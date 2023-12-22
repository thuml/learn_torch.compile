from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[64, 3, 4, 4]"; primals_2: "f32[64]"; primals_3: "f32[64]"; primals_4: "f32[64]"; primals_5: "f32[64]"; primals_6: "f32[64]"; primals_7: "f32[64, 64]"; primals_8: "f32[64]"; primals_9: "f32[64, 64, 8, 8]"; primals_10: "f32[64]"; primals_11: "f32[64]"; primals_12: "f32[64]"; primals_13: "f32[128, 64]"; primals_14: "f32[128]"; primals_15: "f32[64, 64]"; primals_16: "f32[64]"; primals_17: "f32[64]"; primals_18: "f32[64]"; primals_19: "f32[512, 64]"; primals_20: "f32[512]"; primals_21: "f32[64, 512]"; primals_22: "f32[64]"; primals_23: "f32[64, 1, 3, 3]"; primals_24: "f32[64]"; primals_25: "f32[64]"; primals_26: "f32[64]"; primals_27: "f32[64, 64]"; primals_28: "f32[64]"; primals_29: "f32[64, 64, 8, 8]"; primals_30: "f32[64]"; primals_31: "f32[64]"; primals_32: "f32[64]"; primals_33: "f32[128, 64]"; primals_34: "f32[128]"; primals_35: "f32[64, 64]"; primals_36: "f32[64]"; primals_37: "f32[64]"; primals_38: "f32[64]"; primals_39: "f32[512, 64]"; primals_40: "f32[512]"; primals_41: "f32[64, 512]"; primals_42: "f32[64]"; primals_43: "f32[64]"; primals_44: "f32[64]"; primals_45: "f32[64, 64]"; primals_46: "f32[64]"; primals_47: "f32[64, 64, 8, 8]"; primals_48: "f32[64]"; primals_49: "f32[64]"; primals_50: "f32[64]"; primals_51: "f32[128, 64]"; primals_52: "f32[128]"; primals_53: "f32[64, 64]"; primals_54: "f32[64]"; primals_55: "f32[64]"; primals_56: "f32[64]"; primals_57: "f32[512, 64]"; primals_58: "f32[512]"; primals_59: "f32[64, 512]"; primals_60: "f32[64]"; primals_61: "f32[128, 64, 2, 2]"; primals_62: "f32[128]"; primals_63: "f32[128]"; primals_64: "f32[128]"; primals_65: "f32[128]"; primals_66: "f32[128]"; primals_67: "f32[128, 128]"; primals_68: "f32[128]"; primals_69: "f32[128, 128, 4, 4]"; primals_70: "f32[128]"; primals_71: "f32[128]"; primals_72: "f32[128]"; primals_73: "f32[256, 128]"; primals_74: "f32[256]"; primals_75: "f32[128, 128]"; primals_76: "f32[128]"; primals_77: "f32[128]"; primals_78: "f32[128]"; primals_79: "f32[1024, 128]"; primals_80: "f32[1024]"; primals_81: "f32[128, 1024]"; primals_82: "f32[128]"; primals_83: "f32[128, 1, 3, 3]"; primals_84: "f32[128]"; primals_85: "f32[128]"; primals_86: "f32[128]"; primals_87: "f32[128, 128]"; primals_88: "f32[128]"; primals_89: "f32[128, 128, 4, 4]"; primals_90: "f32[128]"; primals_91: "f32[128]"; primals_92: "f32[128]"; primals_93: "f32[256, 128]"; primals_94: "f32[256]"; primals_95: "f32[128, 128]"; primals_96: "f32[128]"; primals_97: "f32[128]"; primals_98: "f32[128]"; primals_99: "f32[1024, 128]"; primals_100: "f32[1024]"; primals_101: "f32[128, 1024]"; primals_102: "f32[128]"; primals_103: "f32[128]"; primals_104: "f32[128]"; primals_105: "f32[128, 128]"; primals_106: "f32[128]"; primals_107: "f32[128, 128, 4, 4]"; primals_108: "f32[128]"; primals_109: "f32[128]"; primals_110: "f32[128]"; primals_111: "f32[256, 128]"; primals_112: "f32[256]"; primals_113: "f32[128, 128]"; primals_114: "f32[128]"; primals_115: "f32[128]"; primals_116: "f32[128]"; primals_117: "f32[1024, 128]"; primals_118: "f32[1024]"; primals_119: "f32[128, 1024]"; primals_120: "f32[128]"; primals_121: "f32[128]"; primals_122: "f32[128]"; primals_123: "f32[128, 128]"; primals_124: "f32[128]"; primals_125: "f32[128, 128, 4, 4]"; primals_126: "f32[128]"; primals_127: "f32[128]"; primals_128: "f32[128]"; primals_129: "f32[256, 128]"; primals_130: "f32[256]"; primals_131: "f32[128, 128]"; primals_132: "f32[128]"; primals_133: "f32[128]"; primals_134: "f32[128]"; primals_135: "f32[1024, 128]"; primals_136: "f32[1024]"; primals_137: "f32[128, 1024]"; primals_138: "f32[128]"; primals_139: "f32[320, 128, 2, 2]"; primals_140: "f32[320]"; primals_141: "f32[320]"; primals_142: "f32[320]"; primals_143: "f32[320]"; primals_144: "f32[320]"; primals_145: "f32[320, 320]"; primals_146: "f32[320]"; primals_147: "f32[320, 320, 2, 2]"; primals_148: "f32[320]"; primals_149: "f32[320]"; primals_150: "f32[320]"; primals_151: "f32[640, 320]"; primals_152: "f32[640]"; primals_153: "f32[320, 320]"; primals_154: "f32[320]"; primals_155: "f32[320]"; primals_156: "f32[320]"; primals_157: "f32[1280, 320]"; primals_158: "f32[1280]"; primals_159: "f32[320, 1280]"; primals_160: "f32[320]"; primals_161: "f32[320, 1, 3, 3]"; primals_162: "f32[320]"; primals_163: "f32[320]"; primals_164: "f32[320]"; primals_165: "f32[320, 320]"; primals_166: "f32[320]"; primals_167: "f32[320, 320, 2, 2]"; primals_168: "f32[320]"; primals_169: "f32[320]"; primals_170: "f32[320]"; primals_171: "f32[640, 320]"; primals_172: "f32[640]"; primals_173: "f32[320, 320]"; primals_174: "f32[320]"; primals_175: "f32[320]"; primals_176: "f32[320]"; primals_177: "f32[1280, 320]"; primals_178: "f32[1280]"; primals_179: "f32[320, 1280]"; primals_180: "f32[320]"; primals_181: "f32[320]"; primals_182: "f32[320]"; primals_183: "f32[320, 320]"; primals_184: "f32[320]"; primals_185: "f32[320, 320, 2, 2]"; primals_186: "f32[320]"; primals_187: "f32[320]"; primals_188: "f32[320]"; primals_189: "f32[640, 320]"; primals_190: "f32[640]"; primals_191: "f32[320, 320]"; primals_192: "f32[320]"; primals_193: "f32[320]"; primals_194: "f32[320]"; primals_195: "f32[1280, 320]"; primals_196: "f32[1280]"; primals_197: "f32[320, 1280]"; primals_198: "f32[320]"; primals_199: "f32[320]"; primals_200: "f32[320]"; primals_201: "f32[320, 320]"; primals_202: "f32[320]"; primals_203: "f32[320, 320, 2, 2]"; primals_204: "f32[320]"; primals_205: "f32[320]"; primals_206: "f32[320]"; primals_207: "f32[640, 320]"; primals_208: "f32[640]"; primals_209: "f32[320, 320]"; primals_210: "f32[320]"; primals_211: "f32[320]"; primals_212: "f32[320]"; primals_213: "f32[1280, 320]"; primals_214: "f32[1280]"; primals_215: "f32[320, 1280]"; primals_216: "f32[320]"; primals_217: "f32[320]"; primals_218: "f32[320]"; primals_219: "f32[320, 320]"; primals_220: "f32[320]"; primals_221: "f32[320, 320, 2, 2]"; primals_222: "f32[320]"; primals_223: "f32[320]"; primals_224: "f32[320]"; primals_225: "f32[640, 320]"; primals_226: "f32[640]"; primals_227: "f32[320, 320]"; primals_228: "f32[320]"; primals_229: "f32[320]"; primals_230: "f32[320]"; primals_231: "f32[1280, 320]"; primals_232: "f32[1280]"; primals_233: "f32[320, 1280]"; primals_234: "f32[320]"; primals_235: "f32[320]"; primals_236: "f32[320]"; primals_237: "f32[320, 320]"; primals_238: "f32[320]"; primals_239: "f32[320, 320, 2, 2]"; primals_240: "f32[320]"; primals_241: "f32[320]"; primals_242: "f32[320]"; primals_243: "f32[640, 320]"; primals_244: "f32[640]"; primals_245: "f32[320, 320]"; primals_246: "f32[320]"; primals_247: "f32[320]"; primals_248: "f32[320]"; primals_249: "f32[1280, 320]"; primals_250: "f32[1280]"; primals_251: "f32[320, 1280]"; primals_252: "f32[320]"; primals_253: "f32[320]"; primals_254: "f32[320]"; primals_255: "f32[320, 320]"; primals_256: "f32[320]"; primals_257: "f32[320, 320, 2, 2]"; primals_258: "f32[320]"; primals_259: "f32[320]"; primals_260: "f32[320]"; primals_261: "f32[640, 320]"; primals_262: "f32[640]"; primals_263: "f32[320, 320]"; primals_264: "f32[320]"; primals_265: "f32[320]"; primals_266: "f32[320]"; primals_267: "f32[1280, 320]"; primals_268: "f32[1280]"; primals_269: "f32[320, 1280]"; primals_270: "f32[320]"; primals_271: "f32[320]"; primals_272: "f32[320]"; primals_273: "f32[320, 320]"; primals_274: "f32[320]"; primals_275: "f32[320, 320, 2, 2]"; primals_276: "f32[320]"; primals_277: "f32[320]"; primals_278: "f32[320]"; primals_279: "f32[640, 320]"; primals_280: "f32[640]"; primals_281: "f32[320, 320]"; primals_282: "f32[320]"; primals_283: "f32[320]"; primals_284: "f32[320]"; primals_285: "f32[1280, 320]"; primals_286: "f32[1280]"; primals_287: "f32[320, 1280]"; primals_288: "f32[320]"; primals_289: "f32[320]"; primals_290: "f32[320]"; primals_291: "f32[320, 320]"; primals_292: "f32[320]"; primals_293: "f32[320, 320, 2, 2]"; primals_294: "f32[320]"; primals_295: "f32[320]"; primals_296: "f32[320]"; primals_297: "f32[640, 320]"; primals_298: "f32[640]"; primals_299: "f32[320, 320]"; primals_300: "f32[320]"; primals_301: "f32[320]"; primals_302: "f32[320]"; primals_303: "f32[1280, 320]"; primals_304: "f32[1280]"; primals_305: "f32[320, 1280]"; primals_306: "f32[320]"; primals_307: "f32[320]"; primals_308: "f32[320]"; primals_309: "f32[320, 320]"; primals_310: "f32[320]"; primals_311: "f32[320, 320, 2, 2]"; primals_312: "f32[320]"; primals_313: "f32[320]"; primals_314: "f32[320]"; primals_315: "f32[640, 320]"; primals_316: "f32[640]"; primals_317: "f32[320, 320]"; primals_318: "f32[320]"; primals_319: "f32[320]"; primals_320: "f32[320]"; primals_321: "f32[1280, 320]"; primals_322: "f32[1280]"; primals_323: "f32[320, 1280]"; primals_324: "f32[320]"; primals_325: "f32[320]"; primals_326: "f32[320]"; primals_327: "f32[320, 320]"; primals_328: "f32[320]"; primals_329: "f32[320, 320, 2, 2]"; primals_330: "f32[320]"; primals_331: "f32[320]"; primals_332: "f32[320]"; primals_333: "f32[640, 320]"; primals_334: "f32[640]"; primals_335: "f32[320, 320]"; primals_336: "f32[320]"; primals_337: "f32[320]"; primals_338: "f32[320]"; primals_339: "f32[1280, 320]"; primals_340: "f32[1280]"; primals_341: "f32[320, 1280]"; primals_342: "f32[320]"; primals_343: "f32[320]"; primals_344: "f32[320]"; primals_345: "f32[320, 320]"; primals_346: "f32[320]"; primals_347: "f32[320, 320, 2, 2]"; primals_348: "f32[320]"; primals_349: "f32[320]"; primals_350: "f32[320]"; primals_351: "f32[640, 320]"; primals_352: "f32[640]"; primals_353: "f32[320, 320]"; primals_354: "f32[320]"; primals_355: "f32[320]"; primals_356: "f32[320]"; primals_357: "f32[1280, 320]"; primals_358: "f32[1280]"; primals_359: "f32[320, 1280]"; primals_360: "f32[320]"; primals_361: "f32[320]"; primals_362: "f32[320]"; primals_363: "f32[320, 320]"; primals_364: "f32[320]"; primals_365: "f32[320, 320, 2, 2]"; primals_366: "f32[320]"; primals_367: "f32[320]"; primals_368: "f32[320]"; primals_369: "f32[640, 320]"; primals_370: "f32[640]"; primals_371: "f32[320, 320]"; primals_372: "f32[320]"; primals_373: "f32[320]"; primals_374: "f32[320]"; primals_375: "f32[1280, 320]"; primals_376: "f32[1280]"; primals_377: "f32[320, 1280]"; primals_378: "f32[320]"; primals_379: "f32[320]"; primals_380: "f32[320]"; primals_381: "f32[320, 320]"; primals_382: "f32[320]"; primals_383: "f32[320, 320, 2, 2]"; primals_384: "f32[320]"; primals_385: "f32[320]"; primals_386: "f32[320]"; primals_387: "f32[640, 320]"; primals_388: "f32[640]"; primals_389: "f32[320, 320]"; primals_390: "f32[320]"; primals_391: "f32[320]"; primals_392: "f32[320]"; primals_393: "f32[1280, 320]"; primals_394: "f32[1280]"; primals_395: "f32[320, 1280]"; primals_396: "f32[320]"; primals_397: "f32[320]"; primals_398: "f32[320]"; primals_399: "f32[320, 320]"; primals_400: "f32[320]"; primals_401: "f32[320, 320, 2, 2]"; primals_402: "f32[320]"; primals_403: "f32[320]"; primals_404: "f32[320]"; primals_405: "f32[640, 320]"; primals_406: "f32[640]"; primals_407: "f32[320, 320]"; primals_408: "f32[320]"; primals_409: "f32[320]"; primals_410: "f32[320]"; primals_411: "f32[1280, 320]"; primals_412: "f32[1280]"; primals_413: "f32[320, 1280]"; primals_414: "f32[320]"; primals_415: "f32[320]"; primals_416: "f32[320]"; primals_417: "f32[320, 320]"; primals_418: "f32[320]"; primals_419: "f32[320, 320, 2, 2]"; primals_420: "f32[320]"; primals_421: "f32[320]"; primals_422: "f32[320]"; primals_423: "f32[640, 320]"; primals_424: "f32[640]"; primals_425: "f32[320, 320]"; primals_426: "f32[320]"; primals_427: "f32[320]"; primals_428: "f32[320]"; primals_429: "f32[1280, 320]"; primals_430: "f32[1280]"; primals_431: "f32[320, 1280]"; primals_432: "f32[320]"; primals_433: "f32[320]"; primals_434: "f32[320]"; primals_435: "f32[320, 320]"; primals_436: "f32[320]"; primals_437: "f32[320, 320, 2, 2]"; primals_438: "f32[320]"; primals_439: "f32[320]"; primals_440: "f32[320]"; primals_441: "f32[640, 320]"; primals_442: "f32[640]"; primals_443: "f32[320, 320]"; primals_444: "f32[320]"; primals_445: "f32[320]"; primals_446: "f32[320]"; primals_447: "f32[1280, 320]"; primals_448: "f32[1280]"; primals_449: "f32[320, 1280]"; primals_450: "f32[320]"; primals_451: "f32[320]"; primals_452: "f32[320]"; primals_453: "f32[320, 320]"; primals_454: "f32[320]"; primals_455: "f32[320, 320, 2, 2]"; primals_456: "f32[320]"; primals_457: "f32[320]"; primals_458: "f32[320]"; primals_459: "f32[640, 320]"; primals_460: "f32[640]"; primals_461: "f32[320, 320]"; primals_462: "f32[320]"; primals_463: "f32[320]"; primals_464: "f32[320]"; primals_465: "f32[1280, 320]"; primals_466: "f32[1280]"; primals_467: "f32[320, 1280]"; primals_468: "f32[320]"; primals_469: "f32[512, 320, 2, 2]"; primals_470: "f32[512]"; primals_471: "f32[512]"; primals_472: "f32[512]"; primals_473: "f32[512]"; primals_474: "f32[512]"; primals_475: "f32[512, 512]"; primals_476: "f32[512]"; primals_477: "f32[1024, 512]"; primals_478: "f32[1024]"; primals_479: "f32[512, 512]"; primals_480: "f32[512]"; primals_481: "f32[512]"; primals_482: "f32[512]"; primals_483: "f32[2048, 512]"; primals_484: "f32[2048]"; primals_485: "f32[512, 2048]"; primals_486: "f32[512]"; primals_487: "f32[512, 1, 3, 3]"; primals_488: "f32[512]"; primals_489: "f32[512]"; primals_490: "f32[512]"; primals_491: "f32[512, 512]"; primals_492: "f32[512]"; primals_493: "f32[1024, 512]"; primals_494: "f32[1024]"; primals_495: "f32[512, 512]"; primals_496: "f32[512]"; primals_497: "f32[512]"; primals_498: "f32[512]"; primals_499: "f32[2048, 512]"; primals_500: "f32[2048]"; primals_501: "f32[512, 2048]"; primals_502: "f32[512]"; primals_503: "f32[512]"; primals_504: "f32[512]"; primals_505: "f32[512, 512]"; primals_506: "f32[512]"; primals_507: "f32[1024, 512]"; primals_508: "f32[1024]"; primals_509: "f32[512, 512]"; primals_510: "f32[512]"; primals_511: "f32[512]"; primals_512: "f32[512]"; primals_513: "f32[2048, 512]"; primals_514: "f32[2048]"; primals_515: "f32[512, 2048]"; primals_516: "f32[512]"; primals_517: "f32[512]"; primals_518: "f32[512]"; primals_519: "f32[1000, 512]"; primals_520: "f32[1000]"; primals_521: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    convolution: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(primals_521, primals_1, primals_2, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_2 = None
    view: "f32[8, 64, 3136]" = torch.ops.aten.view.default(convolution, [8, 64, 3136]);  convolution = None
    permute: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 3136, 1]" = var_mean[0]
    getitem_1: "f32[8, 3136, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = None
    mul: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul, primals_3);  mul = None
    add_1: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:409, code: x = drop(x)
    clone_1: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_1, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 3136, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 3136, 1]" = var_mean_1[1];  var_mean_1 = None
    add_2: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_1, getitem_3)
    mul_2: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_3: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_2, primals_5);  mul_2 = None
    add_3: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_3, primals_6);  mul_3 = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_1: "f32[25088, 64]" = torch.ops.aten.view.default(add_3, [25088, 64])
    permute_1: "f32[64, 64]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm: "f32[25088, 64]" = torch.ops.aten.addmm.default(primals_8, view_1, permute_1);  primals_8 = None
    view_2: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm, [8, 3136, 64]);  addmm = None
    view_3: "f32[8, 3136, 1, 64]" = torch.ops.aten.view.default(view_2, [8, 3136, 1, 64]);  view_2 = None
    permute_2: "f32[8, 1, 3136, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_3: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(add_3, [0, 2, 1]);  add_3 = None
    view_4: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_3, [8, 64, 56, 56]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_1: "f32[8, 64, 7, 7]" = torch.ops.aten.convolution.default(view_4, primals_9, primals_10, [8, 8], [0, 0], [1, 1], False, [0, 0], 1);  primals_10 = None
    view_5: "f32[8, 64, 49]" = torch.ops.aten.view.default(convolution_1, [8, 64, -1]);  convolution_1 = None
    permute_4: "f32[8, 49, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_2: "f32[8, 49, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_2, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 49, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 49, 1]" = var_mean_2[1];  var_mean_2 = None
    add_4: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(clone_2, getitem_5);  clone_2 = None
    mul_4: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_5: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_4, primals_11);  mul_4 = None
    add_5: "f32[8, 49, 64]" = torch.ops.aten.add.Tensor(mul_5, primals_12);  mul_5 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_6: "f32[392, 64]" = torch.ops.aten.view.default(add_5, [392, 64]);  add_5 = None
    permute_5: "f32[64, 128]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_1: "f32[392, 128]" = torch.ops.aten.addmm.default(primals_14, view_6, permute_5);  primals_14 = None
    view_7: "f32[8, 49, 128]" = torch.ops.aten.view.default(addmm_1, [8, 49, 128]);  addmm_1 = None
    view_8: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.view.default(view_7, [8, -1, 2, 1, 64]);  view_7 = None
    permute_6: "f32[2, 8, 1, 49, 64]" = torch.ops.aten.permute.default(view_8, [2, 0, 3, 1, 4]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_6);  permute_6 = None
    getitem_6: "f32[8, 1, 49, 64]" = unbind[0]
    getitem_7: "f32[8, 1, 49, 64]" = unbind[1];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, getitem_6, getitem_7, None, True)
    getitem_8: "f32[8, 1, 3136, 64]" = _scaled_dot_product_efficient_attention[0]
    getitem_9: "f32[8, 1, 3136]" = _scaled_dot_product_efficient_attention[1]
    getitem_10: "i64[]" = _scaled_dot_product_efficient_attention[2]
    getitem_11: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
    alias: "f32[8, 1, 3136, 64]" = torch.ops.aten.alias.default(getitem_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_7: "f32[8, 3136, 1, 64]" = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None
    view_9: "f32[8, 3136, 64]" = torch.ops.aten.view.default(permute_7, [8, 3136, 64]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_10: "f32[25088, 64]" = torch.ops.aten.view.default(view_9, [25088, 64]);  view_9 = None
    permute_8: "f32[64, 64]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_2: "f32[25088, 64]" = torch.ops.aten.addmm.default(primals_16, view_10, permute_8);  primals_16 = None
    view_11: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_2, [8, 3136, 64]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_3: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(view_11);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_6: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(clone_1, clone_3);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_3 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 3136, 1]" = var_mean_3[0]
    getitem_13: "f32[8, 3136, 1]" = var_mean_3[1];  var_mean_3 = None
    add_7: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_3: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_3: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(add_6, getitem_13)
    mul_6: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_7: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_6, primals_17);  mul_6 = None
    add_8: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_7, primals_18);  mul_7 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_12: "f32[25088, 64]" = torch.ops.aten.view.default(add_8, [25088, 64]);  add_8 = None
    permute_9: "f32[64, 512]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    addmm_3: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_20, view_12, permute_9);  primals_20 = None
    view_13: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_3, [8, 3136, 512]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_8: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
    mul_9: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476)
    erf: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
    add_9: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_10: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_8, add_9);  mul_8 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_4: "f32[8, 3136, 512]" = torch.ops.aten.clone.default(mul_10);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_14: "f32[25088, 512]" = torch.ops.aten.view.default(clone_4, [25088, 512]);  clone_4 = None
    permute_10: "f32[512, 64]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
    addmm_4: "f32[25088, 64]" = torch.ops.aten.addmm.default(primals_22, view_14, permute_10);  primals_22 = None
    view_15: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_4, [8, 3136, 64]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_5: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(view_15);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_10: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_6, clone_5);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    permute_11: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(add_10, [0, 2, 1]);  add_10 = None
    view_16: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_11, [8, 64, 56, 56]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    convolution_2: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(view_16, primals_23, primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:245, code: x += cnn_feat_token
    add_11: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(convolution_2, view_16);  convolution_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    view_18: "f32[8, 64, 3136]" = torch.ops.aten.view.default(add_11, [8, 64, 3136]);  add_11 = None
    permute_13: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
    clone_6: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_6, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 3136, 1]" = var_mean_4[0]
    getitem_15: "f32[8, 3136, 1]" = var_mean_4[1];  var_mean_4 = None
    add_12: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_4: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_4: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_6, getitem_15);  clone_6 = None
    mul_11: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_12: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_11, primals_25);  mul_11 = None
    add_13: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_12, primals_26);  mul_12 = primals_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_19: "f32[25088, 64]" = torch.ops.aten.view.default(add_13, [25088, 64])
    permute_14: "f32[64, 64]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    addmm_5: "f32[25088, 64]" = torch.ops.aten.addmm.default(primals_28, view_19, permute_14);  primals_28 = None
    view_20: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_5, [8, 3136, 64]);  addmm_5 = None
    view_21: "f32[8, 3136, 1, 64]" = torch.ops.aten.view.default(view_20, [8, 3136, 1, 64]);  view_20 = None
    permute_15: "f32[8, 1, 3136, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_16: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(add_13, [0, 2, 1]);  add_13 = None
    view_22: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_16, [8, 64, 56, 56]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_3: "f32[8, 64, 7, 7]" = torch.ops.aten.convolution.default(view_22, primals_29, primals_30, [8, 8], [0, 0], [1, 1], False, [0, 0], 1);  primals_30 = None
    view_23: "f32[8, 64, 49]" = torch.ops.aten.view.default(convolution_3, [8, 64, -1]);  convolution_3 = None
    permute_17: "f32[8, 49, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_7: "f32[8, 49, 64]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_7, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 49, 1]" = var_mean_5[0]
    getitem_17: "f32[8, 49, 1]" = var_mean_5[1];  var_mean_5 = None
    add_14: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_5: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_5: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(clone_7, getitem_17);  clone_7 = None
    mul_13: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_14: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_13, primals_31);  mul_13 = None
    add_15: "f32[8, 49, 64]" = torch.ops.aten.add.Tensor(mul_14, primals_32);  mul_14 = primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_24: "f32[392, 64]" = torch.ops.aten.view.default(add_15, [392, 64]);  add_15 = None
    permute_18: "f32[64, 128]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
    addmm_6: "f32[392, 128]" = torch.ops.aten.addmm.default(primals_34, view_24, permute_18);  primals_34 = None
    view_25: "f32[8, 49, 128]" = torch.ops.aten.view.default(addmm_6, [8, 49, 128]);  addmm_6 = None
    view_26: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.view.default(view_25, [8, -1, 2, 1, 64]);  view_25 = None
    permute_19: "f32[2, 8, 1, 49, 64]" = torch.ops.aten.permute.default(view_26, [2, 0, 3, 1, 4]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_19);  permute_19 = None
    getitem_18: "f32[8, 1, 49, 64]" = unbind_1[0]
    getitem_19: "f32[8, 1, 49, 64]" = unbind_1[1];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_15, getitem_18, getitem_19, None, True)
    getitem_20: "f32[8, 1, 3136, 64]" = _scaled_dot_product_efficient_attention_1[0]
    getitem_21: "f32[8, 1, 3136]" = _scaled_dot_product_efficient_attention_1[1]
    getitem_22: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
    getitem_23: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
    alias_1: "f32[8, 1, 3136, 64]" = torch.ops.aten.alias.default(getitem_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_20: "f32[8, 3136, 1, 64]" = torch.ops.aten.permute.default(getitem_20, [0, 2, 1, 3]);  getitem_20 = None
    view_27: "f32[8, 3136, 64]" = torch.ops.aten.view.default(permute_20, [8, 3136, 64]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_28: "f32[25088, 64]" = torch.ops.aten.view.default(view_27, [25088, 64]);  view_27 = None
    permute_21: "f32[64, 64]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
    addmm_7: "f32[25088, 64]" = torch.ops.aten.addmm.default(primals_36, view_28, permute_21);  primals_36 = None
    view_29: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_7, [8, 3136, 64]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_8: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(view_29);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_16: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(permute_13, clone_8);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_9: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_16, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_9, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 3136, 1]" = var_mean_6[0]
    getitem_25: "f32[8, 3136, 1]" = var_mean_6[1];  var_mean_6 = None
    add_17: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_6: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_6: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_9, getitem_25);  clone_9 = None
    mul_15: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_16: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_15, primals_37);  mul_15 = None
    add_18: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_16, primals_38);  mul_16 = primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_30: "f32[25088, 64]" = torch.ops.aten.view.default(add_18, [25088, 64]);  add_18 = None
    permute_22: "f32[64, 512]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
    addmm_8: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_40, view_30, permute_22);  primals_40 = None
    view_31: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_8, [8, 3136, 512]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
    mul_18: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476)
    erf_1: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_19: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_19: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_17, add_19);  mul_17 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_10: "f32[8, 3136, 512]" = torch.ops.aten.clone.default(mul_19);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_32: "f32[25088, 512]" = torch.ops.aten.view.default(clone_10, [25088, 512]);  clone_10 = None
    permute_23: "f32[512, 64]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
    addmm_9: "f32[25088, 64]" = torch.ops.aten.addmm.default(primals_42, view_32, permute_23);  primals_42 = None
    view_33: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_9, [8, 3136, 64]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_11: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(view_33);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_20: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_16, clone_11);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_12: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_12, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 3136, 1]" = var_mean_7[0]
    getitem_27: "f32[8, 3136, 1]" = var_mean_7[1];  var_mean_7 = None
    add_21: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_7: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_7: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_12, getitem_27);  clone_12 = None
    mul_20: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_21: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_20, primals_43);  mul_20 = None
    add_22: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_21, primals_44);  mul_21 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_34: "f32[25088, 64]" = torch.ops.aten.view.default(add_22, [25088, 64])
    permute_24: "f32[64, 64]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    addmm_10: "f32[25088, 64]" = torch.ops.aten.addmm.default(primals_46, view_34, permute_24);  primals_46 = None
    view_35: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_10, [8, 3136, 64]);  addmm_10 = None
    view_36: "f32[8, 3136, 1, 64]" = torch.ops.aten.view.default(view_35, [8, 3136, 1, 64]);  view_35 = None
    permute_25: "f32[8, 1, 3136, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_26: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(add_22, [0, 2, 1]);  add_22 = None
    view_37: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_26, [8, 64, 56, 56]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_4: "f32[8, 64, 7, 7]" = torch.ops.aten.convolution.default(view_37, primals_47, primals_48, [8, 8], [0, 0], [1, 1], False, [0, 0], 1);  primals_48 = None
    view_38: "f32[8, 64, 49]" = torch.ops.aten.view.default(convolution_4, [8, 64, -1]);  convolution_4 = None
    permute_27: "f32[8, 49, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_13: "f32[8, 49, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_13, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 49, 1]" = var_mean_8[0]
    getitem_29: "f32[8, 49, 1]" = var_mean_8[1];  var_mean_8 = None
    add_23: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_8: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_8: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(clone_13, getitem_29);  clone_13 = None
    mul_22: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_23: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_22, primals_49);  mul_22 = None
    add_24: "f32[8, 49, 64]" = torch.ops.aten.add.Tensor(mul_23, primals_50);  mul_23 = primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_39: "f32[392, 64]" = torch.ops.aten.view.default(add_24, [392, 64]);  add_24 = None
    permute_28: "f32[64, 128]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    addmm_11: "f32[392, 128]" = torch.ops.aten.addmm.default(primals_52, view_39, permute_28);  primals_52 = None
    view_40: "f32[8, 49, 128]" = torch.ops.aten.view.default(addmm_11, [8, 49, 128]);  addmm_11 = None
    view_41: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.view.default(view_40, [8, -1, 2, 1, 64]);  view_40 = None
    permute_29: "f32[2, 8, 1, 49, 64]" = torch.ops.aten.permute.default(view_41, [2, 0, 3, 1, 4]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_29);  permute_29 = None
    getitem_30: "f32[8, 1, 49, 64]" = unbind_2[0]
    getitem_31: "f32[8, 1, 49, 64]" = unbind_2[1];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_25, getitem_30, getitem_31, None, True)
    getitem_32: "f32[8, 1, 3136, 64]" = _scaled_dot_product_efficient_attention_2[0]
    getitem_33: "f32[8, 1, 3136]" = _scaled_dot_product_efficient_attention_2[1]
    getitem_34: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
    getitem_35: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
    alias_2: "f32[8, 1, 3136, 64]" = torch.ops.aten.alias.default(getitem_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_30: "f32[8, 3136, 1, 64]" = torch.ops.aten.permute.default(getitem_32, [0, 2, 1, 3]);  getitem_32 = None
    view_42: "f32[8, 3136, 64]" = torch.ops.aten.view.default(permute_30, [8, 3136, 64]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_43: "f32[25088, 64]" = torch.ops.aten.view.default(view_42, [25088, 64]);  view_42 = None
    permute_31: "f32[64, 64]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    addmm_12: "f32[25088, 64]" = torch.ops.aten.addmm.default(primals_54, view_43, permute_31);  primals_54 = None
    view_44: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_12, [8, 3136, 64]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_14: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_25: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_20, clone_14);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_15: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_25, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_15, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 3136, 1]" = var_mean_9[0]
    getitem_37: "f32[8, 3136, 1]" = var_mean_9[1];  var_mean_9 = None
    add_26: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_9: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_9: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_15, getitem_37);  clone_15 = None
    mul_24: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_25: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_24, primals_55);  mul_24 = None
    add_27: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_25, primals_56);  mul_25 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_45: "f32[25088, 64]" = torch.ops.aten.view.default(add_27, [25088, 64]);  add_27 = None
    permute_32: "f32[64, 512]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    addmm_13: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_58, view_45, permute_32);  primals_58 = None
    view_46: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_13, [8, 3136, 512]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_26: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
    mul_27: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476)
    erf_2: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_28: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_28: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_26, add_28);  mul_26 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_16: "f32[8, 3136, 512]" = torch.ops.aten.clone.default(mul_28);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[25088, 512]" = torch.ops.aten.view.default(clone_16, [25088, 512]);  clone_16 = None
    permute_33: "f32[512, 64]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    addmm_14: "f32[25088, 64]" = torch.ops.aten.addmm.default(primals_60, view_47, permute_33);  primals_60 = None
    view_48: "f32[8, 3136, 64]" = torch.ops.aten.view.default(addmm_14, [8, 3136, 64]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_17: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_29: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_25, clone_17);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:415, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
    view_49: "f32[8, 56, 56, 64]" = torch.ops.aten.view.default(add_29, [8, 56, 56, 64]);  add_29 = None
    permute_34: "f32[8, 64, 56, 56]" = torch.ops.aten.permute.default(view_49, [0, 3, 1, 2]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    convolution_5: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(permute_34, primals_61, primals_62, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_62 = None
    view_50: "f32[8, 128, 784]" = torch.ops.aten.view.default(convolution_5, [8, 128, 784]);  convolution_5 = None
    permute_35: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone_18: "f32[8, 784, 128]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_18, [2], correction = 0, keepdim = True)
    getitem_38: "f32[8, 784, 1]" = var_mean_10[0]
    getitem_39: "f32[8, 784, 1]" = var_mean_10[1];  var_mean_10 = None
    add_30: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_10: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_10: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_18, getitem_39);  clone_18 = None
    mul_29: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_30: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_29, primals_63);  mul_29 = None
    add_31: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_30, primals_64);  mul_30 = primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:409, code: x = drop(x)
    clone_19: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_31);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_19, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 784, 1]" = var_mean_11[0]
    getitem_41: "f32[8, 784, 1]" = var_mean_11[1];  var_mean_11 = None
    add_32: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_11: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_11: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_19, getitem_41)
    mul_31: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_32: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_31, primals_65);  mul_31 = None
    add_33: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_32, primals_66);  mul_32 = primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_51: "f32[6272, 128]" = torch.ops.aten.view.default(add_33, [6272, 128])
    permute_36: "f32[128, 128]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    addmm_15: "f32[6272, 128]" = torch.ops.aten.addmm.default(primals_68, view_51, permute_36);  primals_68 = None
    view_52: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_15, [8, 784, 128]);  addmm_15 = None
    view_53: "f32[8, 784, 2, 64]" = torch.ops.aten.view.default(view_52, [8, 784, 2, 64]);  view_52 = None
    permute_37: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_38: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_33, [0, 2, 1]);  add_33 = None
    view_54: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_38, [8, 128, 28, 28]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_6: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(view_54, primals_69, primals_70, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_70 = None
    view_55: "f32[8, 128, 49]" = torch.ops.aten.view.default(convolution_6, [8, 128, -1]);  convolution_6 = None
    permute_39: "f32[8, 49, 128]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_20: "f32[8, 49, 128]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_20, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 49, 1]" = var_mean_12[0]
    getitem_43: "f32[8, 49, 1]" = var_mean_12[1];  var_mean_12 = None
    add_34: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_12: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_12: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(clone_20, getitem_43);  clone_20 = None
    mul_33: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_34: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_33, primals_71);  mul_33 = None
    add_35: "f32[8, 49, 128]" = torch.ops.aten.add.Tensor(mul_34, primals_72);  mul_34 = primals_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_56: "f32[392, 128]" = torch.ops.aten.view.default(add_35, [392, 128]);  add_35 = None
    permute_40: "f32[128, 256]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_16: "f32[392, 256]" = torch.ops.aten.addmm.default(primals_74, view_56, permute_40);  primals_74 = None
    view_57: "f32[8, 49, 256]" = torch.ops.aten.view.default(addmm_16, [8, 49, 256]);  addmm_16 = None
    view_58: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.view.default(view_57, [8, -1, 2, 2, 64]);  view_57 = None
    permute_41: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.permute.default(view_58, [2, 0, 3, 1, 4]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_41);  permute_41 = None
    getitem_44: "f32[8, 2, 49, 64]" = unbind_3[0]
    getitem_45: "f32[8, 2, 49, 64]" = unbind_3[1];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_37, getitem_44, getitem_45, None, True)
    getitem_46: "f32[8, 2, 784, 64]" = _scaled_dot_product_efficient_attention_3[0]
    getitem_47: "f32[8, 2, 800]" = _scaled_dot_product_efficient_attention_3[1]
    getitem_48: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
    getitem_49: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
    alias_3: "f32[8, 2, 784, 64]" = torch.ops.aten.alias.default(getitem_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_42: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_46, [0, 2, 1, 3]);  getitem_46 = None
    view_59: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_42, [8, 784, 128]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_60: "f32[6272, 128]" = torch.ops.aten.view.default(view_59, [6272, 128]);  view_59 = None
    permute_43: "f32[128, 128]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    addmm_17: "f32[6272, 128]" = torch.ops.aten.addmm.default(primals_76, view_60, permute_43);  primals_76 = None
    view_61: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_17, [8, 784, 128]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_21: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_61);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_36: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(clone_19, clone_21);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_13 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
    getitem_50: "f32[8, 784, 1]" = var_mean_13[0]
    getitem_51: "f32[8, 784, 1]" = var_mean_13[1];  var_mean_13 = None
    add_37: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
    rsqrt_13: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_13: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_36, getitem_51)
    mul_35: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_36: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_35, primals_77);  mul_35 = None
    add_38: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_36, primals_78);  mul_36 = primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_62: "f32[6272, 128]" = torch.ops.aten.view.default(add_38, [6272, 128]);  add_38 = None
    permute_44: "f32[128, 1024]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    addmm_18: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_80, view_62, permute_44);  primals_80 = None
    view_63: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_18, [8, 784, 1024]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_38: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_3: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_39: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_39: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_37, add_39);  mul_37 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_22: "f32[8, 784, 1024]" = torch.ops.aten.clone.default(mul_39);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_64: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_22, [6272, 1024]);  clone_22 = None
    permute_45: "f32[1024, 128]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_19: "f32[6272, 128]" = torch.ops.aten.addmm.default(primals_82, view_64, permute_45);  primals_82 = None
    view_65: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_19, [8, 784, 128]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_23: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_65);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_40: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_36, clone_23);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    permute_46: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_40, [0, 2, 1]);  add_40 = None
    view_66: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_46, [8, 128, 28, 28]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    convolution_7: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(view_66, primals_83, primals_84, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:245, code: x += cnn_feat_token
    add_41: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(convolution_7, view_66);  convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    view_68: "f32[8, 128, 784]" = torch.ops.aten.view.default(add_41, [8, 128, 784]);  add_41 = None
    permute_48: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
    clone_24: "f32[8, 784, 128]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_24, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 784, 1]" = var_mean_14[0]
    getitem_53: "f32[8, 784, 1]" = var_mean_14[1];  var_mean_14 = None
    add_42: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_14: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_14: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_24, getitem_53);  clone_24 = None
    mul_40: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_41: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_40, primals_85);  mul_40 = None
    add_43: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_41, primals_86);  mul_41 = primals_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_69: "f32[6272, 128]" = torch.ops.aten.view.default(add_43, [6272, 128])
    permute_49: "f32[128, 128]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_20: "f32[6272, 128]" = torch.ops.aten.addmm.default(primals_88, view_69, permute_49);  primals_88 = None
    view_70: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_20, [8, 784, 128]);  addmm_20 = None
    view_71: "f32[8, 784, 2, 64]" = torch.ops.aten.view.default(view_70, [8, 784, 2, 64]);  view_70 = None
    permute_50: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_51: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_43, [0, 2, 1]);  add_43 = None
    view_72: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_51, [8, 128, 28, 28]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_8: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(view_72, primals_89, primals_90, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_90 = None
    view_73: "f32[8, 128, 49]" = torch.ops.aten.view.default(convolution_8, [8, 128, -1]);  convolution_8 = None
    permute_52: "f32[8, 49, 128]" = torch.ops.aten.permute.default(view_73, [0, 2, 1]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_25: "f32[8, 49, 128]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_25, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 49, 1]" = var_mean_15[0]
    getitem_55: "f32[8, 49, 1]" = var_mean_15[1];  var_mean_15 = None
    add_44: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_15: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_15: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(clone_25, getitem_55);  clone_25 = None
    mul_42: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_43: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_42, primals_91);  mul_42 = None
    add_45: "f32[8, 49, 128]" = torch.ops.aten.add.Tensor(mul_43, primals_92);  mul_43 = primals_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_74: "f32[392, 128]" = torch.ops.aten.view.default(add_45, [392, 128]);  add_45 = None
    permute_53: "f32[128, 256]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    addmm_21: "f32[392, 256]" = torch.ops.aten.addmm.default(primals_94, view_74, permute_53);  primals_94 = None
    view_75: "f32[8, 49, 256]" = torch.ops.aten.view.default(addmm_21, [8, 49, 256]);  addmm_21 = None
    view_76: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.view.default(view_75, [8, -1, 2, 2, 64]);  view_75 = None
    permute_54: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.permute.default(view_76, [2, 0, 3, 1, 4]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_54);  permute_54 = None
    getitem_56: "f32[8, 2, 49, 64]" = unbind_4[0]
    getitem_57: "f32[8, 2, 49, 64]" = unbind_4[1];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_50, getitem_56, getitem_57, None, True)
    getitem_58: "f32[8, 2, 784, 64]" = _scaled_dot_product_efficient_attention_4[0]
    getitem_59: "f32[8, 2, 800]" = _scaled_dot_product_efficient_attention_4[1]
    getitem_60: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
    getitem_61: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None
    alias_4: "f32[8, 2, 784, 64]" = torch.ops.aten.alias.default(getitem_58)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_55: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
    view_77: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_55, [8, 784, 128]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_78: "f32[6272, 128]" = torch.ops.aten.view.default(view_77, [6272, 128]);  view_77 = None
    permute_56: "f32[128, 128]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_22: "f32[6272, 128]" = torch.ops.aten.addmm.default(primals_96, view_78, permute_56);  primals_96 = None
    view_79: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_22, [8, 784, 128]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_26: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_79);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_46: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(permute_48, clone_26);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_27: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_46, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_27, [2], correction = 0, keepdim = True)
    getitem_62: "f32[8, 784, 1]" = var_mean_16[0]
    getitem_63: "f32[8, 784, 1]" = var_mean_16[1];  var_mean_16 = None
    add_47: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
    rsqrt_16: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_16: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_27, getitem_63);  clone_27 = None
    mul_44: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_45: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_44, primals_97);  mul_44 = None
    add_48: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_45, primals_98);  mul_45 = primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_80: "f32[6272, 128]" = torch.ops.aten.view.default(add_48, [6272, 128]);  add_48 = None
    permute_57: "f32[128, 1024]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_23: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_100, view_80, permute_57);  primals_100 = None
    view_81: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_23, [8, 784, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_46: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_81, 0.5)
    mul_47: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_81, 0.7071067811865476)
    erf_4: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_49: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_48: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_28: "f32[8, 784, 1024]" = torch.ops.aten.clone.default(mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_82: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_28, [6272, 1024]);  clone_28 = None
    permute_58: "f32[1024, 128]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_24: "f32[6272, 128]" = torch.ops.aten.addmm.default(primals_102, view_82, permute_58);  primals_102 = None
    view_83: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_24, [8, 784, 128]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_29: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_83);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_50: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_46, clone_29);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_30: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_30, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 784, 1]" = var_mean_17[0]
    getitem_65: "f32[8, 784, 1]" = var_mean_17[1];  var_mean_17 = None
    add_51: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_17: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_17: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_30, getitem_65);  clone_30 = None
    mul_49: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_50: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_49, primals_103);  mul_49 = None
    add_52: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_50, primals_104);  mul_50 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_84: "f32[6272, 128]" = torch.ops.aten.view.default(add_52, [6272, 128])
    permute_59: "f32[128, 128]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_25: "f32[6272, 128]" = torch.ops.aten.addmm.default(primals_106, view_84, permute_59);  primals_106 = None
    view_85: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_25, [8, 784, 128]);  addmm_25 = None
    view_86: "f32[8, 784, 2, 64]" = torch.ops.aten.view.default(view_85, [8, 784, 2, 64]);  view_85 = None
    permute_60: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_61: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_52, [0, 2, 1]);  add_52 = None
    view_87: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_61, [8, 128, 28, 28]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_9: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(view_87, primals_107, primals_108, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_108 = None
    view_88: "f32[8, 128, 49]" = torch.ops.aten.view.default(convolution_9, [8, 128, -1]);  convolution_9 = None
    permute_62: "f32[8, 49, 128]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_31: "f32[8, 49, 128]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_31, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 49, 1]" = var_mean_18[0]
    getitem_67: "f32[8, 49, 1]" = var_mean_18[1];  var_mean_18 = None
    add_53: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_18: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_18: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(clone_31, getitem_67);  clone_31 = None
    mul_51: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_52: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_51, primals_109);  mul_51 = None
    add_54: "f32[8, 49, 128]" = torch.ops.aten.add.Tensor(mul_52, primals_110);  mul_52 = primals_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_89: "f32[392, 128]" = torch.ops.aten.view.default(add_54, [392, 128]);  add_54 = None
    permute_63: "f32[128, 256]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_26: "f32[392, 256]" = torch.ops.aten.addmm.default(primals_112, view_89, permute_63);  primals_112 = None
    view_90: "f32[8, 49, 256]" = torch.ops.aten.view.default(addmm_26, [8, 49, 256]);  addmm_26 = None
    view_91: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.view.default(view_90, [8, -1, 2, 2, 64]);  view_90 = None
    permute_64: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.permute.default(view_91, [2, 0, 3, 1, 4]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_64);  permute_64 = None
    getitem_68: "f32[8, 2, 49, 64]" = unbind_5[0]
    getitem_69: "f32[8, 2, 49, 64]" = unbind_5[1];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_60, getitem_68, getitem_69, None, True)
    getitem_70: "f32[8, 2, 784, 64]" = _scaled_dot_product_efficient_attention_5[0]
    getitem_71: "f32[8, 2, 800]" = _scaled_dot_product_efficient_attention_5[1]
    getitem_72: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
    getitem_73: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None
    alias_5: "f32[8, 2, 784, 64]" = torch.ops.aten.alias.default(getitem_70)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_65: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_70, [0, 2, 1, 3]);  getitem_70 = None
    view_92: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_65, [8, 784, 128]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_93: "f32[6272, 128]" = torch.ops.aten.view.default(view_92, [6272, 128]);  view_92 = None
    permute_66: "f32[128, 128]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    addmm_27: "f32[6272, 128]" = torch.ops.aten.addmm.default(primals_114, view_93, permute_66);  primals_114 = None
    view_94: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_27, [8, 784, 128]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_32: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_94);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_55: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_50, clone_32);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_33: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_55, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_33, [2], correction = 0, keepdim = True)
    getitem_74: "f32[8, 784, 1]" = var_mean_19[0]
    getitem_75: "f32[8, 784, 1]" = var_mean_19[1];  var_mean_19 = None
    add_56: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-06);  getitem_74 = None
    rsqrt_19: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_19: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_33, getitem_75);  clone_33 = None
    mul_53: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_54: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_53, primals_115);  mul_53 = None
    add_57: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_54, primals_116);  mul_54 = primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_95: "f32[6272, 128]" = torch.ops.aten.view.default(add_57, [6272, 128]);  add_57 = None
    permute_67: "f32[128, 1024]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    addmm_28: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_118, view_95, permute_67);  primals_118 = None
    view_96: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_28, [8, 784, 1024]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_55: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_96, 0.5)
    mul_56: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_96, 0.7071067811865476)
    erf_5: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_58: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_57: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_55, add_58);  mul_55 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_34: "f32[8, 784, 1024]" = torch.ops.aten.clone.default(mul_57);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_97: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_34, [6272, 1024]);  clone_34 = None
    permute_68: "f32[1024, 128]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_29: "f32[6272, 128]" = torch.ops.aten.addmm.default(primals_120, view_97, permute_68);  primals_120 = None
    view_98: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_29, [8, 784, 128]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_35: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_98);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_59: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_55, clone_35);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_36: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_59, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_36, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 784, 1]" = var_mean_20[0]
    getitem_77: "f32[8, 784, 1]" = var_mean_20[1];  var_mean_20 = None
    add_60: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_20: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_20: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_36, getitem_77);  clone_36 = None
    mul_58: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_59: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_58, primals_121);  mul_58 = None
    add_61: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_59, primals_122);  mul_59 = primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_99: "f32[6272, 128]" = torch.ops.aten.view.default(add_61, [6272, 128])
    permute_69: "f32[128, 128]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    addmm_30: "f32[6272, 128]" = torch.ops.aten.addmm.default(primals_124, view_99, permute_69);  primals_124 = None
    view_100: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_30, [8, 784, 128]);  addmm_30 = None
    view_101: "f32[8, 784, 2, 64]" = torch.ops.aten.view.default(view_100, [8, 784, 2, 64]);  view_100 = None
    permute_70: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_71: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_61, [0, 2, 1]);  add_61 = None
    view_102: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_71, [8, 128, 28, 28]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_10: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(view_102, primals_125, primals_126, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_126 = None
    view_103: "f32[8, 128, 49]" = torch.ops.aten.view.default(convolution_10, [8, 128, -1]);  convolution_10 = None
    permute_72: "f32[8, 49, 128]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_37: "f32[8, 49, 128]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_37, [2], correction = 0, keepdim = True)
    getitem_78: "f32[8, 49, 1]" = var_mean_21[0]
    getitem_79: "f32[8, 49, 1]" = var_mean_21[1];  var_mean_21 = None
    add_62: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_21: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_21: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(clone_37, getitem_79);  clone_37 = None
    mul_60: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_61: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_60, primals_127);  mul_60 = None
    add_63: "f32[8, 49, 128]" = torch.ops.aten.add.Tensor(mul_61, primals_128);  mul_61 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_104: "f32[392, 128]" = torch.ops.aten.view.default(add_63, [392, 128]);  add_63 = None
    permute_73: "f32[128, 256]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_31: "f32[392, 256]" = torch.ops.aten.addmm.default(primals_130, view_104, permute_73);  primals_130 = None
    view_105: "f32[8, 49, 256]" = torch.ops.aten.view.default(addmm_31, [8, 49, 256]);  addmm_31 = None
    view_106: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.view.default(view_105, [8, -1, 2, 2, 64]);  view_105 = None
    permute_74: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.permute.default(view_106, [2, 0, 3, 1, 4]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_74);  permute_74 = None
    getitem_80: "f32[8, 2, 49, 64]" = unbind_6[0]
    getitem_81: "f32[8, 2, 49, 64]" = unbind_6[1];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_70, getitem_80, getitem_81, None, True)
    getitem_82: "f32[8, 2, 784, 64]" = _scaled_dot_product_efficient_attention_6[0]
    getitem_83: "f32[8, 2, 800]" = _scaled_dot_product_efficient_attention_6[1]
    getitem_84: "i64[]" = _scaled_dot_product_efficient_attention_6[2]
    getitem_85: "i64[]" = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None
    alias_6: "f32[8, 2, 784, 64]" = torch.ops.aten.alias.default(getitem_82)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_75: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
    view_107: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_75, [8, 784, 128]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_108: "f32[6272, 128]" = torch.ops.aten.view.default(view_107, [6272, 128]);  view_107 = None
    permute_76: "f32[128, 128]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_32: "f32[6272, 128]" = torch.ops.aten.addmm.default(primals_132, view_108, permute_76);  primals_132 = None
    view_109: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_32, [8, 784, 128]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_38: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_109);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_64: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_59, clone_38);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_39: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_64, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_39, [2], correction = 0, keepdim = True)
    getitem_86: "f32[8, 784, 1]" = var_mean_22[0]
    getitem_87: "f32[8, 784, 1]" = var_mean_22[1];  var_mean_22 = None
    add_65: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
    rsqrt_22: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_22: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_39, getitem_87);  clone_39 = None
    mul_62: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_63: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_62, primals_133);  mul_62 = None
    add_66: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_63, primals_134);  mul_63 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_110: "f32[6272, 128]" = torch.ops.aten.view.default(add_66, [6272, 128]);  add_66 = None
    permute_77: "f32[128, 1024]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_33: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_136, view_110, permute_77);  primals_136 = None
    view_111: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_33, [8, 784, 1024]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_64: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_111, 0.5)
    mul_65: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476)
    erf_6: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_65);  mul_65 = None
    add_67: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_66: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_64, add_67);  mul_64 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_40: "f32[8, 784, 1024]" = torch.ops.aten.clone.default(mul_66);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_112: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_40, [6272, 1024]);  clone_40 = None
    permute_78: "f32[1024, 128]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_34: "f32[6272, 128]" = torch.ops.aten.addmm.default(primals_138, view_112, permute_78);  primals_138 = None
    view_113: "f32[8, 784, 128]" = torch.ops.aten.view.default(addmm_34, [8, 784, 128]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_41: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_113);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_68: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_64, clone_41);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:415, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
    view_114: "f32[8, 28, 28, 128]" = torch.ops.aten.view.default(add_68, [8, 28, 28, 128]);  add_68 = None
    permute_79: "f32[8, 128, 28, 28]" = torch.ops.aten.permute.default(view_114, [0, 3, 1, 2]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    convolution_11: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(permute_79, primals_139, primals_140, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_140 = None
    view_115: "f32[8, 320, 196]" = torch.ops.aten.view.default(convolution_11, [8, 320, 196]);  convolution_11 = None
    permute_80: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_115, [0, 2, 1]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone_42: "f32[8, 196, 320]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 196, 1]" = var_mean_23[0]
    getitem_89: "f32[8, 196, 1]" = var_mean_23[1];  var_mean_23 = None
    add_69: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_23: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_23: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_42, getitem_89);  clone_42 = None
    mul_67: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_68: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_67, primals_141);  mul_67 = None
    add_70: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_68, primals_142);  mul_68 = primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:409, code: x = drop(x)
    clone_43: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_70);  add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_43, [2], correction = 0, keepdim = True)
    getitem_90: "f32[8, 196, 1]" = var_mean_24[0]
    getitem_91: "f32[8, 196, 1]" = var_mean_24[1];  var_mean_24 = None
    add_71: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
    rsqrt_24: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_24: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_43, getitem_91)
    mul_69: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_70: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_69, primals_143);  mul_69 = None
    add_72: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_70, primals_144);  mul_70 = primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_116: "f32[1568, 320]" = torch.ops.aten.view.default(add_72, [1568, 320])
    permute_81: "f32[320, 320]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_35: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_146, view_116, permute_81);  primals_146 = None
    view_117: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_35, [8, 196, 320]);  addmm_35 = None
    view_118: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_117, [8, 196, 5, 64]);  view_117 = None
    permute_82: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_83: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_72, [0, 2, 1]);  add_72 = None
    view_119: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_83, [8, 320, 14, 14]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_12: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_119, primals_147, primals_148, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_148 = None
    view_120: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_12, [8, 320, -1]);  convolution_12 = None
    permute_84: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_44: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format)
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_44, [2], correction = 0, keepdim = True)
    getitem_92: "f32[8, 49, 1]" = var_mean_25[0]
    getitem_93: "f32[8, 49, 1]" = var_mean_25[1];  var_mean_25 = None
    add_73: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_25: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_25: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_44, getitem_93);  clone_44 = None
    mul_71: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    mul_72: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_71, primals_149);  mul_71 = None
    add_74: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_72, primals_150);  mul_72 = primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_121: "f32[392, 320]" = torch.ops.aten.view.default(add_74, [392, 320]);  add_74 = None
    permute_85: "f32[320, 640]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_36: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_152, view_121, permute_85);  primals_152 = None
    view_122: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_36, [8, 49, 640]);  addmm_36 = None
    view_123: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_122, [8, -1, 2, 5, 64]);  view_122 = None
    permute_86: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_123, [2, 0, 3, 1, 4]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_86);  permute_86 = None
    getitem_94: "f32[8, 5, 49, 64]" = unbind_7[0]
    getitem_95: "f32[8, 5, 49, 64]" = unbind_7[1];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_82, getitem_94, getitem_95, None, True)
    getitem_96: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_7[0]
    getitem_97: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_7[1]
    getitem_98: "i64[]" = _scaled_dot_product_efficient_attention_7[2]
    getitem_99: "i64[]" = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None
    alias_7: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_87: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_96, [0, 2, 1, 3]);  getitem_96 = None
    view_124: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_87, [8, 196, 320]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_125: "f32[1568, 320]" = torch.ops.aten.view.default(view_124, [1568, 320]);  view_124 = None
    permute_88: "f32[320, 320]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    addmm_37: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_154, view_125, permute_88);  primals_154 = None
    view_126: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_37, [8, 196, 320]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_45: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_126);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_75: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(clone_43, clone_45);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_26 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_100: "f32[8, 196, 1]" = var_mean_26[0]
    getitem_101: "f32[8, 196, 1]" = var_mean_26[1];  var_mean_26 = None
    add_76: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-06);  getitem_100 = None
    rsqrt_26: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_26: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_75, getitem_101)
    mul_73: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    mul_74: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_73, primals_155);  mul_73 = None
    add_77: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_74, primals_156);  mul_74 = primals_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_127: "f32[1568, 320]" = torch.ops.aten.view.default(add_77, [1568, 320]);  add_77 = None
    permute_89: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm_38: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_158, view_127, permute_89);  primals_158 = None
    view_128: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_38, [8, 196, 1280]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_75: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_128, 0.5)
    mul_76: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476)
    erf_7: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_78: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_77: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_75, add_78);  mul_75 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_46: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_77);  mul_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_129: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_46, [1568, 1280]);  clone_46 = None
    permute_90: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    addmm_39: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_160, view_129, permute_90);  primals_160 = None
    view_130: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_39, [8, 196, 320]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_47: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_130);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_79: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_75, clone_47);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    permute_91: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_79, [0, 2, 1]);  add_79 = None
    view_131: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_91, [8, 320, 14, 14]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    convolution_13: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(view_131, primals_161, primals_162, [1, 1], [1, 1], [1, 1], False, [0, 0], 320);  primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:245, code: x += cnn_feat_token
    add_80: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(convolution_13, view_131);  convolution_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    view_133: "f32[8, 320, 196]" = torch.ops.aten.view.default(add_80, [8, 320, 196]);  add_80 = None
    permute_93: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    clone_48: "f32[8, 196, 320]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_48, [2], correction = 0, keepdim = True)
    getitem_102: "f32[8, 196, 1]" = var_mean_27[0]
    getitem_103: "f32[8, 196, 1]" = var_mean_27[1];  var_mean_27 = None
    add_81: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-06);  getitem_102 = None
    rsqrt_27: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_27: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_48, getitem_103);  clone_48 = None
    mul_78: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    mul_79: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_78, primals_163);  mul_78 = None
    add_82: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_79, primals_164);  mul_79 = primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_134: "f32[1568, 320]" = torch.ops.aten.view.default(add_82, [1568, 320])
    permute_94: "f32[320, 320]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    addmm_40: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_166, view_134, permute_94);  primals_166 = None
    view_135: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_40, [8, 196, 320]);  addmm_40 = None
    view_136: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_135, [8, 196, 5, 64]);  view_135 = None
    permute_95: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_96: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_82, [0, 2, 1]);  add_82 = None
    view_137: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_96, [8, 320, 14, 14]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_14: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_137, primals_167, primals_168, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_168 = None
    view_138: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_14, [8, 320, -1]);  convolution_14 = None
    permute_97: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_138, [0, 2, 1]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_49: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_49, [2], correction = 0, keepdim = True)
    getitem_104: "f32[8, 49, 1]" = var_mean_28[0]
    getitem_105: "f32[8, 49, 1]" = var_mean_28[1];  var_mean_28 = None
    add_83: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_28: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_28: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_49, getitem_105);  clone_49 = None
    mul_80: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    mul_81: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_80, primals_169);  mul_80 = None
    add_84: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_81, primals_170);  mul_81 = primals_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_139: "f32[392, 320]" = torch.ops.aten.view.default(add_84, [392, 320]);  add_84 = None
    permute_98: "f32[320, 640]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm_41: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_172, view_139, permute_98);  primals_172 = None
    view_140: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_41, [8, 49, 640]);  addmm_41 = None
    view_141: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_140, [8, -1, 2, 5, 64]);  view_140 = None
    permute_99: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_141, [2, 0, 3, 1, 4]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_99);  permute_99 = None
    getitem_106: "f32[8, 5, 49, 64]" = unbind_8[0]
    getitem_107: "f32[8, 5, 49, 64]" = unbind_8[1];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_95, getitem_106, getitem_107, None, True)
    getitem_108: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_8[0]
    getitem_109: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_8[1]
    getitem_110: "i64[]" = _scaled_dot_product_efficient_attention_8[2]
    getitem_111: "i64[]" = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None
    alias_8: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_100: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3]);  getitem_108 = None
    view_142: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_100, [8, 196, 320]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_143: "f32[1568, 320]" = torch.ops.aten.view.default(view_142, [1568, 320]);  view_142 = None
    permute_101: "f32[320, 320]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
    addmm_42: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_174, view_143, permute_101);  primals_174 = None
    view_144: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_42, [8, 196, 320]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_50: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_85: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_93, clone_50);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_51: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_85, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_51, [2], correction = 0, keepdim = True)
    getitem_112: "f32[8, 196, 1]" = var_mean_29[0]
    getitem_113: "f32[8, 196, 1]" = var_mean_29[1];  var_mean_29 = None
    add_86: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
    rsqrt_29: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_29: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_51, getitem_113);  clone_51 = None
    mul_82: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    mul_83: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_82, primals_175);  mul_82 = None
    add_87: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_83, primals_176);  mul_83 = primals_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_145: "f32[1568, 320]" = torch.ops.aten.view.default(add_87, [1568, 320]);  add_87 = None
    permute_102: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    addmm_43: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_178, view_145, permute_102);  primals_178 = None
    view_146: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_43, [8, 196, 1280]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_84: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_146, 0.5)
    mul_85: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476)
    erf_8: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_85);  mul_85 = None
    add_88: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_86: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_84, add_88);  mul_84 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_52: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_86);  mul_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_147: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_52, [1568, 1280]);  clone_52 = None
    permute_103: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    addmm_44: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_180, view_147, permute_103);  primals_180 = None
    view_148: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_44, [8, 196, 320]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_53: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_148);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_89: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_85, clone_53);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_54: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_89, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_54, [2], correction = 0, keepdim = True)
    getitem_114: "f32[8, 196, 1]" = var_mean_30[0]
    getitem_115: "f32[8, 196, 1]" = var_mean_30[1];  var_mean_30 = None
    add_90: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-06);  getitem_114 = None
    rsqrt_30: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_30: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_54, getitem_115);  clone_54 = None
    mul_87: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    mul_88: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_87, primals_181);  mul_87 = None
    add_91: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_88, primals_182);  mul_88 = primals_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_149: "f32[1568, 320]" = torch.ops.aten.view.default(add_91, [1568, 320])
    permute_104: "f32[320, 320]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    addmm_45: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_184, view_149, permute_104);  primals_184 = None
    view_150: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_45, [8, 196, 320]);  addmm_45 = None
    view_151: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_150, [8, 196, 5, 64]);  view_150 = None
    permute_105: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_106: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_91, [0, 2, 1]);  add_91 = None
    view_152: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_106, [8, 320, 14, 14]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_15: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_152, primals_185, primals_186, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_186 = None
    view_153: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_15, [8, 320, -1]);  convolution_15 = None
    permute_107: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_55: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format)
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_55, [2], correction = 0, keepdim = True)
    getitem_116: "f32[8, 49, 1]" = var_mean_31[0]
    getitem_117: "f32[8, 49, 1]" = var_mean_31[1];  var_mean_31 = None
    add_92: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
    rsqrt_31: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_31: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_55, getitem_117);  clone_55 = None
    mul_89: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    mul_90: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_89, primals_187);  mul_89 = None
    add_93: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_90, primals_188);  mul_90 = primals_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_154: "f32[392, 320]" = torch.ops.aten.view.default(add_93, [392, 320]);  add_93 = None
    permute_108: "f32[320, 640]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    addmm_46: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_190, view_154, permute_108);  primals_190 = None
    view_155: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_46, [8, 49, 640]);  addmm_46 = None
    view_156: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_155, [8, -1, 2, 5, 64]);  view_155 = None
    permute_109: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_156, [2, 0, 3, 1, 4]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_109);  permute_109 = None
    getitem_118: "f32[8, 5, 49, 64]" = unbind_9[0]
    getitem_119: "f32[8, 5, 49, 64]" = unbind_9[1];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_105, getitem_118, getitem_119, None, True)
    getitem_120: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_9[0]
    getitem_121: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_9[1]
    getitem_122: "i64[]" = _scaled_dot_product_efficient_attention_9[2]
    getitem_123: "i64[]" = _scaled_dot_product_efficient_attention_9[3];  _scaled_dot_product_efficient_attention_9 = None
    alias_9: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_110: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_120, [0, 2, 1, 3]);  getitem_120 = None
    view_157: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_110, [8, 196, 320]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_158: "f32[1568, 320]" = torch.ops.aten.view.default(view_157, [1568, 320]);  view_157 = None
    permute_111: "f32[320, 320]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
    addmm_47: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_192, view_158, permute_111);  primals_192 = None
    view_159: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_47, [8, 196, 320]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_56: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_159);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_94: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_89, clone_56);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_57: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_57, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 196, 1]" = var_mean_32[0]
    getitem_125: "f32[8, 196, 1]" = var_mean_32[1];  var_mean_32 = None
    add_95: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-06);  getitem_124 = None
    rsqrt_32: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_32: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_57, getitem_125);  clone_57 = None
    mul_91: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    mul_92: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_91, primals_193);  mul_91 = None
    add_96: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_92, primals_194);  mul_92 = primals_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_160: "f32[1568, 320]" = torch.ops.aten.view.default(add_96, [1568, 320]);  add_96 = None
    permute_112: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    addmm_48: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_196, view_160, permute_112);  primals_196 = None
    view_161: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_48, [8, 196, 1280]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_93: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_161, 0.5)
    mul_94: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_161, 0.7071067811865476)
    erf_9: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
    add_97: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_95: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_93, add_97);  mul_93 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_58: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_95);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_162: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_58, [1568, 1280]);  clone_58 = None
    permute_113: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
    addmm_49: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_198, view_162, permute_113);  primals_198 = None
    view_163: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_49, [8, 196, 320]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_59: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_163);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_98: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_94, clone_59);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_60: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_98, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_60, [2], correction = 0, keepdim = True)
    getitem_126: "f32[8, 196, 1]" = var_mean_33[0]
    getitem_127: "f32[8, 196, 1]" = var_mean_33[1];  var_mean_33 = None
    add_99: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
    rsqrt_33: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_33: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_60, getitem_127);  clone_60 = None
    mul_96: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    mul_97: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_96, primals_199);  mul_96 = None
    add_100: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_97, primals_200);  mul_97 = primals_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_164: "f32[1568, 320]" = torch.ops.aten.view.default(add_100, [1568, 320])
    permute_114: "f32[320, 320]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
    addmm_50: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_202, view_164, permute_114);  primals_202 = None
    view_165: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_50, [8, 196, 320]);  addmm_50 = None
    view_166: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_165, [8, 196, 5, 64]);  view_165 = None
    permute_115: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_116: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_100, [0, 2, 1]);  add_100 = None
    view_167: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_116, [8, 320, 14, 14]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_16: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_167, primals_203, primals_204, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_204 = None
    view_168: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_16, [8, 320, -1]);  convolution_16 = None
    permute_117: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_61: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_61, [2], correction = 0, keepdim = True)
    getitem_128: "f32[8, 49, 1]" = var_mean_34[0]
    getitem_129: "f32[8, 49, 1]" = var_mean_34[1];  var_mean_34 = None
    add_101: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
    rsqrt_34: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_34: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_61, getitem_129);  clone_61 = None
    mul_98: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    mul_99: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_98, primals_205);  mul_98 = None
    add_102: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_99, primals_206);  mul_99 = primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_169: "f32[392, 320]" = torch.ops.aten.view.default(add_102, [392, 320]);  add_102 = None
    permute_118: "f32[320, 640]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    addmm_51: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_208, view_169, permute_118);  primals_208 = None
    view_170: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_51, [8, 49, 640]);  addmm_51 = None
    view_171: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_170, [8, -1, 2, 5, 64]);  view_170 = None
    permute_119: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_171, [2, 0, 3, 1, 4]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_119);  permute_119 = None
    getitem_130: "f32[8, 5, 49, 64]" = unbind_10[0]
    getitem_131: "f32[8, 5, 49, 64]" = unbind_10[1];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_115, getitem_130, getitem_131, None, True)
    getitem_132: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_10[0]
    getitem_133: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_10[1]
    getitem_134: "i64[]" = _scaled_dot_product_efficient_attention_10[2]
    getitem_135: "i64[]" = _scaled_dot_product_efficient_attention_10[3];  _scaled_dot_product_efficient_attention_10 = None
    alias_10: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_132)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_120: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None
    view_172: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_120, [8, 196, 320]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_173: "f32[1568, 320]" = torch.ops.aten.view.default(view_172, [1568, 320]);  view_172 = None
    permute_121: "f32[320, 320]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    addmm_52: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_210, view_173, permute_121);  primals_210 = None
    view_174: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_52, [8, 196, 320]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_62: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_174);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_103: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_98, clone_62);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_63: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_103, memory_format = torch.contiguous_format)
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_63, [2], correction = 0, keepdim = True)
    getitem_136: "f32[8, 196, 1]" = var_mean_35[0]
    getitem_137: "f32[8, 196, 1]" = var_mean_35[1];  var_mean_35 = None
    add_104: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-06);  getitem_136 = None
    rsqrt_35: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_35: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_63, getitem_137);  clone_63 = None
    mul_100: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    mul_101: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_100, primals_211);  mul_100 = None
    add_105: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_101, primals_212);  mul_101 = primals_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_175: "f32[1568, 320]" = torch.ops.aten.view.default(add_105, [1568, 320]);  add_105 = None
    permute_122: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
    addmm_53: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_214, view_175, permute_122);  primals_214 = None
    view_176: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_53, [8, 196, 1280]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_102: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_176, 0.5)
    mul_103: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476)
    erf_10: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
    add_106: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_104: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_102, add_106);  mul_102 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_64: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_104);  mul_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_177: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_64, [1568, 1280]);  clone_64 = None
    permute_123: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    addmm_54: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_216, view_177, permute_123);  primals_216 = None
    view_178: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_54, [8, 196, 320]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_65: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_178);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_107: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_103, clone_65);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_66: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_107, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_66, [2], correction = 0, keepdim = True)
    getitem_138: "f32[8, 196, 1]" = var_mean_36[0]
    getitem_139: "f32[8, 196, 1]" = var_mean_36[1];  var_mean_36 = None
    add_108: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-06);  getitem_138 = None
    rsqrt_36: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_36: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_66, getitem_139);  clone_66 = None
    mul_105: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    mul_106: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_105, primals_217);  mul_105 = None
    add_109: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_106, primals_218);  mul_106 = primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_179: "f32[1568, 320]" = torch.ops.aten.view.default(add_109, [1568, 320])
    permute_124: "f32[320, 320]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
    addmm_55: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_220, view_179, permute_124);  primals_220 = None
    view_180: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_55, [8, 196, 320]);  addmm_55 = None
    view_181: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_180, [8, 196, 5, 64]);  view_180 = None
    permute_125: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_126: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_109, [0, 2, 1]);  add_109 = None
    view_182: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_126, [8, 320, 14, 14]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_17: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_182, primals_221, primals_222, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_222 = None
    view_183: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_17, [8, 320, -1]);  convolution_17 = None
    permute_127: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_67: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format)
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_67, [2], correction = 0, keepdim = True)
    getitem_140: "f32[8, 49, 1]" = var_mean_37[0]
    getitem_141: "f32[8, 49, 1]" = var_mean_37[1];  var_mean_37 = None
    add_110: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05);  getitem_140 = None
    rsqrt_37: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_37: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_67, getitem_141);  clone_67 = None
    mul_107: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    mul_108: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_107, primals_223);  mul_107 = None
    add_111: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_108, primals_224);  mul_108 = primals_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_184: "f32[392, 320]" = torch.ops.aten.view.default(add_111, [392, 320]);  add_111 = None
    permute_128: "f32[320, 640]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
    addmm_56: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_226, view_184, permute_128);  primals_226 = None
    view_185: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_56, [8, 49, 640]);  addmm_56 = None
    view_186: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_185, [8, -1, 2, 5, 64]);  view_185 = None
    permute_129: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_186, [2, 0, 3, 1, 4]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_129);  permute_129 = None
    getitem_142: "f32[8, 5, 49, 64]" = unbind_11[0]
    getitem_143: "f32[8, 5, 49, 64]" = unbind_11[1];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_125, getitem_142, getitem_143, None, True)
    getitem_144: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_11[0]
    getitem_145: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_11[1]
    getitem_146: "i64[]" = _scaled_dot_product_efficient_attention_11[2]
    getitem_147: "i64[]" = _scaled_dot_product_efficient_attention_11[3];  _scaled_dot_product_efficient_attention_11 = None
    alias_11: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_130: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None
    view_187: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_130, [8, 196, 320]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_188: "f32[1568, 320]" = torch.ops.aten.view.default(view_187, [1568, 320]);  view_187 = None
    permute_131: "f32[320, 320]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
    addmm_57: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_228, view_188, permute_131);  primals_228 = None
    view_189: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_57, [8, 196, 320]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_68: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_189);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_112: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_107, clone_68);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_69: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_112, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_69, [2], correction = 0, keepdim = True)
    getitem_148: "f32[8, 196, 1]" = var_mean_38[0]
    getitem_149: "f32[8, 196, 1]" = var_mean_38[1];  var_mean_38 = None
    add_113: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-06);  getitem_148 = None
    rsqrt_38: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_38: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_69, getitem_149);  clone_69 = None
    mul_109: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    mul_110: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_109, primals_229);  mul_109 = None
    add_114: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_110, primals_230);  mul_110 = primals_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_190: "f32[1568, 320]" = torch.ops.aten.view.default(add_114, [1568, 320]);  add_114 = None
    permute_132: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
    addmm_58: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_232, view_190, permute_132);  primals_232 = None
    view_191: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_58, [8, 196, 1280]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_111: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_191, 0.5)
    mul_112: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_191, 0.7071067811865476)
    erf_11: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
    add_115: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_113: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_111, add_115);  mul_111 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_70: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_113);  mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_192: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_70, [1568, 1280]);  clone_70 = None
    permute_133: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_233, [1, 0]);  primals_233 = None
    addmm_59: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_234, view_192, permute_133);  primals_234 = None
    view_193: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_59, [8, 196, 320]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_71: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_193);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_116: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_112, clone_71);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_72: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_116, memory_format = torch.contiguous_format)
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_72, [2], correction = 0, keepdim = True)
    getitem_150: "f32[8, 196, 1]" = var_mean_39[0]
    getitem_151: "f32[8, 196, 1]" = var_mean_39[1];  var_mean_39 = None
    add_117: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-06);  getitem_150 = None
    rsqrt_39: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_39: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_72, getitem_151);  clone_72 = None
    mul_114: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    mul_115: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_114, primals_235);  mul_114 = None
    add_118: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_115, primals_236);  mul_115 = primals_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_194: "f32[1568, 320]" = torch.ops.aten.view.default(add_118, [1568, 320])
    permute_134: "f32[320, 320]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
    addmm_60: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_238, view_194, permute_134);  primals_238 = None
    view_195: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_60, [8, 196, 320]);  addmm_60 = None
    view_196: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_195, [8, 196, 5, 64]);  view_195 = None
    permute_135: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_136: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_118, [0, 2, 1]);  add_118 = None
    view_197: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_136, [8, 320, 14, 14]);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_18: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_197, primals_239, primals_240, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_240 = None
    view_198: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_18, [8, 320, -1]);  convolution_18 = None
    permute_137: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_198, [0, 2, 1]);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_73: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format)
    var_mean_40 = torch.ops.aten.var_mean.correction(clone_73, [2], correction = 0, keepdim = True)
    getitem_152: "f32[8, 49, 1]" = var_mean_40[0]
    getitem_153: "f32[8, 49, 1]" = var_mean_40[1];  var_mean_40 = None
    add_119: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05);  getitem_152 = None
    rsqrt_40: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_40: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_73, getitem_153);  clone_73 = None
    mul_116: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    mul_117: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_116, primals_241);  mul_116 = None
    add_120: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_117, primals_242);  mul_117 = primals_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_199: "f32[392, 320]" = torch.ops.aten.view.default(add_120, [392, 320]);  add_120 = None
    permute_138: "f32[320, 640]" = torch.ops.aten.permute.default(primals_243, [1, 0]);  primals_243 = None
    addmm_61: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_244, view_199, permute_138);  primals_244 = None
    view_200: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_61, [8, 49, 640]);  addmm_61 = None
    view_201: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_200, [8, -1, 2, 5, 64]);  view_200 = None
    permute_139: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_201, [2, 0, 3, 1, 4]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_12 = torch.ops.aten.unbind.int(permute_139);  permute_139 = None
    getitem_154: "f32[8, 5, 49, 64]" = unbind_12[0]
    getitem_155: "f32[8, 5, 49, 64]" = unbind_12[1];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_135, getitem_154, getitem_155, None, True)
    getitem_156: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_12[0]
    getitem_157: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_12[1]
    getitem_158: "i64[]" = _scaled_dot_product_efficient_attention_12[2]
    getitem_159: "i64[]" = _scaled_dot_product_efficient_attention_12[3];  _scaled_dot_product_efficient_attention_12 = None
    alias_12: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_140: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_156, [0, 2, 1, 3]);  getitem_156 = None
    view_202: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_140, [8, 196, 320]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_203: "f32[1568, 320]" = torch.ops.aten.view.default(view_202, [1568, 320]);  view_202 = None
    permute_141: "f32[320, 320]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
    addmm_62: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_246, view_203, permute_141);  primals_246 = None
    view_204: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_62, [8, 196, 320]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_74: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_204);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_121: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_116, clone_74);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_75: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_121, memory_format = torch.contiguous_format)
    var_mean_41 = torch.ops.aten.var_mean.correction(clone_75, [2], correction = 0, keepdim = True)
    getitem_160: "f32[8, 196, 1]" = var_mean_41[0]
    getitem_161: "f32[8, 196, 1]" = var_mean_41[1];  var_mean_41 = None
    add_122: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
    rsqrt_41: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_41: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_75, getitem_161);  clone_75 = None
    mul_118: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    mul_119: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_118, primals_247);  mul_118 = None
    add_123: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_119, primals_248);  mul_119 = primals_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_205: "f32[1568, 320]" = torch.ops.aten.view.default(add_123, [1568, 320]);  add_123 = None
    permute_142: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
    addmm_63: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_250, view_205, permute_142);  primals_250 = None
    view_206: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_63, [8, 196, 1280]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_120: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_206, 0.5)
    mul_121: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476)
    erf_12: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_121);  mul_121 = None
    add_124: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_122: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_120, add_124);  mul_120 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_76: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_122);  mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_207: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_76, [1568, 1280]);  clone_76 = None
    permute_143: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
    addmm_64: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_252, view_207, permute_143);  primals_252 = None
    view_208: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_64, [8, 196, 320]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_77: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_208);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_125: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_121, clone_77);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_78: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format)
    var_mean_42 = torch.ops.aten.var_mean.correction(clone_78, [2], correction = 0, keepdim = True)
    getitem_162: "f32[8, 196, 1]" = var_mean_42[0]
    getitem_163: "f32[8, 196, 1]" = var_mean_42[1];  var_mean_42 = None
    add_126: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
    rsqrt_42: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_42: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_78, getitem_163);  clone_78 = None
    mul_123: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    mul_124: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_123, primals_253);  mul_123 = None
    add_127: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_124, primals_254);  mul_124 = primals_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_209: "f32[1568, 320]" = torch.ops.aten.view.default(add_127, [1568, 320])
    permute_144: "f32[320, 320]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_65: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_256, view_209, permute_144);  primals_256 = None
    view_210: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_65, [8, 196, 320]);  addmm_65 = None
    view_211: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_210, [8, 196, 5, 64]);  view_210 = None
    permute_145: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_146: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_127, [0, 2, 1]);  add_127 = None
    view_212: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_146, [8, 320, 14, 14]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_19: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_212, primals_257, primals_258, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_258 = None
    view_213: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_19, [8, 320, -1]);  convolution_19 = None
    permute_147: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_79: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format)
    var_mean_43 = torch.ops.aten.var_mean.correction(clone_79, [2], correction = 0, keepdim = True)
    getitem_164: "f32[8, 49, 1]" = var_mean_43[0]
    getitem_165: "f32[8, 49, 1]" = var_mean_43[1];  var_mean_43 = None
    add_128: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-05);  getitem_164 = None
    rsqrt_43: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    sub_43: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_79, getitem_165);  clone_79 = None
    mul_125: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    mul_126: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_125, primals_259);  mul_125 = None
    add_129: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_126, primals_260);  mul_126 = primals_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_214: "f32[392, 320]" = torch.ops.aten.view.default(add_129, [392, 320]);  add_129 = None
    permute_148: "f32[320, 640]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
    addmm_66: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_262, view_214, permute_148);  primals_262 = None
    view_215: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_66, [8, 49, 640]);  addmm_66 = None
    view_216: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_215, [8, -1, 2, 5, 64]);  view_215 = None
    permute_149: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_216, [2, 0, 3, 1, 4]);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_13 = torch.ops.aten.unbind.int(permute_149);  permute_149 = None
    getitem_166: "f32[8, 5, 49, 64]" = unbind_13[0]
    getitem_167: "f32[8, 5, 49, 64]" = unbind_13[1];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_145, getitem_166, getitem_167, None, True)
    getitem_168: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_13[0]
    getitem_169: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_13[1]
    getitem_170: "i64[]" = _scaled_dot_product_efficient_attention_13[2]
    getitem_171: "i64[]" = _scaled_dot_product_efficient_attention_13[3];  _scaled_dot_product_efficient_attention_13 = None
    alias_13: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_168)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_150: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_168, [0, 2, 1, 3]);  getitem_168 = None
    view_217: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_150, [8, 196, 320]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_218: "f32[1568, 320]" = torch.ops.aten.view.default(view_217, [1568, 320]);  view_217 = None
    permute_151: "f32[320, 320]" = torch.ops.aten.permute.default(primals_263, [1, 0]);  primals_263 = None
    addmm_67: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_264, view_218, permute_151);  primals_264 = None
    view_219: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_67, [8, 196, 320]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_80: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_219);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_130: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_125, clone_80);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_81: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_130, memory_format = torch.contiguous_format)
    var_mean_44 = torch.ops.aten.var_mean.correction(clone_81, [2], correction = 0, keepdim = True)
    getitem_172: "f32[8, 196, 1]" = var_mean_44[0]
    getitem_173: "f32[8, 196, 1]" = var_mean_44[1];  var_mean_44 = None
    add_131: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-06);  getitem_172 = None
    rsqrt_44: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    sub_44: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_81, getitem_173);  clone_81 = None
    mul_127: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    mul_128: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_127, primals_265);  mul_127 = None
    add_132: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_128, primals_266);  mul_128 = primals_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_220: "f32[1568, 320]" = torch.ops.aten.view.default(add_132, [1568, 320]);  add_132 = None
    permute_152: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
    addmm_68: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_268, view_220, permute_152);  primals_268 = None
    view_221: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_68, [8, 196, 1280]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_129: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_221, 0.5)
    mul_130: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_221, 0.7071067811865476)
    erf_13: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_130);  mul_130 = None
    add_133: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_131: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_129, add_133);  mul_129 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_82: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_131);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_222: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_82, [1568, 1280]);  clone_82 = None
    permute_153: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
    addmm_69: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_270, view_222, permute_153);  primals_270 = None
    view_223: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_69, [8, 196, 320]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_83: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_223);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_134: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_130, clone_83);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_84: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_134, memory_format = torch.contiguous_format)
    var_mean_45 = torch.ops.aten.var_mean.correction(clone_84, [2], correction = 0, keepdim = True)
    getitem_174: "f32[8, 196, 1]" = var_mean_45[0]
    getitem_175: "f32[8, 196, 1]" = var_mean_45[1];  var_mean_45 = None
    add_135: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
    rsqrt_45: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    sub_45: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_84, getitem_175);  clone_84 = None
    mul_132: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    mul_133: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_132, primals_271);  mul_132 = None
    add_136: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_133, primals_272);  mul_133 = primals_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_224: "f32[1568, 320]" = torch.ops.aten.view.default(add_136, [1568, 320])
    permute_154: "f32[320, 320]" = torch.ops.aten.permute.default(primals_273, [1, 0]);  primals_273 = None
    addmm_70: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_274, view_224, permute_154);  primals_274 = None
    view_225: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_70, [8, 196, 320]);  addmm_70 = None
    view_226: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_225, [8, 196, 5, 64]);  view_225 = None
    permute_155: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_156: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_136, [0, 2, 1]);  add_136 = None
    view_227: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_156, [8, 320, 14, 14]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_20: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_227, primals_275, primals_276, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_276 = None
    view_228: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_20, [8, 320, -1]);  convolution_20 = None
    permute_157: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_85: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format)
    var_mean_46 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
    getitem_176: "f32[8, 49, 1]" = var_mean_46[0]
    getitem_177: "f32[8, 49, 1]" = var_mean_46[1];  var_mean_46 = None
    add_137: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
    rsqrt_46: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_46: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_85, getitem_177);  clone_85 = None
    mul_134: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    mul_135: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_134, primals_277);  mul_134 = None
    add_138: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_135, primals_278);  mul_135 = primals_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_229: "f32[392, 320]" = torch.ops.aten.view.default(add_138, [392, 320]);  add_138 = None
    permute_158: "f32[320, 640]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
    addmm_71: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_280, view_229, permute_158);  primals_280 = None
    view_230: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_71, [8, 49, 640]);  addmm_71 = None
    view_231: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_230, [8, -1, 2, 5, 64]);  view_230 = None
    permute_159: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_231, [2, 0, 3, 1, 4]);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_14 = torch.ops.aten.unbind.int(permute_159);  permute_159 = None
    getitem_178: "f32[8, 5, 49, 64]" = unbind_14[0]
    getitem_179: "f32[8, 5, 49, 64]" = unbind_14[1];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_155, getitem_178, getitem_179, None, True)
    getitem_180: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_14[0]
    getitem_181: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_14[1]
    getitem_182: "i64[]" = _scaled_dot_product_efficient_attention_14[2]
    getitem_183: "i64[]" = _scaled_dot_product_efficient_attention_14[3];  _scaled_dot_product_efficient_attention_14 = None
    alias_14: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_180)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_160: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None
    view_232: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_160, [8, 196, 320]);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_233: "f32[1568, 320]" = torch.ops.aten.view.default(view_232, [1568, 320]);  view_232 = None
    permute_161: "f32[320, 320]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
    addmm_72: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_282, view_233, permute_161);  primals_282 = None
    view_234: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_72, [8, 196, 320]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_86: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_234);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_139: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_134, clone_86);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_87: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_139, memory_format = torch.contiguous_format)
    var_mean_47 = torch.ops.aten.var_mean.correction(clone_87, [2], correction = 0, keepdim = True)
    getitem_184: "f32[8, 196, 1]" = var_mean_47[0]
    getitem_185: "f32[8, 196, 1]" = var_mean_47[1];  var_mean_47 = None
    add_140: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-06);  getitem_184 = None
    rsqrt_47: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_47: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_87, getitem_185);  clone_87 = None
    mul_136: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    mul_137: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_136, primals_283);  mul_136 = None
    add_141: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_137, primals_284);  mul_137 = primals_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_235: "f32[1568, 320]" = torch.ops.aten.view.default(add_141, [1568, 320]);  add_141 = None
    permute_162: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
    addmm_73: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_286, view_235, permute_162);  primals_286 = None
    view_236: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_73, [8, 196, 1280]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_138: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_236, 0.5)
    mul_139: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_236, 0.7071067811865476)
    erf_14: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_142: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_140: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_138, add_142);  mul_138 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_88: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_140);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_237: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_88, [1568, 1280]);  clone_88 = None
    permute_163: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_287, [1, 0]);  primals_287 = None
    addmm_74: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_288, view_237, permute_163);  primals_288 = None
    view_238: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_74, [8, 196, 320]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_89: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_238);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_143: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_139, clone_89);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_90: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_143, memory_format = torch.contiguous_format)
    var_mean_48 = torch.ops.aten.var_mean.correction(clone_90, [2], correction = 0, keepdim = True)
    getitem_186: "f32[8, 196, 1]" = var_mean_48[0]
    getitem_187: "f32[8, 196, 1]" = var_mean_48[1];  var_mean_48 = None
    add_144: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-06);  getitem_186 = None
    rsqrt_48: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_48: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_90, getitem_187);  clone_90 = None
    mul_141: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    mul_142: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_141, primals_289);  mul_141 = None
    add_145: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_142, primals_290);  mul_142 = primals_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_239: "f32[1568, 320]" = torch.ops.aten.view.default(add_145, [1568, 320])
    permute_164: "f32[320, 320]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
    addmm_75: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_292, view_239, permute_164);  primals_292 = None
    view_240: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_75, [8, 196, 320]);  addmm_75 = None
    view_241: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_240, [8, 196, 5, 64]);  view_240 = None
    permute_165: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_166: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_145, [0, 2, 1]);  add_145 = None
    view_242: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_166, [8, 320, 14, 14]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_21: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_242, primals_293, primals_294, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_294 = None
    view_243: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_21, [8, 320, -1]);  convolution_21 = None
    permute_167: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_243, [0, 2, 1]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_91: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format)
    var_mean_49 = torch.ops.aten.var_mean.correction(clone_91, [2], correction = 0, keepdim = True)
    getitem_188: "f32[8, 49, 1]" = var_mean_49[0]
    getitem_189: "f32[8, 49, 1]" = var_mean_49[1];  var_mean_49 = None
    add_146: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-05);  getitem_188 = None
    rsqrt_49: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_49: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_91, getitem_189);  clone_91 = None
    mul_143: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    mul_144: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_143, primals_295);  mul_143 = None
    add_147: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_144, primals_296);  mul_144 = primals_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_244: "f32[392, 320]" = torch.ops.aten.view.default(add_147, [392, 320]);  add_147 = None
    permute_168: "f32[320, 640]" = torch.ops.aten.permute.default(primals_297, [1, 0]);  primals_297 = None
    addmm_76: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_298, view_244, permute_168);  primals_298 = None
    view_245: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_76, [8, 49, 640]);  addmm_76 = None
    view_246: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_245, [8, -1, 2, 5, 64]);  view_245 = None
    permute_169: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_246, [2, 0, 3, 1, 4]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_15 = torch.ops.aten.unbind.int(permute_169);  permute_169 = None
    getitem_190: "f32[8, 5, 49, 64]" = unbind_15[0]
    getitem_191: "f32[8, 5, 49, 64]" = unbind_15[1];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_165, getitem_190, getitem_191, None, True)
    getitem_192: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_15[0]
    getitem_193: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_15[1]
    getitem_194: "i64[]" = _scaled_dot_product_efficient_attention_15[2]
    getitem_195: "i64[]" = _scaled_dot_product_efficient_attention_15[3];  _scaled_dot_product_efficient_attention_15 = None
    alias_15: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_192)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_170: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_192, [0, 2, 1, 3]);  getitem_192 = None
    view_247: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_170, [8, 196, 320]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_248: "f32[1568, 320]" = torch.ops.aten.view.default(view_247, [1568, 320]);  view_247 = None
    permute_171: "f32[320, 320]" = torch.ops.aten.permute.default(primals_299, [1, 0]);  primals_299 = None
    addmm_77: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_300, view_248, permute_171);  primals_300 = None
    view_249: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_77, [8, 196, 320]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_92: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_249);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_148: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_143, clone_92);  clone_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_93: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_148, memory_format = torch.contiguous_format)
    var_mean_50 = torch.ops.aten.var_mean.correction(clone_93, [2], correction = 0, keepdim = True)
    getitem_196: "f32[8, 196, 1]" = var_mean_50[0]
    getitem_197: "f32[8, 196, 1]" = var_mean_50[1];  var_mean_50 = None
    add_149: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-06);  getitem_196 = None
    rsqrt_50: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    sub_50: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_93, getitem_197);  clone_93 = None
    mul_145: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    mul_146: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_145, primals_301);  mul_145 = None
    add_150: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_146, primals_302);  mul_146 = primals_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_250: "f32[1568, 320]" = torch.ops.aten.view.default(add_150, [1568, 320]);  add_150 = None
    permute_172: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_303, [1, 0]);  primals_303 = None
    addmm_78: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_304, view_250, permute_172);  primals_304 = None
    view_251: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_78, [8, 196, 1280]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_147: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_251, 0.5)
    mul_148: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_251, 0.7071067811865476)
    erf_15: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
    add_151: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_149: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_147, add_151);  mul_147 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_94: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_149);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_252: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_94, [1568, 1280]);  clone_94 = None
    permute_173: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_305, [1, 0]);  primals_305 = None
    addmm_79: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_306, view_252, permute_173);  primals_306 = None
    view_253: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_79, [8, 196, 320]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_95: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_253);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_152: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_148, clone_95);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_96: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_152, memory_format = torch.contiguous_format)
    var_mean_51 = torch.ops.aten.var_mean.correction(clone_96, [2], correction = 0, keepdim = True)
    getitem_198: "f32[8, 196, 1]" = var_mean_51[0]
    getitem_199: "f32[8, 196, 1]" = var_mean_51[1];  var_mean_51 = None
    add_153: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-06);  getitem_198 = None
    rsqrt_51: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    sub_51: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_96, getitem_199);  clone_96 = None
    mul_150: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    mul_151: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_150, primals_307);  mul_150 = None
    add_154: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_151, primals_308);  mul_151 = primals_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_254: "f32[1568, 320]" = torch.ops.aten.view.default(add_154, [1568, 320])
    permute_174: "f32[320, 320]" = torch.ops.aten.permute.default(primals_309, [1, 0]);  primals_309 = None
    addmm_80: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_310, view_254, permute_174);  primals_310 = None
    view_255: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_80, [8, 196, 320]);  addmm_80 = None
    view_256: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_255, [8, 196, 5, 64]);  view_255 = None
    permute_175: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_176: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_154, [0, 2, 1]);  add_154 = None
    view_257: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_176, [8, 320, 14, 14]);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_22: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_257, primals_311, primals_312, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_312 = None
    view_258: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_22, [8, 320, -1]);  convolution_22 = None
    permute_177: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_258, [0, 2, 1]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_97: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format)
    var_mean_52 = torch.ops.aten.var_mean.correction(clone_97, [2], correction = 0, keepdim = True)
    getitem_200: "f32[8, 49, 1]" = var_mean_52[0]
    getitem_201: "f32[8, 49, 1]" = var_mean_52[1];  var_mean_52 = None
    add_155: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-05);  getitem_200 = None
    rsqrt_52: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_52: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_97, getitem_201);  clone_97 = None
    mul_152: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    mul_153: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_152, primals_313);  mul_152 = None
    add_156: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_153, primals_314);  mul_153 = primals_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_259: "f32[392, 320]" = torch.ops.aten.view.default(add_156, [392, 320]);  add_156 = None
    permute_178: "f32[320, 640]" = torch.ops.aten.permute.default(primals_315, [1, 0]);  primals_315 = None
    addmm_81: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_316, view_259, permute_178);  primals_316 = None
    view_260: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_81, [8, 49, 640]);  addmm_81 = None
    view_261: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_260, [8, -1, 2, 5, 64]);  view_260 = None
    permute_179: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_261, [2, 0, 3, 1, 4]);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_16 = torch.ops.aten.unbind.int(permute_179);  permute_179 = None
    getitem_202: "f32[8, 5, 49, 64]" = unbind_16[0]
    getitem_203: "f32[8, 5, 49, 64]" = unbind_16[1];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_175, getitem_202, getitem_203, None, True)
    getitem_204: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_16[0]
    getitem_205: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_16[1]
    getitem_206: "i64[]" = _scaled_dot_product_efficient_attention_16[2]
    getitem_207: "i64[]" = _scaled_dot_product_efficient_attention_16[3];  _scaled_dot_product_efficient_attention_16 = None
    alias_16: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_204)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_180: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_204, [0, 2, 1, 3]);  getitem_204 = None
    view_262: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_180, [8, 196, 320]);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_263: "f32[1568, 320]" = torch.ops.aten.view.default(view_262, [1568, 320]);  view_262 = None
    permute_181: "f32[320, 320]" = torch.ops.aten.permute.default(primals_317, [1, 0]);  primals_317 = None
    addmm_82: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_318, view_263, permute_181);  primals_318 = None
    view_264: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_82, [8, 196, 320]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_98: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_264);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_157: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_152, clone_98);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_99: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_157, memory_format = torch.contiguous_format)
    var_mean_53 = torch.ops.aten.var_mean.correction(clone_99, [2], correction = 0, keepdim = True)
    getitem_208: "f32[8, 196, 1]" = var_mean_53[0]
    getitem_209: "f32[8, 196, 1]" = var_mean_53[1];  var_mean_53 = None
    add_158: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-06);  getitem_208 = None
    rsqrt_53: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_53: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_99, getitem_209);  clone_99 = None
    mul_154: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    mul_155: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_154, primals_319);  mul_154 = None
    add_159: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_155, primals_320);  mul_155 = primals_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_265: "f32[1568, 320]" = torch.ops.aten.view.default(add_159, [1568, 320]);  add_159 = None
    permute_182: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_321, [1, 0]);  primals_321 = None
    addmm_83: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_322, view_265, permute_182);  primals_322 = None
    view_266: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_83, [8, 196, 1280]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_156: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_266, 0.5)
    mul_157: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_266, 0.7071067811865476)
    erf_16: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_157);  mul_157 = None
    add_160: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_158: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_156, add_160);  mul_156 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_100: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_158);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_267: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_100, [1568, 1280]);  clone_100 = None
    permute_183: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_323, [1, 0]);  primals_323 = None
    addmm_84: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_324, view_267, permute_183);  primals_324 = None
    view_268: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_84, [8, 196, 320]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_101: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_268);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_161: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_157, clone_101);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_102: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_161, memory_format = torch.contiguous_format)
    var_mean_54 = torch.ops.aten.var_mean.correction(clone_102, [2], correction = 0, keepdim = True)
    getitem_210: "f32[8, 196, 1]" = var_mean_54[0]
    getitem_211: "f32[8, 196, 1]" = var_mean_54[1];  var_mean_54 = None
    add_162: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-06);  getitem_210 = None
    rsqrt_54: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    sub_54: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_102, getitem_211);  clone_102 = None
    mul_159: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    mul_160: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_159, primals_325);  mul_159 = None
    add_163: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_160, primals_326);  mul_160 = primals_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_269: "f32[1568, 320]" = torch.ops.aten.view.default(add_163, [1568, 320])
    permute_184: "f32[320, 320]" = torch.ops.aten.permute.default(primals_327, [1, 0]);  primals_327 = None
    addmm_85: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_328, view_269, permute_184);  primals_328 = None
    view_270: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_85, [8, 196, 320]);  addmm_85 = None
    view_271: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_270, [8, 196, 5, 64]);  view_270 = None
    permute_185: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_186: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_163, [0, 2, 1]);  add_163 = None
    view_272: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_186, [8, 320, 14, 14]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_23: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_272, primals_329, primals_330, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_330 = None
    view_273: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_23, [8, 320, -1]);  convolution_23 = None
    permute_187: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_103: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format)
    var_mean_55 = torch.ops.aten.var_mean.correction(clone_103, [2], correction = 0, keepdim = True)
    getitem_212: "f32[8, 49, 1]" = var_mean_55[0]
    getitem_213: "f32[8, 49, 1]" = var_mean_55[1];  var_mean_55 = None
    add_164: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-05);  getitem_212 = None
    rsqrt_55: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_55: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_103, getitem_213);  clone_103 = None
    mul_161: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    mul_162: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_161, primals_331);  mul_161 = None
    add_165: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_162, primals_332);  mul_162 = primals_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_274: "f32[392, 320]" = torch.ops.aten.view.default(add_165, [392, 320]);  add_165 = None
    permute_188: "f32[320, 640]" = torch.ops.aten.permute.default(primals_333, [1, 0]);  primals_333 = None
    addmm_86: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_334, view_274, permute_188);  primals_334 = None
    view_275: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_86, [8, 49, 640]);  addmm_86 = None
    view_276: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_275, [8, -1, 2, 5, 64]);  view_275 = None
    permute_189: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_276, [2, 0, 3, 1, 4]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_17 = torch.ops.aten.unbind.int(permute_189);  permute_189 = None
    getitem_214: "f32[8, 5, 49, 64]" = unbind_17[0]
    getitem_215: "f32[8, 5, 49, 64]" = unbind_17[1];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_185, getitem_214, getitem_215, None, True)
    getitem_216: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_17[0]
    getitem_217: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_17[1]
    getitem_218: "i64[]" = _scaled_dot_product_efficient_attention_17[2]
    getitem_219: "i64[]" = _scaled_dot_product_efficient_attention_17[3];  _scaled_dot_product_efficient_attention_17 = None
    alias_17: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_190: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None
    view_277: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_190, [8, 196, 320]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_278: "f32[1568, 320]" = torch.ops.aten.view.default(view_277, [1568, 320]);  view_277 = None
    permute_191: "f32[320, 320]" = torch.ops.aten.permute.default(primals_335, [1, 0]);  primals_335 = None
    addmm_87: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_336, view_278, permute_191);  primals_336 = None
    view_279: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_87, [8, 196, 320]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_104: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_279);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_166: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_161, clone_104);  clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_105: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_166, memory_format = torch.contiguous_format)
    var_mean_56 = torch.ops.aten.var_mean.correction(clone_105, [2], correction = 0, keepdim = True)
    getitem_220: "f32[8, 196, 1]" = var_mean_56[0]
    getitem_221: "f32[8, 196, 1]" = var_mean_56[1];  var_mean_56 = None
    add_167: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-06);  getitem_220 = None
    rsqrt_56: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    sub_56: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_105, getitem_221);  clone_105 = None
    mul_163: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    mul_164: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_163, primals_337);  mul_163 = None
    add_168: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_164, primals_338);  mul_164 = primals_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_280: "f32[1568, 320]" = torch.ops.aten.view.default(add_168, [1568, 320]);  add_168 = None
    permute_192: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_339, [1, 0]);  primals_339 = None
    addmm_88: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_340, view_280, permute_192);  primals_340 = None
    view_281: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_88, [8, 196, 1280]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_165: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_281, 0.5)
    mul_166: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_281, 0.7071067811865476)
    erf_17: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_166);  mul_166 = None
    add_169: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_167: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_165, add_169);  mul_165 = add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_106: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_167);  mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_282: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_106, [1568, 1280]);  clone_106 = None
    permute_193: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_341, [1, 0]);  primals_341 = None
    addmm_89: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_342, view_282, permute_193);  primals_342 = None
    view_283: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_89, [8, 196, 320]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_107: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_283);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_170: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_166, clone_107);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_108: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_170, memory_format = torch.contiguous_format)
    var_mean_57 = torch.ops.aten.var_mean.correction(clone_108, [2], correction = 0, keepdim = True)
    getitem_222: "f32[8, 196, 1]" = var_mean_57[0]
    getitem_223: "f32[8, 196, 1]" = var_mean_57[1];  var_mean_57 = None
    add_171: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-06);  getitem_222 = None
    rsqrt_57: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_57: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_108, getitem_223);  clone_108 = None
    mul_168: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    mul_169: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_168, primals_343);  mul_168 = None
    add_172: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_169, primals_344);  mul_169 = primals_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_284: "f32[1568, 320]" = torch.ops.aten.view.default(add_172, [1568, 320])
    permute_194: "f32[320, 320]" = torch.ops.aten.permute.default(primals_345, [1, 0]);  primals_345 = None
    addmm_90: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_346, view_284, permute_194);  primals_346 = None
    view_285: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_90, [8, 196, 320]);  addmm_90 = None
    view_286: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_285, [8, 196, 5, 64]);  view_285 = None
    permute_195: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_196: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_172, [0, 2, 1]);  add_172 = None
    view_287: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_196, [8, 320, 14, 14]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_24: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_287, primals_347, primals_348, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_348 = None
    view_288: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_24, [8, 320, -1]);  convolution_24 = None
    permute_197: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_109: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format)
    var_mean_58 = torch.ops.aten.var_mean.correction(clone_109, [2], correction = 0, keepdim = True)
    getitem_224: "f32[8, 49, 1]" = var_mean_58[0]
    getitem_225: "f32[8, 49, 1]" = var_mean_58[1];  var_mean_58 = None
    add_173: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-05);  getitem_224 = None
    rsqrt_58: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    sub_58: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_109, getitem_225);  clone_109 = None
    mul_170: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    mul_171: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_170, primals_349);  mul_170 = None
    add_174: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_171, primals_350);  mul_171 = primals_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_289: "f32[392, 320]" = torch.ops.aten.view.default(add_174, [392, 320]);  add_174 = None
    permute_198: "f32[320, 640]" = torch.ops.aten.permute.default(primals_351, [1, 0]);  primals_351 = None
    addmm_91: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_352, view_289, permute_198);  primals_352 = None
    view_290: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_91, [8, 49, 640]);  addmm_91 = None
    view_291: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_290, [8, -1, 2, 5, 64]);  view_290 = None
    permute_199: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_291, [2, 0, 3, 1, 4]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_18 = torch.ops.aten.unbind.int(permute_199);  permute_199 = None
    getitem_226: "f32[8, 5, 49, 64]" = unbind_18[0]
    getitem_227: "f32[8, 5, 49, 64]" = unbind_18[1];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_195, getitem_226, getitem_227, None, True)
    getitem_228: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_18[0]
    getitem_229: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_18[1]
    getitem_230: "i64[]" = _scaled_dot_product_efficient_attention_18[2]
    getitem_231: "i64[]" = _scaled_dot_product_efficient_attention_18[3];  _scaled_dot_product_efficient_attention_18 = None
    alias_18: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_228)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_200: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_228, [0, 2, 1, 3]);  getitem_228 = None
    view_292: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_200, [8, 196, 320]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_293: "f32[1568, 320]" = torch.ops.aten.view.default(view_292, [1568, 320]);  view_292 = None
    permute_201: "f32[320, 320]" = torch.ops.aten.permute.default(primals_353, [1, 0]);  primals_353 = None
    addmm_92: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_354, view_293, permute_201);  primals_354 = None
    view_294: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_92, [8, 196, 320]);  addmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_110: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_294);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_175: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_170, clone_110);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_111: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_175, memory_format = torch.contiguous_format)
    var_mean_59 = torch.ops.aten.var_mean.correction(clone_111, [2], correction = 0, keepdim = True)
    getitem_232: "f32[8, 196, 1]" = var_mean_59[0]
    getitem_233: "f32[8, 196, 1]" = var_mean_59[1];  var_mean_59 = None
    add_176: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_232, 1e-06);  getitem_232 = None
    rsqrt_59: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_59: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_111, getitem_233);  clone_111 = None
    mul_172: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    mul_173: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_172, primals_355);  mul_172 = None
    add_177: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_173, primals_356);  mul_173 = primals_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_295: "f32[1568, 320]" = torch.ops.aten.view.default(add_177, [1568, 320]);  add_177 = None
    permute_202: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_357, [1, 0]);  primals_357 = None
    addmm_93: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_358, view_295, permute_202);  primals_358 = None
    view_296: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_93, [8, 196, 1280]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_174: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_296, 0.5)
    mul_175: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_296, 0.7071067811865476)
    erf_18: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_175);  mul_175 = None
    add_178: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_176: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_174, add_178);  mul_174 = add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_112: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_176);  mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_297: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_112, [1568, 1280]);  clone_112 = None
    permute_203: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_359, [1, 0]);  primals_359 = None
    addmm_94: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_360, view_297, permute_203);  primals_360 = None
    view_298: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_94, [8, 196, 320]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_113: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_298);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_179: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_175, clone_113);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_114: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_179, memory_format = torch.contiguous_format)
    var_mean_60 = torch.ops.aten.var_mean.correction(clone_114, [2], correction = 0, keepdim = True)
    getitem_234: "f32[8, 196, 1]" = var_mean_60[0]
    getitem_235: "f32[8, 196, 1]" = var_mean_60[1];  var_mean_60 = None
    add_180: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_234, 1e-06);  getitem_234 = None
    rsqrt_60: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
    sub_60: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_114, getitem_235);  clone_114 = None
    mul_177: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    mul_178: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_177, primals_361);  mul_177 = None
    add_181: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_178, primals_362);  mul_178 = primals_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_299: "f32[1568, 320]" = torch.ops.aten.view.default(add_181, [1568, 320])
    permute_204: "f32[320, 320]" = torch.ops.aten.permute.default(primals_363, [1, 0]);  primals_363 = None
    addmm_95: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_364, view_299, permute_204);  primals_364 = None
    view_300: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_95, [8, 196, 320]);  addmm_95 = None
    view_301: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_300, [8, 196, 5, 64]);  view_300 = None
    permute_205: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_206: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_181, [0, 2, 1]);  add_181 = None
    view_302: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_206, [8, 320, 14, 14]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_25: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_302, primals_365, primals_366, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_366 = None
    view_303: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_25, [8, 320, -1]);  convolution_25 = None
    permute_207: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_115: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format)
    var_mean_61 = torch.ops.aten.var_mean.correction(clone_115, [2], correction = 0, keepdim = True)
    getitem_236: "f32[8, 49, 1]" = var_mean_61[0]
    getitem_237: "f32[8, 49, 1]" = var_mean_61[1];  var_mean_61 = None
    add_182: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_236, 1e-05);  getitem_236 = None
    rsqrt_61: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_61: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_115, getitem_237);  clone_115 = None
    mul_179: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    mul_180: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_179, primals_367);  mul_179 = None
    add_183: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_180, primals_368);  mul_180 = primals_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_304: "f32[392, 320]" = torch.ops.aten.view.default(add_183, [392, 320]);  add_183 = None
    permute_208: "f32[320, 640]" = torch.ops.aten.permute.default(primals_369, [1, 0]);  primals_369 = None
    addmm_96: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_370, view_304, permute_208);  primals_370 = None
    view_305: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_96, [8, 49, 640]);  addmm_96 = None
    view_306: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_305, [8, -1, 2, 5, 64]);  view_305 = None
    permute_209: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_306, [2, 0, 3, 1, 4]);  view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_19 = torch.ops.aten.unbind.int(permute_209);  permute_209 = None
    getitem_238: "f32[8, 5, 49, 64]" = unbind_19[0]
    getitem_239: "f32[8, 5, 49, 64]" = unbind_19[1];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_205, getitem_238, getitem_239, None, True)
    getitem_240: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_19[0]
    getitem_241: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_19[1]
    getitem_242: "i64[]" = _scaled_dot_product_efficient_attention_19[2]
    getitem_243: "i64[]" = _scaled_dot_product_efficient_attention_19[3];  _scaled_dot_product_efficient_attention_19 = None
    alias_19: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_210: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_240, [0, 2, 1, 3]);  getitem_240 = None
    view_307: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_210, [8, 196, 320]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_308: "f32[1568, 320]" = torch.ops.aten.view.default(view_307, [1568, 320]);  view_307 = None
    permute_211: "f32[320, 320]" = torch.ops.aten.permute.default(primals_371, [1, 0]);  primals_371 = None
    addmm_97: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_372, view_308, permute_211);  primals_372 = None
    view_309: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_97, [8, 196, 320]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_116: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_309);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_184: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_179, clone_116);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_117: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_184, memory_format = torch.contiguous_format)
    var_mean_62 = torch.ops.aten.var_mean.correction(clone_117, [2], correction = 0, keepdim = True)
    getitem_244: "f32[8, 196, 1]" = var_mean_62[0]
    getitem_245: "f32[8, 196, 1]" = var_mean_62[1];  var_mean_62 = None
    add_185: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_244, 1e-06);  getitem_244 = None
    rsqrt_62: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_62: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_117, getitem_245);  clone_117 = None
    mul_181: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
    mul_182: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_181, primals_373);  mul_181 = None
    add_186: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_182, primals_374);  mul_182 = primals_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_310: "f32[1568, 320]" = torch.ops.aten.view.default(add_186, [1568, 320]);  add_186 = None
    permute_212: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_375, [1, 0]);  primals_375 = None
    addmm_98: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_376, view_310, permute_212);  primals_376 = None
    view_311: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_98, [8, 196, 1280]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_183: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_311, 0.5)
    mul_184: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_311, 0.7071067811865476)
    erf_19: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_184);  mul_184 = None
    add_187: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_185: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_183, add_187);  mul_183 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_118: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_185);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_312: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_118, [1568, 1280]);  clone_118 = None
    permute_213: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_377, [1, 0]);  primals_377 = None
    addmm_99: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_378, view_312, permute_213);  primals_378 = None
    view_313: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_99, [8, 196, 320]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_119: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_313);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_188: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_184, clone_119);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_120: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_188, memory_format = torch.contiguous_format)
    var_mean_63 = torch.ops.aten.var_mean.correction(clone_120, [2], correction = 0, keepdim = True)
    getitem_246: "f32[8, 196, 1]" = var_mean_63[0]
    getitem_247: "f32[8, 196, 1]" = var_mean_63[1];  var_mean_63 = None
    add_189: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-06);  getitem_246 = None
    rsqrt_63: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_63: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_120, getitem_247);  clone_120 = None
    mul_186: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
    mul_187: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_186, primals_379);  mul_186 = None
    add_190: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_187, primals_380);  mul_187 = primals_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_314: "f32[1568, 320]" = torch.ops.aten.view.default(add_190, [1568, 320])
    permute_214: "f32[320, 320]" = torch.ops.aten.permute.default(primals_381, [1, 0]);  primals_381 = None
    addmm_100: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_382, view_314, permute_214);  primals_382 = None
    view_315: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_100, [8, 196, 320]);  addmm_100 = None
    view_316: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_315, [8, 196, 5, 64]);  view_315 = None
    permute_215: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_216: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_190, [0, 2, 1]);  add_190 = None
    view_317: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_216, [8, 320, 14, 14]);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_26: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_317, primals_383, primals_384, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_384 = None
    view_318: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_26, [8, 320, -1]);  convolution_26 = None
    permute_217: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_121: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format)
    var_mean_64 = torch.ops.aten.var_mean.correction(clone_121, [2], correction = 0, keepdim = True)
    getitem_248: "f32[8, 49, 1]" = var_mean_64[0]
    getitem_249: "f32[8, 49, 1]" = var_mean_64[1];  var_mean_64 = None
    add_191: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_248, 1e-05);  getitem_248 = None
    rsqrt_64: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_64: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_121, getitem_249);  clone_121 = None
    mul_188: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
    mul_189: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_188, primals_385);  mul_188 = None
    add_192: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_189, primals_386);  mul_189 = primals_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_319: "f32[392, 320]" = torch.ops.aten.view.default(add_192, [392, 320]);  add_192 = None
    permute_218: "f32[320, 640]" = torch.ops.aten.permute.default(primals_387, [1, 0]);  primals_387 = None
    addmm_101: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_388, view_319, permute_218);  primals_388 = None
    view_320: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_101, [8, 49, 640]);  addmm_101 = None
    view_321: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_320, [8, -1, 2, 5, 64]);  view_320 = None
    permute_219: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_321, [2, 0, 3, 1, 4]);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_20 = torch.ops.aten.unbind.int(permute_219);  permute_219 = None
    getitem_250: "f32[8, 5, 49, 64]" = unbind_20[0]
    getitem_251: "f32[8, 5, 49, 64]" = unbind_20[1];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_215, getitem_250, getitem_251, None, True)
    getitem_252: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_20[0]
    getitem_253: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_20[1]
    getitem_254: "i64[]" = _scaled_dot_product_efficient_attention_20[2]
    getitem_255: "i64[]" = _scaled_dot_product_efficient_attention_20[3];  _scaled_dot_product_efficient_attention_20 = None
    alias_20: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_252)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_220: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_252, [0, 2, 1, 3]);  getitem_252 = None
    view_322: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_220, [8, 196, 320]);  permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_323: "f32[1568, 320]" = torch.ops.aten.view.default(view_322, [1568, 320]);  view_322 = None
    permute_221: "f32[320, 320]" = torch.ops.aten.permute.default(primals_389, [1, 0]);  primals_389 = None
    addmm_102: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_390, view_323, permute_221);  primals_390 = None
    view_324: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_102, [8, 196, 320]);  addmm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_122: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_324);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_193: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_188, clone_122);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_123: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_193, memory_format = torch.contiguous_format)
    var_mean_65 = torch.ops.aten.var_mean.correction(clone_123, [2], correction = 0, keepdim = True)
    getitem_256: "f32[8, 196, 1]" = var_mean_65[0]
    getitem_257: "f32[8, 196, 1]" = var_mean_65[1];  var_mean_65 = None
    add_194: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_256, 1e-06);  getitem_256 = None
    rsqrt_65: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_65: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_123, getitem_257);  clone_123 = None
    mul_190: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
    mul_191: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_190, primals_391);  mul_190 = None
    add_195: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_191, primals_392);  mul_191 = primals_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_325: "f32[1568, 320]" = torch.ops.aten.view.default(add_195, [1568, 320]);  add_195 = None
    permute_222: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_393, [1, 0]);  primals_393 = None
    addmm_103: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_394, view_325, permute_222);  primals_394 = None
    view_326: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_103, [8, 196, 1280]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_192: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_326, 0.5)
    mul_193: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_326, 0.7071067811865476)
    erf_20: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_193);  mul_193 = None
    add_196: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_194: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_192, add_196);  mul_192 = add_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_124: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_194);  mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_327: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_124, [1568, 1280]);  clone_124 = None
    permute_223: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_395, [1, 0]);  primals_395 = None
    addmm_104: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_396, view_327, permute_223);  primals_396 = None
    view_328: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_104, [8, 196, 320]);  addmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_125: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_328);  view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_197: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_193, clone_125);  clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_126: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_197, memory_format = torch.contiguous_format)
    var_mean_66 = torch.ops.aten.var_mean.correction(clone_126, [2], correction = 0, keepdim = True)
    getitem_258: "f32[8, 196, 1]" = var_mean_66[0]
    getitem_259: "f32[8, 196, 1]" = var_mean_66[1];  var_mean_66 = None
    add_198: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_258, 1e-06);  getitem_258 = None
    rsqrt_66: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    sub_66: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_126, getitem_259);  clone_126 = None
    mul_195: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
    mul_196: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_195, primals_397);  mul_195 = None
    add_199: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_196, primals_398);  mul_196 = primals_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_329: "f32[1568, 320]" = torch.ops.aten.view.default(add_199, [1568, 320])
    permute_224: "f32[320, 320]" = torch.ops.aten.permute.default(primals_399, [1, 0]);  primals_399 = None
    addmm_105: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_400, view_329, permute_224);  primals_400 = None
    view_330: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_105, [8, 196, 320]);  addmm_105 = None
    view_331: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_330, [8, 196, 5, 64]);  view_330 = None
    permute_225: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_226: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_199, [0, 2, 1]);  add_199 = None
    view_332: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_226, [8, 320, 14, 14]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_27: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_332, primals_401, primals_402, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_402 = None
    view_333: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_27, [8, 320, -1]);  convolution_27 = None
    permute_227: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_127: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format)
    var_mean_67 = torch.ops.aten.var_mean.correction(clone_127, [2], correction = 0, keepdim = True)
    getitem_260: "f32[8, 49, 1]" = var_mean_67[0]
    getitem_261: "f32[8, 49, 1]" = var_mean_67[1];  var_mean_67 = None
    add_200: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_260, 1e-05);  getitem_260 = None
    rsqrt_67: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
    sub_67: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_127, getitem_261);  clone_127 = None
    mul_197: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
    mul_198: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_197, primals_403);  mul_197 = None
    add_201: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_198, primals_404);  mul_198 = primals_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_334: "f32[392, 320]" = torch.ops.aten.view.default(add_201, [392, 320]);  add_201 = None
    permute_228: "f32[320, 640]" = torch.ops.aten.permute.default(primals_405, [1, 0]);  primals_405 = None
    addmm_106: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_406, view_334, permute_228);  primals_406 = None
    view_335: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_106, [8, 49, 640]);  addmm_106 = None
    view_336: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_335, [8, -1, 2, 5, 64]);  view_335 = None
    permute_229: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_336, [2, 0, 3, 1, 4]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_21 = torch.ops.aten.unbind.int(permute_229);  permute_229 = None
    getitem_262: "f32[8, 5, 49, 64]" = unbind_21[0]
    getitem_263: "f32[8, 5, 49, 64]" = unbind_21[1];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_225, getitem_262, getitem_263, None, True)
    getitem_264: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_21[0]
    getitem_265: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_21[1]
    getitem_266: "i64[]" = _scaled_dot_product_efficient_attention_21[2]
    getitem_267: "i64[]" = _scaled_dot_product_efficient_attention_21[3];  _scaled_dot_product_efficient_attention_21 = None
    alias_21: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_264)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_230: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_264, [0, 2, 1, 3]);  getitem_264 = None
    view_337: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_230, [8, 196, 320]);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_338: "f32[1568, 320]" = torch.ops.aten.view.default(view_337, [1568, 320]);  view_337 = None
    permute_231: "f32[320, 320]" = torch.ops.aten.permute.default(primals_407, [1, 0]);  primals_407 = None
    addmm_107: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_408, view_338, permute_231);  primals_408 = None
    view_339: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_107, [8, 196, 320]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_128: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_339);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_202: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_197, clone_128);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_129: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_202, memory_format = torch.contiguous_format)
    var_mean_68 = torch.ops.aten.var_mean.correction(clone_129, [2], correction = 0, keepdim = True)
    getitem_268: "f32[8, 196, 1]" = var_mean_68[0]
    getitem_269: "f32[8, 196, 1]" = var_mean_68[1];  var_mean_68 = None
    add_203: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_268, 1e-06);  getitem_268 = None
    rsqrt_68: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    sub_68: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_129, getitem_269);  clone_129 = None
    mul_199: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
    mul_200: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_199, primals_409);  mul_199 = None
    add_204: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_200, primals_410);  mul_200 = primals_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_340: "f32[1568, 320]" = torch.ops.aten.view.default(add_204, [1568, 320]);  add_204 = None
    permute_232: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_411, [1, 0]);  primals_411 = None
    addmm_108: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_412, view_340, permute_232);  primals_412 = None
    view_341: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_108, [8, 196, 1280]);  addmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_201: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_341, 0.5)
    mul_202: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_341, 0.7071067811865476)
    erf_21: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_202);  mul_202 = None
    add_205: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_203: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_201, add_205);  mul_201 = add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_130: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_203);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_342: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_130, [1568, 1280]);  clone_130 = None
    permute_233: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_413, [1, 0]);  primals_413 = None
    addmm_109: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_414, view_342, permute_233);  primals_414 = None
    view_343: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_109, [8, 196, 320]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_131: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_343);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_206: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_202, clone_131);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_132: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_206, memory_format = torch.contiguous_format)
    var_mean_69 = torch.ops.aten.var_mean.correction(clone_132, [2], correction = 0, keepdim = True)
    getitem_270: "f32[8, 196, 1]" = var_mean_69[0]
    getitem_271: "f32[8, 196, 1]" = var_mean_69[1];  var_mean_69 = None
    add_207: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_270, 1e-06);  getitem_270 = None
    rsqrt_69: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
    sub_69: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_132, getitem_271);  clone_132 = None
    mul_204: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
    mul_205: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_204, primals_415);  mul_204 = None
    add_208: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_205, primals_416);  mul_205 = primals_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_344: "f32[1568, 320]" = torch.ops.aten.view.default(add_208, [1568, 320])
    permute_234: "f32[320, 320]" = torch.ops.aten.permute.default(primals_417, [1, 0]);  primals_417 = None
    addmm_110: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_418, view_344, permute_234);  primals_418 = None
    view_345: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_110, [8, 196, 320]);  addmm_110 = None
    view_346: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_345, [8, 196, 5, 64]);  view_345 = None
    permute_235: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_236: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_208, [0, 2, 1]);  add_208 = None
    view_347: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_236, [8, 320, 14, 14]);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_28: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_347, primals_419, primals_420, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_420 = None
    view_348: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_28, [8, 320, -1]);  convolution_28 = None
    permute_237: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_348, [0, 2, 1]);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_133: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format)
    var_mean_70 = torch.ops.aten.var_mean.correction(clone_133, [2], correction = 0, keepdim = True)
    getitem_272: "f32[8, 49, 1]" = var_mean_70[0]
    getitem_273: "f32[8, 49, 1]" = var_mean_70[1];  var_mean_70 = None
    add_209: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_272, 1e-05);  getitem_272 = None
    rsqrt_70: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
    sub_70: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_133, getitem_273);  clone_133 = None
    mul_206: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
    mul_207: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_206, primals_421);  mul_206 = None
    add_210: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_207, primals_422);  mul_207 = primals_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_349: "f32[392, 320]" = torch.ops.aten.view.default(add_210, [392, 320]);  add_210 = None
    permute_238: "f32[320, 640]" = torch.ops.aten.permute.default(primals_423, [1, 0]);  primals_423 = None
    addmm_111: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_424, view_349, permute_238);  primals_424 = None
    view_350: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_111, [8, 49, 640]);  addmm_111 = None
    view_351: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_350, [8, -1, 2, 5, 64]);  view_350 = None
    permute_239: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_351, [2, 0, 3, 1, 4]);  view_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_22 = torch.ops.aten.unbind.int(permute_239);  permute_239 = None
    getitem_274: "f32[8, 5, 49, 64]" = unbind_22[0]
    getitem_275: "f32[8, 5, 49, 64]" = unbind_22[1];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_235, getitem_274, getitem_275, None, True)
    getitem_276: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_22[0]
    getitem_277: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_22[1]
    getitem_278: "i64[]" = _scaled_dot_product_efficient_attention_22[2]
    getitem_279: "i64[]" = _scaled_dot_product_efficient_attention_22[3];  _scaled_dot_product_efficient_attention_22 = None
    alias_22: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_276)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_240: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_276, [0, 2, 1, 3]);  getitem_276 = None
    view_352: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_240, [8, 196, 320]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_353: "f32[1568, 320]" = torch.ops.aten.view.default(view_352, [1568, 320]);  view_352 = None
    permute_241: "f32[320, 320]" = torch.ops.aten.permute.default(primals_425, [1, 0]);  primals_425 = None
    addmm_112: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_426, view_353, permute_241);  primals_426 = None
    view_354: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_112, [8, 196, 320]);  addmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_134: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_354);  view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_211: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_206, clone_134);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_135: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_211, memory_format = torch.contiguous_format)
    var_mean_71 = torch.ops.aten.var_mean.correction(clone_135, [2], correction = 0, keepdim = True)
    getitem_280: "f32[8, 196, 1]" = var_mean_71[0]
    getitem_281: "f32[8, 196, 1]" = var_mean_71[1];  var_mean_71 = None
    add_212: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_280, 1e-06);  getitem_280 = None
    rsqrt_71: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
    sub_71: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_135, getitem_281);  clone_135 = None
    mul_208: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
    mul_209: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_208, primals_427);  mul_208 = None
    add_213: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_209, primals_428);  mul_209 = primals_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_355: "f32[1568, 320]" = torch.ops.aten.view.default(add_213, [1568, 320]);  add_213 = None
    permute_242: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_429, [1, 0]);  primals_429 = None
    addmm_113: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_430, view_355, permute_242);  primals_430 = None
    view_356: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_113, [8, 196, 1280]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_210: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_356, 0.5)
    mul_211: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_356, 0.7071067811865476)
    erf_22: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
    add_214: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_212: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_210, add_214);  mul_210 = add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_136: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_212);  mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_357: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_136, [1568, 1280]);  clone_136 = None
    permute_243: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_431, [1, 0]);  primals_431 = None
    addmm_114: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_432, view_357, permute_243);  primals_432 = None
    view_358: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_114, [8, 196, 320]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_137: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_358);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_215: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_211, clone_137);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_138: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_215, memory_format = torch.contiguous_format)
    var_mean_72 = torch.ops.aten.var_mean.correction(clone_138, [2], correction = 0, keepdim = True)
    getitem_282: "f32[8, 196, 1]" = var_mean_72[0]
    getitem_283: "f32[8, 196, 1]" = var_mean_72[1];  var_mean_72 = None
    add_216: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_282, 1e-06);  getitem_282 = None
    rsqrt_72: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
    sub_72: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_138, getitem_283);  clone_138 = None
    mul_213: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
    mul_214: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_213, primals_433);  mul_213 = None
    add_217: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_214, primals_434);  mul_214 = primals_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_359: "f32[1568, 320]" = torch.ops.aten.view.default(add_217, [1568, 320])
    permute_244: "f32[320, 320]" = torch.ops.aten.permute.default(primals_435, [1, 0]);  primals_435 = None
    addmm_115: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_436, view_359, permute_244);  primals_436 = None
    view_360: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_115, [8, 196, 320]);  addmm_115 = None
    view_361: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_360, [8, 196, 5, 64]);  view_360 = None
    permute_245: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_246: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_217, [0, 2, 1]);  add_217 = None
    view_362: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_246, [8, 320, 14, 14]);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_29: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_362, primals_437, primals_438, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_438 = None
    view_363: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_29, [8, 320, -1]);  convolution_29 = None
    permute_247: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_139: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format)
    var_mean_73 = torch.ops.aten.var_mean.correction(clone_139, [2], correction = 0, keepdim = True)
    getitem_284: "f32[8, 49, 1]" = var_mean_73[0]
    getitem_285: "f32[8, 49, 1]" = var_mean_73[1];  var_mean_73 = None
    add_218: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_284, 1e-05);  getitem_284 = None
    rsqrt_73: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    sub_73: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_139, getitem_285);  clone_139 = None
    mul_215: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
    mul_216: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_215, primals_439);  mul_215 = None
    add_219: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_216, primals_440);  mul_216 = primals_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_364: "f32[392, 320]" = torch.ops.aten.view.default(add_219, [392, 320]);  add_219 = None
    permute_248: "f32[320, 640]" = torch.ops.aten.permute.default(primals_441, [1, 0]);  primals_441 = None
    addmm_116: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_442, view_364, permute_248);  primals_442 = None
    view_365: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_116, [8, 49, 640]);  addmm_116 = None
    view_366: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_365, [8, -1, 2, 5, 64]);  view_365 = None
    permute_249: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_366, [2, 0, 3, 1, 4]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_23 = torch.ops.aten.unbind.int(permute_249);  permute_249 = None
    getitem_286: "f32[8, 5, 49, 64]" = unbind_23[0]
    getitem_287: "f32[8, 5, 49, 64]" = unbind_23[1];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_245, getitem_286, getitem_287, None, True)
    getitem_288: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_23[0]
    getitem_289: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_23[1]
    getitem_290: "i64[]" = _scaled_dot_product_efficient_attention_23[2]
    getitem_291: "i64[]" = _scaled_dot_product_efficient_attention_23[3];  _scaled_dot_product_efficient_attention_23 = None
    alias_23: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_288)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_250: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_288, [0, 2, 1, 3]);  getitem_288 = None
    view_367: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_250, [8, 196, 320]);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_368: "f32[1568, 320]" = torch.ops.aten.view.default(view_367, [1568, 320]);  view_367 = None
    permute_251: "f32[320, 320]" = torch.ops.aten.permute.default(primals_443, [1, 0]);  primals_443 = None
    addmm_117: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_444, view_368, permute_251);  primals_444 = None
    view_369: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_117, [8, 196, 320]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_140: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_369);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_220: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_215, clone_140);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_141: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_220, memory_format = torch.contiguous_format)
    var_mean_74 = torch.ops.aten.var_mean.correction(clone_141, [2], correction = 0, keepdim = True)
    getitem_292: "f32[8, 196, 1]" = var_mean_74[0]
    getitem_293: "f32[8, 196, 1]" = var_mean_74[1];  var_mean_74 = None
    add_221: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_292, 1e-06);  getitem_292 = None
    rsqrt_74: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
    sub_74: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_141, getitem_293);  clone_141 = None
    mul_217: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
    mul_218: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_217, primals_445);  mul_217 = None
    add_222: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_218, primals_446);  mul_218 = primals_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_370: "f32[1568, 320]" = torch.ops.aten.view.default(add_222, [1568, 320]);  add_222 = None
    permute_252: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_447, [1, 0]);  primals_447 = None
    addmm_118: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_448, view_370, permute_252);  primals_448 = None
    view_371: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_118, [8, 196, 1280]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_219: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_371, 0.5)
    mul_220: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476)
    erf_23: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_220);  mul_220 = None
    add_223: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_221: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_219, add_223);  mul_219 = add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_142: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_221);  mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_372: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_142, [1568, 1280]);  clone_142 = None
    permute_253: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_449, [1, 0]);  primals_449 = None
    addmm_119: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_450, view_372, permute_253);  primals_450 = None
    view_373: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_119, [8, 196, 320]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_143: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_373);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_224: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_220, clone_143);  clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_144: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_224, memory_format = torch.contiguous_format)
    var_mean_75 = torch.ops.aten.var_mean.correction(clone_144, [2], correction = 0, keepdim = True)
    getitem_294: "f32[8, 196, 1]" = var_mean_75[0]
    getitem_295: "f32[8, 196, 1]" = var_mean_75[1];  var_mean_75 = None
    add_225: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_294, 1e-06);  getitem_294 = None
    rsqrt_75: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
    sub_75: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_144, getitem_295);  clone_144 = None
    mul_222: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
    mul_223: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_222, primals_451);  mul_222 = None
    add_226: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_223, primals_452);  mul_223 = primals_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_374: "f32[1568, 320]" = torch.ops.aten.view.default(add_226, [1568, 320])
    permute_254: "f32[320, 320]" = torch.ops.aten.permute.default(primals_453, [1, 0]);  primals_453 = None
    addmm_120: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_454, view_374, permute_254);  primals_454 = None
    view_375: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_120, [8, 196, 320]);  addmm_120 = None
    view_376: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_375, [8, 196, 5, 64]);  view_375 = None
    permute_255: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_256: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_226, [0, 2, 1]);  add_226 = None
    view_377: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_256, [8, 320, 14, 14]);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    convolution_30: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(view_377, primals_455, primals_456, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_456 = None
    view_378: "f32[8, 320, 49]" = torch.ops.aten.view.default(convolution_30, [8, 320, -1]);  convolution_30 = None
    permute_257: "f32[8, 49, 320]" = torch.ops.aten.permute.default(view_378, [0, 2, 1]);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_145: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format)
    var_mean_76 = torch.ops.aten.var_mean.correction(clone_145, [2], correction = 0, keepdim = True)
    getitem_296: "f32[8, 49, 1]" = var_mean_76[0]
    getitem_297: "f32[8, 49, 1]" = var_mean_76[1];  var_mean_76 = None
    add_227: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_296, 1e-05);  getitem_296 = None
    rsqrt_76: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
    sub_76: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_145, getitem_297);  clone_145 = None
    mul_224: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
    mul_225: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_224, primals_457);  mul_224 = None
    add_228: "f32[8, 49, 320]" = torch.ops.aten.add.Tensor(mul_225, primals_458);  mul_225 = primals_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_379: "f32[392, 320]" = torch.ops.aten.view.default(add_228, [392, 320]);  add_228 = None
    permute_258: "f32[320, 640]" = torch.ops.aten.permute.default(primals_459, [1, 0]);  primals_459 = None
    addmm_121: "f32[392, 640]" = torch.ops.aten.addmm.default(primals_460, view_379, permute_258);  primals_460 = None
    view_380: "f32[8, 49, 640]" = torch.ops.aten.view.default(addmm_121, [8, 49, 640]);  addmm_121 = None
    view_381: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.view.default(view_380, [8, -1, 2, 5, 64]);  view_380 = None
    permute_259: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.permute.default(view_381, [2, 0, 3, 1, 4]);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_24 = torch.ops.aten.unbind.int(permute_259);  permute_259 = None
    getitem_298: "f32[8, 5, 49, 64]" = unbind_24[0]
    getitem_299: "f32[8, 5, 49, 64]" = unbind_24[1];  unbind_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_24 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_255, getitem_298, getitem_299, None, True)
    getitem_300: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_24[0]
    getitem_301: "f32[8, 5, 224]" = _scaled_dot_product_efficient_attention_24[1]
    getitem_302: "i64[]" = _scaled_dot_product_efficient_attention_24[2]
    getitem_303: "i64[]" = _scaled_dot_product_efficient_attention_24[3];  _scaled_dot_product_efficient_attention_24 = None
    alias_24: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(getitem_300)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_260: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_300, [0, 2, 1, 3]);  getitem_300 = None
    view_382: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_260, [8, 196, 320]);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_383: "f32[1568, 320]" = torch.ops.aten.view.default(view_382, [1568, 320]);  view_382 = None
    permute_261: "f32[320, 320]" = torch.ops.aten.permute.default(primals_461, [1, 0]);  primals_461 = None
    addmm_122: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_462, view_383, permute_261);  primals_462 = None
    view_384: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_122, [8, 196, 320]);  addmm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_146: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_384);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_229: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_224, clone_146);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_147: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_229, memory_format = torch.contiguous_format)
    var_mean_77 = torch.ops.aten.var_mean.correction(clone_147, [2], correction = 0, keepdim = True)
    getitem_304: "f32[8, 196, 1]" = var_mean_77[0]
    getitem_305: "f32[8, 196, 1]" = var_mean_77[1];  var_mean_77 = None
    add_230: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_304, 1e-06);  getitem_304 = None
    rsqrt_77: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_230);  add_230 = None
    sub_77: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_147, getitem_305);  clone_147 = None
    mul_226: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
    mul_227: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_226, primals_463);  mul_226 = None
    add_231: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_227, primals_464);  mul_227 = primals_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_385: "f32[1568, 320]" = torch.ops.aten.view.default(add_231, [1568, 320]);  add_231 = None
    permute_262: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_465, [1, 0]);  primals_465 = None
    addmm_123: "f32[1568, 1280]" = torch.ops.aten.addmm.default(primals_466, view_385, permute_262);  primals_466 = None
    view_386: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_123, [8, 196, 1280]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_228: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_386, 0.5)
    mul_229: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_386, 0.7071067811865476)
    erf_24: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_229);  mul_229 = None
    add_232: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_230: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_228, add_232);  mul_228 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_148: "f32[8, 196, 1280]" = torch.ops.aten.clone.default(mul_230);  mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_387: "f32[1568, 1280]" = torch.ops.aten.view.default(clone_148, [1568, 1280]);  clone_148 = None
    permute_263: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_467, [1, 0]);  primals_467 = None
    addmm_124: "f32[1568, 320]" = torch.ops.aten.addmm.default(primals_468, view_387, permute_263);  primals_468 = None
    view_388: "f32[8, 196, 320]" = torch.ops.aten.view.default(addmm_124, [8, 196, 320]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_149: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_388);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_233: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_229, clone_149);  clone_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:415, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
    view_389: "f32[8, 14, 14, 320]" = torch.ops.aten.view.default(add_233, [8, 14, 14, 320]);  add_233 = None
    permute_264: "f32[8, 320, 14, 14]" = torch.ops.aten.permute.default(view_389, [0, 3, 1, 2]);  view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    convolution_31: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(permute_264, primals_469, primals_470, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_470 = None
    view_390: "f32[8, 512, 49]" = torch.ops.aten.view.default(convolution_31, [8, 512, 49]);  convolution_31 = None
    permute_265: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_390, [0, 2, 1]);  view_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone_150: "f32[8, 49, 512]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format)
    var_mean_78 = torch.ops.aten.var_mean.correction(clone_150, [2], correction = 0, keepdim = True)
    getitem_306: "f32[8, 49, 1]" = var_mean_78[0]
    getitem_307: "f32[8, 49, 1]" = var_mean_78[1];  var_mean_78 = None
    add_234: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_306, 1e-05);  getitem_306 = None
    rsqrt_78: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
    sub_78: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_150, getitem_307);  clone_150 = None
    mul_231: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
    mul_232: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_231, primals_471);  mul_231 = None
    add_235: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_232, primals_472);  mul_232 = primals_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:409, code: x = drop(x)
    clone_151: "f32[8, 49, 512]" = torch.ops.aten.clone.default(add_235);  add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    var_mean_79 = torch.ops.aten.var_mean.correction(clone_151, [2], correction = 0, keepdim = True)
    getitem_308: "f32[8, 49, 1]" = var_mean_79[0]
    getitem_309: "f32[8, 49, 1]" = var_mean_79[1];  var_mean_79 = None
    add_236: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_308, 1e-06);  getitem_308 = None
    rsqrt_79: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
    sub_79: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_151, getitem_309)
    mul_233: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
    mul_234: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_233, primals_473);  mul_233 = None
    add_237: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_234, primals_474);  mul_234 = primals_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_391: "f32[392, 512]" = torch.ops.aten.view.default(add_237, [392, 512])
    permute_266: "f32[512, 512]" = torch.ops.aten.permute.default(primals_475, [1, 0]);  primals_475 = None
    addmm_125: "f32[392, 512]" = torch.ops.aten.addmm.default(primals_476, view_391, permute_266);  primals_476 = None
    view_392: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_125, [8, 49, 512]);  addmm_125 = None
    view_393: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_392, [8, 49, 8, 64]);  view_392 = None
    permute_267: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_394: "f32[392, 512]" = torch.ops.aten.view.default(add_237, [392, 512]);  add_237 = None
    permute_268: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_477, [1, 0]);  primals_477 = None
    addmm_126: "f32[392, 1024]" = torch.ops.aten.addmm.default(primals_478, view_394, permute_268);  primals_478 = None
    view_395: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_126, [8, 49, 1024]);  addmm_126 = None
    view_396: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.view.default(view_395, [8, -1, 2, 8, 64]);  view_395 = None
    permute_269: "f32[2, 8, 8, 49, 64]" = torch.ops.aten.permute.default(view_396, [2, 0, 3, 1, 4]);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_25 = torch.ops.aten.unbind.int(permute_269);  permute_269 = None
    getitem_310: "f32[8, 8, 49, 64]" = unbind_25[0]
    getitem_311: "f32[8, 8, 49, 64]" = unbind_25[1];  unbind_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_25 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_267, getitem_310, getitem_311, None, True)
    getitem_312: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_25[0]
    getitem_313: "f32[8, 8, 64]" = _scaled_dot_product_efficient_attention_25[1]
    getitem_314: "i64[]" = _scaled_dot_product_efficient_attention_25[2]
    getitem_315: "i64[]" = _scaled_dot_product_efficient_attention_25[3];  _scaled_dot_product_efficient_attention_25 = None
    alias_25: "f32[8, 8, 49, 64]" = torch.ops.aten.alias.default(getitem_312)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_270: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_312, [0, 2, 1, 3]);  getitem_312 = None
    view_397: "f32[8, 49, 512]" = torch.ops.aten.view.default(permute_270, [8, 49, 512]);  permute_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_398: "f32[392, 512]" = torch.ops.aten.view.default(view_397, [392, 512]);  view_397 = None
    permute_271: "f32[512, 512]" = torch.ops.aten.permute.default(primals_479, [1, 0]);  primals_479 = None
    addmm_127: "f32[392, 512]" = torch.ops.aten.addmm.default(primals_480, view_398, permute_271);  primals_480 = None
    view_399: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_127, [8, 49, 512]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_152: "f32[8, 49, 512]" = torch.ops.aten.clone.default(view_399);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_238: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(clone_151, clone_152);  clone_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    var_mean_80 = torch.ops.aten.var_mean.correction(add_238, [2], correction = 0, keepdim = True)
    getitem_316: "f32[8, 49, 1]" = var_mean_80[0]
    getitem_317: "f32[8, 49, 1]" = var_mean_80[1];  var_mean_80 = None
    add_239: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_316, 1e-06);  getitem_316 = None
    rsqrt_80: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
    sub_80: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(add_238, getitem_317)
    mul_235: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = None
    mul_236: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_235, primals_481);  mul_235 = None
    add_240: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_236, primals_482);  mul_236 = primals_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_400: "f32[392, 512]" = torch.ops.aten.view.default(add_240, [392, 512]);  add_240 = None
    permute_272: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_483, [1, 0]);  primals_483 = None
    addmm_128: "f32[392, 2048]" = torch.ops.aten.addmm.default(primals_484, view_400, permute_272);  primals_484 = None
    view_401: "f32[8, 49, 2048]" = torch.ops.aten.view.default(addmm_128, [8, 49, 2048]);  addmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_237: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_401, 0.5)
    mul_238: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_401, 0.7071067811865476)
    erf_25: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_238);  mul_238 = None
    add_241: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_239: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_237, add_241);  mul_237 = add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_153: "f32[8, 49, 2048]" = torch.ops.aten.clone.default(mul_239);  mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_402: "f32[392, 2048]" = torch.ops.aten.view.default(clone_153, [392, 2048]);  clone_153 = None
    permute_273: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_485, [1, 0]);  primals_485 = None
    addmm_129: "f32[392, 512]" = torch.ops.aten.addmm.default(primals_486, view_402, permute_273);  primals_486 = None
    view_403: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_129, [8, 49, 512]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_154: "f32[8, 49, 512]" = torch.ops.aten.clone.default(view_403);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_242: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_238, clone_154);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    permute_274: "f32[8, 512, 49]" = torch.ops.aten.permute.default(add_242, [0, 2, 1]);  add_242 = None
    view_404: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_274, [8, 512, 7, 7]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    convolution_32: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(view_404, primals_487, primals_488, [1, 1], [1, 1], [1, 1], False, [0, 0], 512);  primals_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:245, code: x += cnn_feat_token
    add_243: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(convolution_32, view_404);  convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    view_406: "f32[8, 512, 49]" = torch.ops.aten.view.default(add_243, [8, 512, 49]);  add_243 = None
    permute_276: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None
    clone_155: "f32[8, 49, 512]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format)
    var_mean_81 = torch.ops.aten.var_mean.correction(clone_155, [2], correction = 0, keepdim = True)
    getitem_318: "f32[8, 49, 1]" = var_mean_81[0]
    getitem_319: "f32[8, 49, 1]" = var_mean_81[1];  var_mean_81 = None
    add_244: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_318, 1e-06);  getitem_318 = None
    rsqrt_81: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
    sub_81: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_155, getitem_319);  clone_155 = None
    mul_240: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = None
    mul_241: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_240, primals_489);  mul_240 = None
    add_245: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_241, primals_490);  mul_241 = primals_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_407: "f32[392, 512]" = torch.ops.aten.view.default(add_245, [392, 512])
    permute_277: "f32[512, 512]" = torch.ops.aten.permute.default(primals_491, [1, 0]);  primals_491 = None
    addmm_130: "f32[392, 512]" = torch.ops.aten.addmm.default(primals_492, view_407, permute_277);  primals_492 = None
    view_408: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_130, [8, 49, 512]);  addmm_130 = None
    view_409: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_408, [8, 49, 8, 64]);  view_408 = None
    permute_278: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_410: "f32[392, 512]" = torch.ops.aten.view.default(add_245, [392, 512]);  add_245 = None
    permute_279: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_493, [1, 0]);  primals_493 = None
    addmm_131: "f32[392, 1024]" = torch.ops.aten.addmm.default(primals_494, view_410, permute_279);  primals_494 = None
    view_411: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_131, [8, 49, 1024]);  addmm_131 = None
    view_412: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.view.default(view_411, [8, -1, 2, 8, 64]);  view_411 = None
    permute_280: "f32[2, 8, 8, 49, 64]" = torch.ops.aten.permute.default(view_412, [2, 0, 3, 1, 4]);  view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_26 = torch.ops.aten.unbind.int(permute_280);  permute_280 = None
    getitem_320: "f32[8, 8, 49, 64]" = unbind_26[0]
    getitem_321: "f32[8, 8, 49, 64]" = unbind_26[1];  unbind_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_26 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_278, getitem_320, getitem_321, None, True)
    getitem_322: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_26[0]
    getitem_323: "f32[8, 8, 64]" = _scaled_dot_product_efficient_attention_26[1]
    getitem_324: "i64[]" = _scaled_dot_product_efficient_attention_26[2]
    getitem_325: "i64[]" = _scaled_dot_product_efficient_attention_26[3];  _scaled_dot_product_efficient_attention_26 = None
    alias_26: "f32[8, 8, 49, 64]" = torch.ops.aten.alias.default(getitem_322)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_281: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_322, [0, 2, 1, 3]);  getitem_322 = None
    view_413: "f32[8, 49, 512]" = torch.ops.aten.view.default(permute_281, [8, 49, 512]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_414: "f32[392, 512]" = torch.ops.aten.view.default(view_413, [392, 512]);  view_413 = None
    permute_282: "f32[512, 512]" = torch.ops.aten.permute.default(primals_495, [1, 0]);  primals_495 = None
    addmm_132: "f32[392, 512]" = torch.ops.aten.addmm.default(primals_496, view_414, permute_282);  primals_496 = None
    view_415: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_132, [8, 49, 512]);  addmm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_156: "f32[8, 49, 512]" = torch.ops.aten.clone.default(view_415);  view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_246: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(permute_276, clone_156);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_157: "f32[8, 49, 512]" = torch.ops.aten.clone.default(add_246, memory_format = torch.contiguous_format)
    var_mean_82 = torch.ops.aten.var_mean.correction(clone_157, [2], correction = 0, keepdim = True)
    getitem_326: "f32[8, 49, 1]" = var_mean_82[0]
    getitem_327: "f32[8, 49, 1]" = var_mean_82[1];  var_mean_82 = None
    add_247: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_326, 1e-06);  getitem_326 = None
    rsqrt_82: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
    sub_82: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_157, getitem_327);  clone_157 = None
    mul_242: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
    mul_243: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_242, primals_497);  mul_242 = None
    add_248: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_243, primals_498);  mul_243 = primals_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_416: "f32[392, 512]" = torch.ops.aten.view.default(add_248, [392, 512]);  add_248 = None
    permute_283: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_499, [1, 0]);  primals_499 = None
    addmm_133: "f32[392, 2048]" = torch.ops.aten.addmm.default(primals_500, view_416, permute_283);  primals_500 = None
    view_417: "f32[8, 49, 2048]" = torch.ops.aten.view.default(addmm_133, [8, 49, 2048]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_244: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_417, 0.5)
    mul_245: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_417, 0.7071067811865476)
    erf_26: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_245);  mul_245 = None
    add_249: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_246: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_244, add_249);  mul_244 = add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_158: "f32[8, 49, 2048]" = torch.ops.aten.clone.default(mul_246);  mul_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_418: "f32[392, 2048]" = torch.ops.aten.view.default(clone_158, [392, 2048]);  clone_158 = None
    permute_284: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_501, [1, 0]);  primals_501 = None
    addmm_134: "f32[392, 512]" = torch.ops.aten.addmm.default(primals_502, view_418, permute_284);  primals_502 = None
    view_419: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_134, [8, 49, 512]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_159: "f32[8, 49, 512]" = torch.ops.aten.clone.default(view_419);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_250: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_246, clone_159);  clone_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_160: "f32[8, 49, 512]" = torch.ops.aten.clone.default(add_250, memory_format = torch.contiguous_format)
    var_mean_83 = torch.ops.aten.var_mean.correction(clone_160, [2], correction = 0, keepdim = True)
    getitem_328: "f32[8, 49, 1]" = var_mean_83[0]
    getitem_329: "f32[8, 49, 1]" = var_mean_83[1];  var_mean_83 = None
    add_251: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_328, 1e-06);  getitem_328 = None
    rsqrt_83: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
    sub_83: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_160, getitem_329);  clone_160 = None
    mul_247: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
    mul_248: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_247, primals_503);  mul_247 = None
    add_252: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_248, primals_504);  mul_248 = primals_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_420: "f32[392, 512]" = torch.ops.aten.view.default(add_252, [392, 512])
    permute_285: "f32[512, 512]" = torch.ops.aten.permute.default(primals_505, [1, 0]);  primals_505 = None
    addmm_135: "f32[392, 512]" = torch.ops.aten.addmm.default(primals_506, view_420, permute_285);  primals_506 = None
    view_421: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_135, [8, 49, 512]);  addmm_135 = None
    view_422: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_421, [8, 49, 8, 64]);  view_421 = None
    permute_286: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_423: "f32[392, 512]" = torch.ops.aten.view.default(add_252, [392, 512]);  add_252 = None
    permute_287: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_507, [1, 0]);  primals_507 = None
    addmm_136: "f32[392, 1024]" = torch.ops.aten.addmm.default(primals_508, view_423, permute_287);  primals_508 = None
    view_424: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_136, [8, 49, 1024]);  addmm_136 = None
    view_425: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.view.default(view_424, [8, -1, 2, 8, 64]);  view_424 = None
    permute_288: "f32[2, 8, 8, 49, 64]" = torch.ops.aten.permute.default(view_425, [2, 0, 3, 1, 4]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_27 = torch.ops.aten.unbind.int(permute_288);  permute_288 = None
    getitem_330: "f32[8, 8, 49, 64]" = unbind_27[0]
    getitem_331: "f32[8, 8, 49, 64]" = unbind_27[1];  unbind_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_27 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_286, getitem_330, getitem_331, None, True)
    getitem_332: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_27[0]
    getitem_333: "f32[8, 8, 64]" = _scaled_dot_product_efficient_attention_27[1]
    getitem_334: "i64[]" = _scaled_dot_product_efficient_attention_27[2]
    getitem_335: "i64[]" = _scaled_dot_product_efficient_attention_27[3];  _scaled_dot_product_efficient_attention_27 = None
    alias_27: "f32[8, 8, 49, 64]" = torch.ops.aten.alias.default(getitem_332)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_289: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_332, [0, 2, 1, 3]);  getitem_332 = None
    view_426: "f32[8, 49, 512]" = torch.ops.aten.view.default(permute_289, [8, 49, 512]);  permute_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_427: "f32[392, 512]" = torch.ops.aten.view.default(view_426, [392, 512]);  view_426 = None
    permute_290: "f32[512, 512]" = torch.ops.aten.permute.default(primals_509, [1, 0]);  primals_509 = None
    addmm_137: "f32[392, 512]" = torch.ops.aten.addmm.default(primals_510, view_427, permute_290);  primals_510 = None
    view_428: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_137, [8, 49, 512]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    clone_161: "f32[8, 49, 512]" = torch.ops.aten.clone.default(view_428);  view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_253: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_250, clone_161);  clone_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_162: "f32[8, 49, 512]" = torch.ops.aten.clone.default(add_253, memory_format = torch.contiguous_format)
    var_mean_84 = torch.ops.aten.var_mean.correction(clone_162, [2], correction = 0, keepdim = True)
    getitem_336: "f32[8, 49, 1]" = var_mean_84[0]
    getitem_337: "f32[8, 49, 1]" = var_mean_84[1];  var_mean_84 = None
    add_254: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_336, 1e-06);  getitem_336 = None
    rsqrt_84: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
    sub_84: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_162, getitem_337);  clone_162 = None
    mul_249: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
    mul_250: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_249, primals_511);  mul_249 = None
    add_255: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_250, primals_512);  mul_250 = primals_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_429: "f32[392, 512]" = torch.ops.aten.view.default(add_255, [392, 512]);  add_255 = None
    permute_291: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_513, [1, 0]);  primals_513 = None
    addmm_138: "f32[392, 2048]" = torch.ops.aten.addmm.default(primals_514, view_429, permute_291);  primals_514 = None
    view_430: "f32[8, 49, 2048]" = torch.ops.aten.view.default(addmm_138, [8, 49, 2048]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_251: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_430, 0.5)
    mul_252: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_430, 0.7071067811865476)
    erf_27: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_252);  mul_252 = None
    add_256: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_253: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_251, add_256);  mul_251 = add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_163: "f32[8, 49, 2048]" = torch.ops.aten.clone.default(mul_253);  mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_431: "f32[392, 2048]" = torch.ops.aten.view.default(clone_163, [392, 2048]);  clone_163 = None
    permute_292: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_515, [1, 0]);  primals_515 = None
    addmm_139: "f32[392, 512]" = torch.ops.aten.addmm.default(primals_516, view_431, permute_292);  primals_516 = None
    view_432: "f32[8, 49, 512]" = torch.ops.aten.view.default(addmm_139, [8, 49, 512]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_164: "f32[8, 49, 512]" = torch.ops.aten.clone.default(view_432);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_257: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_253, clone_164);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:416, code: x = self.norm(x)
    clone_165: "f32[8, 49, 512]" = torch.ops.aten.clone.default(add_257, memory_format = torch.contiguous_format)
    var_mean_85 = torch.ops.aten.var_mean.correction(clone_165, [2], correction = 0, keepdim = True)
    getitem_338: "f32[8, 49, 1]" = var_mean_85[0]
    getitem_339: "f32[8, 49, 1]" = var_mean_85[1];  var_mean_85 = None
    add_258: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_338, 1e-06);  getitem_338 = None
    rsqrt_85: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
    sub_85: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_165, getitem_339);  clone_165 = None
    mul_254: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = None
    mul_255: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_254, primals_517);  mul_254 = None
    add_259: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_255, primals_518);  mul_255 = primals_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:421, code: x = x.mean(dim=1)
    mean: "f32[8, 512]" = torch.ops.aten.mean.dim(add_259, [1]);  add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:422, code: x = self.head_drop(x)
    clone_166: "f32[8, 512]" = torch.ops.aten.clone.default(mean);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:423, code: return x if pre_logits else self.head(x)
    permute_293: "f32[512, 1000]" = torch.ops.aten.permute.default(primals_519, [1, 0]);  primals_519 = None
    addmm_140: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_520, clone_166, permute_293);  primals_520 = None
    permute_294: "f32[1000, 512]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
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
    clone_167: "f32[8, 49, 512]" = torch.ops.aten.clone.default(add_257, memory_format = torch.contiguous_format);  add_257 = None
    sub_86: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_167, getitem_339);  clone_167 = getitem_339 = None
    mul_256: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_85);  sub_86 = None
    mul_257: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div, primals_517);  primals_517 = None
    mul_258: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_257, 512)
    sum_2: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True)
    mul_259: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_257, mul_256);  mul_257 = None
    sum_3: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [2], True);  mul_259 = None
    mul_260: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_256, sum_3);  sum_3 = None
    sub_87: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_258, sum_2);  mul_258 = sum_2 = None
    sub_88: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_87, mul_260);  sub_87 = mul_260 = None
    div_1: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_85, 512);  rsqrt_85 = None
    mul_261: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_1, sub_88);  div_1 = sub_88 = None
    mul_262: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div, mul_256);  mul_256 = None
    sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_262, [0, 1]);  mul_262 = None
    sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(div, [0, 1]);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_434: "f32[392, 512]" = torch.ops.aten.view.default(mul_261, [392, 512])
    permute_298: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
    mm_2: "f32[392, 2048]" = torch.ops.aten.mm.default(view_434, permute_298);  permute_298 = None
    permute_299: "f32[512, 392]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_3: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_299, view_431);  permute_299 = view_431 = None
    permute_300: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_6: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[512]" = torch.ops.aten.view.default(sum_6, [512]);  sum_6 = None
    permute_301: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
    view_436: "f32[8, 49, 2048]" = torch.ops.aten.view.default(mm_2, [8, 49, 2048]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_263: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_430, 0.7071067811865476)
    erf_28: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_263);  mul_263 = None
    add_260: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_264: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(add_260, 0.5);  add_260 = None
    mul_265: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_430, view_430)
    mul_266: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_265, -0.5);  mul_265 = None
    exp: "f32[8, 49, 2048]" = torch.ops.aten.exp.default(mul_266);  mul_266 = None
    mul_267: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_268: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_430, mul_267);  view_430 = mul_267 = None
    add_261: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(mul_264, mul_268);  mul_264 = mul_268 = None
    mul_269: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_436, add_261);  view_436 = add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_437: "f32[392, 2048]" = torch.ops.aten.view.default(mul_269, [392, 2048]);  mul_269 = None
    permute_302: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
    mm_4: "f32[392, 512]" = torch.ops.aten.mm.default(view_437, permute_302);  permute_302 = None
    permute_303: "f32[2048, 392]" = torch.ops.aten.permute.default(view_437, [1, 0])
    mm_5: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_303, view_429);  permute_303 = view_429 = None
    permute_304: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_7: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_437, [0], True);  view_437 = None
    view_438: "f32[2048]" = torch.ops.aten.view.default(sum_7, [2048]);  sum_7 = None
    permute_305: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
    view_439: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_4, [8, 49, 512]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_168: "f32[8, 49, 512]" = torch.ops.aten.clone.default(add_253, memory_format = torch.contiguous_format);  add_253 = None
    sub_89: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_168, getitem_337);  clone_168 = getitem_337 = None
    mul_270: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_84);  sub_89 = None
    mul_271: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_439, primals_511);  primals_511 = None
    mul_272: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_271, 512)
    sum_8: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True)
    mul_273: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_271, mul_270);  mul_271 = None
    sum_9: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2], True);  mul_273 = None
    mul_274: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_270, sum_9);  sum_9 = None
    sub_90: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_272, sum_8);  mul_272 = sum_8 = None
    sub_91: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_90, mul_274);  sub_90 = mul_274 = None
    div_2: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_84, 512);  rsqrt_84 = None
    mul_275: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_2, sub_91);  div_2 = sub_91 = None
    mul_276: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_439, mul_270);  mul_270 = None
    sum_10: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1]);  mul_276 = None
    sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_439, [0, 1]);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_262: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_261, mul_275);  mul_261 = mul_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_440: "f32[392, 512]" = torch.ops.aten.view.default(add_262, [392, 512])
    permute_306: "f32[512, 512]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
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
    alias_28: "f32[8, 8, 49, 64]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
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
    permute_312: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
    mm_8: "f32[392, 512]" = torch.ops.aten.mm.default(view_446, permute_312);  permute_312 = None
    permute_313: "f32[1024, 392]" = torch.ops.aten.permute.default(view_446, [1, 0])
    mm_9: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_313, view_423);  permute_313 = view_423 = None
    permute_314: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_13: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_446, [0], True);  view_446 = None
    view_447: "f32[1024]" = torch.ops.aten.view.default(sum_13, [1024]);  sum_13 = None
    permute_315: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    view_448: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_8, [8, 49, 512]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_316: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_340, [0, 2, 1, 3]);  getitem_340 = None
    view_449: "f32[8, 49, 512]" = torch.ops.aten.view.default(permute_316, [8, 49, 512]);  permute_316 = None
    view_450: "f32[392, 512]" = torch.ops.aten.view.default(view_449, [392, 512]);  view_449 = None
    permute_317: "f32[512, 512]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
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
    clone_170: "f32[8, 49, 512]" = torch.ops.aten.clone.default(add_250, memory_format = torch.contiguous_format);  add_250 = None
    sub_92: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_170, getitem_329);  clone_170 = getitem_329 = None
    mul_277: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_83);  sub_92 = None
    mul_278: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_263, primals_503);  primals_503 = None
    mul_279: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_278, 512)
    sum_15: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True)
    mul_280: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_278, mul_277);  mul_278 = None
    sum_16: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True);  mul_280 = None
    mul_281: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_277, sum_16);  sum_16 = None
    sub_93: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_279, sum_15);  mul_279 = sum_15 = None
    sub_94: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_93, mul_281);  sub_93 = mul_281 = None
    div_3: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_83, 512);  rsqrt_83 = None
    mul_282: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_3, sub_94);  div_3 = sub_94 = None
    mul_283: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_263, mul_277);  mul_277 = None
    sum_17: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1]);  mul_283 = None
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_264: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_262, mul_282);  add_262 = mul_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_453: "f32[392, 512]" = torch.ops.aten.view.default(add_264, [392, 512])
    permute_321: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    mm_12: "f32[392, 2048]" = torch.ops.aten.mm.default(view_453, permute_321);  permute_321 = None
    permute_322: "f32[512, 392]" = torch.ops.aten.permute.default(view_453, [1, 0])
    mm_13: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_322, view_418);  permute_322 = view_418 = None
    permute_323: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_19: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_453, [0], True);  view_453 = None
    view_454: "f32[512]" = torch.ops.aten.view.default(sum_19, [512]);  sum_19 = None
    permute_324: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_323, [1, 0]);  permute_323 = None
    view_455: "f32[8, 49, 2048]" = torch.ops.aten.view.default(mm_12, [8, 49, 2048]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_284: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_417, 0.7071067811865476)
    erf_29: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_284);  mul_284 = None
    add_265: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_285: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(add_265, 0.5);  add_265 = None
    mul_286: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_417, view_417)
    mul_287: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_286, -0.5);  mul_286 = None
    exp_1: "f32[8, 49, 2048]" = torch.ops.aten.exp.default(mul_287);  mul_287 = None
    mul_288: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_289: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_417, mul_288);  view_417 = mul_288 = None
    add_266: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(mul_285, mul_289);  mul_285 = mul_289 = None
    mul_290: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_455, add_266);  view_455 = add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_456: "f32[392, 2048]" = torch.ops.aten.view.default(mul_290, [392, 2048]);  mul_290 = None
    permute_325: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_283, [1, 0]);  permute_283 = None
    mm_14: "f32[392, 512]" = torch.ops.aten.mm.default(view_456, permute_325);  permute_325 = None
    permute_326: "f32[2048, 392]" = torch.ops.aten.permute.default(view_456, [1, 0])
    mm_15: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_326, view_416);  permute_326 = view_416 = None
    permute_327: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_20: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_456, [0], True);  view_456 = None
    view_457: "f32[2048]" = torch.ops.aten.view.default(sum_20, [2048]);  sum_20 = None
    permute_328: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_327, [1, 0]);  permute_327 = None
    view_458: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_14, [8, 49, 512]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_171: "f32[8, 49, 512]" = torch.ops.aten.clone.default(add_246, memory_format = torch.contiguous_format);  add_246 = None
    sub_95: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_171, getitem_327);  clone_171 = getitem_327 = None
    mul_291: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_82);  sub_95 = None
    mul_292: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_458, primals_497);  primals_497 = None
    mul_293: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_292, 512)
    sum_21: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True)
    mul_294: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_292, mul_291);  mul_292 = None
    sum_22: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [2], True);  mul_294 = None
    mul_295: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_291, sum_22);  sum_22 = None
    sub_96: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_293, sum_21);  mul_293 = sum_21 = None
    sub_97: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_96, mul_295);  sub_96 = mul_295 = None
    div_4: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_82, 512);  rsqrt_82 = None
    mul_296: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_4, sub_97);  div_4 = sub_97 = None
    mul_297: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_458, mul_291);  mul_291 = None
    sum_23: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 1]);  mul_297 = None
    sum_24: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_458, [0, 1]);  view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_267: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_264, mul_296);  add_264 = mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_459: "f32[392, 512]" = torch.ops.aten.view.default(add_267, [392, 512])
    permute_329: "f32[512, 512]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
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
    alias_29: "f32[8, 8, 49, 64]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
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
    permute_335: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    mm_18: "f32[392, 512]" = torch.ops.aten.mm.default(view_465, permute_335);  permute_335 = None
    permute_336: "f32[1024, 392]" = torch.ops.aten.permute.default(view_465, [1, 0])
    mm_19: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_336, view_410);  permute_336 = view_410 = None
    permute_337: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_26: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_465, [0], True);  view_465 = None
    view_466: "f32[1024]" = torch.ops.aten.view.default(sum_26, [1024]);  sum_26 = None
    permute_338: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
    view_467: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_18, [8, 49, 512]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_339: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_344, [0, 2, 1, 3]);  getitem_344 = None
    view_468: "f32[8, 49, 512]" = torch.ops.aten.view.default(permute_339, [8, 49, 512]);  permute_339 = None
    view_469: "f32[392, 512]" = torch.ops.aten.view.default(view_468, [392, 512]);  view_468 = None
    permute_340: "f32[512, 512]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
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
    clone_173: "f32[8, 49, 512]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
    sub_98: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_173, getitem_319);  clone_173 = getitem_319 = None
    mul_298: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_81);  sub_98 = None
    mul_299: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_268, primals_489);  primals_489 = None
    mul_300: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_299, 512)
    sum_28: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True)
    mul_301: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_299, mul_298);  mul_299 = None
    sum_29: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [2], True);  mul_301 = None
    mul_302: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_298, sum_29);  sum_29 = None
    sub_99: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_300, sum_28);  mul_300 = sum_28 = None
    sub_100: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_99, mul_302);  sub_99 = mul_302 = None
    div_5: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_81, 512);  rsqrt_81 = None
    mul_303: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_5, sub_100);  div_5 = sub_100 = None
    mul_304: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_268, mul_298);  mul_298 = None
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
    permute_346: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
    mm_22: "f32[392, 2048]" = torch.ops.aten.mm.default(view_474, permute_346);  permute_346 = None
    permute_347: "f32[512, 392]" = torch.ops.aten.permute.default(view_474, [1, 0])
    mm_23: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_347, view_402);  permute_347 = view_402 = None
    permute_348: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_33: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_474, [0], True);  view_474 = None
    view_475: "f32[512]" = torch.ops.aten.view.default(sum_33, [512]);  sum_33 = None
    permute_349: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_348, [1, 0]);  permute_348 = None
    view_476: "f32[8, 49, 2048]" = torch.ops.aten.view.default(mm_22, [8, 49, 2048]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_305: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_401, 0.7071067811865476)
    erf_30: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_305);  mul_305 = None
    add_271: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_306: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(add_271, 0.5);  add_271 = None
    mul_307: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_401, view_401)
    mul_308: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_307, -0.5);  mul_307 = None
    exp_2: "f32[8, 49, 2048]" = torch.ops.aten.exp.default(mul_308);  mul_308 = None
    mul_309: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_310: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_401, mul_309);  view_401 = mul_309 = None
    add_272: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(mul_306, mul_310);  mul_306 = mul_310 = None
    mul_311: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_476, add_272);  view_476 = add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_477: "f32[392, 2048]" = torch.ops.aten.view.default(mul_311, [392, 2048]);  mul_311 = None
    permute_350: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    mm_24: "f32[392, 512]" = torch.ops.aten.mm.default(view_477, permute_350);  permute_350 = None
    permute_351: "f32[2048, 392]" = torch.ops.aten.permute.default(view_477, [1, 0])
    mm_25: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_351, view_400);  permute_351 = view_400 = None
    permute_352: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_34: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_477, [0], True);  view_477 = None
    view_478: "f32[2048]" = torch.ops.aten.view.default(sum_34, [2048]);  sum_34 = None
    permute_353: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    view_479: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_24, [8, 49, 512]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    sub_101: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(add_238, getitem_317);  add_238 = getitem_317 = None
    mul_312: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_80);  sub_101 = None
    mul_313: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_479, primals_481);  primals_481 = None
    mul_314: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_313, 512)
    sum_35: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True)
    mul_315: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_313, mul_312);  mul_313 = None
    sum_36: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True);  mul_315 = None
    mul_316: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_312, sum_36);  sum_36 = None
    sub_102: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_314, sum_35);  mul_314 = sum_35 = None
    sub_103: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_102, mul_316);  sub_102 = mul_316 = None
    div_6: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_80, 512);  rsqrt_80 = None
    mul_317: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_6, sub_103);  div_6 = sub_103 = None
    mul_318: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_479, mul_312);  mul_312 = None
    sum_37: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1]);  mul_318 = None
    sum_38: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_479, [0, 1]);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_273: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(permute_345, mul_317);  permute_345 = mul_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_480: "f32[392, 512]" = torch.ops.aten.view.default(add_273, [392, 512])
    permute_354: "f32[512, 512]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
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
    alias_30: "f32[8, 8, 49, 64]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
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
    permute_360: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    mm_28: "f32[392, 512]" = torch.ops.aten.mm.default(view_486, permute_360);  permute_360 = None
    permute_361: "f32[1024, 392]" = torch.ops.aten.permute.default(view_486, [1, 0])
    mm_29: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_361, view_394);  permute_361 = view_394 = None
    permute_362: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_40: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_486, [0], True);  view_486 = None
    view_487: "f32[1024]" = torch.ops.aten.view.default(sum_40, [1024]);  sum_40 = None
    permute_363: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_488: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_28, [8, 49, 512]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_364: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_351, [0, 2, 1, 3]);  getitem_351 = None
    view_489: "f32[8, 49, 512]" = torch.ops.aten.view.default(permute_364, [8, 49, 512]);  permute_364 = None
    view_490: "f32[392, 512]" = torch.ops.aten.view.default(view_489, [392, 512]);  view_489 = None
    permute_365: "f32[512, 512]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
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
    sub_104: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_151, getitem_309);  clone_151 = getitem_309 = None
    mul_319: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_79);  sub_104 = None
    mul_320: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_274, primals_473);  primals_473 = None
    mul_321: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_320, 512)
    sum_42: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True)
    mul_322: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_320, mul_319);  mul_320 = None
    sum_43: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [2], True);  mul_322 = None
    mul_323: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_319, sum_43);  sum_43 = None
    sub_105: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_321, sum_42);  mul_321 = sum_42 = None
    sub_106: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_105, mul_323);  sub_105 = mul_323 = None
    div_7: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_79, 512);  rsqrt_79 = None
    mul_324: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_7, sub_106);  div_7 = sub_106 = None
    mul_325: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_274, mul_319);  mul_319 = None
    sum_44: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 1]);  mul_325 = None
    sum_45: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 1]);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_275: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_273, mul_324);  add_273 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone_175: "f32[8, 49, 512]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    sub_107: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_175, getitem_307);  clone_175 = getitem_307 = None
    mul_326: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_78);  sub_107 = None
    mul_327: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_275, primals_471);  primals_471 = None
    mul_328: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_327, 512)
    sum_46: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True)
    mul_329: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_327, mul_326);  mul_327 = None
    sum_47: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True);  mul_329 = None
    mul_330: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_326, sum_47);  sum_47 = None
    sub_108: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_328, sum_46);  mul_328 = sum_46 = None
    sub_109: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_108, mul_330);  sub_108 = mul_330 = None
    div_8: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_78, 512);  rsqrt_78 = None
    mul_331: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_8, sub_109);  div_8 = sub_109 = None
    mul_332: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_275, mul_326);  mul_326 = None
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
    permute_371: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    mm_32: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_495, permute_371);  permute_371 = None
    permute_372: "f32[320, 1568]" = torch.ops.aten.permute.default(view_495, [1, 0])
    mm_33: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_372, view_387);  permute_372 = view_387 = None
    permute_373: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_51: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_495, [0], True);  view_495 = None
    view_496: "f32[320]" = torch.ops.aten.view.default(sum_51, [320]);  sum_51 = None
    permute_374: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_373, [1, 0]);  permute_373 = None
    view_497: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_32, [8, 196, 1280]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_333: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_386, 0.7071067811865476)
    erf_31: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_333);  mul_333 = None
    add_276: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_334: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_276, 0.5);  add_276 = None
    mul_335: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_386, view_386)
    mul_336: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_335, -0.5);  mul_335 = None
    exp_3: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_336);  mul_336 = None
    mul_337: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_338: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_386, mul_337);  view_386 = mul_337 = None
    add_277: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_334, mul_338);  mul_334 = mul_338 = None
    mul_339: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_497, add_277);  view_497 = add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_498: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_339, [1568, 1280]);  mul_339 = None
    permute_375: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    mm_34: "f32[1568, 320]" = torch.ops.aten.mm.default(view_498, permute_375);  permute_375 = None
    permute_376: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_498, [1, 0])
    mm_35: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_376, view_385);  permute_376 = view_385 = None
    permute_377: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_52: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True);  view_498 = None
    view_499: "f32[1280]" = torch.ops.aten.view.default(sum_52, [1280]);  sum_52 = None
    permute_378: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    view_500: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_34, [8, 196, 320]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_177: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_229, memory_format = torch.contiguous_format);  add_229 = None
    sub_110: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_177, getitem_305);  clone_177 = getitem_305 = None
    mul_340: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_77);  sub_110 = None
    mul_341: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_500, primals_463);  primals_463 = None
    mul_342: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_341, 320)
    sum_53: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
    mul_343: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_341, mul_340);  mul_341 = None
    sum_54: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
    mul_344: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_340, sum_54);  sum_54 = None
    sub_111: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_342, sum_53);  mul_342 = sum_53 = None
    sub_112: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_111, mul_344);  sub_111 = mul_344 = None
    div_9: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_77, 320);  rsqrt_77 = None
    mul_345: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_9, sub_112);  div_9 = sub_112 = None
    mul_346: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_500, mul_340);  mul_340 = None
    sum_55: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
    sum_56: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_500, [0, 1]);  view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_278: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(view_494, mul_345);  view_494 = mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_178: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_278, memory_format = torch.contiguous_format)
    view_501: "f32[1568, 320]" = torch.ops.aten.view.default(clone_178, [1568, 320]);  clone_178 = None
    permute_379: "f32[320, 320]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
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
    alias_31: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
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
    permute_385: "f32[640, 320]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
    mm_38: "f32[392, 320]" = torch.ops.aten.mm.default(view_507, permute_385);  permute_385 = None
    permute_386: "f32[640, 392]" = torch.ops.aten.permute.default(view_507, [1, 0])
    mm_39: "f32[640, 320]" = torch.ops.aten.mm.default(permute_386, view_379);  permute_386 = view_379 = None
    permute_387: "f32[320, 640]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_58: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_507, [0], True);  view_507 = None
    view_508: "f32[640]" = torch.ops.aten.view.default(sum_58, [640]);  sum_58 = None
    permute_388: "f32[640, 320]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
    view_509: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_38, [8, 49, 320]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_180: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
    sub_113: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_180, getitem_297);  clone_180 = getitem_297 = None
    mul_347: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_76);  sub_113 = None
    mul_348: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_509, primals_457);  primals_457 = None
    mul_349: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_348, 320)
    sum_59: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True)
    mul_350: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_348, mul_347);  mul_348 = None
    sum_60: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [2], True);  mul_350 = None
    mul_351: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_347, sum_60);  sum_60 = None
    sub_114: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_349, sum_59);  mul_349 = sum_59 = None
    sub_115: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_114, mul_351);  sub_114 = mul_351 = None
    div_10: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_76, 320);  rsqrt_76 = None
    mul_352: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_10, sub_115);  div_10 = sub_115 = None
    mul_353: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_509, mul_347);  mul_347 = None
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
    permute_392: "f32[320, 320]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
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
    clone_182: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_224, memory_format = torch.contiguous_format);  add_224 = None
    sub_116: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_182, getitem_295);  clone_182 = getitem_295 = None
    mul_354: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_75);  sub_116 = None
    mul_355: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_181, primals_451);  primals_451 = None
    mul_356: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_355, 320)
    sum_65: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True)
    mul_357: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_355, mul_354);  mul_355 = None
    sum_66: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True);  mul_357 = None
    mul_358: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_354, sum_66);  sum_66 = None
    sub_117: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_356, sum_65);  mul_356 = sum_65 = None
    sub_118: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_117, mul_358);  sub_117 = mul_358 = None
    div_11: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_75, 320);  rsqrt_75 = None
    mul_359: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_11, sub_118);  div_11 = sub_118 = None
    mul_360: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_181, mul_354);  mul_354 = None
    sum_67: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1]);  mul_360 = None
    sum_68: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_181, [0, 1]);  clone_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_280: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_278, mul_359);  add_278 = mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_183: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_280, memory_format = torch.contiguous_format)
    view_516: "f32[1568, 320]" = torch.ops.aten.view.default(clone_183, [1568, 320]);  clone_183 = None
    permute_396: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    mm_42: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_516, permute_396);  permute_396 = None
    permute_397: "f32[320, 1568]" = torch.ops.aten.permute.default(view_516, [1, 0])
    mm_43: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_397, view_372);  permute_397 = view_372 = None
    permute_398: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_69: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_516, [0], True);  view_516 = None
    view_517: "f32[320]" = torch.ops.aten.view.default(sum_69, [320]);  sum_69 = None
    permute_399: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
    view_518: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_42, [8, 196, 1280]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_361: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476)
    erf_32: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_361);  mul_361 = None
    add_281: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_362: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_281, 0.5);  add_281 = None
    mul_363: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_371, view_371)
    mul_364: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_363, -0.5);  mul_363 = None
    exp_4: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_364);  mul_364 = None
    mul_365: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_366: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_371, mul_365);  view_371 = mul_365 = None
    add_282: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_362, mul_366);  mul_362 = mul_366 = None
    mul_367: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_518, add_282);  view_518 = add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_519: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_367, [1568, 1280]);  mul_367 = None
    permute_400: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    mm_44: "f32[1568, 320]" = torch.ops.aten.mm.default(view_519, permute_400);  permute_400 = None
    permute_401: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_519, [1, 0])
    mm_45: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_401, view_370);  permute_401 = view_370 = None
    permute_402: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_70: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_519, [0], True);  view_519 = None
    view_520: "f32[1280]" = torch.ops.aten.view.default(sum_70, [1280]);  sum_70 = None
    permute_403: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    view_521: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_44, [8, 196, 320]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_184: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_220, memory_format = torch.contiguous_format);  add_220 = None
    sub_119: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_184, getitem_293);  clone_184 = getitem_293 = None
    mul_368: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_74);  sub_119 = None
    mul_369: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_521, primals_445);  primals_445 = None
    mul_370: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_369, 320)
    sum_71: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True)
    mul_371: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_369, mul_368);  mul_369 = None
    sum_72: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True);  mul_371 = None
    mul_372: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_368, sum_72);  sum_72 = None
    sub_120: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_370, sum_71);  mul_370 = sum_71 = None
    sub_121: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_120, mul_372);  sub_120 = mul_372 = None
    div_12: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_74, 320);  rsqrt_74 = None
    mul_373: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_12, sub_121);  div_12 = sub_121 = None
    mul_374: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_521, mul_368);  mul_368 = None
    sum_73: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1]);  mul_374 = None
    sum_74: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_521, [0, 1]);  view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_283: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_280, mul_373);  add_280 = mul_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_185: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_283, memory_format = torch.contiguous_format)
    view_522: "f32[1568, 320]" = torch.ops.aten.view.default(clone_185, [1568, 320]);  clone_185 = None
    permute_404: "f32[320, 320]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
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
    alias_32: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
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
    permute_410: "f32[640, 320]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    mm_48: "f32[392, 320]" = torch.ops.aten.mm.default(view_528, permute_410);  permute_410 = None
    permute_411: "f32[640, 392]" = torch.ops.aten.permute.default(view_528, [1, 0])
    mm_49: "f32[640, 320]" = torch.ops.aten.mm.default(permute_411, view_364);  permute_411 = view_364 = None
    permute_412: "f32[320, 640]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_76: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
    view_529: "f32[640]" = torch.ops.aten.view.default(sum_76, [640]);  sum_76 = None
    permute_413: "f32[640, 320]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_530: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_48, [8, 49, 320]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_187: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    sub_122: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_187, getitem_285);  clone_187 = getitem_285 = None
    mul_375: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_73);  sub_122 = None
    mul_376: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_530, primals_439);  primals_439 = None
    mul_377: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_376, 320)
    sum_77: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [2], True)
    mul_378: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_376, mul_375);  mul_376 = None
    sum_78: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [2], True);  mul_378 = None
    mul_379: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_375, sum_78);  sum_78 = None
    sub_123: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_377, sum_77);  mul_377 = sum_77 = None
    sub_124: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_123, mul_379);  sub_123 = mul_379 = None
    div_13: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_73, 320);  rsqrt_73 = None
    mul_380: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_13, sub_124);  div_13 = sub_124 = None
    mul_381: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_530, mul_375);  mul_375 = None
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
    permute_417: "f32[320, 320]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
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
    clone_189: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_215, memory_format = torch.contiguous_format);  add_215 = None
    sub_125: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_189, getitem_283);  clone_189 = getitem_283 = None
    mul_382: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_72);  sub_125 = None
    mul_383: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_188, primals_433);  primals_433 = None
    mul_384: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_383, 320)
    sum_83: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
    mul_385: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_383, mul_382);  mul_383 = None
    sum_84: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
    mul_386: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_382, sum_84);  sum_84 = None
    sub_126: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_384, sum_83);  mul_384 = sum_83 = None
    sub_127: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_126, mul_386);  sub_126 = mul_386 = None
    div_14: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_72, 320);  rsqrt_72 = None
    mul_387: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_14, sub_127);  div_14 = sub_127 = None
    mul_388: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_188, mul_382);  mul_382 = None
    sum_85: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
    sum_86: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_188, [0, 1]);  clone_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_285: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_283, mul_387);  add_283 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_190: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_285, memory_format = torch.contiguous_format)
    view_537: "f32[1568, 320]" = torch.ops.aten.view.default(clone_190, [1568, 320]);  clone_190 = None
    permute_421: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    mm_52: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_537, permute_421);  permute_421 = None
    permute_422: "f32[320, 1568]" = torch.ops.aten.permute.default(view_537, [1, 0])
    mm_53: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_422, view_357);  permute_422 = view_357 = None
    permute_423: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_87: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
    view_538: "f32[320]" = torch.ops.aten.view.default(sum_87, [320]);  sum_87 = None
    permute_424: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    view_539: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_52, [8, 196, 1280]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_389: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_356, 0.7071067811865476)
    erf_33: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_389);  mul_389 = None
    add_286: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_390: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_286, 0.5);  add_286 = None
    mul_391: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_356, view_356)
    mul_392: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_391, -0.5);  mul_391 = None
    exp_5: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_392);  mul_392 = None
    mul_393: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_394: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_356, mul_393);  view_356 = mul_393 = None
    add_287: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_390, mul_394);  mul_390 = mul_394 = None
    mul_395: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_539, add_287);  view_539 = add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_540: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_395, [1568, 1280]);  mul_395 = None
    permute_425: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    mm_54: "f32[1568, 320]" = torch.ops.aten.mm.default(view_540, permute_425);  permute_425 = None
    permute_426: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_540, [1, 0])
    mm_55: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_426, view_355);  permute_426 = view_355 = None
    permute_427: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_88: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_540, [0], True);  view_540 = None
    view_541: "f32[1280]" = torch.ops.aten.view.default(sum_88, [1280]);  sum_88 = None
    permute_428: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    view_542: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_54, [8, 196, 320]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_191: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_211, memory_format = torch.contiguous_format);  add_211 = None
    sub_128: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_191, getitem_281);  clone_191 = getitem_281 = None
    mul_396: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_71);  sub_128 = None
    mul_397: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_542, primals_427);  primals_427 = None
    mul_398: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_397, 320)
    sum_89: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True)
    mul_399: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_397, mul_396);  mul_397 = None
    sum_90: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True);  mul_399 = None
    mul_400: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_396, sum_90);  sum_90 = None
    sub_129: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_398, sum_89);  mul_398 = sum_89 = None
    sub_130: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_129, mul_400);  sub_129 = mul_400 = None
    div_15: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_71, 320);  rsqrt_71 = None
    mul_401: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_15, sub_130);  div_15 = sub_130 = None
    mul_402: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_542, mul_396);  mul_396 = None
    sum_91: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_402, [0, 1]);  mul_402 = None
    sum_92: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_542, [0, 1]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_288: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_285, mul_401);  add_285 = mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_192: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_288, memory_format = torch.contiguous_format)
    view_543: "f32[1568, 320]" = torch.ops.aten.view.default(clone_192, [1568, 320]);  clone_192 = None
    permute_429: "f32[320, 320]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
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
    alias_33: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
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
    permute_435: "f32[640, 320]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    mm_58: "f32[392, 320]" = torch.ops.aten.mm.default(view_549, permute_435);  permute_435 = None
    permute_436: "f32[640, 392]" = torch.ops.aten.permute.default(view_549, [1, 0])
    mm_59: "f32[640, 320]" = torch.ops.aten.mm.default(permute_436, view_349);  permute_436 = view_349 = None
    permute_437: "f32[320, 640]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_94: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_549, [0], True);  view_549 = None
    view_550: "f32[640]" = torch.ops.aten.view.default(sum_94, [640]);  sum_94 = None
    permute_438: "f32[640, 320]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    view_551: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_58, [8, 49, 320]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_194: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
    sub_131: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_194, getitem_273);  clone_194 = getitem_273 = None
    mul_403: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_70);  sub_131 = None
    mul_404: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_551, primals_421);  primals_421 = None
    mul_405: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_404, 320)
    sum_95: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True)
    mul_406: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_404, mul_403);  mul_404 = None
    sum_96: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [2], True);  mul_406 = None
    mul_407: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_403, sum_96);  sum_96 = None
    sub_132: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_405, sum_95);  mul_405 = sum_95 = None
    sub_133: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_132, mul_407);  sub_132 = mul_407 = None
    div_16: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_70, 320);  rsqrt_70 = None
    mul_408: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_16, sub_133);  div_16 = sub_133 = None
    mul_409: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_551, mul_403);  mul_403 = None
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
    permute_442: "f32[320, 320]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
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
    clone_196: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_206, memory_format = torch.contiguous_format);  add_206 = None
    sub_134: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_196, getitem_271);  clone_196 = getitem_271 = None
    mul_410: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_69);  sub_134 = None
    mul_411: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_195, primals_415);  primals_415 = None
    mul_412: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_411, 320)
    sum_101: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
    mul_413: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_411, mul_410);  mul_411 = None
    sum_102: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
    mul_414: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_410, sum_102);  sum_102 = None
    sub_135: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_412, sum_101);  mul_412 = sum_101 = None
    sub_136: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_135, mul_414);  sub_135 = mul_414 = None
    div_17: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_69, 320);  rsqrt_69 = None
    mul_415: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_17, sub_136);  div_17 = sub_136 = None
    mul_416: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_195, mul_410);  mul_410 = None
    sum_103: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
    sum_104: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_195, [0, 1]);  clone_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_290: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_288, mul_415);  add_288 = mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_197: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_290, memory_format = torch.contiguous_format)
    view_558: "f32[1568, 320]" = torch.ops.aten.view.default(clone_197, [1568, 320]);  clone_197 = None
    permute_446: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    mm_62: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_558, permute_446);  permute_446 = None
    permute_447: "f32[320, 1568]" = torch.ops.aten.permute.default(view_558, [1, 0])
    mm_63: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_447, view_342);  permute_447 = view_342 = None
    permute_448: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_105: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_558, [0], True);  view_558 = None
    view_559: "f32[320]" = torch.ops.aten.view.default(sum_105, [320]);  sum_105 = None
    permute_449: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_448, [1, 0]);  permute_448 = None
    view_560: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_62, [8, 196, 1280]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_417: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_341, 0.7071067811865476)
    erf_34: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_417);  mul_417 = None
    add_291: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_418: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_291, 0.5);  add_291 = None
    mul_419: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_341, view_341)
    mul_420: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_419, -0.5);  mul_419 = None
    exp_6: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_420);  mul_420 = None
    mul_421: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_422: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_341, mul_421);  view_341 = mul_421 = None
    add_292: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_418, mul_422);  mul_418 = mul_422 = None
    mul_423: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_560, add_292);  view_560 = add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_561: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_423, [1568, 1280]);  mul_423 = None
    permute_450: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    mm_64: "f32[1568, 320]" = torch.ops.aten.mm.default(view_561, permute_450);  permute_450 = None
    permute_451: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_561, [1, 0])
    mm_65: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_451, view_340);  permute_451 = view_340 = None
    permute_452: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_106: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
    view_562: "f32[1280]" = torch.ops.aten.view.default(sum_106, [1280]);  sum_106 = None
    permute_453: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_452, [1, 0]);  permute_452 = None
    view_563: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_64, [8, 196, 320]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_198: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_202, memory_format = torch.contiguous_format);  add_202 = None
    sub_137: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_198, getitem_269);  clone_198 = getitem_269 = None
    mul_424: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_68);  sub_137 = None
    mul_425: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_563, primals_409);  primals_409 = None
    mul_426: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_425, 320)
    sum_107: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True)
    mul_427: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_425, mul_424);  mul_425 = None
    sum_108: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
    mul_428: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_424, sum_108);  sum_108 = None
    sub_138: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_426, sum_107);  mul_426 = sum_107 = None
    sub_139: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_138, mul_428);  sub_138 = mul_428 = None
    div_18: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_68, 320);  rsqrt_68 = None
    mul_429: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_18, sub_139);  div_18 = sub_139 = None
    mul_430: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_563, mul_424);  mul_424 = None
    sum_109: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1]);  mul_430 = None
    sum_110: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_563, [0, 1]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_293: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_290, mul_429);  add_290 = mul_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_199: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_293, memory_format = torch.contiguous_format)
    view_564: "f32[1568, 320]" = torch.ops.aten.view.default(clone_199, [1568, 320]);  clone_199 = None
    permute_454: "f32[320, 320]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
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
    alias_34: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
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
    permute_460: "f32[640, 320]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    mm_68: "f32[392, 320]" = torch.ops.aten.mm.default(view_570, permute_460);  permute_460 = None
    permute_461: "f32[640, 392]" = torch.ops.aten.permute.default(view_570, [1, 0])
    mm_69: "f32[640, 320]" = torch.ops.aten.mm.default(permute_461, view_334);  permute_461 = view_334 = None
    permute_462: "f32[320, 640]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_112: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
    view_571: "f32[640]" = torch.ops.aten.view.default(sum_112, [640]);  sum_112 = None
    permute_463: "f32[640, 320]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    view_572: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_68, [8, 49, 320]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_201: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    sub_140: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_201, getitem_261);  clone_201 = getitem_261 = None
    mul_431: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_67);  sub_140 = None
    mul_432: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_572, primals_403);  primals_403 = None
    mul_433: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_432, 320)
    sum_113: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True)
    mul_434: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_432, mul_431);  mul_432 = None
    sum_114: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True);  mul_434 = None
    mul_435: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_431, sum_114);  sum_114 = None
    sub_141: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_433, sum_113);  mul_433 = sum_113 = None
    sub_142: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_141, mul_435);  sub_141 = mul_435 = None
    div_19: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_67, 320);  rsqrt_67 = None
    mul_436: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_19, sub_142);  div_19 = sub_142 = None
    mul_437: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_572, mul_431);  mul_431 = None
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
    permute_467: "f32[320, 320]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
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
    clone_203: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_197, memory_format = torch.contiguous_format);  add_197 = None
    sub_143: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_203, getitem_259);  clone_203 = getitem_259 = None
    mul_438: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_143, rsqrt_66);  sub_143 = None
    mul_439: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_202, primals_397);  primals_397 = None
    mul_440: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_439, 320)
    sum_119: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True)
    mul_441: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_439, mul_438);  mul_439 = None
    sum_120: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True);  mul_441 = None
    mul_442: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_438, sum_120);  sum_120 = None
    sub_144: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_440, sum_119);  mul_440 = sum_119 = None
    sub_145: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_144, mul_442);  sub_144 = mul_442 = None
    div_20: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_66, 320);  rsqrt_66 = None
    mul_443: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_20, sub_145);  div_20 = sub_145 = None
    mul_444: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_202, mul_438);  mul_438 = None
    sum_121: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 1]);  mul_444 = None
    sum_122: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_202, [0, 1]);  clone_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_295: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_293, mul_443);  add_293 = mul_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_204: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_295, memory_format = torch.contiguous_format)
    view_579: "f32[1568, 320]" = torch.ops.aten.view.default(clone_204, [1568, 320]);  clone_204 = None
    permute_471: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
    mm_72: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_579, permute_471);  permute_471 = None
    permute_472: "f32[320, 1568]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_73: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_472, view_327);  permute_472 = view_327 = None
    permute_473: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_123: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[320]" = torch.ops.aten.view.default(sum_123, [320]);  sum_123 = None
    permute_474: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_581: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_72, [8, 196, 1280]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_445: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_326, 0.7071067811865476)
    erf_35: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_445);  mul_445 = None
    add_296: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_446: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_296, 0.5);  add_296 = None
    mul_447: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_326, view_326)
    mul_448: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_447, -0.5);  mul_447 = None
    exp_7: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_448);  mul_448 = None
    mul_449: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_450: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_326, mul_449);  view_326 = mul_449 = None
    add_297: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_446, mul_450);  mul_446 = mul_450 = None
    mul_451: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_581, add_297);  view_581 = add_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_582: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_451, [1568, 1280]);  mul_451 = None
    permute_475: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    mm_74: "f32[1568, 320]" = torch.ops.aten.mm.default(view_582, permute_475);  permute_475 = None
    permute_476: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_75: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_476, view_325);  permute_476 = view_325 = None
    permute_477: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_124: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
    view_583: "f32[1280]" = torch.ops.aten.view.default(sum_124, [1280]);  sum_124 = None
    permute_478: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_584: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_74, [8, 196, 320]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_205: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_193, memory_format = torch.contiguous_format);  add_193 = None
    sub_146: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_205, getitem_257);  clone_205 = getitem_257 = None
    mul_452: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_146, rsqrt_65);  sub_146 = None
    mul_453: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_584, primals_391);  primals_391 = None
    mul_454: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_453, 320)
    sum_125: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True)
    mul_455: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_453, mul_452);  mul_453 = None
    sum_126: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True);  mul_455 = None
    mul_456: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_452, sum_126);  sum_126 = None
    sub_147: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_454, sum_125);  mul_454 = sum_125 = None
    sub_148: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_147, mul_456);  sub_147 = mul_456 = None
    div_21: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_65, 320);  rsqrt_65 = None
    mul_457: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_21, sub_148);  div_21 = sub_148 = None
    mul_458: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_584, mul_452);  mul_452 = None
    sum_127: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 1]);  mul_458 = None
    sum_128: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_584, [0, 1]);  view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_298: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_295, mul_457);  add_295 = mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_206: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_298, memory_format = torch.contiguous_format)
    view_585: "f32[1568, 320]" = torch.ops.aten.view.default(clone_206, [1568, 320]);  clone_206 = None
    permute_479: "f32[320, 320]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
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
    alias_35: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
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
    permute_485: "f32[640, 320]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    mm_78: "f32[392, 320]" = torch.ops.aten.mm.default(view_591, permute_485);  permute_485 = None
    permute_486: "f32[640, 392]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_79: "f32[640, 320]" = torch.ops.aten.mm.default(permute_486, view_319);  permute_486 = view_319 = None
    permute_487: "f32[320, 640]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_130: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
    view_592: "f32[640]" = torch.ops.aten.view.default(sum_130, [640]);  sum_130 = None
    permute_488: "f32[640, 320]" = torch.ops.aten.permute.default(permute_487, [1, 0]);  permute_487 = None
    view_593: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_78, [8, 49, 320]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_208: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
    sub_149: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_208, getitem_249);  clone_208 = getitem_249 = None
    mul_459: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_149, rsqrt_64);  sub_149 = None
    mul_460: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_593, primals_385);  primals_385 = None
    mul_461: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_460, 320)
    sum_131: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [2], True)
    mul_462: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_460, mul_459);  mul_460 = None
    sum_132: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_462, [2], True);  mul_462 = None
    mul_463: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_459, sum_132);  sum_132 = None
    sub_150: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_461, sum_131);  mul_461 = sum_131 = None
    sub_151: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_150, mul_463);  sub_150 = mul_463 = None
    div_22: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_64, 320);  rsqrt_64 = None
    mul_464: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_22, sub_151);  div_22 = sub_151 = None
    mul_465: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_593, mul_459);  mul_459 = None
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
    permute_492: "f32[320, 320]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
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
    clone_210: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_188, memory_format = torch.contiguous_format);  add_188 = None
    sub_152: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_210, getitem_247);  clone_210 = getitem_247 = None
    mul_466: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_152, rsqrt_63);  sub_152 = None
    mul_467: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_209, primals_379);  primals_379 = None
    mul_468: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_467, 320)
    sum_137: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True)
    mul_469: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_467, mul_466);  mul_467 = None
    sum_138: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [2], True);  mul_469 = None
    mul_470: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_466, sum_138);  sum_138 = None
    sub_153: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_468, sum_137);  mul_468 = sum_137 = None
    sub_154: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_153, mul_470);  sub_153 = mul_470 = None
    div_23: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_63, 320);  rsqrt_63 = None
    mul_471: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_23, sub_154);  div_23 = sub_154 = None
    mul_472: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_209, mul_466);  mul_466 = None
    sum_139: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1]);  mul_472 = None
    sum_140: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_209, [0, 1]);  clone_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_300: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_298, mul_471);  add_298 = mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_211: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_300, memory_format = torch.contiguous_format)
    view_600: "f32[1568, 320]" = torch.ops.aten.view.default(clone_211, [1568, 320]);  clone_211 = None
    permute_496: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    mm_82: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_600, permute_496);  permute_496 = None
    permute_497: "f32[320, 1568]" = torch.ops.aten.permute.default(view_600, [1, 0])
    mm_83: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_497, view_312);  permute_497 = view_312 = None
    permute_498: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_141: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True);  view_600 = None
    view_601: "f32[320]" = torch.ops.aten.view.default(sum_141, [320]);  sum_141 = None
    permute_499: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_498, [1, 0]);  permute_498 = None
    view_602: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_82, [8, 196, 1280]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_473: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_311, 0.7071067811865476)
    erf_36: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_473);  mul_473 = None
    add_301: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_474: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_301, 0.5);  add_301 = None
    mul_475: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_311, view_311)
    mul_476: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_475, -0.5);  mul_475 = None
    exp_8: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_476);  mul_476 = None
    mul_477: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_478: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_311, mul_477);  view_311 = mul_477 = None
    add_302: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_474, mul_478);  mul_474 = mul_478 = None
    mul_479: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_602, add_302);  view_602 = add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_603: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_479, [1568, 1280]);  mul_479 = None
    permute_500: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
    mm_84: "f32[1568, 320]" = torch.ops.aten.mm.default(view_603, permute_500);  permute_500 = None
    permute_501: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_603, [1, 0])
    mm_85: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_501, view_310);  permute_501 = view_310 = None
    permute_502: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_142: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
    view_604: "f32[1280]" = torch.ops.aten.view.default(sum_142, [1280]);  sum_142 = None
    permute_503: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    view_605: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_84, [8, 196, 320]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_212: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_184, memory_format = torch.contiguous_format);  add_184 = None
    sub_155: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_212, getitem_245);  clone_212 = getitem_245 = None
    mul_480: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_155, rsqrt_62);  sub_155 = None
    mul_481: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_605, primals_373);  primals_373 = None
    mul_482: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_481, 320)
    sum_143: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_481, [2], True)
    mul_483: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_481, mul_480);  mul_481 = None
    sum_144: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True);  mul_483 = None
    mul_484: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_480, sum_144);  sum_144 = None
    sub_156: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_482, sum_143);  mul_482 = sum_143 = None
    sub_157: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_156, mul_484);  sub_156 = mul_484 = None
    div_24: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_62, 320);  rsqrt_62 = None
    mul_485: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_24, sub_157);  div_24 = sub_157 = None
    mul_486: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_605, mul_480);  mul_480 = None
    sum_145: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_486, [0, 1]);  mul_486 = None
    sum_146: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_605, [0, 1]);  view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_303: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_300, mul_485);  add_300 = mul_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_213: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_303, memory_format = torch.contiguous_format)
    view_606: "f32[1568, 320]" = torch.ops.aten.view.default(clone_213, [1568, 320]);  clone_213 = None
    permute_504: "f32[320, 320]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
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
    alias_36: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
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
    permute_510: "f32[640, 320]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    mm_88: "f32[392, 320]" = torch.ops.aten.mm.default(view_612, permute_510);  permute_510 = None
    permute_511: "f32[640, 392]" = torch.ops.aten.permute.default(view_612, [1, 0])
    mm_89: "f32[640, 320]" = torch.ops.aten.mm.default(permute_511, view_304);  permute_511 = view_304 = None
    permute_512: "f32[320, 640]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_148: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
    view_613: "f32[640]" = torch.ops.aten.view.default(sum_148, [640]);  sum_148 = None
    permute_513: "f32[640, 320]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    view_614: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_88, [8, 49, 320]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_215: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
    sub_158: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_215, getitem_237);  clone_215 = getitem_237 = None
    mul_487: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_158, rsqrt_61);  sub_158 = None
    mul_488: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_614, primals_367);  primals_367 = None
    mul_489: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_488, 320)
    sum_149: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_488, [2], True)
    mul_490: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_488, mul_487);  mul_488 = None
    sum_150: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_490, [2], True);  mul_490 = None
    mul_491: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_487, sum_150);  sum_150 = None
    sub_159: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_489, sum_149);  mul_489 = sum_149 = None
    sub_160: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_159, mul_491);  sub_159 = mul_491 = None
    div_25: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_61, 320);  rsqrt_61 = None
    mul_492: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_25, sub_160);  div_25 = sub_160 = None
    mul_493: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_614, mul_487);  mul_487 = None
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
    permute_517: "f32[320, 320]" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
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
    clone_217: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_179, memory_format = torch.contiguous_format);  add_179 = None
    sub_161: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_217, getitem_235);  clone_217 = getitem_235 = None
    mul_494: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_161, rsqrt_60);  sub_161 = None
    mul_495: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_216, primals_361);  primals_361 = None
    mul_496: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_495, 320)
    sum_155: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True)
    mul_497: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_495, mul_494);  mul_495 = None
    sum_156: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
    mul_498: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_494, sum_156);  sum_156 = None
    sub_162: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_496, sum_155);  mul_496 = sum_155 = None
    sub_163: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_162, mul_498);  sub_162 = mul_498 = None
    div_26: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_60, 320);  rsqrt_60 = None
    mul_499: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_26, sub_163);  div_26 = sub_163 = None
    mul_500: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_216, mul_494);  mul_494 = None
    sum_157: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1]);  mul_500 = None
    sum_158: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_216, [0, 1]);  clone_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_305: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_303, mul_499);  add_303 = mul_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_218: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_305, memory_format = torch.contiguous_format)
    view_621: "f32[1568, 320]" = torch.ops.aten.view.default(clone_218, [1568, 320]);  clone_218 = None
    permute_521: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
    mm_92: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_621, permute_521);  permute_521 = None
    permute_522: "f32[320, 1568]" = torch.ops.aten.permute.default(view_621, [1, 0])
    mm_93: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_522, view_297);  permute_522 = view_297 = None
    permute_523: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_159: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_621, [0], True);  view_621 = None
    view_622: "f32[320]" = torch.ops.aten.view.default(sum_159, [320]);  sum_159 = None
    permute_524: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_523, [1, 0]);  permute_523 = None
    view_623: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_92, [8, 196, 1280]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_501: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_296, 0.7071067811865476)
    erf_37: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_501);  mul_501 = None
    add_306: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_502: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_306, 0.5);  add_306 = None
    mul_503: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_296, view_296)
    mul_504: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_503, -0.5);  mul_503 = None
    exp_9: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_504);  mul_504 = None
    mul_505: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_506: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_296, mul_505);  view_296 = mul_505 = None
    add_307: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_502, mul_506);  mul_502 = mul_506 = None
    mul_507: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_623, add_307);  view_623 = add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_624: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_507, [1568, 1280]);  mul_507 = None
    permute_525: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    mm_94: "f32[1568, 320]" = torch.ops.aten.mm.default(view_624, permute_525);  permute_525 = None
    permute_526: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_624, [1, 0])
    mm_95: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_526, view_295);  permute_526 = view_295 = None
    permute_527: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_160: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_624, [0], True);  view_624 = None
    view_625: "f32[1280]" = torch.ops.aten.view.default(sum_160, [1280]);  sum_160 = None
    permute_528: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_626: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_94, [8, 196, 320]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_219: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_175, memory_format = torch.contiguous_format);  add_175 = None
    sub_164: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_219, getitem_233);  clone_219 = getitem_233 = None
    mul_508: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_164, rsqrt_59);  sub_164 = None
    mul_509: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_626, primals_355);  primals_355 = None
    mul_510: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_509, 320)
    sum_161: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [2], True)
    mul_511: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_509, mul_508);  mul_509 = None
    sum_162: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_511, [2], True);  mul_511 = None
    mul_512: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_508, sum_162);  sum_162 = None
    sub_165: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_510, sum_161);  mul_510 = sum_161 = None
    sub_166: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_165, mul_512);  sub_165 = mul_512 = None
    div_27: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_59, 320);  rsqrt_59 = None
    mul_513: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_27, sub_166);  div_27 = sub_166 = None
    mul_514: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_626, mul_508);  mul_508 = None
    sum_163: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_514, [0, 1]);  mul_514 = None
    sum_164: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_626, [0, 1]);  view_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_308: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_305, mul_513);  add_305 = mul_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_220: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_308, memory_format = torch.contiguous_format)
    view_627: "f32[1568, 320]" = torch.ops.aten.view.default(clone_220, [1568, 320]);  clone_220 = None
    permute_529: "f32[320, 320]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
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
    alias_37: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
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
    permute_535: "f32[640, 320]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    mm_98: "f32[392, 320]" = torch.ops.aten.mm.default(view_633, permute_535);  permute_535 = None
    permute_536: "f32[640, 392]" = torch.ops.aten.permute.default(view_633, [1, 0])
    mm_99: "f32[640, 320]" = torch.ops.aten.mm.default(permute_536, view_289);  permute_536 = view_289 = None
    permute_537: "f32[320, 640]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_166: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_633, [0], True);  view_633 = None
    view_634: "f32[640]" = torch.ops.aten.view.default(sum_166, [640]);  sum_166 = None
    permute_538: "f32[640, 320]" = torch.ops.aten.permute.default(permute_537, [1, 0]);  permute_537 = None
    view_635: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_98, [8, 49, 320]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_222: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
    sub_167: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_222, getitem_225);  clone_222 = getitem_225 = None
    mul_515: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_167, rsqrt_58);  sub_167 = None
    mul_516: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_635, primals_349);  primals_349 = None
    mul_517: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_516, 320)
    sum_167: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_516, [2], True)
    mul_518: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_516, mul_515);  mul_516 = None
    sum_168: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_518, [2], True);  mul_518 = None
    mul_519: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_515, sum_168);  sum_168 = None
    sub_168: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_517, sum_167);  mul_517 = sum_167 = None
    sub_169: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_168, mul_519);  sub_168 = mul_519 = None
    div_28: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_58, 320);  rsqrt_58 = None
    mul_520: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_28, sub_169);  div_28 = sub_169 = None
    mul_521: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_635, mul_515);  mul_515 = None
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
    permute_542: "f32[320, 320]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
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
    clone_224: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_170, memory_format = torch.contiguous_format);  add_170 = None
    sub_170: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_224, getitem_223);  clone_224 = getitem_223 = None
    mul_522: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_170, rsqrt_57);  sub_170 = None
    mul_523: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_223, primals_343);  primals_343 = None
    mul_524: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_523, 320)
    sum_173: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_523, [2], True)
    mul_525: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_523, mul_522);  mul_523 = None
    sum_174: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_525, [2], True);  mul_525 = None
    mul_526: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_522, sum_174);  sum_174 = None
    sub_171: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_524, sum_173);  mul_524 = sum_173 = None
    sub_172: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_171, mul_526);  sub_171 = mul_526 = None
    div_29: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_57, 320);  rsqrt_57 = None
    mul_527: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_29, sub_172);  div_29 = sub_172 = None
    mul_528: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_223, mul_522);  mul_522 = None
    sum_175: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 1]);  mul_528 = None
    sum_176: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_223, [0, 1]);  clone_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_310: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_308, mul_527);  add_308 = mul_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_225: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_310, memory_format = torch.contiguous_format)
    view_642: "f32[1568, 320]" = torch.ops.aten.view.default(clone_225, [1568, 320]);  clone_225 = None
    permute_546: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    mm_102: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_642, permute_546);  permute_546 = None
    permute_547: "f32[320, 1568]" = torch.ops.aten.permute.default(view_642, [1, 0])
    mm_103: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_547, view_282);  permute_547 = view_282 = None
    permute_548: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_177: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_642, [0], True);  view_642 = None
    view_643: "f32[320]" = torch.ops.aten.view.default(sum_177, [320]);  sum_177 = None
    permute_549: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_548, [1, 0]);  permute_548 = None
    view_644: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_102, [8, 196, 1280]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_529: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_281, 0.7071067811865476)
    erf_38: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_529);  mul_529 = None
    add_311: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    mul_530: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_311, 0.5);  add_311 = None
    mul_531: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_281, view_281)
    mul_532: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_531, -0.5);  mul_531 = None
    exp_10: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_532);  mul_532 = None
    mul_533: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_534: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_281, mul_533);  view_281 = mul_533 = None
    add_312: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_530, mul_534);  mul_530 = mul_534 = None
    mul_535: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_644, add_312);  view_644 = add_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_645: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_535, [1568, 1280]);  mul_535 = None
    permute_550: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    mm_104: "f32[1568, 320]" = torch.ops.aten.mm.default(view_645, permute_550);  permute_550 = None
    permute_551: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_645, [1, 0])
    mm_105: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_551, view_280);  permute_551 = view_280 = None
    permute_552: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_178: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_645, [0], True);  view_645 = None
    view_646: "f32[1280]" = torch.ops.aten.view.default(sum_178, [1280]);  sum_178 = None
    permute_553: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_552, [1, 0]);  permute_552 = None
    view_647: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_104, [8, 196, 320]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_226: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_166, memory_format = torch.contiguous_format);  add_166 = None
    sub_173: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_226, getitem_221);  clone_226 = getitem_221 = None
    mul_536: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_173, rsqrt_56);  sub_173 = None
    mul_537: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_647, primals_337);  primals_337 = None
    mul_538: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_537, 320)
    sum_179: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_537, [2], True)
    mul_539: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_537, mul_536);  mul_537 = None
    sum_180: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_539, [2], True);  mul_539 = None
    mul_540: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_536, sum_180);  sum_180 = None
    sub_174: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_538, sum_179);  mul_538 = sum_179 = None
    sub_175: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_174, mul_540);  sub_174 = mul_540 = None
    div_30: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_56, 320);  rsqrt_56 = None
    mul_541: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_30, sub_175);  div_30 = sub_175 = None
    mul_542: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_647, mul_536);  mul_536 = None
    sum_181: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_542, [0, 1]);  mul_542 = None
    sum_182: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_647, [0, 1]);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_313: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_310, mul_541);  add_310 = mul_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_227: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_313, memory_format = torch.contiguous_format)
    view_648: "f32[1568, 320]" = torch.ops.aten.view.default(clone_227, [1568, 320]);  clone_227 = None
    permute_554: "f32[320, 320]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
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
    alias_38: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
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
    permute_560: "f32[640, 320]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    mm_108: "f32[392, 320]" = torch.ops.aten.mm.default(view_654, permute_560);  permute_560 = None
    permute_561: "f32[640, 392]" = torch.ops.aten.permute.default(view_654, [1, 0])
    mm_109: "f32[640, 320]" = torch.ops.aten.mm.default(permute_561, view_274);  permute_561 = view_274 = None
    permute_562: "f32[320, 640]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_184: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_654, [0], True);  view_654 = None
    view_655: "f32[640]" = torch.ops.aten.view.default(sum_184, [640]);  sum_184 = None
    permute_563: "f32[640, 320]" = torch.ops.aten.permute.default(permute_562, [1, 0]);  permute_562 = None
    view_656: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_108, [8, 49, 320]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_229: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
    sub_176: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_229, getitem_213);  clone_229 = getitem_213 = None
    mul_543: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_176, rsqrt_55);  sub_176 = None
    mul_544: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_656, primals_331);  primals_331 = None
    mul_545: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_544, 320)
    sum_185: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_544, [2], True)
    mul_546: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_544, mul_543);  mul_544 = None
    sum_186: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_546, [2], True);  mul_546 = None
    mul_547: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_543, sum_186);  sum_186 = None
    sub_177: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_545, sum_185);  mul_545 = sum_185 = None
    sub_178: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_177, mul_547);  sub_177 = mul_547 = None
    div_31: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_55, 320);  rsqrt_55 = None
    mul_548: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_31, sub_178);  div_31 = sub_178 = None
    mul_549: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_656, mul_543);  mul_543 = None
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
    permute_567: "f32[320, 320]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
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
    clone_231: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_161, memory_format = torch.contiguous_format);  add_161 = None
    sub_179: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_231, getitem_211);  clone_231 = getitem_211 = None
    mul_550: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_179, rsqrt_54);  sub_179 = None
    mul_551: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_230, primals_325);  primals_325 = None
    mul_552: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_551, 320)
    sum_191: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_551, [2], True)
    mul_553: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_551, mul_550);  mul_551 = None
    sum_192: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_553, [2], True);  mul_553 = None
    mul_554: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_550, sum_192);  sum_192 = None
    sub_180: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_552, sum_191);  mul_552 = sum_191 = None
    sub_181: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_180, mul_554);  sub_180 = mul_554 = None
    div_32: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_54, 320);  rsqrt_54 = None
    mul_555: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_32, sub_181);  div_32 = sub_181 = None
    mul_556: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_230, mul_550);  mul_550 = None
    sum_193: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 1]);  mul_556 = None
    sum_194: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_230, [0, 1]);  clone_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_315: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_313, mul_555);  add_313 = mul_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_232: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_315, memory_format = torch.contiguous_format)
    view_663: "f32[1568, 320]" = torch.ops.aten.view.default(clone_232, [1568, 320]);  clone_232 = None
    permute_571: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
    mm_112: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_663, permute_571);  permute_571 = None
    permute_572: "f32[320, 1568]" = torch.ops.aten.permute.default(view_663, [1, 0])
    mm_113: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_572, view_267);  permute_572 = view_267 = None
    permute_573: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_195: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_663, [0], True);  view_663 = None
    view_664: "f32[320]" = torch.ops.aten.view.default(sum_195, [320]);  sum_195 = None
    permute_574: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_573, [1, 0]);  permute_573 = None
    view_665: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_112, [8, 196, 1280]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_557: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_266, 0.7071067811865476)
    erf_39: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_557);  mul_557 = None
    add_316: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    mul_558: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_316, 0.5);  add_316 = None
    mul_559: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_266, view_266)
    mul_560: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_559, -0.5);  mul_559 = None
    exp_11: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_560);  mul_560 = None
    mul_561: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_562: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_266, mul_561);  view_266 = mul_561 = None
    add_317: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_558, mul_562);  mul_558 = mul_562 = None
    mul_563: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_665, add_317);  view_665 = add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_666: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_563, [1568, 1280]);  mul_563 = None
    permute_575: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    mm_114: "f32[1568, 320]" = torch.ops.aten.mm.default(view_666, permute_575);  permute_575 = None
    permute_576: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_666, [1, 0])
    mm_115: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_576, view_265);  permute_576 = view_265 = None
    permute_577: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_196: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_666, [0], True);  view_666 = None
    view_667: "f32[1280]" = torch.ops.aten.view.default(sum_196, [1280]);  sum_196 = None
    permute_578: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_577, [1, 0]);  permute_577 = None
    view_668: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_114, [8, 196, 320]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_233: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_157, memory_format = torch.contiguous_format);  add_157 = None
    sub_182: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_233, getitem_209);  clone_233 = getitem_209 = None
    mul_564: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_182, rsqrt_53);  sub_182 = None
    mul_565: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_668, primals_319);  primals_319 = None
    mul_566: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_565, 320)
    sum_197: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_565, [2], True)
    mul_567: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_565, mul_564);  mul_565 = None
    sum_198: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_567, [2], True);  mul_567 = None
    mul_568: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_564, sum_198);  sum_198 = None
    sub_183: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_566, sum_197);  mul_566 = sum_197 = None
    sub_184: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_183, mul_568);  sub_183 = mul_568 = None
    div_33: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_53, 320);  rsqrt_53 = None
    mul_569: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_33, sub_184);  div_33 = sub_184 = None
    mul_570: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_668, mul_564);  mul_564 = None
    sum_199: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_570, [0, 1]);  mul_570 = None
    sum_200: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_668, [0, 1]);  view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_318: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_315, mul_569);  add_315 = mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_234: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_318, memory_format = torch.contiguous_format)
    view_669: "f32[1568, 320]" = torch.ops.aten.view.default(clone_234, [1568, 320]);  clone_234 = None
    permute_579: "f32[320, 320]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
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
    alias_39: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
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
    permute_585: "f32[640, 320]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    mm_118: "f32[392, 320]" = torch.ops.aten.mm.default(view_675, permute_585);  permute_585 = None
    permute_586: "f32[640, 392]" = torch.ops.aten.permute.default(view_675, [1, 0])
    mm_119: "f32[640, 320]" = torch.ops.aten.mm.default(permute_586, view_259);  permute_586 = view_259 = None
    permute_587: "f32[320, 640]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_202: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_675, [0], True);  view_675 = None
    view_676: "f32[640]" = torch.ops.aten.view.default(sum_202, [640]);  sum_202 = None
    permute_588: "f32[640, 320]" = torch.ops.aten.permute.default(permute_587, [1, 0]);  permute_587 = None
    view_677: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_118, [8, 49, 320]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_236: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    sub_185: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_236, getitem_201);  clone_236 = getitem_201 = None
    mul_571: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_185, rsqrt_52);  sub_185 = None
    mul_572: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_677, primals_313);  primals_313 = None
    mul_573: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_572, 320)
    sum_203: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [2], True)
    mul_574: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_572, mul_571);  mul_572 = None
    sum_204: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_574, [2], True);  mul_574 = None
    mul_575: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_571, sum_204);  sum_204 = None
    sub_186: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_573, sum_203);  mul_573 = sum_203 = None
    sub_187: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_186, mul_575);  sub_186 = mul_575 = None
    div_34: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_52, 320);  rsqrt_52 = None
    mul_576: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_34, sub_187);  div_34 = sub_187 = None
    mul_577: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_677, mul_571);  mul_571 = None
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
    permute_592: "f32[320, 320]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
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
    clone_238: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_152, memory_format = torch.contiguous_format);  add_152 = None
    sub_188: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_238, getitem_199);  clone_238 = getitem_199 = None
    mul_578: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_188, rsqrt_51);  sub_188 = None
    mul_579: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_237, primals_307);  primals_307 = None
    mul_580: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_579, 320)
    sum_209: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_579, [2], True)
    mul_581: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_579, mul_578);  mul_579 = None
    sum_210: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [2], True);  mul_581 = None
    mul_582: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_578, sum_210);  sum_210 = None
    sub_189: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_580, sum_209);  mul_580 = sum_209 = None
    sub_190: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_189, mul_582);  sub_189 = mul_582 = None
    div_35: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_51, 320);  rsqrt_51 = None
    mul_583: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_35, sub_190);  div_35 = sub_190 = None
    mul_584: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_237, mul_578);  mul_578 = None
    sum_211: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 1]);  mul_584 = None
    sum_212: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_237, [0, 1]);  clone_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_320: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_318, mul_583);  add_318 = mul_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_239: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_320, memory_format = torch.contiguous_format)
    view_684: "f32[1568, 320]" = torch.ops.aten.view.default(clone_239, [1568, 320]);  clone_239 = None
    permute_596: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    mm_122: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_684, permute_596);  permute_596 = None
    permute_597: "f32[320, 1568]" = torch.ops.aten.permute.default(view_684, [1, 0])
    mm_123: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_597, view_252);  permute_597 = view_252 = None
    permute_598: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_213: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_684, [0], True);  view_684 = None
    view_685: "f32[320]" = torch.ops.aten.view.default(sum_213, [320]);  sum_213 = None
    permute_599: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_598, [1, 0]);  permute_598 = None
    view_686: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_122, [8, 196, 1280]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_585: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_251, 0.7071067811865476)
    erf_40: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_585);  mul_585 = None
    add_321: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
    mul_586: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_321, 0.5);  add_321 = None
    mul_587: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_251, view_251)
    mul_588: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_587, -0.5);  mul_587 = None
    exp_12: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_588);  mul_588 = None
    mul_589: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_590: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_251, mul_589);  view_251 = mul_589 = None
    add_322: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_586, mul_590);  mul_586 = mul_590 = None
    mul_591: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_686, add_322);  view_686 = add_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_687: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_591, [1568, 1280]);  mul_591 = None
    permute_600: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    mm_124: "f32[1568, 320]" = torch.ops.aten.mm.default(view_687, permute_600);  permute_600 = None
    permute_601: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_687, [1, 0])
    mm_125: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_601, view_250);  permute_601 = view_250 = None
    permute_602: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_214: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_687, [0], True);  view_687 = None
    view_688: "f32[1280]" = torch.ops.aten.view.default(sum_214, [1280]);  sum_214 = None
    permute_603: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_602, [1, 0]);  permute_602 = None
    view_689: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_124, [8, 196, 320]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_240: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_148, memory_format = torch.contiguous_format);  add_148 = None
    sub_191: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_240, getitem_197);  clone_240 = getitem_197 = None
    mul_592: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_191, rsqrt_50);  sub_191 = None
    mul_593: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_689, primals_301);  primals_301 = None
    mul_594: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_593, 320)
    sum_215: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_593, [2], True)
    mul_595: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_593, mul_592);  mul_593 = None
    sum_216: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_595, [2], True);  mul_595 = None
    mul_596: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_592, sum_216);  sum_216 = None
    sub_192: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_594, sum_215);  mul_594 = sum_215 = None
    sub_193: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_192, mul_596);  sub_192 = mul_596 = None
    div_36: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 320);  rsqrt_50 = None
    mul_597: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_36, sub_193);  div_36 = sub_193 = None
    mul_598: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_689, mul_592);  mul_592 = None
    sum_217: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 1]);  mul_598 = None
    sum_218: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_689, [0, 1]);  view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_323: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_320, mul_597);  add_320 = mul_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_241: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_323, memory_format = torch.contiguous_format)
    view_690: "f32[1568, 320]" = torch.ops.aten.view.default(clone_241, [1568, 320]);  clone_241 = None
    permute_604: "f32[320, 320]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
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
    alias_40: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
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
    permute_610: "f32[640, 320]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    mm_128: "f32[392, 320]" = torch.ops.aten.mm.default(view_696, permute_610);  permute_610 = None
    permute_611: "f32[640, 392]" = torch.ops.aten.permute.default(view_696, [1, 0])
    mm_129: "f32[640, 320]" = torch.ops.aten.mm.default(permute_611, view_244);  permute_611 = view_244 = None
    permute_612: "f32[320, 640]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_220: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_696, [0], True);  view_696 = None
    view_697: "f32[640]" = torch.ops.aten.view.default(sum_220, [640]);  sum_220 = None
    permute_613: "f32[640, 320]" = torch.ops.aten.permute.default(permute_612, [1, 0]);  permute_612 = None
    view_698: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_128, [8, 49, 320]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_243: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
    sub_194: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_243, getitem_189);  clone_243 = getitem_189 = None
    mul_599: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_194, rsqrt_49);  sub_194 = None
    mul_600: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_698, primals_295);  primals_295 = None
    mul_601: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_600, 320)
    sum_221: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_600, [2], True)
    mul_602: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_600, mul_599);  mul_600 = None
    sum_222: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_602, [2], True);  mul_602 = None
    mul_603: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_599, sum_222);  sum_222 = None
    sub_195: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_601, sum_221);  mul_601 = sum_221 = None
    sub_196: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_195, mul_603);  sub_195 = mul_603 = None
    div_37: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 320);  rsqrt_49 = None
    mul_604: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_37, sub_196);  div_37 = sub_196 = None
    mul_605: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_698, mul_599);  mul_599 = None
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
    permute_617: "f32[320, 320]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
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
    clone_245: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_143, memory_format = torch.contiguous_format);  add_143 = None
    sub_197: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_245, getitem_187);  clone_245 = getitem_187 = None
    mul_606: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_197, rsqrt_48);  sub_197 = None
    mul_607: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_244, primals_289);  primals_289 = None
    mul_608: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_607, 320)
    sum_227: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_607, [2], True)
    mul_609: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_607, mul_606);  mul_607 = None
    sum_228: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_609, [2], True);  mul_609 = None
    mul_610: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_606, sum_228);  sum_228 = None
    sub_198: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_608, sum_227);  mul_608 = sum_227 = None
    sub_199: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_198, mul_610);  sub_198 = mul_610 = None
    div_38: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 320);  rsqrt_48 = None
    mul_611: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_38, sub_199);  div_38 = sub_199 = None
    mul_612: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_244, mul_606);  mul_606 = None
    sum_229: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_612, [0, 1]);  mul_612 = None
    sum_230: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_244, [0, 1]);  clone_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_325: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_323, mul_611);  add_323 = mul_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_246: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_325, memory_format = torch.contiguous_format)
    view_705: "f32[1568, 320]" = torch.ops.aten.view.default(clone_246, [1568, 320]);  clone_246 = None
    permute_621: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    mm_132: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_705, permute_621);  permute_621 = None
    permute_622: "f32[320, 1568]" = torch.ops.aten.permute.default(view_705, [1, 0])
    mm_133: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_622, view_237);  permute_622 = view_237 = None
    permute_623: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_231: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_705, [0], True);  view_705 = None
    view_706: "f32[320]" = torch.ops.aten.view.default(sum_231, [320]);  sum_231 = None
    permute_624: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_623, [1, 0]);  permute_623 = None
    view_707: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_132, [8, 196, 1280]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_613: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_236, 0.7071067811865476)
    erf_41: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_613);  mul_613 = None
    add_326: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
    mul_614: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_326, 0.5);  add_326 = None
    mul_615: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_236, view_236)
    mul_616: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_615, -0.5);  mul_615 = None
    exp_13: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_616);  mul_616 = None
    mul_617: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_618: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_236, mul_617);  view_236 = mul_617 = None
    add_327: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_614, mul_618);  mul_614 = mul_618 = None
    mul_619: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_707, add_327);  view_707 = add_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_708: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_619, [1568, 1280]);  mul_619 = None
    permute_625: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    mm_134: "f32[1568, 320]" = torch.ops.aten.mm.default(view_708, permute_625);  permute_625 = None
    permute_626: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_708, [1, 0])
    mm_135: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_626, view_235);  permute_626 = view_235 = None
    permute_627: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_232: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_708, [0], True);  view_708 = None
    view_709: "f32[1280]" = torch.ops.aten.view.default(sum_232, [1280]);  sum_232 = None
    permute_628: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_627, [1, 0]);  permute_627 = None
    view_710: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_134, [8, 196, 320]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_247: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_139, memory_format = torch.contiguous_format);  add_139 = None
    sub_200: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_247, getitem_185);  clone_247 = getitem_185 = None
    mul_620: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_200, rsqrt_47);  sub_200 = None
    mul_621: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_710, primals_283);  primals_283 = None
    mul_622: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_621, 320)
    sum_233: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_621, [2], True)
    mul_623: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_621, mul_620);  mul_621 = None
    sum_234: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_623, [2], True);  mul_623 = None
    mul_624: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_620, sum_234);  sum_234 = None
    sub_201: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_622, sum_233);  mul_622 = sum_233 = None
    sub_202: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_201, mul_624);  sub_201 = mul_624 = None
    div_39: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 320);  rsqrt_47 = None
    mul_625: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_39, sub_202);  div_39 = sub_202 = None
    mul_626: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_710, mul_620);  mul_620 = None
    sum_235: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 1]);  mul_626 = None
    sum_236: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_710, [0, 1]);  view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_328: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_325, mul_625);  add_325 = mul_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_248: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_328, memory_format = torch.contiguous_format)
    view_711: "f32[1568, 320]" = torch.ops.aten.view.default(clone_248, [1568, 320]);  clone_248 = None
    permute_629: "f32[320, 320]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
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
    alias_41: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
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
    permute_635: "f32[640, 320]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    mm_138: "f32[392, 320]" = torch.ops.aten.mm.default(view_717, permute_635);  permute_635 = None
    permute_636: "f32[640, 392]" = torch.ops.aten.permute.default(view_717, [1, 0])
    mm_139: "f32[640, 320]" = torch.ops.aten.mm.default(permute_636, view_229);  permute_636 = view_229 = None
    permute_637: "f32[320, 640]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_238: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_717, [0], True);  view_717 = None
    view_718: "f32[640]" = torch.ops.aten.view.default(sum_238, [640]);  sum_238 = None
    permute_638: "f32[640, 320]" = torch.ops.aten.permute.default(permute_637, [1, 0]);  permute_637 = None
    view_719: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_138, [8, 49, 320]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_250: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    sub_203: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_250, getitem_177);  clone_250 = getitem_177 = None
    mul_627: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_203, rsqrt_46);  sub_203 = None
    mul_628: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_719, primals_277);  primals_277 = None
    mul_629: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_628, 320)
    sum_239: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_628, [2], True)
    mul_630: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_628, mul_627);  mul_628 = None
    sum_240: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_630, [2], True);  mul_630 = None
    mul_631: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_627, sum_240);  sum_240 = None
    sub_204: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_629, sum_239);  mul_629 = sum_239 = None
    sub_205: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_204, mul_631);  sub_204 = mul_631 = None
    div_40: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 320);  rsqrt_46 = None
    mul_632: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_40, sub_205);  div_40 = sub_205 = None
    mul_633: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_719, mul_627);  mul_627 = None
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
    permute_642: "f32[320, 320]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
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
    clone_252: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_134, memory_format = torch.contiguous_format);  add_134 = None
    sub_206: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_252, getitem_175);  clone_252 = getitem_175 = None
    mul_634: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_206, rsqrt_45);  sub_206 = None
    mul_635: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_251, primals_271);  primals_271 = None
    mul_636: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_635, 320)
    sum_245: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_635, [2], True)
    mul_637: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_635, mul_634);  mul_635 = None
    sum_246: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_637, [2], True);  mul_637 = None
    mul_638: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_634, sum_246);  sum_246 = None
    sub_207: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_636, sum_245);  mul_636 = sum_245 = None
    sub_208: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_207, mul_638);  sub_207 = mul_638 = None
    div_41: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 320);  rsqrt_45 = None
    mul_639: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_41, sub_208);  div_41 = sub_208 = None
    mul_640: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_251, mul_634);  mul_634 = None
    sum_247: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 1]);  mul_640 = None
    sum_248: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_251, [0, 1]);  clone_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_330: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_328, mul_639);  add_328 = mul_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_253: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_330, memory_format = torch.contiguous_format)
    view_726: "f32[1568, 320]" = torch.ops.aten.view.default(clone_253, [1568, 320]);  clone_253 = None
    permute_646: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    mm_142: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_726, permute_646);  permute_646 = None
    permute_647: "f32[320, 1568]" = torch.ops.aten.permute.default(view_726, [1, 0])
    mm_143: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_647, view_222);  permute_647 = view_222 = None
    permute_648: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_249: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_726, [0], True);  view_726 = None
    view_727: "f32[320]" = torch.ops.aten.view.default(sum_249, [320]);  sum_249 = None
    permute_649: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_648, [1, 0]);  permute_648 = None
    view_728: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_142, [8, 196, 1280]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_641: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_221, 0.7071067811865476)
    erf_42: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_641);  mul_641 = None
    add_331: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
    mul_642: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_331, 0.5);  add_331 = None
    mul_643: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_221, view_221)
    mul_644: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_643, -0.5);  mul_643 = None
    exp_14: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_644);  mul_644 = None
    mul_645: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_646: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_221, mul_645);  view_221 = mul_645 = None
    add_332: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_642, mul_646);  mul_642 = mul_646 = None
    mul_647: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_728, add_332);  view_728 = add_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_729: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_647, [1568, 1280]);  mul_647 = None
    permute_650: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    mm_144: "f32[1568, 320]" = torch.ops.aten.mm.default(view_729, permute_650);  permute_650 = None
    permute_651: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_729, [1, 0])
    mm_145: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_651, view_220);  permute_651 = view_220 = None
    permute_652: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_250: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_729, [0], True);  view_729 = None
    view_730: "f32[1280]" = torch.ops.aten.view.default(sum_250, [1280]);  sum_250 = None
    permute_653: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_652, [1, 0]);  permute_652 = None
    view_731: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_144, [8, 196, 320]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_254: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_130, memory_format = torch.contiguous_format);  add_130 = None
    sub_209: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_254, getitem_173);  clone_254 = getitem_173 = None
    mul_648: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_209, rsqrt_44);  sub_209 = None
    mul_649: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_731, primals_265);  primals_265 = None
    mul_650: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_649, 320)
    sum_251: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_649, [2], True)
    mul_651: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_649, mul_648);  mul_649 = None
    sum_252: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_651, [2], True);  mul_651 = None
    mul_652: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_648, sum_252);  sum_252 = None
    sub_210: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_650, sum_251);  mul_650 = sum_251 = None
    sub_211: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_210, mul_652);  sub_210 = mul_652 = None
    div_42: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 320);  rsqrt_44 = None
    mul_653: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_42, sub_211);  div_42 = sub_211 = None
    mul_654: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_731, mul_648);  mul_648 = None
    sum_253: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_654, [0, 1]);  mul_654 = None
    sum_254: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_731, [0, 1]);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_333: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_330, mul_653);  add_330 = mul_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_255: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_333, memory_format = torch.contiguous_format)
    view_732: "f32[1568, 320]" = torch.ops.aten.view.default(clone_255, [1568, 320]);  clone_255 = None
    permute_654: "f32[320, 320]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
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
    alias_42: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
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
    permute_660: "f32[640, 320]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    mm_148: "f32[392, 320]" = torch.ops.aten.mm.default(view_738, permute_660);  permute_660 = None
    permute_661: "f32[640, 392]" = torch.ops.aten.permute.default(view_738, [1, 0])
    mm_149: "f32[640, 320]" = torch.ops.aten.mm.default(permute_661, view_214);  permute_661 = view_214 = None
    permute_662: "f32[320, 640]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_256: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_738, [0], True);  view_738 = None
    view_739: "f32[640]" = torch.ops.aten.view.default(sum_256, [640]);  sum_256 = None
    permute_663: "f32[640, 320]" = torch.ops.aten.permute.default(permute_662, [1, 0]);  permute_662 = None
    view_740: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_148, [8, 49, 320]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_257: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    sub_212: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_257, getitem_165);  clone_257 = getitem_165 = None
    mul_655: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_212, rsqrt_43);  sub_212 = None
    mul_656: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_740, primals_259);  primals_259 = None
    mul_657: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_656, 320)
    sum_257: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_656, [2], True)
    mul_658: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_656, mul_655);  mul_656 = None
    sum_258: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_658, [2], True);  mul_658 = None
    mul_659: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_655, sum_258);  sum_258 = None
    sub_213: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_657, sum_257);  mul_657 = sum_257 = None
    sub_214: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_213, mul_659);  sub_213 = mul_659 = None
    div_43: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 320);  rsqrt_43 = None
    mul_660: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_43, sub_214);  div_43 = sub_214 = None
    mul_661: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_740, mul_655);  mul_655 = None
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
    permute_667: "f32[320, 320]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
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
    clone_259: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format);  add_125 = None
    sub_215: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_259, getitem_163);  clone_259 = getitem_163 = None
    mul_662: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_215, rsqrt_42);  sub_215 = None
    mul_663: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_258, primals_253);  primals_253 = None
    mul_664: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_663, 320)
    sum_263: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_663, [2], True)
    mul_665: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_663, mul_662);  mul_663 = None
    sum_264: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_665, [2], True);  mul_665 = None
    mul_666: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_662, sum_264);  sum_264 = None
    sub_216: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_664, sum_263);  mul_664 = sum_263 = None
    sub_217: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_216, mul_666);  sub_216 = mul_666 = None
    div_44: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 320);  rsqrt_42 = None
    mul_667: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_44, sub_217);  div_44 = sub_217 = None
    mul_668: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_258, mul_662);  mul_662 = None
    sum_265: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 1]);  mul_668 = None
    sum_266: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_258, [0, 1]);  clone_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_335: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_333, mul_667);  add_333 = mul_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_260: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_335, memory_format = torch.contiguous_format)
    view_747: "f32[1568, 320]" = torch.ops.aten.view.default(clone_260, [1568, 320]);  clone_260 = None
    permute_671: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    mm_152: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_747, permute_671);  permute_671 = None
    permute_672: "f32[320, 1568]" = torch.ops.aten.permute.default(view_747, [1, 0])
    mm_153: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_672, view_207);  permute_672 = view_207 = None
    permute_673: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_267: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_747, [0], True);  view_747 = None
    view_748: "f32[320]" = torch.ops.aten.view.default(sum_267, [320]);  sum_267 = None
    permute_674: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_673, [1, 0]);  permute_673 = None
    view_749: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_152, [8, 196, 1280]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_669: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476)
    erf_43: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_669);  mul_669 = None
    add_336: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
    mul_670: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_336, 0.5);  add_336 = None
    mul_671: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_206, view_206)
    mul_672: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_671, -0.5);  mul_671 = None
    exp_15: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_672);  mul_672 = None
    mul_673: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_674: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_206, mul_673);  view_206 = mul_673 = None
    add_337: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_670, mul_674);  mul_670 = mul_674 = None
    mul_675: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_749, add_337);  view_749 = add_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_750: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_675, [1568, 1280]);  mul_675 = None
    permute_675: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    mm_154: "f32[1568, 320]" = torch.ops.aten.mm.default(view_750, permute_675);  permute_675 = None
    permute_676: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_750, [1, 0])
    mm_155: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_676, view_205);  permute_676 = view_205 = None
    permute_677: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_268: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_750, [0], True);  view_750 = None
    view_751: "f32[1280]" = torch.ops.aten.view.default(sum_268, [1280]);  sum_268 = None
    permute_678: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_677, [1, 0]);  permute_677 = None
    view_752: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_154, [8, 196, 320]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_261: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_121, memory_format = torch.contiguous_format);  add_121 = None
    sub_218: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_261, getitem_161);  clone_261 = getitem_161 = None
    mul_676: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_218, rsqrt_41);  sub_218 = None
    mul_677: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_752, primals_247);  primals_247 = None
    mul_678: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_677, 320)
    sum_269: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_677, [2], True)
    mul_679: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_677, mul_676);  mul_677 = None
    sum_270: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_679, [2], True);  mul_679 = None
    mul_680: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_676, sum_270);  sum_270 = None
    sub_219: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_678, sum_269);  mul_678 = sum_269 = None
    sub_220: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_219, mul_680);  sub_219 = mul_680 = None
    div_45: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 320);  rsqrt_41 = None
    mul_681: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_45, sub_220);  div_45 = sub_220 = None
    mul_682: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_752, mul_676);  mul_676 = None
    sum_271: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_682, [0, 1]);  mul_682 = None
    sum_272: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_752, [0, 1]);  view_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_338: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_335, mul_681);  add_335 = mul_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_262: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_338, memory_format = torch.contiguous_format)
    view_753: "f32[1568, 320]" = torch.ops.aten.view.default(clone_262, [1568, 320]);  clone_262 = None
    permute_679: "f32[320, 320]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
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
    alias_43: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
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
    permute_685: "f32[640, 320]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    mm_158: "f32[392, 320]" = torch.ops.aten.mm.default(view_759, permute_685);  permute_685 = None
    permute_686: "f32[640, 392]" = torch.ops.aten.permute.default(view_759, [1, 0])
    mm_159: "f32[640, 320]" = torch.ops.aten.mm.default(permute_686, view_199);  permute_686 = view_199 = None
    permute_687: "f32[320, 640]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_274: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_759, [0], True);  view_759 = None
    view_760: "f32[640]" = torch.ops.aten.view.default(sum_274, [640]);  sum_274 = None
    permute_688: "f32[640, 320]" = torch.ops.aten.permute.default(permute_687, [1, 0]);  permute_687 = None
    view_761: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_158, [8, 49, 320]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_264: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    sub_221: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_264, getitem_153);  clone_264 = getitem_153 = None
    mul_683: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_221, rsqrt_40);  sub_221 = None
    mul_684: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_761, primals_241);  primals_241 = None
    mul_685: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_684, 320)
    sum_275: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_684, [2], True)
    mul_686: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_684, mul_683);  mul_684 = None
    sum_276: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_686, [2], True);  mul_686 = None
    mul_687: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_683, sum_276);  sum_276 = None
    sub_222: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_685, sum_275);  mul_685 = sum_275 = None
    sub_223: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_222, mul_687);  sub_222 = mul_687 = None
    div_46: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 320);  rsqrt_40 = None
    mul_688: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_46, sub_223);  div_46 = sub_223 = None
    mul_689: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_761, mul_683);  mul_683 = None
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
    permute_692: "f32[320, 320]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
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
    clone_266: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_116, memory_format = torch.contiguous_format);  add_116 = None
    sub_224: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_266, getitem_151);  clone_266 = getitem_151 = None
    mul_690: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_224, rsqrt_39);  sub_224 = None
    mul_691: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_265, primals_235);  primals_235 = None
    mul_692: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_691, 320)
    sum_281: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_691, [2], True)
    mul_693: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_691, mul_690);  mul_691 = None
    sum_282: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_693, [2], True);  mul_693 = None
    mul_694: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_690, sum_282);  sum_282 = None
    sub_225: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_692, sum_281);  mul_692 = sum_281 = None
    sub_226: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_225, mul_694);  sub_225 = mul_694 = None
    div_47: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 320);  rsqrt_39 = None
    mul_695: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_47, sub_226);  div_47 = sub_226 = None
    mul_696: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_265, mul_690);  mul_690 = None
    sum_283: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_696, [0, 1]);  mul_696 = None
    sum_284: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_265, [0, 1]);  clone_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_340: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_338, mul_695);  add_338 = mul_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_267: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_340, memory_format = torch.contiguous_format)
    view_768: "f32[1568, 320]" = torch.ops.aten.view.default(clone_267, [1568, 320]);  clone_267 = None
    permute_696: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    mm_162: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_768, permute_696);  permute_696 = None
    permute_697: "f32[320, 1568]" = torch.ops.aten.permute.default(view_768, [1, 0])
    mm_163: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_697, view_192);  permute_697 = view_192 = None
    permute_698: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_285: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_768, [0], True);  view_768 = None
    view_769: "f32[320]" = torch.ops.aten.view.default(sum_285, [320]);  sum_285 = None
    permute_699: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_698, [1, 0]);  permute_698 = None
    view_770: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_162, [8, 196, 1280]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_697: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_191, 0.7071067811865476)
    erf_44: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_697);  mul_697 = None
    add_341: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
    mul_698: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_341, 0.5);  add_341 = None
    mul_699: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_191, view_191)
    mul_700: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_699, -0.5);  mul_699 = None
    exp_16: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_700);  mul_700 = None
    mul_701: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_702: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_191, mul_701);  view_191 = mul_701 = None
    add_342: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_698, mul_702);  mul_698 = mul_702 = None
    mul_703: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_770, add_342);  view_770 = add_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_771: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_703, [1568, 1280]);  mul_703 = None
    permute_700: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_164: "f32[1568, 320]" = torch.ops.aten.mm.default(view_771, permute_700);  permute_700 = None
    permute_701: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_771, [1, 0])
    mm_165: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_701, view_190);  permute_701 = view_190 = None
    permute_702: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_286: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_771, [0], True);  view_771 = None
    view_772: "f32[1280]" = torch.ops.aten.view.default(sum_286, [1280]);  sum_286 = None
    permute_703: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_702, [1, 0]);  permute_702 = None
    view_773: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_164, [8, 196, 320]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_268: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_112, memory_format = torch.contiguous_format);  add_112 = None
    sub_227: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_268, getitem_149);  clone_268 = getitem_149 = None
    mul_704: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_227, rsqrt_38);  sub_227 = None
    mul_705: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_773, primals_229);  primals_229 = None
    mul_706: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_705, 320)
    sum_287: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_705, [2], True)
    mul_707: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_705, mul_704);  mul_705 = None
    sum_288: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_707, [2], True);  mul_707 = None
    mul_708: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_704, sum_288);  sum_288 = None
    sub_228: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_706, sum_287);  mul_706 = sum_287 = None
    sub_229: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_228, mul_708);  sub_228 = mul_708 = None
    div_48: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 320);  rsqrt_38 = None
    mul_709: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_48, sub_229);  div_48 = sub_229 = None
    mul_710: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_773, mul_704);  mul_704 = None
    sum_289: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_710, [0, 1]);  mul_710 = None
    sum_290: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_773, [0, 1]);  view_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_343: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_340, mul_709);  add_340 = mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_269: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_343, memory_format = torch.contiguous_format)
    view_774: "f32[1568, 320]" = torch.ops.aten.view.default(clone_269, [1568, 320]);  clone_269 = None
    permute_704: "f32[320, 320]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
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
    alias_44: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
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
    permute_710: "f32[640, 320]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    mm_168: "f32[392, 320]" = torch.ops.aten.mm.default(view_780, permute_710);  permute_710 = None
    permute_711: "f32[640, 392]" = torch.ops.aten.permute.default(view_780, [1, 0])
    mm_169: "f32[640, 320]" = torch.ops.aten.mm.default(permute_711, view_184);  permute_711 = view_184 = None
    permute_712: "f32[320, 640]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    sum_292: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_780, [0], True);  view_780 = None
    view_781: "f32[640]" = torch.ops.aten.view.default(sum_292, [640]);  sum_292 = None
    permute_713: "f32[640, 320]" = torch.ops.aten.permute.default(permute_712, [1, 0]);  permute_712 = None
    view_782: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_168, [8, 49, 320]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_271: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    sub_230: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_271, getitem_141);  clone_271 = getitem_141 = None
    mul_711: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_230, rsqrt_37);  sub_230 = None
    mul_712: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_782, primals_223);  primals_223 = None
    mul_713: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_712, 320)
    sum_293: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_712, [2], True)
    mul_714: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_712, mul_711);  mul_712 = None
    sum_294: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_714, [2], True);  mul_714 = None
    mul_715: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_711, sum_294);  sum_294 = None
    sub_231: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_713, sum_293);  mul_713 = sum_293 = None
    sub_232: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_231, mul_715);  sub_231 = mul_715 = None
    div_49: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 320);  rsqrt_37 = None
    mul_716: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_49, sub_232);  div_49 = sub_232 = None
    mul_717: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_782, mul_711);  mul_711 = None
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
    permute_717: "f32[320, 320]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
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
    clone_273: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_107, memory_format = torch.contiguous_format);  add_107 = None
    sub_233: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_273, getitem_139);  clone_273 = getitem_139 = None
    mul_718: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_233, rsqrt_36);  sub_233 = None
    mul_719: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_272, primals_217);  primals_217 = None
    mul_720: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_719, 320)
    sum_299: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_719, [2], True)
    mul_721: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_719, mul_718);  mul_719 = None
    sum_300: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_721, [2], True);  mul_721 = None
    mul_722: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_718, sum_300);  sum_300 = None
    sub_234: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_720, sum_299);  mul_720 = sum_299 = None
    sub_235: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_234, mul_722);  sub_234 = mul_722 = None
    div_50: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 320);  rsqrt_36 = None
    mul_723: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_50, sub_235);  div_50 = sub_235 = None
    mul_724: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_272, mul_718);  mul_718 = None
    sum_301: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_724, [0, 1]);  mul_724 = None
    sum_302: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_272, [0, 1]);  clone_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_345: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_343, mul_723);  add_343 = mul_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_274: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_345, memory_format = torch.contiguous_format)
    view_789: "f32[1568, 320]" = torch.ops.aten.view.default(clone_274, [1568, 320]);  clone_274 = None
    permute_721: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    mm_172: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_789, permute_721);  permute_721 = None
    permute_722: "f32[320, 1568]" = torch.ops.aten.permute.default(view_789, [1, 0])
    mm_173: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_722, view_177);  permute_722 = view_177 = None
    permute_723: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_303: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_789, [0], True);  view_789 = None
    view_790: "f32[320]" = torch.ops.aten.view.default(sum_303, [320]);  sum_303 = None
    permute_724: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_723, [1, 0]);  permute_723 = None
    view_791: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_172, [8, 196, 1280]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_725: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476)
    erf_45: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_725);  mul_725 = None
    add_346: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
    mul_726: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_346, 0.5);  add_346 = None
    mul_727: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_176, view_176)
    mul_728: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_727, -0.5);  mul_727 = None
    exp_17: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_728);  mul_728 = None
    mul_729: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_730: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_176, mul_729);  view_176 = mul_729 = None
    add_347: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_726, mul_730);  mul_726 = mul_730 = None
    mul_731: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_791, add_347);  view_791 = add_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_792: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_731, [1568, 1280]);  mul_731 = None
    permute_725: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    mm_174: "f32[1568, 320]" = torch.ops.aten.mm.default(view_792, permute_725);  permute_725 = None
    permute_726: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_792, [1, 0])
    mm_175: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_726, view_175);  permute_726 = view_175 = None
    permute_727: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_304: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_792, [0], True);  view_792 = None
    view_793: "f32[1280]" = torch.ops.aten.view.default(sum_304, [1280]);  sum_304 = None
    permute_728: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_727, [1, 0]);  permute_727 = None
    view_794: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_174, [8, 196, 320]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_275: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_103, memory_format = torch.contiguous_format);  add_103 = None
    sub_236: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_275, getitem_137);  clone_275 = getitem_137 = None
    mul_732: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_236, rsqrt_35);  sub_236 = None
    mul_733: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_794, primals_211);  primals_211 = None
    mul_734: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_733, 320)
    sum_305: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_733, [2], True)
    mul_735: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_733, mul_732);  mul_733 = None
    sum_306: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_735, [2], True);  mul_735 = None
    mul_736: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_732, sum_306);  sum_306 = None
    sub_237: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_734, sum_305);  mul_734 = sum_305 = None
    sub_238: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_237, mul_736);  sub_237 = mul_736 = None
    div_51: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 320);  rsqrt_35 = None
    mul_737: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_51, sub_238);  div_51 = sub_238 = None
    mul_738: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_794, mul_732);  mul_732 = None
    sum_307: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_738, [0, 1]);  mul_738 = None
    sum_308: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_794, [0, 1]);  view_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_348: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_345, mul_737);  add_345 = mul_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_276: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_348, memory_format = torch.contiguous_format)
    view_795: "f32[1568, 320]" = torch.ops.aten.view.default(clone_276, [1568, 320]);  clone_276 = None
    permute_729: "f32[320, 320]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
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
    alias_45: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
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
    permute_735: "f32[640, 320]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    mm_178: "f32[392, 320]" = torch.ops.aten.mm.default(view_801, permute_735);  permute_735 = None
    permute_736: "f32[640, 392]" = torch.ops.aten.permute.default(view_801, [1, 0])
    mm_179: "f32[640, 320]" = torch.ops.aten.mm.default(permute_736, view_169);  permute_736 = view_169 = None
    permute_737: "f32[320, 640]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    sum_310: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_801, [0], True);  view_801 = None
    view_802: "f32[640]" = torch.ops.aten.view.default(sum_310, [640]);  sum_310 = None
    permute_738: "f32[640, 320]" = torch.ops.aten.permute.default(permute_737, [1, 0]);  permute_737 = None
    view_803: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_178, [8, 49, 320]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_278: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    sub_239: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_278, getitem_129);  clone_278 = getitem_129 = None
    mul_739: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_239, rsqrt_34);  sub_239 = None
    mul_740: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_803, primals_205);  primals_205 = None
    mul_741: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_740, 320)
    sum_311: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_740, [2], True)
    mul_742: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_740, mul_739);  mul_740 = None
    sum_312: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_742, [2], True);  mul_742 = None
    mul_743: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_739, sum_312);  sum_312 = None
    sub_240: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_741, sum_311);  mul_741 = sum_311 = None
    sub_241: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_240, mul_743);  sub_240 = mul_743 = None
    div_52: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 320);  rsqrt_34 = None
    mul_744: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_52, sub_241);  div_52 = sub_241 = None
    mul_745: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_803, mul_739);  mul_739 = None
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
    permute_742: "f32[320, 320]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
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
    clone_280: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_98, memory_format = torch.contiguous_format);  add_98 = None
    sub_242: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_280, getitem_127);  clone_280 = getitem_127 = None
    mul_746: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_242, rsqrt_33);  sub_242 = None
    mul_747: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_279, primals_199);  primals_199 = None
    mul_748: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_747, 320)
    sum_317: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_747, [2], True)
    mul_749: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_747, mul_746);  mul_747 = None
    sum_318: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True);  mul_749 = None
    mul_750: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_746, sum_318);  sum_318 = None
    sub_243: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_748, sum_317);  mul_748 = sum_317 = None
    sub_244: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_243, mul_750);  sub_243 = mul_750 = None
    div_53: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 320);  rsqrt_33 = None
    mul_751: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_53, sub_244);  div_53 = sub_244 = None
    mul_752: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_279, mul_746);  mul_746 = None
    sum_319: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 1]);  mul_752 = None
    sum_320: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_279, [0, 1]);  clone_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_350: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_348, mul_751);  add_348 = mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_281: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_350, memory_format = torch.contiguous_format)
    view_810: "f32[1568, 320]" = torch.ops.aten.view.default(clone_281, [1568, 320]);  clone_281 = None
    permute_746: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    mm_182: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_810, permute_746);  permute_746 = None
    permute_747: "f32[320, 1568]" = torch.ops.aten.permute.default(view_810, [1, 0])
    mm_183: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_747, view_162);  permute_747 = view_162 = None
    permute_748: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_321: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_810, [0], True);  view_810 = None
    view_811: "f32[320]" = torch.ops.aten.view.default(sum_321, [320]);  sum_321 = None
    permute_749: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_748, [1, 0]);  permute_748 = None
    view_812: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_182, [8, 196, 1280]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_753: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_161, 0.7071067811865476)
    erf_46: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_753);  mul_753 = None
    add_351: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
    mul_754: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_351, 0.5);  add_351 = None
    mul_755: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_161, view_161)
    mul_756: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_755, -0.5);  mul_755 = None
    exp_18: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_756);  mul_756 = None
    mul_757: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_758: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_161, mul_757);  view_161 = mul_757 = None
    add_352: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_754, mul_758);  mul_754 = mul_758 = None
    mul_759: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_812, add_352);  view_812 = add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_813: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_759, [1568, 1280]);  mul_759 = None
    permute_750: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_184: "f32[1568, 320]" = torch.ops.aten.mm.default(view_813, permute_750);  permute_750 = None
    permute_751: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_813, [1, 0])
    mm_185: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_751, view_160);  permute_751 = view_160 = None
    permute_752: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    sum_322: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_813, [0], True);  view_813 = None
    view_814: "f32[1280]" = torch.ops.aten.view.default(sum_322, [1280]);  sum_322 = None
    permute_753: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_752, [1, 0]);  permute_752 = None
    view_815: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_184, [8, 196, 320]);  mm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_282: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format);  add_94 = None
    sub_245: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_282, getitem_125);  clone_282 = getitem_125 = None
    mul_760: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_245, rsqrt_32);  sub_245 = None
    mul_761: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_815, primals_193);  primals_193 = None
    mul_762: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_761, 320)
    sum_323: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_761, [2], True)
    mul_763: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_761, mul_760);  mul_761 = None
    sum_324: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_763, [2], True);  mul_763 = None
    mul_764: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_760, sum_324);  sum_324 = None
    sub_246: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_762, sum_323);  mul_762 = sum_323 = None
    sub_247: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_246, mul_764);  sub_246 = mul_764 = None
    div_54: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 320);  rsqrt_32 = None
    mul_765: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_54, sub_247);  div_54 = sub_247 = None
    mul_766: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_815, mul_760);  mul_760 = None
    sum_325: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_766, [0, 1]);  mul_766 = None
    sum_326: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_815, [0, 1]);  view_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_353: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_350, mul_765);  add_350 = mul_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_283: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_353, memory_format = torch.contiguous_format)
    view_816: "f32[1568, 320]" = torch.ops.aten.view.default(clone_283, [1568, 320]);  clone_283 = None
    permute_754: "f32[320, 320]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
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
    alias_46: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
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
    permute_760: "f32[640, 320]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_188: "f32[392, 320]" = torch.ops.aten.mm.default(view_822, permute_760);  permute_760 = None
    permute_761: "f32[640, 392]" = torch.ops.aten.permute.default(view_822, [1, 0])
    mm_189: "f32[640, 320]" = torch.ops.aten.mm.default(permute_761, view_154);  permute_761 = view_154 = None
    permute_762: "f32[320, 640]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_328: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_822, [0], True);  view_822 = None
    view_823: "f32[640]" = torch.ops.aten.view.default(sum_328, [640]);  sum_328 = None
    permute_763: "f32[640, 320]" = torch.ops.aten.permute.default(permute_762, [1, 0]);  permute_762 = None
    view_824: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_188, [8, 49, 320]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_285: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    sub_248: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_285, getitem_117);  clone_285 = getitem_117 = None
    mul_767: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_248, rsqrt_31);  sub_248 = None
    mul_768: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_824, primals_187);  primals_187 = None
    mul_769: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_768, 320)
    sum_329: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_768, [2], True)
    mul_770: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_768, mul_767);  mul_768 = None
    sum_330: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_770, [2], True);  mul_770 = None
    mul_771: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_767, sum_330);  sum_330 = None
    sub_249: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_769, sum_329);  mul_769 = sum_329 = None
    sub_250: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_249, mul_771);  sub_249 = mul_771 = None
    div_55: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 320);  rsqrt_31 = None
    mul_772: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_55, sub_250);  div_55 = sub_250 = None
    mul_773: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_824, mul_767);  mul_767 = None
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
    permute_767: "f32[320, 320]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
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
    clone_287: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_89, memory_format = torch.contiguous_format);  add_89 = None
    sub_251: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_287, getitem_115);  clone_287 = getitem_115 = None
    mul_774: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_251, rsqrt_30);  sub_251 = None
    mul_775: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_286, primals_181);  primals_181 = None
    mul_776: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_775, 320)
    sum_335: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_775, [2], True)
    mul_777: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_775, mul_774);  mul_775 = None
    sum_336: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_777, [2], True);  mul_777 = None
    mul_778: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_774, sum_336);  sum_336 = None
    sub_252: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_776, sum_335);  mul_776 = sum_335 = None
    sub_253: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_252, mul_778);  sub_252 = mul_778 = None
    div_56: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 320);  rsqrt_30 = None
    mul_779: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_56, sub_253);  div_56 = sub_253 = None
    mul_780: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_286, mul_774);  mul_774 = None
    sum_337: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 1]);  mul_780 = None
    sum_338: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_286, [0, 1]);  clone_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_355: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_353, mul_779);  add_353 = mul_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_288: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_355, memory_format = torch.contiguous_format)
    view_831: "f32[1568, 320]" = torch.ops.aten.view.default(clone_288, [1568, 320]);  clone_288 = None
    permute_771: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    mm_192: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_831, permute_771);  permute_771 = None
    permute_772: "f32[320, 1568]" = torch.ops.aten.permute.default(view_831, [1, 0])
    mm_193: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_772, view_147);  permute_772 = view_147 = None
    permute_773: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_339: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_831, [0], True);  view_831 = None
    view_832: "f32[320]" = torch.ops.aten.view.default(sum_339, [320]);  sum_339 = None
    permute_774: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_773, [1, 0]);  permute_773 = None
    view_833: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_192, [8, 196, 1280]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_781: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476)
    erf_47: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_781);  mul_781 = None
    add_356: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
    mul_782: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_356, 0.5);  add_356 = None
    mul_783: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_146, view_146)
    mul_784: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_783, -0.5);  mul_783 = None
    exp_19: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_784);  mul_784 = None
    mul_785: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_786: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_146, mul_785);  view_146 = mul_785 = None
    add_357: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_782, mul_786);  mul_782 = mul_786 = None
    mul_787: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_833, add_357);  view_833 = add_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_834: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_787, [1568, 1280]);  mul_787 = None
    permute_775: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    mm_194: "f32[1568, 320]" = torch.ops.aten.mm.default(view_834, permute_775);  permute_775 = None
    permute_776: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_834, [1, 0])
    mm_195: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_776, view_145);  permute_776 = view_145 = None
    permute_777: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_340: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_834, [0], True);  view_834 = None
    view_835: "f32[1280]" = torch.ops.aten.view.default(sum_340, [1280]);  sum_340 = None
    permute_778: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_777, [1, 0]);  permute_777 = None
    view_836: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_194, [8, 196, 320]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_289: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_85, memory_format = torch.contiguous_format);  add_85 = None
    sub_254: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_289, getitem_113);  clone_289 = getitem_113 = None
    mul_788: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_254, rsqrt_29);  sub_254 = None
    mul_789: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_836, primals_175);  primals_175 = None
    mul_790: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_789, 320)
    sum_341: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_789, [2], True)
    mul_791: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_789, mul_788);  mul_789 = None
    sum_342: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_791, [2], True);  mul_791 = None
    mul_792: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_788, sum_342);  sum_342 = None
    sub_255: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_790, sum_341);  mul_790 = sum_341 = None
    sub_256: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_255, mul_792);  sub_255 = mul_792 = None
    div_57: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 320);  rsqrt_29 = None
    mul_793: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_57, sub_256);  div_57 = sub_256 = None
    mul_794: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_836, mul_788);  mul_788 = None
    sum_343: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_794, [0, 1]);  mul_794 = None
    sum_344: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_836, [0, 1]);  view_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_358: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_355, mul_793);  add_355 = mul_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_290: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_358, memory_format = torch.contiguous_format)
    view_837: "f32[1568, 320]" = torch.ops.aten.view.default(clone_290, [1568, 320]);  clone_290 = None
    permute_779: "f32[320, 320]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
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
    alias_47: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
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
    permute_785: "f32[640, 320]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_198: "f32[392, 320]" = torch.ops.aten.mm.default(view_843, permute_785);  permute_785 = None
    permute_786: "f32[640, 392]" = torch.ops.aten.permute.default(view_843, [1, 0])
    mm_199: "f32[640, 320]" = torch.ops.aten.mm.default(permute_786, view_139);  permute_786 = view_139 = None
    permute_787: "f32[320, 640]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_346: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_843, [0], True);  view_843 = None
    view_844: "f32[640]" = torch.ops.aten.view.default(sum_346, [640]);  sum_346 = None
    permute_788: "f32[640, 320]" = torch.ops.aten.permute.default(permute_787, [1, 0]);  permute_787 = None
    view_845: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_198, [8, 49, 320]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_292: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    sub_257: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_292, getitem_105);  clone_292 = getitem_105 = None
    mul_795: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_257, rsqrt_28);  sub_257 = None
    mul_796: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_845, primals_169);  primals_169 = None
    mul_797: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_796, 320)
    sum_347: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_796, [2], True)
    mul_798: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_796, mul_795);  mul_796 = None
    sum_348: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_798, [2], True);  mul_798 = None
    mul_799: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_795, sum_348);  sum_348 = None
    sub_258: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_797, sum_347);  mul_797 = sum_347 = None
    sub_259: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_258, mul_799);  sub_258 = mul_799 = None
    div_58: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 320);  rsqrt_28 = None
    mul_800: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_58, sub_259);  div_58 = sub_259 = None
    mul_801: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_845, mul_795);  mul_795 = None
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
    permute_792: "f32[320, 320]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
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
    clone_294: "f32[8, 196, 320]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    sub_260: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_294, getitem_103);  clone_294 = getitem_103 = None
    mul_802: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_260, rsqrt_27);  sub_260 = None
    mul_803: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_293, primals_163);  primals_163 = None
    mul_804: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_803, 320)
    sum_353: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_803, [2], True)
    mul_805: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_803, mul_802);  mul_803 = None
    sum_354: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_805, [2], True);  mul_805 = None
    mul_806: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_802, sum_354);  sum_354 = None
    sub_261: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_804, sum_353);  mul_804 = sum_353 = None
    sub_262: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_261, mul_806);  sub_261 = mul_806 = None
    div_59: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 320);  rsqrt_27 = None
    mul_807: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_59, sub_262);  div_59 = sub_262 = None
    mul_808: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_293, mul_802);  mul_802 = None
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
    permute_798: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    mm_202: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_854, permute_798);  permute_798 = None
    permute_799: "f32[320, 1568]" = torch.ops.aten.permute.default(view_854, [1, 0])
    mm_203: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_799, view_129);  permute_799 = view_129 = None
    permute_800: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    sum_358: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_854, [0], True);  view_854 = None
    view_855: "f32[320]" = torch.ops.aten.view.default(sum_358, [320]);  sum_358 = None
    permute_801: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_800, [1, 0]);  permute_800 = None
    view_856: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_202, [8, 196, 1280]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_809: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476)
    erf_48: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_809);  mul_809 = None
    add_362: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
    mul_810: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_362, 0.5);  add_362 = None
    mul_811: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_128, view_128)
    mul_812: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_811, -0.5);  mul_811 = None
    exp_20: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_812);  mul_812 = None
    mul_813: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_814: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_128, mul_813);  view_128 = mul_813 = None
    add_363: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_810, mul_814);  mul_810 = mul_814 = None
    mul_815: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_856, add_363);  view_856 = add_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_857: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_815, [1568, 1280]);  mul_815 = None
    permute_802: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_204: "f32[1568, 320]" = torch.ops.aten.mm.default(view_857, permute_802);  permute_802 = None
    permute_803: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_857, [1, 0])
    mm_205: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_803, view_127);  permute_803 = view_127 = None
    permute_804: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_359: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_857, [0], True);  view_857 = None
    view_858: "f32[1280]" = torch.ops.aten.view.default(sum_359, [1280]);  sum_359 = None
    permute_805: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_804, [1, 0]);  permute_804 = None
    view_859: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_204, [8, 196, 320]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    sub_263: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(add_75, getitem_101);  add_75 = getitem_101 = None
    mul_816: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_263, rsqrt_26);  sub_263 = None
    mul_817: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_859, primals_155);  primals_155 = None
    mul_818: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_817, 320)
    sum_360: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_817, [2], True)
    mul_819: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_817, mul_816);  mul_817 = None
    sum_361: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_819, [2], True);  mul_819 = None
    mul_820: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_816, sum_361);  sum_361 = None
    sub_264: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_818, sum_360);  mul_818 = sum_360 = None
    sub_265: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_264, mul_820);  sub_264 = mul_820 = None
    div_60: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 320);  rsqrt_26 = None
    mul_821: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_60, sub_265);  div_60 = sub_265 = None
    mul_822: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_859, mul_816);  mul_816 = None
    sum_362: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_822, [0, 1]);  mul_822 = None
    sum_363: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_859, [0, 1]);  view_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_364: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_797, mul_821);  permute_797 = mul_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_296: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_364, memory_format = torch.contiguous_format)
    view_860: "f32[1568, 320]" = torch.ops.aten.view.default(clone_296, [1568, 320]);  clone_296 = None
    permute_806: "f32[320, 320]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
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
    alias_48: "f32[8, 5, 196, 64]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
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
    permute_812: "f32[640, 320]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    mm_208: "f32[392, 320]" = torch.ops.aten.mm.default(view_866, permute_812);  permute_812 = None
    permute_813: "f32[640, 392]" = torch.ops.aten.permute.default(view_866, [1, 0])
    mm_209: "f32[640, 320]" = torch.ops.aten.mm.default(permute_813, view_121);  permute_813 = view_121 = None
    permute_814: "f32[320, 640]" = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
    sum_365: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_866, [0], True);  view_866 = None
    view_867: "f32[640]" = torch.ops.aten.view.default(sum_365, [640]);  sum_365 = None
    permute_815: "f32[640, 320]" = torch.ops.aten.permute.default(permute_814, [1, 0]);  permute_814 = None
    view_868: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_208, [8, 49, 320]);  mm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_298: "f32[8, 49, 320]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    sub_266: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(clone_298, getitem_93);  clone_298 = getitem_93 = None
    mul_823: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(sub_266, rsqrt_25);  sub_266 = None
    mul_824: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_868, primals_149);  primals_149 = None
    mul_825: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_824, 320)
    sum_366: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_824, [2], True)
    mul_826: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_824, mul_823);  mul_824 = None
    sum_367: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_826, [2], True);  mul_826 = None
    mul_827: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_823, sum_367);  sum_367 = None
    sub_267: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_825, sum_366);  mul_825 = sum_366 = None
    sub_268: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_267, mul_827);  sub_267 = mul_827 = None
    div_61: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 320);  rsqrt_25 = None
    mul_828: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_61, sub_268);  div_61 = sub_268 = None
    mul_829: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_868, mul_823);  mul_823 = None
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
    permute_819: "f32[320, 320]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
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
    sub_269: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_43, getitem_91);  clone_43 = getitem_91 = None
    mul_830: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_269, rsqrt_24);  sub_269 = None
    mul_831: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_299, primals_143);  primals_143 = None
    mul_832: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_831, 320)
    sum_372: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_831, [2], True)
    mul_833: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_831, mul_830);  mul_831 = None
    sum_373: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_833, [2], True);  mul_833 = None
    mul_834: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_830, sum_373);  sum_373 = None
    sub_270: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_832, sum_372);  mul_832 = sum_372 = None
    sub_271: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_270, mul_834);  sub_270 = mul_834 = None
    div_62: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 320);  rsqrt_24 = None
    mul_835: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_62, sub_271);  div_62 = sub_271 = None
    mul_836: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_299, mul_830);  mul_830 = None
    sum_374: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 1]);  mul_836 = None
    sum_375: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_299, [0, 1]);  clone_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_366: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_364, mul_835);  add_364 = mul_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone_300: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_366, memory_format = torch.contiguous_format);  add_366 = None
    clone_301: "f32[8, 196, 320]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    sub_272: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_301, getitem_89);  clone_301 = getitem_89 = None
    mul_837: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_272, rsqrt_23);  sub_272 = None
    mul_838: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_300, primals_141);  primals_141 = None
    mul_839: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_838, 320)
    sum_376: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_838, [2], True)
    mul_840: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_838, mul_837);  mul_838 = None
    sum_377: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_840, [2], True);  mul_840 = None
    mul_841: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_837, sum_377);  sum_377 = None
    sub_273: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_839, sum_376);  mul_839 = sum_376 = None
    sub_274: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_273, mul_841);  sub_273 = mul_841 = None
    div_63: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 320);  rsqrt_23 = None
    mul_842: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_63, sub_274);  div_63 = sub_274 = None
    mul_843: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_300, mul_837);  mul_837 = None
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
    permute_825: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_212: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_877, permute_825);  permute_825 = None
    permute_826: "f32[128, 6272]" = torch.ops.aten.permute.default(view_877, [1, 0])
    mm_213: "f32[128, 1024]" = torch.ops.aten.mm.default(permute_826, view_112);  permute_826 = view_112 = None
    permute_827: "f32[1024, 128]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    sum_381: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
    view_878: "f32[128]" = torch.ops.aten.view.default(sum_381, [128]);  sum_381 = None
    permute_828: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_827, [1, 0]);  permute_827 = None
    view_879: "f32[8, 784, 1024]" = torch.ops.aten.view.default(mm_212, [8, 784, 1024]);  mm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_844: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476)
    erf_49: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_844);  mul_844 = None
    add_367: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
    mul_845: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(add_367, 0.5);  add_367 = None
    mul_846: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_111, view_111)
    mul_847: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_846, -0.5);  mul_846 = None
    exp_21: "f32[8, 784, 1024]" = torch.ops.aten.exp.default(mul_847);  mul_847 = None
    mul_848: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_849: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_111, mul_848);  view_111 = mul_848 = None
    add_368: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(mul_845, mul_849);  mul_845 = mul_849 = None
    mul_850: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_879, add_368);  view_879 = add_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_880: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_850, [6272, 1024]);  mul_850 = None
    permute_829: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_214: "f32[6272, 128]" = torch.ops.aten.mm.default(view_880, permute_829);  permute_829 = None
    permute_830: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_880, [1, 0])
    mm_215: "f32[1024, 128]" = torch.ops.aten.mm.default(permute_830, view_110);  permute_830 = view_110 = None
    permute_831: "f32[128, 1024]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    sum_382: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
    view_881: "f32[1024]" = torch.ops.aten.view.default(sum_382, [1024]);  sum_382 = None
    permute_832: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_831, [1, 0]);  permute_831 = None
    view_882: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_214, [8, 784, 128]);  mm_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_303: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_64, memory_format = torch.contiguous_format);  add_64 = None
    sub_275: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_303, getitem_87);  clone_303 = getitem_87 = None
    mul_851: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_275, rsqrt_22);  sub_275 = None
    mul_852: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_882, primals_133);  primals_133 = None
    mul_853: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_852, 128)
    sum_383: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_852, [2], True)
    mul_854: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_852, mul_851);  mul_852 = None
    sum_384: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_854, [2], True);  mul_854 = None
    mul_855: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_851, sum_384);  sum_384 = None
    sub_276: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_853, sum_383);  mul_853 = sum_383 = None
    sub_277: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_276, mul_855);  sub_276 = mul_855 = None
    div_64: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 128);  rsqrt_22 = None
    mul_856: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_64, sub_277);  div_64 = sub_277 = None
    mul_857: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_882, mul_851);  mul_851 = None
    sum_385: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_857, [0, 1]);  mul_857 = None
    sum_386: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_882, [0, 1]);  view_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_369: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(view_876, mul_856);  view_876 = mul_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_304: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_369, memory_format = torch.contiguous_format)
    view_883: "f32[6272, 128]" = torch.ops.aten.view.default(clone_304, [6272, 128]);  clone_304 = None
    permute_833: "f32[128, 128]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
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
    alias_49: "f32[8, 2, 784, 64]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
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
    permute_839: "f32[256, 128]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    mm_218: "f32[392, 128]" = torch.ops.aten.mm.default(view_889, permute_839);  permute_839 = None
    permute_840: "f32[256, 392]" = torch.ops.aten.permute.default(view_889, [1, 0])
    mm_219: "f32[256, 128]" = torch.ops.aten.mm.default(permute_840, view_104);  permute_840 = view_104 = None
    permute_841: "f32[128, 256]" = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
    sum_388: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_889, [0], True);  view_889 = None
    view_890: "f32[256]" = torch.ops.aten.view.default(sum_388, [256]);  sum_388 = None
    permute_842: "f32[256, 128]" = torch.ops.aten.permute.default(permute_841, [1, 0]);  permute_841 = None
    view_891: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_218, [8, 49, 128]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_306: "f32[8, 49, 128]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    sub_278: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(clone_306, getitem_79);  clone_306 = getitem_79 = None
    mul_858: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_278, rsqrt_21);  sub_278 = None
    mul_859: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_891, primals_127);  primals_127 = None
    mul_860: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_859, 128)
    sum_389: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_859, [2], True)
    mul_861: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_859, mul_858);  mul_859 = None
    sum_390: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_861, [2], True);  mul_861 = None
    mul_862: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_858, sum_390);  sum_390 = None
    sub_279: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(mul_860, sum_389);  mul_860 = sum_389 = None
    sub_280: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(sub_279, mul_862);  sub_279 = mul_862 = None
    div_65: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 128);  rsqrt_21 = None
    mul_863: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(div_65, sub_280);  div_65 = sub_280 = None
    mul_864: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_891, mul_858);  mul_858 = None
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
    permute_846: "f32[128, 128]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
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
    clone_308: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_59, memory_format = torch.contiguous_format);  add_59 = None
    sub_281: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_308, getitem_77);  clone_308 = getitem_77 = None
    mul_865: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_281, rsqrt_20);  sub_281 = None
    mul_866: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_307, primals_121);  primals_121 = None
    mul_867: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_866, 128)
    sum_395: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_866, [2], True)
    mul_868: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_866, mul_865);  mul_866 = None
    sum_396: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_868, [2], True);  mul_868 = None
    mul_869: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_865, sum_396);  sum_396 = None
    sub_282: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_867, sum_395);  mul_867 = sum_395 = None
    sub_283: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_282, mul_869);  sub_282 = mul_869 = None
    div_66: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 128);  rsqrt_20 = None
    mul_870: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_66, sub_283);  div_66 = sub_283 = None
    mul_871: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_307, mul_865);  mul_865 = None
    sum_397: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_871, [0, 1]);  mul_871 = None
    sum_398: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_307, [0, 1]);  clone_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_371: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_369, mul_870);  add_369 = mul_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_309: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_371, memory_format = torch.contiguous_format)
    view_898: "f32[6272, 128]" = torch.ops.aten.view.default(clone_309, [6272, 128]);  clone_309 = None
    permute_850: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    mm_222: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_898, permute_850);  permute_850 = None
    permute_851: "f32[128, 6272]" = torch.ops.aten.permute.default(view_898, [1, 0])
    mm_223: "f32[128, 1024]" = torch.ops.aten.mm.default(permute_851, view_97);  permute_851 = view_97 = None
    permute_852: "f32[1024, 128]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    sum_399: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_898, [0], True);  view_898 = None
    view_899: "f32[128]" = torch.ops.aten.view.default(sum_399, [128]);  sum_399 = None
    permute_853: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_852, [1, 0]);  permute_852 = None
    view_900: "f32[8, 784, 1024]" = torch.ops.aten.view.default(mm_222, [8, 784, 1024]);  mm_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_872: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_96, 0.7071067811865476)
    erf_50: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_872);  mul_872 = None
    add_372: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
    mul_873: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(add_372, 0.5);  add_372 = None
    mul_874: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_96, view_96)
    mul_875: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_874, -0.5);  mul_874 = None
    exp_22: "f32[8, 784, 1024]" = torch.ops.aten.exp.default(mul_875);  mul_875 = None
    mul_876: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_877: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_96, mul_876);  view_96 = mul_876 = None
    add_373: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(mul_873, mul_877);  mul_873 = mul_877 = None
    mul_878: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_900, add_373);  view_900 = add_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_901: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_878, [6272, 1024]);  mul_878 = None
    permute_854: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_224: "f32[6272, 128]" = torch.ops.aten.mm.default(view_901, permute_854);  permute_854 = None
    permute_855: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_901, [1, 0])
    mm_225: "f32[1024, 128]" = torch.ops.aten.mm.default(permute_855, view_95);  permute_855 = view_95 = None
    permute_856: "f32[128, 1024]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    sum_400: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_901, [0], True);  view_901 = None
    view_902: "f32[1024]" = torch.ops.aten.view.default(sum_400, [1024]);  sum_400 = None
    permute_857: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_856, [1, 0]);  permute_856 = None
    view_903: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_224, [8, 784, 128]);  mm_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_310: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_55, memory_format = torch.contiguous_format);  add_55 = None
    sub_284: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_310, getitem_75);  clone_310 = getitem_75 = None
    mul_879: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_284, rsqrt_19);  sub_284 = None
    mul_880: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_903, primals_115);  primals_115 = None
    mul_881: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_880, 128)
    sum_401: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_880, [2], True)
    mul_882: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_880, mul_879);  mul_880 = None
    sum_402: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_882, [2], True);  mul_882 = None
    mul_883: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_879, sum_402);  sum_402 = None
    sub_285: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_881, sum_401);  mul_881 = sum_401 = None
    sub_286: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_285, mul_883);  sub_285 = mul_883 = None
    div_67: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 128);  rsqrt_19 = None
    mul_884: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_67, sub_286);  div_67 = sub_286 = None
    mul_885: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_903, mul_879);  mul_879 = None
    sum_403: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_885, [0, 1]);  mul_885 = None
    sum_404: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_903, [0, 1]);  view_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_374: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_371, mul_884);  add_371 = mul_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_311: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_374, memory_format = torch.contiguous_format)
    view_904: "f32[6272, 128]" = torch.ops.aten.view.default(clone_311, [6272, 128]);  clone_311 = None
    permute_858: "f32[128, 128]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
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
    alias_50: "f32[8, 2, 784, 64]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
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
    permute_864: "f32[256, 128]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_228: "f32[392, 128]" = torch.ops.aten.mm.default(view_910, permute_864);  permute_864 = None
    permute_865: "f32[256, 392]" = torch.ops.aten.permute.default(view_910, [1, 0])
    mm_229: "f32[256, 128]" = torch.ops.aten.mm.default(permute_865, view_89);  permute_865 = view_89 = None
    permute_866: "f32[128, 256]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    sum_406: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_910, [0], True);  view_910 = None
    view_911: "f32[256]" = torch.ops.aten.view.default(sum_406, [256]);  sum_406 = None
    permute_867: "f32[256, 128]" = torch.ops.aten.permute.default(permute_866, [1, 0]);  permute_866 = None
    view_912: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_228, [8, 49, 128]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_313: "f32[8, 49, 128]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    sub_287: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(clone_313, getitem_67);  clone_313 = getitem_67 = None
    mul_886: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_287, rsqrt_18);  sub_287 = None
    mul_887: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_912, primals_109);  primals_109 = None
    mul_888: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_887, 128)
    sum_407: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_887, [2], True)
    mul_889: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_887, mul_886);  mul_887 = None
    sum_408: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_889, [2], True);  mul_889 = None
    mul_890: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_886, sum_408);  sum_408 = None
    sub_288: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(mul_888, sum_407);  mul_888 = sum_407 = None
    sub_289: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(sub_288, mul_890);  sub_288 = mul_890 = None
    div_68: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 128);  rsqrt_18 = None
    mul_891: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(div_68, sub_289);  div_68 = sub_289 = None
    mul_892: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_912, mul_886);  mul_886 = None
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
    permute_871: "f32[128, 128]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
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
    clone_315: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format);  add_50 = None
    sub_290: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_315, getitem_65);  clone_315 = getitem_65 = None
    mul_893: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_290, rsqrt_17);  sub_290 = None
    mul_894: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_314, primals_103);  primals_103 = None
    mul_895: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_894, 128)
    sum_413: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_894, [2], True)
    mul_896: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_894, mul_893);  mul_894 = None
    sum_414: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_896, [2], True);  mul_896 = None
    mul_897: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_893, sum_414);  sum_414 = None
    sub_291: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_895, sum_413);  mul_895 = sum_413 = None
    sub_292: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_291, mul_897);  sub_291 = mul_897 = None
    div_69: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 128);  rsqrt_17 = None
    mul_898: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_69, sub_292);  div_69 = sub_292 = None
    mul_899: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_314, mul_893);  mul_893 = None
    sum_415: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_899, [0, 1]);  mul_899 = None
    sum_416: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_314, [0, 1]);  clone_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_376: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_374, mul_898);  add_374 = mul_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_316: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_376, memory_format = torch.contiguous_format)
    view_919: "f32[6272, 128]" = torch.ops.aten.view.default(clone_316, [6272, 128]);  clone_316 = None
    permute_875: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_232: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_919, permute_875);  permute_875 = None
    permute_876: "f32[128, 6272]" = torch.ops.aten.permute.default(view_919, [1, 0])
    mm_233: "f32[128, 1024]" = torch.ops.aten.mm.default(permute_876, view_82);  permute_876 = view_82 = None
    permute_877: "f32[1024, 128]" = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
    sum_417: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_919, [0], True);  view_919 = None
    view_920: "f32[128]" = torch.ops.aten.view.default(sum_417, [128]);  sum_417 = None
    permute_878: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_877, [1, 0]);  permute_877 = None
    view_921: "f32[8, 784, 1024]" = torch.ops.aten.view.default(mm_232, [8, 784, 1024]);  mm_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_900: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_81, 0.7071067811865476)
    erf_51: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_900);  mul_900 = None
    add_377: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
    mul_901: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(add_377, 0.5);  add_377 = None
    mul_902: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_81, view_81)
    mul_903: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_902, -0.5);  mul_902 = None
    exp_23: "f32[8, 784, 1024]" = torch.ops.aten.exp.default(mul_903);  mul_903 = None
    mul_904: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_905: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_81, mul_904);  view_81 = mul_904 = None
    add_378: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(mul_901, mul_905);  mul_901 = mul_905 = None
    mul_906: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_921, add_378);  view_921 = add_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_922: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_906, [6272, 1024]);  mul_906 = None
    permute_879: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_234: "f32[6272, 128]" = torch.ops.aten.mm.default(view_922, permute_879);  permute_879 = None
    permute_880: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_922, [1, 0])
    mm_235: "f32[1024, 128]" = torch.ops.aten.mm.default(permute_880, view_80);  permute_880 = view_80 = None
    permute_881: "f32[128, 1024]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    sum_418: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_922, [0], True);  view_922 = None
    view_923: "f32[1024]" = torch.ops.aten.view.default(sum_418, [1024]);  sum_418 = None
    permute_882: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_881, [1, 0]);  permute_881 = None
    view_924: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_234, [8, 784, 128]);  mm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_317: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_46, memory_format = torch.contiguous_format);  add_46 = None
    sub_293: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_317, getitem_63);  clone_317 = getitem_63 = None
    mul_907: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_293, rsqrt_16);  sub_293 = None
    mul_908: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_924, primals_97);  primals_97 = None
    mul_909: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_908, 128)
    sum_419: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_908, [2], True)
    mul_910: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_908, mul_907);  mul_908 = None
    sum_420: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_910, [2], True);  mul_910 = None
    mul_911: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_907, sum_420);  sum_420 = None
    sub_294: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_909, sum_419);  mul_909 = sum_419 = None
    sub_295: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_294, mul_911);  sub_294 = mul_911 = None
    div_70: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 128);  rsqrt_16 = None
    mul_912: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_70, sub_295);  div_70 = sub_295 = None
    mul_913: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_924, mul_907);  mul_907 = None
    sum_421: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_913, [0, 1]);  mul_913 = None
    sum_422: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_924, [0, 1]);  view_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_379: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_376, mul_912);  add_376 = mul_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_318: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_379, memory_format = torch.contiguous_format)
    view_925: "f32[6272, 128]" = torch.ops.aten.view.default(clone_318, [6272, 128]);  clone_318 = None
    permute_883: "f32[128, 128]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
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
    alias_51: "f32[8, 2, 784, 64]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
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
    permute_889: "f32[256, 128]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_238: "f32[392, 128]" = torch.ops.aten.mm.default(view_931, permute_889);  permute_889 = None
    permute_890: "f32[256, 392]" = torch.ops.aten.permute.default(view_931, [1, 0])
    mm_239: "f32[256, 128]" = torch.ops.aten.mm.default(permute_890, view_74);  permute_890 = view_74 = None
    permute_891: "f32[128, 256]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    sum_424: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_931, [0], True);  view_931 = None
    view_932: "f32[256]" = torch.ops.aten.view.default(sum_424, [256]);  sum_424 = None
    permute_892: "f32[256, 128]" = torch.ops.aten.permute.default(permute_891, [1, 0]);  permute_891 = None
    view_933: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_238, [8, 49, 128]);  mm_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_320: "f32[8, 49, 128]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    sub_296: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(clone_320, getitem_55);  clone_320 = getitem_55 = None
    mul_914: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_296, rsqrt_15);  sub_296 = None
    mul_915: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_933, primals_91);  primals_91 = None
    mul_916: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_915, 128)
    sum_425: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_915, [2], True)
    mul_917: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_915, mul_914);  mul_915 = None
    sum_426: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_917, [2], True);  mul_917 = None
    mul_918: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_914, sum_426);  sum_426 = None
    sub_297: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(mul_916, sum_425);  mul_916 = sum_425 = None
    sub_298: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(sub_297, mul_918);  sub_297 = mul_918 = None
    div_71: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 128);  rsqrt_15 = None
    mul_919: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(div_71, sub_298);  div_71 = sub_298 = None
    mul_920: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_933, mul_914);  mul_914 = None
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
    permute_896: "f32[128, 128]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
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
    clone_322: "f32[8, 784, 128]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    sub_299: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_322, getitem_53);  clone_322 = getitem_53 = None
    mul_921: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_299, rsqrt_14);  sub_299 = None
    mul_922: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_321, primals_85);  primals_85 = None
    mul_923: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_922, 128)
    sum_431: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_922, [2], True)
    mul_924: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_922, mul_921);  mul_922 = None
    sum_432: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_924, [2], True);  mul_924 = None
    mul_925: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_921, sum_432);  sum_432 = None
    sub_300: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_923, sum_431);  mul_923 = sum_431 = None
    sub_301: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_300, mul_925);  sub_300 = mul_925 = None
    div_72: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 128);  rsqrt_14 = None
    mul_926: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_72, sub_301);  div_72 = sub_301 = None
    mul_927: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_321, mul_921);  mul_921 = None
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
    permute_902: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_242: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_942, permute_902);  permute_902 = None
    permute_903: "f32[128, 6272]" = torch.ops.aten.permute.default(view_942, [1, 0])
    mm_243: "f32[128, 1024]" = torch.ops.aten.mm.default(permute_903, view_64);  permute_903 = view_64 = None
    permute_904: "f32[1024, 128]" = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
    sum_436: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_942, [0], True);  view_942 = None
    view_943: "f32[128]" = torch.ops.aten.view.default(sum_436, [128]);  sum_436 = None
    permute_905: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_904, [1, 0]);  permute_904 = None
    view_944: "f32[8, 784, 1024]" = torch.ops.aten.view.default(mm_242, [8, 784, 1024]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_928: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_52: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_928);  mul_928 = None
    add_383: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
    mul_929: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(add_383, 0.5);  add_383 = None
    mul_930: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_931: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_930, -0.5);  mul_930 = None
    exp_24: "f32[8, 784, 1024]" = torch.ops.aten.exp.default(mul_931);  mul_931 = None
    mul_932: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_933: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_63, mul_932);  view_63 = mul_932 = None
    add_384: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(mul_929, mul_933);  mul_929 = mul_933 = None
    mul_934: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_944, add_384);  view_944 = add_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_945: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_934, [6272, 1024]);  mul_934 = None
    permute_906: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_244: "f32[6272, 128]" = torch.ops.aten.mm.default(view_945, permute_906);  permute_906 = None
    permute_907: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_945, [1, 0])
    mm_245: "f32[1024, 128]" = torch.ops.aten.mm.default(permute_907, view_62);  permute_907 = view_62 = None
    permute_908: "f32[128, 1024]" = torch.ops.aten.permute.default(mm_245, [1, 0]);  mm_245 = None
    sum_437: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_945, [0], True);  view_945 = None
    view_946: "f32[1024]" = torch.ops.aten.view.default(sum_437, [1024]);  sum_437 = None
    permute_909: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_908, [1, 0]);  permute_908 = None
    view_947: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_244, [8, 784, 128]);  mm_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    sub_302: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(add_36, getitem_51);  add_36 = getitem_51 = None
    mul_935: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_302, rsqrt_13);  sub_302 = None
    mul_936: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_947, primals_77);  primals_77 = None
    mul_937: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_936, 128)
    sum_438: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_936, [2], True)
    mul_938: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_936, mul_935);  mul_936 = None
    sum_439: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_938, [2], True);  mul_938 = None
    mul_939: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_935, sum_439);  sum_439 = None
    sub_303: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_937, sum_438);  mul_937 = sum_438 = None
    sub_304: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_303, mul_939);  sub_303 = mul_939 = None
    div_73: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 128);  rsqrt_13 = None
    mul_940: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_73, sub_304);  div_73 = sub_304 = None
    mul_941: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_947, mul_935);  mul_935 = None
    sum_440: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_941, [0, 1]);  mul_941 = None
    sum_441: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_947, [0, 1]);  view_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_385: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(permute_901, mul_940);  permute_901 = mul_940 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_324: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_385, memory_format = torch.contiguous_format)
    view_948: "f32[6272, 128]" = torch.ops.aten.view.default(clone_324, [6272, 128]);  clone_324 = None
    permute_910: "f32[128, 128]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
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
    alias_52: "f32[8, 2, 784, 64]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
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
    permute_916: "f32[256, 128]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    mm_248: "f32[392, 128]" = torch.ops.aten.mm.default(view_954, permute_916);  permute_916 = None
    permute_917: "f32[256, 392]" = torch.ops.aten.permute.default(view_954, [1, 0])
    mm_249: "f32[256, 128]" = torch.ops.aten.mm.default(permute_917, view_56);  permute_917 = view_56 = None
    permute_918: "f32[128, 256]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    sum_443: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_954, [0], True);  view_954 = None
    view_955: "f32[256]" = torch.ops.aten.view.default(sum_443, [256]);  sum_443 = None
    permute_919: "f32[256, 128]" = torch.ops.aten.permute.default(permute_918, [1, 0]);  permute_918 = None
    view_956: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_248, [8, 49, 128]);  mm_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_326: "f32[8, 49, 128]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
    sub_305: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(clone_326, getitem_43);  clone_326 = getitem_43 = None
    mul_942: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(sub_305, rsqrt_12);  sub_305 = None
    mul_943: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_956, primals_71);  primals_71 = None
    mul_944: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_943, 128)
    sum_444: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_943, [2], True)
    mul_945: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_943, mul_942);  mul_943 = None
    sum_445: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_945, [2], True);  mul_945 = None
    mul_946: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_942, sum_445);  sum_445 = None
    sub_306: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(mul_944, sum_444);  mul_944 = sum_444 = None
    sub_307: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(sub_306, mul_946);  sub_306 = mul_946 = None
    div_74: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 128);  rsqrt_12 = None
    mul_947: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(div_74, sub_307);  div_74 = sub_307 = None
    mul_948: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_956, mul_942);  mul_942 = None
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
    permute_923: "f32[128, 128]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
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
    sub_308: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_19, getitem_41);  clone_19 = getitem_41 = None
    mul_949: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_308, rsqrt_11);  sub_308 = None
    mul_950: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_327, primals_65);  primals_65 = None
    mul_951: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_950, 128)
    sum_450: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_950, [2], True)
    mul_952: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_950, mul_949);  mul_950 = None
    sum_451: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_952, [2], True);  mul_952 = None
    mul_953: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_949, sum_451);  sum_451 = None
    sub_309: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_951, sum_450);  mul_951 = sum_450 = None
    sub_310: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_309, mul_953);  sub_309 = mul_953 = None
    div_75: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 128);  rsqrt_11 = None
    mul_954: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_75, sub_310);  div_75 = sub_310 = None
    mul_955: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_327, mul_949);  mul_949 = None
    sum_452: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_955, [0, 1]);  mul_955 = None
    sum_453: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_327, [0, 1]);  clone_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_387: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_385, mul_954);  add_385 = mul_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone_328: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_387, memory_format = torch.contiguous_format);  add_387 = None
    clone_329: "f32[8, 784, 128]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    sub_311: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_329, getitem_39);  clone_329 = getitem_39 = None
    mul_956: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_311, rsqrt_10);  sub_311 = None
    mul_957: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_328, primals_63);  primals_63 = None
    mul_958: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_957, 128)
    sum_454: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_957, [2], True)
    mul_959: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_957, mul_956);  mul_957 = None
    sum_455: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_959, [2], True);  mul_959 = None
    mul_960: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_956, sum_455);  sum_455 = None
    sub_312: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_958, sum_454);  mul_958 = sum_454 = None
    sub_313: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_312, mul_960);  sub_312 = mul_960 = None
    div_76: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 128);  rsqrt_10 = None
    mul_961: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_76, sub_313);  div_76 = sub_313 = None
    mul_962: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_328, mul_956);  mul_956 = None
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
    permute_929: "f32[64, 512]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_252: "f32[25088, 512]" = torch.ops.aten.mm.default(view_965, permute_929);  permute_929 = None
    permute_930: "f32[64, 25088]" = torch.ops.aten.permute.default(view_965, [1, 0])
    mm_253: "f32[64, 512]" = torch.ops.aten.mm.default(permute_930, view_47);  permute_930 = view_47 = None
    permute_931: "f32[512, 64]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    sum_459: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_965, [0], True);  view_965 = None
    view_966: "f32[64]" = torch.ops.aten.view.default(sum_459, [64]);  sum_459 = None
    permute_932: "f32[64, 512]" = torch.ops.aten.permute.default(permute_931, [1, 0]);  permute_931 = None
    view_967: "f32[8, 3136, 512]" = torch.ops.aten.view.default(mm_252, [8, 3136, 512]);  mm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_963: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476)
    erf_53: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_963);  mul_963 = None
    add_388: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
    mul_964: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(add_388, 0.5);  add_388 = None
    mul_965: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_46, view_46)
    mul_966: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_965, -0.5);  mul_965 = None
    exp_25: "f32[8, 3136, 512]" = torch.ops.aten.exp.default(mul_966);  mul_966 = None
    mul_967: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_968: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_46, mul_967);  view_46 = mul_967 = None
    add_389: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(mul_964, mul_968);  mul_964 = mul_968 = None
    mul_969: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_967, add_389);  view_967 = add_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_968: "f32[25088, 512]" = torch.ops.aten.view.default(mul_969, [25088, 512]);  mul_969 = None
    permute_933: "f32[512, 64]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_254: "f32[25088, 64]" = torch.ops.aten.mm.default(view_968, permute_933);  permute_933 = None
    permute_934: "f32[512, 25088]" = torch.ops.aten.permute.default(view_968, [1, 0])
    mm_255: "f32[512, 64]" = torch.ops.aten.mm.default(permute_934, view_45);  permute_934 = view_45 = None
    permute_935: "f32[64, 512]" = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
    sum_460: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_968, [0], True);  view_968 = None
    view_969: "f32[512]" = torch.ops.aten.view.default(sum_460, [512]);  sum_460 = None
    permute_936: "f32[512, 64]" = torch.ops.aten.permute.default(permute_935, [1, 0]);  permute_935 = None
    view_970: "f32[8, 3136, 64]" = torch.ops.aten.view.default(mm_254, [8, 3136, 64]);  mm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_331: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_25, memory_format = torch.contiguous_format);  add_25 = None
    sub_314: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_331, getitem_37);  clone_331 = getitem_37 = None
    mul_970: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_314, rsqrt_9);  sub_314 = None
    mul_971: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(view_970, primals_55);  primals_55 = None
    mul_972: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_971, 64)
    sum_461: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_971, [2], True)
    mul_973: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_971, mul_970);  mul_971 = None
    sum_462: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_973, [2], True);  mul_973 = None
    mul_974: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_970, sum_462);  sum_462 = None
    sub_315: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_972, sum_461);  mul_972 = sum_461 = None
    sub_316: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_315, mul_974);  sub_315 = mul_974 = None
    div_77: "f32[8, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 64);  rsqrt_9 = None
    mul_975: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_77, sub_316);  div_77 = sub_316 = None
    mul_976: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(view_970, mul_970);  mul_970 = None
    sum_463: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_976, [0, 1]);  mul_976 = None
    sum_464: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_970, [0, 1]);  view_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_390: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(view_964, mul_975);  view_964 = mul_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_332: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_390, memory_format = torch.contiguous_format)
    view_971: "f32[25088, 64]" = torch.ops.aten.view.default(clone_332, [25088, 64]);  clone_332 = None
    permute_937: "f32[64, 64]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
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
    alias_53: "f32[8, 1, 3136, 64]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
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
    permute_943: "f32[128, 64]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    mm_258: "f32[392, 64]" = torch.ops.aten.mm.default(view_977, permute_943);  permute_943 = None
    permute_944: "f32[128, 392]" = torch.ops.aten.permute.default(view_977, [1, 0])
    mm_259: "f32[128, 64]" = torch.ops.aten.mm.default(permute_944, view_39);  permute_944 = view_39 = None
    permute_945: "f32[64, 128]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    sum_466: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_977, [0], True);  view_977 = None
    view_978: "f32[128]" = torch.ops.aten.view.default(sum_466, [128]);  sum_466 = None
    permute_946: "f32[128, 64]" = torch.ops.aten.permute.default(permute_945, [1, 0]);  permute_945 = None
    view_979: "f32[8, 49, 64]" = torch.ops.aten.view.default(mm_258, [8, 49, 64]);  mm_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_334: "f32[8, 49, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    sub_317: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(clone_334, getitem_29);  clone_334 = getitem_29 = None
    mul_977: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(sub_317, rsqrt_8);  sub_317 = None
    mul_978: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(view_979, primals_49);  primals_49 = None
    mul_979: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_978, 64)
    sum_467: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_978, [2], True)
    mul_980: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_978, mul_977);  mul_978 = None
    sum_468: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_980, [2], True);  mul_980 = None
    mul_981: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_977, sum_468);  sum_468 = None
    sub_318: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(mul_979, sum_467);  mul_979 = sum_467 = None
    sub_319: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(sub_318, mul_981);  sub_318 = mul_981 = None
    div_78: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 64);  rsqrt_8 = None
    mul_982: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(div_78, sub_319);  div_78 = sub_319 = None
    mul_983: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(view_979, mul_977);  mul_977 = None
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
    permute_950: "f32[64, 64]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
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
    clone_336: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format);  add_20 = None
    sub_320: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_336, getitem_27);  clone_336 = getitem_27 = None
    mul_984: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_320, rsqrt_7);  sub_320 = None
    mul_985: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_335, primals_43);  primals_43 = None
    mul_986: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_985, 64)
    sum_473: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_985, [2], True)
    mul_987: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_985, mul_984);  mul_985 = None
    sum_474: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_987, [2], True);  mul_987 = None
    mul_988: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_984, sum_474);  sum_474 = None
    sub_321: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_986, sum_473);  mul_986 = sum_473 = None
    sub_322: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_321, mul_988);  sub_321 = mul_988 = None
    div_79: "f32[8, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 64);  rsqrt_7 = None
    mul_989: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_79, sub_322);  div_79 = sub_322 = None
    mul_990: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_335, mul_984);  mul_984 = None
    sum_475: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_990, [0, 1]);  mul_990 = None
    sum_476: "f32[64]" = torch.ops.aten.sum.dim_IntList(clone_335, [0, 1]);  clone_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_392: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_390, mul_989);  add_390 = mul_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_337: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_392, memory_format = torch.contiguous_format)
    view_986: "f32[25088, 64]" = torch.ops.aten.view.default(clone_337, [25088, 64]);  clone_337 = None
    permute_954: "f32[64, 512]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_262: "f32[25088, 512]" = torch.ops.aten.mm.default(view_986, permute_954);  permute_954 = None
    permute_955: "f32[64, 25088]" = torch.ops.aten.permute.default(view_986, [1, 0])
    mm_263: "f32[64, 512]" = torch.ops.aten.mm.default(permute_955, view_32);  permute_955 = view_32 = None
    permute_956: "f32[512, 64]" = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
    sum_477: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_986, [0], True);  view_986 = None
    view_987: "f32[64]" = torch.ops.aten.view.default(sum_477, [64]);  sum_477 = None
    permute_957: "f32[64, 512]" = torch.ops.aten.permute.default(permute_956, [1, 0]);  permute_956 = None
    view_988: "f32[8, 3136, 512]" = torch.ops.aten.view.default(mm_262, [8, 3136, 512]);  mm_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_991: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476)
    erf_54: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_991);  mul_991 = None
    add_393: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
    mul_992: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(add_393, 0.5);  add_393 = None
    mul_993: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_31, view_31)
    mul_994: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_993, -0.5);  mul_993 = None
    exp_26: "f32[8, 3136, 512]" = torch.ops.aten.exp.default(mul_994);  mul_994 = None
    mul_995: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_996: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_31, mul_995);  view_31 = mul_995 = None
    add_394: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(mul_992, mul_996);  mul_992 = mul_996 = None
    mul_997: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_988, add_394);  view_988 = add_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_989: "f32[25088, 512]" = torch.ops.aten.view.default(mul_997, [25088, 512]);  mul_997 = None
    permute_958: "f32[512, 64]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_264: "f32[25088, 64]" = torch.ops.aten.mm.default(view_989, permute_958);  permute_958 = None
    permute_959: "f32[512, 25088]" = torch.ops.aten.permute.default(view_989, [1, 0])
    mm_265: "f32[512, 64]" = torch.ops.aten.mm.default(permute_959, view_30);  permute_959 = view_30 = None
    permute_960: "f32[64, 512]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    sum_478: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_989, [0], True);  view_989 = None
    view_990: "f32[512]" = torch.ops.aten.view.default(sum_478, [512]);  sum_478 = None
    permute_961: "f32[512, 64]" = torch.ops.aten.permute.default(permute_960, [1, 0]);  permute_960 = None
    view_991: "f32[8, 3136, 64]" = torch.ops.aten.view.default(mm_264, [8, 3136, 64]);  mm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    clone_338: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_16, memory_format = torch.contiguous_format);  add_16 = None
    sub_323: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_338, getitem_25);  clone_338 = getitem_25 = None
    mul_998: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_323, rsqrt_6);  sub_323 = None
    mul_999: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(view_991, primals_37);  primals_37 = None
    mul_1000: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_999, 64)
    sum_479: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_999, [2], True)
    mul_1001: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_999, mul_998);  mul_999 = None
    sum_480: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1001, [2], True);  mul_1001 = None
    mul_1002: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_998, sum_480);  sum_480 = None
    sub_324: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_1000, sum_479);  mul_1000 = sum_479 = None
    sub_325: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_324, mul_1002);  sub_324 = mul_1002 = None
    div_80: "f32[8, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 64);  rsqrt_6 = None
    mul_1003: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_80, sub_325);  div_80 = sub_325 = None
    mul_1004: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(view_991, mul_998);  mul_998 = None
    sum_481: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1004, [0, 1]);  mul_1004 = None
    sum_482: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_991, [0, 1]);  view_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_395: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_392, mul_1003);  add_392 = mul_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_339: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_395, memory_format = torch.contiguous_format)
    view_992: "f32[25088, 64]" = torch.ops.aten.view.default(clone_339, [25088, 64]);  clone_339 = None
    permute_962: "f32[64, 64]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
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
    alias_54: "f32[8, 1, 3136, 64]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
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
    permute_968: "f32[128, 64]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_268: "f32[392, 64]" = torch.ops.aten.mm.default(view_998, permute_968);  permute_968 = None
    permute_969: "f32[128, 392]" = torch.ops.aten.permute.default(view_998, [1, 0])
    mm_269: "f32[128, 64]" = torch.ops.aten.mm.default(permute_969, view_24);  permute_969 = view_24 = None
    permute_970: "f32[64, 128]" = torch.ops.aten.permute.default(mm_269, [1, 0]);  mm_269 = None
    sum_484: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_998, [0], True);  view_998 = None
    view_999: "f32[128]" = torch.ops.aten.view.default(sum_484, [128]);  sum_484 = None
    permute_971: "f32[128, 64]" = torch.ops.aten.permute.default(permute_970, [1, 0]);  permute_970 = None
    view_1000: "f32[8, 49, 64]" = torch.ops.aten.view.default(mm_268, [8, 49, 64]);  mm_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_341: "f32[8, 49, 64]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    sub_326: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(clone_341, getitem_17);  clone_341 = getitem_17 = None
    mul_1005: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(sub_326, rsqrt_5);  sub_326 = None
    mul_1006: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(view_1000, primals_31);  primals_31 = None
    mul_1007: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_1006, 64)
    sum_485: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_1006, [2], True)
    mul_1008: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_1006, mul_1005);  mul_1006 = None
    sum_486: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_1008, [2], True);  mul_1008 = None
    mul_1009: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_1005, sum_486);  sum_486 = None
    sub_327: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(mul_1007, sum_485);  mul_1007 = sum_485 = None
    sub_328: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(sub_327, mul_1009);  sub_327 = mul_1009 = None
    div_81: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 64);  rsqrt_5 = None
    mul_1010: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(div_81, sub_328);  div_81 = sub_328 = None
    mul_1011: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(view_1000, mul_1005);  mul_1005 = None
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
    permute_975: "f32[64, 64]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
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
    clone_343: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    sub_329: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_343, getitem_15);  clone_343 = getitem_15 = None
    mul_1012: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_329, rsqrt_4);  sub_329 = None
    mul_1013: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_342, primals_25);  primals_25 = None
    mul_1014: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1013, 64)
    sum_491: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1013, [2], True)
    mul_1015: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1013, mul_1012);  mul_1013 = None
    sum_492: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1015, [2], True);  mul_1015 = None
    mul_1016: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1012, sum_492);  sum_492 = None
    sub_330: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_1014, sum_491);  mul_1014 = sum_491 = None
    sub_331: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_330, mul_1016);  sub_330 = mul_1016 = None
    div_82: "f32[8, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 64);  rsqrt_4 = None
    mul_1017: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_82, sub_331);  div_82 = sub_331 = None
    mul_1018: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_342, mul_1012);  mul_1012 = None
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
    permute_981: "f32[64, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_272: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1009, permute_981);  permute_981 = None
    permute_982: "f32[64, 25088]" = torch.ops.aten.permute.default(view_1009, [1, 0])
    mm_273: "f32[64, 512]" = torch.ops.aten.mm.default(permute_982, view_14);  permute_982 = view_14 = None
    permute_983: "f32[512, 64]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    sum_496: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_1009, [0], True);  view_1009 = None
    view_1010: "f32[64]" = torch.ops.aten.view.default(sum_496, [64]);  sum_496 = None
    permute_984: "f32[64, 512]" = torch.ops.aten.permute.default(permute_983, [1, 0]);  permute_983 = None
    view_1011: "f32[8, 3136, 512]" = torch.ops.aten.view.default(mm_272, [8, 3136, 512]);  mm_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1019: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476)
    erf_55: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_1019);  mul_1019 = None
    add_399: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
    mul_1020: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(add_399, 0.5);  add_399 = None
    mul_1021: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_13, view_13)
    mul_1022: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_1021, -0.5);  mul_1021 = None
    exp_27: "f32[8, 3136, 512]" = torch.ops.aten.exp.default(mul_1022);  mul_1022 = None
    mul_1023: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_1024: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_13, mul_1023);  view_13 = mul_1023 = None
    add_400: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(mul_1020, mul_1024);  mul_1020 = mul_1024 = None
    mul_1025: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_1011, add_400);  view_1011 = add_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1012: "f32[25088, 512]" = torch.ops.aten.view.default(mul_1025, [25088, 512]);  mul_1025 = None
    permute_985: "f32[512, 64]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_274: "f32[25088, 64]" = torch.ops.aten.mm.default(view_1012, permute_985);  permute_985 = None
    permute_986: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1012, [1, 0])
    mm_275: "f32[512, 64]" = torch.ops.aten.mm.default(permute_986, view_12);  permute_986 = view_12 = None
    permute_987: "f32[64, 512]" = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
    sum_497: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1012, [0], True);  view_1012 = None
    view_1013: "f32[512]" = torch.ops.aten.view.default(sum_497, [512]);  sum_497 = None
    permute_988: "f32[512, 64]" = torch.ops.aten.permute.default(permute_987, [1, 0]);  permute_987 = None
    view_1014: "f32[8, 3136, 64]" = torch.ops.aten.view.default(mm_274, [8, 3136, 64]);  mm_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    sub_332: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(add_6, getitem_13);  add_6 = getitem_13 = None
    mul_1026: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_332, rsqrt_3);  sub_332 = None
    mul_1027: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(view_1014, primals_17);  primals_17 = None
    mul_1028: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1027, 64)
    sum_498: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1027, [2], True)
    mul_1029: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1027, mul_1026);  mul_1027 = None
    sum_499: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1029, [2], True);  mul_1029 = None
    mul_1030: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1026, sum_499);  sum_499 = None
    sub_333: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_1028, sum_498);  mul_1028 = sum_498 = None
    sub_334: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_333, mul_1030);  sub_333 = mul_1030 = None
    div_83: "f32[8, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 64);  rsqrt_3 = None
    mul_1031: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_83, sub_334);  div_83 = sub_334 = None
    mul_1032: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(view_1014, mul_1026);  mul_1026 = None
    sum_500: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1032, [0, 1]);  mul_1032 = None
    sum_501: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_1014, [0, 1]);  view_1014 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_401: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(permute_980, mul_1031);  permute_980 = mul_1031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_345: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_401, memory_format = torch.contiguous_format)
    view_1015: "f32[25088, 64]" = torch.ops.aten.view.default(clone_345, [25088, 64]);  clone_345 = None
    permute_989: "f32[64, 64]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
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
    alias_55: "f32[8, 1, 3136, 64]" = torch.ops.aten.alias.default(alias);  alias = None
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
    permute_995: "f32[128, 64]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    mm_278: "f32[392, 64]" = torch.ops.aten.mm.default(view_1021, permute_995);  permute_995 = None
    permute_996: "f32[128, 392]" = torch.ops.aten.permute.default(view_1021, [1, 0])
    mm_279: "f32[128, 64]" = torch.ops.aten.mm.default(permute_996, view_6);  permute_996 = view_6 = None
    permute_997: "f32[64, 128]" = torch.ops.aten.permute.default(mm_279, [1, 0]);  mm_279 = None
    sum_503: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1021, [0], True);  view_1021 = None
    view_1022: "f32[128]" = torch.ops.aten.view.default(sum_503, [128]);  sum_503 = None
    permute_998: "f32[128, 64]" = torch.ops.aten.permute.default(permute_997, [1, 0]);  permute_997 = None
    view_1023: "f32[8, 49, 64]" = torch.ops.aten.view.default(mm_278, [8, 49, 64]);  mm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    clone_347: "f32[8, 49, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    sub_335: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(clone_347, getitem_5);  clone_347 = getitem_5 = None
    mul_1033: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(sub_335, rsqrt_2);  sub_335 = None
    mul_1034: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(view_1023, primals_11);  primals_11 = None
    mul_1035: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_1034, 64)
    sum_504: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_1034, [2], True)
    mul_1036: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_1034, mul_1033);  mul_1034 = None
    sum_505: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_1036, [2], True);  mul_1036 = None
    mul_1037: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_1033, sum_505);  sum_505 = None
    sub_336: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(mul_1035, sum_504);  mul_1035 = sum_504 = None
    sub_337: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(sub_336, mul_1037);  sub_336 = mul_1037 = None
    div_84: "f32[8, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 64);  rsqrt_2 = None
    mul_1038: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(div_84, sub_337);  div_84 = sub_337 = None
    mul_1039: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(view_1023, mul_1033);  mul_1033 = None
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
    permute_1002: "f32[64, 64]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
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
    sub_338: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_1, getitem_3);  clone_1 = getitem_3 = None
    mul_1040: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_338, rsqrt_1);  sub_338 = None
    mul_1041: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_348, primals_5);  primals_5 = None
    mul_1042: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1041, 64)
    sum_510: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1041, [2], True)
    mul_1043: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1041, mul_1040);  mul_1041 = None
    sum_511: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1043, [2], True);  mul_1043 = None
    mul_1044: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1040, sum_511);  sum_511 = None
    sub_339: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_1042, sum_510);  mul_1042 = sum_510 = None
    sub_340: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_339, mul_1044);  sub_339 = mul_1044 = None
    div_85: "f32[8, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 64);  rsqrt_1 = None
    mul_1045: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_85, sub_340);  div_85 = sub_340 = None
    mul_1046: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_348, mul_1040);  mul_1040 = None
    sum_512: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1046, [0, 1]);  mul_1046 = None
    sum_513: "f32[64]" = torch.ops.aten.sum.dim_IntList(clone_348, [0, 1]);  clone_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_403: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_401, mul_1045);  add_401 = mul_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone_349: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_403, memory_format = torch.contiguous_format);  add_403 = None
    clone_350: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    sub_341: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_350, getitem_1);  clone_350 = getitem_1 = None
    mul_1047: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_341, rsqrt);  sub_341 = None
    mul_1048: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_349, primals_3);  primals_3 = None
    mul_1049: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1048, 64)
    sum_514: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1048, [2], True)
    mul_1050: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1048, mul_1047);  mul_1048 = None
    sum_515: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1050, [2], True);  mul_1050 = None
    mul_1051: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1047, sum_515);  sum_515 = None
    sub_342: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_1049, sum_514);  mul_1049 = sum_514 = None
    sub_343: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_342, mul_1051);  sub_342 = mul_1051 = None
    div_86: "f32[8, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt, 64);  rsqrt = None
    mul_1052: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_86, sub_343);  div_86 = sub_343 = None
    mul_1053: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_349, mul_1047);  mul_1047 = None
    sum_516: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1053, [0, 1]);  mul_1053 = None
    sum_517: "f32[64]" = torch.ops.aten.sum.dim_IntList(clone_349, [0, 1]);  clone_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    permute_1006: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(mul_1052, [0, 2, 1]);  mul_1052 = None
    view_1030: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_1006, [8, 64, 56, 56]);  permute_1006 = None
    sum_518: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_1030, [0, 2, 3])
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(view_1030, primals_521, primals_1, [64], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  view_1030 = primals_521 = primals_1 = None
    getitem_549: "f32[64, 3, 4, 4]" = convolution_backward_32[1];  convolution_backward_32 = None
    return pytree.tree_unflatten([addmm_140, getitem_549, sum_518, sum_516, sum_517, sum_512, sum_513, permute_1005, view_1028, getitem_546, sum_508, sum_506, sum_507, permute_998, view_1022, permute_992, view_1016, sum_500, sum_501, permute_988, view_1013, permute_984, view_1010, getitem_539, sum_495, sum_493, sum_494, permute_978, view_1005, getitem_536, sum_489, sum_487, sum_488, permute_971, view_999, permute_965, view_993, sum_481, sum_482, permute_961, view_990, permute_957, view_987, sum_475, sum_476, permute_953, view_984, getitem_529, sum_471, sum_469, sum_470, permute_946, view_978, permute_940, view_972, sum_463, sum_464, permute_936, view_969, permute_932, view_966, getitem_522, sum_458, sum_456, sum_457, sum_452, sum_453, permute_926, view_961, getitem_519, sum_448, sum_446, sum_447, permute_919, view_955, permute_913, view_949, sum_440, sum_441, permute_909, view_946, permute_905, view_943, getitem_512, sum_435, sum_433, sum_434, permute_899, view_938, getitem_509, sum_429, sum_427, sum_428, permute_892, view_932, permute_886, view_926, sum_421, sum_422, permute_882, view_923, permute_878, view_920, sum_415, sum_416, permute_874, view_917, getitem_502, sum_411, sum_409, sum_410, permute_867, view_911, permute_861, view_905, sum_403, sum_404, permute_857, view_902, permute_853, view_899, sum_397, sum_398, permute_849, view_896, getitem_495, sum_393, sum_391, sum_392, permute_842, view_890, permute_836, view_884, sum_385, sum_386, permute_832, view_881, permute_828, view_878, getitem_488, sum_380, sum_378, sum_379, sum_374, sum_375, permute_822, view_873, getitem_485, sum_370, sum_368, sum_369, permute_815, view_867, permute_809, view_861, sum_362, sum_363, permute_805, view_858, permute_801, view_855, getitem_478, sum_357, sum_355, sum_356, permute_795, view_850, getitem_475, sum_351, sum_349, sum_350, permute_788, view_844, permute_782, view_838, sum_343, sum_344, permute_778, view_835, permute_774, view_832, sum_337, sum_338, permute_770, view_829, getitem_468, sum_333, sum_331, sum_332, permute_763, view_823, permute_757, view_817, sum_325, sum_326, permute_753, view_814, permute_749, view_811, sum_319, sum_320, permute_745, view_808, getitem_461, sum_315, sum_313, sum_314, permute_738, view_802, permute_732, view_796, sum_307, sum_308, permute_728, view_793, permute_724, view_790, sum_301, sum_302, permute_720, view_787, getitem_454, sum_297, sum_295, sum_296, permute_713, view_781, permute_707, view_775, sum_289, sum_290, permute_703, view_772, permute_699, view_769, sum_283, sum_284, permute_695, view_766, getitem_447, sum_279, sum_277, sum_278, permute_688, view_760, permute_682, view_754, sum_271, sum_272, permute_678, view_751, permute_674, view_748, sum_265, sum_266, permute_670, view_745, getitem_440, sum_261, sum_259, sum_260, permute_663, view_739, permute_657, view_733, sum_253, sum_254, permute_653, view_730, permute_649, view_727, sum_247, sum_248, permute_645, view_724, getitem_433, sum_243, sum_241, sum_242, permute_638, view_718, permute_632, view_712, sum_235, sum_236, permute_628, view_709, permute_624, view_706, sum_229, sum_230, permute_620, view_703, getitem_426, sum_225, sum_223, sum_224, permute_613, view_697, permute_607, view_691, sum_217, sum_218, permute_603, view_688, permute_599, view_685, sum_211, sum_212, permute_595, view_682, getitem_419, sum_207, sum_205, sum_206, permute_588, view_676, permute_582, view_670, sum_199, sum_200, permute_578, view_667, permute_574, view_664, sum_193, sum_194, permute_570, view_661, getitem_412, sum_189, sum_187, sum_188, permute_563, view_655, permute_557, view_649, sum_181, sum_182, permute_553, view_646, permute_549, view_643, sum_175, sum_176, permute_545, view_640, getitem_405, sum_171, sum_169, sum_170, permute_538, view_634, permute_532, view_628, sum_163, sum_164, permute_528, view_625, permute_524, view_622, sum_157, sum_158, permute_520, view_619, getitem_398, sum_153, sum_151, sum_152, permute_513, view_613, permute_507, view_607, sum_145, sum_146, permute_503, view_604, permute_499, view_601, sum_139, sum_140, permute_495, view_598, getitem_391, sum_135, sum_133, sum_134, permute_488, view_592, permute_482, view_586, sum_127, sum_128, permute_478, view_583, permute_474, view_580, sum_121, sum_122, permute_470, view_577, getitem_384, sum_117, sum_115, sum_116, permute_463, view_571, permute_457, view_565, sum_109, sum_110, permute_453, view_562, permute_449, view_559, sum_103, sum_104, permute_445, view_556, getitem_377, sum_99, sum_97, sum_98, permute_438, view_550, permute_432, view_544, sum_91, sum_92, permute_428, view_541, permute_424, view_538, sum_85, sum_86, permute_420, view_535, getitem_370, sum_81, sum_79, sum_80, permute_413, view_529, permute_407, view_523, sum_73, sum_74, permute_403, view_520, permute_399, view_517, sum_67, sum_68, permute_395, view_514, getitem_363, sum_63, sum_61, sum_62, permute_388, view_508, permute_382, view_502, sum_55, sum_56, permute_378, view_499, permute_374, view_496, getitem_356, sum_50, sum_48, sum_49, sum_44, sum_45, permute_368, view_491, permute_363, view_487, permute_357, view_481, sum_37, sum_38, permute_353, view_478, permute_349, view_475, getitem_349, sum_32, sum_30, sum_31, permute_343, view_470, permute_338, view_466, permute_332, view_460, sum_23, sum_24, permute_328, view_457, permute_324, view_454, sum_17, sum_18, permute_320, view_451, permute_315, view_447, permute_309, view_441, sum_10, sum_11, permute_305, view_438, permute_301, view_435, sum_4, sum_5, permute_297, view_433, None], self._out_spec)
    