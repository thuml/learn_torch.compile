from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_2: "f32[64]", primals_4: "f32[64, 64, 3, 3]", primals_5: "f32[64]", primals_7: "f32[128, 64, 3, 3]", primals_8: "f32[128]", primals_10: "f32[64, 128, 1, 1]", primals_11: "f32[64]", primals_13: "f32[128, 32, 3, 3]", primals_14: "f32[128]", primals_16: "f32[32, 64, 1, 1]", primals_18: "f32[32]", primals_20: "f32[128, 32, 1, 1]", primals_22: "f32[256, 64, 1, 1]", primals_23: "f32[256]", primals_25: "f32[256, 128, 1, 1]", primals_26: "f32[256]", primals_28: "f32[64, 256, 1, 1]", primals_29: "f32[64]", primals_31: "f32[128, 32, 3, 3]", primals_32: "f32[128]", primals_34: "f32[32, 64, 1, 1]", primals_36: "f32[32]", primals_38: "f32[128, 32, 1, 1]", primals_40: "f32[256, 64, 1, 1]", primals_41: "f32[256]", primals_43: "f32[64, 256, 1, 1]", primals_44: "f32[64]", primals_46: "f32[128, 32, 3, 3]", primals_47: "f32[128]", primals_49: "f32[32, 64, 1, 1]", primals_51: "f32[32]", primals_53: "f32[128, 32, 1, 1]", primals_55: "f32[256, 64, 1, 1]", primals_56: "f32[256]", primals_58: "f32[128, 256, 1, 1]", primals_59: "f32[128]", primals_61: "f32[256, 64, 3, 3]", primals_62: "f32[256]", primals_64: "f32[64, 128, 1, 1]", primals_66: "f32[64]", primals_68: "f32[256, 64, 1, 1]", primals_70: "f32[512, 128, 1, 1]", primals_71: "f32[512]", primals_73: "f32[512, 256, 1, 1]", primals_74: "f32[512]", primals_76: "f32[128, 512, 1, 1]", primals_77: "f32[128]", primals_79: "f32[256, 64, 3, 3]", primals_80: "f32[256]", primals_82: "f32[64, 128, 1, 1]", primals_84: "f32[64]", primals_86: "f32[256, 64, 1, 1]", primals_88: "f32[512, 128, 1, 1]", primals_89: "f32[512]", primals_91: "f32[128, 512, 1, 1]", primals_92: "f32[128]", primals_94: "f32[256, 64, 3, 3]", primals_95: "f32[256]", primals_97: "f32[64, 128, 1, 1]", primals_99: "f32[64]", primals_101: "f32[256, 64, 1, 1]", primals_103: "f32[512, 128, 1, 1]", primals_104: "f32[512]", primals_106: "f32[128, 512, 1, 1]", primals_107: "f32[128]", primals_109: "f32[256, 64, 3, 3]", primals_110: "f32[256]", primals_112: "f32[64, 128, 1, 1]", primals_114: "f32[64]", primals_116: "f32[256, 64, 1, 1]", primals_118: "f32[512, 128, 1, 1]", primals_119: "f32[512]", primals_121: "f32[256, 512, 1, 1]", primals_122: "f32[256]", primals_124: "f32[512, 128, 3, 3]", primals_125: "f32[512]", primals_127: "f32[128, 256, 1, 1]", primals_129: "f32[128]", primals_131: "f32[512, 128, 1, 1]", primals_133: "f32[1024, 256, 1, 1]", primals_134: "f32[1024]", primals_136: "f32[1024, 512, 1, 1]", primals_137: "f32[1024]", primals_139: "f32[256, 1024, 1, 1]", primals_140: "f32[256]", primals_142: "f32[512, 128, 3, 3]", primals_143: "f32[512]", primals_145: "f32[128, 256, 1, 1]", primals_147: "f32[128]", primals_149: "f32[512, 128, 1, 1]", primals_151: "f32[1024, 256, 1, 1]", primals_152: "f32[1024]", primals_154: "f32[256, 1024, 1, 1]", primals_155: "f32[256]", primals_157: "f32[512, 128, 3, 3]", primals_158: "f32[512]", primals_160: "f32[128, 256, 1, 1]", primals_162: "f32[128]", primals_164: "f32[512, 128, 1, 1]", primals_166: "f32[1024, 256, 1, 1]", primals_167: "f32[1024]", primals_169: "f32[256, 1024, 1, 1]", primals_170: "f32[256]", primals_172: "f32[512, 128, 3, 3]", primals_173: "f32[512]", primals_175: "f32[128, 256, 1, 1]", primals_177: "f32[128]", primals_179: "f32[512, 128, 1, 1]", primals_181: "f32[1024, 256, 1, 1]", primals_182: "f32[1024]", primals_184: "f32[256, 1024, 1, 1]", primals_185: "f32[256]", primals_187: "f32[512, 128, 3, 3]", primals_188: "f32[512]", primals_190: "f32[128, 256, 1, 1]", primals_192: "f32[128]", primals_194: "f32[512, 128, 1, 1]", primals_196: "f32[1024, 256, 1, 1]", primals_197: "f32[1024]", primals_199: "f32[256, 1024, 1, 1]", primals_200: "f32[256]", primals_202: "f32[512, 128, 3, 3]", primals_203: "f32[512]", primals_205: "f32[128, 256, 1, 1]", primals_207: "f32[128]", primals_209: "f32[512, 128, 1, 1]", primals_211: "f32[1024, 256, 1, 1]", primals_212: "f32[1024]", primals_214: "f32[256, 1024, 1, 1]", primals_215: "f32[256]", primals_217: "f32[512, 128, 3, 3]", primals_218: "f32[512]", primals_220: "f32[128, 256, 1, 1]", primals_222: "f32[128]", primals_224: "f32[512, 128, 1, 1]", primals_226: "f32[1024, 256, 1, 1]", primals_227: "f32[1024]", primals_229: "f32[256, 1024, 1, 1]", primals_230: "f32[256]", primals_232: "f32[512, 128, 3, 3]", primals_233: "f32[512]", primals_235: "f32[128, 256, 1, 1]", primals_237: "f32[128]", primals_239: "f32[512, 128, 1, 1]", primals_241: "f32[1024, 256, 1, 1]", primals_242: "f32[1024]", primals_244: "f32[256, 1024, 1, 1]", primals_245: "f32[256]", primals_247: "f32[512, 128, 3, 3]", primals_248: "f32[512]", primals_250: "f32[128, 256, 1, 1]", primals_252: "f32[128]", primals_254: "f32[512, 128, 1, 1]", primals_256: "f32[1024, 256, 1, 1]", primals_257: "f32[1024]", primals_259: "f32[256, 1024, 1, 1]", primals_260: "f32[256]", primals_262: "f32[512, 128, 3, 3]", primals_263: "f32[512]", primals_265: "f32[128, 256, 1, 1]", primals_267: "f32[128]", primals_269: "f32[512, 128, 1, 1]", primals_271: "f32[1024, 256, 1, 1]", primals_272: "f32[1024]", primals_274: "f32[256, 1024, 1, 1]", primals_275: "f32[256]", primals_277: "f32[512, 128, 3, 3]", primals_278: "f32[512]", primals_280: "f32[128, 256, 1, 1]", primals_282: "f32[128]", primals_284: "f32[512, 128, 1, 1]", primals_286: "f32[1024, 256, 1, 1]", primals_287: "f32[1024]", primals_289: "f32[256, 1024, 1, 1]", primals_290: "f32[256]", primals_292: "f32[512, 128, 3, 3]", primals_293: "f32[512]", primals_295: "f32[128, 256, 1, 1]", primals_297: "f32[128]", primals_299: "f32[512, 128, 1, 1]", primals_301: "f32[1024, 256, 1, 1]", primals_302: "f32[1024]", primals_304: "f32[256, 1024, 1, 1]", primals_305: "f32[256]", primals_307: "f32[512, 128, 3, 3]", primals_308: "f32[512]", primals_310: "f32[128, 256, 1, 1]", primals_312: "f32[128]", primals_314: "f32[512, 128, 1, 1]", primals_316: "f32[1024, 256, 1, 1]", primals_317: "f32[1024]", primals_319: "f32[256, 1024, 1, 1]", primals_320: "f32[256]", primals_322: "f32[512, 128, 3, 3]", primals_323: "f32[512]", primals_325: "f32[128, 256, 1, 1]", primals_327: "f32[128]", primals_329: "f32[512, 128, 1, 1]", primals_331: "f32[1024, 256, 1, 1]", primals_332: "f32[1024]", primals_334: "f32[256, 1024, 1, 1]", primals_335: "f32[256]", primals_337: "f32[512, 128, 3, 3]", primals_338: "f32[512]", primals_340: "f32[128, 256, 1, 1]", primals_342: "f32[128]", primals_344: "f32[512, 128, 1, 1]", primals_346: "f32[1024, 256, 1, 1]", primals_347: "f32[1024]", primals_349: "f32[256, 1024, 1, 1]", primals_350: "f32[256]", primals_352: "f32[512, 128, 3, 3]", primals_353: "f32[512]", primals_355: "f32[128, 256, 1, 1]", primals_357: "f32[128]", primals_359: "f32[512, 128, 1, 1]", primals_361: "f32[1024, 256, 1, 1]", primals_362: "f32[1024]", primals_364: "f32[256, 1024, 1, 1]", primals_365: "f32[256]", primals_367: "f32[512, 128, 3, 3]", primals_368: "f32[512]", primals_370: "f32[128, 256, 1, 1]", primals_372: "f32[128]", primals_374: "f32[512, 128, 1, 1]", primals_376: "f32[1024, 256, 1, 1]", primals_377: "f32[1024]", primals_379: "f32[256, 1024, 1, 1]", primals_380: "f32[256]", primals_382: "f32[512, 128, 3, 3]", primals_383: "f32[512]", primals_385: "f32[128, 256, 1, 1]", primals_387: "f32[128]", primals_389: "f32[512, 128, 1, 1]", primals_391: "f32[1024, 256, 1, 1]", primals_392: "f32[1024]", primals_394: "f32[256, 1024, 1, 1]", primals_395: "f32[256]", primals_397: "f32[512, 128, 3, 3]", primals_398: "f32[512]", primals_400: "f32[128, 256, 1, 1]", primals_402: "f32[128]", primals_404: "f32[512, 128, 1, 1]", primals_406: "f32[1024, 256, 1, 1]", primals_407: "f32[1024]", primals_409: "f32[256, 1024, 1, 1]", primals_410: "f32[256]", primals_412: "f32[512, 128, 3, 3]", primals_413: "f32[512]", primals_415: "f32[128, 256, 1, 1]", primals_417: "f32[128]", primals_419: "f32[512, 128, 1, 1]", primals_421: "f32[1024, 256, 1, 1]", primals_422: "f32[1024]", primals_424: "f32[256, 1024, 1, 1]", primals_425: "f32[256]", primals_427: "f32[512, 128, 3, 3]", primals_428: "f32[512]", primals_430: "f32[128, 256, 1, 1]", primals_432: "f32[128]", primals_434: "f32[512, 128, 1, 1]", primals_436: "f32[1024, 256, 1, 1]", primals_437: "f32[1024]", primals_439: "f32[256, 1024, 1, 1]", primals_440: "f32[256]", primals_442: "f32[512, 128, 3, 3]", primals_443: "f32[512]", primals_445: "f32[128, 256, 1, 1]", primals_447: "f32[128]", primals_449: "f32[512, 128, 1, 1]", primals_451: "f32[1024, 256, 1, 1]", primals_452: "f32[1024]", primals_454: "f32[256, 1024, 1, 1]", primals_455: "f32[256]", primals_457: "f32[512, 128, 3, 3]", primals_458: "f32[512]", primals_460: "f32[128, 256, 1, 1]", primals_462: "f32[128]", primals_464: "f32[512, 128, 1, 1]", primals_466: "f32[1024, 256, 1, 1]", primals_467: "f32[1024]", primals_469: "f32[512, 1024, 1, 1]", primals_470: "f32[512]", primals_472: "f32[1024, 256, 3, 3]", primals_473: "f32[1024]", primals_475: "f32[256, 512, 1, 1]", primals_477: "f32[256]", primals_479: "f32[1024, 256, 1, 1]", primals_481: "f32[2048, 512, 1, 1]", primals_482: "f32[2048]", primals_484: "f32[2048, 1024, 1, 1]", primals_485: "f32[2048]", primals_487: "f32[512, 2048, 1, 1]", primals_488: "f32[512]", primals_490: "f32[1024, 256, 3, 3]", primals_491: "f32[1024]", primals_493: "f32[256, 512, 1, 1]", primals_495: "f32[256]", primals_497: "f32[1024, 256, 1, 1]", primals_499: "f32[2048, 512, 1, 1]", primals_500: "f32[2048]", primals_502: "f32[512, 2048, 1, 1]", primals_503: "f32[512]", primals_505: "f32[1024, 256, 3, 3]", primals_506: "f32[1024]", primals_508: "f32[256, 512, 1, 1]", primals_510: "f32[256]", primals_512: "f32[1024, 256, 1, 1]", primals_514: "f32[2048, 512, 1, 1]", primals_515: "f32[2048]", primals_936: "f32[8, 3, 256, 256]", convolution: "f32[8, 64, 128, 128]", squeeze_1: "f32[64]", relu: "f32[8, 64, 128, 128]", convolution_1: "f32[8, 64, 128, 128]", squeeze_4: "f32[64]", relu_1: "f32[8, 64, 128, 128]", convolution_2: "f32[8, 128, 128, 128]", squeeze_7: "f32[128]", relu_2: "f32[8, 128, 128, 128]", getitem_6: "f32[8, 128, 64, 64]", getitem_7: "i64[8, 128, 64, 64]", convolution_3: "f32[8, 64, 64, 64]", squeeze_10: "f32[64]", relu_3: "f32[8, 64, 64, 64]", convolution_4: "f32[8, 128, 64, 64]", squeeze_13: "f32[128]", relu_4: "f32[8, 128, 64, 64]", mean: "f32[8, 64, 1, 1]", convolution_5: "f32[8, 32, 1, 1]", relu_5: "f32[8, 32, 1, 1]", div: "f32[8, 2, 1, 64]", sum_3: "f32[8, 64, 64, 64]", convolution_7: "f32[8, 256, 64, 64]", squeeze_19: "f32[256]", convolution_8: "f32[8, 256, 64, 64]", squeeze_22: "f32[256]", relu_6: "f32[8, 256, 64, 64]", convolution_9: "f32[8, 64, 64, 64]", squeeze_25: "f32[64]", relu_7: "f32[8, 64, 64, 64]", convolution_10: "f32[8, 128, 64, 64]", squeeze_28: "f32[128]", relu_8: "f32[8, 128, 64, 64]", mean_1: "f32[8, 64, 1, 1]", convolution_11: "f32[8, 32, 1, 1]", relu_9: "f32[8, 32, 1, 1]", div_1: "f32[8, 2, 1, 64]", sum_6: "f32[8, 64, 64, 64]", convolution_13: "f32[8, 256, 64, 64]", squeeze_34: "f32[256]", relu_10: "f32[8, 256, 64, 64]", convolution_14: "f32[8, 64, 64, 64]", squeeze_37: "f32[64]", relu_11: "f32[8, 64, 64, 64]", convolution_15: "f32[8, 128, 64, 64]", squeeze_40: "f32[128]", relu_12: "f32[8, 128, 64, 64]", mean_2: "f32[8, 64, 1, 1]", convolution_16: "f32[8, 32, 1, 1]", relu_13: "f32[8, 32, 1, 1]", div_2: "f32[8, 2, 1, 64]", sum_9: "f32[8, 64, 64, 64]", convolution_18: "f32[8, 256, 64, 64]", squeeze_46: "f32[256]", relu_14: "f32[8, 256, 64, 64]", convolution_19: "f32[8, 128, 64, 64]", squeeze_49: "f32[128]", relu_15: "f32[8, 128, 64, 64]", convolution_20: "f32[8, 256, 64, 64]", squeeze_52: "f32[256]", relu_16: "f32[8, 256, 64, 64]", mean_3: "f32[8, 128, 1, 1]", convolution_21: "f32[8, 64, 1, 1]", relu_17: "f32[8, 64, 1, 1]", div_3: "f32[8, 2, 1, 128]", sum_12: "f32[8, 128, 64, 64]", avg_pool2d: "f32[8, 128, 32, 32]", convolution_23: "f32[8, 512, 32, 32]", squeeze_58: "f32[512]", avg_pool2d_1: "f32[8, 256, 32, 32]", convolution_24: "f32[8, 512, 32, 32]", squeeze_61: "f32[512]", relu_18: "f32[8, 512, 32, 32]", convolution_25: "f32[8, 128, 32, 32]", squeeze_64: "f32[128]", relu_19: "f32[8, 128, 32, 32]", convolution_26: "f32[8, 256, 32, 32]", squeeze_67: "f32[256]", relu_20: "f32[8, 256, 32, 32]", mean_4: "f32[8, 128, 1, 1]", convolution_27: "f32[8, 64, 1, 1]", relu_21: "f32[8, 64, 1, 1]", div_4: "f32[8, 2, 1, 128]", sum_15: "f32[8, 128, 32, 32]", convolution_29: "f32[8, 512, 32, 32]", squeeze_73: "f32[512]", relu_22: "f32[8, 512, 32, 32]", convolution_30: "f32[8, 128, 32, 32]", squeeze_76: "f32[128]", relu_23: "f32[8, 128, 32, 32]", convolution_31: "f32[8, 256, 32, 32]", squeeze_79: "f32[256]", relu_24: "f32[8, 256, 32, 32]", mean_5: "f32[8, 128, 1, 1]", convolution_32: "f32[8, 64, 1, 1]", relu_25: "f32[8, 64, 1, 1]", div_5: "f32[8, 2, 1, 128]", sum_18: "f32[8, 128, 32, 32]", convolution_34: "f32[8, 512, 32, 32]", squeeze_85: "f32[512]", relu_26: "f32[8, 512, 32, 32]", convolution_35: "f32[8, 128, 32, 32]", squeeze_88: "f32[128]", relu_27: "f32[8, 128, 32, 32]", convolution_36: "f32[8, 256, 32, 32]", squeeze_91: "f32[256]", relu_28: "f32[8, 256, 32, 32]", mean_6: "f32[8, 128, 1, 1]", convolution_37: "f32[8, 64, 1, 1]", relu_29: "f32[8, 64, 1, 1]", div_6: "f32[8, 2, 1, 128]", sum_21: "f32[8, 128, 32, 32]", convolution_39: "f32[8, 512, 32, 32]", squeeze_97: "f32[512]", relu_30: "f32[8, 512, 32, 32]", convolution_40: "f32[8, 256, 32, 32]", squeeze_100: "f32[256]", relu_31: "f32[8, 256, 32, 32]", convolution_41: "f32[8, 512, 32, 32]", squeeze_103: "f32[512]", relu_32: "f32[8, 512, 32, 32]", mean_7: "f32[8, 256, 1, 1]", convolution_42: "f32[8, 128, 1, 1]", relu_33: "f32[8, 128, 1, 1]", div_7: "f32[8, 2, 1, 256]", sum_24: "f32[8, 256, 32, 32]", avg_pool2d_2: "f32[8, 256, 16, 16]", convolution_44: "f32[8, 1024, 16, 16]", squeeze_109: "f32[1024]", avg_pool2d_3: "f32[8, 512, 16, 16]", convolution_45: "f32[8, 1024, 16, 16]", squeeze_112: "f32[1024]", relu_34: "f32[8, 1024, 16, 16]", convolution_46: "f32[8, 256, 16, 16]", squeeze_115: "f32[256]", relu_35: "f32[8, 256, 16, 16]", convolution_47: "f32[8, 512, 16, 16]", squeeze_118: "f32[512]", relu_36: "f32[8, 512, 16, 16]", mean_8: "f32[8, 256, 1, 1]", convolution_48: "f32[8, 128, 1, 1]", relu_37: "f32[8, 128, 1, 1]", div_8: "f32[8, 2, 1, 256]", sum_27: "f32[8, 256, 16, 16]", convolution_50: "f32[8, 1024, 16, 16]", squeeze_124: "f32[1024]", relu_38: "f32[8, 1024, 16, 16]", convolution_51: "f32[8, 256, 16, 16]", squeeze_127: "f32[256]", relu_39: "f32[8, 256, 16, 16]", convolution_52: "f32[8, 512, 16, 16]", squeeze_130: "f32[512]", relu_40: "f32[8, 512, 16, 16]", mean_9: "f32[8, 256, 1, 1]", convolution_53: "f32[8, 128, 1, 1]", relu_41: "f32[8, 128, 1, 1]", div_9: "f32[8, 2, 1, 256]", sum_30: "f32[8, 256, 16, 16]", convolution_55: "f32[8, 1024, 16, 16]", squeeze_136: "f32[1024]", relu_42: "f32[8, 1024, 16, 16]", convolution_56: "f32[8, 256, 16, 16]", squeeze_139: "f32[256]", relu_43: "f32[8, 256, 16, 16]", convolution_57: "f32[8, 512, 16, 16]", squeeze_142: "f32[512]", relu_44: "f32[8, 512, 16, 16]", mean_10: "f32[8, 256, 1, 1]", convolution_58: "f32[8, 128, 1, 1]", relu_45: "f32[8, 128, 1, 1]", div_10: "f32[8, 2, 1, 256]", sum_33: "f32[8, 256, 16, 16]", convolution_60: "f32[8, 1024, 16, 16]", squeeze_148: "f32[1024]", relu_46: "f32[8, 1024, 16, 16]", convolution_61: "f32[8, 256, 16, 16]", squeeze_151: "f32[256]", relu_47: "f32[8, 256, 16, 16]", convolution_62: "f32[8, 512, 16, 16]", squeeze_154: "f32[512]", relu_48: "f32[8, 512, 16, 16]", mean_11: "f32[8, 256, 1, 1]", convolution_63: "f32[8, 128, 1, 1]", relu_49: "f32[8, 128, 1, 1]", div_11: "f32[8, 2, 1, 256]", sum_36: "f32[8, 256, 16, 16]", convolution_65: "f32[8, 1024, 16, 16]", squeeze_160: "f32[1024]", relu_50: "f32[8, 1024, 16, 16]", convolution_66: "f32[8, 256, 16, 16]", squeeze_163: "f32[256]", relu_51: "f32[8, 256, 16, 16]", convolution_67: "f32[8, 512, 16, 16]", squeeze_166: "f32[512]", relu_52: "f32[8, 512, 16, 16]", mean_12: "f32[8, 256, 1, 1]", convolution_68: "f32[8, 128, 1, 1]", relu_53: "f32[8, 128, 1, 1]", div_12: "f32[8, 2, 1, 256]", sum_39: "f32[8, 256, 16, 16]", convolution_70: "f32[8, 1024, 16, 16]", squeeze_172: "f32[1024]", relu_54: "f32[8, 1024, 16, 16]", convolution_71: "f32[8, 256, 16, 16]", squeeze_175: "f32[256]", relu_55: "f32[8, 256, 16, 16]", convolution_72: "f32[8, 512, 16, 16]", squeeze_178: "f32[512]", relu_56: "f32[8, 512, 16, 16]", mean_13: "f32[8, 256, 1, 1]", convolution_73: "f32[8, 128, 1, 1]", relu_57: "f32[8, 128, 1, 1]", div_13: "f32[8, 2, 1, 256]", sum_42: "f32[8, 256, 16, 16]", convolution_75: "f32[8, 1024, 16, 16]", squeeze_184: "f32[1024]", relu_58: "f32[8, 1024, 16, 16]", convolution_76: "f32[8, 256, 16, 16]", squeeze_187: "f32[256]", relu_59: "f32[8, 256, 16, 16]", convolution_77: "f32[8, 512, 16, 16]", squeeze_190: "f32[512]", relu_60: "f32[8, 512, 16, 16]", mean_14: "f32[8, 256, 1, 1]", convolution_78: "f32[8, 128, 1, 1]", relu_61: "f32[8, 128, 1, 1]", div_14: "f32[8, 2, 1, 256]", sum_45: "f32[8, 256, 16, 16]", convolution_80: "f32[8, 1024, 16, 16]", squeeze_196: "f32[1024]", relu_62: "f32[8, 1024, 16, 16]", convolution_81: "f32[8, 256, 16, 16]", squeeze_199: "f32[256]", relu_63: "f32[8, 256, 16, 16]", convolution_82: "f32[8, 512, 16, 16]", squeeze_202: "f32[512]", relu_64: "f32[8, 512, 16, 16]", mean_15: "f32[8, 256, 1, 1]", convolution_83: "f32[8, 128, 1, 1]", relu_65: "f32[8, 128, 1, 1]", div_15: "f32[8, 2, 1, 256]", sum_48: "f32[8, 256, 16, 16]", convolution_85: "f32[8, 1024, 16, 16]", squeeze_208: "f32[1024]", relu_66: "f32[8, 1024, 16, 16]", convolution_86: "f32[8, 256, 16, 16]", squeeze_211: "f32[256]", relu_67: "f32[8, 256, 16, 16]", convolution_87: "f32[8, 512, 16, 16]", squeeze_214: "f32[512]", relu_68: "f32[8, 512, 16, 16]", mean_16: "f32[8, 256, 1, 1]", convolution_88: "f32[8, 128, 1, 1]", relu_69: "f32[8, 128, 1, 1]", div_16: "f32[8, 2, 1, 256]", sum_51: "f32[8, 256, 16, 16]", convolution_90: "f32[8, 1024, 16, 16]", squeeze_220: "f32[1024]", relu_70: "f32[8, 1024, 16, 16]", convolution_91: "f32[8, 256, 16, 16]", squeeze_223: "f32[256]", relu_71: "f32[8, 256, 16, 16]", convolution_92: "f32[8, 512, 16, 16]", squeeze_226: "f32[512]", relu_72: "f32[8, 512, 16, 16]", mean_17: "f32[8, 256, 1, 1]", convolution_93: "f32[8, 128, 1, 1]", relu_73: "f32[8, 128, 1, 1]", div_17: "f32[8, 2, 1, 256]", sum_54: "f32[8, 256, 16, 16]", convolution_95: "f32[8, 1024, 16, 16]", squeeze_232: "f32[1024]", relu_74: "f32[8, 1024, 16, 16]", convolution_96: "f32[8, 256, 16, 16]", squeeze_235: "f32[256]", relu_75: "f32[8, 256, 16, 16]", convolution_97: "f32[8, 512, 16, 16]", squeeze_238: "f32[512]", relu_76: "f32[8, 512, 16, 16]", mean_18: "f32[8, 256, 1, 1]", convolution_98: "f32[8, 128, 1, 1]", relu_77: "f32[8, 128, 1, 1]", div_18: "f32[8, 2, 1, 256]", sum_57: "f32[8, 256, 16, 16]", convolution_100: "f32[8, 1024, 16, 16]", squeeze_244: "f32[1024]", relu_78: "f32[8, 1024, 16, 16]", convolution_101: "f32[8, 256, 16, 16]", squeeze_247: "f32[256]", relu_79: "f32[8, 256, 16, 16]", convolution_102: "f32[8, 512, 16, 16]", squeeze_250: "f32[512]", relu_80: "f32[8, 512, 16, 16]", mean_19: "f32[8, 256, 1, 1]", convolution_103: "f32[8, 128, 1, 1]", relu_81: "f32[8, 128, 1, 1]", div_19: "f32[8, 2, 1, 256]", sum_60: "f32[8, 256, 16, 16]", convolution_105: "f32[8, 1024, 16, 16]", squeeze_256: "f32[1024]", relu_82: "f32[8, 1024, 16, 16]", convolution_106: "f32[8, 256, 16, 16]", squeeze_259: "f32[256]", relu_83: "f32[8, 256, 16, 16]", convolution_107: "f32[8, 512, 16, 16]", squeeze_262: "f32[512]", relu_84: "f32[8, 512, 16, 16]", mean_20: "f32[8, 256, 1, 1]", convolution_108: "f32[8, 128, 1, 1]", relu_85: "f32[8, 128, 1, 1]", div_20: "f32[8, 2, 1, 256]", sum_63: "f32[8, 256, 16, 16]", convolution_110: "f32[8, 1024, 16, 16]", squeeze_268: "f32[1024]", relu_86: "f32[8, 1024, 16, 16]", convolution_111: "f32[8, 256, 16, 16]", squeeze_271: "f32[256]", relu_87: "f32[8, 256, 16, 16]", convolution_112: "f32[8, 512, 16, 16]", squeeze_274: "f32[512]", relu_88: "f32[8, 512, 16, 16]", mean_21: "f32[8, 256, 1, 1]", convolution_113: "f32[8, 128, 1, 1]", relu_89: "f32[8, 128, 1, 1]", div_21: "f32[8, 2, 1, 256]", sum_66: "f32[8, 256, 16, 16]", convolution_115: "f32[8, 1024, 16, 16]", squeeze_280: "f32[1024]", relu_90: "f32[8, 1024, 16, 16]", convolution_116: "f32[8, 256, 16, 16]", squeeze_283: "f32[256]", relu_91: "f32[8, 256, 16, 16]", convolution_117: "f32[8, 512, 16, 16]", squeeze_286: "f32[512]", relu_92: "f32[8, 512, 16, 16]", mean_22: "f32[8, 256, 1, 1]", convolution_118: "f32[8, 128, 1, 1]", relu_93: "f32[8, 128, 1, 1]", div_22: "f32[8, 2, 1, 256]", sum_69: "f32[8, 256, 16, 16]", convolution_120: "f32[8, 1024, 16, 16]", squeeze_292: "f32[1024]", relu_94: "f32[8, 1024, 16, 16]", convolution_121: "f32[8, 256, 16, 16]", squeeze_295: "f32[256]", relu_95: "f32[8, 256, 16, 16]", convolution_122: "f32[8, 512, 16, 16]", squeeze_298: "f32[512]", relu_96: "f32[8, 512, 16, 16]", mean_23: "f32[8, 256, 1, 1]", convolution_123: "f32[8, 128, 1, 1]", relu_97: "f32[8, 128, 1, 1]", div_23: "f32[8, 2, 1, 256]", sum_72: "f32[8, 256, 16, 16]", convolution_125: "f32[8, 1024, 16, 16]", squeeze_304: "f32[1024]", relu_98: "f32[8, 1024, 16, 16]", convolution_126: "f32[8, 256, 16, 16]", squeeze_307: "f32[256]", relu_99: "f32[8, 256, 16, 16]", convolution_127: "f32[8, 512, 16, 16]", squeeze_310: "f32[512]", relu_100: "f32[8, 512, 16, 16]", mean_24: "f32[8, 256, 1, 1]", convolution_128: "f32[8, 128, 1, 1]", relu_101: "f32[8, 128, 1, 1]", div_24: "f32[8, 2, 1, 256]", sum_75: "f32[8, 256, 16, 16]", convolution_130: "f32[8, 1024, 16, 16]", squeeze_316: "f32[1024]", relu_102: "f32[8, 1024, 16, 16]", convolution_131: "f32[8, 256, 16, 16]", squeeze_319: "f32[256]", relu_103: "f32[8, 256, 16, 16]", convolution_132: "f32[8, 512, 16, 16]", squeeze_322: "f32[512]", relu_104: "f32[8, 512, 16, 16]", mean_25: "f32[8, 256, 1, 1]", convolution_133: "f32[8, 128, 1, 1]", relu_105: "f32[8, 128, 1, 1]", div_25: "f32[8, 2, 1, 256]", sum_78: "f32[8, 256, 16, 16]", convolution_135: "f32[8, 1024, 16, 16]", squeeze_328: "f32[1024]", relu_106: "f32[8, 1024, 16, 16]", convolution_136: "f32[8, 256, 16, 16]", squeeze_331: "f32[256]", relu_107: "f32[8, 256, 16, 16]", convolution_137: "f32[8, 512, 16, 16]", squeeze_334: "f32[512]", relu_108: "f32[8, 512, 16, 16]", mean_26: "f32[8, 256, 1, 1]", convolution_138: "f32[8, 128, 1, 1]", relu_109: "f32[8, 128, 1, 1]", div_26: "f32[8, 2, 1, 256]", sum_81: "f32[8, 256, 16, 16]", convolution_140: "f32[8, 1024, 16, 16]", squeeze_340: "f32[1024]", relu_110: "f32[8, 1024, 16, 16]", convolution_141: "f32[8, 256, 16, 16]", squeeze_343: "f32[256]", relu_111: "f32[8, 256, 16, 16]", convolution_142: "f32[8, 512, 16, 16]", squeeze_346: "f32[512]", relu_112: "f32[8, 512, 16, 16]", mean_27: "f32[8, 256, 1, 1]", convolution_143: "f32[8, 128, 1, 1]", relu_113: "f32[8, 128, 1, 1]", div_27: "f32[8, 2, 1, 256]", sum_84: "f32[8, 256, 16, 16]", convolution_145: "f32[8, 1024, 16, 16]", squeeze_352: "f32[1024]", relu_114: "f32[8, 1024, 16, 16]", convolution_146: "f32[8, 256, 16, 16]", squeeze_355: "f32[256]", relu_115: "f32[8, 256, 16, 16]", convolution_147: "f32[8, 512, 16, 16]", squeeze_358: "f32[512]", relu_116: "f32[8, 512, 16, 16]", mean_28: "f32[8, 256, 1, 1]", convolution_148: "f32[8, 128, 1, 1]", relu_117: "f32[8, 128, 1, 1]", div_28: "f32[8, 2, 1, 256]", sum_87: "f32[8, 256, 16, 16]", convolution_150: "f32[8, 1024, 16, 16]", squeeze_364: "f32[1024]", relu_118: "f32[8, 1024, 16, 16]", convolution_151: "f32[8, 256, 16, 16]", squeeze_367: "f32[256]", relu_119: "f32[8, 256, 16, 16]", convolution_152: "f32[8, 512, 16, 16]", squeeze_370: "f32[512]", relu_120: "f32[8, 512, 16, 16]", mean_29: "f32[8, 256, 1, 1]", convolution_153: "f32[8, 128, 1, 1]", relu_121: "f32[8, 128, 1, 1]", div_29: "f32[8, 2, 1, 256]", sum_90: "f32[8, 256, 16, 16]", convolution_155: "f32[8, 1024, 16, 16]", squeeze_376: "f32[1024]", relu_122: "f32[8, 1024, 16, 16]", convolution_156: "f32[8, 512, 16, 16]", squeeze_379: "f32[512]", relu_123: "f32[8, 512, 16, 16]", convolution_157: "f32[8, 1024, 16, 16]", squeeze_382: "f32[1024]", relu_124: "f32[8, 1024, 16, 16]", mean_30: "f32[8, 512, 1, 1]", convolution_158: "f32[8, 256, 1, 1]", relu_125: "f32[8, 256, 1, 1]", div_30: "f32[8, 2, 1, 512]", sum_93: "f32[8, 512, 16, 16]", avg_pool2d_4: "f32[8, 512, 8, 8]", convolution_160: "f32[8, 2048, 8, 8]", squeeze_388: "f32[2048]", avg_pool2d_5: "f32[8, 1024, 8, 8]", convolution_161: "f32[8, 2048, 8, 8]", squeeze_391: "f32[2048]", relu_126: "f32[8, 2048, 8, 8]", convolution_162: "f32[8, 512, 8, 8]", squeeze_394: "f32[512]", relu_127: "f32[8, 512, 8, 8]", convolution_163: "f32[8, 1024, 8, 8]", squeeze_397: "f32[1024]", relu_128: "f32[8, 1024, 8, 8]", mean_31: "f32[8, 512, 1, 1]", convolution_164: "f32[8, 256, 1, 1]", relu_129: "f32[8, 256, 1, 1]", div_31: "f32[8, 2, 1, 512]", sum_96: "f32[8, 512, 8, 8]", convolution_166: "f32[8, 2048, 8, 8]", squeeze_403: "f32[2048]", relu_130: "f32[8, 2048, 8, 8]", convolution_167: "f32[8, 512, 8, 8]", squeeze_406: "f32[512]", relu_131: "f32[8, 512, 8, 8]", convolution_168: "f32[8, 1024, 8, 8]", squeeze_409: "f32[1024]", relu_132: "f32[8, 1024, 8, 8]", mean_32: "f32[8, 512, 1, 1]", convolution_169: "f32[8, 256, 1, 1]", relu_133: "f32[8, 256, 1, 1]", div_32: "f32[8, 2, 1, 512]", sum_99: "f32[8, 512, 8, 8]", convolution_171: "f32[8, 2048, 8, 8]", squeeze_415: "f32[2048]", view_198: "f32[8, 2048]", permute_34: "f32[1000, 2048]", le: "b8[8, 2048, 8, 8]", unsqueeze_558: "f32[1, 2048, 1, 1]", unsqueeze_584: "f32[1, 1024, 1, 1]", unsqueeze_596: "f32[1, 512, 1, 1]", unsqueeze_608: "f32[1, 2048, 1, 1]", unsqueeze_634: "f32[1, 1024, 1, 1]", unsqueeze_646: "f32[1, 512, 1, 1]", unsqueeze_658: "f32[1, 2048, 1, 1]", unsqueeze_670: "f32[1, 2048, 1, 1]", unsqueeze_696: "f32[1, 1024, 1, 1]", unsqueeze_708: "f32[1, 512, 1, 1]", unsqueeze_720: "f32[1, 1024, 1, 1]", unsqueeze_746: "f32[1, 512, 1, 1]", unsqueeze_758: "f32[1, 256, 1, 1]", unsqueeze_770: "f32[1, 1024, 1, 1]", unsqueeze_796: "f32[1, 512, 1, 1]", unsqueeze_808: "f32[1, 256, 1, 1]", unsqueeze_820: "f32[1, 1024, 1, 1]", unsqueeze_846: "f32[1, 512, 1, 1]", unsqueeze_858: "f32[1, 256, 1, 1]", unsqueeze_870: "f32[1, 1024, 1, 1]", unsqueeze_896: "f32[1, 512, 1, 1]", unsqueeze_908: "f32[1, 256, 1, 1]", unsqueeze_920: "f32[1, 1024, 1, 1]", unsqueeze_946: "f32[1, 512, 1, 1]", unsqueeze_958: "f32[1, 256, 1, 1]", unsqueeze_970: "f32[1, 1024, 1, 1]", unsqueeze_996: "f32[1, 512, 1, 1]", unsqueeze_1008: "f32[1, 256, 1, 1]", unsqueeze_1020: "f32[1, 1024, 1, 1]", unsqueeze_1046: "f32[1, 512, 1, 1]", unsqueeze_1058: "f32[1, 256, 1, 1]", unsqueeze_1070: "f32[1, 1024, 1, 1]", unsqueeze_1096: "f32[1, 512, 1, 1]", unsqueeze_1108: "f32[1, 256, 1, 1]", unsqueeze_1120: "f32[1, 1024, 1, 1]", unsqueeze_1146: "f32[1, 512, 1, 1]", unsqueeze_1158: "f32[1, 256, 1, 1]", unsqueeze_1170: "f32[1, 1024, 1, 1]", unsqueeze_1196: "f32[1, 512, 1, 1]", unsqueeze_1208: "f32[1, 256, 1, 1]", unsqueeze_1220: "f32[1, 1024, 1, 1]", unsqueeze_1246: "f32[1, 512, 1, 1]", unsqueeze_1258: "f32[1, 256, 1, 1]", unsqueeze_1270: "f32[1, 1024, 1, 1]", unsqueeze_1296: "f32[1, 512, 1, 1]", unsqueeze_1308: "f32[1, 256, 1, 1]", unsqueeze_1320: "f32[1, 1024, 1, 1]", unsqueeze_1346: "f32[1, 512, 1, 1]", unsqueeze_1358: "f32[1, 256, 1, 1]", unsqueeze_1370: "f32[1, 1024, 1, 1]", unsqueeze_1396: "f32[1, 512, 1, 1]", unsqueeze_1408: "f32[1, 256, 1, 1]", unsqueeze_1420: "f32[1, 1024, 1, 1]", unsqueeze_1446: "f32[1, 512, 1, 1]", unsqueeze_1458: "f32[1, 256, 1, 1]", unsqueeze_1470: "f32[1, 1024, 1, 1]", unsqueeze_1496: "f32[1, 512, 1, 1]", unsqueeze_1508: "f32[1, 256, 1, 1]", unsqueeze_1520: "f32[1, 1024, 1, 1]", unsqueeze_1546: "f32[1, 512, 1, 1]", unsqueeze_1558: "f32[1, 256, 1, 1]", unsqueeze_1570: "f32[1, 1024, 1, 1]", unsqueeze_1596: "f32[1, 512, 1, 1]", unsqueeze_1608: "f32[1, 256, 1, 1]", unsqueeze_1620: "f32[1, 1024, 1, 1]", unsqueeze_1646: "f32[1, 512, 1, 1]", unsqueeze_1658: "f32[1, 256, 1, 1]", unsqueeze_1670: "f32[1, 1024, 1, 1]", unsqueeze_1696: "f32[1, 512, 1, 1]", unsqueeze_1708: "f32[1, 256, 1, 1]", unsqueeze_1720: "f32[1, 1024, 1, 1]", unsqueeze_1746: "f32[1, 512, 1, 1]", unsqueeze_1758: "f32[1, 256, 1, 1]", unsqueeze_1770: "f32[1, 1024, 1, 1]", unsqueeze_1796: "f32[1, 512, 1, 1]", unsqueeze_1808: "f32[1, 256, 1, 1]", unsqueeze_1820: "f32[1, 1024, 1, 1]", unsqueeze_1832: "f32[1, 1024, 1, 1]", unsqueeze_1858: "f32[1, 512, 1, 1]", unsqueeze_1870: "f32[1, 256, 1, 1]", unsqueeze_1882: "f32[1, 512, 1, 1]", unsqueeze_1908: "f32[1, 256, 1, 1]", unsqueeze_1920: "f32[1, 128, 1, 1]", unsqueeze_1932: "f32[1, 512, 1, 1]", unsqueeze_1958: "f32[1, 256, 1, 1]", unsqueeze_1970: "f32[1, 128, 1, 1]", unsqueeze_1982: "f32[1, 512, 1, 1]", unsqueeze_2008: "f32[1, 256, 1, 1]", unsqueeze_2020: "f32[1, 128, 1, 1]", unsqueeze_2032: "f32[1, 512, 1, 1]", unsqueeze_2044: "f32[1, 512, 1, 1]", unsqueeze_2070: "f32[1, 256, 1, 1]", unsqueeze_2082: "f32[1, 128, 1, 1]", unsqueeze_2094: "f32[1, 256, 1, 1]", unsqueeze_2120: "f32[1, 128, 1, 1]", unsqueeze_2132: "f32[1, 64, 1, 1]", unsqueeze_2144: "f32[1, 256, 1, 1]", unsqueeze_2170: "f32[1, 128, 1, 1]", unsqueeze_2182: "f32[1, 64, 1, 1]", unsqueeze_2194: "f32[1, 256, 1, 1]", unsqueeze_2206: "f32[1, 256, 1, 1]", unsqueeze_2232: "f32[1, 128, 1, 1]", unsqueeze_2244: "f32[1, 64, 1, 1]", unsqueeze_2256: "f32[1, 128, 1, 1]", unsqueeze_2268: "f32[1, 64, 1, 1]", unsqueeze_2280: "f32[1, 64, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_1: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.view.default(relu_4, [8, 2, 64, 64, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 32, 1, 1]" = var_mean_5[0]
    getitem_13: "f32[1, 32, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_5: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    squeeze_15: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_16: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_6: "f32[8, 2, 1, 64]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_3: "f32[8, 128]" = torch.ops.aten.view.default(div, [8, -1]);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_4: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(view_3, [8, -1, 1, 1]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_5: "f32[8, 2, 64, 1, 1]" = torch.ops.aten.view.default(view_4, [8, 2, 64, 1, 1]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_7: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.view.default(relu_8, [8, 2, 64, 64, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 32, 1, 1]" = var_mean_10[0]
    getitem_23: "f32[1, 32, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_52: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_10: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    squeeze_30: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_31: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_11: "f32[8, 2, 1, 64]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_9: "f32[8, 128]" = torch.ops.aten.view.default(div_1, [8, -1]);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_10: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(view_9, [8, -1, 1, 1]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_11: "f32[8, 2, 64, 1, 1]" = torch.ops.aten.view.default(view_10, [8, 2, 64, 1, 1]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_13: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.view.default(relu_12, [8, 2, 64, 64, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 32, 1, 1]" = var_mean_14[0]
    getitem_31: "f32[1, 32, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_73: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_14: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    squeeze_42: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_43: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_16: "f32[8, 2, 1, 64]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_15: "f32[8, 128]" = torch.ops.aten.view.default(div_2, [8, -1]);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_16: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(view_15, [8, -1, 1, 1]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_17: "f32[8, 2, 64, 1, 1]" = torch.ops.aten.view.default(view_16, [8, 2, 64, 1, 1]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_19: "f32[8, 2, 128, 64, 64]" = torch.ops.aten.view.default(relu_16, [8, 2, 128, 64, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 64, 1, 1]" = var_mean_18[0]
    getitem_39: "f32[1, 64, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_94: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_18: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    squeeze_54: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_55: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_21: "f32[8, 2, 1, 128]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_21: "f32[8, 256]" = torch.ops.aten.view.default(div_3, [8, -1]);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_22: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(view_21, [8, -1, 1, 1]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_23: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.view.default(view_22, [8, 2, 128, 1, 1]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_25: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.view.default(relu_20, [8, 2, 128, 32, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 64, 1, 1]" = var_mean_23[0]
    getitem_49: "f32[1, 64, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_120: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_23: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    squeeze_69: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_70: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_26: "f32[8, 2, 1, 128]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_27: "f32[8, 256]" = torch.ops.aten.view.default(div_4, [8, -1]);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_28: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(view_27, [8, -1, 1, 1]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_29: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.view.default(view_28, [8, 2, 128, 1, 1]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_31: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.view.default(relu_24, [8, 2, 128, 32, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 64, 1, 1]" = var_mean_27[0]
    getitem_57: "f32[1, 64, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_141: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_27: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    squeeze_81: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_82: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_31: "f32[8, 2, 1, 128]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_33: "f32[8, 256]" = torch.ops.aten.view.default(div_5, [8, -1]);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_34: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(view_33, [8, -1, 1, 1]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_35: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.view.default(view_34, [8, 2, 128, 1, 1]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_37: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.view.default(relu_28, [8, 2, 128, 32, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 64, 1, 1]" = var_mean_31[0]
    getitem_65: "f32[1, 64, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_162: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_31: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    squeeze_93: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_94: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_36: "f32[8, 2, 1, 128]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_39: "f32[8, 256]" = torch.ops.aten.view.default(div_6, [8, -1]);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_40: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(view_39, [8, -1, 1, 1]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_41: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.view.default(view_40, [8, 2, 128, 1, 1]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_43: "f32[8, 2, 256, 32, 32]" = torch.ops.aten.view.default(relu_32, [8, 2, 256, 32, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1, 1]" = var_mean_35[0]
    getitem_73: "f32[1, 128, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_183: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_35: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    squeeze_105: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_106: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_41: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_45: "f32[8, 512]" = torch.ops.aten.view.default(div_7, [8, -1]);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_46: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_45, [8, -1, 1, 1]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_47: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_46, [8, 2, 256, 1, 1]);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_49: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_36, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1, 1]" = var_mean_40[0]
    getitem_83: "f32[1, 128, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_209: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_40: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
    squeeze_120: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_121: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_46: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_51: "f32[8, 512]" = torch.ops.aten.view.default(div_8, [8, -1]);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_52: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_51, [8, -1, 1, 1]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_53: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_52, [8, 2, 256, 1, 1]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_55: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_40, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 128, 1, 1]" = var_mean_44[0]
    getitem_91: "f32[1, 128, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_230: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_44: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_230);  add_230 = None
    squeeze_132: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_133: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_51: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_57: "f32[8, 512]" = torch.ops.aten.view.default(div_9, [8, -1]);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_58: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_57, [8, -1, 1, 1]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_59: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_58, [8, 2, 256, 1, 1]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_61: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_44, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 128, 1, 1]" = var_mean_48[0]
    getitem_99: "f32[1, 128, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_251: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_48: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
    squeeze_144: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_145: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_56: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_63: "f32[8, 512]" = torch.ops.aten.view.default(div_10, [8, -1]);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_64: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_63, [8, -1, 1, 1]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_65: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_64, [8, 2, 256, 1, 1]);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_67: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_48, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 128, 1, 1]" = var_mean_52[0]
    getitem_107: "f32[1, 128, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_272: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_52: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
    squeeze_156: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_157: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_61: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_69: "f32[8, 512]" = torch.ops.aten.view.default(div_11, [8, -1]);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_70: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_69, [8, -1, 1, 1]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_71: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_70, [8, 2, 256, 1, 1]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_73: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_52, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 128, 1, 1]" = var_mean_56[0]
    getitem_115: "f32[1, 128, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_293: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_56: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
    squeeze_168: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_169: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_66: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_75: "f32[8, 512]" = torch.ops.aten.view.default(div_12, [8, -1]);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_76: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_75, [8, -1, 1, 1]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_77: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_76, [8, 2, 256, 1, 1]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_79: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_56, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_73, [0, 2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[1, 128, 1, 1]" = var_mean_60[0]
    getitem_123: "f32[1, 128, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_314: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
    rsqrt_60: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
    squeeze_180: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    squeeze_181: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_71: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_81: "f32[8, 512]" = torch.ops.aten.view.default(div_13, [8, -1]);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_82: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_81, [8, -1, 1, 1]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_83: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_82, [8, 2, 256, 1, 1]);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_85: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_60, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_64 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
    getitem_130: "f32[1, 128, 1, 1]" = var_mean_64[0]
    getitem_131: "f32[1, 128, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_335: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
    rsqrt_64: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_335);  add_335 = None
    squeeze_192: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
    squeeze_193: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_76: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_87: "f32[8, 512]" = torch.ops.aten.view.default(div_14, [8, -1]);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_88: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_87, [8, -1, 1, 1]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_89: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_88, [8, 2, 256, 1, 1]);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_91: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_64, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_68 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
    getitem_138: "f32[1, 128, 1, 1]" = var_mean_68[0]
    getitem_139: "f32[1, 128, 1, 1]" = var_mean_68[1];  var_mean_68 = None
    add_356: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
    rsqrt_68: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_356);  add_356 = None
    squeeze_204: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
    squeeze_205: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_81: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_93: "f32[8, 512]" = torch.ops.aten.view.default(div_15, [8, -1]);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_94: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_93, [8, -1, 1, 1]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_95: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_94, [8, 2, 256, 1, 1]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_97: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_68, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_72 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
    getitem_146: "f32[1, 128, 1, 1]" = var_mean_72[0]
    getitem_147: "f32[1, 128, 1, 1]" = var_mean_72[1];  var_mean_72 = None
    add_377: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
    rsqrt_72: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_377);  add_377 = None
    squeeze_216: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
    squeeze_217: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_86: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_99: "f32[8, 512]" = torch.ops.aten.view.default(div_16, [8, -1]);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_100: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_99, [8, -1, 1, 1]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_101: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_100, [8, 2, 256, 1, 1]);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_103: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_72, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_76 = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True)
    getitem_154: "f32[1, 128, 1, 1]" = var_mean_76[0]
    getitem_155: "f32[1, 128, 1, 1]" = var_mean_76[1];  var_mean_76 = None
    add_398: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
    rsqrt_76: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_398);  add_398 = None
    squeeze_228: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
    squeeze_229: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_91: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_105: "f32[8, 512]" = torch.ops.aten.view.default(div_17, [8, -1]);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_106: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_105, [8, -1, 1, 1]);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_107: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_106, [8, 2, 256, 1, 1]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_109: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_76, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_80 = torch.ops.aten.var_mean.correction(convolution_98, [0, 2, 3], correction = 0, keepdim = True)
    getitem_162: "f32[1, 128, 1, 1]" = var_mean_80[0]
    getitem_163: "f32[1, 128, 1, 1]" = var_mean_80[1];  var_mean_80 = None
    add_419: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
    rsqrt_80: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_419);  add_419 = None
    squeeze_240: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
    squeeze_241: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_96: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_111: "f32[8, 512]" = torch.ops.aten.view.default(div_18, [8, -1]);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_112: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_111, [8, -1, 1, 1]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_113: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_112, [8, 2, 256, 1, 1]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_115: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_80, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_84 = torch.ops.aten.var_mean.correction(convolution_103, [0, 2, 3], correction = 0, keepdim = True)
    getitem_170: "f32[1, 128, 1, 1]" = var_mean_84[0]
    getitem_171: "f32[1, 128, 1, 1]" = var_mean_84[1];  var_mean_84 = None
    add_440: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-05);  getitem_170 = None
    rsqrt_84: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_440);  add_440 = None
    squeeze_252: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
    squeeze_253: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_101: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_117: "f32[8, 512]" = torch.ops.aten.view.default(div_19, [8, -1]);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_118: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_117, [8, -1, 1, 1]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_119: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_118, [8, 2, 256, 1, 1]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_121: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_84, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_88 = torch.ops.aten.var_mean.correction(convolution_108, [0, 2, 3], correction = 0, keepdim = True)
    getitem_178: "f32[1, 128, 1, 1]" = var_mean_88[0]
    getitem_179: "f32[1, 128, 1, 1]" = var_mean_88[1];  var_mean_88 = None
    add_461: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
    rsqrt_88: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_461);  add_461 = None
    squeeze_264: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
    squeeze_265: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_106: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_123: "f32[8, 512]" = torch.ops.aten.view.default(div_20, [8, -1]);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_124: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_123, [8, -1, 1, 1]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_125: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_124, [8, 2, 256, 1, 1]);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_127: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_88, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_92 = torch.ops.aten.var_mean.correction(convolution_113, [0, 2, 3], correction = 0, keepdim = True)
    getitem_186: "f32[1, 128, 1, 1]" = var_mean_92[0]
    getitem_187: "f32[1, 128, 1, 1]" = var_mean_92[1];  var_mean_92 = None
    add_482: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
    rsqrt_92: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_482);  add_482 = None
    squeeze_276: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
    squeeze_277: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_92, [0, 2, 3]);  rsqrt_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_111: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_129: "f32[8, 512]" = torch.ops.aten.view.default(div_21, [8, -1]);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_130: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_129, [8, -1, 1, 1]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_131: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_130, [8, 2, 256, 1, 1]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_133: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_92, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_96 = torch.ops.aten.var_mean.correction(convolution_118, [0, 2, 3], correction = 0, keepdim = True)
    getitem_194: "f32[1, 128, 1, 1]" = var_mean_96[0]
    getitem_195: "f32[1, 128, 1, 1]" = var_mean_96[1];  var_mean_96 = None
    add_503: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-05);  getitem_194 = None
    rsqrt_96: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_503);  add_503 = None
    squeeze_288: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3]);  getitem_195 = None
    squeeze_289: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_96, [0, 2, 3]);  rsqrt_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_116: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_135: "f32[8, 512]" = torch.ops.aten.view.default(div_22, [8, -1]);  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_136: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_135, [8, -1, 1, 1]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_137: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_136, [8, 2, 256, 1, 1]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_139: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_96, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_100 = torch.ops.aten.var_mean.correction(convolution_123, [0, 2, 3], correction = 0, keepdim = True)
    getitem_202: "f32[1, 128, 1, 1]" = var_mean_100[0]
    getitem_203: "f32[1, 128, 1, 1]" = var_mean_100[1];  var_mean_100 = None
    add_524: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_202, 1e-05);  getitem_202 = None
    rsqrt_100: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_524);  add_524 = None
    squeeze_300: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_203, [0, 2, 3]);  getitem_203 = None
    squeeze_301: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_100, [0, 2, 3]);  rsqrt_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_121: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_141: "f32[8, 512]" = torch.ops.aten.view.default(div_23, [8, -1]);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_142: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_141, [8, -1, 1, 1]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_143: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_142, [8, 2, 256, 1, 1]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_145: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_100, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_104 = torch.ops.aten.var_mean.correction(convolution_128, [0, 2, 3], correction = 0, keepdim = True)
    getitem_210: "f32[1, 128, 1, 1]" = var_mean_104[0]
    getitem_211: "f32[1, 128, 1, 1]" = var_mean_104[1];  var_mean_104 = None
    add_545: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
    rsqrt_104: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_545);  add_545 = None
    squeeze_312: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_211, [0, 2, 3]);  getitem_211 = None
    squeeze_313: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_104, [0, 2, 3]);  rsqrt_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_126: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_147: "f32[8, 512]" = torch.ops.aten.view.default(div_24, [8, -1]);  div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_148: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_147, [8, -1, 1, 1]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_149: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_148, [8, 2, 256, 1, 1]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_151: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_104, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_108 = torch.ops.aten.var_mean.correction(convolution_133, [0, 2, 3], correction = 0, keepdim = True)
    getitem_218: "f32[1, 128, 1, 1]" = var_mean_108[0]
    getitem_219: "f32[1, 128, 1, 1]" = var_mean_108[1];  var_mean_108 = None
    add_566: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-05);  getitem_218 = None
    rsqrt_108: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_566);  add_566 = None
    squeeze_324: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_219, [0, 2, 3]);  getitem_219 = None
    squeeze_325: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_108, [0, 2, 3]);  rsqrt_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_131: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_153: "f32[8, 512]" = torch.ops.aten.view.default(div_25, [8, -1]);  div_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_154: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_153, [8, -1, 1, 1]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_155: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_154, [8, 2, 256, 1, 1]);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_157: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_108, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_112 = torch.ops.aten.var_mean.correction(convolution_138, [0, 2, 3], correction = 0, keepdim = True)
    getitem_226: "f32[1, 128, 1, 1]" = var_mean_112[0]
    getitem_227: "f32[1, 128, 1, 1]" = var_mean_112[1];  var_mean_112 = None
    add_587: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-05);  getitem_226 = None
    rsqrt_112: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_587);  add_587 = None
    squeeze_336: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_227, [0, 2, 3]);  getitem_227 = None
    squeeze_337: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_112, [0, 2, 3]);  rsqrt_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_136: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_159: "f32[8, 512]" = torch.ops.aten.view.default(div_26, [8, -1]);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_160: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_159, [8, -1, 1, 1]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_161: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_160, [8, 2, 256, 1, 1]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_163: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_112, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_116 = torch.ops.aten.var_mean.correction(convolution_143, [0, 2, 3], correction = 0, keepdim = True)
    getitem_234: "f32[1, 128, 1, 1]" = var_mean_116[0]
    getitem_235: "f32[1, 128, 1, 1]" = var_mean_116[1];  var_mean_116 = None
    add_608: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_234, 1e-05);  getitem_234 = None
    rsqrt_116: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_608);  add_608 = None
    squeeze_348: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_235, [0, 2, 3]);  getitem_235 = None
    squeeze_349: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_116, [0, 2, 3]);  rsqrt_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_141: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_165: "f32[8, 512]" = torch.ops.aten.view.default(div_27, [8, -1]);  div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_166: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_165, [8, -1, 1, 1]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_167: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_166, [8, 2, 256, 1, 1]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_169: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_116, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_120 = torch.ops.aten.var_mean.correction(convolution_148, [0, 2, 3], correction = 0, keepdim = True)
    getitem_242: "f32[1, 128, 1, 1]" = var_mean_120[0]
    getitem_243: "f32[1, 128, 1, 1]" = var_mean_120[1];  var_mean_120 = None
    add_629: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-05);  getitem_242 = None
    rsqrt_120: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_629);  add_629 = None
    squeeze_360: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_243, [0, 2, 3]);  getitem_243 = None
    squeeze_361: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_120, [0, 2, 3]);  rsqrt_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_146: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_171: "f32[8, 512]" = torch.ops.aten.view.default(div_28, [8, -1]);  div_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_172: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_171, [8, -1, 1, 1]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_173: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_172, [8, 2, 256, 1, 1]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_175: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.view.default(relu_120, [8, 2, 256, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_124 = torch.ops.aten.var_mean.correction(convolution_153, [0, 2, 3], correction = 0, keepdim = True)
    getitem_250: "f32[1, 128, 1, 1]" = var_mean_124[0]
    getitem_251: "f32[1, 128, 1, 1]" = var_mean_124[1];  var_mean_124 = None
    add_650: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_250, 1e-05);  getitem_250 = None
    rsqrt_124: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_650);  add_650 = None
    squeeze_372: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_251, [0, 2, 3]);  getitem_251 = None
    squeeze_373: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_124, [0, 2, 3]);  rsqrt_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_151: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(div_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_177: "f32[8, 512]" = torch.ops.aten.view.default(div_29, [8, -1]);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_178: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(view_177, [8, -1, 1, 1]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_179: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.view.default(view_178, [8, 2, 256, 1, 1]);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_181: "f32[8, 2, 512, 16, 16]" = torch.ops.aten.view.default(relu_124, [8, 2, 512, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_128 = torch.ops.aten.var_mean.correction(convolution_158, [0, 2, 3], correction = 0, keepdim = True)
    getitem_258: "f32[1, 256, 1, 1]" = var_mean_128[0]
    getitem_259: "f32[1, 256, 1, 1]" = var_mean_128[1];  var_mean_128 = None
    add_671: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_258, 1e-05);  getitem_258 = None
    rsqrt_128: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_671);  add_671 = None
    squeeze_384: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_259, [0, 2, 3]);  getitem_259 = None
    squeeze_385: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_128, [0, 2, 3]);  rsqrt_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_156: "f32[8, 2, 1, 512]" = torch.ops.aten.alias.default(div_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_183: "f32[8, 1024]" = torch.ops.aten.view.default(div_30, [8, -1]);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_184: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(view_183, [8, -1, 1, 1]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_185: "f32[8, 2, 512, 1, 1]" = torch.ops.aten.view.default(view_184, [8, 2, 512, 1, 1]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_187: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.view.default(relu_128, [8, 2, 512, 8, 8])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_133 = torch.ops.aten.var_mean.correction(convolution_164, [0, 2, 3], correction = 0, keepdim = True)
    getitem_268: "f32[1, 256, 1, 1]" = var_mean_133[0]
    getitem_269: "f32[1, 256, 1, 1]" = var_mean_133[1];  var_mean_133 = None
    add_697: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_268, 1e-05);  getitem_268 = None
    rsqrt_133: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_697);  add_697 = None
    squeeze_399: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_269, [0, 2, 3]);  getitem_269 = None
    squeeze_400: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_133, [0, 2, 3]);  rsqrt_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_161: "f32[8, 2, 1, 512]" = torch.ops.aten.alias.default(div_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_189: "f32[8, 1024]" = torch.ops.aten.view.default(div_31, [8, -1]);  div_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_190: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(view_189, [8, -1, 1, 1]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_191: "f32[8, 2, 512, 1, 1]" = torch.ops.aten.view.default(view_190, [8, 2, 512, 1, 1]);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_193: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.view.default(relu_132, [8, 2, 512, 8, 8])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    var_mean_137 = torch.ops.aten.var_mean.correction(convolution_169, [0, 2, 3], correction = 0, keepdim = True)
    getitem_276: "f32[1, 256, 1, 1]" = var_mean_137[0]
    getitem_277: "f32[1, 256, 1, 1]" = var_mean_137[1];  var_mean_137 = None
    add_718: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_276, 1e-05);  getitem_276 = None
    rsqrt_137: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_718);  add_718 = None
    squeeze_411: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_277, [0, 2, 3]);  getitem_277 = None
    squeeze_412: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_137, [0, 2, 3]);  rsqrt_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_166: "f32[8, 2, 1, 512]" = torch.ops.aten.alias.default(div_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_195: "f32[8, 1024]" = torch.ops.aten.view.default(div_32, [8, -1]);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_196: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(view_195, [8, -1, 1, 1]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_197: "f32[8, 2, 512, 1, 1]" = torch.ops.aten.view.default(view_196, [8, 2, 512, 1, 1]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    mm: "f32[8, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_34);  permute_34 = None
    permute_35: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_35, view_198);  permute_35 = view_198 = None
    permute_36: "f32[2048, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_100: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_199: "f32[1000]" = torch.ops.aten.view.default(sum_100, [1000]);  sum_100 = None
    permute_37: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_200: "f32[8, 2048, 1, 1]" = torch.ops.aten.view.default(mm, [8, 2048, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 2048, 8, 8]" = torch.ops.aten.expand.default(view_200, [8, 2048, 8, 8]);  view_200 = None
    div_33: "f32[8, 2048, 8, 8]" = torch.ops.aten.div.Scalar(expand, 64);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[8, 2048, 8, 8]" = torch.ops.aten.where.self(le, full_default, div_33);  le = div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_101: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_172: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_171, unsqueeze_558);  convolution_171 = unsqueeze_558 = None
    mul_1006: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(where, sub_172)
    sum_102: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1006, [0, 2, 3]);  mul_1006 = None
    mul_1007: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_101, 0.001953125)
    unsqueeze_559: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1007, 0);  mul_1007 = None
    unsqueeze_560: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_1008: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_102, 0.001953125)
    mul_1009: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_415, squeeze_415)
    mul_1010: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1008, mul_1009);  mul_1008 = mul_1009 = None
    unsqueeze_562: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1010, 0);  mul_1010 = None
    unsqueeze_563: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    mul_1011: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_415, primals_515);  primals_515 = None
    unsqueeze_565: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1011, 0);  mul_1011 = None
    unsqueeze_566: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    mul_1012: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_564);  sub_172 = unsqueeze_564 = None
    sub_174: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(where, mul_1012);  mul_1012 = None
    sub_175: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_561);  sub_174 = unsqueeze_561 = None
    mul_1013: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_567);  sub_175 = unsqueeze_567 = None
    mul_1014: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_415);  sum_102 = squeeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_1013, sum_99, primals_514, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1013 = sum_99 = primals_514 = None
    getitem_280: "f32[8, 512, 8, 8]" = convolution_backward[0]
    getitem_281: "f32[2048, 512, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_568: "f32[8, 1, 512, 8, 8]" = torch.ops.aten.unsqueeze.default(getitem_280, 1);  getitem_280 = None
    expand_1: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.expand.default(unsqueeze_568, [8, 2, 512, 8, 8]);  unsqueeze_568 = None
    mul_1015: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.mul.Tensor(expand_1, view_193);  view_193 = None
    mul_1016: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.mul.Tensor(expand_1, view_197);  expand_1 = view_197 = None
    sum_103: "f32[8, 2, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1015, [3, 4], True);  mul_1015 = None
    view_201: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(sum_103, [8, 1024, 1, 1]);  sum_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_202: "f32[8, 1024]" = torch.ops.aten.view.default(view_201, [8, 1024]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_203: "f32[8, 2, 1, 512]" = torch.ops.aten.view.default(view_202, [8, 2, 1, 512]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_171: "f32[8, 2, 1, 512]" = torch.ops.aten.alias.default(alias_166);  alias_166 = None
    mul_1017: "f32[8, 2, 1, 512]" = torch.ops.aten.mul.Tensor(view_203, alias_171);  view_203 = None
    sum_104: "f32[8, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_1017, [1], True)
    mul_1018: "f32[8, 2, 1, 512]" = torch.ops.aten.mul.Tensor(alias_171, sum_104);  alias_171 = sum_104 = None
    sub_176: "f32[8, 2, 1, 512]" = torch.ops.aten.sub.Tensor(mul_1017, mul_1018);  mul_1017 = mul_1018 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_38: "f32[8, 1, 2, 512]" = torch.ops.aten.permute.default(sub_176, [0, 2, 1, 3]);  sub_176 = None
    view_204: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(permute_38, [8, 1024, 1, 1]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(view_204, relu_133, primals_512, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_204 = primals_512 = None
    getitem_283: "f32[8, 256, 1, 1]" = convolution_backward_1[0]
    getitem_284: "f32[1024, 256, 1, 1]" = convolution_backward_1[1]
    getitem_285: "f32[1024]" = convolution_backward_1[2];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_173: "f32[8, 256, 1, 1]" = torch.ops.aten.alias.default(relu_133);  relu_133 = None
    alias_174: "f32[8, 256, 1, 1]" = torch.ops.aten.alias.default(alias_173);  alias_173 = None
    le_1: "b8[8, 256, 1, 1]" = torch.ops.aten.le.Scalar(alias_174, 0);  alias_174 = None
    where_1: "f32[8, 256, 1, 1]" = torch.ops.aten.where.self(le_1, full_default, getitem_283);  le_1 = getitem_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_569: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_411, 0);  squeeze_411 = None
    unsqueeze_570: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    sum_105: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_177: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_169, unsqueeze_571);  convolution_169 = unsqueeze_571 = None
    mul_1019: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(where_1, sub_177)
    sum_106: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1019, [0, 2, 3]);  mul_1019 = None
    mul_1020: "f32[256]" = torch.ops.aten.mul.Tensor(sum_105, 0.125)
    unsqueeze_572: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1020, 0);  mul_1020 = None
    unsqueeze_573: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    mul_1021: "f32[256]" = torch.ops.aten.mul.Tensor(sum_106, 0.125)
    mul_1022: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_412, squeeze_412)
    mul_1023: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1021, mul_1022);  mul_1021 = mul_1022 = None
    unsqueeze_575: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
    unsqueeze_576: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_1024: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_412, primals_510);  primals_510 = None
    unsqueeze_578: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1024, 0);  mul_1024 = None
    unsqueeze_579: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_1025: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_577);  sub_177 = unsqueeze_577 = None
    sub_179: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(where_1, mul_1025);  where_1 = mul_1025 = None
    sub_180: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_574);  sub_179 = unsqueeze_574 = None
    mul_1026: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_580);  sub_180 = unsqueeze_580 = None
    mul_1027: "f32[256]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_412);  sum_106 = squeeze_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_1026, mean_32, primals_508, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1026 = mean_32 = primals_508 = None
    getitem_286: "f32[8, 512, 1, 1]" = convolution_backward_2[0]
    getitem_287: "f32[256, 512, 1, 1]" = convolution_backward_2[1]
    getitem_288: "f32[256]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 512, 8, 8]" = torch.ops.aten.expand.default(getitem_286, [8, 512, 8, 8]);  getitem_286 = None
    div_34: "f32[8, 512, 8, 8]" = torch.ops.aten.div.Scalar(expand_2, 64);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_581: "f32[8, 1, 512, 8, 8]" = torch.ops.aten.unsqueeze.default(div_34, 1);  div_34 = None
    expand_3: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.expand.default(unsqueeze_581, [8, 2, 512, 8, 8]);  unsqueeze_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_728: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_1016, expand_3);  mul_1016 = expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_205: "f32[8, 1024, 8, 8]" = torch.ops.aten.view.default(add_728, [8, 1024, 8, 8]);  add_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_176: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(relu_132);  relu_132 = None
    alias_177: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(alias_176);  alias_176 = None
    le_2: "b8[8, 1024, 8, 8]" = torch.ops.aten.le.Scalar(alias_177, 0);  alias_177 = None
    where_2: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(le_2, full_default, view_205);  le_2 = view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_107: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_181: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_168, unsqueeze_584);  convolution_168 = unsqueeze_584 = None
    mul_1028: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(where_2, sub_181)
    sum_108: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1028, [0, 2, 3]);  mul_1028 = None
    mul_1029: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_107, 0.001953125)
    unsqueeze_585: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1029, 0);  mul_1029 = None
    unsqueeze_586: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
    unsqueeze_587: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
    mul_1030: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_108, 0.001953125)
    mul_1031: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_409, squeeze_409)
    mul_1032: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1030, mul_1031);  mul_1030 = mul_1031 = None
    unsqueeze_588: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
    unsqueeze_589: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 2);  unsqueeze_588 = None
    unsqueeze_590: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 3);  unsqueeze_589 = None
    mul_1033: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_409, primals_506);  primals_506 = None
    unsqueeze_591: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1033, 0);  mul_1033 = None
    unsqueeze_592: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 2);  unsqueeze_591 = None
    unsqueeze_593: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 3);  unsqueeze_592 = None
    mul_1034: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_590);  sub_181 = unsqueeze_590 = None
    sub_183: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(where_2, mul_1034);  where_2 = mul_1034 = None
    sub_184: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_587);  sub_183 = unsqueeze_587 = None
    mul_1035: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_593);  sub_184 = unsqueeze_593 = None
    mul_1036: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_409);  sum_108 = squeeze_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_1035, relu_131, primals_505, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1035 = primals_505 = None
    getitem_289: "f32[8, 512, 8, 8]" = convolution_backward_3[0]
    getitem_290: "f32[1024, 256, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_179: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(relu_131);  relu_131 = None
    alias_180: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_179);  alias_179 = None
    le_3: "b8[8, 512, 8, 8]" = torch.ops.aten.le.Scalar(alias_180, 0);  alias_180 = None
    where_3: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(le_3, full_default, getitem_289);  le_3 = getitem_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_109: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_185: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_596);  convolution_167 = unsqueeze_596 = None
    mul_1037: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_3, sub_185)
    sum_110: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1037, [0, 2, 3]);  mul_1037 = None
    mul_1038: "f32[512]" = torch.ops.aten.mul.Tensor(sum_109, 0.001953125)
    unsqueeze_597: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1038, 0);  mul_1038 = None
    unsqueeze_598: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
    unsqueeze_599: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
    mul_1039: "f32[512]" = torch.ops.aten.mul.Tensor(sum_110, 0.001953125)
    mul_1040: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_406, squeeze_406)
    mul_1041: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1039, mul_1040);  mul_1039 = mul_1040 = None
    unsqueeze_600: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1041, 0);  mul_1041 = None
    unsqueeze_601: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 2);  unsqueeze_600 = None
    unsqueeze_602: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 3);  unsqueeze_601 = None
    mul_1042: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_406, primals_503);  primals_503 = None
    unsqueeze_603: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1042, 0);  mul_1042 = None
    unsqueeze_604: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 2);  unsqueeze_603 = None
    unsqueeze_605: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 3);  unsqueeze_604 = None
    mul_1043: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_602);  sub_185 = unsqueeze_602 = None
    sub_187: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_3, mul_1043);  where_3 = mul_1043 = None
    sub_188: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_599);  sub_187 = unsqueeze_599 = None
    mul_1044: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_605);  sub_188 = unsqueeze_605 = None
    mul_1045: "f32[512]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_406);  sum_110 = squeeze_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_1044, relu_130, primals_502, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1044 = primals_502 = None
    getitem_292: "f32[8, 2048, 8, 8]" = convolution_backward_4[0]
    getitem_293: "f32[512, 2048, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_729: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(where, getitem_292);  where = getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_182: "f32[8, 2048, 8, 8]" = torch.ops.aten.alias.default(relu_130);  relu_130 = None
    alias_183: "f32[8, 2048, 8, 8]" = torch.ops.aten.alias.default(alias_182);  alias_182 = None
    le_4: "b8[8, 2048, 8, 8]" = torch.ops.aten.le.Scalar(alias_183, 0);  alias_183 = None
    where_4: "f32[8, 2048, 8, 8]" = torch.ops.aten.where.self(le_4, full_default, add_729);  le_4 = add_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_111: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_189: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_608);  convolution_166 = unsqueeze_608 = None
    mul_1046: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(where_4, sub_189)
    sum_112: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1046, [0, 2, 3]);  mul_1046 = None
    mul_1047: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_111, 0.001953125)
    unsqueeze_609: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1047, 0);  mul_1047 = None
    unsqueeze_610: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
    unsqueeze_611: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
    mul_1048: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_112, 0.001953125)
    mul_1049: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_403, squeeze_403)
    mul_1050: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1048, mul_1049);  mul_1048 = mul_1049 = None
    unsqueeze_612: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1050, 0);  mul_1050 = None
    unsqueeze_613: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 2);  unsqueeze_612 = None
    unsqueeze_614: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 3);  unsqueeze_613 = None
    mul_1051: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_403, primals_500);  primals_500 = None
    unsqueeze_615: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1051, 0);  mul_1051 = None
    unsqueeze_616: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 2);  unsqueeze_615 = None
    unsqueeze_617: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 3);  unsqueeze_616 = None
    mul_1052: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_614);  sub_189 = unsqueeze_614 = None
    sub_191: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(where_4, mul_1052);  mul_1052 = None
    sub_192: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_611);  sub_191 = unsqueeze_611 = None
    mul_1053: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_617);  sub_192 = unsqueeze_617 = None
    mul_1054: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_112, squeeze_403);  sum_112 = squeeze_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_1053, sum_96, primals_499, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1053 = sum_96 = primals_499 = None
    getitem_295: "f32[8, 512, 8, 8]" = convolution_backward_5[0]
    getitem_296: "f32[2048, 512, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_618: "f32[8, 1, 512, 8, 8]" = torch.ops.aten.unsqueeze.default(getitem_295, 1);  getitem_295 = None
    expand_4: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.expand.default(unsqueeze_618, [8, 2, 512, 8, 8]);  unsqueeze_618 = None
    mul_1055: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.mul.Tensor(expand_4, view_187);  view_187 = None
    mul_1056: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.mul.Tensor(expand_4, view_191);  expand_4 = view_191 = None
    sum_113: "f32[8, 2, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1055, [3, 4], True);  mul_1055 = None
    view_206: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(sum_113, [8, 1024, 1, 1]);  sum_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_207: "f32[8, 1024]" = torch.ops.aten.view.default(view_206, [8, 1024]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_208: "f32[8, 2, 1, 512]" = torch.ops.aten.view.default(view_207, [8, 2, 1, 512]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_184: "f32[8, 2, 1, 512]" = torch.ops.aten.alias.default(alias_161);  alias_161 = None
    mul_1057: "f32[8, 2, 1, 512]" = torch.ops.aten.mul.Tensor(view_208, alias_184);  view_208 = None
    sum_114: "f32[8, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_1057, [1], True)
    mul_1058: "f32[8, 2, 1, 512]" = torch.ops.aten.mul.Tensor(alias_184, sum_114);  alias_184 = sum_114 = None
    sub_193: "f32[8, 2, 1, 512]" = torch.ops.aten.sub.Tensor(mul_1057, mul_1058);  mul_1057 = mul_1058 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_39: "f32[8, 1, 2, 512]" = torch.ops.aten.permute.default(sub_193, [0, 2, 1, 3]);  sub_193 = None
    view_209: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(permute_39, [8, 1024, 1, 1]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(view_209, relu_129, primals_497, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_209 = primals_497 = None
    getitem_298: "f32[8, 256, 1, 1]" = convolution_backward_6[0]
    getitem_299: "f32[1024, 256, 1, 1]" = convolution_backward_6[1]
    getitem_300: "f32[1024]" = convolution_backward_6[2];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_186: "f32[8, 256, 1, 1]" = torch.ops.aten.alias.default(relu_129);  relu_129 = None
    alias_187: "f32[8, 256, 1, 1]" = torch.ops.aten.alias.default(alias_186);  alias_186 = None
    le_5: "b8[8, 256, 1, 1]" = torch.ops.aten.le.Scalar(alias_187, 0);  alias_187 = None
    where_5: "f32[8, 256, 1, 1]" = torch.ops.aten.where.self(le_5, full_default, getitem_298);  le_5 = getitem_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_619: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_399, 0);  squeeze_399 = None
    unsqueeze_620: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
    unsqueeze_621: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
    sum_115: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_194: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_164, unsqueeze_621);  convolution_164 = unsqueeze_621 = None
    mul_1059: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(where_5, sub_194)
    sum_116: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1059, [0, 2, 3]);  mul_1059 = None
    mul_1060: "f32[256]" = torch.ops.aten.mul.Tensor(sum_115, 0.125)
    unsqueeze_622: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1060, 0);  mul_1060 = None
    unsqueeze_623: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    mul_1061: "f32[256]" = torch.ops.aten.mul.Tensor(sum_116, 0.125)
    mul_1062: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_400, squeeze_400)
    mul_1063: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1061, mul_1062);  mul_1061 = mul_1062 = None
    unsqueeze_625: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1063, 0);  mul_1063 = None
    unsqueeze_626: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    mul_1064: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_400, primals_495);  primals_495 = None
    unsqueeze_628: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
    unsqueeze_629: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    mul_1065: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_627);  sub_194 = unsqueeze_627 = None
    sub_196: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(where_5, mul_1065);  where_5 = mul_1065 = None
    sub_197: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(sub_196, unsqueeze_624);  sub_196 = unsqueeze_624 = None
    mul_1066: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_630);  sub_197 = unsqueeze_630 = None
    mul_1067: "f32[256]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_400);  sum_116 = squeeze_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_1066, mean_31, primals_493, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1066 = mean_31 = primals_493 = None
    getitem_301: "f32[8, 512, 1, 1]" = convolution_backward_7[0]
    getitem_302: "f32[256, 512, 1, 1]" = convolution_backward_7[1]
    getitem_303: "f32[256]" = convolution_backward_7[2];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_5: "f32[8, 512, 8, 8]" = torch.ops.aten.expand.default(getitem_301, [8, 512, 8, 8]);  getitem_301 = None
    div_35: "f32[8, 512, 8, 8]" = torch.ops.aten.div.Scalar(expand_5, 64);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_631: "f32[8, 1, 512, 8, 8]" = torch.ops.aten.unsqueeze.default(div_35, 1);  div_35 = None
    expand_6: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.expand.default(unsqueeze_631, [8, 2, 512, 8, 8]);  unsqueeze_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_730: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_1056, expand_6);  mul_1056 = expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_210: "f32[8, 1024, 8, 8]" = torch.ops.aten.view.default(add_730, [8, 1024, 8, 8]);  add_730 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_189: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(relu_128);  relu_128 = None
    alias_190: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(alias_189);  alias_189 = None
    le_6: "b8[8, 1024, 8, 8]" = torch.ops.aten.le.Scalar(alias_190, 0);  alias_190 = None
    where_6: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(le_6, full_default, view_210);  le_6 = view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_117: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_198: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_163, unsqueeze_634);  convolution_163 = unsqueeze_634 = None
    mul_1068: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(where_6, sub_198)
    sum_118: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1068, [0, 2, 3]);  mul_1068 = None
    mul_1069: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_117, 0.001953125)
    unsqueeze_635: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1069, 0);  mul_1069 = None
    unsqueeze_636: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
    unsqueeze_637: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
    mul_1070: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_118, 0.001953125)
    mul_1071: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_397, squeeze_397)
    mul_1072: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1070, mul_1071);  mul_1070 = mul_1071 = None
    unsqueeze_638: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1072, 0);  mul_1072 = None
    unsqueeze_639: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
    unsqueeze_640: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
    mul_1073: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_397, primals_491);  primals_491 = None
    unsqueeze_641: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
    unsqueeze_642: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_1074: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_640);  sub_198 = unsqueeze_640 = None
    sub_200: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(where_6, mul_1074);  where_6 = mul_1074 = None
    sub_201: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_637);  sub_200 = unsqueeze_637 = None
    mul_1075: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_643);  sub_201 = unsqueeze_643 = None
    mul_1076: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_397);  sum_118 = squeeze_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_1075, relu_127, primals_490, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1075 = primals_490 = None
    getitem_304: "f32[8, 512, 8, 8]" = convolution_backward_8[0]
    getitem_305: "f32[1024, 256, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_192: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(relu_127);  relu_127 = None
    alias_193: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_192);  alias_192 = None
    le_7: "b8[8, 512, 8, 8]" = torch.ops.aten.le.Scalar(alias_193, 0);  alias_193 = None
    where_7: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(le_7, full_default, getitem_304);  le_7 = getitem_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_119: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_202: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_162, unsqueeze_646);  convolution_162 = unsqueeze_646 = None
    mul_1077: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_7, sub_202)
    sum_120: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1077, [0, 2, 3]);  mul_1077 = None
    mul_1078: "f32[512]" = torch.ops.aten.mul.Tensor(sum_119, 0.001953125)
    unsqueeze_647: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1078, 0);  mul_1078 = None
    unsqueeze_648: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
    unsqueeze_649: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
    mul_1079: "f32[512]" = torch.ops.aten.mul.Tensor(sum_120, 0.001953125)
    mul_1080: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_394, squeeze_394)
    mul_1081: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1079, mul_1080);  mul_1079 = mul_1080 = None
    unsqueeze_650: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1081, 0);  mul_1081 = None
    unsqueeze_651: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
    unsqueeze_652: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
    mul_1082: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_394, primals_488);  primals_488 = None
    unsqueeze_653: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
    unsqueeze_654: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_1083: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_652);  sub_202 = unsqueeze_652 = None
    sub_204: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_7, mul_1083);  where_7 = mul_1083 = None
    sub_205: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_204, unsqueeze_649);  sub_204 = unsqueeze_649 = None
    mul_1084: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_655);  sub_205 = unsqueeze_655 = None
    mul_1085: "f32[512]" = torch.ops.aten.mul.Tensor(sum_120, squeeze_394);  sum_120 = squeeze_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_1084, relu_126, primals_487, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1084 = primals_487 = None
    getitem_307: "f32[8, 2048, 8, 8]" = convolution_backward_9[0]
    getitem_308: "f32[512, 2048, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_731: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(where_4, getitem_307);  where_4 = getitem_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_195: "f32[8, 2048, 8, 8]" = torch.ops.aten.alias.default(relu_126);  relu_126 = None
    alias_196: "f32[8, 2048, 8, 8]" = torch.ops.aten.alias.default(alias_195);  alias_195 = None
    le_8: "b8[8, 2048, 8, 8]" = torch.ops.aten.le.Scalar(alias_196, 0);  alias_196 = None
    where_8: "f32[8, 2048, 8, 8]" = torch.ops.aten.where.self(le_8, full_default, add_731);  le_8 = add_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    sum_121: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_206: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_658);  convolution_161 = unsqueeze_658 = None
    mul_1086: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(where_8, sub_206)
    sum_122: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1086, [0, 2, 3]);  mul_1086 = None
    mul_1087: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_121, 0.001953125)
    unsqueeze_659: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1087, 0);  mul_1087 = None
    unsqueeze_660: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
    unsqueeze_661: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
    mul_1088: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_122, 0.001953125)
    mul_1089: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_391, squeeze_391)
    mul_1090: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1088, mul_1089);  mul_1088 = mul_1089 = None
    unsqueeze_662: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1090, 0);  mul_1090 = None
    unsqueeze_663: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
    unsqueeze_664: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
    mul_1091: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_391, primals_485);  primals_485 = None
    unsqueeze_665: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1091, 0);  mul_1091 = None
    unsqueeze_666: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    mul_1092: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_664);  sub_206 = unsqueeze_664 = None
    sub_208: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(where_8, mul_1092);  mul_1092 = None
    sub_209: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_208, unsqueeze_661);  sub_208 = None
    mul_1093: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_667);  sub_209 = unsqueeze_667 = None
    mul_1094: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_391);  sum_122 = squeeze_391 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_1093, avg_pool2d_5, primals_484, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1093 = avg_pool2d_5 = primals_484 = None
    getitem_310: "f32[8, 1024, 8, 8]" = convolution_backward_10[0]
    getitem_311: "f32[2048, 1024, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    avg_pool2d_backward: "f32[8, 1024, 16, 16]" = torch.ops.aten.avg_pool2d_backward.default(getitem_310, relu_122, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sub_210: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_670);  convolution_160 = unsqueeze_670 = None
    mul_1095: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(where_8, sub_210)
    sum_124: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1095, [0, 2, 3]);  mul_1095 = None
    mul_1097: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_124, 0.001953125)
    mul_1098: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_388, squeeze_388)
    mul_1099: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1097, mul_1098);  mul_1097 = mul_1098 = None
    unsqueeze_674: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1099, 0);  mul_1099 = None
    unsqueeze_675: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
    unsqueeze_676: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
    mul_1100: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_388, primals_482);  primals_482 = None
    unsqueeze_677: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1100, 0);  mul_1100 = None
    unsqueeze_678: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    mul_1101: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_676);  sub_210 = unsqueeze_676 = None
    sub_212: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(where_8, mul_1101);  where_8 = mul_1101 = None
    sub_213: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_212, unsqueeze_661);  sub_212 = unsqueeze_661 = None
    mul_1102: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_679);  sub_213 = unsqueeze_679 = None
    mul_1103: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_124, squeeze_388);  sum_124 = squeeze_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_1102, avg_pool2d_4, primals_481, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1102 = avg_pool2d_4 = primals_481 = None
    getitem_313: "f32[8, 512, 8, 8]" = convolution_backward_11[0]
    getitem_314: "f32[2048, 512, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    avg_pool2d_backward_1: "f32[8, 512, 16, 16]" = torch.ops.aten.avg_pool2d_backward.default(getitem_313, sum_93, [3, 3], [2, 2], [1, 1], False, True, None);  getitem_313 = sum_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_680: "f32[8, 1, 512, 16, 16]" = torch.ops.aten.unsqueeze.default(avg_pool2d_backward_1, 1);  avg_pool2d_backward_1 = None
    expand_7: "f32[8, 2, 512, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_680, [8, 2, 512, 16, 16]);  unsqueeze_680 = None
    mul_1104: "f32[8, 2, 512, 16, 16]" = torch.ops.aten.mul.Tensor(expand_7, view_181);  view_181 = None
    mul_1105: "f32[8, 2, 512, 16, 16]" = torch.ops.aten.mul.Tensor(expand_7, view_185);  expand_7 = view_185 = None
    sum_125: "f32[8, 2, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1104, [3, 4], True);  mul_1104 = None
    view_211: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(sum_125, [8, 1024, 1, 1]);  sum_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_212: "f32[8, 1024]" = torch.ops.aten.view.default(view_211, [8, 1024]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_213: "f32[8, 2, 1, 512]" = torch.ops.aten.view.default(view_212, [8, 2, 1, 512]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_197: "f32[8, 2, 1, 512]" = torch.ops.aten.alias.default(alias_156);  alias_156 = None
    mul_1106: "f32[8, 2, 1, 512]" = torch.ops.aten.mul.Tensor(view_213, alias_197);  view_213 = None
    sum_126: "f32[8, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_1106, [1], True)
    mul_1107: "f32[8, 2, 1, 512]" = torch.ops.aten.mul.Tensor(alias_197, sum_126);  alias_197 = sum_126 = None
    sub_214: "f32[8, 2, 1, 512]" = torch.ops.aten.sub.Tensor(mul_1106, mul_1107);  mul_1106 = mul_1107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_40: "f32[8, 1, 2, 512]" = torch.ops.aten.permute.default(sub_214, [0, 2, 1, 3]);  sub_214 = None
    view_214: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(permute_40, [8, 1024, 1, 1]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(view_214, relu_125, primals_479, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_214 = primals_479 = None
    getitem_316: "f32[8, 256, 1, 1]" = convolution_backward_12[0]
    getitem_317: "f32[1024, 256, 1, 1]" = convolution_backward_12[1]
    getitem_318: "f32[1024]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_199: "f32[8, 256, 1, 1]" = torch.ops.aten.alias.default(relu_125);  relu_125 = None
    alias_200: "f32[8, 256, 1, 1]" = torch.ops.aten.alias.default(alias_199);  alias_199 = None
    le_9: "b8[8, 256, 1, 1]" = torch.ops.aten.le.Scalar(alias_200, 0);  alias_200 = None
    where_9: "f32[8, 256, 1, 1]" = torch.ops.aten.where.self(le_9, full_default, getitem_316);  le_9 = getitem_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_681: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_384, 0);  squeeze_384 = None
    unsqueeze_682: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 2);  unsqueeze_681 = None
    unsqueeze_683: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 3);  unsqueeze_682 = None
    sum_127: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_215: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_158, unsqueeze_683);  convolution_158 = unsqueeze_683 = None
    mul_1108: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(where_9, sub_215)
    sum_128: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1108, [0, 2, 3]);  mul_1108 = None
    mul_1109: "f32[256]" = torch.ops.aten.mul.Tensor(sum_127, 0.125)
    unsqueeze_684: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1109, 0);  mul_1109 = None
    unsqueeze_685: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 2);  unsqueeze_684 = None
    unsqueeze_686: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 3);  unsqueeze_685 = None
    mul_1110: "f32[256]" = torch.ops.aten.mul.Tensor(sum_128, 0.125)
    mul_1111: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_385, squeeze_385)
    mul_1112: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1110, mul_1111);  mul_1110 = mul_1111 = None
    unsqueeze_687: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
    unsqueeze_688: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 2);  unsqueeze_687 = None
    unsqueeze_689: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 3);  unsqueeze_688 = None
    mul_1113: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_385, primals_477);  primals_477 = None
    unsqueeze_690: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1113, 0);  mul_1113 = None
    unsqueeze_691: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 2);  unsqueeze_690 = None
    unsqueeze_692: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 3);  unsqueeze_691 = None
    mul_1114: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_689);  sub_215 = unsqueeze_689 = None
    sub_217: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(where_9, mul_1114);  where_9 = mul_1114 = None
    sub_218: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(sub_217, unsqueeze_686);  sub_217 = unsqueeze_686 = None
    mul_1115: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_692);  sub_218 = unsqueeze_692 = None
    mul_1116: "f32[256]" = torch.ops.aten.mul.Tensor(sum_128, squeeze_385);  sum_128 = squeeze_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_1115, mean_30, primals_475, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1115 = mean_30 = primals_475 = None
    getitem_319: "f32[8, 512, 1, 1]" = convolution_backward_13[0]
    getitem_320: "f32[256, 512, 1, 1]" = convolution_backward_13[1]
    getitem_321: "f32[256]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_8: "f32[8, 512, 16, 16]" = torch.ops.aten.expand.default(getitem_319, [8, 512, 16, 16]);  getitem_319 = None
    div_36: "f32[8, 512, 16, 16]" = torch.ops.aten.div.Scalar(expand_8, 256);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_693: "f32[8, 1, 512, 16, 16]" = torch.ops.aten.unsqueeze.default(div_36, 1);  div_36 = None
    expand_9: "f32[8, 2, 512, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_693, [8, 2, 512, 16, 16]);  unsqueeze_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_732: "f32[8, 2, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_1105, expand_9);  mul_1105 = expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_215: "f32[8, 1024, 16, 16]" = torch.ops.aten.view.default(add_732, [8, 1024, 16, 16]);  add_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_202: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_124);  relu_124 = None
    alias_203: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_202);  alias_202 = None
    le_10: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_203, 0);  alias_203 = None
    where_10: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_10, full_default, view_215);  le_10 = view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_129: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_219: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_696);  convolution_157 = unsqueeze_696 = None
    mul_1117: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_10, sub_219)
    sum_130: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1117, [0, 2, 3]);  mul_1117 = None
    mul_1118: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_129, 0.00048828125)
    unsqueeze_697: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1118, 0);  mul_1118 = None
    unsqueeze_698: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    mul_1119: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_130, 0.00048828125)
    mul_1120: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_382, squeeze_382)
    mul_1121: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1119, mul_1120);  mul_1119 = mul_1120 = None
    unsqueeze_700: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1121, 0);  mul_1121 = None
    unsqueeze_701: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    mul_1122: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_382, primals_473);  primals_473 = None
    unsqueeze_703: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1122, 0);  mul_1122 = None
    unsqueeze_704: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_1123: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_702);  sub_219 = unsqueeze_702 = None
    sub_221: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_10, mul_1123);  where_10 = mul_1123 = None
    sub_222: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_699);  sub_221 = unsqueeze_699 = None
    mul_1124: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_705);  sub_222 = unsqueeze_705 = None
    mul_1125: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_130, squeeze_382);  sum_130 = squeeze_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_1124, relu_123, primals_472, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1124 = primals_472 = None
    getitem_322: "f32[8, 512, 16, 16]" = convolution_backward_14[0]
    getitem_323: "f32[1024, 256, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_205: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_123);  relu_123 = None
    alias_206: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_205);  alias_205 = None
    le_11: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_206, 0);  alias_206 = None
    where_11: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_11, full_default, getitem_322);  le_11 = getitem_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_131: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_223: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_156, unsqueeze_708);  convolution_156 = unsqueeze_708 = None
    mul_1126: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_11, sub_223)
    sum_132: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1126, [0, 2, 3]);  mul_1126 = None
    mul_1127: "f32[512]" = torch.ops.aten.mul.Tensor(sum_131, 0.00048828125)
    unsqueeze_709: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1127, 0);  mul_1127 = None
    unsqueeze_710: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    mul_1128: "f32[512]" = torch.ops.aten.mul.Tensor(sum_132, 0.00048828125)
    mul_1129: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_379, squeeze_379)
    mul_1130: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1128, mul_1129);  mul_1128 = mul_1129 = None
    unsqueeze_712: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1130, 0);  mul_1130 = None
    unsqueeze_713: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    mul_1131: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_379, primals_470);  primals_470 = None
    unsqueeze_715: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1131, 0);  mul_1131 = None
    unsqueeze_716: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
    unsqueeze_717: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
    mul_1132: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_714);  sub_223 = unsqueeze_714 = None
    sub_225: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_11, mul_1132);  where_11 = mul_1132 = None
    sub_226: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_225, unsqueeze_711);  sub_225 = unsqueeze_711 = None
    mul_1133: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_717);  sub_226 = unsqueeze_717 = None
    mul_1134: "f32[512]" = torch.ops.aten.mul.Tensor(sum_132, squeeze_379);  sum_132 = squeeze_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_1133, relu_122, primals_469, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1133 = primals_469 = None
    getitem_325: "f32[8, 1024, 16, 16]" = convolution_backward_15[0]
    getitem_326: "f32[512, 1024, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_733: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(avg_pool2d_backward, getitem_325);  avg_pool2d_backward = getitem_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_208: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_122);  relu_122 = None
    alias_209: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_208);  alias_208 = None
    le_12: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_209, 0);  alias_209 = None
    where_12: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_12, full_default, add_733);  le_12 = add_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_133: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_227: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_720);  convolution_155 = unsqueeze_720 = None
    mul_1135: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_12, sub_227)
    sum_134: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1135, [0, 2, 3]);  mul_1135 = None
    mul_1136: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_133, 0.00048828125)
    unsqueeze_721: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1136, 0);  mul_1136 = None
    unsqueeze_722: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    mul_1137: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_134, 0.00048828125)
    mul_1138: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_376, squeeze_376)
    mul_1139: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1137, mul_1138);  mul_1137 = mul_1138 = None
    unsqueeze_724: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1139, 0);  mul_1139 = None
    unsqueeze_725: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    mul_1140: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_376, primals_467);  primals_467 = None
    unsqueeze_727: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1140, 0);  mul_1140 = None
    unsqueeze_728: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
    unsqueeze_729: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
    mul_1141: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_726);  sub_227 = unsqueeze_726 = None
    sub_229: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_12, mul_1141);  mul_1141 = None
    sub_230: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_229, unsqueeze_723);  sub_229 = unsqueeze_723 = None
    mul_1142: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_729);  sub_230 = unsqueeze_729 = None
    mul_1143: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_134, squeeze_376);  sum_134 = squeeze_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_1142, sum_90, primals_466, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1142 = sum_90 = primals_466 = None
    getitem_328: "f32[8, 256, 16, 16]" = convolution_backward_16[0]
    getitem_329: "f32[1024, 256, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_730: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_328, 1);  getitem_328 = None
    expand_10: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_730, [8, 2, 256, 16, 16]);  unsqueeze_730 = None
    mul_1144: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_10, view_175);  view_175 = None
    mul_1145: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_10, view_179);  expand_10 = view_179 = None
    sum_135: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1144, [3, 4], True);  mul_1144 = None
    view_216: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_135, [8, 512, 1, 1]);  sum_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_217: "f32[8, 512]" = torch.ops.aten.view.default(view_216, [8, 512]);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_218: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_217, [8, 2, 1, 256]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_210: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_151);  alias_151 = None
    mul_1146: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_218, alias_210);  view_218 = None
    sum_136: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1146, [1], True)
    mul_1147: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_210, sum_136);  alias_210 = sum_136 = None
    sub_231: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1146, mul_1147);  mul_1146 = mul_1147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_41: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_231, [0, 2, 1, 3]);  sub_231 = None
    view_219: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_41, [8, 512, 1, 1]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(view_219, relu_121, primals_464, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_219 = primals_464 = None
    getitem_331: "f32[8, 128, 1, 1]" = convolution_backward_17[0]
    getitem_332: "f32[512, 128, 1, 1]" = convolution_backward_17[1]
    getitem_333: "f32[512]" = convolution_backward_17[2];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_212: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_121);  relu_121 = None
    alias_213: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_212);  alias_212 = None
    le_13: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_213, 0);  alias_213 = None
    where_13: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_13, full_default, getitem_331);  le_13 = getitem_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_731: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_372, 0);  squeeze_372 = None
    unsqueeze_732: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
    unsqueeze_733: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
    sum_137: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_232: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_153, unsqueeze_733);  convolution_153 = unsqueeze_733 = None
    mul_1148: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_13, sub_232)
    sum_138: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1148, [0, 2, 3]);  mul_1148 = None
    mul_1149: "f32[128]" = torch.ops.aten.mul.Tensor(sum_137, 0.125)
    unsqueeze_734: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1149, 0);  mul_1149 = None
    unsqueeze_735: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
    unsqueeze_736: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
    mul_1150: "f32[128]" = torch.ops.aten.mul.Tensor(sum_138, 0.125)
    mul_1151: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_373, squeeze_373)
    mul_1152: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1150, mul_1151);  mul_1150 = mul_1151 = None
    unsqueeze_737: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1152, 0);  mul_1152 = None
    unsqueeze_738: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    mul_1153: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_373, primals_462);  primals_462 = None
    unsqueeze_740: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1153, 0);  mul_1153 = None
    unsqueeze_741: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
    unsqueeze_742: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
    mul_1154: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_739);  sub_232 = unsqueeze_739 = None
    sub_234: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_13, mul_1154);  where_13 = mul_1154 = None
    sub_235: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_736);  sub_234 = unsqueeze_736 = None
    mul_1155: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_742);  sub_235 = unsqueeze_742 = None
    mul_1156: "f32[128]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_373);  sum_138 = squeeze_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_1155, mean_29, primals_460, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1155 = mean_29 = primals_460 = None
    getitem_334: "f32[8, 256, 1, 1]" = convolution_backward_18[0]
    getitem_335: "f32[128, 256, 1, 1]" = convolution_backward_18[1]
    getitem_336: "f32[128]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_11: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_334, [8, 256, 16, 16]);  getitem_334 = None
    div_37: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_11, 256);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_743: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_37, 1);  div_37 = None
    expand_12: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_743, [8, 2, 256, 16, 16]);  unsqueeze_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_734: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1145, expand_12);  mul_1145 = expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_220: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_734, [8, 512, 16, 16]);  add_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_215: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_120);  relu_120 = None
    alias_216: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_215);  alias_215 = None
    le_14: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_216, 0);  alias_216 = None
    where_14: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_14, full_default, view_220);  le_14 = view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_139: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_236: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_746);  convolution_152 = unsqueeze_746 = None
    mul_1157: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_14, sub_236)
    sum_140: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1157, [0, 2, 3]);  mul_1157 = None
    mul_1158: "f32[512]" = torch.ops.aten.mul.Tensor(sum_139, 0.00048828125)
    unsqueeze_747: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1158, 0);  mul_1158 = None
    unsqueeze_748: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 2);  unsqueeze_747 = None
    unsqueeze_749: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 3);  unsqueeze_748 = None
    mul_1159: "f32[512]" = torch.ops.aten.mul.Tensor(sum_140, 0.00048828125)
    mul_1160: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_370, squeeze_370)
    mul_1161: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1159, mul_1160);  mul_1159 = mul_1160 = None
    unsqueeze_750: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1161, 0);  mul_1161 = None
    unsqueeze_751: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 2);  unsqueeze_750 = None
    unsqueeze_752: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 3);  unsqueeze_751 = None
    mul_1162: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_370, primals_458);  primals_458 = None
    unsqueeze_753: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1162, 0);  mul_1162 = None
    unsqueeze_754: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 2);  unsqueeze_753 = None
    unsqueeze_755: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 3);  unsqueeze_754 = None
    mul_1163: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_752);  sub_236 = unsqueeze_752 = None
    sub_238: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_14, mul_1163);  where_14 = mul_1163 = None
    sub_239: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_749);  sub_238 = unsqueeze_749 = None
    mul_1164: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_755);  sub_239 = unsqueeze_755 = None
    mul_1165: "f32[512]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_370);  sum_140 = squeeze_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_1164, relu_119, primals_457, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1164 = primals_457 = None
    getitem_337: "f32[8, 256, 16, 16]" = convolution_backward_19[0]
    getitem_338: "f32[512, 128, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_218: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_119);  relu_119 = None
    alias_219: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_218);  alias_218 = None
    le_15: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_219, 0);  alias_219 = None
    where_15: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_15, full_default, getitem_337);  le_15 = getitem_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_141: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_240: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_758);  convolution_151 = unsqueeze_758 = None
    mul_1166: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_15, sub_240)
    sum_142: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1166, [0, 2, 3]);  mul_1166 = None
    mul_1167: "f32[256]" = torch.ops.aten.mul.Tensor(sum_141, 0.00048828125)
    unsqueeze_759: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1167, 0);  mul_1167 = None
    unsqueeze_760: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 2);  unsqueeze_759 = None
    unsqueeze_761: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 3);  unsqueeze_760 = None
    mul_1168: "f32[256]" = torch.ops.aten.mul.Tensor(sum_142, 0.00048828125)
    mul_1169: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_367, squeeze_367)
    mul_1170: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1168, mul_1169);  mul_1168 = mul_1169 = None
    unsqueeze_762: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1170, 0);  mul_1170 = None
    unsqueeze_763: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 2);  unsqueeze_762 = None
    unsqueeze_764: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 3);  unsqueeze_763 = None
    mul_1171: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_367, primals_455);  primals_455 = None
    unsqueeze_765: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1171, 0);  mul_1171 = None
    unsqueeze_766: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 2);  unsqueeze_765 = None
    unsqueeze_767: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 3);  unsqueeze_766 = None
    mul_1172: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_764);  sub_240 = unsqueeze_764 = None
    sub_242: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_15, mul_1172);  where_15 = mul_1172 = None
    sub_243: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_761);  sub_242 = unsqueeze_761 = None
    mul_1173: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_767);  sub_243 = unsqueeze_767 = None
    mul_1174: "f32[256]" = torch.ops.aten.mul.Tensor(sum_142, squeeze_367);  sum_142 = squeeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_1173, relu_118, primals_454, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1173 = primals_454 = None
    getitem_340: "f32[8, 1024, 16, 16]" = convolution_backward_20[0]
    getitem_341: "f32[256, 1024, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_735: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_12, getitem_340);  where_12 = getitem_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_221: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_118);  relu_118 = None
    alias_222: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_221);  alias_221 = None
    le_16: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_222, 0);  alias_222 = None
    where_16: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_16, full_default, add_735);  le_16 = add_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_143: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_244: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_770);  convolution_150 = unsqueeze_770 = None
    mul_1175: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_16, sub_244)
    sum_144: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1175, [0, 2, 3]);  mul_1175 = None
    mul_1176: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_143, 0.00048828125)
    unsqueeze_771: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1176, 0);  mul_1176 = None
    unsqueeze_772: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 2);  unsqueeze_771 = None
    unsqueeze_773: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 3);  unsqueeze_772 = None
    mul_1177: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_144, 0.00048828125)
    mul_1178: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_364, squeeze_364)
    mul_1179: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1177, mul_1178);  mul_1177 = mul_1178 = None
    unsqueeze_774: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1179, 0);  mul_1179 = None
    unsqueeze_775: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 2);  unsqueeze_774 = None
    unsqueeze_776: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 3);  unsqueeze_775 = None
    mul_1180: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_364, primals_452);  primals_452 = None
    unsqueeze_777: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1180, 0);  mul_1180 = None
    unsqueeze_778: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 2);  unsqueeze_777 = None
    unsqueeze_779: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 3);  unsqueeze_778 = None
    mul_1181: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_776);  sub_244 = unsqueeze_776 = None
    sub_246: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_16, mul_1181);  mul_1181 = None
    sub_247: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_773);  sub_246 = unsqueeze_773 = None
    mul_1182: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_779);  sub_247 = unsqueeze_779 = None
    mul_1183: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_144, squeeze_364);  sum_144 = squeeze_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_1182, sum_87, primals_451, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1182 = sum_87 = primals_451 = None
    getitem_343: "f32[8, 256, 16, 16]" = convolution_backward_21[0]
    getitem_344: "f32[1024, 256, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_780: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_343, 1);  getitem_343 = None
    expand_13: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_780, [8, 2, 256, 16, 16]);  unsqueeze_780 = None
    mul_1184: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_13, view_169);  view_169 = None
    mul_1185: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_13, view_173);  expand_13 = view_173 = None
    sum_145: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1184, [3, 4], True);  mul_1184 = None
    view_221: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_145, [8, 512, 1, 1]);  sum_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_222: "f32[8, 512]" = torch.ops.aten.view.default(view_221, [8, 512]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_223: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_222, [8, 2, 1, 256]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_223: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_146);  alias_146 = None
    mul_1186: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_223, alias_223);  view_223 = None
    sum_146: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1186, [1], True)
    mul_1187: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_223, sum_146);  alias_223 = sum_146 = None
    sub_248: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1186, mul_1187);  mul_1186 = mul_1187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_42: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_248, [0, 2, 1, 3]);  sub_248 = None
    view_224: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_42, [8, 512, 1, 1]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(view_224, relu_117, primals_449, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_224 = primals_449 = None
    getitem_346: "f32[8, 128, 1, 1]" = convolution_backward_22[0]
    getitem_347: "f32[512, 128, 1, 1]" = convolution_backward_22[1]
    getitem_348: "f32[512]" = convolution_backward_22[2];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_225: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_117);  relu_117 = None
    alias_226: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_225);  alias_225 = None
    le_17: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_226, 0);  alias_226 = None
    where_17: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_17, full_default, getitem_346);  le_17 = getitem_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_781: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_360, 0);  squeeze_360 = None
    unsqueeze_782: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    sum_147: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_249: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_148, unsqueeze_783);  convolution_148 = unsqueeze_783 = None
    mul_1188: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_17, sub_249)
    sum_148: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1188, [0, 2, 3]);  mul_1188 = None
    mul_1189: "f32[128]" = torch.ops.aten.mul.Tensor(sum_147, 0.125)
    unsqueeze_784: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1189, 0);  mul_1189 = None
    unsqueeze_785: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
    unsqueeze_786: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
    mul_1190: "f32[128]" = torch.ops.aten.mul.Tensor(sum_148, 0.125)
    mul_1191: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_361, squeeze_361)
    mul_1192: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1190, mul_1191);  mul_1190 = mul_1191 = None
    unsqueeze_787: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1192, 0);  mul_1192 = None
    unsqueeze_788: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
    unsqueeze_789: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
    mul_1193: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_361, primals_447);  primals_447 = None
    unsqueeze_790: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1193, 0);  mul_1193 = None
    unsqueeze_791: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
    unsqueeze_792: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
    mul_1194: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_789);  sub_249 = unsqueeze_789 = None
    sub_251: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_17, mul_1194);  where_17 = mul_1194 = None
    sub_252: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_786);  sub_251 = unsqueeze_786 = None
    mul_1195: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_792);  sub_252 = unsqueeze_792 = None
    mul_1196: "f32[128]" = torch.ops.aten.mul.Tensor(sum_148, squeeze_361);  sum_148 = squeeze_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_1195, mean_28, primals_445, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1195 = mean_28 = primals_445 = None
    getitem_349: "f32[8, 256, 1, 1]" = convolution_backward_23[0]
    getitem_350: "f32[128, 256, 1, 1]" = convolution_backward_23[1]
    getitem_351: "f32[128]" = convolution_backward_23[2];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_14: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_349, [8, 256, 16, 16]);  getitem_349 = None
    div_38: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_14, 256);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_793: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_38, 1);  div_38 = None
    expand_15: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_793, [8, 2, 256, 16, 16]);  unsqueeze_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_736: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1185, expand_15);  mul_1185 = expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_225: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_736, [8, 512, 16, 16]);  add_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_228: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_116);  relu_116 = None
    alias_229: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_228);  alias_228 = None
    le_18: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_229, 0);  alias_229 = None
    where_18: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_18, full_default, view_225);  le_18 = view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_149: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_253: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_796);  convolution_147 = unsqueeze_796 = None
    mul_1197: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_18, sub_253)
    sum_150: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1197, [0, 2, 3]);  mul_1197 = None
    mul_1198: "f32[512]" = torch.ops.aten.mul.Tensor(sum_149, 0.00048828125)
    unsqueeze_797: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1198, 0);  mul_1198 = None
    unsqueeze_798: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    mul_1199: "f32[512]" = torch.ops.aten.mul.Tensor(sum_150, 0.00048828125)
    mul_1200: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_358, squeeze_358)
    mul_1201: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1199, mul_1200);  mul_1199 = mul_1200 = None
    unsqueeze_800: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1201, 0);  mul_1201 = None
    unsqueeze_801: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
    unsqueeze_802: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
    mul_1202: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_358, primals_443);  primals_443 = None
    unsqueeze_803: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1202, 0);  mul_1202 = None
    unsqueeze_804: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
    unsqueeze_805: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
    mul_1203: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_802);  sub_253 = unsqueeze_802 = None
    sub_255: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_18, mul_1203);  where_18 = mul_1203 = None
    sub_256: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_799);  sub_255 = unsqueeze_799 = None
    mul_1204: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_805);  sub_256 = unsqueeze_805 = None
    mul_1205: "f32[512]" = torch.ops.aten.mul.Tensor(sum_150, squeeze_358);  sum_150 = squeeze_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_1204, relu_115, primals_442, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1204 = primals_442 = None
    getitem_352: "f32[8, 256, 16, 16]" = convolution_backward_24[0]
    getitem_353: "f32[512, 128, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_231: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_115);  relu_115 = None
    alias_232: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_231);  alias_231 = None
    le_19: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_232, 0);  alias_232 = None
    where_19: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_19, full_default, getitem_352);  le_19 = getitem_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_151: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_257: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_808);  convolution_146 = unsqueeze_808 = None
    mul_1206: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_19, sub_257)
    sum_152: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1206, [0, 2, 3]);  mul_1206 = None
    mul_1207: "f32[256]" = torch.ops.aten.mul.Tensor(sum_151, 0.00048828125)
    unsqueeze_809: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1207, 0);  mul_1207 = None
    unsqueeze_810: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    mul_1208: "f32[256]" = torch.ops.aten.mul.Tensor(sum_152, 0.00048828125)
    mul_1209: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_355, squeeze_355)
    mul_1210: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1208, mul_1209);  mul_1208 = mul_1209 = None
    unsqueeze_812: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1210, 0);  mul_1210 = None
    unsqueeze_813: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
    unsqueeze_814: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
    mul_1211: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_355, primals_440);  primals_440 = None
    unsqueeze_815: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1211, 0);  mul_1211 = None
    unsqueeze_816: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
    unsqueeze_817: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
    mul_1212: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_814);  sub_257 = unsqueeze_814 = None
    sub_259: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_19, mul_1212);  where_19 = mul_1212 = None
    sub_260: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_811);  sub_259 = unsqueeze_811 = None
    mul_1213: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_817);  sub_260 = unsqueeze_817 = None
    mul_1214: "f32[256]" = torch.ops.aten.mul.Tensor(sum_152, squeeze_355);  sum_152 = squeeze_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_1213, relu_114, primals_439, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1213 = primals_439 = None
    getitem_355: "f32[8, 1024, 16, 16]" = convolution_backward_25[0]
    getitem_356: "f32[256, 1024, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_737: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_16, getitem_355);  where_16 = getitem_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_234: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_114);  relu_114 = None
    alias_235: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_234);  alias_234 = None
    le_20: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_235, 0);  alias_235 = None
    where_20: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_20, full_default, add_737);  le_20 = add_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_153: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_261: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_820);  convolution_145 = unsqueeze_820 = None
    mul_1215: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_20, sub_261)
    sum_154: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1215, [0, 2, 3]);  mul_1215 = None
    mul_1216: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_153, 0.00048828125)
    unsqueeze_821: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1216, 0);  mul_1216 = None
    unsqueeze_822: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    mul_1217: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_154, 0.00048828125)
    mul_1218: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_352, squeeze_352)
    mul_1219: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1217, mul_1218);  mul_1217 = mul_1218 = None
    unsqueeze_824: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1219, 0);  mul_1219 = None
    unsqueeze_825: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
    unsqueeze_826: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
    mul_1220: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_352, primals_437);  primals_437 = None
    unsqueeze_827: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1220, 0);  mul_1220 = None
    unsqueeze_828: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
    unsqueeze_829: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
    mul_1221: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_826);  sub_261 = unsqueeze_826 = None
    sub_263: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_20, mul_1221);  mul_1221 = None
    sub_264: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_823);  sub_263 = unsqueeze_823 = None
    mul_1222: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_829);  sub_264 = unsqueeze_829 = None
    mul_1223: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_154, squeeze_352);  sum_154 = squeeze_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_1222, sum_84, primals_436, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1222 = sum_84 = primals_436 = None
    getitem_358: "f32[8, 256, 16, 16]" = convolution_backward_26[0]
    getitem_359: "f32[1024, 256, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_830: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_358, 1);  getitem_358 = None
    expand_16: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_830, [8, 2, 256, 16, 16]);  unsqueeze_830 = None
    mul_1224: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_16, view_163);  view_163 = None
    mul_1225: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_16, view_167);  expand_16 = view_167 = None
    sum_155: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1224, [3, 4], True);  mul_1224 = None
    view_226: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_155, [8, 512, 1, 1]);  sum_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_227: "f32[8, 512]" = torch.ops.aten.view.default(view_226, [8, 512]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_228: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_227, [8, 2, 1, 256]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_236: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_141);  alias_141 = None
    mul_1226: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_228, alias_236);  view_228 = None
    sum_156: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1226, [1], True)
    mul_1227: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_236, sum_156);  alias_236 = sum_156 = None
    sub_265: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1226, mul_1227);  mul_1226 = mul_1227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_43: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_265, [0, 2, 1, 3]);  sub_265 = None
    view_229: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_43, [8, 512, 1, 1]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(view_229, relu_113, primals_434, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_229 = primals_434 = None
    getitem_361: "f32[8, 128, 1, 1]" = convolution_backward_27[0]
    getitem_362: "f32[512, 128, 1, 1]" = convolution_backward_27[1]
    getitem_363: "f32[512]" = convolution_backward_27[2];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_238: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_113);  relu_113 = None
    alias_239: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_238);  alias_238 = None
    le_21: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_239, 0);  alias_239 = None
    where_21: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_21, full_default, getitem_361);  le_21 = getitem_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_831: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_348, 0);  squeeze_348 = None
    unsqueeze_832: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 2);  unsqueeze_831 = None
    unsqueeze_833: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 3);  unsqueeze_832 = None
    sum_157: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_266: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_143, unsqueeze_833);  convolution_143 = unsqueeze_833 = None
    mul_1228: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_21, sub_266)
    sum_158: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1228, [0, 2, 3]);  mul_1228 = None
    mul_1229: "f32[128]" = torch.ops.aten.mul.Tensor(sum_157, 0.125)
    unsqueeze_834: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1229, 0);  mul_1229 = None
    unsqueeze_835: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 2);  unsqueeze_834 = None
    unsqueeze_836: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 3);  unsqueeze_835 = None
    mul_1230: "f32[128]" = torch.ops.aten.mul.Tensor(sum_158, 0.125)
    mul_1231: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_349, squeeze_349)
    mul_1232: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1230, mul_1231);  mul_1230 = mul_1231 = None
    unsqueeze_837: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1232, 0);  mul_1232 = None
    unsqueeze_838: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 2);  unsqueeze_837 = None
    unsqueeze_839: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 3);  unsqueeze_838 = None
    mul_1233: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_349, primals_432);  primals_432 = None
    unsqueeze_840: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1233, 0);  mul_1233 = None
    unsqueeze_841: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 2);  unsqueeze_840 = None
    unsqueeze_842: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 3);  unsqueeze_841 = None
    mul_1234: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_839);  sub_266 = unsqueeze_839 = None
    sub_268: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_21, mul_1234);  where_21 = mul_1234 = None
    sub_269: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_268, unsqueeze_836);  sub_268 = unsqueeze_836 = None
    mul_1235: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_842);  sub_269 = unsqueeze_842 = None
    mul_1236: "f32[128]" = torch.ops.aten.mul.Tensor(sum_158, squeeze_349);  sum_158 = squeeze_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_1235, mean_27, primals_430, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1235 = mean_27 = primals_430 = None
    getitem_364: "f32[8, 256, 1, 1]" = convolution_backward_28[0]
    getitem_365: "f32[128, 256, 1, 1]" = convolution_backward_28[1]
    getitem_366: "f32[128]" = convolution_backward_28[2];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_17: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_364, [8, 256, 16, 16]);  getitem_364 = None
    div_39: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_17, 256);  expand_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_843: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_39, 1);  div_39 = None
    expand_18: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_843, [8, 2, 256, 16, 16]);  unsqueeze_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_738: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1225, expand_18);  mul_1225 = expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_230: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_738, [8, 512, 16, 16]);  add_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_241: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_112);  relu_112 = None
    alias_242: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_241);  alias_241 = None
    le_22: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_242, 0);  alias_242 = None
    where_22: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_22, full_default, view_230);  le_22 = view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_159: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_270: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_846);  convolution_142 = unsqueeze_846 = None
    mul_1237: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_22, sub_270)
    sum_160: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1237, [0, 2, 3]);  mul_1237 = None
    mul_1238: "f32[512]" = torch.ops.aten.mul.Tensor(sum_159, 0.00048828125)
    unsqueeze_847: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1238, 0);  mul_1238 = None
    unsqueeze_848: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
    unsqueeze_849: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
    mul_1239: "f32[512]" = torch.ops.aten.mul.Tensor(sum_160, 0.00048828125)
    mul_1240: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_346, squeeze_346)
    mul_1241: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1239, mul_1240);  mul_1239 = mul_1240 = None
    unsqueeze_850: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1241, 0);  mul_1241 = None
    unsqueeze_851: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
    unsqueeze_852: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
    mul_1242: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_346, primals_428);  primals_428 = None
    unsqueeze_853: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1242, 0);  mul_1242 = None
    unsqueeze_854: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    mul_1243: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_852);  sub_270 = unsqueeze_852 = None
    sub_272: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_22, mul_1243);  where_22 = mul_1243 = None
    sub_273: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_272, unsqueeze_849);  sub_272 = unsqueeze_849 = None
    mul_1244: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_855);  sub_273 = unsqueeze_855 = None
    mul_1245: "f32[512]" = torch.ops.aten.mul.Tensor(sum_160, squeeze_346);  sum_160 = squeeze_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_1244, relu_111, primals_427, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1244 = primals_427 = None
    getitem_367: "f32[8, 256, 16, 16]" = convolution_backward_29[0]
    getitem_368: "f32[512, 128, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_244: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_111);  relu_111 = None
    alias_245: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_244);  alias_244 = None
    le_23: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_245, 0);  alias_245 = None
    where_23: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_23, full_default, getitem_367);  le_23 = getitem_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_161: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_274: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_858);  convolution_141 = unsqueeze_858 = None
    mul_1246: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_23, sub_274)
    sum_162: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1246, [0, 2, 3]);  mul_1246 = None
    mul_1247: "f32[256]" = torch.ops.aten.mul.Tensor(sum_161, 0.00048828125)
    unsqueeze_859: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1247, 0);  mul_1247 = None
    unsqueeze_860: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
    unsqueeze_861: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
    mul_1248: "f32[256]" = torch.ops.aten.mul.Tensor(sum_162, 0.00048828125)
    mul_1249: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_343, squeeze_343)
    mul_1250: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1248, mul_1249);  mul_1248 = mul_1249 = None
    unsqueeze_862: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1250, 0);  mul_1250 = None
    unsqueeze_863: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
    unsqueeze_864: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
    mul_1251: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_343, primals_425);  primals_425 = None
    unsqueeze_865: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1251, 0);  mul_1251 = None
    unsqueeze_866: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    mul_1252: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_864);  sub_274 = unsqueeze_864 = None
    sub_276: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_23, mul_1252);  where_23 = mul_1252 = None
    sub_277: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_276, unsqueeze_861);  sub_276 = unsqueeze_861 = None
    mul_1253: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_867);  sub_277 = unsqueeze_867 = None
    mul_1254: "f32[256]" = torch.ops.aten.mul.Tensor(sum_162, squeeze_343);  sum_162 = squeeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_1253, relu_110, primals_424, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1253 = primals_424 = None
    getitem_370: "f32[8, 1024, 16, 16]" = convolution_backward_30[0]
    getitem_371: "f32[256, 1024, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_739: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_20, getitem_370);  where_20 = getitem_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_247: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_110);  relu_110 = None
    alias_248: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_247);  alias_247 = None
    le_24: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_248, 0);  alias_248 = None
    where_24: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_24, full_default, add_739);  le_24 = add_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_163: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_278: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_870);  convolution_140 = unsqueeze_870 = None
    mul_1255: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_24, sub_278)
    sum_164: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1255, [0, 2, 3]);  mul_1255 = None
    mul_1256: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_163, 0.00048828125)
    unsqueeze_871: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1256, 0);  mul_1256 = None
    unsqueeze_872: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
    unsqueeze_873: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
    mul_1257: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_164, 0.00048828125)
    mul_1258: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_340, squeeze_340)
    mul_1259: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1257, mul_1258);  mul_1257 = mul_1258 = None
    unsqueeze_874: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1259, 0);  mul_1259 = None
    unsqueeze_875: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
    unsqueeze_876: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
    mul_1260: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_340, primals_422);  primals_422 = None
    unsqueeze_877: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1260, 0);  mul_1260 = None
    unsqueeze_878: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    mul_1261: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_876);  sub_278 = unsqueeze_876 = None
    sub_280: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_24, mul_1261);  mul_1261 = None
    sub_281: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_280, unsqueeze_873);  sub_280 = unsqueeze_873 = None
    mul_1262: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_879);  sub_281 = unsqueeze_879 = None
    mul_1263: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_164, squeeze_340);  sum_164 = squeeze_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_1262, sum_81, primals_421, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1262 = sum_81 = primals_421 = None
    getitem_373: "f32[8, 256, 16, 16]" = convolution_backward_31[0]
    getitem_374: "f32[1024, 256, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_880: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_373, 1);  getitem_373 = None
    expand_19: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_880, [8, 2, 256, 16, 16]);  unsqueeze_880 = None
    mul_1264: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_19, view_157);  view_157 = None
    mul_1265: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_19, view_161);  expand_19 = view_161 = None
    sum_165: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1264, [3, 4], True);  mul_1264 = None
    view_231: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_165, [8, 512, 1, 1]);  sum_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_232: "f32[8, 512]" = torch.ops.aten.view.default(view_231, [8, 512]);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_233: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_232, [8, 2, 1, 256]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_249: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_136);  alias_136 = None
    mul_1266: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_233, alias_249);  view_233 = None
    sum_166: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1266, [1], True)
    mul_1267: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_249, sum_166);  alias_249 = sum_166 = None
    sub_282: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1266, mul_1267);  mul_1266 = mul_1267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_44: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_282, [0, 2, 1, 3]);  sub_282 = None
    view_234: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_44, [8, 512, 1, 1]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(view_234, relu_109, primals_419, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_234 = primals_419 = None
    getitem_376: "f32[8, 128, 1, 1]" = convolution_backward_32[0]
    getitem_377: "f32[512, 128, 1, 1]" = convolution_backward_32[1]
    getitem_378: "f32[512]" = convolution_backward_32[2];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_251: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_109);  relu_109 = None
    alias_252: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_251);  alias_251 = None
    le_25: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_252, 0);  alias_252 = None
    where_25: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_25, full_default, getitem_376);  le_25 = getitem_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_881: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_336, 0);  squeeze_336 = None
    unsqueeze_882: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    sum_167: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_283: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_883);  convolution_138 = unsqueeze_883 = None
    mul_1268: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_25, sub_283)
    sum_168: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1268, [0, 2, 3]);  mul_1268 = None
    mul_1269: "f32[128]" = torch.ops.aten.mul.Tensor(sum_167, 0.125)
    unsqueeze_884: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1269, 0);  mul_1269 = None
    unsqueeze_885: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
    unsqueeze_886: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
    mul_1270: "f32[128]" = torch.ops.aten.mul.Tensor(sum_168, 0.125)
    mul_1271: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_337, squeeze_337)
    mul_1272: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1270, mul_1271);  mul_1270 = mul_1271 = None
    unsqueeze_887: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1272, 0);  mul_1272 = None
    unsqueeze_888: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 2);  unsqueeze_887 = None
    unsqueeze_889: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 3);  unsqueeze_888 = None
    mul_1273: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_337, primals_417);  primals_417 = None
    unsqueeze_890: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1273, 0);  mul_1273 = None
    unsqueeze_891: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 2);  unsqueeze_890 = None
    unsqueeze_892: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 3);  unsqueeze_891 = None
    mul_1274: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_889);  sub_283 = unsqueeze_889 = None
    sub_285: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_25, mul_1274);  where_25 = mul_1274 = None
    sub_286: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_285, unsqueeze_886);  sub_285 = unsqueeze_886 = None
    mul_1275: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_892);  sub_286 = unsqueeze_892 = None
    mul_1276: "f32[128]" = torch.ops.aten.mul.Tensor(sum_168, squeeze_337);  sum_168 = squeeze_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_1275, mean_26, primals_415, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1275 = mean_26 = primals_415 = None
    getitem_379: "f32[8, 256, 1, 1]" = convolution_backward_33[0]
    getitem_380: "f32[128, 256, 1, 1]" = convolution_backward_33[1]
    getitem_381: "f32[128]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_20: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_379, [8, 256, 16, 16]);  getitem_379 = None
    div_40: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_20, 256);  expand_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_893: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_40, 1);  div_40 = None
    expand_21: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_893, [8, 2, 256, 16, 16]);  unsqueeze_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_740: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1265, expand_21);  mul_1265 = expand_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_235: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_740, [8, 512, 16, 16]);  add_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_254: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_108);  relu_108 = None
    alias_255: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_254);  alias_254 = None
    le_26: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_255, 0);  alias_255 = None
    where_26: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_26, full_default, view_235);  le_26 = view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_169: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_287: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_896);  convolution_137 = unsqueeze_896 = None
    mul_1277: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_26, sub_287)
    sum_170: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1277, [0, 2, 3]);  mul_1277 = None
    mul_1278: "f32[512]" = torch.ops.aten.mul.Tensor(sum_169, 0.00048828125)
    unsqueeze_897: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1278, 0);  mul_1278 = None
    unsqueeze_898: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 2);  unsqueeze_897 = None
    unsqueeze_899: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 3);  unsqueeze_898 = None
    mul_1279: "f32[512]" = torch.ops.aten.mul.Tensor(sum_170, 0.00048828125)
    mul_1280: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_334, squeeze_334)
    mul_1281: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1279, mul_1280);  mul_1279 = mul_1280 = None
    unsqueeze_900: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1281, 0);  mul_1281 = None
    unsqueeze_901: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 2);  unsqueeze_900 = None
    unsqueeze_902: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 3);  unsqueeze_901 = None
    mul_1282: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_334, primals_413);  primals_413 = None
    unsqueeze_903: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1282, 0);  mul_1282 = None
    unsqueeze_904: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 2);  unsqueeze_903 = None
    unsqueeze_905: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 3);  unsqueeze_904 = None
    mul_1283: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_902);  sub_287 = unsqueeze_902 = None
    sub_289: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_26, mul_1283);  where_26 = mul_1283 = None
    sub_290: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_289, unsqueeze_899);  sub_289 = unsqueeze_899 = None
    mul_1284: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_905);  sub_290 = unsqueeze_905 = None
    mul_1285: "f32[512]" = torch.ops.aten.mul.Tensor(sum_170, squeeze_334);  sum_170 = squeeze_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_1284, relu_107, primals_412, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1284 = primals_412 = None
    getitem_382: "f32[8, 256, 16, 16]" = convolution_backward_34[0]
    getitem_383: "f32[512, 128, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_257: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_107);  relu_107 = None
    alias_258: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_257);  alias_257 = None
    le_27: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_258, 0);  alias_258 = None
    where_27: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_27, full_default, getitem_382);  le_27 = getitem_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_171: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_291: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_908);  convolution_136 = unsqueeze_908 = None
    mul_1286: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_27, sub_291)
    sum_172: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1286, [0, 2, 3]);  mul_1286 = None
    mul_1287: "f32[256]" = torch.ops.aten.mul.Tensor(sum_171, 0.00048828125)
    unsqueeze_909: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1287, 0);  mul_1287 = None
    unsqueeze_910: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 2);  unsqueeze_909 = None
    unsqueeze_911: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 3);  unsqueeze_910 = None
    mul_1288: "f32[256]" = torch.ops.aten.mul.Tensor(sum_172, 0.00048828125)
    mul_1289: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_331, squeeze_331)
    mul_1290: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1288, mul_1289);  mul_1288 = mul_1289 = None
    unsqueeze_912: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1290, 0);  mul_1290 = None
    unsqueeze_913: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 2);  unsqueeze_912 = None
    unsqueeze_914: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 3);  unsqueeze_913 = None
    mul_1291: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_331, primals_410);  primals_410 = None
    unsqueeze_915: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1291, 0);  mul_1291 = None
    unsqueeze_916: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 2);  unsqueeze_915 = None
    unsqueeze_917: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 3);  unsqueeze_916 = None
    mul_1292: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_914);  sub_291 = unsqueeze_914 = None
    sub_293: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_27, mul_1292);  where_27 = mul_1292 = None
    sub_294: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_293, unsqueeze_911);  sub_293 = unsqueeze_911 = None
    mul_1293: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_917);  sub_294 = unsqueeze_917 = None
    mul_1294: "f32[256]" = torch.ops.aten.mul.Tensor(sum_172, squeeze_331);  sum_172 = squeeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_1293, relu_106, primals_409, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1293 = primals_409 = None
    getitem_385: "f32[8, 1024, 16, 16]" = convolution_backward_35[0]
    getitem_386: "f32[256, 1024, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_741: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_24, getitem_385);  where_24 = getitem_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_260: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_106);  relu_106 = None
    alias_261: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_260);  alias_260 = None
    le_28: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_261, 0);  alias_261 = None
    where_28: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_28, full_default, add_741);  le_28 = add_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_173: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_295: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_920);  convolution_135 = unsqueeze_920 = None
    mul_1295: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_28, sub_295)
    sum_174: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1295, [0, 2, 3]);  mul_1295 = None
    mul_1296: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_173, 0.00048828125)
    unsqueeze_921: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1296, 0);  mul_1296 = None
    unsqueeze_922: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 2);  unsqueeze_921 = None
    unsqueeze_923: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 3);  unsqueeze_922 = None
    mul_1297: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_174, 0.00048828125)
    mul_1298: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_328, squeeze_328)
    mul_1299: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1297, mul_1298);  mul_1297 = mul_1298 = None
    unsqueeze_924: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1299, 0);  mul_1299 = None
    unsqueeze_925: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 2);  unsqueeze_924 = None
    unsqueeze_926: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 3);  unsqueeze_925 = None
    mul_1300: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_328, primals_407);  primals_407 = None
    unsqueeze_927: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1300, 0);  mul_1300 = None
    unsqueeze_928: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 2);  unsqueeze_927 = None
    unsqueeze_929: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 3);  unsqueeze_928 = None
    mul_1301: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_926);  sub_295 = unsqueeze_926 = None
    sub_297: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_28, mul_1301);  mul_1301 = None
    sub_298: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_297, unsqueeze_923);  sub_297 = unsqueeze_923 = None
    mul_1302: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_929);  sub_298 = unsqueeze_929 = None
    mul_1303: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_174, squeeze_328);  sum_174 = squeeze_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_1302, sum_78, primals_406, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1302 = sum_78 = primals_406 = None
    getitem_388: "f32[8, 256, 16, 16]" = convolution_backward_36[0]
    getitem_389: "f32[1024, 256, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_930: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_388, 1);  getitem_388 = None
    expand_22: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_930, [8, 2, 256, 16, 16]);  unsqueeze_930 = None
    mul_1304: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_22, view_151);  view_151 = None
    mul_1305: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_22, view_155);  expand_22 = view_155 = None
    sum_175: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1304, [3, 4], True);  mul_1304 = None
    view_236: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_175, [8, 512, 1, 1]);  sum_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_237: "f32[8, 512]" = torch.ops.aten.view.default(view_236, [8, 512]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_238: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_237, [8, 2, 1, 256]);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_262: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_131);  alias_131 = None
    mul_1306: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_238, alias_262);  view_238 = None
    sum_176: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1306, [1], True)
    mul_1307: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_262, sum_176);  alias_262 = sum_176 = None
    sub_299: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1306, mul_1307);  mul_1306 = mul_1307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_45: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_299, [0, 2, 1, 3]);  sub_299 = None
    view_239: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_45, [8, 512, 1, 1]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(view_239, relu_105, primals_404, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_239 = primals_404 = None
    getitem_391: "f32[8, 128, 1, 1]" = convolution_backward_37[0]
    getitem_392: "f32[512, 128, 1, 1]" = convolution_backward_37[1]
    getitem_393: "f32[512]" = convolution_backward_37[2];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_264: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_105);  relu_105 = None
    alias_265: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_264);  alias_264 = None
    le_29: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_265, 0);  alias_265 = None
    where_29: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_29, full_default, getitem_391);  le_29 = getitem_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_931: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_324, 0);  squeeze_324 = None
    unsqueeze_932: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 2);  unsqueeze_931 = None
    unsqueeze_933: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 3);  unsqueeze_932 = None
    sum_177: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_300: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_933);  convolution_133 = unsqueeze_933 = None
    mul_1308: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_29, sub_300)
    sum_178: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1308, [0, 2, 3]);  mul_1308 = None
    mul_1309: "f32[128]" = torch.ops.aten.mul.Tensor(sum_177, 0.125)
    unsqueeze_934: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1309, 0);  mul_1309 = None
    unsqueeze_935: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 2);  unsqueeze_934 = None
    unsqueeze_936: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 3);  unsqueeze_935 = None
    mul_1310: "f32[128]" = torch.ops.aten.mul.Tensor(sum_178, 0.125)
    mul_1311: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_325, squeeze_325)
    mul_1312: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1310, mul_1311);  mul_1310 = mul_1311 = None
    unsqueeze_937: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1312, 0);  mul_1312 = None
    unsqueeze_938: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
    unsqueeze_939: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
    mul_1313: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_325, primals_402);  primals_402 = None
    unsqueeze_940: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1313, 0);  mul_1313 = None
    unsqueeze_941: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 2);  unsqueeze_940 = None
    unsqueeze_942: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 3);  unsqueeze_941 = None
    mul_1314: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_939);  sub_300 = unsqueeze_939 = None
    sub_302: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_29, mul_1314);  where_29 = mul_1314 = None
    sub_303: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_936);  sub_302 = unsqueeze_936 = None
    mul_1315: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_942);  sub_303 = unsqueeze_942 = None
    mul_1316: "f32[128]" = torch.ops.aten.mul.Tensor(sum_178, squeeze_325);  sum_178 = squeeze_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_1315, mean_25, primals_400, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1315 = mean_25 = primals_400 = None
    getitem_394: "f32[8, 256, 1, 1]" = convolution_backward_38[0]
    getitem_395: "f32[128, 256, 1, 1]" = convolution_backward_38[1]
    getitem_396: "f32[128]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_23: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_394, [8, 256, 16, 16]);  getitem_394 = None
    div_41: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_23, 256);  expand_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_943: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_41, 1);  div_41 = None
    expand_24: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_943, [8, 2, 256, 16, 16]);  unsqueeze_943 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_742: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1305, expand_24);  mul_1305 = expand_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_240: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_742, [8, 512, 16, 16]);  add_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_267: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_104);  relu_104 = None
    alias_268: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_267);  alias_267 = None
    le_30: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_268, 0);  alias_268 = None
    where_30: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_30, full_default, view_240);  le_30 = view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_179: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_304: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_946);  convolution_132 = unsqueeze_946 = None
    mul_1317: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_30, sub_304)
    sum_180: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1317, [0, 2, 3]);  mul_1317 = None
    mul_1318: "f32[512]" = torch.ops.aten.mul.Tensor(sum_179, 0.00048828125)
    unsqueeze_947: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1318, 0);  mul_1318 = None
    unsqueeze_948: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 2);  unsqueeze_947 = None
    unsqueeze_949: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 3);  unsqueeze_948 = None
    mul_1319: "f32[512]" = torch.ops.aten.mul.Tensor(sum_180, 0.00048828125)
    mul_1320: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_322, squeeze_322)
    mul_1321: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1319, mul_1320);  mul_1319 = mul_1320 = None
    unsqueeze_950: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1321, 0);  mul_1321 = None
    unsqueeze_951: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 2);  unsqueeze_950 = None
    unsqueeze_952: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 3);  unsqueeze_951 = None
    mul_1322: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_322, primals_398);  primals_398 = None
    unsqueeze_953: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1322, 0);  mul_1322 = None
    unsqueeze_954: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
    unsqueeze_955: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
    mul_1323: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_952);  sub_304 = unsqueeze_952 = None
    sub_306: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_30, mul_1323);  where_30 = mul_1323 = None
    sub_307: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_306, unsqueeze_949);  sub_306 = unsqueeze_949 = None
    mul_1324: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_955);  sub_307 = unsqueeze_955 = None
    mul_1325: "f32[512]" = torch.ops.aten.mul.Tensor(sum_180, squeeze_322);  sum_180 = squeeze_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_1324, relu_103, primals_397, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1324 = primals_397 = None
    getitem_397: "f32[8, 256, 16, 16]" = convolution_backward_39[0]
    getitem_398: "f32[512, 128, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_270: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_103);  relu_103 = None
    alias_271: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_270);  alias_270 = None
    le_31: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_271, 0);  alias_271 = None
    where_31: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_31, full_default, getitem_397);  le_31 = getitem_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_181: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_308: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_958);  convolution_131 = unsqueeze_958 = None
    mul_1326: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_31, sub_308)
    sum_182: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1326, [0, 2, 3]);  mul_1326 = None
    mul_1327: "f32[256]" = torch.ops.aten.mul.Tensor(sum_181, 0.00048828125)
    unsqueeze_959: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1327, 0);  mul_1327 = None
    unsqueeze_960: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 2);  unsqueeze_959 = None
    unsqueeze_961: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 3);  unsqueeze_960 = None
    mul_1328: "f32[256]" = torch.ops.aten.mul.Tensor(sum_182, 0.00048828125)
    mul_1329: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_319, squeeze_319)
    mul_1330: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1328, mul_1329);  mul_1328 = mul_1329 = None
    unsqueeze_962: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1330, 0);  mul_1330 = None
    unsqueeze_963: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 2);  unsqueeze_962 = None
    unsqueeze_964: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 3);  unsqueeze_963 = None
    mul_1331: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_319, primals_395);  primals_395 = None
    unsqueeze_965: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1331, 0);  mul_1331 = None
    unsqueeze_966: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
    unsqueeze_967: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
    mul_1332: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_964);  sub_308 = unsqueeze_964 = None
    sub_310: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_31, mul_1332);  where_31 = mul_1332 = None
    sub_311: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_310, unsqueeze_961);  sub_310 = unsqueeze_961 = None
    mul_1333: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_967);  sub_311 = unsqueeze_967 = None
    mul_1334: "f32[256]" = torch.ops.aten.mul.Tensor(sum_182, squeeze_319);  sum_182 = squeeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_1333, relu_102, primals_394, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1333 = primals_394 = None
    getitem_400: "f32[8, 1024, 16, 16]" = convolution_backward_40[0]
    getitem_401: "f32[256, 1024, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_743: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_28, getitem_400);  where_28 = getitem_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_273: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_102);  relu_102 = None
    alias_274: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_273);  alias_273 = None
    le_32: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_274, 0);  alias_274 = None
    where_32: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_32, full_default, add_743);  le_32 = add_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_183: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_312: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_970);  convolution_130 = unsqueeze_970 = None
    mul_1335: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_32, sub_312)
    sum_184: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1335, [0, 2, 3]);  mul_1335 = None
    mul_1336: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_183, 0.00048828125)
    unsqueeze_971: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1336, 0);  mul_1336 = None
    unsqueeze_972: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 2);  unsqueeze_971 = None
    unsqueeze_973: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 3);  unsqueeze_972 = None
    mul_1337: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_184, 0.00048828125)
    mul_1338: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_316, squeeze_316)
    mul_1339: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1337, mul_1338);  mul_1337 = mul_1338 = None
    unsqueeze_974: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1339, 0);  mul_1339 = None
    unsqueeze_975: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 2);  unsqueeze_974 = None
    unsqueeze_976: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 3);  unsqueeze_975 = None
    mul_1340: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_316, primals_392);  primals_392 = None
    unsqueeze_977: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1340, 0);  mul_1340 = None
    unsqueeze_978: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    mul_1341: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_976);  sub_312 = unsqueeze_976 = None
    sub_314: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_32, mul_1341);  mul_1341 = None
    sub_315: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_314, unsqueeze_973);  sub_314 = unsqueeze_973 = None
    mul_1342: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_979);  sub_315 = unsqueeze_979 = None
    mul_1343: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_184, squeeze_316);  sum_184 = squeeze_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_1342, sum_75, primals_391, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1342 = sum_75 = primals_391 = None
    getitem_403: "f32[8, 256, 16, 16]" = convolution_backward_41[0]
    getitem_404: "f32[1024, 256, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_980: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_403, 1);  getitem_403 = None
    expand_25: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_980, [8, 2, 256, 16, 16]);  unsqueeze_980 = None
    mul_1344: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_25, view_145);  view_145 = None
    mul_1345: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_25, view_149);  expand_25 = view_149 = None
    sum_185: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1344, [3, 4], True);  mul_1344 = None
    view_241: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_185, [8, 512, 1, 1]);  sum_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_242: "f32[8, 512]" = torch.ops.aten.view.default(view_241, [8, 512]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_243: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_242, [8, 2, 1, 256]);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_275: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_126);  alias_126 = None
    mul_1346: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_243, alias_275);  view_243 = None
    sum_186: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1346, [1], True)
    mul_1347: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_275, sum_186);  alias_275 = sum_186 = None
    sub_316: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1346, mul_1347);  mul_1346 = mul_1347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_46: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_316, [0, 2, 1, 3]);  sub_316 = None
    view_244: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_46, [8, 512, 1, 1]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(view_244, relu_101, primals_389, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_244 = primals_389 = None
    getitem_406: "f32[8, 128, 1, 1]" = convolution_backward_42[0]
    getitem_407: "f32[512, 128, 1, 1]" = convolution_backward_42[1]
    getitem_408: "f32[512]" = convolution_backward_42[2];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_277: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_101);  relu_101 = None
    alias_278: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_277);  alias_277 = None
    le_33: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_278, 0);  alias_278 = None
    where_33: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_33, full_default, getitem_406);  le_33 = getitem_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_981: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_312, 0);  squeeze_312 = None
    unsqueeze_982: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 2);  unsqueeze_981 = None
    unsqueeze_983: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 3);  unsqueeze_982 = None
    sum_187: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_317: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_983);  convolution_128 = unsqueeze_983 = None
    mul_1348: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_33, sub_317)
    sum_188: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1348, [0, 2, 3]);  mul_1348 = None
    mul_1349: "f32[128]" = torch.ops.aten.mul.Tensor(sum_187, 0.125)
    unsqueeze_984: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1349, 0);  mul_1349 = None
    unsqueeze_985: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 2);  unsqueeze_984 = None
    unsqueeze_986: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 3);  unsqueeze_985 = None
    mul_1350: "f32[128]" = torch.ops.aten.mul.Tensor(sum_188, 0.125)
    mul_1351: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_313, squeeze_313)
    mul_1352: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1350, mul_1351);  mul_1350 = mul_1351 = None
    unsqueeze_987: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1352, 0);  mul_1352 = None
    unsqueeze_988: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 2);  unsqueeze_987 = None
    unsqueeze_989: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 3);  unsqueeze_988 = None
    mul_1353: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_313, primals_387);  primals_387 = None
    unsqueeze_990: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1353, 0);  mul_1353 = None
    unsqueeze_991: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 2);  unsqueeze_990 = None
    unsqueeze_992: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 3);  unsqueeze_991 = None
    mul_1354: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_989);  sub_317 = unsqueeze_989 = None
    sub_319: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_33, mul_1354);  where_33 = mul_1354 = None
    sub_320: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_319, unsqueeze_986);  sub_319 = unsqueeze_986 = None
    mul_1355: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_992);  sub_320 = unsqueeze_992 = None
    mul_1356: "f32[128]" = torch.ops.aten.mul.Tensor(sum_188, squeeze_313);  sum_188 = squeeze_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1355, mean_24, primals_385, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1355 = mean_24 = primals_385 = None
    getitem_409: "f32[8, 256, 1, 1]" = convolution_backward_43[0]
    getitem_410: "f32[128, 256, 1, 1]" = convolution_backward_43[1]
    getitem_411: "f32[128]" = convolution_backward_43[2];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_26: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_409, [8, 256, 16, 16]);  getitem_409 = None
    div_42: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_26, 256);  expand_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_993: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_42, 1);  div_42 = None
    expand_27: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_993, [8, 2, 256, 16, 16]);  unsqueeze_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_744: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1345, expand_27);  mul_1345 = expand_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_245: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_744, [8, 512, 16, 16]);  add_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_280: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_100);  relu_100 = None
    alias_281: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_280);  alias_280 = None
    le_34: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_281, 0);  alias_281 = None
    where_34: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_34, full_default, view_245);  le_34 = view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_189: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_321: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_996);  convolution_127 = unsqueeze_996 = None
    mul_1357: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_34, sub_321)
    sum_190: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1357, [0, 2, 3]);  mul_1357 = None
    mul_1358: "f32[512]" = torch.ops.aten.mul.Tensor(sum_189, 0.00048828125)
    unsqueeze_997: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1358, 0);  mul_1358 = None
    unsqueeze_998: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 2);  unsqueeze_997 = None
    unsqueeze_999: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 3);  unsqueeze_998 = None
    mul_1359: "f32[512]" = torch.ops.aten.mul.Tensor(sum_190, 0.00048828125)
    mul_1360: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_310, squeeze_310)
    mul_1361: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1359, mul_1360);  mul_1359 = mul_1360 = None
    unsqueeze_1000: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1361, 0);  mul_1361 = None
    unsqueeze_1001: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 2);  unsqueeze_1000 = None
    unsqueeze_1002: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 3);  unsqueeze_1001 = None
    mul_1362: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_310, primals_383);  primals_383 = None
    unsqueeze_1003: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1362, 0);  mul_1362 = None
    unsqueeze_1004: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 2);  unsqueeze_1003 = None
    unsqueeze_1005: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 3);  unsqueeze_1004 = None
    mul_1363: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_1002);  sub_321 = unsqueeze_1002 = None
    sub_323: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_34, mul_1363);  where_34 = mul_1363 = None
    sub_324: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_999);  sub_323 = unsqueeze_999 = None
    mul_1364: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1005);  sub_324 = unsqueeze_1005 = None
    mul_1365: "f32[512]" = torch.ops.aten.mul.Tensor(sum_190, squeeze_310);  sum_190 = squeeze_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1364, relu_99, primals_382, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1364 = primals_382 = None
    getitem_412: "f32[8, 256, 16, 16]" = convolution_backward_44[0]
    getitem_413: "f32[512, 128, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_283: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_99);  relu_99 = None
    alias_284: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_283);  alias_283 = None
    le_35: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_284, 0);  alias_284 = None
    where_35: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_35, full_default, getitem_412);  le_35 = getitem_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_191: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_325: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_1008);  convolution_126 = unsqueeze_1008 = None
    mul_1366: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_35, sub_325)
    sum_192: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1366, [0, 2, 3]);  mul_1366 = None
    mul_1367: "f32[256]" = torch.ops.aten.mul.Tensor(sum_191, 0.00048828125)
    unsqueeze_1009: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1367, 0);  mul_1367 = None
    unsqueeze_1010: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 2);  unsqueeze_1009 = None
    unsqueeze_1011: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 3);  unsqueeze_1010 = None
    mul_1368: "f32[256]" = torch.ops.aten.mul.Tensor(sum_192, 0.00048828125)
    mul_1369: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_307, squeeze_307)
    mul_1370: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1368, mul_1369);  mul_1368 = mul_1369 = None
    unsqueeze_1012: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1370, 0);  mul_1370 = None
    unsqueeze_1013: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 2);  unsqueeze_1012 = None
    unsqueeze_1014: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 3);  unsqueeze_1013 = None
    mul_1371: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_307, primals_380);  primals_380 = None
    unsqueeze_1015: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1371, 0);  mul_1371 = None
    unsqueeze_1016: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 2);  unsqueeze_1015 = None
    unsqueeze_1017: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 3);  unsqueeze_1016 = None
    mul_1372: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_1014);  sub_325 = unsqueeze_1014 = None
    sub_327: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_35, mul_1372);  where_35 = mul_1372 = None
    sub_328: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_327, unsqueeze_1011);  sub_327 = unsqueeze_1011 = None
    mul_1373: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1017);  sub_328 = unsqueeze_1017 = None
    mul_1374: "f32[256]" = torch.ops.aten.mul.Tensor(sum_192, squeeze_307);  sum_192 = squeeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1373, relu_98, primals_379, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1373 = primals_379 = None
    getitem_415: "f32[8, 1024, 16, 16]" = convolution_backward_45[0]
    getitem_416: "f32[256, 1024, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_745: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_32, getitem_415);  where_32 = getitem_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_286: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_98);  relu_98 = None
    alias_287: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_286);  alias_286 = None
    le_36: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_287, 0);  alias_287 = None
    where_36: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_36, full_default, add_745);  le_36 = add_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_193: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_329: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_1020);  convolution_125 = unsqueeze_1020 = None
    mul_1375: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_36, sub_329)
    sum_194: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1375, [0, 2, 3]);  mul_1375 = None
    mul_1376: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_193, 0.00048828125)
    unsqueeze_1021: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1376, 0);  mul_1376 = None
    unsqueeze_1022: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 2);  unsqueeze_1021 = None
    unsqueeze_1023: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 3);  unsqueeze_1022 = None
    mul_1377: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_194, 0.00048828125)
    mul_1378: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_304, squeeze_304)
    mul_1379: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1377, mul_1378);  mul_1377 = mul_1378 = None
    unsqueeze_1024: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1379, 0);  mul_1379 = None
    unsqueeze_1025: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 2);  unsqueeze_1024 = None
    unsqueeze_1026: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 3);  unsqueeze_1025 = None
    mul_1380: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_304, primals_377);  primals_377 = None
    unsqueeze_1027: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1380, 0);  mul_1380 = None
    unsqueeze_1028: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 2);  unsqueeze_1027 = None
    unsqueeze_1029: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 3);  unsqueeze_1028 = None
    mul_1381: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_1026);  sub_329 = unsqueeze_1026 = None
    sub_331: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_36, mul_1381);  mul_1381 = None
    sub_332: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_331, unsqueeze_1023);  sub_331 = unsqueeze_1023 = None
    mul_1382: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1029);  sub_332 = unsqueeze_1029 = None
    mul_1383: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_194, squeeze_304);  sum_194 = squeeze_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1382, sum_72, primals_376, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1382 = sum_72 = primals_376 = None
    getitem_418: "f32[8, 256, 16, 16]" = convolution_backward_46[0]
    getitem_419: "f32[1024, 256, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1030: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_418, 1);  getitem_418 = None
    expand_28: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1030, [8, 2, 256, 16, 16]);  unsqueeze_1030 = None
    mul_1384: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_28, view_139);  view_139 = None
    mul_1385: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_28, view_143);  expand_28 = view_143 = None
    sum_195: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1384, [3, 4], True);  mul_1384 = None
    view_246: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_195, [8, 512, 1, 1]);  sum_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_247: "f32[8, 512]" = torch.ops.aten.view.default(view_246, [8, 512]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_248: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_247, [8, 2, 1, 256]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_288: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_121);  alias_121 = None
    mul_1386: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_248, alias_288);  view_248 = None
    sum_196: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1386, [1], True)
    mul_1387: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_288, sum_196);  alias_288 = sum_196 = None
    sub_333: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1386, mul_1387);  mul_1386 = mul_1387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_47: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_333, [0, 2, 1, 3]);  sub_333 = None
    view_249: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_47, [8, 512, 1, 1]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(view_249, relu_97, primals_374, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_249 = primals_374 = None
    getitem_421: "f32[8, 128, 1, 1]" = convolution_backward_47[0]
    getitem_422: "f32[512, 128, 1, 1]" = convolution_backward_47[1]
    getitem_423: "f32[512]" = convolution_backward_47[2];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_290: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_97);  relu_97 = None
    alias_291: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_290);  alias_290 = None
    le_37: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_291, 0);  alias_291 = None
    where_37: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_37, full_default, getitem_421);  le_37 = getitem_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1031: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_300, 0);  squeeze_300 = None
    unsqueeze_1032: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 2);  unsqueeze_1031 = None
    unsqueeze_1033: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 3);  unsqueeze_1032 = None
    sum_197: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_334: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_1033);  convolution_123 = unsqueeze_1033 = None
    mul_1388: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_37, sub_334)
    sum_198: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1388, [0, 2, 3]);  mul_1388 = None
    mul_1389: "f32[128]" = torch.ops.aten.mul.Tensor(sum_197, 0.125)
    unsqueeze_1034: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1389, 0);  mul_1389 = None
    unsqueeze_1035: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 2);  unsqueeze_1034 = None
    unsqueeze_1036: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 3);  unsqueeze_1035 = None
    mul_1390: "f32[128]" = torch.ops.aten.mul.Tensor(sum_198, 0.125)
    mul_1391: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_301, squeeze_301)
    mul_1392: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1390, mul_1391);  mul_1390 = mul_1391 = None
    unsqueeze_1037: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1392, 0);  mul_1392 = None
    unsqueeze_1038: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 2);  unsqueeze_1037 = None
    unsqueeze_1039: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 3);  unsqueeze_1038 = None
    mul_1393: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_301, primals_372);  primals_372 = None
    unsqueeze_1040: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1393, 0);  mul_1393 = None
    unsqueeze_1041: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 2);  unsqueeze_1040 = None
    unsqueeze_1042: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 3);  unsqueeze_1041 = None
    mul_1394: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_1039);  sub_334 = unsqueeze_1039 = None
    sub_336: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_37, mul_1394);  where_37 = mul_1394 = None
    sub_337: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_336, unsqueeze_1036);  sub_336 = unsqueeze_1036 = None
    mul_1395: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_1042);  sub_337 = unsqueeze_1042 = None
    mul_1396: "f32[128]" = torch.ops.aten.mul.Tensor(sum_198, squeeze_301);  sum_198 = squeeze_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1395, mean_23, primals_370, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1395 = mean_23 = primals_370 = None
    getitem_424: "f32[8, 256, 1, 1]" = convolution_backward_48[0]
    getitem_425: "f32[128, 256, 1, 1]" = convolution_backward_48[1]
    getitem_426: "f32[128]" = convolution_backward_48[2];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_29: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_424, [8, 256, 16, 16]);  getitem_424 = None
    div_43: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_29, 256);  expand_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1043: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_43, 1);  div_43 = None
    expand_30: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1043, [8, 2, 256, 16, 16]);  unsqueeze_1043 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_746: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1385, expand_30);  mul_1385 = expand_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_250: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_746, [8, 512, 16, 16]);  add_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_293: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_96);  relu_96 = None
    alias_294: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_293);  alias_293 = None
    le_38: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_294, 0);  alias_294 = None
    where_38: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_38, full_default, view_250);  le_38 = view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_199: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_338: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_1046);  convolution_122 = unsqueeze_1046 = None
    mul_1397: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_38, sub_338)
    sum_200: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1397, [0, 2, 3]);  mul_1397 = None
    mul_1398: "f32[512]" = torch.ops.aten.mul.Tensor(sum_199, 0.00048828125)
    unsqueeze_1047: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1398, 0);  mul_1398 = None
    unsqueeze_1048: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 2);  unsqueeze_1047 = None
    unsqueeze_1049: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 3);  unsqueeze_1048 = None
    mul_1399: "f32[512]" = torch.ops.aten.mul.Tensor(sum_200, 0.00048828125)
    mul_1400: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_298, squeeze_298)
    mul_1401: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1399, mul_1400);  mul_1399 = mul_1400 = None
    unsqueeze_1050: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1401, 0);  mul_1401 = None
    unsqueeze_1051: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 2);  unsqueeze_1050 = None
    unsqueeze_1052: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 3);  unsqueeze_1051 = None
    mul_1402: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_298, primals_368);  primals_368 = None
    unsqueeze_1053: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1402, 0);  mul_1402 = None
    unsqueeze_1054: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 2);  unsqueeze_1053 = None
    unsqueeze_1055: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 3);  unsqueeze_1054 = None
    mul_1403: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_1052);  sub_338 = unsqueeze_1052 = None
    sub_340: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_38, mul_1403);  where_38 = mul_1403 = None
    sub_341: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_340, unsqueeze_1049);  sub_340 = unsqueeze_1049 = None
    mul_1404: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_1055);  sub_341 = unsqueeze_1055 = None
    mul_1405: "f32[512]" = torch.ops.aten.mul.Tensor(sum_200, squeeze_298);  sum_200 = squeeze_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1404, relu_95, primals_367, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1404 = primals_367 = None
    getitem_427: "f32[8, 256, 16, 16]" = convolution_backward_49[0]
    getitem_428: "f32[512, 128, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_296: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_95);  relu_95 = None
    alias_297: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_296);  alias_296 = None
    le_39: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_297, 0);  alias_297 = None
    where_39: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_39, full_default, getitem_427);  le_39 = getitem_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_201: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_342: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_1058);  convolution_121 = unsqueeze_1058 = None
    mul_1406: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_39, sub_342)
    sum_202: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1406, [0, 2, 3]);  mul_1406 = None
    mul_1407: "f32[256]" = torch.ops.aten.mul.Tensor(sum_201, 0.00048828125)
    unsqueeze_1059: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1407, 0);  mul_1407 = None
    unsqueeze_1060: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 2);  unsqueeze_1059 = None
    unsqueeze_1061: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 3);  unsqueeze_1060 = None
    mul_1408: "f32[256]" = torch.ops.aten.mul.Tensor(sum_202, 0.00048828125)
    mul_1409: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_295, squeeze_295)
    mul_1410: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1408, mul_1409);  mul_1408 = mul_1409 = None
    unsqueeze_1062: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1410, 0);  mul_1410 = None
    unsqueeze_1063: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 2);  unsqueeze_1062 = None
    unsqueeze_1064: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 3);  unsqueeze_1063 = None
    mul_1411: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_295, primals_365);  primals_365 = None
    unsqueeze_1065: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1411, 0);  mul_1411 = None
    unsqueeze_1066: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 2);  unsqueeze_1065 = None
    unsqueeze_1067: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 3);  unsqueeze_1066 = None
    mul_1412: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_1064);  sub_342 = unsqueeze_1064 = None
    sub_344: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_39, mul_1412);  where_39 = mul_1412 = None
    sub_345: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_344, unsqueeze_1061);  sub_344 = unsqueeze_1061 = None
    mul_1413: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_1067);  sub_345 = unsqueeze_1067 = None
    mul_1414: "f32[256]" = torch.ops.aten.mul.Tensor(sum_202, squeeze_295);  sum_202 = squeeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1413, relu_94, primals_364, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1413 = primals_364 = None
    getitem_430: "f32[8, 1024, 16, 16]" = convolution_backward_50[0]
    getitem_431: "f32[256, 1024, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_747: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_36, getitem_430);  where_36 = getitem_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_299: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_94);  relu_94 = None
    alias_300: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_299);  alias_299 = None
    le_40: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_300, 0);  alias_300 = None
    where_40: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_40, full_default, add_747);  le_40 = add_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_203: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_346: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_1070);  convolution_120 = unsqueeze_1070 = None
    mul_1415: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_40, sub_346)
    sum_204: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1415, [0, 2, 3]);  mul_1415 = None
    mul_1416: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_203, 0.00048828125)
    unsqueeze_1071: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1416, 0);  mul_1416 = None
    unsqueeze_1072: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 2);  unsqueeze_1071 = None
    unsqueeze_1073: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 3);  unsqueeze_1072 = None
    mul_1417: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_204, 0.00048828125)
    mul_1418: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_292, squeeze_292)
    mul_1419: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1417, mul_1418);  mul_1417 = mul_1418 = None
    unsqueeze_1074: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1419, 0);  mul_1419 = None
    unsqueeze_1075: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 2);  unsqueeze_1074 = None
    unsqueeze_1076: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 3);  unsqueeze_1075 = None
    mul_1420: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_292, primals_362);  primals_362 = None
    unsqueeze_1077: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1420, 0);  mul_1420 = None
    unsqueeze_1078: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 2);  unsqueeze_1077 = None
    unsqueeze_1079: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 3);  unsqueeze_1078 = None
    mul_1421: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_1076);  sub_346 = unsqueeze_1076 = None
    sub_348: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_40, mul_1421);  mul_1421 = None
    sub_349: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_348, unsqueeze_1073);  sub_348 = unsqueeze_1073 = None
    mul_1422: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_1079);  sub_349 = unsqueeze_1079 = None
    mul_1423: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_204, squeeze_292);  sum_204 = squeeze_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1422, sum_69, primals_361, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1422 = sum_69 = primals_361 = None
    getitem_433: "f32[8, 256, 16, 16]" = convolution_backward_51[0]
    getitem_434: "f32[1024, 256, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1080: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_433, 1);  getitem_433 = None
    expand_31: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1080, [8, 2, 256, 16, 16]);  unsqueeze_1080 = None
    mul_1424: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_31, view_133);  view_133 = None
    mul_1425: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_31, view_137);  expand_31 = view_137 = None
    sum_205: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1424, [3, 4], True);  mul_1424 = None
    view_251: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_205, [8, 512, 1, 1]);  sum_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_252: "f32[8, 512]" = torch.ops.aten.view.default(view_251, [8, 512]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_253: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_252, [8, 2, 1, 256]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_301: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_116);  alias_116 = None
    mul_1426: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_253, alias_301);  view_253 = None
    sum_206: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1426, [1], True)
    mul_1427: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_301, sum_206);  alias_301 = sum_206 = None
    sub_350: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1426, mul_1427);  mul_1426 = mul_1427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_48: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_350, [0, 2, 1, 3]);  sub_350 = None
    view_254: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_48, [8, 512, 1, 1]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(view_254, relu_93, primals_359, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_254 = primals_359 = None
    getitem_436: "f32[8, 128, 1, 1]" = convolution_backward_52[0]
    getitem_437: "f32[512, 128, 1, 1]" = convolution_backward_52[1]
    getitem_438: "f32[512]" = convolution_backward_52[2];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_303: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_93);  relu_93 = None
    alias_304: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_303);  alias_303 = None
    le_41: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_304, 0);  alias_304 = None
    where_41: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_41, full_default, getitem_436);  le_41 = getitem_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1081: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_288, 0);  squeeze_288 = None
    unsqueeze_1082: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1081, 2);  unsqueeze_1081 = None
    unsqueeze_1083: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 3);  unsqueeze_1082 = None
    sum_207: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_351: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_1083);  convolution_118 = unsqueeze_1083 = None
    mul_1428: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_41, sub_351)
    sum_208: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1428, [0, 2, 3]);  mul_1428 = None
    mul_1429: "f32[128]" = torch.ops.aten.mul.Tensor(sum_207, 0.125)
    unsqueeze_1084: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1429, 0);  mul_1429 = None
    unsqueeze_1085: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 2);  unsqueeze_1084 = None
    unsqueeze_1086: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 3);  unsqueeze_1085 = None
    mul_1430: "f32[128]" = torch.ops.aten.mul.Tensor(sum_208, 0.125)
    mul_1431: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_289, squeeze_289)
    mul_1432: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1430, mul_1431);  mul_1430 = mul_1431 = None
    unsqueeze_1087: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1432, 0);  mul_1432 = None
    unsqueeze_1088: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 2);  unsqueeze_1087 = None
    unsqueeze_1089: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, 3);  unsqueeze_1088 = None
    mul_1433: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_289, primals_357);  primals_357 = None
    unsqueeze_1090: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1433, 0);  mul_1433 = None
    unsqueeze_1091: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 2);  unsqueeze_1090 = None
    unsqueeze_1092: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1091, 3);  unsqueeze_1091 = None
    mul_1434: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_1089);  sub_351 = unsqueeze_1089 = None
    sub_353: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_41, mul_1434);  where_41 = mul_1434 = None
    sub_354: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_353, unsqueeze_1086);  sub_353 = unsqueeze_1086 = None
    mul_1435: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_354, unsqueeze_1092);  sub_354 = unsqueeze_1092 = None
    mul_1436: "f32[128]" = torch.ops.aten.mul.Tensor(sum_208, squeeze_289);  sum_208 = squeeze_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1435, mean_22, primals_355, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1435 = mean_22 = primals_355 = None
    getitem_439: "f32[8, 256, 1, 1]" = convolution_backward_53[0]
    getitem_440: "f32[128, 256, 1, 1]" = convolution_backward_53[1]
    getitem_441: "f32[128]" = convolution_backward_53[2];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_32: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_439, [8, 256, 16, 16]);  getitem_439 = None
    div_44: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_32, 256);  expand_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1093: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_44, 1);  div_44 = None
    expand_33: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1093, [8, 2, 256, 16, 16]);  unsqueeze_1093 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_748: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1425, expand_33);  mul_1425 = expand_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_255: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_748, [8, 512, 16, 16]);  add_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_306: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_92);  relu_92 = None
    alias_307: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_306);  alias_306 = None
    le_42: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_307, 0);  alias_307 = None
    where_42: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_42, full_default, view_255);  le_42 = view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_209: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_355: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_1096);  convolution_117 = unsqueeze_1096 = None
    mul_1437: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_42, sub_355)
    sum_210: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1437, [0, 2, 3]);  mul_1437 = None
    mul_1438: "f32[512]" = torch.ops.aten.mul.Tensor(sum_209, 0.00048828125)
    unsqueeze_1097: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1438, 0);  mul_1438 = None
    unsqueeze_1098: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 2);  unsqueeze_1097 = None
    unsqueeze_1099: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 3);  unsqueeze_1098 = None
    mul_1439: "f32[512]" = torch.ops.aten.mul.Tensor(sum_210, 0.00048828125)
    mul_1440: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
    mul_1441: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1439, mul_1440);  mul_1439 = mul_1440 = None
    unsqueeze_1100: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1441, 0);  mul_1441 = None
    unsqueeze_1101: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, 2);  unsqueeze_1100 = None
    unsqueeze_1102: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 3);  unsqueeze_1101 = None
    mul_1442: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_286, primals_353);  primals_353 = None
    unsqueeze_1103: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1442, 0);  mul_1442 = None
    unsqueeze_1104: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1103, 2);  unsqueeze_1103 = None
    unsqueeze_1105: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, 3);  unsqueeze_1104 = None
    mul_1443: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_355, unsqueeze_1102);  sub_355 = unsqueeze_1102 = None
    sub_357: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_42, mul_1443);  where_42 = mul_1443 = None
    sub_358: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_357, unsqueeze_1099);  sub_357 = unsqueeze_1099 = None
    mul_1444: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_358, unsqueeze_1105);  sub_358 = unsqueeze_1105 = None
    mul_1445: "f32[512]" = torch.ops.aten.mul.Tensor(sum_210, squeeze_286);  sum_210 = squeeze_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1444, relu_91, primals_352, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1444 = primals_352 = None
    getitem_442: "f32[8, 256, 16, 16]" = convolution_backward_54[0]
    getitem_443: "f32[512, 128, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_309: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_91);  relu_91 = None
    alias_310: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_309);  alias_309 = None
    le_43: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_310, 0);  alias_310 = None
    where_43: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_43, full_default, getitem_442);  le_43 = getitem_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_211: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_359: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_1108);  convolution_116 = unsqueeze_1108 = None
    mul_1446: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_43, sub_359)
    sum_212: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1446, [0, 2, 3]);  mul_1446 = None
    mul_1447: "f32[256]" = torch.ops.aten.mul.Tensor(sum_211, 0.00048828125)
    unsqueeze_1109: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1447, 0);  mul_1447 = None
    unsqueeze_1110: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 2);  unsqueeze_1109 = None
    unsqueeze_1111: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 3);  unsqueeze_1110 = None
    mul_1448: "f32[256]" = torch.ops.aten.mul.Tensor(sum_212, 0.00048828125)
    mul_1449: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
    mul_1450: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1448, mul_1449);  mul_1448 = mul_1449 = None
    unsqueeze_1112: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1450, 0);  mul_1450 = None
    unsqueeze_1113: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, 2);  unsqueeze_1112 = None
    unsqueeze_1114: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 3);  unsqueeze_1113 = None
    mul_1451: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_283, primals_350);  primals_350 = None
    unsqueeze_1115: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1451, 0);  mul_1451 = None
    unsqueeze_1116: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1115, 2);  unsqueeze_1115 = None
    unsqueeze_1117: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, 3);  unsqueeze_1116 = None
    mul_1452: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_359, unsqueeze_1114);  sub_359 = unsqueeze_1114 = None
    sub_361: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_43, mul_1452);  where_43 = mul_1452 = None
    sub_362: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_361, unsqueeze_1111);  sub_361 = unsqueeze_1111 = None
    mul_1453: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_362, unsqueeze_1117);  sub_362 = unsqueeze_1117 = None
    mul_1454: "f32[256]" = torch.ops.aten.mul.Tensor(sum_212, squeeze_283);  sum_212 = squeeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1453, relu_90, primals_349, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1453 = primals_349 = None
    getitem_445: "f32[8, 1024, 16, 16]" = convolution_backward_55[0]
    getitem_446: "f32[256, 1024, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_749: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_40, getitem_445);  where_40 = getitem_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_312: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_90);  relu_90 = None
    alias_313: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_312);  alias_312 = None
    le_44: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_313, 0);  alias_313 = None
    where_44: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_44, full_default, add_749);  le_44 = add_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_213: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_363: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_1120);  convolution_115 = unsqueeze_1120 = None
    mul_1455: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_44, sub_363)
    sum_214: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1455, [0, 2, 3]);  mul_1455 = None
    mul_1456: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_213, 0.00048828125)
    unsqueeze_1121: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1456, 0);  mul_1456 = None
    unsqueeze_1122: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 2);  unsqueeze_1121 = None
    unsqueeze_1123: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 3);  unsqueeze_1122 = None
    mul_1457: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_214, 0.00048828125)
    mul_1458: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
    mul_1459: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1457, mul_1458);  mul_1457 = mul_1458 = None
    unsqueeze_1124: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1459, 0);  mul_1459 = None
    unsqueeze_1125: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, 2);  unsqueeze_1124 = None
    unsqueeze_1126: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 3);  unsqueeze_1125 = None
    mul_1460: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_280, primals_347);  primals_347 = None
    unsqueeze_1127: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1460, 0);  mul_1460 = None
    unsqueeze_1128: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1127, 2);  unsqueeze_1127 = None
    unsqueeze_1129: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, 3);  unsqueeze_1128 = None
    mul_1461: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_363, unsqueeze_1126);  sub_363 = unsqueeze_1126 = None
    sub_365: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_44, mul_1461);  mul_1461 = None
    sub_366: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_365, unsqueeze_1123);  sub_365 = unsqueeze_1123 = None
    mul_1462: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_366, unsqueeze_1129);  sub_366 = unsqueeze_1129 = None
    mul_1463: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_214, squeeze_280);  sum_214 = squeeze_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1462, sum_66, primals_346, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1462 = sum_66 = primals_346 = None
    getitem_448: "f32[8, 256, 16, 16]" = convolution_backward_56[0]
    getitem_449: "f32[1024, 256, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1130: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_448, 1);  getitem_448 = None
    expand_34: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1130, [8, 2, 256, 16, 16]);  unsqueeze_1130 = None
    mul_1464: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_34, view_127);  view_127 = None
    mul_1465: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_34, view_131);  expand_34 = view_131 = None
    sum_215: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1464, [3, 4], True);  mul_1464 = None
    view_256: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_215, [8, 512, 1, 1]);  sum_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_257: "f32[8, 512]" = torch.ops.aten.view.default(view_256, [8, 512]);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_258: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_257, [8, 2, 1, 256]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_314: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_111);  alias_111 = None
    mul_1466: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_258, alias_314);  view_258 = None
    sum_216: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1466, [1], True)
    mul_1467: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_314, sum_216);  alias_314 = sum_216 = None
    sub_367: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1466, mul_1467);  mul_1466 = mul_1467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_49: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_367, [0, 2, 1, 3]);  sub_367 = None
    view_259: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_49, [8, 512, 1, 1]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(view_259, relu_89, primals_344, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_259 = primals_344 = None
    getitem_451: "f32[8, 128, 1, 1]" = convolution_backward_57[0]
    getitem_452: "f32[512, 128, 1, 1]" = convolution_backward_57[1]
    getitem_453: "f32[512]" = convolution_backward_57[2];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_316: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_89);  relu_89 = None
    alias_317: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_316);  alias_316 = None
    le_45: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_317, 0);  alias_317 = None
    where_45: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_45, full_default, getitem_451);  le_45 = getitem_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1131: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_276, 0);  squeeze_276 = None
    unsqueeze_1132: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 2);  unsqueeze_1131 = None
    unsqueeze_1133: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 3);  unsqueeze_1132 = None
    sum_217: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_368: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_1133);  convolution_113 = unsqueeze_1133 = None
    mul_1468: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_45, sub_368)
    sum_218: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1468, [0, 2, 3]);  mul_1468 = None
    mul_1469: "f32[128]" = torch.ops.aten.mul.Tensor(sum_217, 0.125)
    unsqueeze_1134: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1469, 0);  mul_1469 = None
    unsqueeze_1135: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 2);  unsqueeze_1134 = None
    unsqueeze_1136: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 3);  unsqueeze_1135 = None
    mul_1470: "f32[128]" = torch.ops.aten.mul.Tensor(sum_218, 0.125)
    mul_1471: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
    mul_1472: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1470, mul_1471);  mul_1470 = mul_1471 = None
    unsqueeze_1137: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1472, 0);  mul_1472 = None
    unsqueeze_1138: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 2);  unsqueeze_1137 = None
    unsqueeze_1139: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 3);  unsqueeze_1138 = None
    mul_1473: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_277, primals_342);  primals_342 = None
    unsqueeze_1140: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1473, 0);  mul_1473 = None
    unsqueeze_1141: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, 2);  unsqueeze_1140 = None
    unsqueeze_1142: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1141, 3);  unsqueeze_1141 = None
    mul_1474: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_1139);  sub_368 = unsqueeze_1139 = None
    sub_370: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_45, mul_1474);  where_45 = mul_1474 = None
    sub_371: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_370, unsqueeze_1136);  sub_370 = unsqueeze_1136 = None
    mul_1475: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_371, unsqueeze_1142);  sub_371 = unsqueeze_1142 = None
    mul_1476: "f32[128]" = torch.ops.aten.mul.Tensor(sum_218, squeeze_277);  sum_218 = squeeze_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1475, mean_21, primals_340, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1475 = mean_21 = primals_340 = None
    getitem_454: "f32[8, 256, 1, 1]" = convolution_backward_58[0]
    getitem_455: "f32[128, 256, 1, 1]" = convolution_backward_58[1]
    getitem_456: "f32[128]" = convolution_backward_58[2];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_35: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_454, [8, 256, 16, 16]);  getitem_454 = None
    div_45: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_35, 256);  expand_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1143: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_45, 1);  div_45 = None
    expand_36: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1143, [8, 2, 256, 16, 16]);  unsqueeze_1143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_750: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1465, expand_36);  mul_1465 = expand_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_260: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_750, [8, 512, 16, 16]);  add_750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_319: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_88);  relu_88 = None
    alias_320: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_319);  alias_319 = None
    le_46: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_320, 0);  alias_320 = None
    where_46: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_46, full_default, view_260);  le_46 = view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_219: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_372: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_1146);  convolution_112 = unsqueeze_1146 = None
    mul_1477: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_46, sub_372)
    sum_220: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1477, [0, 2, 3]);  mul_1477 = None
    mul_1478: "f32[512]" = torch.ops.aten.mul.Tensor(sum_219, 0.00048828125)
    unsqueeze_1147: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1478, 0);  mul_1478 = None
    unsqueeze_1148: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 2);  unsqueeze_1147 = None
    unsqueeze_1149: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, 3);  unsqueeze_1148 = None
    mul_1479: "f32[512]" = torch.ops.aten.mul.Tensor(sum_220, 0.00048828125)
    mul_1480: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
    mul_1481: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1479, mul_1480);  mul_1479 = mul_1480 = None
    unsqueeze_1150: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1481, 0);  mul_1481 = None
    unsqueeze_1151: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 2);  unsqueeze_1150 = None
    unsqueeze_1152: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1151, 3);  unsqueeze_1151 = None
    mul_1482: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_274, primals_338);  primals_338 = None
    unsqueeze_1153: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1482, 0);  mul_1482 = None
    unsqueeze_1154: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1153, 2);  unsqueeze_1153 = None
    unsqueeze_1155: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 3);  unsqueeze_1154 = None
    mul_1483: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_1152);  sub_372 = unsqueeze_1152 = None
    sub_374: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_46, mul_1483);  where_46 = mul_1483 = None
    sub_375: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_374, unsqueeze_1149);  sub_374 = unsqueeze_1149 = None
    mul_1484: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_375, unsqueeze_1155);  sub_375 = unsqueeze_1155 = None
    mul_1485: "f32[512]" = torch.ops.aten.mul.Tensor(sum_220, squeeze_274);  sum_220 = squeeze_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1484, relu_87, primals_337, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1484 = primals_337 = None
    getitem_457: "f32[8, 256, 16, 16]" = convolution_backward_59[0]
    getitem_458: "f32[512, 128, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_322: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_87);  relu_87 = None
    alias_323: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_322);  alias_322 = None
    le_47: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_323, 0);  alias_323 = None
    where_47: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_47, full_default, getitem_457);  le_47 = getitem_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_221: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_376: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_1158);  convolution_111 = unsqueeze_1158 = None
    mul_1486: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_47, sub_376)
    sum_222: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1486, [0, 2, 3]);  mul_1486 = None
    mul_1487: "f32[256]" = torch.ops.aten.mul.Tensor(sum_221, 0.00048828125)
    unsqueeze_1159: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1487, 0);  mul_1487 = None
    unsqueeze_1160: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 2);  unsqueeze_1159 = None
    unsqueeze_1161: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, 3);  unsqueeze_1160 = None
    mul_1488: "f32[256]" = torch.ops.aten.mul.Tensor(sum_222, 0.00048828125)
    mul_1489: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
    mul_1490: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1488, mul_1489);  mul_1488 = mul_1489 = None
    unsqueeze_1162: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1490, 0);  mul_1490 = None
    unsqueeze_1163: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 2);  unsqueeze_1162 = None
    unsqueeze_1164: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1163, 3);  unsqueeze_1163 = None
    mul_1491: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_271, primals_335);  primals_335 = None
    unsqueeze_1165: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1491, 0);  mul_1491 = None
    unsqueeze_1166: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1165, 2);  unsqueeze_1165 = None
    unsqueeze_1167: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 3);  unsqueeze_1166 = None
    mul_1492: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_1164);  sub_376 = unsqueeze_1164 = None
    sub_378: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_47, mul_1492);  where_47 = mul_1492 = None
    sub_379: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_378, unsqueeze_1161);  sub_378 = unsqueeze_1161 = None
    mul_1493: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_379, unsqueeze_1167);  sub_379 = unsqueeze_1167 = None
    mul_1494: "f32[256]" = torch.ops.aten.mul.Tensor(sum_222, squeeze_271);  sum_222 = squeeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1493, relu_86, primals_334, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1493 = primals_334 = None
    getitem_460: "f32[8, 1024, 16, 16]" = convolution_backward_60[0]
    getitem_461: "f32[256, 1024, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_751: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_44, getitem_460);  where_44 = getitem_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_325: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_86);  relu_86 = None
    alias_326: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_325);  alias_325 = None
    le_48: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_326, 0);  alias_326 = None
    where_48: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_48, full_default, add_751);  le_48 = add_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_223: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_380: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_1170);  convolution_110 = unsqueeze_1170 = None
    mul_1495: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_48, sub_380)
    sum_224: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1495, [0, 2, 3]);  mul_1495 = None
    mul_1496: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_223, 0.00048828125)
    unsqueeze_1171: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1496, 0);  mul_1496 = None
    unsqueeze_1172: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 2);  unsqueeze_1171 = None
    unsqueeze_1173: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, 3);  unsqueeze_1172 = None
    mul_1497: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_224, 0.00048828125)
    mul_1498: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
    mul_1499: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1497, mul_1498);  mul_1497 = mul_1498 = None
    unsqueeze_1174: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1499, 0);  mul_1499 = None
    unsqueeze_1175: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 2);  unsqueeze_1174 = None
    unsqueeze_1176: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1175, 3);  unsqueeze_1175 = None
    mul_1500: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_268, primals_332);  primals_332 = None
    unsqueeze_1177: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1500, 0);  mul_1500 = None
    unsqueeze_1178: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1177, 2);  unsqueeze_1177 = None
    unsqueeze_1179: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 3);  unsqueeze_1178 = None
    mul_1501: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_1176);  sub_380 = unsqueeze_1176 = None
    sub_382: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_48, mul_1501);  mul_1501 = None
    sub_383: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_382, unsqueeze_1173);  sub_382 = unsqueeze_1173 = None
    mul_1502: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_383, unsqueeze_1179);  sub_383 = unsqueeze_1179 = None
    mul_1503: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_224, squeeze_268);  sum_224 = squeeze_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1502, sum_63, primals_331, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1502 = sum_63 = primals_331 = None
    getitem_463: "f32[8, 256, 16, 16]" = convolution_backward_61[0]
    getitem_464: "f32[1024, 256, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1180: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_463, 1);  getitem_463 = None
    expand_37: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1180, [8, 2, 256, 16, 16]);  unsqueeze_1180 = None
    mul_1504: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_37, view_121);  view_121 = None
    mul_1505: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_37, view_125);  expand_37 = view_125 = None
    sum_225: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1504, [3, 4], True);  mul_1504 = None
    view_261: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_225, [8, 512, 1, 1]);  sum_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_262: "f32[8, 512]" = torch.ops.aten.view.default(view_261, [8, 512]);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_263: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_262, [8, 2, 1, 256]);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_327: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_106);  alias_106 = None
    mul_1506: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_263, alias_327);  view_263 = None
    sum_226: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1506, [1], True)
    mul_1507: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_327, sum_226);  alias_327 = sum_226 = None
    sub_384: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1506, mul_1507);  mul_1506 = mul_1507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_50: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_384, [0, 2, 1, 3]);  sub_384 = None
    view_264: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_50, [8, 512, 1, 1]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(view_264, relu_85, primals_329, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_264 = primals_329 = None
    getitem_466: "f32[8, 128, 1, 1]" = convolution_backward_62[0]
    getitem_467: "f32[512, 128, 1, 1]" = convolution_backward_62[1]
    getitem_468: "f32[512]" = convolution_backward_62[2];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_329: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_85);  relu_85 = None
    alias_330: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_329);  alias_329 = None
    le_49: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_330, 0);  alias_330 = None
    where_49: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_49, full_default, getitem_466);  le_49 = getitem_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1181: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_264, 0);  squeeze_264 = None
    unsqueeze_1182: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 2);  unsqueeze_1181 = None
    unsqueeze_1183: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 3);  unsqueeze_1182 = None
    sum_227: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_385: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_1183);  convolution_108 = unsqueeze_1183 = None
    mul_1508: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_49, sub_385)
    sum_228: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1508, [0, 2, 3]);  mul_1508 = None
    mul_1509: "f32[128]" = torch.ops.aten.mul.Tensor(sum_227, 0.125)
    unsqueeze_1184: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1509, 0);  mul_1509 = None
    unsqueeze_1185: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, 2);  unsqueeze_1184 = None
    unsqueeze_1186: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 3);  unsqueeze_1185 = None
    mul_1510: "f32[128]" = torch.ops.aten.mul.Tensor(sum_228, 0.125)
    mul_1511: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
    mul_1512: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1510, mul_1511);  mul_1510 = mul_1511 = None
    unsqueeze_1187: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1512, 0);  mul_1512 = None
    unsqueeze_1188: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1187, 2);  unsqueeze_1187 = None
    unsqueeze_1189: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, 3);  unsqueeze_1188 = None
    mul_1513: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_265, primals_327);  primals_327 = None
    unsqueeze_1190: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1513, 0);  mul_1513 = None
    unsqueeze_1191: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 2);  unsqueeze_1190 = None
    unsqueeze_1192: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 3);  unsqueeze_1191 = None
    mul_1514: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_385, unsqueeze_1189);  sub_385 = unsqueeze_1189 = None
    sub_387: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_49, mul_1514);  where_49 = mul_1514 = None
    sub_388: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_387, unsqueeze_1186);  sub_387 = unsqueeze_1186 = None
    mul_1515: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_1192);  sub_388 = unsqueeze_1192 = None
    mul_1516: "f32[128]" = torch.ops.aten.mul.Tensor(sum_228, squeeze_265);  sum_228 = squeeze_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1515, mean_20, primals_325, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1515 = mean_20 = primals_325 = None
    getitem_469: "f32[8, 256, 1, 1]" = convolution_backward_63[0]
    getitem_470: "f32[128, 256, 1, 1]" = convolution_backward_63[1]
    getitem_471: "f32[128]" = convolution_backward_63[2];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_38: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_469, [8, 256, 16, 16]);  getitem_469 = None
    div_46: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_38, 256);  expand_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1193: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_46, 1);  div_46 = None
    expand_39: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1193, [8, 2, 256, 16, 16]);  unsqueeze_1193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_752: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1505, expand_39);  mul_1505 = expand_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_265: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_752, [8, 512, 16, 16]);  add_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_332: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_84);  relu_84 = None
    alias_333: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_332);  alias_332 = None
    le_50: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_333, 0);  alias_333 = None
    where_50: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_50, full_default, view_265);  le_50 = view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_229: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_389: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_1196);  convolution_107 = unsqueeze_1196 = None
    mul_1517: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_50, sub_389)
    sum_230: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1517, [0, 2, 3]);  mul_1517 = None
    mul_1518: "f32[512]" = torch.ops.aten.mul.Tensor(sum_229, 0.00048828125)
    unsqueeze_1197: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1518, 0);  mul_1518 = None
    unsqueeze_1198: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 2);  unsqueeze_1197 = None
    unsqueeze_1199: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 3);  unsqueeze_1198 = None
    mul_1519: "f32[512]" = torch.ops.aten.mul.Tensor(sum_230, 0.00048828125)
    mul_1520: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
    mul_1521: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1519, mul_1520);  mul_1519 = mul_1520 = None
    unsqueeze_1200: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1521, 0);  mul_1521 = None
    unsqueeze_1201: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, 2);  unsqueeze_1200 = None
    unsqueeze_1202: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1201, 3);  unsqueeze_1201 = None
    mul_1522: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_262, primals_323);  primals_323 = None
    unsqueeze_1203: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1522, 0);  mul_1522 = None
    unsqueeze_1204: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 2);  unsqueeze_1203 = None
    unsqueeze_1205: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 3);  unsqueeze_1204 = None
    mul_1523: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_389, unsqueeze_1202);  sub_389 = unsqueeze_1202 = None
    sub_391: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_50, mul_1523);  where_50 = mul_1523 = None
    sub_392: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_391, unsqueeze_1199);  sub_391 = unsqueeze_1199 = None
    mul_1524: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_1205);  sub_392 = unsqueeze_1205 = None
    mul_1525: "f32[512]" = torch.ops.aten.mul.Tensor(sum_230, squeeze_262);  sum_230 = squeeze_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1524, relu_83, primals_322, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1524 = primals_322 = None
    getitem_472: "f32[8, 256, 16, 16]" = convolution_backward_64[0]
    getitem_473: "f32[512, 128, 3, 3]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_335: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_83);  relu_83 = None
    alias_336: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_335);  alias_335 = None
    le_51: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_336, 0);  alias_336 = None
    where_51: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_51, full_default, getitem_472);  le_51 = getitem_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_231: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_393: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_1208);  convolution_106 = unsqueeze_1208 = None
    mul_1526: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_51, sub_393)
    sum_232: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1526, [0, 2, 3]);  mul_1526 = None
    mul_1527: "f32[256]" = torch.ops.aten.mul.Tensor(sum_231, 0.00048828125)
    unsqueeze_1209: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1527, 0);  mul_1527 = None
    unsqueeze_1210: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 2);  unsqueeze_1209 = None
    unsqueeze_1211: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 3);  unsqueeze_1210 = None
    mul_1528: "f32[256]" = torch.ops.aten.mul.Tensor(sum_232, 0.00048828125)
    mul_1529: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
    mul_1530: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1528, mul_1529);  mul_1528 = mul_1529 = None
    unsqueeze_1212: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1530, 0);  mul_1530 = None
    unsqueeze_1213: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, 2);  unsqueeze_1212 = None
    unsqueeze_1214: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1213, 3);  unsqueeze_1213 = None
    mul_1531: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_259, primals_320);  primals_320 = None
    unsqueeze_1215: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1531, 0);  mul_1531 = None
    unsqueeze_1216: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 2);  unsqueeze_1215 = None
    unsqueeze_1217: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 3);  unsqueeze_1216 = None
    mul_1532: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_393, unsqueeze_1214);  sub_393 = unsqueeze_1214 = None
    sub_395: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_51, mul_1532);  where_51 = mul_1532 = None
    sub_396: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_395, unsqueeze_1211);  sub_395 = unsqueeze_1211 = None
    mul_1533: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_1217);  sub_396 = unsqueeze_1217 = None
    mul_1534: "f32[256]" = torch.ops.aten.mul.Tensor(sum_232, squeeze_259);  sum_232 = squeeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1533, relu_82, primals_319, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1533 = primals_319 = None
    getitem_475: "f32[8, 1024, 16, 16]" = convolution_backward_65[0]
    getitem_476: "f32[256, 1024, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_753: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_48, getitem_475);  where_48 = getitem_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_338: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_82);  relu_82 = None
    alias_339: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_338);  alias_338 = None
    le_52: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_339, 0);  alias_339 = None
    where_52: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_52, full_default, add_753);  le_52 = add_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_233: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_397: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_1220);  convolution_105 = unsqueeze_1220 = None
    mul_1535: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_52, sub_397)
    sum_234: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1535, [0, 2, 3]);  mul_1535 = None
    mul_1536: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_233, 0.00048828125)
    unsqueeze_1221: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1536, 0);  mul_1536 = None
    unsqueeze_1222: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 2);  unsqueeze_1221 = None
    unsqueeze_1223: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 3);  unsqueeze_1222 = None
    mul_1537: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_234, 0.00048828125)
    mul_1538: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
    mul_1539: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1537, mul_1538);  mul_1537 = mul_1538 = None
    unsqueeze_1224: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1539, 0);  mul_1539 = None
    unsqueeze_1225: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, 2);  unsqueeze_1224 = None
    unsqueeze_1226: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1225, 3);  unsqueeze_1225 = None
    mul_1540: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_256, primals_317);  primals_317 = None
    unsqueeze_1227: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1540, 0);  mul_1540 = None
    unsqueeze_1228: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 2);  unsqueeze_1227 = None
    unsqueeze_1229: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 3);  unsqueeze_1228 = None
    mul_1541: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_397, unsqueeze_1226);  sub_397 = unsqueeze_1226 = None
    sub_399: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_52, mul_1541);  mul_1541 = None
    sub_400: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_399, unsqueeze_1223);  sub_399 = unsqueeze_1223 = None
    mul_1542: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_1229);  sub_400 = unsqueeze_1229 = None
    mul_1543: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_234, squeeze_256);  sum_234 = squeeze_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1542, sum_60, primals_316, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1542 = sum_60 = primals_316 = None
    getitem_478: "f32[8, 256, 16, 16]" = convolution_backward_66[0]
    getitem_479: "f32[1024, 256, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1230: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_478, 1);  getitem_478 = None
    expand_40: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1230, [8, 2, 256, 16, 16]);  unsqueeze_1230 = None
    mul_1544: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_40, view_115);  view_115 = None
    mul_1545: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_40, view_119);  expand_40 = view_119 = None
    sum_235: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1544, [3, 4], True);  mul_1544 = None
    view_266: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_235, [8, 512, 1, 1]);  sum_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_267: "f32[8, 512]" = torch.ops.aten.view.default(view_266, [8, 512]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_268: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_267, [8, 2, 1, 256]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_340: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_101);  alias_101 = None
    mul_1546: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_268, alias_340);  view_268 = None
    sum_236: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1546, [1], True)
    mul_1547: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_340, sum_236);  alias_340 = sum_236 = None
    sub_401: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1546, mul_1547);  mul_1546 = mul_1547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_51: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_401, [0, 2, 1, 3]);  sub_401 = None
    view_269: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_51, [8, 512, 1, 1]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(view_269, relu_81, primals_314, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_269 = primals_314 = None
    getitem_481: "f32[8, 128, 1, 1]" = convolution_backward_67[0]
    getitem_482: "f32[512, 128, 1, 1]" = convolution_backward_67[1]
    getitem_483: "f32[512]" = convolution_backward_67[2];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_342: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_81);  relu_81 = None
    alias_343: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_342);  alias_342 = None
    le_53: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_343, 0);  alias_343 = None
    where_53: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_53, full_default, getitem_481);  le_53 = getitem_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1231: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
    unsqueeze_1232: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 2);  unsqueeze_1231 = None
    unsqueeze_1233: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, 3);  unsqueeze_1232 = None
    sum_237: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_402: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_1233);  convolution_103 = unsqueeze_1233 = None
    mul_1548: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_53, sub_402)
    sum_238: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1548, [0, 2, 3]);  mul_1548 = None
    mul_1549: "f32[128]" = torch.ops.aten.mul.Tensor(sum_237, 0.125)
    unsqueeze_1234: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1549, 0);  mul_1549 = None
    unsqueeze_1235: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 2);  unsqueeze_1234 = None
    unsqueeze_1236: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1235, 3);  unsqueeze_1235 = None
    mul_1550: "f32[128]" = torch.ops.aten.mul.Tensor(sum_238, 0.125)
    mul_1551: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
    mul_1552: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1550, mul_1551);  mul_1550 = mul_1551 = None
    unsqueeze_1237: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1552, 0);  mul_1552 = None
    unsqueeze_1238: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1237, 2);  unsqueeze_1237 = None
    unsqueeze_1239: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 3);  unsqueeze_1238 = None
    mul_1553: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_312);  primals_312 = None
    unsqueeze_1240: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1553, 0);  mul_1553 = None
    unsqueeze_1241: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, 2);  unsqueeze_1240 = None
    unsqueeze_1242: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 3);  unsqueeze_1241 = None
    mul_1554: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_402, unsqueeze_1239);  sub_402 = unsqueeze_1239 = None
    sub_404: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_53, mul_1554);  where_53 = mul_1554 = None
    sub_405: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_404, unsqueeze_1236);  sub_404 = unsqueeze_1236 = None
    mul_1555: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_405, unsqueeze_1242);  sub_405 = unsqueeze_1242 = None
    mul_1556: "f32[128]" = torch.ops.aten.mul.Tensor(sum_238, squeeze_253);  sum_238 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1555, mean_19, primals_310, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1555 = mean_19 = primals_310 = None
    getitem_484: "f32[8, 256, 1, 1]" = convolution_backward_68[0]
    getitem_485: "f32[128, 256, 1, 1]" = convolution_backward_68[1]
    getitem_486: "f32[128]" = convolution_backward_68[2];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_41: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_484, [8, 256, 16, 16]);  getitem_484 = None
    div_47: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_41, 256);  expand_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1243: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_47, 1);  div_47 = None
    expand_42: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1243, [8, 2, 256, 16, 16]);  unsqueeze_1243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_754: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1545, expand_42);  mul_1545 = expand_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_270: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_754, [8, 512, 16, 16]);  add_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_345: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_80);  relu_80 = None
    alias_346: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_345);  alias_345 = None
    le_54: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_346, 0);  alias_346 = None
    where_54: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_54, full_default, view_270);  le_54 = view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_239: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_406: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_1246);  convolution_102 = unsqueeze_1246 = None
    mul_1557: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_54, sub_406)
    sum_240: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1557, [0, 2, 3]);  mul_1557 = None
    mul_1558: "f32[512]" = torch.ops.aten.mul.Tensor(sum_239, 0.00048828125)
    unsqueeze_1247: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1558, 0);  mul_1558 = None
    unsqueeze_1248: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1247, 2);  unsqueeze_1247 = None
    unsqueeze_1249: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, 3);  unsqueeze_1248 = None
    mul_1559: "f32[512]" = torch.ops.aten.mul.Tensor(sum_240, 0.00048828125)
    mul_1560: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
    mul_1561: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1559, mul_1560);  mul_1559 = mul_1560 = None
    unsqueeze_1250: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1561, 0);  mul_1561 = None
    unsqueeze_1251: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, 2);  unsqueeze_1250 = None
    unsqueeze_1252: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 3);  unsqueeze_1251 = None
    mul_1562: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_308);  primals_308 = None
    unsqueeze_1253: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1562, 0);  mul_1562 = None
    unsqueeze_1254: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 2);  unsqueeze_1253 = None
    unsqueeze_1255: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 3);  unsqueeze_1254 = None
    mul_1563: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_406, unsqueeze_1252);  sub_406 = unsqueeze_1252 = None
    sub_408: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_54, mul_1563);  where_54 = mul_1563 = None
    sub_409: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_408, unsqueeze_1249);  sub_408 = unsqueeze_1249 = None
    mul_1564: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_409, unsqueeze_1255);  sub_409 = unsqueeze_1255 = None
    mul_1565: "f32[512]" = torch.ops.aten.mul.Tensor(sum_240, squeeze_250);  sum_240 = squeeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1564, relu_79, primals_307, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1564 = primals_307 = None
    getitem_487: "f32[8, 256, 16, 16]" = convolution_backward_69[0]
    getitem_488: "f32[512, 128, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_348: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_79);  relu_79 = None
    alias_349: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_348);  alias_348 = None
    le_55: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_349, 0);  alias_349 = None
    where_55: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_55, full_default, getitem_487);  le_55 = getitem_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_241: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_410: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_1258);  convolution_101 = unsqueeze_1258 = None
    mul_1566: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_55, sub_410)
    sum_242: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1566, [0, 2, 3]);  mul_1566 = None
    mul_1567: "f32[256]" = torch.ops.aten.mul.Tensor(sum_241, 0.00048828125)
    unsqueeze_1259: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1567, 0);  mul_1567 = None
    unsqueeze_1260: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1259, 2);  unsqueeze_1259 = None
    unsqueeze_1261: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, 3);  unsqueeze_1260 = None
    mul_1568: "f32[256]" = torch.ops.aten.mul.Tensor(sum_242, 0.00048828125)
    mul_1569: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
    mul_1570: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1568, mul_1569);  mul_1568 = mul_1569 = None
    unsqueeze_1262: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1570, 0);  mul_1570 = None
    unsqueeze_1263: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, 2);  unsqueeze_1262 = None
    unsqueeze_1264: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 3);  unsqueeze_1263 = None
    mul_1571: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_305);  primals_305 = None
    unsqueeze_1265: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1571, 0);  mul_1571 = None
    unsqueeze_1266: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 2);  unsqueeze_1265 = None
    unsqueeze_1267: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 3);  unsqueeze_1266 = None
    mul_1572: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_410, unsqueeze_1264);  sub_410 = unsqueeze_1264 = None
    sub_412: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_55, mul_1572);  where_55 = mul_1572 = None
    sub_413: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_412, unsqueeze_1261);  sub_412 = unsqueeze_1261 = None
    mul_1573: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_413, unsqueeze_1267);  sub_413 = unsqueeze_1267 = None
    mul_1574: "f32[256]" = torch.ops.aten.mul.Tensor(sum_242, squeeze_247);  sum_242 = squeeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1573, relu_78, primals_304, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1573 = primals_304 = None
    getitem_490: "f32[8, 1024, 16, 16]" = convolution_backward_70[0]
    getitem_491: "f32[256, 1024, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_755: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_52, getitem_490);  where_52 = getitem_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_351: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_78);  relu_78 = None
    alias_352: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_351);  alias_351 = None
    le_56: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_352, 0);  alias_352 = None
    where_56: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_56, full_default, add_755);  le_56 = add_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_243: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_414: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_1270);  convolution_100 = unsqueeze_1270 = None
    mul_1575: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_56, sub_414)
    sum_244: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1575, [0, 2, 3]);  mul_1575 = None
    mul_1576: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_243, 0.00048828125)
    unsqueeze_1271: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1576, 0);  mul_1576 = None
    unsqueeze_1272: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1271, 2);  unsqueeze_1271 = None
    unsqueeze_1273: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, 3);  unsqueeze_1272 = None
    mul_1577: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_244, 0.00048828125)
    mul_1578: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
    mul_1579: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1577, mul_1578);  mul_1577 = mul_1578 = None
    unsqueeze_1274: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1579, 0);  mul_1579 = None
    unsqueeze_1275: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, 2);  unsqueeze_1274 = None
    unsqueeze_1276: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 3);  unsqueeze_1275 = None
    mul_1580: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_302);  primals_302 = None
    unsqueeze_1277: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1580, 0);  mul_1580 = None
    unsqueeze_1278: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 2);  unsqueeze_1277 = None
    unsqueeze_1279: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 3);  unsqueeze_1278 = None
    mul_1581: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_414, unsqueeze_1276);  sub_414 = unsqueeze_1276 = None
    sub_416: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_56, mul_1581);  mul_1581 = None
    sub_417: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_416, unsqueeze_1273);  sub_416 = unsqueeze_1273 = None
    mul_1582: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_417, unsqueeze_1279);  sub_417 = unsqueeze_1279 = None
    mul_1583: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_244, squeeze_244);  sum_244 = squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1582, sum_57, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1582 = sum_57 = primals_301 = None
    getitem_493: "f32[8, 256, 16, 16]" = convolution_backward_71[0]
    getitem_494: "f32[1024, 256, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1280: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_493, 1);  getitem_493 = None
    expand_43: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1280, [8, 2, 256, 16, 16]);  unsqueeze_1280 = None
    mul_1584: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_43, view_109);  view_109 = None
    mul_1585: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_43, view_113);  expand_43 = view_113 = None
    sum_245: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1584, [3, 4], True);  mul_1584 = None
    view_271: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_245, [8, 512, 1, 1]);  sum_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_272: "f32[8, 512]" = torch.ops.aten.view.default(view_271, [8, 512]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_273: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_272, [8, 2, 1, 256]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_353: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_96);  alias_96 = None
    mul_1586: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_273, alias_353);  view_273 = None
    sum_246: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1586, [1], True)
    mul_1587: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_353, sum_246);  alias_353 = sum_246 = None
    sub_418: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1586, mul_1587);  mul_1586 = mul_1587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_52: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_418, [0, 2, 1, 3]);  sub_418 = None
    view_274: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_52, [8, 512, 1, 1]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(view_274, relu_77, primals_299, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_274 = primals_299 = None
    getitem_496: "f32[8, 128, 1, 1]" = convolution_backward_72[0]
    getitem_497: "f32[512, 128, 1, 1]" = convolution_backward_72[1]
    getitem_498: "f32[512]" = convolution_backward_72[2];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_355: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_77);  relu_77 = None
    alias_356: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_355);  alias_355 = None
    le_57: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_356, 0);  alias_356 = None
    where_57: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_57, full_default, getitem_496);  le_57 = getitem_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1281: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
    unsqueeze_1282: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 2);  unsqueeze_1281 = None
    unsqueeze_1283: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 3);  unsqueeze_1282 = None
    sum_247: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_419: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_1283);  convolution_98 = unsqueeze_1283 = None
    mul_1588: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_57, sub_419)
    sum_248: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1588, [0, 2, 3]);  mul_1588 = None
    mul_1589: "f32[128]" = torch.ops.aten.mul.Tensor(sum_247, 0.125)
    unsqueeze_1284: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1589, 0);  mul_1589 = None
    unsqueeze_1285: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, 2);  unsqueeze_1284 = None
    unsqueeze_1286: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1285, 3);  unsqueeze_1285 = None
    mul_1590: "f32[128]" = torch.ops.aten.mul.Tensor(sum_248, 0.125)
    mul_1591: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
    mul_1592: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1590, mul_1591);  mul_1590 = mul_1591 = None
    unsqueeze_1287: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1592, 0);  mul_1592 = None
    unsqueeze_1288: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1287, 2);  unsqueeze_1287 = None
    unsqueeze_1289: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, 3);  unsqueeze_1288 = None
    mul_1593: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_297);  primals_297 = None
    unsqueeze_1290: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1593, 0);  mul_1593 = None
    unsqueeze_1291: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 2);  unsqueeze_1290 = None
    unsqueeze_1292: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 3);  unsqueeze_1291 = None
    mul_1594: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_419, unsqueeze_1289);  sub_419 = unsqueeze_1289 = None
    sub_421: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_57, mul_1594);  where_57 = mul_1594 = None
    sub_422: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_421, unsqueeze_1286);  sub_421 = unsqueeze_1286 = None
    mul_1595: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_422, unsqueeze_1292);  sub_422 = unsqueeze_1292 = None
    mul_1596: "f32[128]" = torch.ops.aten.mul.Tensor(sum_248, squeeze_241);  sum_248 = squeeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1595, mean_18, primals_295, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1595 = mean_18 = primals_295 = None
    getitem_499: "f32[8, 256, 1, 1]" = convolution_backward_73[0]
    getitem_500: "f32[128, 256, 1, 1]" = convolution_backward_73[1]
    getitem_501: "f32[128]" = convolution_backward_73[2];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_44: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_499, [8, 256, 16, 16]);  getitem_499 = None
    div_48: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_44, 256);  expand_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1293: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_48, 1);  div_48 = None
    expand_45: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1293, [8, 2, 256, 16, 16]);  unsqueeze_1293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_756: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1585, expand_45);  mul_1585 = expand_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_275: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_756, [8, 512, 16, 16]);  add_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_358: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_76);  relu_76 = None
    alias_359: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_358);  alias_358 = None
    le_58: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_359, 0);  alias_359 = None
    where_58: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_58, full_default, view_275);  le_58 = view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_249: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_423: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_1296);  convolution_97 = unsqueeze_1296 = None
    mul_1597: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_58, sub_423)
    sum_250: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1597, [0, 2, 3]);  mul_1597 = None
    mul_1598: "f32[512]" = torch.ops.aten.mul.Tensor(sum_249, 0.00048828125)
    unsqueeze_1297: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1598, 0);  mul_1598 = None
    unsqueeze_1298: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1297, 2);  unsqueeze_1297 = None
    unsqueeze_1299: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, 3);  unsqueeze_1298 = None
    mul_1599: "f32[512]" = torch.ops.aten.mul.Tensor(sum_250, 0.00048828125)
    mul_1600: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_1601: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1599, mul_1600);  mul_1599 = mul_1600 = None
    unsqueeze_1300: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1601, 0);  mul_1601 = None
    unsqueeze_1301: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, 2);  unsqueeze_1300 = None
    unsqueeze_1302: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1301, 3);  unsqueeze_1301 = None
    mul_1602: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_293);  primals_293 = None
    unsqueeze_1303: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1602, 0);  mul_1602 = None
    unsqueeze_1304: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 2);  unsqueeze_1303 = None
    unsqueeze_1305: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, 3);  unsqueeze_1304 = None
    mul_1603: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_423, unsqueeze_1302);  sub_423 = unsqueeze_1302 = None
    sub_425: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_58, mul_1603);  where_58 = mul_1603 = None
    sub_426: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_425, unsqueeze_1299);  sub_425 = unsqueeze_1299 = None
    mul_1604: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_426, unsqueeze_1305);  sub_426 = unsqueeze_1305 = None
    mul_1605: "f32[512]" = torch.ops.aten.mul.Tensor(sum_250, squeeze_238);  sum_250 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1604, relu_75, primals_292, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1604 = primals_292 = None
    getitem_502: "f32[8, 256, 16, 16]" = convolution_backward_74[0]
    getitem_503: "f32[512, 128, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_361: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_75);  relu_75 = None
    alias_362: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_361);  alias_361 = None
    le_59: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_362, 0);  alias_362 = None
    where_59: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_59, full_default, getitem_502);  le_59 = getitem_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_251: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_427: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_1308);  convolution_96 = unsqueeze_1308 = None
    mul_1606: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_59, sub_427)
    sum_252: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1606, [0, 2, 3]);  mul_1606 = None
    mul_1607: "f32[256]" = torch.ops.aten.mul.Tensor(sum_251, 0.00048828125)
    unsqueeze_1309: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1607, 0);  mul_1607 = None
    unsqueeze_1310: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1309, 2);  unsqueeze_1309 = None
    unsqueeze_1311: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, 3);  unsqueeze_1310 = None
    mul_1608: "f32[256]" = torch.ops.aten.mul.Tensor(sum_252, 0.00048828125)
    mul_1609: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_1610: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1608, mul_1609);  mul_1608 = mul_1609 = None
    unsqueeze_1312: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1610, 0);  mul_1610 = None
    unsqueeze_1313: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, 2);  unsqueeze_1312 = None
    unsqueeze_1314: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1313, 3);  unsqueeze_1313 = None
    mul_1611: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_290);  primals_290 = None
    unsqueeze_1315: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1611, 0);  mul_1611 = None
    unsqueeze_1316: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 2);  unsqueeze_1315 = None
    unsqueeze_1317: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, 3);  unsqueeze_1316 = None
    mul_1612: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_427, unsqueeze_1314);  sub_427 = unsqueeze_1314 = None
    sub_429: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_59, mul_1612);  where_59 = mul_1612 = None
    sub_430: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_429, unsqueeze_1311);  sub_429 = unsqueeze_1311 = None
    mul_1613: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_430, unsqueeze_1317);  sub_430 = unsqueeze_1317 = None
    mul_1614: "f32[256]" = torch.ops.aten.mul.Tensor(sum_252, squeeze_235);  sum_252 = squeeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1613, relu_74, primals_289, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1613 = primals_289 = None
    getitem_505: "f32[8, 1024, 16, 16]" = convolution_backward_75[0]
    getitem_506: "f32[256, 1024, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_757: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_56, getitem_505);  where_56 = getitem_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_364: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_74);  relu_74 = None
    alias_365: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_364);  alias_364 = None
    le_60: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_365, 0);  alias_365 = None
    where_60: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_60, full_default, add_757);  le_60 = add_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_253: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_431: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_1320);  convolution_95 = unsqueeze_1320 = None
    mul_1615: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_60, sub_431)
    sum_254: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1615, [0, 2, 3]);  mul_1615 = None
    mul_1616: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_253, 0.00048828125)
    unsqueeze_1321: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1616, 0);  mul_1616 = None
    unsqueeze_1322: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1321, 2);  unsqueeze_1321 = None
    unsqueeze_1323: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, 3);  unsqueeze_1322 = None
    mul_1617: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_254, 0.00048828125)
    mul_1618: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_1619: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1617, mul_1618);  mul_1617 = mul_1618 = None
    unsqueeze_1324: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1619, 0);  mul_1619 = None
    unsqueeze_1325: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, 2);  unsqueeze_1324 = None
    unsqueeze_1326: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1325, 3);  unsqueeze_1325 = None
    mul_1620: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_287);  primals_287 = None
    unsqueeze_1327: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1620, 0);  mul_1620 = None
    unsqueeze_1328: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 2);  unsqueeze_1327 = None
    unsqueeze_1329: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, 3);  unsqueeze_1328 = None
    mul_1621: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_431, unsqueeze_1326);  sub_431 = unsqueeze_1326 = None
    sub_433: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_60, mul_1621);  mul_1621 = None
    sub_434: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_433, unsqueeze_1323);  sub_433 = unsqueeze_1323 = None
    mul_1622: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_434, unsqueeze_1329);  sub_434 = unsqueeze_1329 = None
    mul_1623: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_254, squeeze_232);  sum_254 = squeeze_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1622, sum_54, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1622 = sum_54 = primals_286 = None
    getitem_508: "f32[8, 256, 16, 16]" = convolution_backward_76[0]
    getitem_509: "f32[1024, 256, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1330: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_508, 1);  getitem_508 = None
    expand_46: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1330, [8, 2, 256, 16, 16]);  unsqueeze_1330 = None
    mul_1624: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_46, view_103);  view_103 = None
    mul_1625: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_46, view_107);  expand_46 = view_107 = None
    sum_255: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1624, [3, 4], True);  mul_1624 = None
    view_276: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_255, [8, 512, 1, 1]);  sum_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_277: "f32[8, 512]" = torch.ops.aten.view.default(view_276, [8, 512]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_278: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_277, [8, 2, 1, 256]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_366: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_91);  alias_91 = None
    mul_1626: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_278, alias_366);  view_278 = None
    sum_256: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1626, [1], True)
    mul_1627: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_366, sum_256);  alias_366 = sum_256 = None
    sub_435: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1626, mul_1627);  mul_1626 = mul_1627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_53: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_435, [0, 2, 1, 3]);  sub_435 = None
    view_279: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_53, [8, 512, 1, 1]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(view_279, relu_73, primals_284, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_279 = primals_284 = None
    getitem_511: "f32[8, 128, 1, 1]" = convolution_backward_77[0]
    getitem_512: "f32[512, 128, 1, 1]" = convolution_backward_77[1]
    getitem_513: "f32[512]" = convolution_backward_77[2];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_368: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_73);  relu_73 = None
    alias_369: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_368);  alias_368 = None
    le_61: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_369, 0);  alias_369 = None
    where_61: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_61, full_default, getitem_511);  le_61 = getitem_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1331: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
    unsqueeze_1332: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1331, 2);  unsqueeze_1331 = None
    unsqueeze_1333: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, 3);  unsqueeze_1332 = None
    sum_257: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_436: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_1333);  convolution_93 = unsqueeze_1333 = None
    mul_1628: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_61, sub_436)
    sum_258: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1628, [0, 2, 3]);  mul_1628 = None
    mul_1629: "f32[128]" = torch.ops.aten.mul.Tensor(sum_257, 0.125)
    unsqueeze_1334: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1629, 0);  mul_1629 = None
    unsqueeze_1335: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, 2);  unsqueeze_1334 = None
    unsqueeze_1336: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1335, 3);  unsqueeze_1335 = None
    mul_1630: "f32[128]" = torch.ops.aten.mul.Tensor(sum_258, 0.125)
    mul_1631: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_1632: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1630, mul_1631);  mul_1630 = mul_1631 = None
    unsqueeze_1337: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1632, 0);  mul_1632 = None
    unsqueeze_1338: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1337, 2);  unsqueeze_1337 = None
    unsqueeze_1339: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 3);  unsqueeze_1338 = None
    mul_1633: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_282);  primals_282 = None
    unsqueeze_1340: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1633, 0);  mul_1633 = None
    unsqueeze_1341: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, 2);  unsqueeze_1340 = None
    unsqueeze_1342: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 3);  unsqueeze_1341 = None
    mul_1634: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_436, unsqueeze_1339);  sub_436 = unsqueeze_1339 = None
    sub_438: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_61, mul_1634);  where_61 = mul_1634 = None
    sub_439: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_438, unsqueeze_1336);  sub_438 = unsqueeze_1336 = None
    mul_1635: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_439, unsqueeze_1342);  sub_439 = unsqueeze_1342 = None
    mul_1636: "f32[128]" = torch.ops.aten.mul.Tensor(sum_258, squeeze_229);  sum_258 = squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1635, mean_17, primals_280, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1635 = mean_17 = primals_280 = None
    getitem_514: "f32[8, 256, 1, 1]" = convolution_backward_78[0]
    getitem_515: "f32[128, 256, 1, 1]" = convolution_backward_78[1]
    getitem_516: "f32[128]" = convolution_backward_78[2];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_47: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_514, [8, 256, 16, 16]);  getitem_514 = None
    div_49: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_47, 256);  expand_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1343: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_49, 1);  div_49 = None
    expand_48: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1343, [8, 2, 256, 16, 16]);  unsqueeze_1343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_758: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1625, expand_48);  mul_1625 = expand_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_280: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_758, [8, 512, 16, 16]);  add_758 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_371: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_72);  relu_72 = None
    alias_372: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_371);  alias_371 = None
    le_62: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_372, 0);  alias_372 = None
    where_62: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_62, full_default, view_280);  le_62 = view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_259: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_440: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_1346);  convolution_92 = unsqueeze_1346 = None
    mul_1637: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_62, sub_440)
    sum_260: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1637, [0, 2, 3]);  mul_1637 = None
    mul_1638: "f32[512]" = torch.ops.aten.mul.Tensor(sum_259, 0.00048828125)
    unsqueeze_1347: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1638, 0);  mul_1638 = None
    unsqueeze_1348: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1347, 2);  unsqueeze_1347 = None
    unsqueeze_1349: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, 3);  unsqueeze_1348 = None
    mul_1639: "f32[512]" = torch.ops.aten.mul.Tensor(sum_260, 0.00048828125)
    mul_1640: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_1641: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1639, mul_1640);  mul_1639 = mul_1640 = None
    unsqueeze_1350: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1641, 0);  mul_1641 = None
    unsqueeze_1351: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 2);  unsqueeze_1350 = None
    unsqueeze_1352: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 3);  unsqueeze_1351 = None
    mul_1642: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_278);  primals_278 = None
    unsqueeze_1353: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1642, 0);  mul_1642 = None
    unsqueeze_1354: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 2);  unsqueeze_1353 = None
    unsqueeze_1355: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 3);  unsqueeze_1354 = None
    mul_1643: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_440, unsqueeze_1352);  sub_440 = unsqueeze_1352 = None
    sub_442: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_62, mul_1643);  where_62 = mul_1643 = None
    sub_443: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_442, unsqueeze_1349);  sub_442 = unsqueeze_1349 = None
    mul_1644: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_443, unsqueeze_1355);  sub_443 = unsqueeze_1355 = None
    mul_1645: "f32[512]" = torch.ops.aten.mul.Tensor(sum_260, squeeze_226);  sum_260 = squeeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1644, relu_71, primals_277, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1644 = primals_277 = None
    getitem_517: "f32[8, 256, 16, 16]" = convolution_backward_79[0]
    getitem_518: "f32[512, 128, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_374: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_71);  relu_71 = None
    alias_375: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_374);  alias_374 = None
    le_63: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_375, 0);  alias_375 = None
    where_63: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_63, full_default, getitem_517);  le_63 = getitem_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_261: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_444: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_1358);  convolution_91 = unsqueeze_1358 = None
    mul_1646: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_63, sub_444)
    sum_262: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1646, [0, 2, 3]);  mul_1646 = None
    mul_1647: "f32[256]" = torch.ops.aten.mul.Tensor(sum_261, 0.00048828125)
    unsqueeze_1359: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1647, 0);  mul_1647 = None
    unsqueeze_1360: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1359, 2);  unsqueeze_1359 = None
    unsqueeze_1361: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, 3);  unsqueeze_1360 = None
    mul_1648: "f32[256]" = torch.ops.aten.mul.Tensor(sum_262, 0.00048828125)
    mul_1649: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_1650: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1648, mul_1649);  mul_1648 = mul_1649 = None
    unsqueeze_1362: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1650, 0);  mul_1650 = None
    unsqueeze_1363: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 2);  unsqueeze_1362 = None
    unsqueeze_1364: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1363, 3);  unsqueeze_1363 = None
    mul_1651: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_275);  primals_275 = None
    unsqueeze_1365: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1651, 0);  mul_1651 = None
    unsqueeze_1366: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 2);  unsqueeze_1365 = None
    unsqueeze_1367: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, 3);  unsqueeze_1366 = None
    mul_1652: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_444, unsqueeze_1364);  sub_444 = unsqueeze_1364 = None
    sub_446: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_63, mul_1652);  where_63 = mul_1652 = None
    sub_447: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_446, unsqueeze_1361);  sub_446 = unsqueeze_1361 = None
    mul_1653: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_447, unsqueeze_1367);  sub_447 = unsqueeze_1367 = None
    mul_1654: "f32[256]" = torch.ops.aten.mul.Tensor(sum_262, squeeze_223);  sum_262 = squeeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1653, relu_70, primals_274, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1653 = primals_274 = None
    getitem_520: "f32[8, 1024, 16, 16]" = convolution_backward_80[0]
    getitem_521: "f32[256, 1024, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_759: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_60, getitem_520);  where_60 = getitem_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_377: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_70);  relu_70 = None
    alias_378: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_377);  alias_377 = None
    le_64: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_378, 0);  alias_378 = None
    where_64: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_64, full_default, add_759);  le_64 = add_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_263: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_448: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_1370);  convolution_90 = unsqueeze_1370 = None
    mul_1655: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_64, sub_448)
    sum_264: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1655, [0, 2, 3]);  mul_1655 = None
    mul_1656: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_263, 0.00048828125)
    unsqueeze_1371: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1656, 0);  mul_1656 = None
    unsqueeze_1372: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1371, 2);  unsqueeze_1371 = None
    unsqueeze_1373: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, 3);  unsqueeze_1372 = None
    mul_1657: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_264, 0.00048828125)
    mul_1658: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_1659: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1657, mul_1658);  mul_1657 = mul_1658 = None
    unsqueeze_1374: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1659, 0);  mul_1659 = None
    unsqueeze_1375: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 2);  unsqueeze_1374 = None
    unsqueeze_1376: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1375, 3);  unsqueeze_1375 = None
    mul_1660: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_272);  primals_272 = None
    unsqueeze_1377: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1660, 0);  mul_1660 = None
    unsqueeze_1378: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 2);  unsqueeze_1377 = None
    unsqueeze_1379: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, 3);  unsqueeze_1378 = None
    mul_1661: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_448, unsqueeze_1376);  sub_448 = unsqueeze_1376 = None
    sub_450: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_64, mul_1661);  mul_1661 = None
    sub_451: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_450, unsqueeze_1373);  sub_450 = unsqueeze_1373 = None
    mul_1662: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_451, unsqueeze_1379);  sub_451 = unsqueeze_1379 = None
    mul_1663: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_264, squeeze_220);  sum_264 = squeeze_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1662, sum_51, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1662 = sum_51 = primals_271 = None
    getitem_523: "f32[8, 256, 16, 16]" = convolution_backward_81[0]
    getitem_524: "f32[1024, 256, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1380: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_523, 1);  getitem_523 = None
    expand_49: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1380, [8, 2, 256, 16, 16]);  unsqueeze_1380 = None
    mul_1664: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_49, view_97);  view_97 = None
    mul_1665: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_49, view_101);  expand_49 = view_101 = None
    sum_265: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1664, [3, 4], True);  mul_1664 = None
    view_281: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_265, [8, 512, 1, 1]);  sum_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_282: "f32[8, 512]" = torch.ops.aten.view.default(view_281, [8, 512]);  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_283: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_282, [8, 2, 1, 256]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_379: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_86);  alias_86 = None
    mul_1666: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_283, alias_379);  view_283 = None
    sum_266: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1666, [1], True)
    mul_1667: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_379, sum_266);  alias_379 = sum_266 = None
    sub_452: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1666, mul_1667);  mul_1666 = mul_1667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_54: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_452, [0, 2, 1, 3]);  sub_452 = None
    view_284: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_54, [8, 512, 1, 1]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(view_284, relu_69, primals_269, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_284 = primals_269 = None
    getitem_526: "f32[8, 128, 1, 1]" = convolution_backward_82[0]
    getitem_527: "f32[512, 128, 1, 1]" = convolution_backward_82[1]
    getitem_528: "f32[512]" = convolution_backward_82[2];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_381: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_69);  relu_69 = None
    alias_382: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_381);  alias_381 = None
    le_65: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_382, 0);  alias_382 = None
    where_65: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_65, full_default, getitem_526);  le_65 = getitem_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1381: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
    unsqueeze_1382: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1381, 2);  unsqueeze_1381 = None
    unsqueeze_1383: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, 3);  unsqueeze_1382 = None
    sum_267: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_453: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_1383);  convolution_88 = unsqueeze_1383 = None
    mul_1668: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_65, sub_453)
    sum_268: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1668, [0, 2, 3]);  mul_1668 = None
    mul_1669: "f32[128]" = torch.ops.aten.mul.Tensor(sum_267, 0.125)
    unsqueeze_1384: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1669, 0);  mul_1669 = None
    unsqueeze_1385: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, 2);  unsqueeze_1384 = None
    unsqueeze_1386: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1385, 3);  unsqueeze_1385 = None
    mul_1670: "f32[128]" = torch.ops.aten.mul.Tensor(sum_268, 0.125)
    mul_1671: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_1672: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1670, mul_1671);  mul_1670 = mul_1671 = None
    unsqueeze_1387: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1672, 0);  mul_1672 = None
    unsqueeze_1388: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1387, 2);  unsqueeze_1387 = None
    unsqueeze_1389: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, 3);  unsqueeze_1388 = None
    mul_1673: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_267);  primals_267 = None
    unsqueeze_1390: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1673, 0);  mul_1673 = None
    unsqueeze_1391: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, 2);  unsqueeze_1390 = None
    unsqueeze_1392: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1391, 3);  unsqueeze_1391 = None
    mul_1674: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_453, unsqueeze_1389);  sub_453 = unsqueeze_1389 = None
    sub_455: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_65, mul_1674);  where_65 = mul_1674 = None
    sub_456: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_455, unsqueeze_1386);  sub_455 = unsqueeze_1386 = None
    mul_1675: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_456, unsqueeze_1392);  sub_456 = unsqueeze_1392 = None
    mul_1676: "f32[128]" = torch.ops.aten.mul.Tensor(sum_268, squeeze_217);  sum_268 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1675, mean_16, primals_265, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1675 = mean_16 = primals_265 = None
    getitem_529: "f32[8, 256, 1, 1]" = convolution_backward_83[0]
    getitem_530: "f32[128, 256, 1, 1]" = convolution_backward_83[1]
    getitem_531: "f32[128]" = convolution_backward_83[2];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_50: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_529, [8, 256, 16, 16]);  getitem_529 = None
    div_50: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_50, 256);  expand_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1393: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_50, 1);  div_50 = None
    expand_51: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1393, [8, 2, 256, 16, 16]);  unsqueeze_1393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_760: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1665, expand_51);  mul_1665 = expand_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_285: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_760, [8, 512, 16, 16]);  add_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_384: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_68);  relu_68 = None
    alias_385: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_384);  alias_384 = None
    le_66: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_385, 0);  alias_385 = None
    where_66: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_66, full_default, view_285);  le_66 = view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_269: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_457: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_1396);  convolution_87 = unsqueeze_1396 = None
    mul_1677: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_66, sub_457)
    sum_270: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1677, [0, 2, 3]);  mul_1677 = None
    mul_1678: "f32[512]" = torch.ops.aten.mul.Tensor(sum_269, 0.00048828125)
    unsqueeze_1397: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1678, 0);  mul_1678 = None
    unsqueeze_1398: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1397, 2);  unsqueeze_1397 = None
    unsqueeze_1399: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, 3);  unsqueeze_1398 = None
    mul_1679: "f32[512]" = torch.ops.aten.mul.Tensor(sum_270, 0.00048828125)
    mul_1680: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_1681: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1679, mul_1680);  mul_1679 = mul_1680 = None
    unsqueeze_1400: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1681, 0);  mul_1681 = None
    unsqueeze_1401: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, 2);  unsqueeze_1400 = None
    unsqueeze_1402: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1401, 3);  unsqueeze_1401 = None
    mul_1682: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_263);  primals_263 = None
    unsqueeze_1403: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1682, 0);  mul_1682 = None
    unsqueeze_1404: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1403, 2);  unsqueeze_1403 = None
    unsqueeze_1405: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1404, 3);  unsqueeze_1404 = None
    mul_1683: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_457, unsqueeze_1402);  sub_457 = unsqueeze_1402 = None
    sub_459: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_66, mul_1683);  where_66 = mul_1683 = None
    sub_460: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_459, unsqueeze_1399);  sub_459 = unsqueeze_1399 = None
    mul_1684: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_460, unsqueeze_1405);  sub_460 = unsqueeze_1405 = None
    mul_1685: "f32[512]" = torch.ops.aten.mul.Tensor(sum_270, squeeze_214);  sum_270 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1684, relu_67, primals_262, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1684 = primals_262 = None
    getitem_532: "f32[8, 256, 16, 16]" = convolution_backward_84[0]
    getitem_533: "f32[512, 128, 3, 3]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_387: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_67);  relu_67 = None
    alias_388: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_387);  alias_387 = None
    le_67: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_388, 0);  alias_388 = None
    where_67: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_67, full_default, getitem_532);  le_67 = getitem_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_271: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_461: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_1408);  convolution_86 = unsqueeze_1408 = None
    mul_1686: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_67, sub_461)
    sum_272: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1686, [0, 2, 3]);  mul_1686 = None
    mul_1687: "f32[256]" = torch.ops.aten.mul.Tensor(sum_271, 0.00048828125)
    unsqueeze_1409: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1687, 0);  mul_1687 = None
    unsqueeze_1410: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1409, 2);  unsqueeze_1409 = None
    unsqueeze_1411: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, 3);  unsqueeze_1410 = None
    mul_1688: "f32[256]" = torch.ops.aten.mul.Tensor(sum_272, 0.00048828125)
    mul_1689: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_1690: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1688, mul_1689);  mul_1688 = mul_1689 = None
    unsqueeze_1412: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1690, 0);  mul_1690 = None
    unsqueeze_1413: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, 2);  unsqueeze_1412 = None
    unsqueeze_1414: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1413, 3);  unsqueeze_1413 = None
    mul_1691: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_260);  primals_260 = None
    unsqueeze_1415: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1691, 0);  mul_1691 = None
    unsqueeze_1416: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1415, 2);  unsqueeze_1415 = None
    unsqueeze_1417: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1416, 3);  unsqueeze_1416 = None
    mul_1692: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_461, unsqueeze_1414);  sub_461 = unsqueeze_1414 = None
    sub_463: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_67, mul_1692);  where_67 = mul_1692 = None
    sub_464: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_463, unsqueeze_1411);  sub_463 = unsqueeze_1411 = None
    mul_1693: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_464, unsqueeze_1417);  sub_464 = unsqueeze_1417 = None
    mul_1694: "f32[256]" = torch.ops.aten.mul.Tensor(sum_272, squeeze_211);  sum_272 = squeeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1693, relu_66, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1693 = primals_259 = None
    getitem_535: "f32[8, 1024, 16, 16]" = convolution_backward_85[0]
    getitem_536: "f32[256, 1024, 1, 1]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_761: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_64, getitem_535);  where_64 = getitem_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_390: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_66);  relu_66 = None
    alias_391: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_390);  alias_390 = None
    le_68: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_391, 0);  alias_391 = None
    where_68: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_68, full_default, add_761);  le_68 = add_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_273: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_465: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_1420);  convolution_85 = unsqueeze_1420 = None
    mul_1695: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_68, sub_465)
    sum_274: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1695, [0, 2, 3]);  mul_1695 = None
    mul_1696: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_273, 0.00048828125)
    unsqueeze_1421: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1696, 0);  mul_1696 = None
    unsqueeze_1422: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1421, 2);  unsqueeze_1421 = None
    unsqueeze_1423: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, 3);  unsqueeze_1422 = None
    mul_1697: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_274, 0.00048828125)
    mul_1698: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_1699: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1697, mul_1698);  mul_1697 = mul_1698 = None
    unsqueeze_1424: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1699, 0);  mul_1699 = None
    unsqueeze_1425: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, 2);  unsqueeze_1424 = None
    unsqueeze_1426: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1425, 3);  unsqueeze_1425 = None
    mul_1700: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_257);  primals_257 = None
    unsqueeze_1427: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1700, 0);  mul_1700 = None
    unsqueeze_1428: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1427, 2);  unsqueeze_1427 = None
    unsqueeze_1429: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1428, 3);  unsqueeze_1428 = None
    mul_1701: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_465, unsqueeze_1426);  sub_465 = unsqueeze_1426 = None
    sub_467: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_68, mul_1701);  mul_1701 = None
    sub_468: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_467, unsqueeze_1423);  sub_467 = unsqueeze_1423 = None
    mul_1702: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_468, unsqueeze_1429);  sub_468 = unsqueeze_1429 = None
    mul_1703: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_274, squeeze_208);  sum_274 = squeeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1702, sum_48, primals_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1702 = sum_48 = primals_256 = None
    getitem_538: "f32[8, 256, 16, 16]" = convolution_backward_86[0]
    getitem_539: "f32[1024, 256, 1, 1]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1430: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_538, 1);  getitem_538 = None
    expand_52: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1430, [8, 2, 256, 16, 16]);  unsqueeze_1430 = None
    mul_1704: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_52, view_91);  view_91 = None
    mul_1705: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_52, view_95);  expand_52 = view_95 = None
    sum_275: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1704, [3, 4], True);  mul_1704 = None
    view_286: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_275, [8, 512, 1, 1]);  sum_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_287: "f32[8, 512]" = torch.ops.aten.view.default(view_286, [8, 512]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_288: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_287, [8, 2, 1, 256]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_392: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_81);  alias_81 = None
    mul_1706: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_288, alias_392);  view_288 = None
    sum_276: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1706, [1], True)
    mul_1707: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_392, sum_276);  alias_392 = sum_276 = None
    sub_469: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1706, mul_1707);  mul_1706 = mul_1707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_55: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_469, [0, 2, 1, 3]);  sub_469 = None
    view_289: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_55, [8, 512, 1, 1]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(view_289, relu_65, primals_254, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_289 = primals_254 = None
    getitem_541: "f32[8, 128, 1, 1]" = convolution_backward_87[0]
    getitem_542: "f32[512, 128, 1, 1]" = convolution_backward_87[1]
    getitem_543: "f32[512]" = convolution_backward_87[2];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_394: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_65);  relu_65 = None
    alias_395: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_394);  alias_394 = None
    le_69: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_395, 0);  alias_395 = None
    where_69: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_69, full_default, getitem_541);  le_69 = getitem_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1431: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
    unsqueeze_1432: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1431, 2);  unsqueeze_1431 = None
    unsqueeze_1433: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, 3);  unsqueeze_1432 = None
    sum_277: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_470: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_1433);  convolution_83 = unsqueeze_1433 = None
    mul_1708: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_69, sub_470)
    sum_278: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1708, [0, 2, 3]);  mul_1708 = None
    mul_1709: "f32[128]" = torch.ops.aten.mul.Tensor(sum_277, 0.125)
    unsqueeze_1434: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1709, 0);  mul_1709 = None
    unsqueeze_1435: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, 2);  unsqueeze_1434 = None
    unsqueeze_1436: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1435, 3);  unsqueeze_1435 = None
    mul_1710: "f32[128]" = torch.ops.aten.mul.Tensor(sum_278, 0.125)
    mul_1711: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_1712: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1710, mul_1711);  mul_1710 = mul_1711 = None
    unsqueeze_1437: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1712, 0);  mul_1712 = None
    unsqueeze_1438: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1437, 2);  unsqueeze_1437 = None
    unsqueeze_1439: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, 3);  unsqueeze_1438 = None
    mul_1713: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_252);  primals_252 = None
    unsqueeze_1440: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1713, 0);  mul_1713 = None
    unsqueeze_1441: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1440, 2);  unsqueeze_1440 = None
    unsqueeze_1442: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1441, 3);  unsqueeze_1441 = None
    mul_1714: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_470, unsqueeze_1439);  sub_470 = unsqueeze_1439 = None
    sub_472: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_69, mul_1714);  where_69 = mul_1714 = None
    sub_473: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_472, unsqueeze_1436);  sub_472 = unsqueeze_1436 = None
    mul_1715: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_473, unsqueeze_1442);  sub_473 = unsqueeze_1442 = None
    mul_1716: "f32[128]" = torch.ops.aten.mul.Tensor(sum_278, squeeze_205);  sum_278 = squeeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1715, mean_15, primals_250, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1715 = mean_15 = primals_250 = None
    getitem_544: "f32[8, 256, 1, 1]" = convolution_backward_88[0]
    getitem_545: "f32[128, 256, 1, 1]" = convolution_backward_88[1]
    getitem_546: "f32[128]" = convolution_backward_88[2];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_53: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_544, [8, 256, 16, 16]);  getitem_544 = None
    div_51: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_53, 256);  expand_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1443: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_51, 1);  div_51 = None
    expand_54: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1443, [8, 2, 256, 16, 16]);  unsqueeze_1443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_762: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1705, expand_54);  mul_1705 = expand_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_290: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_762, [8, 512, 16, 16]);  add_762 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_397: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_64);  relu_64 = None
    alias_398: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_397);  alias_397 = None
    le_70: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_398, 0);  alias_398 = None
    where_70: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_70, full_default, view_290);  le_70 = view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_279: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_474: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_1446);  convolution_82 = unsqueeze_1446 = None
    mul_1717: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_70, sub_474)
    sum_280: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1717, [0, 2, 3]);  mul_1717 = None
    mul_1718: "f32[512]" = torch.ops.aten.mul.Tensor(sum_279, 0.00048828125)
    unsqueeze_1447: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1718, 0);  mul_1718 = None
    unsqueeze_1448: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1447, 2);  unsqueeze_1447 = None
    unsqueeze_1449: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, 3);  unsqueeze_1448 = None
    mul_1719: "f32[512]" = torch.ops.aten.mul.Tensor(sum_280, 0.00048828125)
    mul_1720: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_1721: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1719, mul_1720);  mul_1719 = mul_1720 = None
    unsqueeze_1450: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1721, 0);  mul_1721 = None
    unsqueeze_1451: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, 2);  unsqueeze_1450 = None
    unsqueeze_1452: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1451, 3);  unsqueeze_1451 = None
    mul_1722: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_248);  primals_248 = None
    unsqueeze_1453: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1722, 0);  mul_1722 = None
    unsqueeze_1454: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1453, 2);  unsqueeze_1453 = None
    unsqueeze_1455: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, 3);  unsqueeze_1454 = None
    mul_1723: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_474, unsqueeze_1452);  sub_474 = unsqueeze_1452 = None
    sub_476: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_70, mul_1723);  where_70 = mul_1723 = None
    sub_477: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_476, unsqueeze_1449);  sub_476 = unsqueeze_1449 = None
    mul_1724: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_477, unsqueeze_1455);  sub_477 = unsqueeze_1455 = None
    mul_1725: "f32[512]" = torch.ops.aten.mul.Tensor(sum_280, squeeze_202);  sum_280 = squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1724, relu_63, primals_247, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1724 = primals_247 = None
    getitem_547: "f32[8, 256, 16, 16]" = convolution_backward_89[0]
    getitem_548: "f32[512, 128, 3, 3]" = convolution_backward_89[1];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_400: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_63);  relu_63 = None
    alias_401: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_400);  alias_400 = None
    le_71: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_401, 0);  alias_401 = None
    where_71: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_71, full_default, getitem_547);  le_71 = getitem_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_281: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_478: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_1458);  convolution_81 = unsqueeze_1458 = None
    mul_1726: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_71, sub_478)
    sum_282: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1726, [0, 2, 3]);  mul_1726 = None
    mul_1727: "f32[256]" = torch.ops.aten.mul.Tensor(sum_281, 0.00048828125)
    unsqueeze_1459: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1727, 0);  mul_1727 = None
    unsqueeze_1460: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1459, 2);  unsqueeze_1459 = None
    unsqueeze_1461: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, 3);  unsqueeze_1460 = None
    mul_1728: "f32[256]" = torch.ops.aten.mul.Tensor(sum_282, 0.00048828125)
    mul_1729: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_1730: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1728, mul_1729);  mul_1728 = mul_1729 = None
    unsqueeze_1462: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1730, 0);  mul_1730 = None
    unsqueeze_1463: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, 2);  unsqueeze_1462 = None
    unsqueeze_1464: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1463, 3);  unsqueeze_1463 = None
    mul_1731: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_245);  primals_245 = None
    unsqueeze_1465: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1731, 0);  mul_1731 = None
    unsqueeze_1466: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1465, 2);  unsqueeze_1465 = None
    unsqueeze_1467: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, 3);  unsqueeze_1466 = None
    mul_1732: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_478, unsqueeze_1464);  sub_478 = unsqueeze_1464 = None
    sub_480: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_71, mul_1732);  where_71 = mul_1732 = None
    sub_481: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_480, unsqueeze_1461);  sub_480 = unsqueeze_1461 = None
    mul_1733: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_481, unsqueeze_1467);  sub_481 = unsqueeze_1467 = None
    mul_1734: "f32[256]" = torch.ops.aten.mul.Tensor(sum_282, squeeze_199);  sum_282 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1733, relu_62, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1733 = primals_244 = None
    getitem_550: "f32[8, 1024, 16, 16]" = convolution_backward_90[0]
    getitem_551: "f32[256, 1024, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_763: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_68, getitem_550);  where_68 = getitem_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_403: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_62);  relu_62 = None
    alias_404: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_403);  alias_403 = None
    le_72: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_404, 0);  alias_404 = None
    where_72: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_72, full_default, add_763);  le_72 = add_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_283: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_482: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_1470);  convolution_80 = unsqueeze_1470 = None
    mul_1735: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_72, sub_482)
    sum_284: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1735, [0, 2, 3]);  mul_1735 = None
    mul_1736: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_283, 0.00048828125)
    unsqueeze_1471: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1736, 0);  mul_1736 = None
    unsqueeze_1472: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1471, 2);  unsqueeze_1471 = None
    unsqueeze_1473: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, 3);  unsqueeze_1472 = None
    mul_1737: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_284, 0.00048828125)
    mul_1738: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_1739: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1737, mul_1738);  mul_1737 = mul_1738 = None
    unsqueeze_1474: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1739, 0);  mul_1739 = None
    unsqueeze_1475: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, 2);  unsqueeze_1474 = None
    unsqueeze_1476: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1475, 3);  unsqueeze_1475 = None
    mul_1740: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_242);  primals_242 = None
    unsqueeze_1477: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1740, 0);  mul_1740 = None
    unsqueeze_1478: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1477, 2);  unsqueeze_1477 = None
    unsqueeze_1479: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, 3);  unsqueeze_1478 = None
    mul_1741: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_482, unsqueeze_1476);  sub_482 = unsqueeze_1476 = None
    sub_484: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_72, mul_1741);  mul_1741 = None
    sub_485: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_484, unsqueeze_1473);  sub_484 = unsqueeze_1473 = None
    mul_1742: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_485, unsqueeze_1479);  sub_485 = unsqueeze_1479 = None
    mul_1743: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_284, squeeze_196);  sum_284 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1742, sum_45, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1742 = sum_45 = primals_241 = None
    getitem_553: "f32[8, 256, 16, 16]" = convolution_backward_91[0]
    getitem_554: "f32[1024, 256, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1480: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_553, 1);  getitem_553 = None
    expand_55: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1480, [8, 2, 256, 16, 16]);  unsqueeze_1480 = None
    mul_1744: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_55, view_85);  view_85 = None
    mul_1745: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_55, view_89);  expand_55 = view_89 = None
    sum_285: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1744, [3, 4], True);  mul_1744 = None
    view_291: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_285, [8, 512, 1, 1]);  sum_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_292: "f32[8, 512]" = torch.ops.aten.view.default(view_291, [8, 512]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_293: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_292, [8, 2, 1, 256]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_405: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_76);  alias_76 = None
    mul_1746: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_293, alias_405);  view_293 = None
    sum_286: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1746, [1], True)
    mul_1747: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_405, sum_286);  alias_405 = sum_286 = None
    sub_486: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1746, mul_1747);  mul_1746 = mul_1747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_56: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_486, [0, 2, 1, 3]);  sub_486 = None
    view_294: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_56, [8, 512, 1, 1]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(view_294, relu_61, primals_239, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_294 = primals_239 = None
    getitem_556: "f32[8, 128, 1, 1]" = convolution_backward_92[0]
    getitem_557: "f32[512, 128, 1, 1]" = convolution_backward_92[1]
    getitem_558: "f32[512]" = convolution_backward_92[2];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_407: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_61);  relu_61 = None
    alias_408: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_407);  alias_407 = None
    le_73: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_408, 0);  alias_408 = None
    where_73: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_73, full_default, getitem_556);  le_73 = getitem_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1481: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_1482: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1481, 2);  unsqueeze_1481 = None
    unsqueeze_1483: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, 3);  unsqueeze_1482 = None
    sum_287: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_487: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_1483);  convolution_78 = unsqueeze_1483 = None
    mul_1748: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_73, sub_487)
    sum_288: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1748, [0, 2, 3]);  mul_1748 = None
    mul_1749: "f32[128]" = torch.ops.aten.mul.Tensor(sum_287, 0.125)
    unsqueeze_1484: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1749, 0);  mul_1749 = None
    unsqueeze_1485: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, 2);  unsqueeze_1484 = None
    unsqueeze_1486: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1485, 3);  unsqueeze_1485 = None
    mul_1750: "f32[128]" = torch.ops.aten.mul.Tensor(sum_288, 0.125)
    mul_1751: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_1752: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1750, mul_1751);  mul_1750 = mul_1751 = None
    unsqueeze_1487: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1752, 0);  mul_1752 = None
    unsqueeze_1488: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1487, 2);  unsqueeze_1487 = None
    unsqueeze_1489: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1488, 3);  unsqueeze_1488 = None
    mul_1753: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_237);  primals_237 = None
    unsqueeze_1490: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1753, 0);  mul_1753 = None
    unsqueeze_1491: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, 2);  unsqueeze_1490 = None
    unsqueeze_1492: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1491, 3);  unsqueeze_1491 = None
    mul_1754: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_487, unsqueeze_1489);  sub_487 = unsqueeze_1489 = None
    sub_489: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_73, mul_1754);  where_73 = mul_1754 = None
    sub_490: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_489, unsqueeze_1486);  sub_489 = unsqueeze_1486 = None
    mul_1755: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_490, unsqueeze_1492);  sub_490 = unsqueeze_1492 = None
    mul_1756: "f32[128]" = torch.ops.aten.mul.Tensor(sum_288, squeeze_193);  sum_288 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1755, mean_14, primals_235, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1755 = mean_14 = primals_235 = None
    getitem_559: "f32[8, 256, 1, 1]" = convolution_backward_93[0]
    getitem_560: "f32[128, 256, 1, 1]" = convolution_backward_93[1]
    getitem_561: "f32[128]" = convolution_backward_93[2];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_56: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_559, [8, 256, 16, 16]);  getitem_559 = None
    div_52: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_56, 256);  expand_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1493: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_52, 1);  div_52 = None
    expand_57: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1493, [8, 2, 256, 16, 16]);  unsqueeze_1493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_764: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1745, expand_57);  mul_1745 = expand_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_295: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_764, [8, 512, 16, 16]);  add_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_410: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_60);  relu_60 = None
    alias_411: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_410);  alias_410 = None
    le_74: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_411, 0);  alias_411 = None
    where_74: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_74, full_default, view_295);  le_74 = view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_289: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_491: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_1496);  convolution_77 = unsqueeze_1496 = None
    mul_1757: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_74, sub_491)
    sum_290: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1757, [0, 2, 3]);  mul_1757 = None
    mul_1758: "f32[512]" = torch.ops.aten.mul.Tensor(sum_289, 0.00048828125)
    unsqueeze_1497: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1758, 0);  mul_1758 = None
    unsqueeze_1498: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1497, 2);  unsqueeze_1497 = None
    unsqueeze_1499: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, 3);  unsqueeze_1498 = None
    mul_1759: "f32[512]" = torch.ops.aten.mul.Tensor(sum_290, 0.00048828125)
    mul_1760: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_1761: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1759, mul_1760);  mul_1759 = mul_1760 = None
    unsqueeze_1500: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1761, 0);  mul_1761 = None
    unsqueeze_1501: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1500, 2);  unsqueeze_1500 = None
    unsqueeze_1502: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1501, 3);  unsqueeze_1501 = None
    mul_1762: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_233);  primals_233 = None
    unsqueeze_1503: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1762, 0);  mul_1762 = None
    unsqueeze_1504: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1503, 2);  unsqueeze_1503 = None
    unsqueeze_1505: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1504, 3);  unsqueeze_1504 = None
    mul_1763: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_491, unsqueeze_1502);  sub_491 = unsqueeze_1502 = None
    sub_493: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_74, mul_1763);  where_74 = mul_1763 = None
    sub_494: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_493, unsqueeze_1499);  sub_493 = unsqueeze_1499 = None
    mul_1764: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_494, unsqueeze_1505);  sub_494 = unsqueeze_1505 = None
    mul_1765: "f32[512]" = torch.ops.aten.mul.Tensor(sum_290, squeeze_190);  sum_290 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1764, relu_59, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1764 = primals_232 = None
    getitem_562: "f32[8, 256, 16, 16]" = convolution_backward_94[0]
    getitem_563: "f32[512, 128, 3, 3]" = convolution_backward_94[1];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_413: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_59);  relu_59 = None
    alias_414: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_413);  alias_413 = None
    le_75: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_414, 0);  alias_414 = None
    where_75: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_75, full_default, getitem_562);  le_75 = getitem_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_291: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_495: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_1508);  convolution_76 = unsqueeze_1508 = None
    mul_1766: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_75, sub_495)
    sum_292: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1766, [0, 2, 3]);  mul_1766 = None
    mul_1767: "f32[256]" = torch.ops.aten.mul.Tensor(sum_291, 0.00048828125)
    unsqueeze_1509: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1767, 0);  mul_1767 = None
    unsqueeze_1510: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1509, 2);  unsqueeze_1509 = None
    unsqueeze_1511: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1510, 3);  unsqueeze_1510 = None
    mul_1768: "f32[256]" = torch.ops.aten.mul.Tensor(sum_292, 0.00048828125)
    mul_1769: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_1770: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1768, mul_1769);  mul_1768 = mul_1769 = None
    unsqueeze_1512: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1770, 0);  mul_1770 = None
    unsqueeze_1513: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1512, 2);  unsqueeze_1512 = None
    unsqueeze_1514: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1513, 3);  unsqueeze_1513 = None
    mul_1771: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_230);  primals_230 = None
    unsqueeze_1515: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1771, 0);  mul_1771 = None
    unsqueeze_1516: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1515, 2);  unsqueeze_1515 = None
    unsqueeze_1517: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1516, 3);  unsqueeze_1516 = None
    mul_1772: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_495, unsqueeze_1514);  sub_495 = unsqueeze_1514 = None
    sub_497: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_75, mul_1772);  where_75 = mul_1772 = None
    sub_498: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_497, unsqueeze_1511);  sub_497 = unsqueeze_1511 = None
    mul_1773: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_498, unsqueeze_1517);  sub_498 = unsqueeze_1517 = None
    mul_1774: "f32[256]" = torch.ops.aten.mul.Tensor(sum_292, squeeze_187);  sum_292 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1773, relu_58, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1773 = primals_229 = None
    getitem_565: "f32[8, 1024, 16, 16]" = convolution_backward_95[0]
    getitem_566: "f32[256, 1024, 1, 1]" = convolution_backward_95[1];  convolution_backward_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_765: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_72, getitem_565);  where_72 = getitem_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_416: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_58);  relu_58 = None
    alias_417: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_416);  alias_416 = None
    le_76: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_417, 0);  alias_417 = None
    where_76: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_76, full_default, add_765);  le_76 = add_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_293: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_499: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_1520);  convolution_75 = unsqueeze_1520 = None
    mul_1775: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_76, sub_499)
    sum_294: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1775, [0, 2, 3]);  mul_1775 = None
    mul_1776: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_293, 0.00048828125)
    unsqueeze_1521: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1776, 0);  mul_1776 = None
    unsqueeze_1522: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1521, 2);  unsqueeze_1521 = None
    unsqueeze_1523: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1522, 3);  unsqueeze_1522 = None
    mul_1777: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_294, 0.00048828125)
    mul_1778: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_1779: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1777, mul_1778);  mul_1777 = mul_1778 = None
    unsqueeze_1524: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1779, 0);  mul_1779 = None
    unsqueeze_1525: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1524, 2);  unsqueeze_1524 = None
    unsqueeze_1526: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1525, 3);  unsqueeze_1525 = None
    mul_1780: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_227);  primals_227 = None
    unsqueeze_1527: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1780, 0);  mul_1780 = None
    unsqueeze_1528: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1527, 2);  unsqueeze_1527 = None
    unsqueeze_1529: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1528, 3);  unsqueeze_1528 = None
    mul_1781: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_499, unsqueeze_1526);  sub_499 = unsqueeze_1526 = None
    sub_501: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_76, mul_1781);  mul_1781 = None
    sub_502: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_501, unsqueeze_1523);  sub_501 = unsqueeze_1523 = None
    mul_1782: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_502, unsqueeze_1529);  sub_502 = unsqueeze_1529 = None
    mul_1783: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_294, squeeze_184);  sum_294 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(mul_1782, sum_42, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1782 = sum_42 = primals_226 = None
    getitem_568: "f32[8, 256, 16, 16]" = convolution_backward_96[0]
    getitem_569: "f32[1024, 256, 1, 1]" = convolution_backward_96[1];  convolution_backward_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1530: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_568, 1);  getitem_568 = None
    expand_58: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1530, [8, 2, 256, 16, 16]);  unsqueeze_1530 = None
    mul_1784: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_58, view_79);  view_79 = None
    mul_1785: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_58, view_83);  expand_58 = view_83 = None
    sum_295: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1784, [3, 4], True);  mul_1784 = None
    view_296: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_295, [8, 512, 1, 1]);  sum_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_297: "f32[8, 512]" = torch.ops.aten.view.default(view_296, [8, 512]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_298: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_297, [8, 2, 1, 256]);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_418: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_71);  alias_71 = None
    mul_1786: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_298, alias_418);  view_298 = None
    sum_296: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1786, [1], True)
    mul_1787: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_418, sum_296);  alias_418 = sum_296 = None
    sub_503: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1786, mul_1787);  mul_1786 = mul_1787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_57: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_503, [0, 2, 1, 3]);  sub_503 = None
    view_299: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_57, [8, 512, 1, 1]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(view_299, relu_57, primals_224, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_299 = primals_224 = None
    getitem_571: "f32[8, 128, 1, 1]" = convolution_backward_97[0]
    getitem_572: "f32[512, 128, 1, 1]" = convolution_backward_97[1]
    getitem_573: "f32[512]" = convolution_backward_97[2];  convolution_backward_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_420: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_57);  relu_57 = None
    alias_421: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_420);  alias_420 = None
    le_77: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_421, 0);  alias_421 = None
    where_77: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_77, full_default, getitem_571);  le_77 = getitem_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1531: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_1532: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1531, 2);  unsqueeze_1531 = None
    unsqueeze_1533: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1532, 3);  unsqueeze_1532 = None
    sum_297: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_504: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_1533);  convolution_73 = unsqueeze_1533 = None
    mul_1788: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_77, sub_504)
    sum_298: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1788, [0, 2, 3]);  mul_1788 = None
    mul_1789: "f32[128]" = torch.ops.aten.mul.Tensor(sum_297, 0.125)
    unsqueeze_1534: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1789, 0);  mul_1789 = None
    unsqueeze_1535: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1534, 2);  unsqueeze_1534 = None
    unsqueeze_1536: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1535, 3);  unsqueeze_1535 = None
    mul_1790: "f32[128]" = torch.ops.aten.mul.Tensor(sum_298, 0.125)
    mul_1791: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_1792: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1790, mul_1791);  mul_1790 = mul_1791 = None
    unsqueeze_1537: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1792, 0);  mul_1792 = None
    unsqueeze_1538: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1537, 2);  unsqueeze_1537 = None
    unsqueeze_1539: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1538, 3);  unsqueeze_1538 = None
    mul_1793: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_222);  primals_222 = None
    unsqueeze_1540: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1793, 0);  mul_1793 = None
    unsqueeze_1541: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1540, 2);  unsqueeze_1540 = None
    unsqueeze_1542: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1541, 3);  unsqueeze_1541 = None
    mul_1794: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_504, unsqueeze_1539);  sub_504 = unsqueeze_1539 = None
    sub_506: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_77, mul_1794);  where_77 = mul_1794 = None
    sub_507: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_506, unsqueeze_1536);  sub_506 = unsqueeze_1536 = None
    mul_1795: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_507, unsqueeze_1542);  sub_507 = unsqueeze_1542 = None
    mul_1796: "f32[128]" = torch.ops.aten.mul.Tensor(sum_298, squeeze_181);  sum_298 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(mul_1795, mean_13, primals_220, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1795 = mean_13 = primals_220 = None
    getitem_574: "f32[8, 256, 1, 1]" = convolution_backward_98[0]
    getitem_575: "f32[128, 256, 1, 1]" = convolution_backward_98[1]
    getitem_576: "f32[128]" = convolution_backward_98[2];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_59: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_574, [8, 256, 16, 16]);  getitem_574 = None
    div_53: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_59, 256);  expand_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1543: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_53, 1);  div_53 = None
    expand_60: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1543, [8, 2, 256, 16, 16]);  unsqueeze_1543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_766: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1785, expand_60);  mul_1785 = expand_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_300: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_766, [8, 512, 16, 16]);  add_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_423: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_56);  relu_56 = None
    alias_424: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_423);  alias_423 = None
    le_78: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_424, 0);  alias_424 = None
    where_78: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_78, full_default, view_300);  le_78 = view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_299: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_508: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_1546);  convolution_72 = unsqueeze_1546 = None
    mul_1797: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_78, sub_508)
    sum_300: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1797, [0, 2, 3]);  mul_1797 = None
    mul_1798: "f32[512]" = torch.ops.aten.mul.Tensor(sum_299, 0.00048828125)
    unsqueeze_1547: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1798, 0);  mul_1798 = None
    unsqueeze_1548: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1547, 2);  unsqueeze_1547 = None
    unsqueeze_1549: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1548, 3);  unsqueeze_1548 = None
    mul_1799: "f32[512]" = torch.ops.aten.mul.Tensor(sum_300, 0.00048828125)
    mul_1800: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_1801: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1799, mul_1800);  mul_1799 = mul_1800 = None
    unsqueeze_1550: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1801, 0);  mul_1801 = None
    unsqueeze_1551: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1550, 2);  unsqueeze_1550 = None
    unsqueeze_1552: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1551, 3);  unsqueeze_1551 = None
    mul_1802: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_218);  primals_218 = None
    unsqueeze_1553: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1802, 0);  mul_1802 = None
    unsqueeze_1554: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1553, 2);  unsqueeze_1553 = None
    unsqueeze_1555: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, 3);  unsqueeze_1554 = None
    mul_1803: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_508, unsqueeze_1552);  sub_508 = unsqueeze_1552 = None
    sub_510: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_78, mul_1803);  where_78 = mul_1803 = None
    sub_511: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_510, unsqueeze_1549);  sub_510 = unsqueeze_1549 = None
    mul_1804: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_511, unsqueeze_1555);  sub_511 = unsqueeze_1555 = None
    mul_1805: "f32[512]" = torch.ops.aten.mul.Tensor(sum_300, squeeze_178);  sum_300 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(mul_1804, relu_55, primals_217, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1804 = primals_217 = None
    getitem_577: "f32[8, 256, 16, 16]" = convolution_backward_99[0]
    getitem_578: "f32[512, 128, 3, 3]" = convolution_backward_99[1];  convolution_backward_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_426: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_55);  relu_55 = None
    alias_427: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_426);  alias_426 = None
    le_79: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_427, 0);  alias_427 = None
    where_79: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_79, full_default, getitem_577);  le_79 = getitem_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_301: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_512: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_1558);  convolution_71 = unsqueeze_1558 = None
    mul_1806: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_79, sub_512)
    sum_302: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1806, [0, 2, 3]);  mul_1806 = None
    mul_1807: "f32[256]" = torch.ops.aten.mul.Tensor(sum_301, 0.00048828125)
    unsqueeze_1559: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1807, 0);  mul_1807 = None
    unsqueeze_1560: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1559, 2);  unsqueeze_1559 = None
    unsqueeze_1561: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1560, 3);  unsqueeze_1560 = None
    mul_1808: "f32[256]" = torch.ops.aten.mul.Tensor(sum_302, 0.00048828125)
    mul_1809: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_1810: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1808, mul_1809);  mul_1808 = mul_1809 = None
    unsqueeze_1562: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1810, 0);  mul_1810 = None
    unsqueeze_1563: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1562, 2);  unsqueeze_1562 = None
    unsqueeze_1564: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1563, 3);  unsqueeze_1563 = None
    mul_1811: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_215);  primals_215 = None
    unsqueeze_1565: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1811, 0);  mul_1811 = None
    unsqueeze_1566: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1565, 2);  unsqueeze_1565 = None
    unsqueeze_1567: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, 3);  unsqueeze_1566 = None
    mul_1812: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_512, unsqueeze_1564);  sub_512 = unsqueeze_1564 = None
    sub_514: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_79, mul_1812);  where_79 = mul_1812 = None
    sub_515: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_514, unsqueeze_1561);  sub_514 = unsqueeze_1561 = None
    mul_1813: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_515, unsqueeze_1567);  sub_515 = unsqueeze_1567 = None
    mul_1814: "f32[256]" = torch.ops.aten.mul.Tensor(sum_302, squeeze_175);  sum_302 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(mul_1813, relu_54, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1813 = primals_214 = None
    getitem_580: "f32[8, 1024, 16, 16]" = convolution_backward_100[0]
    getitem_581: "f32[256, 1024, 1, 1]" = convolution_backward_100[1];  convolution_backward_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_767: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_76, getitem_580);  where_76 = getitem_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_429: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_54);  relu_54 = None
    alias_430: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_429);  alias_429 = None
    le_80: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_430, 0);  alias_430 = None
    where_80: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_80, full_default, add_767);  le_80 = add_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_303: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_516: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_1570);  convolution_70 = unsqueeze_1570 = None
    mul_1815: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_80, sub_516)
    sum_304: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1815, [0, 2, 3]);  mul_1815 = None
    mul_1816: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_303, 0.00048828125)
    unsqueeze_1571: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1816, 0);  mul_1816 = None
    unsqueeze_1572: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1571, 2);  unsqueeze_1571 = None
    unsqueeze_1573: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1572, 3);  unsqueeze_1572 = None
    mul_1817: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_304, 0.00048828125)
    mul_1818: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_1819: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1817, mul_1818);  mul_1817 = mul_1818 = None
    unsqueeze_1574: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1819, 0);  mul_1819 = None
    unsqueeze_1575: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1574, 2);  unsqueeze_1574 = None
    unsqueeze_1576: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1575, 3);  unsqueeze_1575 = None
    mul_1820: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_212);  primals_212 = None
    unsqueeze_1577: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1820, 0);  mul_1820 = None
    unsqueeze_1578: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1577, 2);  unsqueeze_1577 = None
    unsqueeze_1579: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, 3);  unsqueeze_1578 = None
    mul_1821: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_516, unsqueeze_1576);  sub_516 = unsqueeze_1576 = None
    sub_518: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_80, mul_1821);  mul_1821 = None
    sub_519: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_518, unsqueeze_1573);  sub_518 = unsqueeze_1573 = None
    mul_1822: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_519, unsqueeze_1579);  sub_519 = unsqueeze_1579 = None
    mul_1823: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_304, squeeze_172);  sum_304 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(mul_1822, sum_39, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1822 = sum_39 = primals_211 = None
    getitem_583: "f32[8, 256, 16, 16]" = convolution_backward_101[0]
    getitem_584: "f32[1024, 256, 1, 1]" = convolution_backward_101[1];  convolution_backward_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1580: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_583, 1);  getitem_583 = None
    expand_61: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1580, [8, 2, 256, 16, 16]);  unsqueeze_1580 = None
    mul_1824: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_61, view_73);  view_73 = None
    mul_1825: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_61, view_77);  expand_61 = view_77 = None
    sum_305: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1824, [3, 4], True);  mul_1824 = None
    view_301: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_305, [8, 512, 1, 1]);  sum_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_302: "f32[8, 512]" = torch.ops.aten.view.default(view_301, [8, 512]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_303: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_302, [8, 2, 1, 256]);  view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_431: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    mul_1826: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_303, alias_431);  view_303 = None
    sum_306: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1826, [1], True)
    mul_1827: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_431, sum_306);  alias_431 = sum_306 = None
    sub_520: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1826, mul_1827);  mul_1826 = mul_1827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_58: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_520, [0, 2, 1, 3]);  sub_520 = None
    view_304: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_58, [8, 512, 1, 1]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(view_304, relu_53, primals_209, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_304 = primals_209 = None
    getitem_586: "f32[8, 128, 1, 1]" = convolution_backward_102[0]
    getitem_587: "f32[512, 128, 1, 1]" = convolution_backward_102[1]
    getitem_588: "f32[512]" = convolution_backward_102[2];  convolution_backward_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_433: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_53);  relu_53 = None
    alias_434: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_433);  alias_433 = None
    le_81: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_434, 0);  alias_434 = None
    where_81: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_81, full_default, getitem_586);  le_81 = getitem_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1581: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_1582: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1581, 2);  unsqueeze_1581 = None
    unsqueeze_1583: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1582, 3);  unsqueeze_1582 = None
    sum_307: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
    sub_521: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_1583);  convolution_68 = unsqueeze_1583 = None
    mul_1828: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_81, sub_521)
    sum_308: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1828, [0, 2, 3]);  mul_1828 = None
    mul_1829: "f32[128]" = torch.ops.aten.mul.Tensor(sum_307, 0.125)
    unsqueeze_1584: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1829, 0);  mul_1829 = None
    unsqueeze_1585: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1584, 2);  unsqueeze_1584 = None
    unsqueeze_1586: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1585, 3);  unsqueeze_1585 = None
    mul_1830: "f32[128]" = torch.ops.aten.mul.Tensor(sum_308, 0.125)
    mul_1831: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_1832: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1830, mul_1831);  mul_1830 = mul_1831 = None
    unsqueeze_1587: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1832, 0);  mul_1832 = None
    unsqueeze_1588: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1587, 2);  unsqueeze_1587 = None
    unsqueeze_1589: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1588, 3);  unsqueeze_1588 = None
    mul_1833: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_207);  primals_207 = None
    unsqueeze_1590: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1833, 0);  mul_1833 = None
    unsqueeze_1591: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, 2);  unsqueeze_1590 = None
    unsqueeze_1592: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1591, 3);  unsqueeze_1591 = None
    mul_1834: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_521, unsqueeze_1589);  sub_521 = unsqueeze_1589 = None
    sub_523: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_81, mul_1834);  where_81 = mul_1834 = None
    sub_524: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_523, unsqueeze_1586);  sub_523 = unsqueeze_1586 = None
    mul_1835: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_524, unsqueeze_1592);  sub_524 = unsqueeze_1592 = None
    mul_1836: "f32[128]" = torch.ops.aten.mul.Tensor(sum_308, squeeze_169);  sum_308 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(mul_1835, mean_12, primals_205, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1835 = mean_12 = primals_205 = None
    getitem_589: "f32[8, 256, 1, 1]" = convolution_backward_103[0]
    getitem_590: "f32[128, 256, 1, 1]" = convolution_backward_103[1]
    getitem_591: "f32[128]" = convolution_backward_103[2];  convolution_backward_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_62: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_589, [8, 256, 16, 16]);  getitem_589 = None
    div_54: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_62, 256);  expand_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1593: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_54, 1);  div_54 = None
    expand_63: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1593, [8, 2, 256, 16, 16]);  unsqueeze_1593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_768: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1825, expand_63);  mul_1825 = expand_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_305: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_768, [8, 512, 16, 16]);  add_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_436: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_52);  relu_52 = None
    alias_437: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_436);  alias_436 = None
    le_82: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_437, 0);  alias_437 = None
    where_82: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_82, full_default, view_305);  le_82 = view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_309: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
    sub_525: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_1596);  convolution_67 = unsqueeze_1596 = None
    mul_1837: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_82, sub_525)
    sum_310: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1837, [0, 2, 3]);  mul_1837 = None
    mul_1838: "f32[512]" = torch.ops.aten.mul.Tensor(sum_309, 0.00048828125)
    unsqueeze_1597: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1838, 0);  mul_1838 = None
    unsqueeze_1598: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1597, 2);  unsqueeze_1597 = None
    unsqueeze_1599: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1598, 3);  unsqueeze_1598 = None
    mul_1839: "f32[512]" = torch.ops.aten.mul.Tensor(sum_310, 0.00048828125)
    mul_1840: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_1841: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1839, mul_1840);  mul_1839 = mul_1840 = None
    unsqueeze_1600: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1841, 0);  mul_1841 = None
    unsqueeze_1601: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1600, 2);  unsqueeze_1600 = None
    unsqueeze_1602: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1601, 3);  unsqueeze_1601 = None
    mul_1842: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_203);  primals_203 = None
    unsqueeze_1603: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1842, 0);  mul_1842 = None
    unsqueeze_1604: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1603, 2);  unsqueeze_1603 = None
    unsqueeze_1605: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1604, 3);  unsqueeze_1604 = None
    mul_1843: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_525, unsqueeze_1602);  sub_525 = unsqueeze_1602 = None
    sub_527: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_82, mul_1843);  where_82 = mul_1843 = None
    sub_528: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_527, unsqueeze_1599);  sub_527 = unsqueeze_1599 = None
    mul_1844: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_528, unsqueeze_1605);  sub_528 = unsqueeze_1605 = None
    mul_1845: "f32[512]" = torch.ops.aten.mul.Tensor(sum_310, squeeze_166);  sum_310 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_104 = torch.ops.aten.convolution_backward.default(mul_1844, relu_51, primals_202, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1844 = primals_202 = None
    getitem_592: "f32[8, 256, 16, 16]" = convolution_backward_104[0]
    getitem_593: "f32[512, 128, 3, 3]" = convolution_backward_104[1];  convolution_backward_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_439: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_51);  relu_51 = None
    alias_440: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_439);  alias_439 = None
    le_83: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_440, 0);  alias_440 = None
    where_83: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_83, full_default, getitem_592);  le_83 = getitem_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_311: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
    sub_529: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_1608);  convolution_66 = unsqueeze_1608 = None
    mul_1846: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_83, sub_529)
    sum_312: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1846, [0, 2, 3]);  mul_1846 = None
    mul_1847: "f32[256]" = torch.ops.aten.mul.Tensor(sum_311, 0.00048828125)
    unsqueeze_1609: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1847, 0);  mul_1847 = None
    unsqueeze_1610: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1609, 2);  unsqueeze_1609 = None
    unsqueeze_1611: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1610, 3);  unsqueeze_1610 = None
    mul_1848: "f32[256]" = torch.ops.aten.mul.Tensor(sum_312, 0.00048828125)
    mul_1849: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_1850: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1848, mul_1849);  mul_1848 = mul_1849 = None
    unsqueeze_1612: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1850, 0);  mul_1850 = None
    unsqueeze_1613: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1612, 2);  unsqueeze_1612 = None
    unsqueeze_1614: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1613, 3);  unsqueeze_1613 = None
    mul_1851: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_200);  primals_200 = None
    unsqueeze_1615: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1851, 0);  mul_1851 = None
    unsqueeze_1616: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1615, 2);  unsqueeze_1615 = None
    unsqueeze_1617: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1616, 3);  unsqueeze_1616 = None
    mul_1852: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_529, unsqueeze_1614);  sub_529 = unsqueeze_1614 = None
    sub_531: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_83, mul_1852);  where_83 = mul_1852 = None
    sub_532: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_531, unsqueeze_1611);  sub_531 = unsqueeze_1611 = None
    mul_1853: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_532, unsqueeze_1617);  sub_532 = unsqueeze_1617 = None
    mul_1854: "f32[256]" = torch.ops.aten.mul.Tensor(sum_312, squeeze_163);  sum_312 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_105 = torch.ops.aten.convolution_backward.default(mul_1853, relu_50, primals_199, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1853 = primals_199 = None
    getitem_595: "f32[8, 1024, 16, 16]" = convolution_backward_105[0]
    getitem_596: "f32[256, 1024, 1, 1]" = convolution_backward_105[1];  convolution_backward_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_769: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_80, getitem_595);  where_80 = getitem_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_442: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_50);  relu_50 = None
    alias_443: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_442);  alias_442 = None
    le_84: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_443, 0);  alias_443 = None
    where_84: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_84, full_default, add_769);  le_84 = add_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_313: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
    sub_533: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_1620);  convolution_65 = unsqueeze_1620 = None
    mul_1855: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_84, sub_533)
    sum_314: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1855, [0, 2, 3]);  mul_1855 = None
    mul_1856: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_313, 0.00048828125)
    unsqueeze_1621: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1856, 0);  mul_1856 = None
    unsqueeze_1622: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1621, 2);  unsqueeze_1621 = None
    unsqueeze_1623: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1622, 3);  unsqueeze_1622 = None
    mul_1857: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_314, 0.00048828125)
    mul_1858: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_1859: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1857, mul_1858);  mul_1857 = mul_1858 = None
    unsqueeze_1624: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1859, 0);  mul_1859 = None
    unsqueeze_1625: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1624, 2);  unsqueeze_1624 = None
    unsqueeze_1626: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1625, 3);  unsqueeze_1625 = None
    mul_1860: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_197);  primals_197 = None
    unsqueeze_1627: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1860, 0);  mul_1860 = None
    unsqueeze_1628: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1627, 2);  unsqueeze_1627 = None
    unsqueeze_1629: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1628, 3);  unsqueeze_1628 = None
    mul_1861: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_533, unsqueeze_1626);  sub_533 = unsqueeze_1626 = None
    sub_535: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_84, mul_1861);  mul_1861 = None
    sub_536: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_535, unsqueeze_1623);  sub_535 = unsqueeze_1623 = None
    mul_1862: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_536, unsqueeze_1629);  sub_536 = unsqueeze_1629 = None
    mul_1863: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_314, squeeze_160);  sum_314 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_106 = torch.ops.aten.convolution_backward.default(mul_1862, sum_36, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1862 = sum_36 = primals_196 = None
    getitem_598: "f32[8, 256, 16, 16]" = convolution_backward_106[0]
    getitem_599: "f32[1024, 256, 1, 1]" = convolution_backward_106[1];  convolution_backward_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1630: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_598, 1);  getitem_598 = None
    expand_64: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1630, [8, 2, 256, 16, 16]);  unsqueeze_1630 = None
    mul_1864: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_64, view_67);  view_67 = None
    mul_1865: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_64, view_71);  expand_64 = view_71 = None
    sum_315: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1864, [3, 4], True);  mul_1864 = None
    view_306: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_315, [8, 512, 1, 1]);  sum_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_307: "f32[8, 512]" = torch.ops.aten.view.default(view_306, [8, 512]);  view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_308: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_307, [8, 2, 1, 256]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_444: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_61);  alias_61 = None
    mul_1866: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_308, alias_444);  view_308 = None
    sum_316: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1866, [1], True)
    mul_1867: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_444, sum_316);  alias_444 = sum_316 = None
    sub_537: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1866, mul_1867);  mul_1866 = mul_1867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_59: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_537, [0, 2, 1, 3]);  sub_537 = None
    view_309: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_59, [8, 512, 1, 1]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_107 = torch.ops.aten.convolution_backward.default(view_309, relu_49, primals_194, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_309 = primals_194 = None
    getitem_601: "f32[8, 128, 1, 1]" = convolution_backward_107[0]
    getitem_602: "f32[512, 128, 1, 1]" = convolution_backward_107[1]
    getitem_603: "f32[512]" = convolution_backward_107[2];  convolution_backward_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_446: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_49);  relu_49 = None
    alias_447: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_446);  alias_446 = None
    le_85: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_447, 0);  alias_447 = None
    where_85: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_85, full_default, getitem_601);  le_85 = getitem_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1631: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_1632: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1631, 2);  unsqueeze_1631 = None
    unsqueeze_1633: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1632, 3);  unsqueeze_1632 = None
    sum_317: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_538: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_1633);  convolution_63 = unsqueeze_1633 = None
    mul_1868: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_85, sub_538)
    sum_318: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1868, [0, 2, 3]);  mul_1868 = None
    mul_1869: "f32[128]" = torch.ops.aten.mul.Tensor(sum_317, 0.125)
    unsqueeze_1634: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1869, 0);  mul_1869 = None
    unsqueeze_1635: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1634, 2);  unsqueeze_1634 = None
    unsqueeze_1636: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1635, 3);  unsqueeze_1635 = None
    mul_1870: "f32[128]" = torch.ops.aten.mul.Tensor(sum_318, 0.125)
    mul_1871: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_1872: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1870, mul_1871);  mul_1870 = mul_1871 = None
    unsqueeze_1637: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1872, 0);  mul_1872 = None
    unsqueeze_1638: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1637, 2);  unsqueeze_1637 = None
    unsqueeze_1639: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, 3);  unsqueeze_1638 = None
    mul_1873: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_192);  primals_192 = None
    unsqueeze_1640: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1873, 0);  mul_1873 = None
    unsqueeze_1641: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1640, 2);  unsqueeze_1640 = None
    unsqueeze_1642: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1641, 3);  unsqueeze_1641 = None
    mul_1874: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_538, unsqueeze_1639);  sub_538 = unsqueeze_1639 = None
    sub_540: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_85, mul_1874);  where_85 = mul_1874 = None
    sub_541: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_540, unsqueeze_1636);  sub_540 = unsqueeze_1636 = None
    mul_1875: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_541, unsqueeze_1642);  sub_541 = unsqueeze_1642 = None
    mul_1876: "f32[128]" = torch.ops.aten.mul.Tensor(sum_318, squeeze_157);  sum_318 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_108 = torch.ops.aten.convolution_backward.default(mul_1875, mean_11, primals_190, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1875 = mean_11 = primals_190 = None
    getitem_604: "f32[8, 256, 1, 1]" = convolution_backward_108[0]
    getitem_605: "f32[128, 256, 1, 1]" = convolution_backward_108[1]
    getitem_606: "f32[128]" = convolution_backward_108[2];  convolution_backward_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_65: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_604, [8, 256, 16, 16]);  getitem_604 = None
    div_55: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_65, 256);  expand_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1643: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_55, 1);  div_55 = None
    expand_66: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1643, [8, 2, 256, 16, 16]);  unsqueeze_1643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_770: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1865, expand_66);  mul_1865 = expand_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_310: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_770, [8, 512, 16, 16]);  add_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_449: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_48);  relu_48 = None
    alias_450: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_449);  alias_449 = None
    le_86: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_450, 0);  alias_450 = None
    where_86: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_86, full_default, view_310);  le_86 = view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_319: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
    sub_542: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_1646);  convolution_62 = unsqueeze_1646 = None
    mul_1877: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_86, sub_542)
    sum_320: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1877, [0, 2, 3]);  mul_1877 = None
    mul_1878: "f32[512]" = torch.ops.aten.mul.Tensor(sum_319, 0.00048828125)
    unsqueeze_1647: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1878, 0);  mul_1878 = None
    unsqueeze_1648: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1647, 2);  unsqueeze_1647 = None
    unsqueeze_1649: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1648, 3);  unsqueeze_1648 = None
    mul_1879: "f32[512]" = torch.ops.aten.mul.Tensor(sum_320, 0.00048828125)
    mul_1880: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_1881: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1879, mul_1880);  mul_1879 = mul_1880 = None
    unsqueeze_1650: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1881, 0);  mul_1881 = None
    unsqueeze_1651: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, 2);  unsqueeze_1650 = None
    unsqueeze_1652: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1651, 3);  unsqueeze_1651 = None
    mul_1882: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_188);  primals_188 = None
    unsqueeze_1653: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1882, 0);  mul_1882 = None
    unsqueeze_1654: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1653, 2);  unsqueeze_1653 = None
    unsqueeze_1655: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1654, 3);  unsqueeze_1654 = None
    mul_1883: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_542, unsqueeze_1652);  sub_542 = unsqueeze_1652 = None
    sub_544: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_86, mul_1883);  where_86 = mul_1883 = None
    sub_545: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_544, unsqueeze_1649);  sub_544 = unsqueeze_1649 = None
    mul_1884: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_545, unsqueeze_1655);  sub_545 = unsqueeze_1655 = None
    mul_1885: "f32[512]" = torch.ops.aten.mul.Tensor(sum_320, squeeze_154);  sum_320 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_109 = torch.ops.aten.convolution_backward.default(mul_1884, relu_47, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1884 = primals_187 = None
    getitem_607: "f32[8, 256, 16, 16]" = convolution_backward_109[0]
    getitem_608: "f32[512, 128, 3, 3]" = convolution_backward_109[1];  convolution_backward_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_452: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_47);  relu_47 = None
    alias_453: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_452);  alias_452 = None
    le_87: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_453, 0);  alias_453 = None
    where_87: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_87, full_default, getitem_607);  le_87 = getitem_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_321: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_546: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_1658);  convolution_61 = unsqueeze_1658 = None
    mul_1886: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_87, sub_546)
    sum_322: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1886, [0, 2, 3]);  mul_1886 = None
    mul_1887: "f32[256]" = torch.ops.aten.mul.Tensor(sum_321, 0.00048828125)
    unsqueeze_1659: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1887, 0);  mul_1887 = None
    unsqueeze_1660: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1659, 2);  unsqueeze_1659 = None
    unsqueeze_1661: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1660, 3);  unsqueeze_1660 = None
    mul_1888: "f32[256]" = torch.ops.aten.mul.Tensor(sum_322, 0.00048828125)
    mul_1889: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_1890: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1888, mul_1889);  mul_1888 = mul_1889 = None
    unsqueeze_1662: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1890, 0);  mul_1890 = None
    unsqueeze_1663: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, 2);  unsqueeze_1662 = None
    unsqueeze_1664: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1663, 3);  unsqueeze_1663 = None
    mul_1891: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_185);  primals_185 = None
    unsqueeze_1665: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1891, 0);  mul_1891 = None
    unsqueeze_1666: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1665, 2);  unsqueeze_1665 = None
    unsqueeze_1667: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1666, 3);  unsqueeze_1666 = None
    mul_1892: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_546, unsqueeze_1664);  sub_546 = unsqueeze_1664 = None
    sub_548: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_87, mul_1892);  where_87 = mul_1892 = None
    sub_549: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_548, unsqueeze_1661);  sub_548 = unsqueeze_1661 = None
    mul_1893: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_549, unsqueeze_1667);  sub_549 = unsqueeze_1667 = None
    mul_1894: "f32[256]" = torch.ops.aten.mul.Tensor(sum_322, squeeze_151);  sum_322 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_110 = torch.ops.aten.convolution_backward.default(mul_1893, relu_46, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1893 = primals_184 = None
    getitem_610: "f32[8, 1024, 16, 16]" = convolution_backward_110[0]
    getitem_611: "f32[256, 1024, 1, 1]" = convolution_backward_110[1];  convolution_backward_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_771: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_84, getitem_610);  where_84 = getitem_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_455: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_46);  relu_46 = None
    alias_456: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_455);  alias_455 = None
    le_88: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_456, 0);  alias_456 = None
    where_88: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_88, full_default, add_771);  le_88 = add_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_323: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_550: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_1670);  convolution_60 = unsqueeze_1670 = None
    mul_1895: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_88, sub_550)
    sum_324: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1895, [0, 2, 3]);  mul_1895 = None
    mul_1896: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_323, 0.00048828125)
    unsqueeze_1671: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1896, 0);  mul_1896 = None
    unsqueeze_1672: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1671, 2);  unsqueeze_1671 = None
    unsqueeze_1673: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1672, 3);  unsqueeze_1672 = None
    mul_1897: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_324, 0.00048828125)
    mul_1898: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_1899: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1897, mul_1898);  mul_1897 = mul_1898 = None
    unsqueeze_1674: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1899, 0);  mul_1899 = None
    unsqueeze_1675: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1674, 2);  unsqueeze_1674 = None
    unsqueeze_1676: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1675, 3);  unsqueeze_1675 = None
    mul_1900: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_182);  primals_182 = None
    unsqueeze_1677: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1900, 0);  mul_1900 = None
    unsqueeze_1678: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1677, 2);  unsqueeze_1677 = None
    unsqueeze_1679: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1678, 3);  unsqueeze_1678 = None
    mul_1901: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_550, unsqueeze_1676);  sub_550 = unsqueeze_1676 = None
    sub_552: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_88, mul_1901);  mul_1901 = None
    sub_553: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_552, unsqueeze_1673);  sub_552 = unsqueeze_1673 = None
    mul_1902: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_553, unsqueeze_1679);  sub_553 = unsqueeze_1679 = None
    mul_1903: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_324, squeeze_148);  sum_324 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_111 = torch.ops.aten.convolution_backward.default(mul_1902, sum_33, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1902 = sum_33 = primals_181 = None
    getitem_613: "f32[8, 256, 16, 16]" = convolution_backward_111[0]
    getitem_614: "f32[1024, 256, 1, 1]" = convolution_backward_111[1];  convolution_backward_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1680: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_613, 1);  getitem_613 = None
    expand_67: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1680, [8, 2, 256, 16, 16]);  unsqueeze_1680 = None
    mul_1904: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_67, view_61);  view_61 = None
    mul_1905: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_67, view_65);  expand_67 = view_65 = None
    sum_325: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1904, [3, 4], True);  mul_1904 = None
    view_311: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_325, [8, 512, 1, 1]);  sum_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_312: "f32[8, 512]" = torch.ops.aten.view.default(view_311, [8, 512]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_313: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_312, [8, 2, 1, 256]);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_457: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    mul_1906: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_313, alias_457);  view_313 = None
    sum_326: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1906, [1], True)
    mul_1907: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_457, sum_326);  alias_457 = sum_326 = None
    sub_554: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1906, mul_1907);  mul_1906 = mul_1907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_60: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_554, [0, 2, 1, 3]);  sub_554 = None
    view_314: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_60, [8, 512, 1, 1]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_112 = torch.ops.aten.convolution_backward.default(view_314, relu_45, primals_179, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_314 = primals_179 = None
    getitem_616: "f32[8, 128, 1, 1]" = convolution_backward_112[0]
    getitem_617: "f32[512, 128, 1, 1]" = convolution_backward_112[1]
    getitem_618: "f32[512]" = convolution_backward_112[2];  convolution_backward_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_459: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_45);  relu_45 = None
    alias_460: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_459);  alias_459 = None
    le_89: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_460, 0);  alias_460 = None
    where_89: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_89, full_default, getitem_616);  le_89 = getitem_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1681: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_1682: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1681, 2);  unsqueeze_1681 = None
    unsqueeze_1683: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1682, 3);  unsqueeze_1682 = None
    sum_327: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
    sub_555: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_1683);  convolution_58 = unsqueeze_1683 = None
    mul_1908: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_89, sub_555)
    sum_328: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1908, [0, 2, 3]);  mul_1908 = None
    mul_1909: "f32[128]" = torch.ops.aten.mul.Tensor(sum_327, 0.125)
    unsqueeze_1684: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1909, 0);  mul_1909 = None
    unsqueeze_1685: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1684, 2);  unsqueeze_1684 = None
    unsqueeze_1686: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1685, 3);  unsqueeze_1685 = None
    mul_1910: "f32[128]" = torch.ops.aten.mul.Tensor(sum_328, 0.125)
    mul_1911: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_1912: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1910, mul_1911);  mul_1910 = mul_1911 = None
    unsqueeze_1687: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1912, 0);  mul_1912 = None
    unsqueeze_1688: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1687, 2);  unsqueeze_1687 = None
    unsqueeze_1689: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1688, 3);  unsqueeze_1688 = None
    mul_1913: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_177);  primals_177 = None
    unsqueeze_1690: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1913, 0);  mul_1913 = None
    unsqueeze_1691: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1690, 2);  unsqueeze_1690 = None
    unsqueeze_1692: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1691, 3);  unsqueeze_1691 = None
    mul_1914: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_555, unsqueeze_1689);  sub_555 = unsqueeze_1689 = None
    sub_557: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_89, mul_1914);  where_89 = mul_1914 = None
    sub_558: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_557, unsqueeze_1686);  sub_557 = unsqueeze_1686 = None
    mul_1915: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_558, unsqueeze_1692);  sub_558 = unsqueeze_1692 = None
    mul_1916: "f32[128]" = torch.ops.aten.mul.Tensor(sum_328, squeeze_145);  sum_328 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_113 = torch.ops.aten.convolution_backward.default(mul_1915, mean_10, primals_175, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1915 = mean_10 = primals_175 = None
    getitem_619: "f32[8, 256, 1, 1]" = convolution_backward_113[0]
    getitem_620: "f32[128, 256, 1, 1]" = convolution_backward_113[1]
    getitem_621: "f32[128]" = convolution_backward_113[2];  convolution_backward_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_68: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_619, [8, 256, 16, 16]);  getitem_619 = None
    div_56: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_68, 256);  expand_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1693: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_56, 1);  div_56 = None
    expand_69: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1693, [8, 2, 256, 16, 16]);  unsqueeze_1693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_772: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1905, expand_69);  mul_1905 = expand_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_315: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_772, [8, 512, 16, 16]);  add_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_462: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_44);  relu_44 = None
    alias_463: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_462);  alias_462 = None
    le_90: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_463, 0);  alias_463 = None
    where_90: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_90, full_default, view_315);  le_90 = view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_329: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
    sub_559: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_1696);  convolution_57 = unsqueeze_1696 = None
    mul_1917: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_90, sub_559)
    sum_330: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1917, [0, 2, 3]);  mul_1917 = None
    mul_1918: "f32[512]" = torch.ops.aten.mul.Tensor(sum_329, 0.00048828125)
    unsqueeze_1697: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1918, 0);  mul_1918 = None
    unsqueeze_1698: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1697, 2);  unsqueeze_1697 = None
    unsqueeze_1699: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1698, 3);  unsqueeze_1698 = None
    mul_1919: "f32[512]" = torch.ops.aten.mul.Tensor(sum_330, 0.00048828125)
    mul_1920: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_1921: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1919, mul_1920);  mul_1919 = mul_1920 = None
    unsqueeze_1700: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1921, 0);  mul_1921 = None
    unsqueeze_1701: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1700, 2);  unsqueeze_1700 = None
    unsqueeze_1702: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1701, 3);  unsqueeze_1701 = None
    mul_1922: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_173);  primals_173 = None
    unsqueeze_1703: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1922, 0);  mul_1922 = None
    unsqueeze_1704: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1703, 2);  unsqueeze_1703 = None
    unsqueeze_1705: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1704, 3);  unsqueeze_1704 = None
    mul_1923: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_559, unsqueeze_1702);  sub_559 = unsqueeze_1702 = None
    sub_561: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_90, mul_1923);  where_90 = mul_1923 = None
    sub_562: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_561, unsqueeze_1699);  sub_561 = unsqueeze_1699 = None
    mul_1924: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_562, unsqueeze_1705);  sub_562 = unsqueeze_1705 = None
    mul_1925: "f32[512]" = torch.ops.aten.mul.Tensor(sum_330, squeeze_142);  sum_330 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_114 = torch.ops.aten.convolution_backward.default(mul_1924, relu_43, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1924 = primals_172 = None
    getitem_622: "f32[8, 256, 16, 16]" = convolution_backward_114[0]
    getitem_623: "f32[512, 128, 3, 3]" = convolution_backward_114[1];  convolution_backward_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_465: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_43);  relu_43 = None
    alias_466: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_465);  alias_465 = None
    le_91: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_466, 0);  alias_466 = None
    where_91: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_91, full_default, getitem_622);  le_91 = getitem_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_331: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
    sub_563: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_1708);  convolution_56 = unsqueeze_1708 = None
    mul_1926: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_91, sub_563)
    sum_332: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1926, [0, 2, 3]);  mul_1926 = None
    mul_1927: "f32[256]" = torch.ops.aten.mul.Tensor(sum_331, 0.00048828125)
    unsqueeze_1709: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1927, 0);  mul_1927 = None
    unsqueeze_1710: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1709, 2);  unsqueeze_1709 = None
    unsqueeze_1711: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1710, 3);  unsqueeze_1710 = None
    mul_1928: "f32[256]" = torch.ops.aten.mul.Tensor(sum_332, 0.00048828125)
    mul_1929: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_1930: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1928, mul_1929);  mul_1928 = mul_1929 = None
    unsqueeze_1712: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1930, 0);  mul_1930 = None
    unsqueeze_1713: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1712, 2);  unsqueeze_1712 = None
    unsqueeze_1714: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1713, 3);  unsqueeze_1713 = None
    mul_1931: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_170);  primals_170 = None
    unsqueeze_1715: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1931, 0);  mul_1931 = None
    unsqueeze_1716: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1715, 2);  unsqueeze_1715 = None
    unsqueeze_1717: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1716, 3);  unsqueeze_1716 = None
    mul_1932: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_563, unsqueeze_1714);  sub_563 = unsqueeze_1714 = None
    sub_565: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_91, mul_1932);  where_91 = mul_1932 = None
    sub_566: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_565, unsqueeze_1711);  sub_565 = unsqueeze_1711 = None
    mul_1933: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_566, unsqueeze_1717);  sub_566 = unsqueeze_1717 = None
    mul_1934: "f32[256]" = torch.ops.aten.mul.Tensor(sum_332, squeeze_139);  sum_332 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_115 = torch.ops.aten.convolution_backward.default(mul_1933, relu_42, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1933 = primals_169 = None
    getitem_625: "f32[8, 1024, 16, 16]" = convolution_backward_115[0]
    getitem_626: "f32[256, 1024, 1, 1]" = convolution_backward_115[1];  convolution_backward_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_773: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_88, getitem_625);  where_88 = getitem_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_468: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
    alias_469: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_468);  alias_468 = None
    le_92: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_469, 0);  alias_469 = None
    where_92: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_92, full_default, add_773);  le_92 = add_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_333: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_567: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_1720);  convolution_55 = unsqueeze_1720 = None
    mul_1935: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_92, sub_567)
    sum_334: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1935, [0, 2, 3]);  mul_1935 = None
    mul_1936: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_333, 0.00048828125)
    unsqueeze_1721: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1936, 0);  mul_1936 = None
    unsqueeze_1722: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1721, 2);  unsqueeze_1721 = None
    unsqueeze_1723: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1722, 3);  unsqueeze_1722 = None
    mul_1937: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_334, 0.00048828125)
    mul_1938: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_1939: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1937, mul_1938);  mul_1937 = mul_1938 = None
    unsqueeze_1724: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1939, 0);  mul_1939 = None
    unsqueeze_1725: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1724, 2);  unsqueeze_1724 = None
    unsqueeze_1726: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1725, 3);  unsqueeze_1725 = None
    mul_1940: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_167);  primals_167 = None
    unsqueeze_1727: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1940, 0);  mul_1940 = None
    unsqueeze_1728: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1727, 2);  unsqueeze_1727 = None
    unsqueeze_1729: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1728, 3);  unsqueeze_1728 = None
    mul_1941: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_567, unsqueeze_1726);  sub_567 = unsqueeze_1726 = None
    sub_569: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_92, mul_1941);  mul_1941 = None
    sub_570: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_569, unsqueeze_1723);  sub_569 = unsqueeze_1723 = None
    mul_1942: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_570, unsqueeze_1729);  sub_570 = unsqueeze_1729 = None
    mul_1943: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_334, squeeze_136);  sum_334 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_116 = torch.ops.aten.convolution_backward.default(mul_1942, sum_30, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1942 = sum_30 = primals_166 = None
    getitem_628: "f32[8, 256, 16, 16]" = convolution_backward_116[0]
    getitem_629: "f32[1024, 256, 1, 1]" = convolution_backward_116[1];  convolution_backward_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1730: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_628, 1);  getitem_628 = None
    expand_70: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1730, [8, 2, 256, 16, 16]);  unsqueeze_1730 = None
    mul_1944: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_70, view_55);  view_55 = None
    mul_1945: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_70, view_59);  expand_70 = view_59 = None
    sum_335: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1944, [3, 4], True);  mul_1944 = None
    view_316: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_335, [8, 512, 1, 1]);  sum_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_317: "f32[8, 512]" = torch.ops.aten.view.default(view_316, [8, 512]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_318: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_317, [8, 2, 1, 256]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_470: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    mul_1946: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_318, alias_470);  view_318 = None
    sum_336: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1946, [1], True)
    mul_1947: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_470, sum_336);  alias_470 = sum_336 = None
    sub_571: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1946, mul_1947);  mul_1946 = mul_1947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_61: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_571, [0, 2, 1, 3]);  sub_571 = None
    view_319: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_61, [8, 512, 1, 1]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_117 = torch.ops.aten.convolution_backward.default(view_319, relu_41, primals_164, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_319 = primals_164 = None
    getitem_631: "f32[8, 128, 1, 1]" = convolution_backward_117[0]
    getitem_632: "f32[512, 128, 1, 1]" = convolution_backward_117[1]
    getitem_633: "f32[512]" = convolution_backward_117[2];  convolution_backward_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_472: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_473: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_472);  alias_472 = None
    le_93: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_473, 0);  alias_473 = None
    where_93: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_93, full_default, getitem_631);  le_93 = getitem_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1731: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_1732: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1731, 2);  unsqueeze_1731 = None
    unsqueeze_1733: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1732, 3);  unsqueeze_1732 = None
    sum_337: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
    sub_572: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_1733);  convolution_53 = unsqueeze_1733 = None
    mul_1948: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_93, sub_572)
    sum_338: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1948, [0, 2, 3]);  mul_1948 = None
    mul_1949: "f32[128]" = torch.ops.aten.mul.Tensor(sum_337, 0.125)
    unsqueeze_1734: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1949, 0);  mul_1949 = None
    unsqueeze_1735: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1734, 2);  unsqueeze_1734 = None
    unsqueeze_1736: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1735, 3);  unsqueeze_1735 = None
    mul_1950: "f32[128]" = torch.ops.aten.mul.Tensor(sum_338, 0.125)
    mul_1951: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_1952: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1950, mul_1951);  mul_1950 = mul_1951 = None
    unsqueeze_1737: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1952, 0);  mul_1952 = None
    unsqueeze_1738: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1737, 2);  unsqueeze_1737 = None
    unsqueeze_1739: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1738, 3);  unsqueeze_1738 = None
    mul_1953: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_162);  primals_162 = None
    unsqueeze_1740: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1953, 0);  mul_1953 = None
    unsqueeze_1741: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1740, 2);  unsqueeze_1740 = None
    unsqueeze_1742: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1741, 3);  unsqueeze_1741 = None
    mul_1954: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_572, unsqueeze_1739);  sub_572 = unsqueeze_1739 = None
    sub_574: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_93, mul_1954);  where_93 = mul_1954 = None
    sub_575: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_574, unsqueeze_1736);  sub_574 = unsqueeze_1736 = None
    mul_1955: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_575, unsqueeze_1742);  sub_575 = unsqueeze_1742 = None
    mul_1956: "f32[128]" = torch.ops.aten.mul.Tensor(sum_338, squeeze_133);  sum_338 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_118 = torch.ops.aten.convolution_backward.default(mul_1955, mean_9, primals_160, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1955 = mean_9 = primals_160 = None
    getitem_634: "f32[8, 256, 1, 1]" = convolution_backward_118[0]
    getitem_635: "f32[128, 256, 1, 1]" = convolution_backward_118[1]
    getitem_636: "f32[128]" = convolution_backward_118[2];  convolution_backward_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_71: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_634, [8, 256, 16, 16]);  getitem_634 = None
    div_57: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_71, 256);  expand_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1743: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_57, 1);  div_57 = None
    expand_72: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1743, [8, 2, 256, 16, 16]);  unsqueeze_1743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_774: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1945, expand_72);  mul_1945 = expand_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_320: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_774, [8, 512, 16, 16]);  add_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_475: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_40);  relu_40 = None
    alias_476: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_475);  alias_475 = None
    le_94: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_476, 0);  alias_476 = None
    where_94: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_94, full_default, view_320);  le_94 = view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_339: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_576: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_1746);  convolution_52 = unsqueeze_1746 = None
    mul_1957: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_94, sub_576)
    sum_340: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1957, [0, 2, 3]);  mul_1957 = None
    mul_1958: "f32[512]" = torch.ops.aten.mul.Tensor(sum_339, 0.00048828125)
    unsqueeze_1747: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1958, 0);  mul_1958 = None
    unsqueeze_1748: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1747, 2);  unsqueeze_1747 = None
    unsqueeze_1749: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1748, 3);  unsqueeze_1748 = None
    mul_1959: "f32[512]" = torch.ops.aten.mul.Tensor(sum_340, 0.00048828125)
    mul_1960: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_1961: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1959, mul_1960);  mul_1959 = mul_1960 = None
    unsqueeze_1750: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1961, 0);  mul_1961 = None
    unsqueeze_1751: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1750, 2);  unsqueeze_1750 = None
    unsqueeze_1752: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1751, 3);  unsqueeze_1751 = None
    mul_1962: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_158);  primals_158 = None
    unsqueeze_1753: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1962, 0);  mul_1962 = None
    unsqueeze_1754: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1753, 2);  unsqueeze_1753 = None
    unsqueeze_1755: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1754, 3);  unsqueeze_1754 = None
    mul_1963: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_576, unsqueeze_1752);  sub_576 = unsqueeze_1752 = None
    sub_578: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_94, mul_1963);  where_94 = mul_1963 = None
    sub_579: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_578, unsqueeze_1749);  sub_578 = unsqueeze_1749 = None
    mul_1964: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_579, unsqueeze_1755);  sub_579 = unsqueeze_1755 = None
    mul_1965: "f32[512]" = torch.ops.aten.mul.Tensor(sum_340, squeeze_130);  sum_340 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_119 = torch.ops.aten.convolution_backward.default(mul_1964, relu_39, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1964 = primals_157 = None
    getitem_637: "f32[8, 256, 16, 16]" = convolution_backward_119[0]
    getitem_638: "f32[512, 128, 3, 3]" = convolution_backward_119[1];  convolution_backward_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_478: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_479: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_478);  alias_478 = None
    le_95: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_479, 0);  alias_479 = None
    where_95: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_95, full_default, getitem_637);  le_95 = getitem_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_341: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
    sub_580: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_1758);  convolution_51 = unsqueeze_1758 = None
    mul_1966: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_95, sub_580)
    sum_342: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1966, [0, 2, 3]);  mul_1966 = None
    mul_1967: "f32[256]" = torch.ops.aten.mul.Tensor(sum_341, 0.00048828125)
    unsqueeze_1759: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1967, 0);  mul_1967 = None
    unsqueeze_1760: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1759, 2);  unsqueeze_1759 = None
    unsqueeze_1761: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1760, 3);  unsqueeze_1760 = None
    mul_1968: "f32[256]" = torch.ops.aten.mul.Tensor(sum_342, 0.00048828125)
    mul_1969: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_1970: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1968, mul_1969);  mul_1968 = mul_1969 = None
    unsqueeze_1762: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1970, 0);  mul_1970 = None
    unsqueeze_1763: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1762, 2);  unsqueeze_1762 = None
    unsqueeze_1764: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1763, 3);  unsqueeze_1763 = None
    mul_1971: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_155);  primals_155 = None
    unsqueeze_1765: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1971, 0);  mul_1971 = None
    unsqueeze_1766: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1765, 2);  unsqueeze_1765 = None
    unsqueeze_1767: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1766, 3);  unsqueeze_1766 = None
    mul_1972: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_580, unsqueeze_1764);  sub_580 = unsqueeze_1764 = None
    sub_582: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_95, mul_1972);  where_95 = mul_1972 = None
    sub_583: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_582, unsqueeze_1761);  sub_582 = unsqueeze_1761 = None
    mul_1973: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_583, unsqueeze_1767);  sub_583 = unsqueeze_1767 = None
    mul_1974: "f32[256]" = torch.ops.aten.mul.Tensor(sum_342, squeeze_127);  sum_342 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_120 = torch.ops.aten.convolution_backward.default(mul_1973, relu_38, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1973 = primals_154 = None
    getitem_640: "f32[8, 1024, 16, 16]" = convolution_backward_120[0]
    getitem_641: "f32[256, 1024, 1, 1]" = convolution_backward_120[1];  convolution_backward_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_775: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_92, getitem_640);  where_92 = getitem_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_481: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_482: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_481);  alias_481 = None
    le_96: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_482, 0);  alias_482 = None
    where_96: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_96, full_default, add_775);  le_96 = add_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_343: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_584: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_1770);  convolution_50 = unsqueeze_1770 = None
    mul_1975: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_96, sub_584)
    sum_344: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1975, [0, 2, 3]);  mul_1975 = None
    mul_1976: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_343, 0.00048828125)
    unsqueeze_1771: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1976, 0);  mul_1976 = None
    unsqueeze_1772: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1771, 2);  unsqueeze_1771 = None
    unsqueeze_1773: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1772, 3);  unsqueeze_1772 = None
    mul_1977: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_344, 0.00048828125)
    mul_1978: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_1979: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1977, mul_1978);  mul_1977 = mul_1978 = None
    unsqueeze_1774: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1979, 0);  mul_1979 = None
    unsqueeze_1775: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1774, 2);  unsqueeze_1774 = None
    unsqueeze_1776: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1775, 3);  unsqueeze_1775 = None
    mul_1980: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_152);  primals_152 = None
    unsqueeze_1777: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1980, 0);  mul_1980 = None
    unsqueeze_1778: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1777, 2);  unsqueeze_1777 = None
    unsqueeze_1779: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1778, 3);  unsqueeze_1778 = None
    mul_1981: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_584, unsqueeze_1776);  sub_584 = unsqueeze_1776 = None
    sub_586: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_96, mul_1981);  mul_1981 = None
    sub_587: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_586, unsqueeze_1773);  sub_586 = unsqueeze_1773 = None
    mul_1982: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_587, unsqueeze_1779);  sub_587 = unsqueeze_1779 = None
    mul_1983: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_344, squeeze_124);  sum_344 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_121 = torch.ops.aten.convolution_backward.default(mul_1982, sum_27, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1982 = sum_27 = primals_151 = None
    getitem_643: "f32[8, 256, 16, 16]" = convolution_backward_121[0]
    getitem_644: "f32[1024, 256, 1, 1]" = convolution_backward_121[1];  convolution_backward_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1780: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(getitem_643, 1);  getitem_643 = None
    expand_73: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1780, [8, 2, 256, 16, 16]);  unsqueeze_1780 = None
    mul_1984: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_73, view_49);  view_49 = None
    mul_1985: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(expand_73, view_53);  expand_73 = view_53 = None
    sum_345: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1984, [3, 4], True);  mul_1984 = None
    view_321: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_345, [8, 512, 1, 1]);  sum_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_322: "f32[8, 512]" = torch.ops.aten.view.default(view_321, [8, 512]);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_323: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_322, [8, 2, 1, 256]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_483: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    mul_1986: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_323, alias_483);  view_323 = None
    sum_346: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_1986, [1], True)
    mul_1987: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_483, sum_346);  alias_483 = sum_346 = None
    sub_588: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_1986, mul_1987);  mul_1986 = mul_1987 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_62: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_588, [0, 2, 1, 3]);  sub_588 = None
    view_324: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_62, [8, 512, 1, 1]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_122 = torch.ops.aten.convolution_backward.default(view_324, relu_37, primals_149, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_324 = primals_149 = None
    getitem_646: "f32[8, 128, 1, 1]" = convolution_backward_122[0]
    getitem_647: "f32[512, 128, 1, 1]" = convolution_backward_122[1]
    getitem_648: "f32[512]" = convolution_backward_122[2];  convolution_backward_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_485: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_486: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_485);  alias_485 = None
    le_97: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_486, 0);  alias_486 = None
    where_97: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_97, full_default, getitem_646);  le_97 = getitem_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1781: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_1782: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1781, 2);  unsqueeze_1781 = None
    unsqueeze_1783: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1782, 3);  unsqueeze_1782 = None
    sum_347: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_97, [0, 2, 3])
    sub_589: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_1783);  convolution_48 = unsqueeze_1783 = None
    mul_1988: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_97, sub_589)
    sum_348: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1988, [0, 2, 3]);  mul_1988 = None
    mul_1989: "f32[128]" = torch.ops.aten.mul.Tensor(sum_347, 0.125)
    unsqueeze_1784: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1989, 0);  mul_1989 = None
    unsqueeze_1785: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1784, 2);  unsqueeze_1784 = None
    unsqueeze_1786: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1785, 3);  unsqueeze_1785 = None
    mul_1990: "f32[128]" = torch.ops.aten.mul.Tensor(sum_348, 0.125)
    mul_1991: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_1992: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1990, mul_1991);  mul_1990 = mul_1991 = None
    unsqueeze_1787: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1992, 0);  mul_1992 = None
    unsqueeze_1788: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1787, 2);  unsqueeze_1787 = None
    unsqueeze_1789: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1788, 3);  unsqueeze_1788 = None
    mul_1993: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_147);  primals_147 = None
    unsqueeze_1790: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1993, 0);  mul_1993 = None
    unsqueeze_1791: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1790, 2);  unsqueeze_1790 = None
    unsqueeze_1792: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1791, 3);  unsqueeze_1791 = None
    mul_1994: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_589, unsqueeze_1789);  sub_589 = unsqueeze_1789 = None
    sub_591: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_97, mul_1994);  where_97 = mul_1994 = None
    sub_592: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_591, unsqueeze_1786);  sub_591 = unsqueeze_1786 = None
    mul_1995: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_592, unsqueeze_1792);  sub_592 = unsqueeze_1792 = None
    mul_1996: "f32[128]" = torch.ops.aten.mul.Tensor(sum_348, squeeze_121);  sum_348 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_123 = torch.ops.aten.convolution_backward.default(mul_1995, mean_8, primals_145, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1995 = mean_8 = primals_145 = None
    getitem_649: "f32[8, 256, 1, 1]" = convolution_backward_123[0]
    getitem_650: "f32[128, 256, 1, 1]" = convolution_backward_123[1]
    getitem_651: "f32[128]" = convolution_backward_123[2];  convolution_backward_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_74: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_649, [8, 256, 16, 16]);  getitem_649 = None
    div_58: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_74, 256);  expand_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1793: "f32[8, 1, 256, 16, 16]" = torch.ops.aten.unsqueeze.default(div_58, 1);  div_58 = None
    expand_75: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_1793, [8, 2, 256, 16, 16]);  unsqueeze_1793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_776: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1985, expand_75);  mul_1985 = expand_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_325: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(add_776, [8, 512, 16, 16]);  add_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_488: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_489: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_488);  alias_488 = None
    le_98: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_489, 0);  alias_489 = None
    where_98: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_98, full_default, view_325);  le_98 = view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_349: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_98, [0, 2, 3])
    sub_593: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_1796);  convolution_47 = unsqueeze_1796 = None
    mul_1997: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_98, sub_593)
    sum_350: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1997, [0, 2, 3]);  mul_1997 = None
    mul_1998: "f32[512]" = torch.ops.aten.mul.Tensor(sum_349, 0.00048828125)
    unsqueeze_1797: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1998, 0);  mul_1998 = None
    unsqueeze_1798: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1797, 2);  unsqueeze_1797 = None
    unsqueeze_1799: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1798, 3);  unsqueeze_1798 = None
    mul_1999: "f32[512]" = torch.ops.aten.mul.Tensor(sum_350, 0.00048828125)
    mul_2000: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_2001: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1999, mul_2000);  mul_1999 = mul_2000 = None
    unsqueeze_1800: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2001, 0);  mul_2001 = None
    unsqueeze_1801: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1800, 2);  unsqueeze_1800 = None
    unsqueeze_1802: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1801, 3);  unsqueeze_1801 = None
    mul_2002: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_143);  primals_143 = None
    unsqueeze_1803: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2002, 0);  mul_2002 = None
    unsqueeze_1804: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1803, 2);  unsqueeze_1803 = None
    unsqueeze_1805: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1804, 3);  unsqueeze_1804 = None
    mul_2003: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_593, unsqueeze_1802);  sub_593 = unsqueeze_1802 = None
    sub_595: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_98, mul_2003);  where_98 = mul_2003 = None
    sub_596: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_595, unsqueeze_1799);  sub_595 = unsqueeze_1799 = None
    mul_2004: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_596, unsqueeze_1805);  sub_596 = unsqueeze_1805 = None
    mul_2005: "f32[512]" = torch.ops.aten.mul.Tensor(sum_350, squeeze_118);  sum_350 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_124 = torch.ops.aten.convolution_backward.default(mul_2004, relu_35, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_2004 = primals_142 = None
    getitem_652: "f32[8, 256, 16, 16]" = convolution_backward_124[0]
    getitem_653: "f32[512, 128, 3, 3]" = convolution_backward_124[1];  convolution_backward_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_491: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_492: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_491);  alias_491 = None
    le_99: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_492, 0);  alias_492 = None
    where_99: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_99, full_default, getitem_652);  le_99 = getitem_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_351: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_99, [0, 2, 3])
    sub_597: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_1808);  convolution_46 = unsqueeze_1808 = None
    mul_2006: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_99, sub_597)
    sum_352: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2006, [0, 2, 3]);  mul_2006 = None
    mul_2007: "f32[256]" = torch.ops.aten.mul.Tensor(sum_351, 0.00048828125)
    unsqueeze_1809: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2007, 0);  mul_2007 = None
    unsqueeze_1810: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1809, 2);  unsqueeze_1809 = None
    unsqueeze_1811: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1810, 3);  unsqueeze_1810 = None
    mul_2008: "f32[256]" = torch.ops.aten.mul.Tensor(sum_352, 0.00048828125)
    mul_2009: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_2010: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2008, mul_2009);  mul_2008 = mul_2009 = None
    unsqueeze_1812: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2010, 0);  mul_2010 = None
    unsqueeze_1813: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1812, 2);  unsqueeze_1812 = None
    unsqueeze_1814: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1813, 3);  unsqueeze_1813 = None
    mul_2011: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_140);  primals_140 = None
    unsqueeze_1815: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2011, 0);  mul_2011 = None
    unsqueeze_1816: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1815, 2);  unsqueeze_1815 = None
    unsqueeze_1817: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1816, 3);  unsqueeze_1816 = None
    mul_2012: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_597, unsqueeze_1814);  sub_597 = unsqueeze_1814 = None
    sub_599: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_99, mul_2012);  where_99 = mul_2012 = None
    sub_600: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_599, unsqueeze_1811);  sub_599 = unsqueeze_1811 = None
    mul_2013: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_600, unsqueeze_1817);  sub_600 = unsqueeze_1817 = None
    mul_2014: "f32[256]" = torch.ops.aten.mul.Tensor(sum_352, squeeze_115);  sum_352 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_125 = torch.ops.aten.convolution_backward.default(mul_2013, relu_34, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2013 = primals_139 = None
    getitem_655: "f32[8, 1024, 16, 16]" = convolution_backward_125[0]
    getitem_656: "f32[256, 1024, 1, 1]" = convolution_backward_125[1];  convolution_backward_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_777: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_96, getitem_655);  where_96 = getitem_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_494: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_495: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_494);  alias_494 = None
    le_100: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_495, 0);  alias_495 = None
    where_100: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_100, full_default, add_777);  le_100 = add_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    sum_353: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_100, [0, 2, 3])
    sub_601: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_1820);  convolution_45 = unsqueeze_1820 = None
    mul_2015: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_100, sub_601)
    sum_354: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_2015, [0, 2, 3]);  mul_2015 = None
    mul_2016: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_353, 0.00048828125)
    unsqueeze_1821: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2016, 0);  mul_2016 = None
    unsqueeze_1822: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1821, 2);  unsqueeze_1821 = None
    unsqueeze_1823: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1822, 3);  unsqueeze_1822 = None
    mul_2017: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_354, 0.00048828125)
    mul_2018: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_2019: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_2017, mul_2018);  mul_2017 = mul_2018 = None
    unsqueeze_1824: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2019, 0);  mul_2019 = None
    unsqueeze_1825: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1824, 2);  unsqueeze_1824 = None
    unsqueeze_1826: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1825, 3);  unsqueeze_1825 = None
    mul_2020: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_137);  primals_137 = None
    unsqueeze_1827: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2020, 0);  mul_2020 = None
    unsqueeze_1828: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1827, 2);  unsqueeze_1827 = None
    unsqueeze_1829: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1828, 3);  unsqueeze_1828 = None
    mul_2021: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_601, unsqueeze_1826);  sub_601 = unsqueeze_1826 = None
    sub_603: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_100, mul_2021);  mul_2021 = None
    sub_604: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_603, unsqueeze_1823);  sub_603 = None
    mul_2022: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_604, unsqueeze_1829);  sub_604 = unsqueeze_1829 = None
    mul_2023: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_354, squeeze_112);  sum_354 = squeeze_112 = None
    convolution_backward_126 = torch.ops.aten.convolution_backward.default(mul_2022, avg_pool2d_3, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2022 = avg_pool2d_3 = primals_136 = None
    getitem_658: "f32[8, 512, 16, 16]" = convolution_backward_126[0]
    getitem_659: "f32[1024, 512, 1, 1]" = convolution_backward_126[1];  convolution_backward_126 = None
    avg_pool2d_backward_2: "f32[8, 512, 32, 32]" = torch.ops.aten.avg_pool2d_backward.default(getitem_658, relu_30, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sub_605: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_1832);  convolution_44 = unsqueeze_1832 = None
    mul_2024: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_100, sub_605)
    sum_356: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_2024, [0, 2, 3]);  mul_2024 = None
    mul_2026: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_356, 0.00048828125)
    mul_2027: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_2028: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_2026, mul_2027);  mul_2026 = mul_2027 = None
    unsqueeze_1836: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2028, 0);  mul_2028 = None
    unsqueeze_1837: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1836, 2);  unsqueeze_1836 = None
    unsqueeze_1838: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1837, 3);  unsqueeze_1837 = None
    mul_2029: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_134);  primals_134 = None
    unsqueeze_1839: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2029, 0);  mul_2029 = None
    unsqueeze_1840: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1839, 2);  unsqueeze_1839 = None
    unsqueeze_1841: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1840, 3);  unsqueeze_1840 = None
    mul_2030: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_605, unsqueeze_1838);  sub_605 = unsqueeze_1838 = None
    sub_607: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_100, mul_2030);  where_100 = mul_2030 = None
    sub_608: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_607, unsqueeze_1823);  sub_607 = unsqueeze_1823 = None
    mul_2031: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_608, unsqueeze_1841);  sub_608 = unsqueeze_1841 = None
    mul_2032: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_356, squeeze_109);  sum_356 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_127 = torch.ops.aten.convolution_backward.default(mul_2031, avg_pool2d_2, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2031 = avg_pool2d_2 = primals_133 = None
    getitem_661: "f32[8, 256, 16, 16]" = convolution_backward_127[0]
    getitem_662: "f32[1024, 256, 1, 1]" = convolution_backward_127[1];  convolution_backward_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    avg_pool2d_backward_3: "f32[8, 256, 32, 32]" = torch.ops.aten.avg_pool2d_backward.default(getitem_661, sum_24, [3, 3], [2, 2], [1, 1], False, True, None);  getitem_661 = sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1842: "f32[8, 1, 256, 32, 32]" = torch.ops.aten.unsqueeze.default(avg_pool2d_backward_3, 1);  avg_pool2d_backward_3 = None
    expand_76: "f32[8, 2, 256, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_1842, [8, 2, 256, 32, 32]);  unsqueeze_1842 = None
    mul_2033: "f32[8, 2, 256, 32, 32]" = torch.ops.aten.mul.Tensor(expand_76, view_43);  view_43 = None
    mul_2034: "f32[8, 2, 256, 32, 32]" = torch.ops.aten.mul.Tensor(expand_76, view_47);  expand_76 = view_47 = None
    sum_357: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_2033, [3, 4], True);  mul_2033 = None
    view_326: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(sum_357, [8, 512, 1, 1]);  sum_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_327: "f32[8, 512]" = torch.ops.aten.view.default(view_326, [8, 512]);  view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_328: "f32[8, 2, 1, 256]" = torch.ops.aten.view.default(view_327, [8, 2, 1, 256]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_496: "f32[8, 2, 1, 256]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    mul_2035: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(view_328, alias_496);  view_328 = None
    sum_358: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(mul_2035, [1], True)
    mul_2036: "f32[8, 2, 1, 256]" = torch.ops.aten.mul.Tensor(alias_496, sum_358);  alias_496 = sum_358 = None
    sub_609: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(mul_2035, mul_2036);  mul_2035 = mul_2036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_63: "f32[8, 1, 2, 256]" = torch.ops.aten.permute.default(sub_609, [0, 2, 1, 3]);  sub_609 = None
    view_329: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(permute_63, [8, 512, 1, 1]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_128 = torch.ops.aten.convolution_backward.default(view_329, relu_33, primals_131, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_329 = primals_131 = None
    getitem_664: "f32[8, 128, 1, 1]" = convolution_backward_128[0]
    getitem_665: "f32[512, 128, 1, 1]" = convolution_backward_128[1]
    getitem_666: "f32[512]" = convolution_backward_128[2];  convolution_backward_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_498: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_499: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_498);  alias_498 = None
    le_101: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_499, 0);  alias_499 = None
    where_101: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_101, full_default, getitem_664);  le_101 = getitem_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1843: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_1844: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1843, 2);  unsqueeze_1843 = None
    unsqueeze_1845: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1844, 3);  unsqueeze_1844 = None
    sum_359: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_101, [0, 2, 3])
    sub_610: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_1845);  convolution_42 = unsqueeze_1845 = None
    mul_2037: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(where_101, sub_610)
    sum_360: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_2037, [0, 2, 3]);  mul_2037 = None
    mul_2038: "f32[128]" = torch.ops.aten.mul.Tensor(sum_359, 0.125)
    unsqueeze_1846: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2038, 0);  mul_2038 = None
    unsqueeze_1847: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1846, 2);  unsqueeze_1846 = None
    unsqueeze_1848: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1847, 3);  unsqueeze_1847 = None
    mul_2039: "f32[128]" = torch.ops.aten.mul.Tensor(sum_360, 0.125)
    mul_2040: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_2041: "f32[128]" = torch.ops.aten.mul.Tensor(mul_2039, mul_2040);  mul_2039 = mul_2040 = None
    unsqueeze_1849: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2041, 0);  mul_2041 = None
    unsqueeze_1850: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1849, 2);  unsqueeze_1849 = None
    unsqueeze_1851: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1850, 3);  unsqueeze_1850 = None
    mul_2042: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_129);  primals_129 = None
    unsqueeze_1852: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2042, 0);  mul_2042 = None
    unsqueeze_1853: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1852, 2);  unsqueeze_1852 = None
    unsqueeze_1854: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1853, 3);  unsqueeze_1853 = None
    mul_2043: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_610, unsqueeze_1851);  sub_610 = unsqueeze_1851 = None
    sub_612: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(where_101, mul_2043);  where_101 = mul_2043 = None
    sub_613: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(sub_612, unsqueeze_1848);  sub_612 = unsqueeze_1848 = None
    mul_2044: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_613, unsqueeze_1854);  sub_613 = unsqueeze_1854 = None
    mul_2045: "f32[128]" = torch.ops.aten.mul.Tensor(sum_360, squeeze_106);  sum_360 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_129 = torch.ops.aten.convolution_backward.default(mul_2044, mean_7, primals_127, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_2044 = mean_7 = primals_127 = None
    getitem_667: "f32[8, 256, 1, 1]" = convolution_backward_129[0]
    getitem_668: "f32[128, 256, 1, 1]" = convolution_backward_129[1]
    getitem_669: "f32[128]" = convolution_backward_129[2];  convolution_backward_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_77: "f32[8, 256, 32, 32]" = torch.ops.aten.expand.default(getitem_667, [8, 256, 32, 32]);  getitem_667 = None
    div_59: "f32[8, 256, 32, 32]" = torch.ops.aten.div.Scalar(expand_77, 1024);  expand_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1855: "f32[8, 1, 256, 32, 32]" = torch.ops.aten.unsqueeze.default(div_59, 1);  div_59 = None
    expand_78: "f32[8, 2, 256, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_1855, [8, 2, 256, 32, 32]);  unsqueeze_1855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_778: "f32[8, 2, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_2034, expand_78);  mul_2034 = expand_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_330: "f32[8, 512, 32, 32]" = torch.ops.aten.view.default(add_778, [8, 512, 32, 32]);  add_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_501: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_502: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(alias_501);  alias_501 = None
    le_102: "b8[8, 512, 32, 32]" = torch.ops.aten.le.Scalar(alias_502, 0);  alias_502 = None
    where_102: "f32[8, 512, 32, 32]" = torch.ops.aten.where.self(le_102, full_default, view_330);  le_102 = view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_361: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_102, [0, 2, 3])
    sub_614: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1858);  convolution_41 = unsqueeze_1858 = None
    mul_2046: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(where_102, sub_614)
    sum_362: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_2046, [0, 2, 3]);  mul_2046 = None
    mul_2047: "f32[512]" = torch.ops.aten.mul.Tensor(sum_361, 0.0001220703125)
    unsqueeze_1859: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2047, 0);  mul_2047 = None
    unsqueeze_1860: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1859, 2);  unsqueeze_1859 = None
    unsqueeze_1861: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1860, 3);  unsqueeze_1860 = None
    mul_2048: "f32[512]" = torch.ops.aten.mul.Tensor(sum_362, 0.0001220703125)
    mul_2049: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_2050: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2048, mul_2049);  mul_2048 = mul_2049 = None
    unsqueeze_1862: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2050, 0);  mul_2050 = None
    unsqueeze_1863: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1862, 2);  unsqueeze_1862 = None
    unsqueeze_1864: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1863, 3);  unsqueeze_1863 = None
    mul_2051: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_125);  primals_125 = None
    unsqueeze_1865: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2051, 0);  mul_2051 = None
    unsqueeze_1866: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1865, 2);  unsqueeze_1865 = None
    unsqueeze_1867: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1866, 3);  unsqueeze_1866 = None
    mul_2052: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_614, unsqueeze_1864);  sub_614 = unsqueeze_1864 = None
    sub_616: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(where_102, mul_2052);  where_102 = mul_2052 = None
    sub_617: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_616, unsqueeze_1861);  sub_616 = unsqueeze_1861 = None
    mul_2053: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_617, unsqueeze_1867);  sub_617 = unsqueeze_1867 = None
    mul_2054: "f32[512]" = torch.ops.aten.mul.Tensor(sum_362, squeeze_103);  sum_362 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_130 = torch.ops.aten.convolution_backward.default(mul_2053, relu_31, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_2053 = primals_124 = None
    getitem_670: "f32[8, 256, 32, 32]" = convolution_backward_130[0]
    getitem_671: "f32[512, 128, 3, 3]" = convolution_backward_130[1];  convolution_backward_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_504: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_505: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(alias_504);  alias_504 = None
    le_103: "b8[8, 256, 32, 32]" = torch.ops.aten.le.Scalar(alias_505, 0);  alias_505 = None
    where_103: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(le_103, full_default, getitem_670);  le_103 = getitem_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_363: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_103, [0, 2, 3])
    sub_618: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_1870);  convolution_40 = unsqueeze_1870 = None
    mul_2055: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(where_103, sub_618)
    sum_364: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2055, [0, 2, 3]);  mul_2055 = None
    mul_2056: "f32[256]" = torch.ops.aten.mul.Tensor(sum_363, 0.0001220703125)
    unsqueeze_1871: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2056, 0);  mul_2056 = None
    unsqueeze_1872: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1871, 2);  unsqueeze_1871 = None
    unsqueeze_1873: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1872, 3);  unsqueeze_1872 = None
    mul_2057: "f32[256]" = torch.ops.aten.mul.Tensor(sum_364, 0.0001220703125)
    mul_2058: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_2059: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2057, mul_2058);  mul_2057 = mul_2058 = None
    unsqueeze_1874: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2059, 0);  mul_2059 = None
    unsqueeze_1875: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1874, 2);  unsqueeze_1874 = None
    unsqueeze_1876: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1875, 3);  unsqueeze_1875 = None
    mul_2060: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_122);  primals_122 = None
    unsqueeze_1877: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2060, 0);  mul_2060 = None
    unsqueeze_1878: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1877, 2);  unsqueeze_1877 = None
    unsqueeze_1879: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1878, 3);  unsqueeze_1878 = None
    mul_2061: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_618, unsqueeze_1876);  sub_618 = unsqueeze_1876 = None
    sub_620: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(where_103, mul_2061);  where_103 = mul_2061 = None
    sub_621: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_620, unsqueeze_1873);  sub_620 = unsqueeze_1873 = None
    mul_2062: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_621, unsqueeze_1879);  sub_621 = unsqueeze_1879 = None
    mul_2063: "f32[256]" = torch.ops.aten.mul.Tensor(sum_364, squeeze_100);  sum_364 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_131 = torch.ops.aten.convolution_backward.default(mul_2062, relu_30, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2062 = primals_121 = None
    getitem_673: "f32[8, 512, 32, 32]" = convolution_backward_131[0]
    getitem_674: "f32[256, 512, 1, 1]" = convolution_backward_131[1];  convolution_backward_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_779: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_2, getitem_673);  avg_pool2d_backward_2 = getitem_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_507: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_508: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(alias_507);  alias_507 = None
    le_104: "b8[8, 512, 32, 32]" = torch.ops.aten.le.Scalar(alias_508, 0);  alias_508 = None
    where_104: "f32[8, 512, 32, 32]" = torch.ops.aten.where.self(le_104, full_default, add_779);  le_104 = add_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_365: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_104, [0, 2, 3])
    sub_622: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_1882);  convolution_39 = unsqueeze_1882 = None
    mul_2064: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(where_104, sub_622)
    sum_366: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_2064, [0, 2, 3]);  mul_2064 = None
    mul_2065: "f32[512]" = torch.ops.aten.mul.Tensor(sum_365, 0.0001220703125)
    unsqueeze_1883: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2065, 0);  mul_2065 = None
    unsqueeze_1884: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1883, 2);  unsqueeze_1883 = None
    unsqueeze_1885: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1884, 3);  unsqueeze_1884 = None
    mul_2066: "f32[512]" = torch.ops.aten.mul.Tensor(sum_366, 0.0001220703125)
    mul_2067: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_2068: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2066, mul_2067);  mul_2066 = mul_2067 = None
    unsqueeze_1886: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2068, 0);  mul_2068 = None
    unsqueeze_1887: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1886, 2);  unsqueeze_1886 = None
    unsqueeze_1888: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1887, 3);  unsqueeze_1887 = None
    mul_2069: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_119);  primals_119 = None
    unsqueeze_1889: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2069, 0);  mul_2069 = None
    unsqueeze_1890: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1889, 2);  unsqueeze_1889 = None
    unsqueeze_1891: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1890, 3);  unsqueeze_1890 = None
    mul_2070: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_622, unsqueeze_1888);  sub_622 = unsqueeze_1888 = None
    sub_624: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(where_104, mul_2070);  mul_2070 = None
    sub_625: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_624, unsqueeze_1885);  sub_624 = unsqueeze_1885 = None
    mul_2071: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_625, unsqueeze_1891);  sub_625 = unsqueeze_1891 = None
    mul_2072: "f32[512]" = torch.ops.aten.mul.Tensor(sum_366, squeeze_97);  sum_366 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_132 = torch.ops.aten.convolution_backward.default(mul_2071, sum_21, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2071 = sum_21 = primals_118 = None
    getitem_676: "f32[8, 128, 32, 32]" = convolution_backward_132[0]
    getitem_677: "f32[512, 128, 1, 1]" = convolution_backward_132[1];  convolution_backward_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1892: "f32[8, 1, 128, 32, 32]" = torch.ops.aten.unsqueeze.default(getitem_676, 1);  getitem_676 = None
    expand_79: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_1892, [8, 2, 128, 32, 32]);  unsqueeze_1892 = None
    mul_2073: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.mul.Tensor(expand_79, view_37);  view_37 = None
    mul_2074: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.mul.Tensor(expand_79, view_41);  expand_79 = view_41 = None
    sum_367: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_2073, [3, 4], True);  mul_2073 = None
    view_331: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(sum_367, [8, 256, 1, 1]);  sum_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_332: "f32[8, 256]" = torch.ops.aten.view.default(view_331, [8, 256]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_333: "f32[8, 2, 1, 128]" = torch.ops.aten.view.default(view_332, [8, 2, 1, 128]);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_509: "f32[8, 2, 1, 128]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    mul_2075: "f32[8, 2, 1, 128]" = torch.ops.aten.mul.Tensor(view_333, alias_509);  view_333 = None
    sum_368: "f32[8, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_2075, [1], True)
    mul_2076: "f32[8, 2, 1, 128]" = torch.ops.aten.mul.Tensor(alias_509, sum_368);  alias_509 = sum_368 = None
    sub_626: "f32[8, 2, 1, 128]" = torch.ops.aten.sub.Tensor(mul_2075, mul_2076);  mul_2075 = mul_2076 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_64: "f32[8, 1, 2, 128]" = torch.ops.aten.permute.default(sub_626, [0, 2, 1, 3]);  sub_626 = None
    view_334: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(permute_64, [8, 256, 1, 1]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_133 = torch.ops.aten.convolution_backward.default(view_334, relu_29, primals_116, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_334 = primals_116 = None
    getitem_679: "f32[8, 64, 1, 1]" = convolution_backward_133[0]
    getitem_680: "f32[256, 64, 1, 1]" = convolution_backward_133[1]
    getitem_681: "f32[256]" = convolution_backward_133[2];  convolution_backward_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_511: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_512: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(alias_511);  alias_511 = None
    le_105: "b8[8, 64, 1, 1]" = torch.ops.aten.le.Scalar(alias_512, 0);  alias_512 = None
    where_105: "f32[8, 64, 1, 1]" = torch.ops.aten.where.self(le_105, full_default, getitem_679);  le_105 = getitem_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1893: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_1894: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1893, 2);  unsqueeze_1893 = None
    unsqueeze_1895: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1894, 3);  unsqueeze_1894 = None
    sum_369: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_105, [0, 2, 3])
    sub_627: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1895);  convolution_37 = unsqueeze_1895 = None
    mul_2077: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(where_105, sub_627)
    sum_370: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_2077, [0, 2, 3]);  mul_2077 = None
    mul_2078: "f32[64]" = torch.ops.aten.mul.Tensor(sum_369, 0.125)
    unsqueeze_1896: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2078, 0);  mul_2078 = None
    unsqueeze_1897: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1896, 2);  unsqueeze_1896 = None
    unsqueeze_1898: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1897, 3);  unsqueeze_1897 = None
    mul_2079: "f32[64]" = torch.ops.aten.mul.Tensor(sum_370, 0.125)
    mul_2080: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_2081: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2079, mul_2080);  mul_2079 = mul_2080 = None
    unsqueeze_1899: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2081, 0);  mul_2081 = None
    unsqueeze_1900: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1899, 2);  unsqueeze_1899 = None
    unsqueeze_1901: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1900, 3);  unsqueeze_1900 = None
    mul_2082: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_114);  primals_114 = None
    unsqueeze_1902: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2082, 0);  mul_2082 = None
    unsqueeze_1903: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1902, 2);  unsqueeze_1902 = None
    unsqueeze_1904: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1903, 3);  unsqueeze_1903 = None
    mul_2083: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_627, unsqueeze_1901);  sub_627 = unsqueeze_1901 = None
    sub_629: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(where_105, mul_2083);  where_105 = mul_2083 = None
    sub_630: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(sub_629, unsqueeze_1898);  sub_629 = unsqueeze_1898 = None
    mul_2084: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_630, unsqueeze_1904);  sub_630 = unsqueeze_1904 = None
    mul_2085: "f32[64]" = torch.ops.aten.mul.Tensor(sum_370, squeeze_94);  sum_370 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_134 = torch.ops.aten.convolution_backward.default(mul_2084, mean_6, primals_112, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_2084 = mean_6 = primals_112 = None
    getitem_682: "f32[8, 128, 1, 1]" = convolution_backward_134[0]
    getitem_683: "f32[64, 128, 1, 1]" = convolution_backward_134[1]
    getitem_684: "f32[64]" = convolution_backward_134[2];  convolution_backward_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_80: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(getitem_682, [8, 128, 32, 32]);  getitem_682 = None
    div_60: "f32[8, 128, 32, 32]" = torch.ops.aten.div.Scalar(expand_80, 1024);  expand_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1905: "f32[8, 1, 128, 32, 32]" = torch.ops.aten.unsqueeze.default(div_60, 1);  div_60 = None
    expand_81: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_1905, [8, 2, 128, 32, 32]);  unsqueeze_1905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_780: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_2074, expand_81);  mul_2074 = expand_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_335: "f32[8, 256, 32, 32]" = torch.ops.aten.view.default(add_780, [8, 256, 32, 32]);  add_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_514: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_515: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(alias_514);  alias_514 = None
    le_106: "b8[8, 256, 32, 32]" = torch.ops.aten.le.Scalar(alias_515, 0);  alias_515 = None
    where_106: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(le_106, full_default, view_335);  le_106 = view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_371: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_106, [0, 2, 3])
    sub_631: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1908);  convolution_36 = unsqueeze_1908 = None
    mul_2086: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(where_106, sub_631)
    sum_372: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2086, [0, 2, 3]);  mul_2086 = None
    mul_2087: "f32[256]" = torch.ops.aten.mul.Tensor(sum_371, 0.0001220703125)
    unsqueeze_1909: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2087, 0);  mul_2087 = None
    unsqueeze_1910: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1909, 2);  unsqueeze_1909 = None
    unsqueeze_1911: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1910, 3);  unsqueeze_1910 = None
    mul_2088: "f32[256]" = torch.ops.aten.mul.Tensor(sum_372, 0.0001220703125)
    mul_2089: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_2090: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2088, mul_2089);  mul_2088 = mul_2089 = None
    unsqueeze_1912: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2090, 0);  mul_2090 = None
    unsqueeze_1913: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1912, 2);  unsqueeze_1912 = None
    unsqueeze_1914: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1913, 3);  unsqueeze_1913 = None
    mul_2091: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_110);  primals_110 = None
    unsqueeze_1915: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2091, 0);  mul_2091 = None
    unsqueeze_1916: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1915, 2);  unsqueeze_1915 = None
    unsqueeze_1917: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1916, 3);  unsqueeze_1916 = None
    mul_2092: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_631, unsqueeze_1914);  sub_631 = unsqueeze_1914 = None
    sub_633: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(where_106, mul_2092);  where_106 = mul_2092 = None
    sub_634: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_633, unsqueeze_1911);  sub_633 = unsqueeze_1911 = None
    mul_2093: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_634, unsqueeze_1917);  sub_634 = unsqueeze_1917 = None
    mul_2094: "f32[256]" = torch.ops.aten.mul.Tensor(sum_372, squeeze_91);  sum_372 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_135 = torch.ops.aten.convolution_backward.default(mul_2093, relu_27, primals_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_2093 = primals_109 = None
    getitem_685: "f32[8, 128, 32, 32]" = convolution_backward_135[0]
    getitem_686: "f32[256, 64, 3, 3]" = convolution_backward_135[1];  convolution_backward_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_517: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_518: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_517);  alias_517 = None
    le_107: "b8[8, 128, 32, 32]" = torch.ops.aten.le.Scalar(alias_518, 0);  alias_518 = None
    where_107: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(le_107, full_default, getitem_685);  le_107 = getitem_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_373: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_107, [0, 2, 3])
    sub_635: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_1920);  convolution_35 = unsqueeze_1920 = None
    mul_2095: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_107, sub_635)
    sum_374: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_2095, [0, 2, 3]);  mul_2095 = None
    mul_2096: "f32[128]" = torch.ops.aten.mul.Tensor(sum_373, 0.0001220703125)
    unsqueeze_1921: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2096, 0);  mul_2096 = None
    unsqueeze_1922: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1921, 2);  unsqueeze_1921 = None
    unsqueeze_1923: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1922, 3);  unsqueeze_1922 = None
    mul_2097: "f32[128]" = torch.ops.aten.mul.Tensor(sum_374, 0.0001220703125)
    mul_2098: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_2099: "f32[128]" = torch.ops.aten.mul.Tensor(mul_2097, mul_2098);  mul_2097 = mul_2098 = None
    unsqueeze_1924: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2099, 0);  mul_2099 = None
    unsqueeze_1925: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1924, 2);  unsqueeze_1924 = None
    unsqueeze_1926: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1925, 3);  unsqueeze_1925 = None
    mul_2100: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_107);  primals_107 = None
    unsqueeze_1927: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2100, 0);  mul_2100 = None
    unsqueeze_1928: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1927, 2);  unsqueeze_1927 = None
    unsqueeze_1929: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1928, 3);  unsqueeze_1928 = None
    mul_2101: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_635, unsqueeze_1926);  sub_635 = unsqueeze_1926 = None
    sub_637: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_107, mul_2101);  where_107 = mul_2101 = None
    sub_638: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_637, unsqueeze_1923);  sub_637 = unsqueeze_1923 = None
    mul_2102: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_638, unsqueeze_1929);  sub_638 = unsqueeze_1929 = None
    mul_2103: "f32[128]" = torch.ops.aten.mul.Tensor(sum_374, squeeze_88);  sum_374 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_136 = torch.ops.aten.convolution_backward.default(mul_2102, relu_26, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2102 = primals_106 = None
    getitem_688: "f32[8, 512, 32, 32]" = convolution_backward_136[0]
    getitem_689: "f32[128, 512, 1, 1]" = convolution_backward_136[1];  convolution_backward_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_781: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(where_104, getitem_688);  where_104 = getitem_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_520: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_521: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(alias_520);  alias_520 = None
    le_108: "b8[8, 512, 32, 32]" = torch.ops.aten.le.Scalar(alias_521, 0);  alias_521 = None
    where_108: "f32[8, 512, 32, 32]" = torch.ops.aten.where.self(le_108, full_default, add_781);  le_108 = add_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_375: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_108, [0, 2, 3])
    sub_639: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_1932);  convolution_34 = unsqueeze_1932 = None
    mul_2104: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(where_108, sub_639)
    sum_376: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_2104, [0, 2, 3]);  mul_2104 = None
    mul_2105: "f32[512]" = torch.ops.aten.mul.Tensor(sum_375, 0.0001220703125)
    unsqueeze_1933: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2105, 0);  mul_2105 = None
    unsqueeze_1934: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1933, 2);  unsqueeze_1933 = None
    unsqueeze_1935: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1934, 3);  unsqueeze_1934 = None
    mul_2106: "f32[512]" = torch.ops.aten.mul.Tensor(sum_376, 0.0001220703125)
    mul_2107: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_2108: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2106, mul_2107);  mul_2106 = mul_2107 = None
    unsqueeze_1936: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2108, 0);  mul_2108 = None
    unsqueeze_1937: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1936, 2);  unsqueeze_1936 = None
    unsqueeze_1938: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1937, 3);  unsqueeze_1937 = None
    mul_2109: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_104);  primals_104 = None
    unsqueeze_1939: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2109, 0);  mul_2109 = None
    unsqueeze_1940: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1939, 2);  unsqueeze_1939 = None
    unsqueeze_1941: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1940, 3);  unsqueeze_1940 = None
    mul_2110: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_639, unsqueeze_1938);  sub_639 = unsqueeze_1938 = None
    sub_641: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(where_108, mul_2110);  mul_2110 = None
    sub_642: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_641, unsqueeze_1935);  sub_641 = unsqueeze_1935 = None
    mul_2111: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_642, unsqueeze_1941);  sub_642 = unsqueeze_1941 = None
    mul_2112: "f32[512]" = torch.ops.aten.mul.Tensor(sum_376, squeeze_85);  sum_376 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_137 = torch.ops.aten.convolution_backward.default(mul_2111, sum_18, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2111 = sum_18 = primals_103 = None
    getitem_691: "f32[8, 128, 32, 32]" = convolution_backward_137[0]
    getitem_692: "f32[512, 128, 1, 1]" = convolution_backward_137[1];  convolution_backward_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1942: "f32[8, 1, 128, 32, 32]" = torch.ops.aten.unsqueeze.default(getitem_691, 1);  getitem_691 = None
    expand_82: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_1942, [8, 2, 128, 32, 32]);  unsqueeze_1942 = None
    mul_2113: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.mul.Tensor(expand_82, view_31);  view_31 = None
    mul_2114: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.mul.Tensor(expand_82, view_35);  expand_82 = view_35 = None
    sum_377: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_2113, [3, 4], True);  mul_2113 = None
    view_336: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(sum_377, [8, 256, 1, 1]);  sum_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_337: "f32[8, 256]" = torch.ops.aten.view.default(view_336, [8, 256]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_338: "f32[8, 2, 1, 128]" = torch.ops.aten.view.default(view_337, [8, 2, 1, 128]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_522: "f32[8, 2, 1, 128]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    mul_2115: "f32[8, 2, 1, 128]" = torch.ops.aten.mul.Tensor(view_338, alias_522);  view_338 = None
    sum_378: "f32[8, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_2115, [1], True)
    mul_2116: "f32[8, 2, 1, 128]" = torch.ops.aten.mul.Tensor(alias_522, sum_378);  alias_522 = sum_378 = None
    sub_643: "f32[8, 2, 1, 128]" = torch.ops.aten.sub.Tensor(mul_2115, mul_2116);  mul_2115 = mul_2116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_65: "f32[8, 1, 2, 128]" = torch.ops.aten.permute.default(sub_643, [0, 2, 1, 3]);  sub_643 = None
    view_339: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(permute_65, [8, 256, 1, 1]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_138 = torch.ops.aten.convolution_backward.default(view_339, relu_25, primals_101, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_339 = primals_101 = None
    getitem_694: "f32[8, 64, 1, 1]" = convolution_backward_138[0]
    getitem_695: "f32[256, 64, 1, 1]" = convolution_backward_138[1]
    getitem_696: "f32[256]" = convolution_backward_138[2];  convolution_backward_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_524: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_525: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(alias_524);  alias_524 = None
    le_109: "b8[8, 64, 1, 1]" = torch.ops.aten.le.Scalar(alias_525, 0);  alias_525 = None
    where_109: "f32[8, 64, 1, 1]" = torch.ops.aten.where.self(le_109, full_default, getitem_694);  le_109 = getitem_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1943: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_1944: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1943, 2);  unsqueeze_1943 = None
    unsqueeze_1945: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1944, 3);  unsqueeze_1944 = None
    sum_379: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_109, [0, 2, 3])
    sub_644: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1945);  convolution_32 = unsqueeze_1945 = None
    mul_2117: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(where_109, sub_644)
    sum_380: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_2117, [0, 2, 3]);  mul_2117 = None
    mul_2118: "f32[64]" = torch.ops.aten.mul.Tensor(sum_379, 0.125)
    unsqueeze_1946: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2118, 0);  mul_2118 = None
    unsqueeze_1947: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1946, 2);  unsqueeze_1946 = None
    unsqueeze_1948: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1947, 3);  unsqueeze_1947 = None
    mul_2119: "f32[64]" = torch.ops.aten.mul.Tensor(sum_380, 0.125)
    mul_2120: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_2121: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2119, mul_2120);  mul_2119 = mul_2120 = None
    unsqueeze_1949: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2121, 0);  mul_2121 = None
    unsqueeze_1950: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1949, 2);  unsqueeze_1949 = None
    unsqueeze_1951: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1950, 3);  unsqueeze_1950 = None
    mul_2122: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_99);  primals_99 = None
    unsqueeze_1952: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2122, 0);  mul_2122 = None
    unsqueeze_1953: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1952, 2);  unsqueeze_1952 = None
    unsqueeze_1954: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1953, 3);  unsqueeze_1953 = None
    mul_2123: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_644, unsqueeze_1951);  sub_644 = unsqueeze_1951 = None
    sub_646: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(where_109, mul_2123);  where_109 = mul_2123 = None
    sub_647: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(sub_646, unsqueeze_1948);  sub_646 = unsqueeze_1948 = None
    mul_2124: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_647, unsqueeze_1954);  sub_647 = unsqueeze_1954 = None
    mul_2125: "f32[64]" = torch.ops.aten.mul.Tensor(sum_380, squeeze_82);  sum_380 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_139 = torch.ops.aten.convolution_backward.default(mul_2124, mean_5, primals_97, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_2124 = mean_5 = primals_97 = None
    getitem_697: "f32[8, 128, 1, 1]" = convolution_backward_139[0]
    getitem_698: "f32[64, 128, 1, 1]" = convolution_backward_139[1]
    getitem_699: "f32[64]" = convolution_backward_139[2];  convolution_backward_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_83: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(getitem_697, [8, 128, 32, 32]);  getitem_697 = None
    div_61: "f32[8, 128, 32, 32]" = torch.ops.aten.div.Scalar(expand_83, 1024);  expand_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_1955: "f32[8, 1, 128, 32, 32]" = torch.ops.aten.unsqueeze.default(div_61, 1);  div_61 = None
    expand_84: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_1955, [8, 2, 128, 32, 32]);  unsqueeze_1955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_782: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_2114, expand_84);  mul_2114 = expand_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_340: "f32[8, 256, 32, 32]" = torch.ops.aten.view.default(add_782, [8, 256, 32, 32]);  add_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_527: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_528: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(alias_527);  alias_527 = None
    le_110: "b8[8, 256, 32, 32]" = torch.ops.aten.le.Scalar(alias_528, 0);  alias_528 = None
    where_110: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(le_110, full_default, view_340);  le_110 = view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_381: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_110, [0, 2, 3])
    sub_648: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1958);  convolution_31 = unsqueeze_1958 = None
    mul_2126: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(where_110, sub_648)
    sum_382: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2126, [0, 2, 3]);  mul_2126 = None
    mul_2127: "f32[256]" = torch.ops.aten.mul.Tensor(sum_381, 0.0001220703125)
    unsqueeze_1959: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2127, 0);  mul_2127 = None
    unsqueeze_1960: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1959, 2);  unsqueeze_1959 = None
    unsqueeze_1961: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1960, 3);  unsqueeze_1960 = None
    mul_2128: "f32[256]" = torch.ops.aten.mul.Tensor(sum_382, 0.0001220703125)
    mul_2129: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_2130: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2128, mul_2129);  mul_2128 = mul_2129 = None
    unsqueeze_1962: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2130, 0);  mul_2130 = None
    unsqueeze_1963: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1962, 2);  unsqueeze_1962 = None
    unsqueeze_1964: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1963, 3);  unsqueeze_1963 = None
    mul_2131: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_95);  primals_95 = None
    unsqueeze_1965: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2131, 0);  mul_2131 = None
    unsqueeze_1966: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1965, 2);  unsqueeze_1965 = None
    unsqueeze_1967: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1966, 3);  unsqueeze_1966 = None
    mul_2132: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_648, unsqueeze_1964);  sub_648 = unsqueeze_1964 = None
    sub_650: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(where_110, mul_2132);  where_110 = mul_2132 = None
    sub_651: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_650, unsqueeze_1961);  sub_650 = unsqueeze_1961 = None
    mul_2133: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_651, unsqueeze_1967);  sub_651 = unsqueeze_1967 = None
    mul_2134: "f32[256]" = torch.ops.aten.mul.Tensor(sum_382, squeeze_79);  sum_382 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_140 = torch.ops.aten.convolution_backward.default(mul_2133, relu_23, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_2133 = primals_94 = None
    getitem_700: "f32[8, 128, 32, 32]" = convolution_backward_140[0]
    getitem_701: "f32[256, 64, 3, 3]" = convolution_backward_140[1];  convolution_backward_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_530: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_531: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_530);  alias_530 = None
    le_111: "b8[8, 128, 32, 32]" = torch.ops.aten.le.Scalar(alias_531, 0);  alias_531 = None
    where_111: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(le_111, full_default, getitem_700);  le_111 = getitem_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_383: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_111, [0, 2, 3])
    sub_652: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_1970);  convolution_30 = unsqueeze_1970 = None
    mul_2135: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_111, sub_652)
    sum_384: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_2135, [0, 2, 3]);  mul_2135 = None
    mul_2136: "f32[128]" = torch.ops.aten.mul.Tensor(sum_383, 0.0001220703125)
    unsqueeze_1971: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2136, 0);  mul_2136 = None
    unsqueeze_1972: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1971, 2);  unsqueeze_1971 = None
    unsqueeze_1973: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1972, 3);  unsqueeze_1972 = None
    mul_2137: "f32[128]" = torch.ops.aten.mul.Tensor(sum_384, 0.0001220703125)
    mul_2138: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_2139: "f32[128]" = torch.ops.aten.mul.Tensor(mul_2137, mul_2138);  mul_2137 = mul_2138 = None
    unsqueeze_1974: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2139, 0);  mul_2139 = None
    unsqueeze_1975: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1974, 2);  unsqueeze_1974 = None
    unsqueeze_1976: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1975, 3);  unsqueeze_1975 = None
    mul_2140: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_92);  primals_92 = None
    unsqueeze_1977: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2140, 0);  mul_2140 = None
    unsqueeze_1978: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1977, 2);  unsqueeze_1977 = None
    unsqueeze_1979: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1978, 3);  unsqueeze_1978 = None
    mul_2141: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_652, unsqueeze_1976);  sub_652 = unsqueeze_1976 = None
    sub_654: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_111, mul_2141);  where_111 = mul_2141 = None
    sub_655: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_654, unsqueeze_1973);  sub_654 = unsqueeze_1973 = None
    mul_2142: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_655, unsqueeze_1979);  sub_655 = unsqueeze_1979 = None
    mul_2143: "f32[128]" = torch.ops.aten.mul.Tensor(sum_384, squeeze_76);  sum_384 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_141 = torch.ops.aten.convolution_backward.default(mul_2142, relu_22, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2142 = primals_91 = None
    getitem_703: "f32[8, 512, 32, 32]" = convolution_backward_141[0]
    getitem_704: "f32[128, 512, 1, 1]" = convolution_backward_141[1];  convolution_backward_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_783: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(where_108, getitem_703);  where_108 = getitem_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_533: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_534: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(alias_533);  alias_533 = None
    le_112: "b8[8, 512, 32, 32]" = torch.ops.aten.le.Scalar(alias_534, 0);  alias_534 = None
    where_112: "f32[8, 512, 32, 32]" = torch.ops.aten.where.self(le_112, full_default, add_783);  le_112 = add_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_385: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_112, [0, 2, 3])
    sub_656: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1982);  convolution_29 = unsqueeze_1982 = None
    mul_2144: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(where_112, sub_656)
    sum_386: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_2144, [0, 2, 3]);  mul_2144 = None
    mul_2145: "f32[512]" = torch.ops.aten.mul.Tensor(sum_385, 0.0001220703125)
    unsqueeze_1983: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2145, 0);  mul_2145 = None
    unsqueeze_1984: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1983, 2);  unsqueeze_1983 = None
    unsqueeze_1985: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1984, 3);  unsqueeze_1984 = None
    mul_2146: "f32[512]" = torch.ops.aten.mul.Tensor(sum_386, 0.0001220703125)
    mul_2147: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_2148: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2146, mul_2147);  mul_2146 = mul_2147 = None
    unsqueeze_1986: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2148, 0);  mul_2148 = None
    unsqueeze_1987: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1986, 2);  unsqueeze_1986 = None
    unsqueeze_1988: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1987, 3);  unsqueeze_1987 = None
    mul_2149: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_89);  primals_89 = None
    unsqueeze_1989: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2149, 0);  mul_2149 = None
    unsqueeze_1990: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1989, 2);  unsqueeze_1989 = None
    unsqueeze_1991: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1990, 3);  unsqueeze_1990 = None
    mul_2150: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_656, unsqueeze_1988);  sub_656 = unsqueeze_1988 = None
    sub_658: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(where_112, mul_2150);  mul_2150 = None
    sub_659: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_658, unsqueeze_1985);  sub_658 = unsqueeze_1985 = None
    mul_2151: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_659, unsqueeze_1991);  sub_659 = unsqueeze_1991 = None
    mul_2152: "f32[512]" = torch.ops.aten.mul.Tensor(sum_386, squeeze_73);  sum_386 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_142 = torch.ops.aten.convolution_backward.default(mul_2151, sum_15, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2151 = sum_15 = primals_88 = None
    getitem_706: "f32[8, 128, 32, 32]" = convolution_backward_142[0]
    getitem_707: "f32[512, 128, 1, 1]" = convolution_backward_142[1];  convolution_backward_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_1992: "f32[8, 1, 128, 32, 32]" = torch.ops.aten.unsqueeze.default(getitem_706, 1);  getitem_706 = None
    expand_85: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_1992, [8, 2, 128, 32, 32]);  unsqueeze_1992 = None
    mul_2153: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.mul.Tensor(expand_85, view_25);  view_25 = None
    mul_2154: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.mul.Tensor(expand_85, view_29);  expand_85 = view_29 = None
    sum_387: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_2153, [3, 4], True);  mul_2153 = None
    view_341: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(sum_387, [8, 256, 1, 1]);  sum_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_342: "f32[8, 256]" = torch.ops.aten.view.default(view_341, [8, 256]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_343: "f32[8, 2, 1, 128]" = torch.ops.aten.view.default(view_342, [8, 2, 1, 128]);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_535: "f32[8, 2, 1, 128]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    mul_2155: "f32[8, 2, 1, 128]" = torch.ops.aten.mul.Tensor(view_343, alias_535);  view_343 = None
    sum_388: "f32[8, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_2155, [1], True)
    mul_2156: "f32[8, 2, 1, 128]" = torch.ops.aten.mul.Tensor(alias_535, sum_388);  alias_535 = sum_388 = None
    sub_660: "f32[8, 2, 1, 128]" = torch.ops.aten.sub.Tensor(mul_2155, mul_2156);  mul_2155 = mul_2156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_66: "f32[8, 1, 2, 128]" = torch.ops.aten.permute.default(sub_660, [0, 2, 1, 3]);  sub_660 = None
    view_344: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(permute_66, [8, 256, 1, 1]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_143 = torch.ops.aten.convolution_backward.default(view_344, relu_21, primals_86, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_344 = primals_86 = None
    getitem_709: "f32[8, 64, 1, 1]" = convolution_backward_143[0]
    getitem_710: "f32[256, 64, 1, 1]" = convolution_backward_143[1]
    getitem_711: "f32[256]" = convolution_backward_143[2];  convolution_backward_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_537: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_538: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(alias_537);  alias_537 = None
    le_113: "b8[8, 64, 1, 1]" = torch.ops.aten.le.Scalar(alias_538, 0);  alias_538 = None
    where_113: "f32[8, 64, 1, 1]" = torch.ops.aten.where.self(le_113, full_default, getitem_709);  le_113 = getitem_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_1993: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_1994: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1993, 2);  unsqueeze_1993 = None
    unsqueeze_1995: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1994, 3);  unsqueeze_1994 = None
    sum_389: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_113, [0, 2, 3])
    sub_661: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1995);  convolution_27 = unsqueeze_1995 = None
    mul_2157: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(where_113, sub_661)
    sum_390: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_2157, [0, 2, 3]);  mul_2157 = None
    mul_2158: "f32[64]" = torch.ops.aten.mul.Tensor(sum_389, 0.125)
    unsqueeze_1996: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2158, 0);  mul_2158 = None
    unsqueeze_1997: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1996, 2);  unsqueeze_1996 = None
    unsqueeze_1998: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1997, 3);  unsqueeze_1997 = None
    mul_2159: "f32[64]" = torch.ops.aten.mul.Tensor(sum_390, 0.125)
    mul_2160: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_2161: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2159, mul_2160);  mul_2159 = mul_2160 = None
    unsqueeze_1999: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2161, 0);  mul_2161 = None
    unsqueeze_2000: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1999, 2);  unsqueeze_1999 = None
    unsqueeze_2001: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2000, 3);  unsqueeze_2000 = None
    mul_2162: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_84);  primals_84 = None
    unsqueeze_2002: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2162, 0);  mul_2162 = None
    unsqueeze_2003: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2002, 2);  unsqueeze_2002 = None
    unsqueeze_2004: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2003, 3);  unsqueeze_2003 = None
    mul_2163: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_661, unsqueeze_2001);  sub_661 = unsqueeze_2001 = None
    sub_663: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(where_113, mul_2163);  where_113 = mul_2163 = None
    sub_664: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(sub_663, unsqueeze_1998);  sub_663 = unsqueeze_1998 = None
    mul_2164: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_664, unsqueeze_2004);  sub_664 = unsqueeze_2004 = None
    mul_2165: "f32[64]" = torch.ops.aten.mul.Tensor(sum_390, squeeze_70);  sum_390 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_144 = torch.ops.aten.convolution_backward.default(mul_2164, mean_4, primals_82, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_2164 = mean_4 = primals_82 = None
    getitem_712: "f32[8, 128, 1, 1]" = convolution_backward_144[0]
    getitem_713: "f32[64, 128, 1, 1]" = convolution_backward_144[1]
    getitem_714: "f32[64]" = convolution_backward_144[2];  convolution_backward_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_86: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(getitem_712, [8, 128, 32, 32]);  getitem_712 = None
    div_62: "f32[8, 128, 32, 32]" = torch.ops.aten.div.Scalar(expand_86, 1024);  expand_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_2005: "f32[8, 1, 128, 32, 32]" = torch.ops.aten.unsqueeze.default(div_62, 1);  div_62 = None
    expand_87: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_2005, [8, 2, 128, 32, 32]);  unsqueeze_2005 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_784: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_2154, expand_87);  mul_2154 = expand_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_345: "f32[8, 256, 32, 32]" = torch.ops.aten.view.default(add_784, [8, 256, 32, 32]);  add_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_540: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_541: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(alias_540);  alias_540 = None
    le_114: "b8[8, 256, 32, 32]" = torch.ops.aten.le.Scalar(alias_541, 0);  alias_541 = None
    where_114: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(le_114, full_default, view_345);  le_114 = view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_391: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_114, [0, 2, 3])
    sub_665: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_2008);  convolution_26 = unsqueeze_2008 = None
    mul_2166: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(where_114, sub_665)
    sum_392: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2166, [0, 2, 3]);  mul_2166 = None
    mul_2167: "f32[256]" = torch.ops.aten.mul.Tensor(sum_391, 0.0001220703125)
    unsqueeze_2009: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2167, 0);  mul_2167 = None
    unsqueeze_2010: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2009, 2);  unsqueeze_2009 = None
    unsqueeze_2011: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2010, 3);  unsqueeze_2010 = None
    mul_2168: "f32[256]" = torch.ops.aten.mul.Tensor(sum_392, 0.0001220703125)
    mul_2169: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_2170: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2168, mul_2169);  mul_2168 = mul_2169 = None
    unsqueeze_2012: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2170, 0);  mul_2170 = None
    unsqueeze_2013: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2012, 2);  unsqueeze_2012 = None
    unsqueeze_2014: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2013, 3);  unsqueeze_2013 = None
    mul_2171: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_80);  primals_80 = None
    unsqueeze_2015: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2171, 0);  mul_2171 = None
    unsqueeze_2016: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2015, 2);  unsqueeze_2015 = None
    unsqueeze_2017: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2016, 3);  unsqueeze_2016 = None
    mul_2172: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_665, unsqueeze_2014);  sub_665 = unsqueeze_2014 = None
    sub_667: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(where_114, mul_2172);  where_114 = mul_2172 = None
    sub_668: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_667, unsqueeze_2011);  sub_667 = unsqueeze_2011 = None
    mul_2173: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_668, unsqueeze_2017);  sub_668 = unsqueeze_2017 = None
    mul_2174: "f32[256]" = torch.ops.aten.mul.Tensor(sum_392, squeeze_67);  sum_392 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_145 = torch.ops.aten.convolution_backward.default(mul_2173, relu_19, primals_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_2173 = primals_79 = None
    getitem_715: "f32[8, 128, 32, 32]" = convolution_backward_145[0]
    getitem_716: "f32[256, 64, 3, 3]" = convolution_backward_145[1];  convolution_backward_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_543: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_544: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_543);  alias_543 = None
    le_115: "b8[8, 128, 32, 32]" = torch.ops.aten.le.Scalar(alias_544, 0);  alias_544 = None
    where_115: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(le_115, full_default, getitem_715);  le_115 = getitem_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_393: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_115, [0, 2, 3])
    sub_669: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_2020);  convolution_25 = unsqueeze_2020 = None
    mul_2175: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_115, sub_669)
    sum_394: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_2175, [0, 2, 3]);  mul_2175 = None
    mul_2176: "f32[128]" = torch.ops.aten.mul.Tensor(sum_393, 0.0001220703125)
    unsqueeze_2021: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2176, 0);  mul_2176 = None
    unsqueeze_2022: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2021, 2);  unsqueeze_2021 = None
    unsqueeze_2023: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2022, 3);  unsqueeze_2022 = None
    mul_2177: "f32[128]" = torch.ops.aten.mul.Tensor(sum_394, 0.0001220703125)
    mul_2178: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_2179: "f32[128]" = torch.ops.aten.mul.Tensor(mul_2177, mul_2178);  mul_2177 = mul_2178 = None
    unsqueeze_2024: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2179, 0);  mul_2179 = None
    unsqueeze_2025: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2024, 2);  unsqueeze_2024 = None
    unsqueeze_2026: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2025, 3);  unsqueeze_2025 = None
    mul_2180: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_77);  primals_77 = None
    unsqueeze_2027: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2180, 0);  mul_2180 = None
    unsqueeze_2028: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2027, 2);  unsqueeze_2027 = None
    unsqueeze_2029: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2028, 3);  unsqueeze_2028 = None
    mul_2181: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_669, unsqueeze_2026);  sub_669 = unsqueeze_2026 = None
    sub_671: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_115, mul_2181);  where_115 = mul_2181 = None
    sub_672: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_671, unsqueeze_2023);  sub_671 = unsqueeze_2023 = None
    mul_2182: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_672, unsqueeze_2029);  sub_672 = unsqueeze_2029 = None
    mul_2183: "f32[128]" = torch.ops.aten.mul.Tensor(sum_394, squeeze_64);  sum_394 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_146 = torch.ops.aten.convolution_backward.default(mul_2182, relu_18, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2182 = primals_76 = None
    getitem_718: "f32[8, 512, 32, 32]" = convolution_backward_146[0]
    getitem_719: "f32[128, 512, 1, 1]" = convolution_backward_146[1];  convolution_backward_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_785: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(where_112, getitem_718);  where_112 = getitem_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_546: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_547: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(alias_546);  alias_546 = None
    le_116: "b8[8, 512, 32, 32]" = torch.ops.aten.le.Scalar(alias_547, 0);  alias_547 = None
    where_116: "f32[8, 512, 32, 32]" = torch.ops.aten.where.self(le_116, full_default, add_785);  le_116 = add_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    sum_395: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_116, [0, 2, 3])
    sub_673: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_2032);  convolution_24 = unsqueeze_2032 = None
    mul_2184: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(where_116, sub_673)
    sum_396: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_2184, [0, 2, 3]);  mul_2184 = None
    mul_2185: "f32[512]" = torch.ops.aten.mul.Tensor(sum_395, 0.0001220703125)
    unsqueeze_2033: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2185, 0);  mul_2185 = None
    unsqueeze_2034: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2033, 2);  unsqueeze_2033 = None
    unsqueeze_2035: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2034, 3);  unsqueeze_2034 = None
    mul_2186: "f32[512]" = torch.ops.aten.mul.Tensor(sum_396, 0.0001220703125)
    mul_2187: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_2188: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2186, mul_2187);  mul_2186 = mul_2187 = None
    unsqueeze_2036: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2188, 0);  mul_2188 = None
    unsqueeze_2037: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2036, 2);  unsqueeze_2036 = None
    unsqueeze_2038: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2037, 3);  unsqueeze_2037 = None
    mul_2189: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_74);  primals_74 = None
    unsqueeze_2039: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2189, 0);  mul_2189 = None
    unsqueeze_2040: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2039, 2);  unsqueeze_2039 = None
    unsqueeze_2041: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2040, 3);  unsqueeze_2040 = None
    mul_2190: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_673, unsqueeze_2038);  sub_673 = unsqueeze_2038 = None
    sub_675: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(where_116, mul_2190);  mul_2190 = None
    sub_676: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_675, unsqueeze_2035);  sub_675 = None
    mul_2191: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_676, unsqueeze_2041);  sub_676 = unsqueeze_2041 = None
    mul_2192: "f32[512]" = torch.ops.aten.mul.Tensor(sum_396, squeeze_61);  sum_396 = squeeze_61 = None
    convolution_backward_147 = torch.ops.aten.convolution_backward.default(mul_2191, avg_pool2d_1, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2191 = avg_pool2d_1 = primals_73 = None
    getitem_721: "f32[8, 256, 32, 32]" = convolution_backward_147[0]
    getitem_722: "f32[512, 256, 1, 1]" = convolution_backward_147[1];  convolution_backward_147 = None
    avg_pool2d_backward_4: "f32[8, 256, 64, 64]" = torch.ops.aten.avg_pool2d_backward.default(getitem_721, relu_14, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sub_677: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_2044);  convolution_23 = unsqueeze_2044 = None
    mul_2193: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(where_116, sub_677)
    sum_398: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_2193, [0, 2, 3]);  mul_2193 = None
    mul_2195: "f32[512]" = torch.ops.aten.mul.Tensor(sum_398, 0.0001220703125)
    mul_2196: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_2197: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2195, mul_2196);  mul_2195 = mul_2196 = None
    unsqueeze_2048: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2197, 0);  mul_2197 = None
    unsqueeze_2049: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2048, 2);  unsqueeze_2048 = None
    unsqueeze_2050: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2049, 3);  unsqueeze_2049 = None
    mul_2198: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_71);  primals_71 = None
    unsqueeze_2051: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2198, 0);  mul_2198 = None
    unsqueeze_2052: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2051, 2);  unsqueeze_2051 = None
    unsqueeze_2053: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2052, 3);  unsqueeze_2052 = None
    mul_2199: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_677, unsqueeze_2050);  sub_677 = unsqueeze_2050 = None
    sub_679: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(where_116, mul_2199);  where_116 = mul_2199 = None
    sub_680: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_679, unsqueeze_2035);  sub_679 = unsqueeze_2035 = None
    mul_2200: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_680, unsqueeze_2053);  sub_680 = unsqueeze_2053 = None
    mul_2201: "f32[512]" = torch.ops.aten.mul.Tensor(sum_398, squeeze_58);  sum_398 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_148 = torch.ops.aten.convolution_backward.default(mul_2200, avg_pool2d, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2200 = avg_pool2d = primals_70 = None
    getitem_724: "f32[8, 128, 32, 32]" = convolution_backward_148[0]
    getitem_725: "f32[512, 128, 1, 1]" = convolution_backward_148[1];  convolution_backward_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    avg_pool2d_backward_5: "f32[8, 128, 64, 64]" = torch.ops.aten.avg_pool2d_backward.default(getitem_724, sum_12, [3, 3], [2, 2], [1, 1], False, True, None);  getitem_724 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_2054: "f32[8, 1, 128, 64, 64]" = torch.ops.aten.unsqueeze.default(avg_pool2d_backward_5, 1);  avg_pool2d_backward_5 = None
    expand_88: "f32[8, 2, 128, 64, 64]" = torch.ops.aten.expand.default(unsqueeze_2054, [8, 2, 128, 64, 64]);  unsqueeze_2054 = None
    mul_2202: "f32[8, 2, 128, 64, 64]" = torch.ops.aten.mul.Tensor(expand_88, view_19);  view_19 = None
    mul_2203: "f32[8, 2, 128, 64, 64]" = torch.ops.aten.mul.Tensor(expand_88, view_23);  expand_88 = view_23 = None
    sum_399: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_2202, [3, 4], True);  mul_2202 = None
    view_346: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(sum_399, [8, 256, 1, 1]);  sum_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_347: "f32[8, 256]" = torch.ops.aten.view.default(view_346, [8, 256]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_348: "f32[8, 2, 1, 128]" = torch.ops.aten.view.default(view_347, [8, 2, 1, 128]);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_548: "f32[8, 2, 1, 128]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_2204: "f32[8, 2, 1, 128]" = torch.ops.aten.mul.Tensor(view_348, alias_548);  view_348 = None
    sum_400: "f32[8, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_2204, [1], True)
    mul_2205: "f32[8, 2, 1, 128]" = torch.ops.aten.mul.Tensor(alias_548, sum_400);  alias_548 = sum_400 = None
    sub_681: "f32[8, 2, 1, 128]" = torch.ops.aten.sub.Tensor(mul_2204, mul_2205);  mul_2204 = mul_2205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_67: "f32[8, 1, 2, 128]" = torch.ops.aten.permute.default(sub_681, [0, 2, 1, 3]);  sub_681 = None
    view_349: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(permute_67, [8, 256, 1, 1]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_149 = torch.ops.aten.convolution_backward.default(view_349, relu_17, primals_68, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_349 = primals_68 = None
    getitem_727: "f32[8, 64, 1, 1]" = convolution_backward_149[0]
    getitem_728: "f32[256, 64, 1, 1]" = convolution_backward_149[1]
    getitem_729: "f32[256]" = convolution_backward_149[2];  convolution_backward_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_550: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_551: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(alias_550);  alias_550 = None
    le_117: "b8[8, 64, 1, 1]" = torch.ops.aten.le.Scalar(alias_551, 0);  alias_551 = None
    where_117: "f32[8, 64, 1, 1]" = torch.ops.aten.where.self(le_117, full_default, getitem_727);  le_117 = getitem_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_2055: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_2056: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2055, 2);  unsqueeze_2055 = None
    unsqueeze_2057: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2056, 3);  unsqueeze_2056 = None
    sum_401: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_117, [0, 2, 3])
    sub_682: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_2057);  convolution_21 = unsqueeze_2057 = None
    mul_2206: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(where_117, sub_682)
    sum_402: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_2206, [0, 2, 3]);  mul_2206 = None
    mul_2207: "f32[64]" = torch.ops.aten.mul.Tensor(sum_401, 0.125)
    unsqueeze_2058: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2207, 0);  mul_2207 = None
    unsqueeze_2059: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2058, 2);  unsqueeze_2058 = None
    unsqueeze_2060: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2059, 3);  unsqueeze_2059 = None
    mul_2208: "f32[64]" = torch.ops.aten.mul.Tensor(sum_402, 0.125)
    mul_2209: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_2210: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2208, mul_2209);  mul_2208 = mul_2209 = None
    unsqueeze_2061: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2210, 0);  mul_2210 = None
    unsqueeze_2062: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2061, 2);  unsqueeze_2061 = None
    unsqueeze_2063: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2062, 3);  unsqueeze_2062 = None
    mul_2211: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_66);  primals_66 = None
    unsqueeze_2064: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2211, 0);  mul_2211 = None
    unsqueeze_2065: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2064, 2);  unsqueeze_2064 = None
    unsqueeze_2066: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2065, 3);  unsqueeze_2065 = None
    mul_2212: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_682, unsqueeze_2063);  sub_682 = unsqueeze_2063 = None
    sub_684: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(where_117, mul_2212);  where_117 = mul_2212 = None
    sub_685: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(sub_684, unsqueeze_2060);  sub_684 = unsqueeze_2060 = None
    mul_2213: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_685, unsqueeze_2066);  sub_685 = unsqueeze_2066 = None
    mul_2214: "f32[64]" = torch.ops.aten.mul.Tensor(sum_402, squeeze_55);  sum_402 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_150 = torch.ops.aten.convolution_backward.default(mul_2213, mean_3, primals_64, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_2213 = mean_3 = primals_64 = None
    getitem_730: "f32[8, 128, 1, 1]" = convolution_backward_150[0]
    getitem_731: "f32[64, 128, 1, 1]" = convolution_backward_150[1]
    getitem_732: "f32[64]" = convolution_backward_150[2];  convolution_backward_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_89: "f32[8, 128, 64, 64]" = torch.ops.aten.expand.default(getitem_730, [8, 128, 64, 64]);  getitem_730 = None
    div_63: "f32[8, 128, 64, 64]" = torch.ops.aten.div.Scalar(expand_89, 4096);  expand_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_2067: "f32[8, 1, 128, 64, 64]" = torch.ops.aten.unsqueeze.default(div_63, 1);  div_63 = None
    expand_90: "f32[8, 2, 128, 64, 64]" = torch.ops.aten.expand.default(unsqueeze_2067, [8, 2, 128, 64, 64]);  unsqueeze_2067 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_786: "f32[8, 2, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_2203, expand_90);  mul_2203 = expand_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_350: "f32[8, 256, 64, 64]" = torch.ops.aten.view.default(add_786, [8, 256, 64, 64]);  add_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_553: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_554: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(alias_553);  alias_553 = None
    le_118: "b8[8, 256, 64, 64]" = torch.ops.aten.le.Scalar(alias_554, 0);  alias_554 = None
    where_118: "f32[8, 256, 64, 64]" = torch.ops.aten.where.self(le_118, full_default, view_350);  le_118 = view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_403: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_118, [0, 2, 3])
    sub_686: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_2070);  convolution_20 = unsqueeze_2070 = None
    mul_2215: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(where_118, sub_686)
    sum_404: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2215, [0, 2, 3]);  mul_2215 = None
    mul_2216: "f32[256]" = torch.ops.aten.mul.Tensor(sum_403, 3.0517578125e-05)
    unsqueeze_2071: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2216, 0);  mul_2216 = None
    unsqueeze_2072: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2071, 2);  unsqueeze_2071 = None
    unsqueeze_2073: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2072, 3);  unsqueeze_2072 = None
    mul_2217: "f32[256]" = torch.ops.aten.mul.Tensor(sum_404, 3.0517578125e-05)
    mul_2218: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_2219: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2217, mul_2218);  mul_2217 = mul_2218 = None
    unsqueeze_2074: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2219, 0);  mul_2219 = None
    unsqueeze_2075: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2074, 2);  unsqueeze_2074 = None
    unsqueeze_2076: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2075, 3);  unsqueeze_2075 = None
    mul_2220: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_62);  primals_62 = None
    unsqueeze_2077: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2220, 0);  mul_2220 = None
    unsqueeze_2078: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2077, 2);  unsqueeze_2077 = None
    unsqueeze_2079: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2078, 3);  unsqueeze_2078 = None
    mul_2221: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_686, unsqueeze_2076);  sub_686 = unsqueeze_2076 = None
    sub_688: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(where_118, mul_2221);  where_118 = mul_2221 = None
    sub_689: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_688, unsqueeze_2073);  sub_688 = unsqueeze_2073 = None
    mul_2222: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_689, unsqueeze_2079);  sub_689 = unsqueeze_2079 = None
    mul_2223: "f32[256]" = torch.ops.aten.mul.Tensor(sum_404, squeeze_52);  sum_404 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_151 = torch.ops.aten.convolution_backward.default(mul_2222, relu_15, primals_61, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_2222 = primals_61 = None
    getitem_733: "f32[8, 128, 64, 64]" = convolution_backward_151[0]
    getitem_734: "f32[256, 64, 3, 3]" = convolution_backward_151[1];  convolution_backward_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_556: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_557: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_556);  alias_556 = None
    le_119: "b8[8, 128, 64, 64]" = torch.ops.aten.le.Scalar(alias_557, 0);  alias_557 = None
    where_119: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(le_119, full_default, getitem_733);  le_119 = getitem_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_405: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_119, [0, 2, 3])
    sub_690: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_2082);  convolution_19 = unsqueeze_2082 = None
    mul_2224: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_119, sub_690)
    sum_406: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_2224, [0, 2, 3]);  mul_2224 = None
    mul_2225: "f32[128]" = torch.ops.aten.mul.Tensor(sum_405, 3.0517578125e-05)
    unsqueeze_2083: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2225, 0);  mul_2225 = None
    unsqueeze_2084: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2083, 2);  unsqueeze_2083 = None
    unsqueeze_2085: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2084, 3);  unsqueeze_2084 = None
    mul_2226: "f32[128]" = torch.ops.aten.mul.Tensor(sum_406, 3.0517578125e-05)
    mul_2227: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_2228: "f32[128]" = torch.ops.aten.mul.Tensor(mul_2226, mul_2227);  mul_2226 = mul_2227 = None
    unsqueeze_2086: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2228, 0);  mul_2228 = None
    unsqueeze_2087: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2086, 2);  unsqueeze_2086 = None
    unsqueeze_2088: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2087, 3);  unsqueeze_2087 = None
    mul_2229: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_59);  primals_59 = None
    unsqueeze_2089: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2229, 0);  mul_2229 = None
    unsqueeze_2090: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2089, 2);  unsqueeze_2089 = None
    unsqueeze_2091: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2090, 3);  unsqueeze_2090 = None
    mul_2230: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_690, unsqueeze_2088);  sub_690 = unsqueeze_2088 = None
    sub_692: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_119, mul_2230);  where_119 = mul_2230 = None
    sub_693: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_692, unsqueeze_2085);  sub_692 = unsqueeze_2085 = None
    mul_2231: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_693, unsqueeze_2091);  sub_693 = unsqueeze_2091 = None
    mul_2232: "f32[128]" = torch.ops.aten.mul.Tensor(sum_406, squeeze_49);  sum_406 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_152 = torch.ops.aten.convolution_backward.default(mul_2231, relu_14, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2231 = primals_58 = None
    getitem_736: "f32[8, 256, 64, 64]" = convolution_backward_152[0]
    getitem_737: "f32[128, 256, 1, 1]" = convolution_backward_152[1];  convolution_backward_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_787: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_4, getitem_736);  avg_pool2d_backward_4 = getitem_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_559: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_560: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(alias_559);  alias_559 = None
    le_120: "b8[8, 256, 64, 64]" = torch.ops.aten.le.Scalar(alias_560, 0);  alias_560 = None
    where_120: "f32[8, 256, 64, 64]" = torch.ops.aten.where.self(le_120, full_default, add_787);  le_120 = add_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_407: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_120, [0, 2, 3])
    sub_694: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_2094);  convolution_18 = unsqueeze_2094 = None
    mul_2233: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(where_120, sub_694)
    sum_408: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2233, [0, 2, 3]);  mul_2233 = None
    mul_2234: "f32[256]" = torch.ops.aten.mul.Tensor(sum_407, 3.0517578125e-05)
    unsqueeze_2095: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2234, 0);  mul_2234 = None
    unsqueeze_2096: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2095, 2);  unsqueeze_2095 = None
    unsqueeze_2097: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2096, 3);  unsqueeze_2096 = None
    mul_2235: "f32[256]" = torch.ops.aten.mul.Tensor(sum_408, 3.0517578125e-05)
    mul_2236: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_2237: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2235, mul_2236);  mul_2235 = mul_2236 = None
    unsqueeze_2098: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2237, 0);  mul_2237 = None
    unsqueeze_2099: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2098, 2);  unsqueeze_2098 = None
    unsqueeze_2100: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2099, 3);  unsqueeze_2099 = None
    mul_2238: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_56);  primals_56 = None
    unsqueeze_2101: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2238, 0);  mul_2238 = None
    unsqueeze_2102: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2101, 2);  unsqueeze_2101 = None
    unsqueeze_2103: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2102, 3);  unsqueeze_2102 = None
    mul_2239: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_694, unsqueeze_2100);  sub_694 = unsqueeze_2100 = None
    sub_696: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(where_120, mul_2239);  mul_2239 = None
    sub_697: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_696, unsqueeze_2097);  sub_696 = unsqueeze_2097 = None
    mul_2240: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_697, unsqueeze_2103);  sub_697 = unsqueeze_2103 = None
    mul_2241: "f32[256]" = torch.ops.aten.mul.Tensor(sum_408, squeeze_46);  sum_408 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_153 = torch.ops.aten.convolution_backward.default(mul_2240, sum_9, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2240 = sum_9 = primals_55 = None
    getitem_739: "f32[8, 64, 64, 64]" = convolution_backward_153[0]
    getitem_740: "f32[256, 64, 1, 1]" = convolution_backward_153[1];  convolution_backward_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_2104: "f32[8, 1, 64, 64, 64]" = torch.ops.aten.unsqueeze.default(getitem_739, 1);  getitem_739 = None
    expand_91: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.expand.default(unsqueeze_2104, [8, 2, 64, 64, 64]);  unsqueeze_2104 = None
    mul_2242: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.mul.Tensor(expand_91, view_13);  view_13 = None
    mul_2243: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.mul.Tensor(expand_91, view_17);  expand_91 = view_17 = None
    sum_409: "f32[8, 2, 64, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_2242, [3, 4], True);  mul_2242 = None
    view_351: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(sum_409, [8, 128, 1, 1]);  sum_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_352: "f32[8, 128]" = torch.ops.aten.view.default(view_351, [8, 128]);  view_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_353: "f32[8, 2, 1, 64]" = torch.ops.aten.view.default(view_352, [8, 2, 1, 64]);  view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_561: "f32[8, 2, 1, 64]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    mul_2244: "f32[8, 2, 1, 64]" = torch.ops.aten.mul.Tensor(view_353, alias_561);  view_353 = None
    sum_410: "f32[8, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(mul_2244, [1], True)
    mul_2245: "f32[8, 2, 1, 64]" = torch.ops.aten.mul.Tensor(alias_561, sum_410);  alias_561 = sum_410 = None
    sub_698: "f32[8, 2, 1, 64]" = torch.ops.aten.sub.Tensor(mul_2244, mul_2245);  mul_2244 = mul_2245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_68: "f32[8, 1, 2, 64]" = torch.ops.aten.permute.default(sub_698, [0, 2, 1, 3]);  sub_698 = None
    view_354: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(permute_68, [8, 128, 1, 1]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_154 = torch.ops.aten.convolution_backward.default(view_354, relu_13, primals_53, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_354 = primals_53 = None
    getitem_742: "f32[8, 32, 1, 1]" = convolution_backward_154[0]
    getitem_743: "f32[128, 32, 1, 1]" = convolution_backward_154[1]
    getitem_744: "f32[128]" = convolution_backward_154[2];  convolution_backward_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_563: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_564: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(alias_563);  alias_563 = None
    le_121: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(alias_564, 0);  alias_564 = None
    where_121: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_121, full_default, getitem_742);  le_121 = getitem_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_2105: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_2106: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2105, 2);  unsqueeze_2105 = None
    unsqueeze_2107: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2106, 3);  unsqueeze_2106 = None
    sum_411: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_121, [0, 2, 3])
    sub_699: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_2107);  convolution_16 = unsqueeze_2107 = None
    mul_2246: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(where_121, sub_699)
    sum_412: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_2246, [0, 2, 3]);  mul_2246 = None
    mul_2247: "f32[32]" = torch.ops.aten.mul.Tensor(sum_411, 0.125)
    unsqueeze_2108: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_2247, 0);  mul_2247 = None
    unsqueeze_2109: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2108, 2);  unsqueeze_2108 = None
    unsqueeze_2110: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2109, 3);  unsqueeze_2109 = None
    mul_2248: "f32[32]" = torch.ops.aten.mul.Tensor(sum_412, 0.125)
    mul_2249: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_2250: "f32[32]" = torch.ops.aten.mul.Tensor(mul_2248, mul_2249);  mul_2248 = mul_2249 = None
    unsqueeze_2111: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_2250, 0);  mul_2250 = None
    unsqueeze_2112: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2111, 2);  unsqueeze_2111 = None
    unsqueeze_2113: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2112, 3);  unsqueeze_2112 = None
    mul_2251: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_51);  primals_51 = None
    unsqueeze_2114: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_2251, 0);  mul_2251 = None
    unsqueeze_2115: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2114, 2);  unsqueeze_2114 = None
    unsqueeze_2116: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2115, 3);  unsqueeze_2115 = None
    mul_2252: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_699, unsqueeze_2113);  sub_699 = unsqueeze_2113 = None
    sub_701: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(where_121, mul_2252);  where_121 = mul_2252 = None
    sub_702: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(sub_701, unsqueeze_2110);  sub_701 = unsqueeze_2110 = None
    mul_2253: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_702, unsqueeze_2116);  sub_702 = unsqueeze_2116 = None
    mul_2254: "f32[32]" = torch.ops.aten.mul.Tensor(sum_412, squeeze_43);  sum_412 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_155 = torch.ops.aten.convolution_backward.default(mul_2253, mean_2, primals_49, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_2253 = mean_2 = primals_49 = None
    getitem_745: "f32[8, 64, 1, 1]" = convolution_backward_155[0]
    getitem_746: "f32[32, 64, 1, 1]" = convolution_backward_155[1]
    getitem_747: "f32[32]" = convolution_backward_155[2];  convolution_backward_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_92: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(getitem_745, [8, 64, 64, 64]);  getitem_745 = None
    div_64: "f32[8, 64, 64, 64]" = torch.ops.aten.div.Scalar(expand_92, 4096);  expand_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_2117: "f32[8, 1, 64, 64, 64]" = torch.ops.aten.unsqueeze.default(div_64, 1);  div_64 = None
    expand_93: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.expand.default(unsqueeze_2117, [8, 2, 64, 64, 64]);  unsqueeze_2117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_788: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_2243, expand_93);  mul_2243 = expand_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_355: "f32[8, 128, 64, 64]" = torch.ops.aten.view.default(add_788, [8, 128, 64, 64]);  add_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_566: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_567: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_566);  alias_566 = None
    le_122: "b8[8, 128, 64, 64]" = torch.ops.aten.le.Scalar(alias_567, 0);  alias_567 = None
    where_122: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(le_122, full_default, view_355);  le_122 = view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_413: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_122, [0, 2, 3])
    sub_703: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_2120);  convolution_15 = unsqueeze_2120 = None
    mul_2255: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_122, sub_703)
    sum_414: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_2255, [0, 2, 3]);  mul_2255 = None
    mul_2256: "f32[128]" = torch.ops.aten.mul.Tensor(sum_413, 3.0517578125e-05)
    unsqueeze_2121: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2256, 0);  mul_2256 = None
    unsqueeze_2122: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2121, 2);  unsqueeze_2121 = None
    unsqueeze_2123: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2122, 3);  unsqueeze_2122 = None
    mul_2257: "f32[128]" = torch.ops.aten.mul.Tensor(sum_414, 3.0517578125e-05)
    mul_2258: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_2259: "f32[128]" = torch.ops.aten.mul.Tensor(mul_2257, mul_2258);  mul_2257 = mul_2258 = None
    unsqueeze_2124: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2259, 0);  mul_2259 = None
    unsqueeze_2125: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2124, 2);  unsqueeze_2124 = None
    unsqueeze_2126: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2125, 3);  unsqueeze_2125 = None
    mul_2260: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_47);  primals_47 = None
    unsqueeze_2127: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2260, 0);  mul_2260 = None
    unsqueeze_2128: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2127, 2);  unsqueeze_2127 = None
    unsqueeze_2129: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2128, 3);  unsqueeze_2128 = None
    mul_2261: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_703, unsqueeze_2126);  sub_703 = unsqueeze_2126 = None
    sub_705: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_122, mul_2261);  where_122 = mul_2261 = None
    sub_706: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_705, unsqueeze_2123);  sub_705 = unsqueeze_2123 = None
    mul_2262: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_706, unsqueeze_2129);  sub_706 = unsqueeze_2129 = None
    mul_2263: "f32[128]" = torch.ops.aten.mul.Tensor(sum_414, squeeze_40);  sum_414 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_156 = torch.ops.aten.convolution_backward.default(mul_2262, relu_11, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_2262 = primals_46 = None
    getitem_748: "f32[8, 64, 64, 64]" = convolution_backward_156[0]
    getitem_749: "f32[128, 32, 3, 3]" = convolution_backward_156[1];  convolution_backward_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_569: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_570: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_569);  alias_569 = None
    le_123: "b8[8, 64, 64, 64]" = torch.ops.aten.le.Scalar(alias_570, 0);  alias_570 = None
    where_123: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(le_123, full_default, getitem_748);  le_123 = getitem_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_415: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_123, [0, 2, 3])
    sub_707: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_2132);  convolution_14 = unsqueeze_2132 = None
    mul_2264: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_123, sub_707)
    sum_416: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_2264, [0, 2, 3]);  mul_2264 = None
    mul_2265: "f32[64]" = torch.ops.aten.mul.Tensor(sum_415, 3.0517578125e-05)
    unsqueeze_2133: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2265, 0);  mul_2265 = None
    unsqueeze_2134: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2133, 2);  unsqueeze_2133 = None
    unsqueeze_2135: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2134, 3);  unsqueeze_2134 = None
    mul_2266: "f32[64]" = torch.ops.aten.mul.Tensor(sum_416, 3.0517578125e-05)
    mul_2267: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_2268: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2266, mul_2267);  mul_2266 = mul_2267 = None
    unsqueeze_2136: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2268, 0);  mul_2268 = None
    unsqueeze_2137: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2136, 2);  unsqueeze_2136 = None
    unsqueeze_2138: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2137, 3);  unsqueeze_2137 = None
    mul_2269: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_44);  primals_44 = None
    unsqueeze_2139: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2269, 0);  mul_2269 = None
    unsqueeze_2140: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2139, 2);  unsqueeze_2139 = None
    unsqueeze_2141: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2140, 3);  unsqueeze_2140 = None
    mul_2270: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_707, unsqueeze_2138);  sub_707 = unsqueeze_2138 = None
    sub_709: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_123, mul_2270);  where_123 = mul_2270 = None
    sub_710: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_709, unsqueeze_2135);  sub_709 = unsqueeze_2135 = None
    mul_2271: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_710, unsqueeze_2141);  sub_710 = unsqueeze_2141 = None
    mul_2272: "f32[64]" = torch.ops.aten.mul.Tensor(sum_416, squeeze_37);  sum_416 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_157 = torch.ops.aten.convolution_backward.default(mul_2271, relu_10, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2271 = primals_43 = None
    getitem_751: "f32[8, 256, 64, 64]" = convolution_backward_157[0]
    getitem_752: "f32[64, 256, 1, 1]" = convolution_backward_157[1];  convolution_backward_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_789: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(where_120, getitem_751);  where_120 = getitem_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_572: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_573: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(alias_572);  alias_572 = None
    le_124: "b8[8, 256, 64, 64]" = torch.ops.aten.le.Scalar(alias_573, 0);  alias_573 = None
    where_124: "f32[8, 256, 64, 64]" = torch.ops.aten.where.self(le_124, full_default, add_789);  le_124 = add_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sum_417: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_124, [0, 2, 3])
    sub_711: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_2144);  convolution_13 = unsqueeze_2144 = None
    mul_2273: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(where_124, sub_711)
    sum_418: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2273, [0, 2, 3]);  mul_2273 = None
    mul_2274: "f32[256]" = torch.ops.aten.mul.Tensor(sum_417, 3.0517578125e-05)
    unsqueeze_2145: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2274, 0);  mul_2274 = None
    unsqueeze_2146: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2145, 2);  unsqueeze_2145 = None
    unsqueeze_2147: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2146, 3);  unsqueeze_2146 = None
    mul_2275: "f32[256]" = torch.ops.aten.mul.Tensor(sum_418, 3.0517578125e-05)
    mul_2276: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_2277: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2275, mul_2276);  mul_2275 = mul_2276 = None
    unsqueeze_2148: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2277, 0);  mul_2277 = None
    unsqueeze_2149: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2148, 2);  unsqueeze_2148 = None
    unsqueeze_2150: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2149, 3);  unsqueeze_2149 = None
    mul_2278: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_41);  primals_41 = None
    unsqueeze_2151: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2278, 0);  mul_2278 = None
    unsqueeze_2152: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2151, 2);  unsqueeze_2151 = None
    unsqueeze_2153: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2152, 3);  unsqueeze_2152 = None
    mul_2279: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_711, unsqueeze_2150);  sub_711 = unsqueeze_2150 = None
    sub_713: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(where_124, mul_2279);  mul_2279 = None
    sub_714: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_713, unsqueeze_2147);  sub_713 = unsqueeze_2147 = None
    mul_2280: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_714, unsqueeze_2153);  sub_714 = unsqueeze_2153 = None
    mul_2281: "f32[256]" = torch.ops.aten.mul.Tensor(sum_418, squeeze_34);  sum_418 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_158 = torch.ops.aten.convolution_backward.default(mul_2280, sum_6, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2280 = sum_6 = primals_40 = None
    getitem_754: "f32[8, 64, 64, 64]" = convolution_backward_158[0]
    getitem_755: "f32[256, 64, 1, 1]" = convolution_backward_158[1];  convolution_backward_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_2154: "f32[8, 1, 64, 64, 64]" = torch.ops.aten.unsqueeze.default(getitem_754, 1);  getitem_754 = None
    expand_94: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.expand.default(unsqueeze_2154, [8, 2, 64, 64, 64]);  unsqueeze_2154 = None
    mul_2282: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.mul.Tensor(expand_94, view_7);  view_7 = None
    mul_2283: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.mul.Tensor(expand_94, view_11);  expand_94 = view_11 = None
    sum_419: "f32[8, 2, 64, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_2282, [3, 4], True);  mul_2282 = None
    view_356: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(sum_419, [8, 128, 1, 1]);  sum_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_357: "f32[8, 128]" = torch.ops.aten.view.default(view_356, [8, 128]);  view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_358: "f32[8, 2, 1, 64]" = torch.ops.aten.view.default(view_357, [8, 2, 1, 64]);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_574: "f32[8, 2, 1, 64]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_2284: "f32[8, 2, 1, 64]" = torch.ops.aten.mul.Tensor(view_358, alias_574);  view_358 = None
    sum_420: "f32[8, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(mul_2284, [1], True)
    mul_2285: "f32[8, 2, 1, 64]" = torch.ops.aten.mul.Tensor(alias_574, sum_420);  alias_574 = sum_420 = None
    sub_715: "f32[8, 2, 1, 64]" = torch.ops.aten.sub.Tensor(mul_2284, mul_2285);  mul_2284 = mul_2285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_69: "f32[8, 1, 2, 64]" = torch.ops.aten.permute.default(sub_715, [0, 2, 1, 3]);  sub_715 = None
    view_359: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(permute_69, [8, 128, 1, 1]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_159 = torch.ops.aten.convolution_backward.default(view_359, relu_9, primals_38, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_359 = primals_38 = None
    getitem_757: "f32[8, 32, 1, 1]" = convolution_backward_159[0]
    getitem_758: "f32[128, 32, 1, 1]" = convolution_backward_159[1]
    getitem_759: "f32[128]" = convolution_backward_159[2];  convolution_backward_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_576: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_577: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(alias_576);  alias_576 = None
    le_125: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(alias_577, 0);  alias_577 = None
    where_125: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_125, full_default, getitem_757);  le_125 = getitem_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_2155: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_2156: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2155, 2);  unsqueeze_2155 = None
    unsqueeze_2157: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2156, 3);  unsqueeze_2156 = None
    sum_421: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_125, [0, 2, 3])
    sub_716: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_2157);  convolution_11 = unsqueeze_2157 = None
    mul_2286: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(where_125, sub_716)
    sum_422: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_2286, [0, 2, 3]);  mul_2286 = None
    mul_2287: "f32[32]" = torch.ops.aten.mul.Tensor(sum_421, 0.125)
    unsqueeze_2158: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_2287, 0);  mul_2287 = None
    unsqueeze_2159: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2158, 2);  unsqueeze_2158 = None
    unsqueeze_2160: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2159, 3);  unsqueeze_2159 = None
    mul_2288: "f32[32]" = torch.ops.aten.mul.Tensor(sum_422, 0.125)
    mul_2289: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_2290: "f32[32]" = torch.ops.aten.mul.Tensor(mul_2288, mul_2289);  mul_2288 = mul_2289 = None
    unsqueeze_2161: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_2290, 0);  mul_2290 = None
    unsqueeze_2162: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2161, 2);  unsqueeze_2161 = None
    unsqueeze_2163: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2162, 3);  unsqueeze_2162 = None
    mul_2291: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_36);  primals_36 = None
    unsqueeze_2164: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_2291, 0);  mul_2291 = None
    unsqueeze_2165: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2164, 2);  unsqueeze_2164 = None
    unsqueeze_2166: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2165, 3);  unsqueeze_2165 = None
    mul_2292: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_716, unsqueeze_2163);  sub_716 = unsqueeze_2163 = None
    sub_718: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(where_125, mul_2292);  where_125 = mul_2292 = None
    sub_719: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(sub_718, unsqueeze_2160);  sub_718 = unsqueeze_2160 = None
    mul_2293: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_719, unsqueeze_2166);  sub_719 = unsqueeze_2166 = None
    mul_2294: "f32[32]" = torch.ops.aten.mul.Tensor(sum_422, squeeze_31);  sum_422 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_160 = torch.ops.aten.convolution_backward.default(mul_2293, mean_1, primals_34, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_2293 = mean_1 = primals_34 = None
    getitem_760: "f32[8, 64, 1, 1]" = convolution_backward_160[0]
    getitem_761: "f32[32, 64, 1, 1]" = convolution_backward_160[1]
    getitem_762: "f32[32]" = convolution_backward_160[2];  convolution_backward_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_95: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(getitem_760, [8, 64, 64, 64]);  getitem_760 = None
    div_65: "f32[8, 64, 64, 64]" = torch.ops.aten.div.Scalar(expand_95, 4096);  expand_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_2167: "f32[8, 1, 64, 64, 64]" = torch.ops.aten.unsqueeze.default(div_65, 1);  div_65 = None
    expand_96: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.expand.default(unsqueeze_2167, [8, 2, 64, 64, 64]);  unsqueeze_2167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_790: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_2283, expand_96);  mul_2283 = expand_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_360: "f32[8, 128, 64, 64]" = torch.ops.aten.view.default(add_790, [8, 128, 64, 64]);  add_790 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_579: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_580: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_579);  alias_579 = None
    le_126: "b8[8, 128, 64, 64]" = torch.ops.aten.le.Scalar(alias_580, 0);  alias_580 = None
    where_126: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(le_126, full_default, view_360);  le_126 = view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_423: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_126, [0, 2, 3])
    sub_720: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_2170);  convolution_10 = unsqueeze_2170 = None
    mul_2295: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_126, sub_720)
    sum_424: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_2295, [0, 2, 3]);  mul_2295 = None
    mul_2296: "f32[128]" = torch.ops.aten.mul.Tensor(sum_423, 3.0517578125e-05)
    unsqueeze_2171: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2296, 0);  mul_2296 = None
    unsqueeze_2172: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2171, 2);  unsqueeze_2171 = None
    unsqueeze_2173: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2172, 3);  unsqueeze_2172 = None
    mul_2297: "f32[128]" = torch.ops.aten.mul.Tensor(sum_424, 3.0517578125e-05)
    mul_2298: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_2299: "f32[128]" = torch.ops.aten.mul.Tensor(mul_2297, mul_2298);  mul_2297 = mul_2298 = None
    unsqueeze_2174: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2299, 0);  mul_2299 = None
    unsqueeze_2175: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2174, 2);  unsqueeze_2174 = None
    unsqueeze_2176: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2175, 3);  unsqueeze_2175 = None
    mul_2300: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_32);  primals_32 = None
    unsqueeze_2177: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2300, 0);  mul_2300 = None
    unsqueeze_2178: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2177, 2);  unsqueeze_2177 = None
    unsqueeze_2179: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2178, 3);  unsqueeze_2178 = None
    mul_2301: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_720, unsqueeze_2176);  sub_720 = unsqueeze_2176 = None
    sub_722: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_126, mul_2301);  where_126 = mul_2301 = None
    sub_723: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_722, unsqueeze_2173);  sub_722 = unsqueeze_2173 = None
    mul_2302: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_723, unsqueeze_2179);  sub_723 = unsqueeze_2179 = None
    mul_2303: "f32[128]" = torch.ops.aten.mul.Tensor(sum_424, squeeze_28);  sum_424 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_161 = torch.ops.aten.convolution_backward.default(mul_2302, relu_7, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_2302 = primals_31 = None
    getitem_763: "f32[8, 64, 64, 64]" = convolution_backward_161[0]
    getitem_764: "f32[128, 32, 3, 3]" = convolution_backward_161[1];  convolution_backward_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_582: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_583: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_582);  alias_582 = None
    le_127: "b8[8, 64, 64, 64]" = torch.ops.aten.le.Scalar(alias_583, 0);  alias_583 = None
    where_127: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(le_127, full_default, getitem_763);  le_127 = getitem_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_425: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_127, [0, 2, 3])
    sub_724: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_2182);  convolution_9 = unsqueeze_2182 = None
    mul_2304: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_127, sub_724)
    sum_426: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_2304, [0, 2, 3]);  mul_2304 = None
    mul_2305: "f32[64]" = torch.ops.aten.mul.Tensor(sum_425, 3.0517578125e-05)
    unsqueeze_2183: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2305, 0);  mul_2305 = None
    unsqueeze_2184: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2183, 2);  unsqueeze_2183 = None
    unsqueeze_2185: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2184, 3);  unsqueeze_2184 = None
    mul_2306: "f32[64]" = torch.ops.aten.mul.Tensor(sum_426, 3.0517578125e-05)
    mul_2307: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_2308: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2306, mul_2307);  mul_2306 = mul_2307 = None
    unsqueeze_2186: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2308, 0);  mul_2308 = None
    unsqueeze_2187: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2186, 2);  unsqueeze_2186 = None
    unsqueeze_2188: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2187, 3);  unsqueeze_2187 = None
    mul_2309: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_29);  primals_29 = None
    unsqueeze_2189: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2309, 0);  mul_2309 = None
    unsqueeze_2190: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2189, 2);  unsqueeze_2189 = None
    unsqueeze_2191: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2190, 3);  unsqueeze_2190 = None
    mul_2310: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_724, unsqueeze_2188);  sub_724 = unsqueeze_2188 = None
    sub_726: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_127, mul_2310);  where_127 = mul_2310 = None
    sub_727: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_726, unsqueeze_2185);  sub_726 = unsqueeze_2185 = None
    mul_2311: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_727, unsqueeze_2191);  sub_727 = unsqueeze_2191 = None
    mul_2312: "f32[64]" = torch.ops.aten.mul.Tensor(sum_426, squeeze_25);  sum_426 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_162 = torch.ops.aten.convolution_backward.default(mul_2311, relu_6, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2311 = primals_28 = None
    getitem_766: "f32[8, 256, 64, 64]" = convolution_backward_162[0]
    getitem_767: "f32[64, 256, 1, 1]" = convolution_backward_162[1];  convolution_backward_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_791: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(where_124, getitem_766);  where_124 = getitem_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    alias_585: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_586: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(alias_585);  alias_585 = None
    le_128: "b8[8, 256, 64, 64]" = torch.ops.aten.le.Scalar(alias_586, 0);  alias_586 = None
    where_128: "f32[8, 256, 64, 64]" = torch.ops.aten.where.self(le_128, full_default, add_791);  le_128 = add_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    sum_427: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_128, [0, 2, 3])
    sub_728: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_2194);  convolution_8 = unsqueeze_2194 = None
    mul_2313: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(where_128, sub_728)
    sum_428: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2313, [0, 2, 3]);  mul_2313 = None
    mul_2314: "f32[256]" = torch.ops.aten.mul.Tensor(sum_427, 3.0517578125e-05)
    unsqueeze_2195: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2314, 0);  mul_2314 = None
    unsqueeze_2196: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2195, 2);  unsqueeze_2195 = None
    unsqueeze_2197: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2196, 3);  unsqueeze_2196 = None
    mul_2315: "f32[256]" = torch.ops.aten.mul.Tensor(sum_428, 3.0517578125e-05)
    mul_2316: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_2317: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2315, mul_2316);  mul_2315 = mul_2316 = None
    unsqueeze_2198: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2317, 0);  mul_2317 = None
    unsqueeze_2199: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2198, 2);  unsqueeze_2198 = None
    unsqueeze_2200: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2199, 3);  unsqueeze_2199 = None
    mul_2318: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_26);  primals_26 = None
    unsqueeze_2201: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2318, 0);  mul_2318 = None
    unsqueeze_2202: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2201, 2);  unsqueeze_2201 = None
    unsqueeze_2203: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2202, 3);  unsqueeze_2202 = None
    mul_2319: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_728, unsqueeze_2200);  sub_728 = unsqueeze_2200 = None
    sub_730: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(where_128, mul_2319);  mul_2319 = None
    sub_731: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_730, unsqueeze_2197);  sub_730 = None
    mul_2320: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_731, unsqueeze_2203);  sub_731 = unsqueeze_2203 = None
    mul_2321: "f32[256]" = torch.ops.aten.mul.Tensor(sum_428, squeeze_22);  sum_428 = squeeze_22 = None
    convolution_backward_163 = torch.ops.aten.convolution_backward.default(mul_2320, getitem_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2320 = primals_25 = None
    getitem_769: "f32[8, 128, 64, 64]" = convolution_backward_163[0]
    getitem_770: "f32[256, 128, 1, 1]" = convolution_backward_163[1];  convolution_backward_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    sub_732: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_2206);  convolution_7 = unsqueeze_2206 = None
    mul_2322: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(where_128, sub_732)
    sum_430: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2322, [0, 2, 3]);  mul_2322 = None
    mul_2324: "f32[256]" = torch.ops.aten.mul.Tensor(sum_430, 3.0517578125e-05)
    mul_2325: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_2326: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2324, mul_2325);  mul_2324 = mul_2325 = None
    unsqueeze_2210: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2326, 0);  mul_2326 = None
    unsqueeze_2211: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2210, 2);  unsqueeze_2210 = None
    unsqueeze_2212: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2211, 3);  unsqueeze_2211 = None
    mul_2327: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_23);  primals_23 = None
    unsqueeze_2213: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2327, 0);  mul_2327 = None
    unsqueeze_2214: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2213, 2);  unsqueeze_2213 = None
    unsqueeze_2215: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2214, 3);  unsqueeze_2214 = None
    mul_2328: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_732, unsqueeze_2212);  sub_732 = unsqueeze_2212 = None
    sub_734: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(where_128, mul_2328);  where_128 = mul_2328 = None
    sub_735: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_734, unsqueeze_2197);  sub_734 = unsqueeze_2197 = None
    mul_2329: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_735, unsqueeze_2215);  sub_735 = unsqueeze_2215 = None
    mul_2330: "f32[256]" = torch.ops.aten.mul.Tensor(sum_430, squeeze_19);  sum_430 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_backward_164 = torch.ops.aten.convolution_backward.default(mul_2329, sum_3, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2329 = sum_3 = primals_22 = None
    getitem_772: "f32[8, 64, 64, 64]" = convolution_backward_164[0]
    getitem_773: "f32[256, 64, 1, 1]" = convolution_backward_164[1];  convolution_backward_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    unsqueeze_2216: "f32[8, 1, 64, 64, 64]" = torch.ops.aten.unsqueeze.default(getitem_772, 1);  getitem_772 = None
    expand_97: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.expand.default(unsqueeze_2216, [8, 2, 64, 64, 64]);  unsqueeze_2216 = None
    mul_2331: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.mul.Tensor(expand_97, view_1);  view_1 = None
    mul_2332: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.mul.Tensor(expand_97, view_5);  expand_97 = view_5 = None
    sum_431: "f32[8, 2, 64, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_2331, [3, 4], True);  mul_2331 = None
    view_361: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(sum_431, [8, 128, 1, 1]);  sum_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_362: "f32[8, 128]" = torch.ops.aten.view.default(view_361, [8, 128]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_363: "f32[8, 2, 1, 64]" = torch.ops.aten.view.default(view_362, [8, 2, 1, 64]);  view_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    alias_587: "f32[8, 2, 1, 64]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_2333: "f32[8, 2, 1, 64]" = torch.ops.aten.mul.Tensor(view_363, alias_587);  view_363 = None
    sum_432: "f32[8, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(mul_2333, [1], True)
    mul_2334: "f32[8, 2, 1, 64]" = torch.ops.aten.mul.Tensor(alias_587, sum_432);  alias_587 = sum_432 = None
    sub_736: "f32[8, 2, 1, 64]" = torch.ops.aten.sub.Tensor(mul_2333, mul_2334);  mul_2333 = mul_2334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    permute_70: "f32[8, 1, 2, 64]" = torch.ops.aten.permute.default(sub_736, [0, 2, 1, 3]);  sub_736 = None
    view_364: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(permute_70, [8, 128, 1, 1]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_backward_165 = torch.ops.aten.convolution_backward.default(view_364, relu_5, primals_20, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_364 = primals_20 = None
    getitem_775: "f32[8, 32, 1, 1]" = convolution_backward_165[0]
    getitem_776: "f32[128, 32, 1, 1]" = convolution_backward_165[1]
    getitem_777: "f32[128]" = convolution_backward_165[2];  convolution_backward_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    alias_589: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_590: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(alias_589);  alias_589 = None
    le_129: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(alias_590, 0);  alias_590 = None
    where_129: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_129, full_default, getitem_775);  le_129 = getitem_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    unsqueeze_2217: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_2218: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2217, 2);  unsqueeze_2217 = None
    unsqueeze_2219: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2218, 3);  unsqueeze_2218 = None
    sum_433: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_129, [0, 2, 3])
    sub_737: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_2219);  convolution_5 = unsqueeze_2219 = None
    mul_2335: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(where_129, sub_737)
    sum_434: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_2335, [0, 2, 3]);  mul_2335 = None
    mul_2336: "f32[32]" = torch.ops.aten.mul.Tensor(sum_433, 0.125)
    unsqueeze_2220: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_2336, 0);  mul_2336 = None
    unsqueeze_2221: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2220, 2);  unsqueeze_2220 = None
    unsqueeze_2222: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2221, 3);  unsqueeze_2221 = None
    mul_2337: "f32[32]" = torch.ops.aten.mul.Tensor(sum_434, 0.125)
    mul_2338: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_2339: "f32[32]" = torch.ops.aten.mul.Tensor(mul_2337, mul_2338);  mul_2337 = mul_2338 = None
    unsqueeze_2223: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_2339, 0);  mul_2339 = None
    unsqueeze_2224: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2223, 2);  unsqueeze_2223 = None
    unsqueeze_2225: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2224, 3);  unsqueeze_2224 = None
    mul_2340: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_18);  primals_18 = None
    unsqueeze_2226: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_2340, 0);  mul_2340 = None
    unsqueeze_2227: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2226, 2);  unsqueeze_2226 = None
    unsqueeze_2228: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2227, 3);  unsqueeze_2227 = None
    mul_2341: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_737, unsqueeze_2225);  sub_737 = unsqueeze_2225 = None
    sub_739: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(where_129, mul_2341);  where_129 = mul_2341 = None
    sub_740: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(sub_739, unsqueeze_2222);  sub_739 = unsqueeze_2222 = None
    mul_2342: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_740, unsqueeze_2228);  sub_740 = unsqueeze_2228 = None
    mul_2343: "f32[32]" = torch.ops.aten.mul.Tensor(sum_434, squeeze_16);  sum_434 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_backward_166 = torch.ops.aten.convolution_backward.default(mul_2342, mean, primals_16, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_2342 = mean = primals_16 = None
    getitem_778: "f32[8, 64, 1, 1]" = convolution_backward_166[0]
    getitem_779: "f32[32, 64, 1, 1]" = convolution_backward_166[1]
    getitem_780: "f32[32]" = convolution_backward_166[2];  convolution_backward_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    expand_98: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(getitem_778, [8, 64, 64, 64]);  getitem_778 = None
    div_66: "f32[8, 64, 64, 64]" = torch.ops.aten.div.Scalar(expand_98, 4096);  expand_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    unsqueeze_2229: "f32[8, 1, 64, 64, 64]" = torch.ops.aten.unsqueeze.default(div_66, 1);  div_66 = None
    expand_99: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.expand.default(unsqueeze_2229, [8, 2, 64, 64, 64]);  unsqueeze_2229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    add_792: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_2332, expand_99);  mul_2332 = expand_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    view_365: "f32[8, 128, 64, 64]" = torch.ops.aten.view.default(add_792, [8, 128, 64, 64]);  add_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    alias_592: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_593: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_592);  alias_592 = None
    le_130: "b8[8, 128, 64, 64]" = torch.ops.aten.le.Scalar(alias_593, 0);  alias_593 = None
    where_130: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(le_130, full_default, view_365);  le_130 = view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    sum_435: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_130, [0, 2, 3])
    sub_741: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_2232);  convolution_4 = unsqueeze_2232 = None
    mul_2344: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_130, sub_741)
    sum_436: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_2344, [0, 2, 3]);  mul_2344 = None
    mul_2345: "f32[128]" = torch.ops.aten.mul.Tensor(sum_435, 3.0517578125e-05)
    unsqueeze_2233: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2345, 0);  mul_2345 = None
    unsqueeze_2234: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2233, 2);  unsqueeze_2233 = None
    unsqueeze_2235: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2234, 3);  unsqueeze_2234 = None
    mul_2346: "f32[128]" = torch.ops.aten.mul.Tensor(sum_436, 3.0517578125e-05)
    mul_2347: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_2348: "f32[128]" = torch.ops.aten.mul.Tensor(mul_2346, mul_2347);  mul_2346 = mul_2347 = None
    unsqueeze_2236: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2348, 0);  mul_2348 = None
    unsqueeze_2237: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2236, 2);  unsqueeze_2236 = None
    unsqueeze_2238: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2237, 3);  unsqueeze_2237 = None
    mul_2349: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
    unsqueeze_2239: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2349, 0);  mul_2349 = None
    unsqueeze_2240: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2239, 2);  unsqueeze_2239 = None
    unsqueeze_2241: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2240, 3);  unsqueeze_2240 = None
    mul_2350: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_741, unsqueeze_2238);  sub_741 = unsqueeze_2238 = None
    sub_743: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_130, mul_2350);  where_130 = mul_2350 = None
    sub_744: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_743, unsqueeze_2235);  sub_743 = unsqueeze_2235 = None
    mul_2351: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_744, unsqueeze_2241);  sub_744 = unsqueeze_2241 = None
    mul_2352: "f32[128]" = torch.ops.aten.mul.Tensor(sum_436, squeeze_13);  sum_436 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_backward_167 = torch.ops.aten.convolution_backward.default(mul_2351, relu_3, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_2351 = primals_13 = None
    getitem_781: "f32[8, 64, 64, 64]" = convolution_backward_167[0]
    getitem_782: "f32[128, 32, 3, 3]" = convolution_backward_167[1];  convolution_backward_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    alias_595: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_596: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_595);  alias_595 = None
    le_131: "b8[8, 64, 64, 64]" = torch.ops.aten.le.Scalar(alias_596, 0);  alias_596 = None
    where_131: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(le_131, full_default, getitem_781);  le_131 = getitem_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    sum_437: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_131, [0, 2, 3])
    sub_745: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_2244);  convolution_3 = unsqueeze_2244 = None
    mul_2353: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_131, sub_745)
    sum_438: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_2353, [0, 2, 3]);  mul_2353 = None
    mul_2354: "f32[64]" = torch.ops.aten.mul.Tensor(sum_437, 3.0517578125e-05)
    unsqueeze_2245: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2354, 0);  mul_2354 = None
    unsqueeze_2246: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2245, 2);  unsqueeze_2245 = None
    unsqueeze_2247: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2246, 3);  unsqueeze_2246 = None
    mul_2355: "f32[64]" = torch.ops.aten.mul.Tensor(sum_438, 3.0517578125e-05)
    mul_2356: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_2357: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2355, mul_2356);  mul_2355 = mul_2356 = None
    unsqueeze_2248: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2357, 0);  mul_2357 = None
    unsqueeze_2249: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2248, 2);  unsqueeze_2248 = None
    unsqueeze_2250: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2249, 3);  unsqueeze_2249 = None
    mul_2358: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
    unsqueeze_2251: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2358, 0);  mul_2358 = None
    unsqueeze_2252: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2251, 2);  unsqueeze_2251 = None
    unsqueeze_2253: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2252, 3);  unsqueeze_2252 = None
    mul_2359: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_745, unsqueeze_2250);  sub_745 = unsqueeze_2250 = None
    sub_747: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_131, mul_2359);  where_131 = mul_2359 = None
    sub_748: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_747, unsqueeze_2247);  sub_747 = unsqueeze_2247 = None
    mul_2360: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_748, unsqueeze_2253);  sub_748 = unsqueeze_2253 = None
    mul_2361: "f32[64]" = torch.ops.aten.mul.Tensor(sum_438, squeeze_10);  sum_438 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_backward_168 = torch.ops.aten.convolution_backward.default(mul_2360, getitem_6, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2360 = getitem_6 = primals_10 = None
    getitem_784: "f32[8, 128, 64, 64]" = convolution_backward_168[0]
    getitem_785: "f32[64, 128, 1, 1]" = convolution_backward_168[1];  convolution_backward_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    add_793: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(getitem_769, getitem_784);  getitem_769 = getitem_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    max_pool2d_with_indices_backward: "f32[8, 128, 128, 128]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_793, relu_2, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_7);  add_793 = getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    alias_598: "f32[8, 128, 128, 128]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_599: "f32[8, 128, 128, 128]" = torch.ops.aten.alias.default(alias_598);  alias_598 = None
    le_132: "b8[8, 128, 128, 128]" = torch.ops.aten.le.Scalar(alias_599, 0);  alias_599 = None
    where_132: "f32[8, 128, 128, 128]" = torch.ops.aten.where.self(le_132, full_default, max_pool2d_with_indices_backward);  le_132 = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    sum_439: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_132, [0, 2, 3])
    sub_749: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_2256);  convolution_2 = unsqueeze_2256 = None
    mul_2362: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(where_132, sub_749)
    sum_440: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_2362, [0, 2, 3]);  mul_2362 = None
    mul_2363: "f32[128]" = torch.ops.aten.mul.Tensor(sum_439, 7.62939453125e-06)
    unsqueeze_2257: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2363, 0);  mul_2363 = None
    unsqueeze_2258: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2257, 2);  unsqueeze_2257 = None
    unsqueeze_2259: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2258, 3);  unsqueeze_2258 = None
    mul_2364: "f32[128]" = torch.ops.aten.mul.Tensor(sum_440, 7.62939453125e-06)
    mul_2365: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_2366: "f32[128]" = torch.ops.aten.mul.Tensor(mul_2364, mul_2365);  mul_2364 = mul_2365 = None
    unsqueeze_2260: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2366, 0);  mul_2366 = None
    unsqueeze_2261: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2260, 2);  unsqueeze_2260 = None
    unsqueeze_2262: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2261, 3);  unsqueeze_2261 = None
    mul_2367: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
    unsqueeze_2263: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2367, 0);  mul_2367 = None
    unsqueeze_2264: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2263, 2);  unsqueeze_2263 = None
    unsqueeze_2265: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2264, 3);  unsqueeze_2264 = None
    mul_2368: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_749, unsqueeze_2262);  sub_749 = unsqueeze_2262 = None
    sub_751: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(where_132, mul_2368);  where_132 = mul_2368 = None
    sub_752: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(sub_751, unsqueeze_2259);  sub_751 = unsqueeze_2259 = None
    mul_2369: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_752, unsqueeze_2265);  sub_752 = unsqueeze_2265 = None
    mul_2370: "f32[128]" = torch.ops.aten.mul.Tensor(sum_440, squeeze_7);  sum_440 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution_backward_169 = torch.ops.aten.convolution_backward.default(mul_2369, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2369 = primals_7 = None
    getitem_787: "f32[8, 64, 128, 128]" = convolution_backward_169[0]
    getitem_788: "f32[128, 64, 3, 3]" = convolution_backward_169[1];  convolution_backward_169 = None
    alias_601: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_602: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(alias_601);  alias_601 = None
    le_133: "b8[8, 64, 128, 128]" = torch.ops.aten.le.Scalar(alias_602, 0);  alias_602 = None
    where_133: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(le_133, full_default, getitem_787);  le_133 = getitem_787 = None
    sum_441: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_133, [0, 2, 3])
    sub_753: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_2268);  convolution_1 = unsqueeze_2268 = None
    mul_2371: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(where_133, sub_753)
    sum_442: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_2371, [0, 2, 3]);  mul_2371 = None
    mul_2372: "f32[64]" = torch.ops.aten.mul.Tensor(sum_441, 7.62939453125e-06)
    unsqueeze_2269: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2372, 0);  mul_2372 = None
    unsqueeze_2270: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2269, 2);  unsqueeze_2269 = None
    unsqueeze_2271: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2270, 3);  unsqueeze_2270 = None
    mul_2373: "f32[64]" = torch.ops.aten.mul.Tensor(sum_442, 7.62939453125e-06)
    mul_2374: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_2375: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2373, mul_2374);  mul_2373 = mul_2374 = None
    unsqueeze_2272: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2375, 0);  mul_2375 = None
    unsqueeze_2273: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2272, 2);  unsqueeze_2272 = None
    unsqueeze_2274: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2273, 3);  unsqueeze_2273 = None
    mul_2376: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
    unsqueeze_2275: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2376, 0);  mul_2376 = None
    unsqueeze_2276: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2275, 2);  unsqueeze_2275 = None
    unsqueeze_2277: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2276, 3);  unsqueeze_2276 = None
    mul_2377: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_753, unsqueeze_2274);  sub_753 = unsqueeze_2274 = None
    sub_755: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(where_133, mul_2377);  where_133 = mul_2377 = None
    sub_756: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_755, unsqueeze_2271);  sub_755 = unsqueeze_2271 = None
    mul_2378: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_756, unsqueeze_2277);  sub_756 = unsqueeze_2277 = None
    mul_2379: "f32[64]" = torch.ops.aten.mul.Tensor(sum_442, squeeze_4);  sum_442 = squeeze_4 = None
    convolution_backward_170 = torch.ops.aten.convolution_backward.default(mul_2378, relu, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2378 = primals_4 = None
    getitem_790: "f32[8, 64, 128, 128]" = convolution_backward_170[0]
    getitem_791: "f32[64, 64, 3, 3]" = convolution_backward_170[1];  convolution_backward_170 = None
    alias_604: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_605: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(alias_604);  alias_604 = None
    le_134: "b8[8, 64, 128, 128]" = torch.ops.aten.le.Scalar(alias_605, 0);  alias_605 = None
    where_134: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(le_134, full_default, getitem_790);  le_134 = full_default = getitem_790 = None
    sum_443: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_134, [0, 2, 3])
    sub_757: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_2280);  convolution = unsqueeze_2280 = None
    mul_2380: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(where_134, sub_757)
    sum_444: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_2380, [0, 2, 3]);  mul_2380 = None
    mul_2381: "f32[64]" = torch.ops.aten.mul.Tensor(sum_443, 7.62939453125e-06)
    unsqueeze_2281: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2381, 0);  mul_2381 = None
    unsqueeze_2282: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2281, 2);  unsqueeze_2281 = None
    unsqueeze_2283: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2282, 3);  unsqueeze_2282 = None
    mul_2382: "f32[64]" = torch.ops.aten.mul.Tensor(sum_444, 7.62939453125e-06)
    mul_2383: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_2384: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2382, mul_2383);  mul_2382 = mul_2383 = None
    unsqueeze_2284: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2384, 0);  mul_2384 = None
    unsqueeze_2285: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2284, 2);  unsqueeze_2284 = None
    unsqueeze_2286: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2285, 3);  unsqueeze_2285 = None
    mul_2385: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
    unsqueeze_2287: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2385, 0);  mul_2385 = None
    unsqueeze_2288: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2287, 2);  unsqueeze_2287 = None
    unsqueeze_2289: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2288, 3);  unsqueeze_2288 = None
    mul_2386: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_757, unsqueeze_2286);  sub_757 = unsqueeze_2286 = None
    sub_759: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(where_134, mul_2386);  where_134 = mul_2386 = None
    sub_760: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_759, unsqueeze_2283);  sub_759 = unsqueeze_2283 = None
    mul_2387: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_760, unsqueeze_2289);  sub_760 = unsqueeze_2289 = None
    mul_2388: "f32[64]" = torch.ops.aten.mul.Tensor(sum_444, squeeze_1);  sum_444 = squeeze_1 = None
    convolution_backward_171 = torch.ops.aten.convolution_backward.default(mul_2387, primals_936, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_2387 = primals_936 = primals_1 = None
    getitem_794: "f32[64, 3, 3, 3]" = convolution_backward_171[1];  convolution_backward_171 = None
    return [getitem_794, mul_2388, sum_443, getitem_791, mul_2379, sum_441, getitem_788, mul_2370, sum_439, getitem_785, mul_2361, sum_437, getitem_782, mul_2352, sum_435, getitem_779, getitem_780, mul_2343, sum_433, getitem_776, getitem_777, getitem_773, mul_2330, sum_427, getitem_770, mul_2321, sum_427, getitem_767, mul_2312, sum_425, getitem_764, mul_2303, sum_423, getitem_761, getitem_762, mul_2294, sum_421, getitem_758, getitem_759, getitem_755, mul_2281, sum_417, getitem_752, mul_2272, sum_415, getitem_749, mul_2263, sum_413, getitem_746, getitem_747, mul_2254, sum_411, getitem_743, getitem_744, getitem_740, mul_2241, sum_407, getitem_737, mul_2232, sum_405, getitem_734, mul_2223, sum_403, getitem_731, getitem_732, mul_2214, sum_401, getitem_728, getitem_729, getitem_725, mul_2201, sum_395, getitem_722, mul_2192, sum_395, getitem_719, mul_2183, sum_393, getitem_716, mul_2174, sum_391, getitem_713, getitem_714, mul_2165, sum_389, getitem_710, getitem_711, getitem_707, mul_2152, sum_385, getitem_704, mul_2143, sum_383, getitem_701, mul_2134, sum_381, getitem_698, getitem_699, mul_2125, sum_379, getitem_695, getitem_696, getitem_692, mul_2112, sum_375, getitem_689, mul_2103, sum_373, getitem_686, mul_2094, sum_371, getitem_683, getitem_684, mul_2085, sum_369, getitem_680, getitem_681, getitem_677, mul_2072, sum_365, getitem_674, mul_2063, sum_363, getitem_671, mul_2054, sum_361, getitem_668, getitem_669, mul_2045, sum_359, getitem_665, getitem_666, getitem_662, mul_2032, sum_353, getitem_659, mul_2023, sum_353, getitem_656, mul_2014, sum_351, getitem_653, mul_2005, sum_349, getitem_650, getitem_651, mul_1996, sum_347, getitem_647, getitem_648, getitem_644, mul_1983, sum_343, getitem_641, mul_1974, sum_341, getitem_638, mul_1965, sum_339, getitem_635, getitem_636, mul_1956, sum_337, getitem_632, getitem_633, getitem_629, mul_1943, sum_333, getitem_626, mul_1934, sum_331, getitem_623, mul_1925, sum_329, getitem_620, getitem_621, mul_1916, sum_327, getitem_617, getitem_618, getitem_614, mul_1903, sum_323, getitem_611, mul_1894, sum_321, getitem_608, mul_1885, sum_319, getitem_605, getitem_606, mul_1876, sum_317, getitem_602, getitem_603, getitem_599, mul_1863, sum_313, getitem_596, mul_1854, sum_311, getitem_593, mul_1845, sum_309, getitem_590, getitem_591, mul_1836, sum_307, getitem_587, getitem_588, getitem_584, mul_1823, sum_303, getitem_581, mul_1814, sum_301, getitem_578, mul_1805, sum_299, getitem_575, getitem_576, mul_1796, sum_297, getitem_572, getitem_573, getitem_569, mul_1783, sum_293, getitem_566, mul_1774, sum_291, getitem_563, mul_1765, sum_289, getitem_560, getitem_561, mul_1756, sum_287, getitem_557, getitem_558, getitem_554, mul_1743, sum_283, getitem_551, mul_1734, sum_281, getitem_548, mul_1725, sum_279, getitem_545, getitem_546, mul_1716, sum_277, getitem_542, getitem_543, getitem_539, mul_1703, sum_273, getitem_536, mul_1694, sum_271, getitem_533, mul_1685, sum_269, getitem_530, getitem_531, mul_1676, sum_267, getitem_527, getitem_528, getitem_524, mul_1663, sum_263, getitem_521, mul_1654, sum_261, getitem_518, mul_1645, sum_259, getitem_515, getitem_516, mul_1636, sum_257, getitem_512, getitem_513, getitem_509, mul_1623, sum_253, getitem_506, mul_1614, sum_251, getitem_503, mul_1605, sum_249, getitem_500, getitem_501, mul_1596, sum_247, getitem_497, getitem_498, getitem_494, mul_1583, sum_243, getitem_491, mul_1574, sum_241, getitem_488, mul_1565, sum_239, getitem_485, getitem_486, mul_1556, sum_237, getitem_482, getitem_483, getitem_479, mul_1543, sum_233, getitem_476, mul_1534, sum_231, getitem_473, mul_1525, sum_229, getitem_470, getitem_471, mul_1516, sum_227, getitem_467, getitem_468, getitem_464, mul_1503, sum_223, getitem_461, mul_1494, sum_221, getitem_458, mul_1485, sum_219, getitem_455, getitem_456, mul_1476, sum_217, getitem_452, getitem_453, getitem_449, mul_1463, sum_213, getitem_446, mul_1454, sum_211, getitem_443, mul_1445, sum_209, getitem_440, getitem_441, mul_1436, sum_207, getitem_437, getitem_438, getitem_434, mul_1423, sum_203, getitem_431, mul_1414, sum_201, getitem_428, mul_1405, sum_199, getitem_425, getitem_426, mul_1396, sum_197, getitem_422, getitem_423, getitem_419, mul_1383, sum_193, getitem_416, mul_1374, sum_191, getitem_413, mul_1365, sum_189, getitem_410, getitem_411, mul_1356, sum_187, getitem_407, getitem_408, getitem_404, mul_1343, sum_183, getitem_401, mul_1334, sum_181, getitem_398, mul_1325, sum_179, getitem_395, getitem_396, mul_1316, sum_177, getitem_392, getitem_393, getitem_389, mul_1303, sum_173, getitem_386, mul_1294, sum_171, getitem_383, mul_1285, sum_169, getitem_380, getitem_381, mul_1276, sum_167, getitem_377, getitem_378, getitem_374, mul_1263, sum_163, getitem_371, mul_1254, sum_161, getitem_368, mul_1245, sum_159, getitem_365, getitem_366, mul_1236, sum_157, getitem_362, getitem_363, getitem_359, mul_1223, sum_153, getitem_356, mul_1214, sum_151, getitem_353, mul_1205, sum_149, getitem_350, getitem_351, mul_1196, sum_147, getitem_347, getitem_348, getitem_344, mul_1183, sum_143, getitem_341, mul_1174, sum_141, getitem_338, mul_1165, sum_139, getitem_335, getitem_336, mul_1156, sum_137, getitem_332, getitem_333, getitem_329, mul_1143, sum_133, getitem_326, mul_1134, sum_131, getitem_323, mul_1125, sum_129, getitem_320, getitem_321, mul_1116, sum_127, getitem_317, getitem_318, getitem_314, mul_1103, sum_121, getitem_311, mul_1094, sum_121, getitem_308, mul_1085, sum_119, getitem_305, mul_1076, sum_117, getitem_302, getitem_303, mul_1067, sum_115, getitem_299, getitem_300, getitem_296, mul_1054, sum_111, getitem_293, mul_1045, sum_109, getitem_290, mul_1036, sum_107, getitem_287, getitem_288, mul_1027, sum_105, getitem_284, getitem_285, getitem_281, mul_1014, sum_101, permute_37, view_199, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    