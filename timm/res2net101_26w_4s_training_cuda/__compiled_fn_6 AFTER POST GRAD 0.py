from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[64]", primals_4: "f32[104, 64, 1, 1]", primals_5: "f32[104]", primals_7: "f32[26, 26, 3, 3]", primals_8: "f32[26]", primals_10: "f32[26, 26, 3, 3]", primals_11: "f32[26]", primals_13: "f32[26, 26, 3, 3]", primals_14: "f32[26]", primals_16: "f32[256, 104, 1, 1]", primals_17: "f32[256]", primals_19: "f32[256, 64, 1, 1]", primals_20: "f32[256]", primals_22: "f32[104, 256, 1, 1]", primals_23: "f32[104]", primals_25: "f32[26, 26, 3, 3]", primals_26: "f32[26]", primals_28: "f32[26, 26, 3, 3]", primals_29: "f32[26]", primals_31: "f32[26, 26, 3, 3]", primals_32: "f32[26]", primals_34: "f32[256, 104, 1, 1]", primals_35: "f32[256]", primals_37: "f32[104, 256, 1, 1]", primals_38: "f32[104]", primals_40: "f32[26, 26, 3, 3]", primals_41: "f32[26]", primals_43: "f32[26, 26, 3, 3]", primals_44: "f32[26]", primals_46: "f32[26, 26, 3, 3]", primals_47: "f32[26]", primals_49: "f32[256, 104, 1, 1]", primals_50: "f32[256]", primals_52: "f32[208, 256, 1, 1]", primals_53: "f32[208]", primals_55: "f32[52, 52, 3, 3]", primals_56: "f32[52]", primals_58: "f32[52, 52, 3, 3]", primals_59: "f32[52]", primals_61: "f32[52, 52, 3, 3]", primals_62: "f32[52]", primals_64: "f32[512, 208, 1, 1]", primals_65: "f32[512]", primals_67: "f32[512, 256, 1, 1]", primals_68: "f32[512]", primals_70: "f32[208, 512, 1, 1]", primals_71: "f32[208]", primals_73: "f32[52, 52, 3, 3]", primals_74: "f32[52]", primals_76: "f32[52, 52, 3, 3]", primals_77: "f32[52]", primals_79: "f32[52, 52, 3, 3]", primals_80: "f32[52]", primals_82: "f32[512, 208, 1, 1]", primals_83: "f32[512]", primals_85: "f32[208, 512, 1, 1]", primals_86: "f32[208]", primals_88: "f32[52, 52, 3, 3]", primals_89: "f32[52]", primals_91: "f32[52, 52, 3, 3]", primals_92: "f32[52]", primals_94: "f32[52, 52, 3, 3]", primals_95: "f32[52]", primals_97: "f32[512, 208, 1, 1]", primals_98: "f32[512]", primals_100: "f32[208, 512, 1, 1]", primals_101: "f32[208]", primals_103: "f32[52, 52, 3, 3]", primals_104: "f32[52]", primals_106: "f32[52, 52, 3, 3]", primals_107: "f32[52]", primals_109: "f32[52, 52, 3, 3]", primals_110: "f32[52]", primals_112: "f32[512, 208, 1, 1]", primals_113: "f32[512]", primals_115: "f32[416, 512, 1, 1]", primals_116: "f32[416]", primals_118: "f32[104, 104, 3, 3]", primals_119: "f32[104]", primals_121: "f32[104, 104, 3, 3]", primals_122: "f32[104]", primals_124: "f32[104, 104, 3, 3]", primals_125: "f32[104]", primals_127: "f32[1024, 416, 1, 1]", primals_128: "f32[1024]", primals_130: "f32[1024, 512, 1, 1]", primals_131: "f32[1024]", primals_133: "f32[416, 1024, 1, 1]", primals_134: "f32[416]", primals_136: "f32[104, 104, 3, 3]", primals_137: "f32[104]", primals_139: "f32[104, 104, 3, 3]", primals_140: "f32[104]", primals_142: "f32[104, 104, 3, 3]", primals_143: "f32[104]", primals_145: "f32[1024, 416, 1, 1]", primals_146: "f32[1024]", primals_148: "f32[416, 1024, 1, 1]", primals_149: "f32[416]", primals_151: "f32[104, 104, 3, 3]", primals_152: "f32[104]", primals_154: "f32[104, 104, 3, 3]", primals_155: "f32[104]", primals_157: "f32[104, 104, 3, 3]", primals_158: "f32[104]", primals_160: "f32[1024, 416, 1, 1]", primals_161: "f32[1024]", primals_163: "f32[416, 1024, 1, 1]", primals_164: "f32[416]", primals_166: "f32[104, 104, 3, 3]", primals_167: "f32[104]", primals_169: "f32[104, 104, 3, 3]", primals_170: "f32[104]", primals_172: "f32[104, 104, 3, 3]", primals_173: "f32[104]", primals_175: "f32[1024, 416, 1, 1]", primals_176: "f32[1024]", primals_178: "f32[416, 1024, 1, 1]", primals_179: "f32[416]", primals_181: "f32[104, 104, 3, 3]", primals_182: "f32[104]", primals_184: "f32[104, 104, 3, 3]", primals_185: "f32[104]", primals_187: "f32[104, 104, 3, 3]", primals_188: "f32[104]", primals_190: "f32[1024, 416, 1, 1]", primals_191: "f32[1024]", primals_193: "f32[416, 1024, 1, 1]", primals_194: "f32[416]", primals_196: "f32[104, 104, 3, 3]", primals_197: "f32[104]", primals_199: "f32[104, 104, 3, 3]", primals_200: "f32[104]", primals_202: "f32[104, 104, 3, 3]", primals_203: "f32[104]", primals_205: "f32[1024, 416, 1, 1]", primals_206: "f32[1024]", primals_208: "f32[416, 1024, 1, 1]", primals_209: "f32[416]", primals_211: "f32[104, 104, 3, 3]", primals_212: "f32[104]", primals_214: "f32[104, 104, 3, 3]", primals_215: "f32[104]", primals_217: "f32[104, 104, 3, 3]", primals_218: "f32[104]", primals_220: "f32[1024, 416, 1, 1]", primals_221: "f32[1024]", primals_223: "f32[416, 1024, 1, 1]", primals_224: "f32[416]", primals_226: "f32[104, 104, 3, 3]", primals_227: "f32[104]", primals_229: "f32[104, 104, 3, 3]", primals_230: "f32[104]", primals_232: "f32[104, 104, 3, 3]", primals_233: "f32[104]", primals_235: "f32[1024, 416, 1, 1]", primals_236: "f32[1024]", primals_238: "f32[416, 1024, 1, 1]", primals_239: "f32[416]", primals_241: "f32[104, 104, 3, 3]", primals_242: "f32[104]", primals_244: "f32[104, 104, 3, 3]", primals_245: "f32[104]", primals_247: "f32[104, 104, 3, 3]", primals_248: "f32[104]", primals_250: "f32[1024, 416, 1, 1]", primals_251: "f32[1024]", primals_253: "f32[416, 1024, 1, 1]", primals_254: "f32[416]", primals_256: "f32[104, 104, 3, 3]", primals_257: "f32[104]", primals_259: "f32[104, 104, 3, 3]", primals_260: "f32[104]", primals_262: "f32[104, 104, 3, 3]", primals_263: "f32[104]", primals_265: "f32[1024, 416, 1, 1]", primals_266: "f32[1024]", primals_268: "f32[416, 1024, 1, 1]", primals_269: "f32[416]", primals_271: "f32[104, 104, 3, 3]", primals_272: "f32[104]", primals_274: "f32[104, 104, 3, 3]", primals_275: "f32[104]", primals_277: "f32[104, 104, 3, 3]", primals_278: "f32[104]", primals_280: "f32[1024, 416, 1, 1]", primals_281: "f32[1024]", primals_283: "f32[416, 1024, 1, 1]", primals_284: "f32[416]", primals_286: "f32[104, 104, 3, 3]", primals_287: "f32[104]", primals_289: "f32[104, 104, 3, 3]", primals_290: "f32[104]", primals_292: "f32[104, 104, 3, 3]", primals_293: "f32[104]", primals_295: "f32[1024, 416, 1, 1]", primals_296: "f32[1024]", primals_298: "f32[416, 1024, 1, 1]", primals_299: "f32[416]", primals_301: "f32[104, 104, 3, 3]", primals_302: "f32[104]", primals_304: "f32[104, 104, 3, 3]", primals_305: "f32[104]", primals_307: "f32[104, 104, 3, 3]", primals_308: "f32[104]", primals_310: "f32[1024, 416, 1, 1]", primals_311: "f32[1024]", primals_313: "f32[416, 1024, 1, 1]", primals_314: "f32[416]", primals_316: "f32[104, 104, 3, 3]", primals_317: "f32[104]", primals_319: "f32[104, 104, 3, 3]", primals_320: "f32[104]", primals_322: "f32[104, 104, 3, 3]", primals_323: "f32[104]", primals_325: "f32[1024, 416, 1, 1]", primals_326: "f32[1024]", primals_328: "f32[416, 1024, 1, 1]", primals_329: "f32[416]", primals_331: "f32[104, 104, 3, 3]", primals_332: "f32[104]", primals_334: "f32[104, 104, 3, 3]", primals_335: "f32[104]", primals_337: "f32[104, 104, 3, 3]", primals_338: "f32[104]", primals_340: "f32[1024, 416, 1, 1]", primals_341: "f32[1024]", primals_343: "f32[416, 1024, 1, 1]", primals_344: "f32[416]", primals_346: "f32[104, 104, 3, 3]", primals_347: "f32[104]", primals_349: "f32[104, 104, 3, 3]", primals_350: "f32[104]", primals_352: "f32[104, 104, 3, 3]", primals_353: "f32[104]", primals_355: "f32[1024, 416, 1, 1]", primals_356: "f32[1024]", primals_358: "f32[416, 1024, 1, 1]", primals_359: "f32[416]", primals_361: "f32[104, 104, 3, 3]", primals_362: "f32[104]", primals_364: "f32[104, 104, 3, 3]", primals_365: "f32[104]", primals_367: "f32[104, 104, 3, 3]", primals_368: "f32[104]", primals_370: "f32[1024, 416, 1, 1]", primals_371: "f32[1024]", primals_373: "f32[416, 1024, 1, 1]", primals_374: "f32[416]", primals_376: "f32[104, 104, 3, 3]", primals_377: "f32[104]", primals_379: "f32[104, 104, 3, 3]", primals_380: "f32[104]", primals_382: "f32[104, 104, 3, 3]", primals_383: "f32[104]", primals_385: "f32[1024, 416, 1, 1]", primals_386: "f32[1024]", primals_388: "f32[416, 1024, 1, 1]", primals_389: "f32[416]", primals_391: "f32[104, 104, 3, 3]", primals_392: "f32[104]", primals_394: "f32[104, 104, 3, 3]", primals_395: "f32[104]", primals_397: "f32[104, 104, 3, 3]", primals_398: "f32[104]", primals_400: "f32[1024, 416, 1, 1]", primals_401: "f32[1024]", primals_403: "f32[416, 1024, 1, 1]", primals_404: "f32[416]", primals_406: "f32[104, 104, 3, 3]", primals_407: "f32[104]", primals_409: "f32[104, 104, 3, 3]", primals_410: "f32[104]", primals_412: "f32[104, 104, 3, 3]", primals_413: "f32[104]", primals_415: "f32[1024, 416, 1, 1]", primals_416: "f32[1024]", primals_418: "f32[416, 1024, 1, 1]", primals_419: "f32[416]", primals_421: "f32[104, 104, 3, 3]", primals_422: "f32[104]", primals_424: "f32[104, 104, 3, 3]", primals_425: "f32[104]", primals_427: "f32[104, 104, 3, 3]", primals_428: "f32[104]", primals_430: "f32[1024, 416, 1, 1]", primals_431: "f32[1024]", primals_433: "f32[416, 1024, 1, 1]", primals_434: "f32[416]", primals_436: "f32[104, 104, 3, 3]", primals_437: "f32[104]", primals_439: "f32[104, 104, 3, 3]", primals_440: "f32[104]", primals_442: "f32[104, 104, 3, 3]", primals_443: "f32[104]", primals_445: "f32[1024, 416, 1, 1]", primals_446: "f32[1024]", primals_448: "f32[416, 1024, 1, 1]", primals_449: "f32[416]", primals_451: "f32[104, 104, 3, 3]", primals_452: "f32[104]", primals_454: "f32[104, 104, 3, 3]", primals_455: "f32[104]", primals_457: "f32[104, 104, 3, 3]", primals_458: "f32[104]", primals_460: "f32[1024, 416, 1, 1]", primals_461: "f32[1024]", primals_463: "f32[832, 1024, 1, 1]", primals_464: "f32[832]", primals_466: "f32[208, 208, 3, 3]", primals_467: "f32[208]", primals_469: "f32[208, 208, 3, 3]", primals_470: "f32[208]", primals_472: "f32[208, 208, 3, 3]", primals_473: "f32[208]", primals_475: "f32[2048, 832, 1, 1]", primals_476: "f32[2048]", primals_478: "f32[2048, 1024, 1, 1]", primals_479: "f32[2048]", primals_481: "f32[832, 2048, 1, 1]", primals_482: "f32[832]", primals_484: "f32[208, 208, 3, 3]", primals_485: "f32[208]", primals_487: "f32[208, 208, 3, 3]", primals_488: "f32[208]", primals_490: "f32[208, 208, 3, 3]", primals_491: "f32[208]", primals_493: "f32[2048, 832, 1, 1]", primals_494: "f32[2048]", primals_496: "f32[832, 2048, 1, 1]", primals_497: "f32[832]", primals_499: "f32[208, 208, 3, 3]", primals_500: "f32[208]", primals_502: "f32[208, 208, 3, 3]", primals_503: "f32[208]", primals_505: "f32[208, 208, 3, 3]", primals_506: "f32[208]", primals_508: "f32[2048, 832, 1, 1]", primals_509: "f32[2048]", primals_1023: "f32[8, 3, 224, 224]", convolution: "f32[8, 64, 112, 112]", squeeze_1: "f32[64]", relu: "f32[8, 64, 112, 112]", getitem_2: "f32[8, 64, 56, 56]", getitem_3: "i64[8, 64, 56, 56]", convolution_1: "f32[8, 104, 56, 56]", squeeze_4: "f32[104]", getitem_10: "f32[8, 26, 56, 56]", convolution_2: "f32[8, 26, 56, 56]", squeeze_7: "f32[26]", getitem_17: "f32[8, 26, 56, 56]", convolution_3: "f32[8, 26, 56, 56]", squeeze_10: "f32[26]", getitem_24: "f32[8, 26, 56, 56]", convolution_4: "f32[8, 26, 56, 56]", squeeze_13: "f32[26]", getitem_31: "f32[8, 26, 56, 56]", cat: "f32[8, 104, 56, 56]", convolution_5: "f32[8, 256, 56, 56]", squeeze_16: "f32[256]", convolution_6: "f32[8, 256, 56, 56]", squeeze_19: "f32[256]", relu_5: "f32[8, 256, 56, 56]", convolution_7: "f32[8, 104, 56, 56]", squeeze_22: "f32[104]", getitem_42: "f32[8, 26, 56, 56]", convolution_8: "f32[8, 26, 56, 56]", squeeze_25: "f32[26]", add_46: "f32[8, 26, 56, 56]", convolution_9: "f32[8, 26, 56, 56]", squeeze_28: "f32[26]", add_52: "f32[8, 26, 56, 56]", convolution_10: "f32[8, 26, 56, 56]", squeeze_31: "f32[26]", cat_1: "f32[8, 104, 56, 56]", convolution_11: "f32[8, 256, 56, 56]", squeeze_34: "f32[256]", relu_10: "f32[8, 256, 56, 56]", convolution_12: "f32[8, 104, 56, 56]", squeeze_37: "f32[104]", getitem_72: "f32[8, 26, 56, 56]", convolution_13: "f32[8, 26, 56, 56]", squeeze_40: "f32[26]", add_74: "f32[8, 26, 56, 56]", convolution_14: "f32[8, 26, 56, 56]", squeeze_43: "f32[26]", add_80: "f32[8, 26, 56, 56]", convolution_15: "f32[8, 26, 56, 56]", squeeze_46: "f32[26]", cat_2: "f32[8, 104, 56, 56]", convolution_16: "f32[8, 256, 56, 56]", squeeze_49: "f32[256]", relu_15: "f32[8, 256, 56, 56]", convolution_17: "f32[8, 208, 56, 56]", squeeze_52: "f32[208]", getitem_102: "f32[8, 52, 56, 56]", convolution_18: "f32[8, 52, 28, 28]", squeeze_55: "f32[52]", getitem_109: "f32[8, 52, 56, 56]", convolution_19: "f32[8, 52, 28, 28]", squeeze_58: "f32[52]", getitem_116: "f32[8, 52, 56, 56]", convolution_20: "f32[8, 52, 28, 28]", squeeze_61: "f32[52]", getitem_123: "f32[8, 52, 56, 56]", cat_3: "f32[8, 208, 28, 28]", convolution_21: "f32[8, 512, 28, 28]", squeeze_64: "f32[512]", convolution_22: "f32[8, 512, 28, 28]", squeeze_67: "f32[512]", relu_20: "f32[8, 512, 28, 28]", convolution_23: "f32[8, 208, 28, 28]", squeeze_70: "f32[208]", getitem_134: "f32[8, 52, 28, 28]", convolution_24: "f32[8, 52, 28, 28]", squeeze_73: "f32[52]", add_133: "f32[8, 52, 28, 28]", convolution_25: "f32[8, 52, 28, 28]", squeeze_76: "f32[52]", add_139: "f32[8, 52, 28, 28]", convolution_26: "f32[8, 52, 28, 28]", squeeze_79: "f32[52]", cat_4: "f32[8, 208, 28, 28]", convolution_27: "f32[8, 512, 28, 28]", squeeze_82: "f32[512]", relu_25: "f32[8, 512, 28, 28]", convolution_28: "f32[8, 208, 28, 28]", squeeze_85: "f32[208]", getitem_164: "f32[8, 52, 28, 28]", convolution_29: "f32[8, 52, 28, 28]", squeeze_88: "f32[52]", add_161: "f32[8, 52, 28, 28]", convolution_30: "f32[8, 52, 28, 28]", squeeze_91: "f32[52]", add_167: "f32[8, 52, 28, 28]", convolution_31: "f32[8, 52, 28, 28]", squeeze_94: "f32[52]", cat_5: "f32[8, 208, 28, 28]", convolution_32: "f32[8, 512, 28, 28]", squeeze_97: "f32[512]", relu_30: "f32[8, 512, 28, 28]", convolution_33: "f32[8, 208, 28, 28]", squeeze_100: "f32[208]", getitem_194: "f32[8, 52, 28, 28]", convolution_34: "f32[8, 52, 28, 28]", squeeze_103: "f32[52]", add_189: "f32[8, 52, 28, 28]", convolution_35: "f32[8, 52, 28, 28]", squeeze_106: "f32[52]", add_195: "f32[8, 52, 28, 28]", convolution_36: "f32[8, 52, 28, 28]", squeeze_109: "f32[52]", cat_6: "f32[8, 208, 28, 28]", convolution_37: "f32[8, 512, 28, 28]", squeeze_112: "f32[512]", relu_35: "f32[8, 512, 28, 28]", convolution_38: "f32[8, 416, 28, 28]", squeeze_115: "f32[416]", getitem_224: "f32[8, 104, 28, 28]", convolution_39: "f32[8, 104, 14, 14]", squeeze_118: "f32[104]", getitem_231: "f32[8, 104, 28, 28]", convolution_40: "f32[8, 104, 14, 14]", squeeze_121: "f32[104]", getitem_238: "f32[8, 104, 28, 28]", convolution_41: "f32[8, 104, 14, 14]", squeeze_124: "f32[104]", getitem_245: "f32[8, 104, 28, 28]", cat_7: "f32[8, 416, 14, 14]", convolution_42: "f32[8, 1024, 14, 14]", squeeze_127: "f32[1024]", convolution_43: "f32[8, 1024, 14, 14]", squeeze_130: "f32[1024]", relu_40: "f32[8, 1024, 14, 14]", convolution_44: "f32[8, 416, 14, 14]", squeeze_133: "f32[416]", getitem_256: "f32[8, 104, 14, 14]", convolution_45: "f32[8, 104, 14, 14]", squeeze_136: "f32[104]", add_248: "f32[8, 104, 14, 14]", convolution_46: "f32[8, 104, 14, 14]", squeeze_139: "f32[104]", add_254: "f32[8, 104, 14, 14]", convolution_47: "f32[8, 104, 14, 14]", squeeze_142: "f32[104]", cat_8: "f32[8, 416, 14, 14]", convolution_48: "f32[8, 1024, 14, 14]", squeeze_145: "f32[1024]", relu_45: "f32[8, 1024, 14, 14]", convolution_49: "f32[8, 416, 14, 14]", squeeze_148: "f32[416]", getitem_286: "f32[8, 104, 14, 14]", convolution_50: "f32[8, 104, 14, 14]", squeeze_151: "f32[104]", add_276: "f32[8, 104, 14, 14]", convolution_51: "f32[8, 104, 14, 14]", squeeze_154: "f32[104]", add_282: "f32[8, 104, 14, 14]", convolution_52: "f32[8, 104, 14, 14]", squeeze_157: "f32[104]", cat_9: "f32[8, 416, 14, 14]", convolution_53: "f32[8, 1024, 14, 14]", squeeze_160: "f32[1024]", relu_50: "f32[8, 1024, 14, 14]", convolution_54: "f32[8, 416, 14, 14]", squeeze_163: "f32[416]", getitem_316: "f32[8, 104, 14, 14]", convolution_55: "f32[8, 104, 14, 14]", squeeze_166: "f32[104]", add_304: "f32[8, 104, 14, 14]", convolution_56: "f32[8, 104, 14, 14]", squeeze_169: "f32[104]", add_310: "f32[8, 104, 14, 14]", convolution_57: "f32[8, 104, 14, 14]", squeeze_172: "f32[104]", cat_10: "f32[8, 416, 14, 14]", convolution_58: "f32[8, 1024, 14, 14]", squeeze_175: "f32[1024]", relu_55: "f32[8, 1024, 14, 14]", convolution_59: "f32[8, 416, 14, 14]", squeeze_178: "f32[416]", getitem_346: "f32[8, 104, 14, 14]", convolution_60: "f32[8, 104, 14, 14]", squeeze_181: "f32[104]", add_332: "f32[8, 104, 14, 14]", convolution_61: "f32[8, 104, 14, 14]", squeeze_184: "f32[104]", add_338: "f32[8, 104, 14, 14]", convolution_62: "f32[8, 104, 14, 14]", squeeze_187: "f32[104]", cat_11: "f32[8, 416, 14, 14]", convolution_63: "f32[8, 1024, 14, 14]", squeeze_190: "f32[1024]", relu_60: "f32[8, 1024, 14, 14]", convolution_64: "f32[8, 416, 14, 14]", squeeze_193: "f32[416]", getitem_376: "f32[8, 104, 14, 14]", convolution_65: "f32[8, 104, 14, 14]", squeeze_196: "f32[104]", add_360: "f32[8, 104, 14, 14]", convolution_66: "f32[8, 104, 14, 14]", squeeze_199: "f32[104]", add_366: "f32[8, 104, 14, 14]", convolution_67: "f32[8, 104, 14, 14]", squeeze_202: "f32[104]", cat_12: "f32[8, 416, 14, 14]", convolution_68: "f32[8, 1024, 14, 14]", squeeze_205: "f32[1024]", relu_65: "f32[8, 1024, 14, 14]", convolution_69: "f32[8, 416, 14, 14]", squeeze_208: "f32[416]", getitem_406: "f32[8, 104, 14, 14]", convolution_70: "f32[8, 104, 14, 14]", squeeze_211: "f32[104]", add_388: "f32[8, 104, 14, 14]", convolution_71: "f32[8, 104, 14, 14]", squeeze_214: "f32[104]", add_394: "f32[8, 104, 14, 14]", convolution_72: "f32[8, 104, 14, 14]", squeeze_217: "f32[104]", cat_13: "f32[8, 416, 14, 14]", convolution_73: "f32[8, 1024, 14, 14]", squeeze_220: "f32[1024]", relu_70: "f32[8, 1024, 14, 14]", convolution_74: "f32[8, 416, 14, 14]", squeeze_223: "f32[416]", getitem_436: "f32[8, 104, 14, 14]", convolution_75: "f32[8, 104, 14, 14]", squeeze_226: "f32[104]", add_416: "f32[8, 104, 14, 14]", convolution_76: "f32[8, 104, 14, 14]", squeeze_229: "f32[104]", add_422: "f32[8, 104, 14, 14]", convolution_77: "f32[8, 104, 14, 14]", squeeze_232: "f32[104]", cat_14: "f32[8, 416, 14, 14]", convolution_78: "f32[8, 1024, 14, 14]", squeeze_235: "f32[1024]", relu_75: "f32[8, 1024, 14, 14]", convolution_79: "f32[8, 416, 14, 14]", squeeze_238: "f32[416]", getitem_466: "f32[8, 104, 14, 14]", convolution_80: "f32[8, 104, 14, 14]", squeeze_241: "f32[104]", add_444: "f32[8, 104, 14, 14]", convolution_81: "f32[8, 104, 14, 14]", squeeze_244: "f32[104]", add_450: "f32[8, 104, 14, 14]", convolution_82: "f32[8, 104, 14, 14]", squeeze_247: "f32[104]", cat_15: "f32[8, 416, 14, 14]", convolution_83: "f32[8, 1024, 14, 14]", squeeze_250: "f32[1024]", relu_80: "f32[8, 1024, 14, 14]", convolution_84: "f32[8, 416, 14, 14]", squeeze_253: "f32[416]", getitem_496: "f32[8, 104, 14, 14]", convolution_85: "f32[8, 104, 14, 14]", squeeze_256: "f32[104]", add_472: "f32[8, 104, 14, 14]", convolution_86: "f32[8, 104, 14, 14]", squeeze_259: "f32[104]", add_478: "f32[8, 104, 14, 14]", convolution_87: "f32[8, 104, 14, 14]", squeeze_262: "f32[104]", cat_16: "f32[8, 416, 14, 14]", convolution_88: "f32[8, 1024, 14, 14]", squeeze_265: "f32[1024]", relu_85: "f32[8, 1024, 14, 14]", convolution_89: "f32[8, 416, 14, 14]", squeeze_268: "f32[416]", getitem_526: "f32[8, 104, 14, 14]", convolution_90: "f32[8, 104, 14, 14]", squeeze_271: "f32[104]", add_500: "f32[8, 104, 14, 14]", convolution_91: "f32[8, 104, 14, 14]", squeeze_274: "f32[104]", add_506: "f32[8, 104, 14, 14]", convolution_92: "f32[8, 104, 14, 14]", squeeze_277: "f32[104]", cat_17: "f32[8, 416, 14, 14]", convolution_93: "f32[8, 1024, 14, 14]", squeeze_280: "f32[1024]", relu_90: "f32[8, 1024, 14, 14]", convolution_94: "f32[8, 416, 14, 14]", squeeze_283: "f32[416]", getitem_556: "f32[8, 104, 14, 14]", convolution_95: "f32[8, 104, 14, 14]", squeeze_286: "f32[104]", add_528: "f32[8, 104, 14, 14]", convolution_96: "f32[8, 104, 14, 14]", squeeze_289: "f32[104]", add_534: "f32[8, 104, 14, 14]", convolution_97: "f32[8, 104, 14, 14]", squeeze_292: "f32[104]", cat_18: "f32[8, 416, 14, 14]", convolution_98: "f32[8, 1024, 14, 14]", squeeze_295: "f32[1024]", relu_95: "f32[8, 1024, 14, 14]", convolution_99: "f32[8, 416, 14, 14]", squeeze_298: "f32[416]", getitem_586: "f32[8, 104, 14, 14]", convolution_100: "f32[8, 104, 14, 14]", squeeze_301: "f32[104]", add_556: "f32[8, 104, 14, 14]", convolution_101: "f32[8, 104, 14, 14]", squeeze_304: "f32[104]", add_562: "f32[8, 104, 14, 14]", convolution_102: "f32[8, 104, 14, 14]", squeeze_307: "f32[104]", cat_19: "f32[8, 416, 14, 14]", convolution_103: "f32[8, 1024, 14, 14]", squeeze_310: "f32[1024]", relu_100: "f32[8, 1024, 14, 14]", convolution_104: "f32[8, 416, 14, 14]", squeeze_313: "f32[416]", getitem_616: "f32[8, 104, 14, 14]", convolution_105: "f32[8, 104, 14, 14]", squeeze_316: "f32[104]", add_584: "f32[8, 104, 14, 14]", convolution_106: "f32[8, 104, 14, 14]", squeeze_319: "f32[104]", add_590: "f32[8, 104, 14, 14]", convolution_107: "f32[8, 104, 14, 14]", squeeze_322: "f32[104]", cat_20: "f32[8, 416, 14, 14]", convolution_108: "f32[8, 1024, 14, 14]", squeeze_325: "f32[1024]", relu_105: "f32[8, 1024, 14, 14]", convolution_109: "f32[8, 416, 14, 14]", squeeze_328: "f32[416]", getitem_646: "f32[8, 104, 14, 14]", convolution_110: "f32[8, 104, 14, 14]", squeeze_331: "f32[104]", add_612: "f32[8, 104, 14, 14]", convolution_111: "f32[8, 104, 14, 14]", squeeze_334: "f32[104]", add_618: "f32[8, 104, 14, 14]", convolution_112: "f32[8, 104, 14, 14]", squeeze_337: "f32[104]", cat_21: "f32[8, 416, 14, 14]", convolution_113: "f32[8, 1024, 14, 14]", squeeze_340: "f32[1024]", relu_110: "f32[8, 1024, 14, 14]", convolution_114: "f32[8, 416, 14, 14]", squeeze_343: "f32[416]", getitem_676: "f32[8, 104, 14, 14]", convolution_115: "f32[8, 104, 14, 14]", squeeze_346: "f32[104]", add_640: "f32[8, 104, 14, 14]", convolution_116: "f32[8, 104, 14, 14]", squeeze_349: "f32[104]", add_646: "f32[8, 104, 14, 14]", convolution_117: "f32[8, 104, 14, 14]", squeeze_352: "f32[104]", cat_22: "f32[8, 416, 14, 14]", convolution_118: "f32[8, 1024, 14, 14]", squeeze_355: "f32[1024]", relu_115: "f32[8, 1024, 14, 14]", convolution_119: "f32[8, 416, 14, 14]", squeeze_358: "f32[416]", getitem_706: "f32[8, 104, 14, 14]", convolution_120: "f32[8, 104, 14, 14]", squeeze_361: "f32[104]", add_668: "f32[8, 104, 14, 14]", convolution_121: "f32[8, 104, 14, 14]", squeeze_364: "f32[104]", add_674: "f32[8, 104, 14, 14]", convolution_122: "f32[8, 104, 14, 14]", squeeze_367: "f32[104]", cat_23: "f32[8, 416, 14, 14]", convolution_123: "f32[8, 1024, 14, 14]", squeeze_370: "f32[1024]", relu_120: "f32[8, 1024, 14, 14]", convolution_124: "f32[8, 416, 14, 14]", squeeze_373: "f32[416]", getitem_736: "f32[8, 104, 14, 14]", convolution_125: "f32[8, 104, 14, 14]", squeeze_376: "f32[104]", add_696: "f32[8, 104, 14, 14]", convolution_126: "f32[8, 104, 14, 14]", squeeze_379: "f32[104]", add_702: "f32[8, 104, 14, 14]", convolution_127: "f32[8, 104, 14, 14]", squeeze_382: "f32[104]", cat_24: "f32[8, 416, 14, 14]", convolution_128: "f32[8, 1024, 14, 14]", squeeze_385: "f32[1024]", relu_125: "f32[8, 1024, 14, 14]", convolution_129: "f32[8, 416, 14, 14]", squeeze_388: "f32[416]", getitem_766: "f32[8, 104, 14, 14]", convolution_130: "f32[8, 104, 14, 14]", squeeze_391: "f32[104]", add_724: "f32[8, 104, 14, 14]", convolution_131: "f32[8, 104, 14, 14]", squeeze_394: "f32[104]", add_730: "f32[8, 104, 14, 14]", convolution_132: "f32[8, 104, 14, 14]", squeeze_397: "f32[104]", cat_25: "f32[8, 416, 14, 14]", convolution_133: "f32[8, 1024, 14, 14]", squeeze_400: "f32[1024]", relu_130: "f32[8, 1024, 14, 14]", convolution_134: "f32[8, 416, 14, 14]", squeeze_403: "f32[416]", getitem_796: "f32[8, 104, 14, 14]", convolution_135: "f32[8, 104, 14, 14]", squeeze_406: "f32[104]", add_752: "f32[8, 104, 14, 14]", convolution_136: "f32[8, 104, 14, 14]", squeeze_409: "f32[104]", add_758: "f32[8, 104, 14, 14]", convolution_137: "f32[8, 104, 14, 14]", squeeze_412: "f32[104]", cat_26: "f32[8, 416, 14, 14]", convolution_138: "f32[8, 1024, 14, 14]", squeeze_415: "f32[1024]", relu_135: "f32[8, 1024, 14, 14]", convolution_139: "f32[8, 416, 14, 14]", squeeze_418: "f32[416]", getitem_826: "f32[8, 104, 14, 14]", convolution_140: "f32[8, 104, 14, 14]", squeeze_421: "f32[104]", add_780: "f32[8, 104, 14, 14]", convolution_141: "f32[8, 104, 14, 14]", squeeze_424: "f32[104]", add_786: "f32[8, 104, 14, 14]", convolution_142: "f32[8, 104, 14, 14]", squeeze_427: "f32[104]", cat_27: "f32[8, 416, 14, 14]", convolution_143: "f32[8, 1024, 14, 14]", squeeze_430: "f32[1024]", relu_140: "f32[8, 1024, 14, 14]", convolution_144: "f32[8, 416, 14, 14]", squeeze_433: "f32[416]", getitem_856: "f32[8, 104, 14, 14]", convolution_145: "f32[8, 104, 14, 14]", squeeze_436: "f32[104]", add_808: "f32[8, 104, 14, 14]", convolution_146: "f32[8, 104, 14, 14]", squeeze_439: "f32[104]", add_814: "f32[8, 104, 14, 14]", convolution_147: "f32[8, 104, 14, 14]", squeeze_442: "f32[104]", cat_28: "f32[8, 416, 14, 14]", convolution_148: "f32[8, 1024, 14, 14]", squeeze_445: "f32[1024]", relu_145: "f32[8, 1024, 14, 14]", convolution_149: "f32[8, 416, 14, 14]", squeeze_448: "f32[416]", getitem_886: "f32[8, 104, 14, 14]", convolution_150: "f32[8, 104, 14, 14]", squeeze_451: "f32[104]", add_836: "f32[8, 104, 14, 14]", convolution_151: "f32[8, 104, 14, 14]", squeeze_454: "f32[104]", add_842: "f32[8, 104, 14, 14]", convolution_152: "f32[8, 104, 14, 14]", squeeze_457: "f32[104]", cat_29: "f32[8, 416, 14, 14]", convolution_153: "f32[8, 1024, 14, 14]", squeeze_460: "f32[1024]", relu_150: "f32[8, 1024, 14, 14]", convolution_154: "f32[8, 832, 14, 14]", squeeze_463: "f32[832]", getitem_916: "f32[8, 208, 14, 14]", convolution_155: "f32[8, 208, 7, 7]", squeeze_466: "f32[208]", getitem_923: "f32[8, 208, 14, 14]", convolution_156: "f32[8, 208, 7, 7]", squeeze_469: "f32[208]", getitem_930: "f32[8, 208, 14, 14]", convolution_157: "f32[8, 208, 7, 7]", squeeze_472: "f32[208]", getitem_937: "f32[8, 208, 14, 14]", cat_30: "f32[8, 832, 7, 7]", convolution_158: "f32[8, 2048, 7, 7]", squeeze_475: "f32[2048]", convolution_159: "f32[8, 2048, 7, 7]", squeeze_478: "f32[2048]", relu_155: "f32[8, 2048, 7, 7]", convolution_160: "f32[8, 832, 7, 7]", squeeze_481: "f32[832]", getitem_948: "f32[8, 208, 7, 7]", convolution_161: "f32[8, 208, 7, 7]", squeeze_484: "f32[208]", add_895: "f32[8, 208, 7, 7]", convolution_162: "f32[8, 208, 7, 7]", squeeze_487: "f32[208]", add_901: "f32[8, 208, 7, 7]", convolution_163: "f32[8, 208, 7, 7]", squeeze_490: "f32[208]", cat_31: "f32[8, 832, 7, 7]", convolution_164: "f32[8, 2048, 7, 7]", squeeze_493: "f32[2048]", relu_160: "f32[8, 2048, 7, 7]", convolution_165: "f32[8, 832, 7, 7]", squeeze_496: "f32[832]", getitem_978: "f32[8, 208, 7, 7]", convolution_166: "f32[8, 208, 7, 7]", squeeze_499: "f32[208]", add_923: "f32[8, 208, 7, 7]", convolution_167: "f32[8, 208, 7, 7]", squeeze_502: "f32[208]", add_929: "f32[8, 208, 7, 7]", convolution_168: "f32[8, 208, 7, 7]", squeeze_505: "f32[208]", cat_32: "f32[8, 832, 7, 7]", convolution_169: "f32[8, 2048, 7, 7]", squeeze_508: "f32[2048]", view: "f32[8, 2048]", permute_1: "f32[1000, 2048]", le: "b8[8, 2048, 7, 7]", unsqueeze_682: "f32[1, 2048, 1, 1]", le_1: "b8[8, 208, 7, 7]", unsqueeze_694: "f32[1, 208, 1, 1]", le_2: "b8[8, 208, 7, 7]", unsqueeze_706: "f32[1, 208, 1, 1]", le_3: "b8[8, 208, 7, 7]", unsqueeze_718: "f32[1, 208, 1, 1]", le_4: "b8[8, 832, 7, 7]", unsqueeze_730: "f32[1, 832, 1, 1]", unsqueeze_742: "f32[1, 2048, 1, 1]", le_6: "b8[8, 208, 7, 7]", unsqueeze_754: "f32[1, 208, 1, 1]", le_7: "b8[8, 208, 7, 7]", unsqueeze_766: "f32[1, 208, 1, 1]", le_8: "b8[8, 208, 7, 7]", unsqueeze_778: "f32[1, 208, 1, 1]", le_9: "b8[8, 832, 7, 7]", unsqueeze_790: "f32[1, 832, 1, 1]", unsqueeze_802: "f32[1, 2048, 1, 1]", unsqueeze_814: "f32[1, 2048, 1, 1]", le_11: "b8[8, 208, 7, 7]", unsqueeze_826: "f32[1, 208, 1, 1]", le_12: "b8[8, 208, 7, 7]", unsqueeze_838: "f32[1, 208, 1, 1]", le_13: "b8[8, 208, 7, 7]", unsqueeze_850: "f32[1, 208, 1, 1]", le_14: "b8[8, 832, 14, 14]", unsqueeze_862: "f32[1, 832, 1, 1]", unsqueeze_874: "f32[1, 1024, 1, 1]", le_16: "b8[8, 104, 14, 14]", unsqueeze_886: "f32[1, 104, 1, 1]", le_17: "b8[8, 104, 14, 14]", unsqueeze_898: "f32[1, 104, 1, 1]", le_18: "b8[8, 104, 14, 14]", unsqueeze_910: "f32[1, 104, 1, 1]", le_19: "b8[8, 416, 14, 14]", unsqueeze_922: "f32[1, 416, 1, 1]", unsqueeze_934: "f32[1, 1024, 1, 1]", le_21: "b8[8, 104, 14, 14]", unsqueeze_946: "f32[1, 104, 1, 1]", le_22: "b8[8, 104, 14, 14]", unsqueeze_958: "f32[1, 104, 1, 1]", le_23: "b8[8, 104, 14, 14]", unsqueeze_970: "f32[1, 104, 1, 1]", le_24: "b8[8, 416, 14, 14]", unsqueeze_982: "f32[1, 416, 1, 1]", unsqueeze_994: "f32[1, 1024, 1, 1]", le_26: "b8[8, 104, 14, 14]", unsqueeze_1006: "f32[1, 104, 1, 1]", le_27: "b8[8, 104, 14, 14]", unsqueeze_1018: "f32[1, 104, 1, 1]", le_28: "b8[8, 104, 14, 14]", unsqueeze_1030: "f32[1, 104, 1, 1]", le_29: "b8[8, 416, 14, 14]", unsqueeze_1042: "f32[1, 416, 1, 1]", unsqueeze_1054: "f32[1, 1024, 1, 1]", le_31: "b8[8, 104, 14, 14]", unsqueeze_1066: "f32[1, 104, 1, 1]", le_32: "b8[8, 104, 14, 14]", unsqueeze_1078: "f32[1, 104, 1, 1]", le_33: "b8[8, 104, 14, 14]", unsqueeze_1090: "f32[1, 104, 1, 1]", le_34: "b8[8, 416, 14, 14]", unsqueeze_1102: "f32[1, 416, 1, 1]", unsqueeze_1114: "f32[1, 1024, 1, 1]", le_36: "b8[8, 104, 14, 14]", unsqueeze_1126: "f32[1, 104, 1, 1]", le_37: "b8[8, 104, 14, 14]", unsqueeze_1138: "f32[1, 104, 1, 1]", le_38: "b8[8, 104, 14, 14]", unsqueeze_1150: "f32[1, 104, 1, 1]", le_39: "b8[8, 416, 14, 14]", unsqueeze_1162: "f32[1, 416, 1, 1]", unsqueeze_1174: "f32[1, 1024, 1, 1]", le_41: "b8[8, 104, 14, 14]", unsqueeze_1186: "f32[1, 104, 1, 1]", le_42: "b8[8, 104, 14, 14]", unsqueeze_1198: "f32[1, 104, 1, 1]", le_43: "b8[8, 104, 14, 14]", unsqueeze_1210: "f32[1, 104, 1, 1]", le_44: "b8[8, 416, 14, 14]", unsqueeze_1222: "f32[1, 416, 1, 1]", unsqueeze_1234: "f32[1, 1024, 1, 1]", le_46: "b8[8, 104, 14, 14]", unsqueeze_1246: "f32[1, 104, 1, 1]", le_47: "b8[8, 104, 14, 14]", unsqueeze_1258: "f32[1, 104, 1, 1]", le_48: "b8[8, 104, 14, 14]", unsqueeze_1270: "f32[1, 104, 1, 1]", le_49: "b8[8, 416, 14, 14]", unsqueeze_1282: "f32[1, 416, 1, 1]", unsqueeze_1294: "f32[1, 1024, 1, 1]", le_51: "b8[8, 104, 14, 14]", unsqueeze_1306: "f32[1, 104, 1, 1]", le_52: "b8[8, 104, 14, 14]", unsqueeze_1318: "f32[1, 104, 1, 1]", le_53: "b8[8, 104, 14, 14]", unsqueeze_1330: "f32[1, 104, 1, 1]", le_54: "b8[8, 416, 14, 14]", unsqueeze_1342: "f32[1, 416, 1, 1]", unsqueeze_1354: "f32[1, 1024, 1, 1]", le_56: "b8[8, 104, 14, 14]", unsqueeze_1366: "f32[1, 104, 1, 1]", le_57: "b8[8, 104, 14, 14]", unsqueeze_1378: "f32[1, 104, 1, 1]", le_58: "b8[8, 104, 14, 14]", unsqueeze_1390: "f32[1, 104, 1, 1]", le_59: "b8[8, 416, 14, 14]", unsqueeze_1402: "f32[1, 416, 1, 1]", unsqueeze_1414: "f32[1, 1024, 1, 1]", le_61: "b8[8, 104, 14, 14]", unsqueeze_1426: "f32[1, 104, 1, 1]", le_62: "b8[8, 104, 14, 14]", unsqueeze_1438: "f32[1, 104, 1, 1]", le_63: "b8[8, 104, 14, 14]", unsqueeze_1450: "f32[1, 104, 1, 1]", le_64: "b8[8, 416, 14, 14]", unsqueeze_1462: "f32[1, 416, 1, 1]", unsqueeze_1474: "f32[1, 1024, 1, 1]", le_66: "b8[8, 104, 14, 14]", unsqueeze_1486: "f32[1, 104, 1, 1]", le_67: "b8[8, 104, 14, 14]", unsqueeze_1498: "f32[1, 104, 1, 1]", le_68: "b8[8, 104, 14, 14]", unsqueeze_1510: "f32[1, 104, 1, 1]", le_69: "b8[8, 416, 14, 14]", unsqueeze_1522: "f32[1, 416, 1, 1]", unsqueeze_1534: "f32[1, 1024, 1, 1]", le_71: "b8[8, 104, 14, 14]", unsqueeze_1546: "f32[1, 104, 1, 1]", le_72: "b8[8, 104, 14, 14]", unsqueeze_1558: "f32[1, 104, 1, 1]", le_73: "b8[8, 104, 14, 14]", unsqueeze_1570: "f32[1, 104, 1, 1]", le_74: "b8[8, 416, 14, 14]", unsqueeze_1582: "f32[1, 416, 1, 1]", unsqueeze_1594: "f32[1, 1024, 1, 1]", le_76: "b8[8, 104, 14, 14]", unsqueeze_1606: "f32[1, 104, 1, 1]", le_77: "b8[8, 104, 14, 14]", unsqueeze_1618: "f32[1, 104, 1, 1]", le_78: "b8[8, 104, 14, 14]", unsqueeze_1630: "f32[1, 104, 1, 1]", le_79: "b8[8, 416, 14, 14]", unsqueeze_1642: "f32[1, 416, 1, 1]", unsqueeze_1654: "f32[1, 1024, 1, 1]", le_81: "b8[8, 104, 14, 14]", unsqueeze_1666: "f32[1, 104, 1, 1]", le_82: "b8[8, 104, 14, 14]", unsqueeze_1678: "f32[1, 104, 1, 1]", le_83: "b8[8, 104, 14, 14]", unsqueeze_1690: "f32[1, 104, 1, 1]", le_84: "b8[8, 416, 14, 14]", unsqueeze_1702: "f32[1, 416, 1, 1]", unsqueeze_1714: "f32[1, 1024, 1, 1]", le_86: "b8[8, 104, 14, 14]", unsqueeze_1726: "f32[1, 104, 1, 1]", le_87: "b8[8, 104, 14, 14]", unsqueeze_1738: "f32[1, 104, 1, 1]", le_88: "b8[8, 104, 14, 14]", unsqueeze_1750: "f32[1, 104, 1, 1]", le_89: "b8[8, 416, 14, 14]", unsqueeze_1762: "f32[1, 416, 1, 1]", unsqueeze_1774: "f32[1, 1024, 1, 1]", le_91: "b8[8, 104, 14, 14]", unsqueeze_1786: "f32[1, 104, 1, 1]", le_92: "b8[8, 104, 14, 14]", unsqueeze_1798: "f32[1, 104, 1, 1]", le_93: "b8[8, 104, 14, 14]", unsqueeze_1810: "f32[1, 104, 1, 1]", le_94: "b8[8, 416, 14, 14]", unsqueeze_1822: "f32[1, 416, 1, 1]", unsqueeze_1834: "f32[1, 1024, 1, 1]", le_96: "b8[8, 104, 14, 14]", unsqueeze_1846: "f32[1, 104, 1, 1]", le_97: "b8[8, 104, 14, 14]", unsqueeze_1858: "f32[1, 104, 1, 1]", le_98: "b8[8, 104, 14, 14]", unsqueeze_1870: "f32[1, 104, 1, 1]", le_99: "b8[8, 416, 14, 14]", unsqueeze_1882: "f32[1, 416, 1, 1]", unsqueeze_1894: "f32[1, 1024, 1, 1]", le_101: "b8[8, 104, 14, 14]", unsqueeze_1906: "f32[1, 104, 1, 1]", le_102: "b8[8, 104, 14, 14]", unsqueeze_1918: "f32[1, 104, 1, 1]", le_103: "b8[8, 104, 14, 14]", unsqueeze_1930: "f32[1, 104, 1, 1]", le_104: "b8[8, 416, 14, 14]", unsqueeze_1942: "f32[1, 416, 1, 1]", unsqueeze_1954: "f32[1, 1024, 1, 1]", le_106: "b8[8, 104, 14, 14]", unsqueeze_1966: "f32[1, 104, 1, 1]", le_107: "b8[8, 104, 14, 14]", unsqueeze_1978: "f32[1, 104, 1, 1]", le_108: "b8[8, 104, 14, 14]", unsqueeze_1990: "f32[1, 104, 1, 1]", le_109: "b8[8, 416, 14, 14]", unsqueeze_2002: "f32[1, 416, 1, 1]", unsqueeze_2014: "f32[1, 1024, 1, 1]", le_111: "b8[8, 104, 14, 14]", unsqueeze_2026: "f32[1, 104, 1, 1]", le_112: "b8[8, 104, 14, 14]", unsqueeze_2038: "f32[1, 104, 1, 1]", le_113: "b8[8, 104, 14, 14]", unsqueeze_2050: "f32[1, 104, 1, 1]", le_114: "b8[8, 416, 14, 14]", unsqueeze_2062: "f32[1, 416, 1, 1]", unsqueeze_2074: "f32[1, 1024, 1, 1]", le_116: "b8[8, 104, 14, 14]", unsqueeze_2086: "f32[1, 104, 1, 1]", le_117: "b8[8, 104, 14, 14]", unsqueeze_2098: "f32[1, 104, 1, 1]", le_118: "b8[8, 104, 14, 14]", unsqueeze_2110: "f32[1, 104, 1, 1]", le_119: "b8[8, 416, 14, 14]", unsqueeze_2122: "f32[1, 416, 1, 1]", unsqueeze_2134: "f32[1, 1024, 1, 1]", le_121: "b8[8, 104, 14, 14]", unsqueeze_2146: "f32[1, 104, 1, 1]", le_122: "b8[8, 104, 14, 14]", unsqueeze_2158: "f32[1, 104, 1, 1]", le_123: "b8[8, 104, 14, 14]", unsqueeze_2170: "f32[1, 104, 1, 1]", le_124: "b8[8, 416, 14, 14]", unsqueeze_2182: "f32[1, 416, 1, 1]", unsqueeze_2194: "f32[1, 1024, 1, 1]", unsqueeze_2206: "f32[1, 1024, 1, 1]", le_126: "b8[8, 104, 14, 14]", unsqueeze_2218: "f32[1, 104, 1, 1]", le_127: "b8[8, 104, 14, 14]", unsqueeze_2230: "f32[1, 104, 1, 1]", le_128: "b8[8, 104, 14, 14]", unsqueeze_2242: "f32[1, 104, 1, 1]", le_129: "b8[8, 416, 28, 28]", unsqueeze_2254: "f32[1, 416, 1, 1]", unsqueeze_2266: "f32[1, 512, 1, 1]", le_131: "b8[8, 52, 28, 28]", unsqueeze_2278: "f32[1, 52, 1, 1]", le_132: "b8[8, 52, 28, 28]", unsqueeze_2290: "f32[1, 52, 1, 1]", le_133: "b8[8, 52, 28, 28]", unsqueeze_2302: "f32[1, 52, 1, 1]", le_134: "b8[8, 208, 28, 28]", unsqueeze_2314: "f32[1, 208, 1, 1]", unsqueeze_2326: "f32[1, 512, 1, 1]", le_136: "b8[8, 52, 28, 28]", unsqueeze_2338: "f32[1, 52, 1, 1]", le_137: "b8[8, 52, 28, 28]", unsqueeze_2350: "f32[1, 52, 1, 1]", le_138: "b8[8, 52, 28, 28]", unsqueeze_2362: "f32[1, 52, 1, 1]", le_139: "b8[8, 208, 28, 28]", unsqueeze_2374: "f32[1, 208, 1, 1]", unsqueeze_2386: "f32[1, 512, 1, 1]", le_141: "b8[8, 52, 28, 28]", unsqueeze_2398: "f32[1, 52, 1, 1]", le_142: "b8[8, 52, 28, 28]", unsqueeze_2410: "f32[1, 52, 1, 1]", le_143: "b8[8, 52, 28, 28]", unsqueeze_2422: "f32[1, 52, 1, 1]", le_144: "b8[8, 208, 28, 28]", unsqueeze_2434: "f32[1, 208, 1, 1]", unsqueeze_2446: "f32[1, 512, 1, 1]", unsqueeze_2458: "f32[1, 512, 1, 1]", le_146: "b8[8, 52, 28, 28]", unsqueeze_2470: "f32[1, 52, 1, 1]", le_147: "b8[8, 52, 28, 28]", unsqueeze_2482: "f32[1, 52, 1, 1]", le_148: "b8[8, 52, 28, 28]", unsqueeze_2494: "f32[1, 52, 1, 1]", le_149: "b8[8, 208, 56, 56]", unsqueeze_2506: "f32[1, 208, 1, 1]", unsqueeze_2518: "f32[1, 256, 1, 1]", le_151: "b8[8, 26, 56, 56]", unsqueeze_2530: "f32[1, 26, 1, 1]", le_152: "b8[8, 26, 56, 56]", unsqueeze_2542: "f32[1, 26, 1, 1]", le_153: "b8[8, 26, 56, 56]", unsqueeze_2554: "f32[1, 26, 1, 1]", le_154: "b8[8, 104, 56, 56]", unsqueeze_2566: "f32[1, 104, 1, 1]", unsqueeze_2578: "f32[1, 256, 1, 1]", le_156: "b8[8, 26, 56, 56]", unsqueeze_2590: "f32[1, 26, 1, 1]", le_157: "b8[8, 26, 56, 56]", unsqueeze_2602: "f32[1, 26, 1, 1]", le_158: "b8[8, 26, 56, 56]", unsqueeze_2614: "f32[1, 26, 1, 1]", le_159: "b8[8, 104, 56, 56]", unsqueeze_2626: "f32[1, 104, 1, 1]", unsqueeze_2638: "f32[1, 256, 1, 1]", unsqueeze_2650: "f32[1, 256, 1, 1]", le_161: "b8[8, 26, 56, 56]", unsqueeze_2662: "f32[1, 26, 1, 1]", le_162: "b8[8, 26, 56, 56]", unsqueeze_2674: "f32[1, 26, 1, 1]", le_163: "b8[8, 26, 56, 56]", unsqueeze_2686: "f32[1, 26, 1, 1]", le_164: "b8[8, 104, 56, 56]", unsqueeze_2698: "f32[1, 104, 1, 1]", unsqueeze_2710: "f32[1, 64, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    mm: "f32[8, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[2048, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 2048, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 2048, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 2048, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 2048, 7, 7]);  view_2 = None
    div: "f32[8, 2048, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_2: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_170: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_169, unsqueeze_682);  convolution_169 = unsqueeze_682 = None
    mul_1190: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_170)
    sum_3: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1190, [0, 2, 3]);  mul_1190 = None
    mul_1191: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_683: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1191, 0);  mul_1191 = None
    unsqueeze_684: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
    unsqueeze_685: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
    mul_1192: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_1193: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_508, squeeze_508)
    mul_1194: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1192, mul_1193);  mul_1192 = mul_1193 = None
    unsqueeze_686: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1194, 0);  mul_1194 = None
    unsqueeze_687: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
    unsqueeze_688: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
    mul_1195: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_508, primals_509);  primals_509 = None
    unsqueeze_689: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1195, 0);  mul_1195 = None
    unsqueeze_690: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    mul_1196: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_688);  sub_170 = unsqueeze_688 = None
    sub_172: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_1196);  mul_1196 = None
    sub_173: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_172, unsqueeze_685);  sub_172 = unsqueeze_685 = None
    mul_1197: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_691);  sub_173 = unsqueeze_691 = None
    mul_1198: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_508);  sum_3 = squeeze_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_1197, cat_32, primals_508, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1197 = cat_32 = primals_508 = None
    getitem_1002: "f32[8, 832, 7, 7]" = convolution_backward[0]
    getitem_1003: "f32[2048, 832, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_1: "f32[8, 208, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1002, 1, 0, 208)
    slice_2: "f32[8, 208, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1002, 1, 208, 416)
    slice_3: "f32[8, 208, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1002, 1, 416, 624)
    slice_4: "f32[8, 208, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1002, 1, 624, 832);  getitem_1002 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_1: "f32[8, 208, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, slice_3);  le_1 = slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_4: "f32[208]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_174: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_168, unsqueeze_694);  convolution_168 = unsqueeze_694 = None
    mul_1199: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_174)
    sum_5: "f32[208]" = torch.ops.aten.sum.dim_IntList(mul_1199, [0, 2, 3]);  mul_1199 = None
    mul_1200: "f32[208]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_695: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1200, 0);  mul_1200 = None
    unsqueeze_696: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
    unsqueeze_697: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
    mul_1201: "f32[208]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_1202: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_505, squeeze_505)
    mul_1203: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1201, mul_1202);  mul_1201 = mul_1202 = None
    unsqueeze_698: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1203, 0);  mul_1203 = None
    unsqueeze_699: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
    unsqueeze_700: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
    mul_1204: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_505, primals_506);  primals_506 = None
    unsqueeze_701: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1204, 0);  mul_1204 = None
    unsqueeze_702: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    mul_1205: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_700);  sub_174 = unsqueeze_700 = None
    sub_176: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_1205);  where_1 = mul_1205 = None
    sub_177: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_697);  sub_176 = unsqueeze_697 = None
    mul_1206: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_703);  sub_177 = unsqueeze_703 = None
    mul_1207: "f32[208]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_505);  sum_5 = squeeze_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_1206, add_929, primals_505, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1206 = add_929 = primals_505 = None
    getitem_1005: "f32[8, 208, 7, 7]" = convolution_backward_1[0]
    getitem_1006: "f32[208, 208, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_941: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(slice_2, getitem_1005);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_2: "f32[8, 208, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, add_941);  le_2 = add_941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_6: "f32[208]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_178: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_706);  convolution_167 = unsqueeze_706 = None
    mul_1208: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_178)
    sum_7: "f32[208]" = torch.ops.aten.sum.dim_IntList(mul_1208, [0, 2, 3]);  mul_1208 = None
    mul_1209: "f32[208]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_707: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1209, 0);  mul_1209 = None
    unsqueeze_708: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
    unsqueeze_709: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
    mul_1210: "f32[208]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_1211: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_502, squeeze_502)
    mul_1212: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1210, mul_1211);  mul_1210 = mul_1211 = None
    unsqueeze_710: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1212, 0);  mul_1212 = None
    unsqueeze_711: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
    unsqueeze_712: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
    mul_1213: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_502, primals_503);  primals_503 = None
    unsqueeze_713: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1213, 0);  mul_1213 = None
    unsqueeze_714: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    mul_1214: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_712);  sub_178 = unsqueeze_712 = None
    sub_180: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_1214);  where_2 = mul_1214 = None
    sub_181: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(sub_180, unsqueeze_709);  sub_180 = unsqueeze_709 = None
    mul_1215: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_715);  sub_181 = unsqueeze_715 = None
    mul_1216: "f32[208]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_502);  sum_7 = squeeze_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_1215, add_923, primals_502, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1215 = add_923 = primals_502 = None
    getitem_1008: "f32[8, 208, 7, 7]" = convolution_backward_2[0]
    getitem_1009: "f32[208, 208, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_942: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(slice_1, getitem_1008);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_3: "f32[8, 208, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, add_942);  le_3 = add_942 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_8: "f32[208]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_182: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_718);  convolution_166 = unsqueeze_718 = None
    mul_1217: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_182)
    sum_9: "f32[208]" = torch.ops.aten.sum.dim_IntList(mul_1217, [0, 2, 3]);  mul_1217 = None
    mul_1218: "f32[208]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_719: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1218, 0);  mul_1218 = None
    unsqueeze_720: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
    unsqueeze_721: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
    mul_1219: "f32[208]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_1220: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_499, squeeze_499)
    mul_1221: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1219, mul_1220);  mul_1219 = mul_1220 = None
    unsqueeze_722: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1221, 0);  mul_1221 = None
    unsqueeze_723: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
    unsqueeze_724: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
    mul_1222: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_499, primals_500);  primals_500 = None
    unsqueeze_725: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1222, 0);  mul_1222 = None
    unsqueeze_726: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    mul_1223: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_724);  sub_182 = unsqueeze_724 = None
    sub_184: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_1223);  where_3 = mul_1223 = None
    sub_185: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_721);  sub_184 = unsqueeze_721 = None
    mul_1224: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_727);  sub_185 = unsqueeze_727 = None
    mul_1225: "f32[208]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_499);  sum_9 = squeeze_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_1224, getitem_978, primals_499, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1224 = getitem_978 = primals_499 = None
    getitem_1011: "f32[8, 208, 7, 7]" = convolution_backward_3[0]
    getitem_1012: "f32[208, 208, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_33: "f32[8, 832, 7, 7]" = torch.ops.aten.cat.default([getitem_1011, getitem_1008, getitem_1005, slice_4], 1);  getitem_1011 = getitem_1008 = getitem_1005 = slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_4: "f32[8, 832, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, cat_33);  le_4 = cat_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_10: "f32[832]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_186: "f32[8, 832, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_165, unsqueeze_730);  convolution_165 = unsqueeze_730 = None
    mul_1226: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_186)
    sum_11: "f32[832]" = torch.ops.aten.sum.dim_IntList(mul_1226, [0, 2, 3]);  mul_1226 = None
    mul_1227: "f32[832]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_731: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_1227, 0);  mul_1227 = None
    unsqueeze_732: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
    unsqueeze_733: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
    mul_1228: "f32[832]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_1229: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_496, squeeze_496)
    mul_1230: "f32[832]" = torch.ops.aten.mul.Tensor(mul_1228, mul_1229);  mul_1228 = mul_1229 = None
    unsqueeze_734: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_1230, 0);  mul_1230 = None
    unsqueeze_735: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
    unsqueeze_736: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
    mul_1231: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_496, primals_497);  primals_497 = None
    unsqueeze_737: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_1231, 0);  mul_1231 = None
    unsqueeze_738: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    mul_1232: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_736);  sub_186 = unsqueeze_736 = None
    sub_188: "f32[8, 832, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_1232);  where_4 = mul_1232 = None
    sub_189: "f32[8, 832, 7, 7]" = torch.ops.aten.sub.Tensor(sub_188, unsqueeze_733);  sub_188 = unsqueeze_733 = None
    mul_1233: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_739);  sub_189 = unsqueeze_739 = None
    mul_1234: "f32[832]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_496);  sum_11 = squeeze_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_1233, relu_160, primals_496, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1233 = primals_496 = None
    getitem_1014: "f32[8, 2048, 7, 7]" = convolution_backward_4[0]
    getitem_1015: "f32[832, 2048, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_943: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where, getitem_1014);  where = getitem_1014 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_5: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(relu_160, 0);  relu_160 = None
    where_5: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, add_943);  le_5 = add_943 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_12: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_190: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_164, unsqueeze_742);  convolution_164 = unsqueeze_742 = None
    mul_1235: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_190)
    sum_13: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1235, [0, 2, 3]);  mul_1235 = None
    mul_1236: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_743: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1236, 0);  mul_1236 = None
    unsqueeze_744: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
    unsqueeze_745: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
    mul_1237: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_1238: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_493, squeeze_493)
    mul_1239: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1237, mul_1238);  mul_1237 = mul_1238 = None
    unsqueeze_746: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1239, 0);  mul_1239 = None
    unsqueeze_747: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
    unsqueeze_748: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
    mul_1240: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_493, primals_494);  primals_494 = None
    unsqueeze_749: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1240, 0);  mul_1240 = None
    unsqueeze_750: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    mul_1241: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_748);  sub_190 = unsqueeze_748 = None
    sub_192: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_1241);  mul_1241 = None
    sub_193: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_192, unsqueeze_745);  sub_192 = unsqueeze_745 = None
    mul_1242: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_751);  sub_193 = unsqueeze_751 = None
    mul_1243: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_493);  sum_13 = squeeze_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_1242, cat_31, primals_493, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1242 = cat_31 = primals_493 = None
    getitem_1017: "f32[8, 832, 7, 7]" = convolution_backward_5[0]
    getitem_1018: "f32[2048, 832, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_5: "f32[8, 208, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1017, 1, 0, 208)
    slice_6: "f32[8, 208, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1017, 1, 208, 416)
    slice_7: "f32[8, 208, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1017, 1, 416, 624)
    slice_8: "f32[8, 208, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1017, 1, 624, 832);  getitem_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_6: "f32[8, 208, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, slice_7);  le_6 = slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_14: "f32[208]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_194: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_163, unsqueeze_754);  convolution_163 = unsqueeze_754 = None
    mul_1244: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_194)
    sum_15: "f32[208]" = torch.ops.aten.sum.dim_IntList(mul_1244, [0, 2, 3]);  mul_1244 = None
    mul_1245: "f32[208]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_755: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1245, 0);  mul_1245 = None
    unsqueeze_756: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
    unsqueeze_757: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
    mul_1246: "f32[208]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_1247: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_490, squeeze_490)
    mul_1248: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1246, mul_1247);  mul_1246 = mul_1247 = None
    unsqueeze_758: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1248, 0);  mul_1248 = None
    unsqueeze_759: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
    unsqueeze_760: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
    mul_1249: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_490, primals_491);  primals_491 = None
    unsqueeze_761: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1249, 0);  mul_1249 = None
    unsqueeze_762: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    mul_1250: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_760);  sub_194 = unsqueeze_760 = None
    sub_196: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_1250);  where_6 = mul_1250 = None
    sub_197: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(sub_196, unsqueeze_757);  sub_196 = unsqueeze_757 = None
    mul_1251: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_763);  sub_197 = unsqueeze_763 = None
    mul_1252: "f32[208]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_490);  sum_15 = squeeze_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_1251, add_901, primals_490, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1251 = add_901 = primals_490 = None
    getitem_1020: "f32[8, 208, 7, 7]" = convolution_backward_6[0]
    getitem_1021: "f32[208, 208, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_944: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(slice_6, getitem_1020);  slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_7: "f32[8, 208, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, add_944);  le_7 = add_944 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_16: "f32[208]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_198: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_162, unsqueeze_766);  convolution_162 = unsqueeze_766 = None
    mul_1253: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_198)
    sum_17: "f32[208]" = torch.ops.aten.sum.dim_IntList(mul_1253, [0, 2, 3]);  mul_1253 = None
    mul_1254: "f32[208]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_767: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1254, 0);  mul_1254 = None
    unsqueeze_768: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
    unsqueeze_769: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
    mul_1255: "f32[208]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_1256: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_487, squeeze_487)
    mul_1257: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1255, mul_1256);  mul_1255 = mul_1256 = None
    unsqueeze_770: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1257, 0);  mul_1257 = None
    unsqueeze_771: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
    unsqueeze_772: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
    mul_1258: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_487, primals_488);  primals_488 = None
    unsqueeze_773: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1258, 0);  mul_1258 = None
    unsqueeze_774: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    mul_1259: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_772);  sub_198 = unsqueeze_772 = None
    sub_200: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_1259);  where_7 = mul_1259 = None
    sub_201: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_769);  sub_200 = unsqueeze_769 = None
    mul_1260: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_775);  sub_201 = unsqueeze_775 = None
    mul_1261: "f32[208]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_487);  sum_17 = squeeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_1260, add_895, primals_487, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1260 = add_895 = primals_487 = None
    getitem_1023: "f32[8, 208, 7, 7]" = convolution_backward_7[0]
    getitem_1024: "f32[208, 208, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_945: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(slice_5, getitem_1023);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_8: "f32[8, 208, 7, 7]" = torch.ops.aten.where.self(le_8, full_default, add_945);  le_8 = add_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_18: "f32[208]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_202: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_778);  convolution_161 = unsqueeze_778 = None
    mul_1262: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_202)
    sum_19: "f32[208]" = torch.ops.aten.sum.dim_IntList(mul_1262, [0, 2, 3]);  mul_1262 = None
    mul_1263: "f32[208]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_779: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1263, 0);  mul_1263 = None
    unsqueeze_780: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
    unsqueeze_781: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
    mul_1264: "f32[208]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_1265: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_484, squeeze_484)
    mul_1266: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1264, mul_1265);  mul_1264 = mul_1265 = None
    unsqueeze_782: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1266, 0);  mul_1266 = None
    unsqueeze_783: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
    unsqueeze_784: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
    mul_1267: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_484, primals_485);  primals_485 = None
    unsqueeze_785: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1267, 0);  mul_1267 = None
    unsqueeze_786: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    mul_1268: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_784);  sub_202 = unsqueeze_784 = None
    sub_204: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_1268);  where_8 = mul_1268 = None
    sub_205: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(sub_204, unsqueeze_781);  sub_204 = unsqueeze_781 = None
    mul_1269: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_787);  sub_205 = unsqueeze_787 = None
    mul_1270: "f32[208]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_484);  sum_19 = squeeze_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_1269, getitem_948, primals_484, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1269 = getitem_948 = primals_484 = None
    getitem_1026: "f32[8, 208, 7, 7]" = convolution_backward_8[0]
    getitem_1027: "f32[208, 208, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_34: "f32[8, 832, 7, 7]" = torch.ops.aten.cat.default([getitem_1026, getitem_1023, getitem_1020, slice_8], 1);  getitem_1026 = getitem_1023 = getitem_1020 = slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_9: "f32[8, 832, 7, 7]" = torch.ops.aten.where.self(le_9, full_default, cat_34);  le_9 = cat_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_20: "f32[832]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_206: "f32[8, 832, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_790);  convolution_160 = unsqueeze_790 = None
    mul_1271: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_206)
    sum_21: "f32[832]" = torch.ops.aten.sum.dim_IntList(mul_1271, [0, 2, 3]);  mul_1271 = None
    mul_1272: "f32[832]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    unsqueeze_791: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_1272, 0);  mul_1272 = None
    unsqueeze_792: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
    unsqueeze_793: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
    mul_1273: "f32[832]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    mul_1274: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_481, squeeze_481)
    mul_1275: "f32[832]" = torch.ops.aten.mul.Tensor(mul_1273, mul_1274);  mul_1273 = mul_1274 = None
    unsqueeze_794: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_1275, 0);  mul_1275 = None
    unsqueeze_795: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
    unsqueeze_796: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
    mul_1276: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_481, primals_482);  primals_482 = None
    unsqueeze_797: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_1276, 0);  mul_1276 = None
    unsqueeze_798: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    mul_1277: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_796);  sub_206 = unsqueeze_796 = None
    sub_208: "f32[8, 832, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_1277);  where_9 = mul_1277 = None
    sub_209: "f32[8, 832, 7, 7]" = torch.ops.aten.sub.Tensor(sub_208, unsqueeze_793);  sub_208 = unsqueeze_793 = None
    mul_1278: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_799);  sub_209 = unsqueeze_799 = None
    mul_1279: "f32[832]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_481);  sum_21 = squeeze_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_1278, relu_155, primals_481, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1278 = primals_481 = None
    getitem_1029: "f32[8, 2048, 7, 7]" = convolution_backward_9[0]
    getitem_1030: "f32[832, 2048, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_946: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where_5, getitem_1029);  where_5 = getitem_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_10: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(relu_155, 0);  relu_155 = None
    where_10: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le_10, full_default, add_946);  le_10 = add_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    sum_22: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_210: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_159, unsqueeze_802);  convolution_159 = unsqueeze_802 = None
    mul_1280: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_210)
    sum_23: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1280, [0, 2, 3]);  mul_1280 = None
    mul_1281: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    unsqueeze_803: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1281, 0);  mul_1281 = None
    unsqueeze_804: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
    unsqueeze_805: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
    mul_1282: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    mul_1283: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_478, squeeze_478)
    mul_1284: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1282, mul_1283);  mul_1282 = mul_1283 = None
    unsqueeze_806: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1284, 0);  mul_1284 = None
    unsqueeze_807: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
    unsqueeze_808: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
    mul_1285: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_478, primals_479);  primals_479 = None
    unsqueeze_809: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1285, 0);  mul_1285 = None
    unsqueeze_810: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    mul_1286: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_808);  sub_210 = unsqueeze_808 = None
    sub_212: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_10, mul_1286);  mul_1286 = None
    sub_213: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_212, unsqueeze_805);  sub_212 = None
    mul_1287: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_811);  sub_213 = unsqueeze_811 = None
    mul_1288: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_478);  sum_23 = squeeze_478 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_1287, relu_150, primals_478, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1287 = primals_478 = None
    getitem_1032: "f32[8, 1024, 14, 14]" = convolution_backward_10[0]
    getitem_1033: "f32[2048, 1024, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sub_214: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_158, unsqueeze_814);  convolution_158 = unsqueeze_814 = None
    mul_1289: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_214)
    sum_25: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1289, [0, 2, 3]);  mul_1289 = None
    mul_1291: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    mul_1292: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_475, squeeze_475)
    mul_1293: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1291, mul_1292);  mul_1291 = mul_1292 = None
    unsqueeze_818: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1293, 0);  mul_1293 = None
    unsqueeze_819: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
    unsqueeze_820: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
    mul_1294: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_475, primals_476);  primals_476 = None
    unsqueeze_821: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1294, 0);  mul_1294 = None
    unsqueeze_822: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    mul_1295: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_820);  sub_214 = unsqueeze_820 = None
    sub_216: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_10, mul_1295);  where_10 = mul_1295 = None
    sub_217: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_805);  sub_216 = unsqueeze_805 = None
    mul_1296: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_823);  sub_217 = unsqueeze_823 = None
    mul_1297: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_475);  sum_25 = squeeze_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_1296, cat_30, primals_475, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1296 = cat_30 = primals_475 = None
    getitem_1035: "f32[8, 832, 7, 7]" = convolution_backward_11[0]
    getitem_1036: "f32[2048, 832, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_9: "f32[8, 208, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1035, 1, 0, 208)
    slice_10: "f32[8, 208, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1035, 1, 208, 416)
    slice_11: "f32[8, 208, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1035, 1, 416, 624)
    slice_12: "f32[8, 208, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1035, 1, 624, 832);  getitem_1035 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_backward: "f32[8, 208, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(slice_12, getitem_937, [3, 3], [2, 2], [1, 1], False, True, None);  slice_12 = getitem_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_11: "f32[8, 208, 7, 7]" = torch.ops.aten.where.self(le_11, full_default, slice_11);  le_11 = slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_26: "f32[208]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_218: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_826);  convolution_157 = unsqueeze_826 = None
    mul_1298: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(where_11, sub_218)
    sum_27: "f32[208]" = torch.ops.aten.sum.dim_IntList(mul_1298, [0, 2, 3]);  mul_1298 = None
    mul_1299: "f32[208]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    unsqueeze_827: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1299, 0);  mul_1299 = None
    unsqueeze_828: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
    unsqueeze_829: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
    mul_1300: "f32[208]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    mul_1301: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_472, squeeze_472)
    mul_1302: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1300, mul_1301);  mul_1300 = mul_1301 = None
    unsqueeze_830: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1302, 0);  mul_1302 = None
    unsqueeze_831: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
    unsqueeze_832: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
    mul_1303: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_472, primals_473);  primals_473 = None
    unsqueeze_833: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1303, 0);  mul_1303 = None
    unsqueeze_834: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    mul_1304: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_832);  sub_218 = unsqueeze_832 = None
    sub_220: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(where_11, mul_1304);  where_11 = mul_1304 = None
    sub_221: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(sub_220, unsqueeze_829);  sub_220 = unsqueeze_829 = None
    mul_1305: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_835);  sub_221 = unsqueeze_835 = None
    mul_1306: "f32[208]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_472);  sum_27 = squeeze_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_1305, getitem_930, primals_472, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1305 = getitem_930 = primals_472 = None
    getitem_1038: "f32[8, 208, 14, 14]" = convolution_backward_12[0]
    getitem_1039: "f32[208, 208, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_12: "f32[8, 208, 7, 7]" = torch.ops.aten.where.self(le_12, full_default, slice_10);  le_12 = slice_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_28: "f32[208]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_222: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_156, unsqueeze_838);  convolution_156 = unsqueeze_838 = None
    mul_1307: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(where_12, sub_222)
    sum_29: "f32[208]" = torch.ops.aten.sum.dim_IntList(mul_1307, [0, 2, 3]);  mul_1307 = None
    mul_1308: "f32[208]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_839: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1308, 0);  mul_1308 = None
    unsqueeze_840: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
    unsqueeze_841: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
    mul_1309: "f32[208]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_1310: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_469, squeeze_469)
    mul_1311: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1309, mul_1310);  mul_1309 = mul_1310 = None
    unsqueeze_842: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1311, 0);  mul_1311 = None
    unsqueeze_843: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
    unsqueeze_844: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
    mul_1312: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_469, primals_470);  primals_470 = None
    unsqueeze_845: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1312, 0);  mul_1312 = None
    unsqueeze_846: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    mul_1313: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_844);  sub_222 = unsqueeze_844 = None
    sub_224: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(where_12, mul_1313);  where_12 = mul_1313 = None
    sub_225: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_841);  sub_224 = unsqueeze_841 = None
    mul_1314: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_847);  sub_225 = unsqueeze_847 = None
    mul_1315: "f32[208]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_469);  sum_29 = squeeze_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_1314, getitem_923, primals_469, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1314 = getitem_923 = primals_469 = None
    getitem_1041: "f32[8, 208, 14, 14]" = convolution_backward_13[0]
    getitem_1042: "f32[208, 208, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_13: "f32[8, 208, 7, 7]" = torch.ops.aten.where.self(le_13, full_default, slice_9);  le_13 = slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_30: "f32[208]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_226: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_850);  convolution_155 = unsqueeze_850 = None
    mul_1316: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, sub_226)
    sum_31: "f32[208]" = torch.ops.aten.sum.dim_IntList(mul_1316, [0, 2, 3]);  mul_1316 = None
    mul_1317: "f32[208]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_851: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1317, 0);  mul_1317 = None
    unsqueeze_852: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 2);  unsqueeze_851 = None
    unsqueeze_853: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 3);  unsqueeze_852 = None
    mul_1318: "f32[208]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_1319: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_466, squeeze_466)
    mul_1320: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1318, mul_1319);  mul_1318 = mul_1319 = None
    unsqueeze_854: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1320, 0);  mul_1320 = None
    unsqueeze_855: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 2);  unsqueeze_854 = None
    unsqueeze_856: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 3);  unsqueeze_855 = None
    mul_1321: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_466, primals_467);  primals_467 = None
    unsqueeze_857: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_1321, 0);  mul_1321 = None
    unsqueeze_858: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    mul_1322: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_856);  sub_226 = unsqueeze_856 = None
    sub_228: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(where_13, mul_1322);  where_13 = mul_1322 = None
    sub_229: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(sub_228, unsqueeze_853);  sub_228 = unsqueeze_853 = None
    mul_1323: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_859);  sub_229 = unsqueeze_859 = None
    mul_1324: "f32[208]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_466);  sum_31 = squeeze_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_1323, getitem_916, primals_466, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1323 = getitem_916 = primals_466 = None
    getitem_1044: "f32[8, 208, 14, 14]" = convolution_backward_14[0]
    getitem_1045: "f32[208, 208, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_35: "f32[8, 832, 14, 14]" = torch.ops.aten.cat.default([getitem_1044, getitem_1041, getitem_1038, avg_pool2d_backward], 1);  getitem_1044 = getitem_1041 = getitem_1038 = avg_pool2d_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_14: "f32[8, 832, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, cat_35);  le_14 = cat_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_32: "f32[832]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_230: "f32[8, 832, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_862);  convolution_154 = unsqueeze_862 = None
    mul_1325: "f32[8, 832, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_230)
    sum_33: "f32[832]" = torch.ops.aten.sum.dim_IntList(mul_1325, [0, 2, 3]);  mul_1325 = None
    mul_1326: "f32[832]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_863: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_1326, 0);  mul_1326 = None
    unsqueeze_864: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 2);  unsqueeze_863 = None
    unsqueeze_865: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 3);  unsqueeze_864 = None
    mul_1327: "f32[832]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_1328: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_463, squeeze_463)
    mul_1329: "f32[832]" = torch.ops.aten.mul.Tensor(mul_1327, mul_1328);  mul_1327 = mul_1328 = None
    unsqueeze_866: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_1329, 0);  mul_1329 = None
    unsqueeze_867: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 2);  unsqueeze_866 = None
    unsqueeze_868: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 3);  unsqueeze_867 = None
    mul_1330: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_463, primals_464);  primals_464 = None
    unsqueeze_869: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_1330, 0);  mul_1330 = None
    unsqueeze_870: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    mul_1331: "f32[8, 832, 14, 14]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_868);  sub_230 = unsqueeze_868 = None
    sub_232: "f32[8, 832, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_1331);  where_14 = mul_1331 = None
    sub_233: "f32[8, 832, 14, 14]" = torch.ops.aten.sub.Tensor(sub_232, unsqueeze_865);  sub_232 = unsqueeze_865 = None
    mul_1332: "f32[8, 832, 14, 14]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_871);  sub_233 = unsqueeze_871 = None
    mul_1333: "f32[832]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_463);  sum_33 = squeeze_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_1332, relu_150, primals_463, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1332 = primals_463 = None
    getitem_1047: "f32[8, 1024, 14, 14]" = convolution_backward_15[0]
    getitem_1048: "f32[832, 1024, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_947: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(getitem_1032, getitem_1047);  getitem_1032 = getitem_1047 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_15: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_150, 0);  relu_150 = None
    where_15: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, add_947);  le_15 = add_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_234: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_153, unsqueeze_874);  convolution_153 = unsqueeze_874 = None
    mul_1334: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_234)
    sum_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1334, [0, 2, 3]);  mul_1334 = None
    mul_1335: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    unsqueeze_875: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1335, 0);  mul_1335 = None
    unsqueeze_876: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 2);  unsqueeze_875 = None
    unsqueeze_877: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 3);  unsqueeze_876 = None
    mul_1336: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    mul_1337: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_460, squeeze_460)
    mul_1338: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1336, mul_1337);  mul_1336 = mul_1337 = None
    unsqueeze_878: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1338, 0);  mul_1338 = None
    unsqueeze_879: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 2);  unsqueeze_878 = None
    unsqueeze_880: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 3);  unsqueeze_879 = None
    mul_1339: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_460, primals_461);  primals_461 = None
    unsqueeze_881: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1339, 0);  mul_1339 = None
    unsqueeze_882: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    mul_1340: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_880);  sub_234 = unsqueeze_880 = None
    sub_236: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_1340);  mul_1340 = None
    sub_237: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_236, unsqueeze_877);  sub_236 = unsqueeze_877 = None
    mul_1341: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_883);  sub_237 = unsqueeze_883 = None
    mul_1342: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_460);  sum_35 = squeeze_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_1341, cat_29, primals_460, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1341 = cat_29 = primals_460 = None
    getitem_1050: "f32[8, 416, 14, 14]" = convolution_backward_16[0]
    getitem_1051: "f32[1024, 416, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_13: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1050, 1, 0, 104)
    slice_14: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1050, 1, 104, 208)
    slice_15: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1050, 1, 208, 312)
    slice_16: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1050, 1, 312, 416);  getitem_1050 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_16: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, slice_15);  le_16 = slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_36: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_238: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_886);  convolution_152 = unsqueeze_886 = None
    mul_1343: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_238)
    sum_37: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1343, [0, 2, 3]);  mul_1343 = None
    mul_1344: "f32[104]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    unsqueeze_887: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1344, 0);  mul_1344 = None
    unsqueeze_888: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 2);  unsqueeze_887 = None
    unsqueeze_889: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 3);  unsqueeze_888 = None
    mul_1345: "f32[104]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    mul_1346: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_457, squeeze_457)
    mul_1347: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1345, mul_1346);  mul_1345 = mul_1346 = None
    unsqueeze_890: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1347, 0);  mul_1347 = None
    unsqueeze_891: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 2);  unsqueeze_890 = None
    unsqueeze_892: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 3);  unsqueeze_891 = None
    mul_1348: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_457, primals_458);  primals_458 = None
    unsqueeze_893: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1348, 0);  mul_1348 = None
    unsqueeze_894: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    mul_1349: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_892);  sub_238 = unsqueeze_892 = None
    sub_240: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_1349);  where_16 = mul_1349 = None
    sub_241: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_240, unsqueeze_889);  sub_240 = unsqueeze_889 = None
    mul_1350: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_895);  sub_241 = unsqueeze_895 = None
    mul_1351: "f32[104]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_457);  sum_37 = squeeze_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_1350, add_842, primals_457, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1350 = add_842 = primals_457 = None
    getitem_1053: "f32[8, 104, 14, 14]" = convolution_backward_17[0]
    getitem_1054: "f32[104, 104, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_948: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_14, getitem_1053);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_17: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_17, full_default, add_948);  le_17 = add_948 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_38: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_242: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_898);  convolution_151 = unsqueeze_898 = None
    mul_1352: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_242)
    sum_39: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1352, [0, 2, 3]);  mul_1352 = None
    mul_1353: "f32[104]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    unsqueeze_899: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1353, 0);  mul_1353 = None
    unsqueeze_900: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 2);  unsqueeze_899 = None
    unsqueeze_901: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 3);  unsqueeze_900 = None
    mul_1354: "f32[104]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    mul_1355: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_454, squeeze_454)
    mul_1356: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1354, mul_1355);  mul_1354 = mul_1355 = None
    unsqueeze_902: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1356, 0);  mul_1356 = None
    unsqueeze_903: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 2);  unsqueeze_902 = None
    unsqueeze_904: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 3);  unsqueeze_903 = None
    mul_1357: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_454, primals_455);  primals_455 = None
    unsqueeze_905: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1357, 0);  mul_1357 = None
    unsqueeze_906: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    mul_1358: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_904);  sub_242 = unsqueeze_904 = None
    sub_244: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_1358);  where_17 = mul_1358 = None
    sub_245: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_244, unsqueeze_901);  sub_244 = unsqueeze_901 = None
    mul_1359: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_907);  sub_245 = unsqueeze_907 = None
    mul_1360: "f32[104]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_454);  sum_39 = squeeze_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_1359, add_836, primals_454, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1359 = add_836 = primals_454 = None
    getitem_1056: "f32[8, 104, 14, 14]" = convolution_backward_18[0]
    getitem_1057: "f32[104, 104, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_949: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_13, getitem_1056);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_18: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, add_949);  le_18 = add_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_40: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_246: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_910);  convolution_150 = unsqueeze_910 = None
    mul_1361: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_246)
    sum_41: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1361, [0, 2, 3]);  mul_1361 = None
    mul_1362: "f32[104]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    unsqueeze_911: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1362, 0);  mul_1362 = None
    unsqueeze_912: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 2);  unsqueeze_911 = None
    unsqueeze_913: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 3);  unsqueeze_912 = None
    mul_1363: "f32[104]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    mul_1364: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_451, squeeze_451)
    mul_1365: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1363, mul_1364);  mul_1363 = mul_1364 = None
    unsqueeze_914: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1365, 0);  mul_1365 = None
    unsqueeze_915: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 2);  unsqueeze_914 = None
    unsqueeze_916: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 3);  unsqueeze_915 = None
    mul_1366: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_451, primals_452);  primals_452 = None
    unsqueeze_917: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1366, 0);  mul_1366 = None
    unsqueeze_918: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    mul_1367: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_916);  sub_246 = unsqueeze_916 = None
    sub_248: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_1367);  where_18 = mul_1367 = None
    sub_249: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_248, unsqueeze_913);  sub_248 = unsqueeze_913 = None
    mul_1368: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_919);  sub_249 = unsqueeze_919 = None
    mul_1369: "f32[104]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_451);  sum_41 = squeeze_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_1368, getitem_886, primals_451, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1368 = getitem_886 = primals_451 = None
    getitem_1059: "f32[8, 104, 14, 14]" = convolution_backward_19[0]
    getitem_1060: "f32[104, 104, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_36: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1059, getitem_1056, getitem_1053, slice_16], 1);  getitem_1059 = getitem_1056 = getitem_1053 = slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_19: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, cat_36);  le_19 = cat_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_42: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_250: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_149, unsqueeze_922);  convolution_149 = unsqueeze_922 = None
    mul_1370: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_250)
    sum_43: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1370, [0, 2, 3]);  mul_1370 = None
    mul_1371: "f32[416]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    unsqueeze_923: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1371, 0);  mul_1371 = None
    unsqueeze_924: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 2);  unsqueeze_923 = None
    unsqueeze_925: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 3);  unsqueeze_924 = None
    mul_1372: "f32[416]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    mul_1373: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_448, squeeze_448)
    mul_1374: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1372, mul_1373);  mul_1372 = mul_1373 = None
    unsqueeze_926: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1374, 0);  mul_1374 = None
    unsqueeze_927: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 2);  unsqueeze_926 = None
    unsqueeze_928: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 3);  unsqueeze_927 = None
    mul_1375: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_448, primals_449);  primals_449 = None
    unsqueeze_929: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1375, 0);  mul_1375 = None
    unsqueeze_930: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
    unsqueeze_931: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
    mul_1376: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_928);  sub_250 = unsqueeze_928 = None
    sub_252: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_1376);  where_19 = mul_1376 = None
    sub_253: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_252, unsqueeze_925);  sub_252 = unsqueeze_925 = None
    mul_1377: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_931);  sub_253 = unsqueeze_931 = None
    mul_1378: "f32[416]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_448);  sum_43 = squeeze_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_1377, relu_145, primals_448, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1377 = primals_448 = None
    getitem_1062: "f32[8, 1024, 14, 14]" = convolution_backward_20[0]
    getitem_1063: "f32[416, 1024, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_950: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_15, getitem_1062);  where_15 = getitem_1062 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_20: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_145, 0);  relu_145 = None
    where_20: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_20, full_default, add_950);  le_20 = add_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_44: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_254: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_148, unsqueeze_934);  convolution_148 = unsqueeze_934 = None
    mul_1379: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_254)
    sum_45: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1379, [0, 2, 3]);  mul_1379 = None
    mul_1380: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    unsqueeze_935: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1380, 0);  mul_1380 = None
    unsqueeze_936: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 2);  unsqueeze_935 = None
    unsqueeze_937: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 3);  unsqueeze_936 = None
    mul_1381: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    mul_1382: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_445, squeeze_445)
    mul_1383: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1381, mul_1382);  mul_1381 = mul_1382 = None
    unsqueeze_938: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1383, 0);  mul_1383 = None
    unsqueeze_939: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 2);  unsqueeze_938 = None
    unsqueeze_940: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 3);  unsqueeze_939 = None
    mul_1384: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_445, primals_446);  primals_446 = None
    unsqueeze_941: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1384, 0);  mul_1384 = None
    unsqueeze_942: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
    unsqueeze_943: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
    mul_1385: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_940);  sub_254 = unsqueeze_940 = None
    sub_256: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_1385);  mul_1385 = None
    sub_257: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_256, unsqueeze_937);  sub_256 = unsqueeze_937 = None
    mul_1386: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_943);  sub_257 = unsqueeze_943 = None
    mul_1387: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_445);  sum_45 = squeeze_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_1386, cat_28, primals_445, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1386 = cat_28 = primals_445 = None
    getitem_1065: "f32[8, 416, 14, 14]" = convolution_backward_21[0]
    getitem_1066: "f32[1024, 416, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_17: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1065, 1, 0, 104)
    slice_18: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1065, 1, 104, 208)
    slice_19: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1065, 1, 208, 312)
    slice_20: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1065, 1, 312, 416);  getitem_1065 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_21: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_21, full_default, slice_19);  le_21 = slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_46: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_258: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_946);  convolution_147 = unsqueeze_946 = None
    mul_1388: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_258)
    sum_47: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1388, [0, 2, 3]);  mul_1388 = None
    mul_1389: "f32[104]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    unsqueeze_947: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1389, 0);  mul_1389 = None
    unsqueeze_948: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 2);  unsqueeze_947 = None
    unsqueeze_949: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 3);  unsqueeze_948 = None
    mul_1390: "f32[104]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_1391: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_442, squeeze_442)
    mul_1392: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1390, mul_1391);  mul_1390 = mul_1391 = None
    unsqueeze_950: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1392, 0);  mul_1392 = None
    unsqueeze_951: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 2);  unsqueeze_950 = None
    unsqueeze_952: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 3);  unsqueeze_951 = None
    mul_1393: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_442, primals_443);  primals_443 = None
    unsqueeze_953: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1393, 0);  mul_1393 = None
    unsqueeze_954: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
    unsqueeze_955: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
    mul_1394: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_952);  sub_258 = unsqueeze_952 = None
    sub_260: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_1394);  where_21 = mul_1394 = None
    sub_261: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_260, unsqueeze_949);  sub_260 = unsqueeze_949 = None
    mul_1395: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_955);  sub_261 = unsqueeze_955 = None
    mul_1396: "f32[104]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_442);  sum_47 = squeeze_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_1395, add_814, primals_442, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1395 = add_814 = primals_442 = None
    getitem_1068: "f32[8, 104, 14, 14]" = convolution_backward_22[0]
    getitem_1069: "f32[104, 104, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_951: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_18, getitem_1068);  slice_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_22: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_22, full_default, add_951);  le_22 = add_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_48: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_262: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_958);  convolution_146 = unsqueeze_958 = None
    mul_1397: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_262)
    sum_49: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1397, [0, 2, 3]);  mul_1397 = None
    mul_1398: "f32[104]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_959: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1398, 0);  mul_1398 = None
    unsqueeze_960: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 2);  unsqueeze_959 = None
    unsqueeze_961: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 3);  unsqueeze_960 = None
    mul_1399: "f32[104]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_1400: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_439, squeeze_439)
    mul_1401: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1399, mul_1400);  mul_1399 = mul_1400 = None
    unsqueeze_962: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1401, 0);  mul_1401 = None
    unsqueeze_963: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 2);  unsqueeze_962 = None
    unsqueeze_964: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 3);  unsqueeze_963 = None
    mul_1402: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_439, primals_440);  primals_440 = None
    unsqueeze_965: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1402, 0);  mul_1402 = None
    unsqueeze_966: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
    unsqueeze_967: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
    mul_1403: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_964);  sub_262 = unsqueeze_964 = None
    sub_264: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_22, mul_1403);  where_22 = mul_1403 = None
    sub_265: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_264, unsqueeze_961);  sub_264 = unsqueeze_961 = None
    mul_1404: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_967);  sub_265 = unsqueeze_967 = None
    mul_1405: "f32[104]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_439);  sum_49 = squeeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_1404, add_808, primals_439, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1404 = add_808 = primals_439 = None
    getitem_1071: "f32[8, 104, 14, 14]" = convolution_backward_23[0]
    getitem_1072: "f32[104, 104, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_952: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_17, getitem_1071);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_23: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_23, full_default, add_952);  le_23 = add_952 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_50: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_266: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_970);  convolution_145 = unsqueeze_970 = None
    mul_1406: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_266)
    sum_51: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1406, [0, 2, 3]);  mul_1406 = None
    mul_1407: "f32[104]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    unsqueeze_971: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1407, 0);  mul_1407 = None
    unsqueeze_972: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 2);  unsqueeze_971 = None
    unsqueeze_973: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 3);  unsqueeze_972 = None
    mul_1408: "f32[104]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_1409: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_436, squeeze_436)
    mul_1410: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1408, mul_1409);  mul_1408 = mul_1409 = None
    unsqueeze_974: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1410, 0);  mul_1410 = None
    unsqueeze_975: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 2);  unsqueeze_974 = None
    unsqueeze_976: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 3);  unsqueeze_975 = None
    mul_1411: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_436, primals_437);  primals_437 = None
    unsqueeze_977: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1411, 0);  mul_1411 = None
    unsqueeze_978: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    mul_1412: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_976);  sub_266 = unsqueeze_976 = None
    sub_268: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_1412);  where_23 = mul_1412 = None
    sub_269: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_268, unsqueeze_973);  sub_268 = unsqueeze_973 = None
    mul_1413: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_979);  sub_269 = unsqueeze_979 = None
    mul_1414: "f32[104]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_436);  sum_51 = squeeze_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_1413, getitem_856, primals_436, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1413 = getitem_856 = primals_436 = None
    getitem_1074: "f32[8, 104, 14, 14]" = convolution_backward_24[0]
    getitem_1075: "f32[104, 104, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_37: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1074, getitem_1071, getitem_1068, slice_20], 1);  getitem_1074 = getitem_1071 = getitem_1068 = slice_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_24: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_24, full_default, cat_37);  le_24 = cat_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_52: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_270: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_144, unsqueeze_982);  convolution_144 = unsqueeze_982 = None
    mul_1415: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_270)
    sum_53: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1415, [0, 2, 3]);  mul_1415 = None
    mul_1416: "f32[416]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    unsqueeze_983: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1416, 0);  mul_1416 = None
    unsqueeze_984: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 2);  unsqueeze_983 = None
    unsqueeze_985: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 3);  unsqueeze_984 = None
    mul_1417: "f32[416]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    mul_1418: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_433, squeeze_433)
    mul_1419: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1417, mul_1418);  mul_1417 = mul_1418 = None
    unsqueeze_986: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1419, 0);  mul_1419 = None
    unsqueeze_987: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 2);  unsqueeze_986 = None
    unsqueeze_988: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 3);  unsqueeze_987 = None
    mul_1420: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_433, primals_434);  primals_434 = None
    unsqueeze_989: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1420, 0);  mul_1420 = None
    unsqueeze_990: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
    unsqueeze_991: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
    mul_1421: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_988);  sub_270 = unsqueeze_988 = None
    sub_272: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_1421);  where_24 = mul_1421 = None
    sub_273: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_272, unsqueeze_985);  sub_272 = unsqueeze_985 = None
    mul_1422: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_991);  sub_273 = unsqueeze_991 = None
    mul_1423: "f32[416]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_433);  sum_53 = squeeze_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_1422, relu_140, primals_433, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1422 = primals_433 = None
    getitem_1077: "f32[8, 1024, 14, 14]" = convolution_backward_25[0]
    getitem_1078: "f32[416, 1024, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_953: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_20, getitem_1077);  where_20 = getitem_1077 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_25: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_140, 0);  relu_140 = None
    where_25: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_25, full_default, add_953);  le_25 = add_953 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_54: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_274: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_143, unsqueeze_994);  convolution_143 = unsqueeze_994 = None
    mul_1424: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_274)
    sum_55: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1424, [0, 2, 3]);  mul_1424 = None
    mul_1425: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    unsqueeze_995: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1425, 0);  mul_1425 = None
    unsqueeze_996: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 2);  unsqueeze_995 = None
    unsqueeze_997: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 3);  unsqueeze_996 = None
    mul_1426: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    mul_1427: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_430, squeeze_430)
    mul_1428: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1426, mul_1427);  mul_1426 = mul_1427 = None
    unsqueeze_998: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1428, 0);  mul_1428 = None
    unsqueeze_999: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 2);  unsqueeze_998 = None
    unsqueeze_1000: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 3);  unsqueeze_999 = None
    mul_1429: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_430, primals_431);  primals_431 = None
    unsqueeze_1001: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1429, 0);  mul_1429 = None
    unsqueeze_1002: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 2);  unsqueeze_1001 = None
    unsqueeze_1003: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 3);  unsqueeze_1002 = None
    mul_1430: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_1000);  sub_274 = unsqueeze_1000 = None
    sub_276: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_1430);  mul_1430 = None
    sub_277: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_276, unsqueeze_997);  sub_276 = unsqueeze_997 = None
    mul_1431: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_1003);  sub_277 = unsqueeze_1003 = None
    mul_1432: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_430);  sum_55 = squeeze_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_1431, cat_27, primals_430, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1431 = cat_27 = primals_430 = None
    getitem_1080: "f32[8, 416, 14, 14]" = convolution_backward_26[0]
    getitem_1081: "f32[1024, 416, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_21: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1080, 1, 0, 104)
    slice_22: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1080, 1, 104, 208)
    slice_23: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1080, 1, 208, 312)
    slice_24: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1080, 1, 312, 416);  getitem_1080 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_26: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_26, full_default, slice_23);  le_26 = slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_56: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_278: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_1006);  convolution_142 = unsqueeze_1006 = None
    mul_1433: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_278)
    sum_57: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1433, [0, 2, 3]);  mul_1433 = None
    mul_1434: "f32[104]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_1007: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1434, 0);  mul_1434 = None
    unsqueeze_1008: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 2);  unsqueeze_1007 = None
    unsqueeze_1009: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 3);  unsqueeze_1008 = None
    mul_1435: "f32[104]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_1436: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_427, squeeze_427)
    mul_1437: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1435, mul_1436);  mul_1435 = mul_1436 = None
    unsqueeze_1010: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1437, 0);  mul_1437 = None
    unsqueeze_1011: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 2);  unsqueeze_1010 = None
    unsqueeze_1012: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 3);  unsqueeze_1011 = None
    mul_1438: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_427, primals_428);  primals_428 = None
    unsqueeze_1013: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1438, 0);  mul_1438 = None
    unsqueeze_1014: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 2);  unsqueeze_1013 = None
    unsqueeze_1015: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 3);  unsqueeze_1014 = None
    mul_1439: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_1012);  sub_278 = unsqueeze_1012 = None
    sub_280: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_26, mul_1439);  where_26 = mul_1439 = None
    sub_281: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_280, unsqueeze_1009);  sub_280 = unsqueeze_1009 = None
    mul_1440: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_1015);  sub_281 = unsqueeze_1015 = None
    mul_1441: "f32[104]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_427);  sum_57 = squeeze_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_1440, add_786, primals_427, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1440 = add_786 = primals_427 = None
    getitem_1083: "f32[8, 104, 14, 14]" = convolution_backward_27[0]
    getitem_1084: "f32[104, 104, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_954: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_22, getitem_1083);  slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_27: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_27, full_default, add_954);  le_27 = add_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_58: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_282: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_1018);  convolution_141 = unsqueeze_1018 = None
    mul_1442: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_282)
    sum_59: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1442, [0, 2, 3]);  mul_1442 = None
    mul_1443: "f32[104]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_1019: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1443, 0);  mul_1443 = None
    unsqueeze_1020: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 2);  unsqueeze_1019 = None
    unsqueeze_1021: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 3);  unsqueeze_1020 = None
    mul_1444: "f32[104]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_1445: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_424, squeeze_424)
    mul_1446: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1444, mul_1445);  mul_1444 = mul_1445 = None
    unsqueeze_1022: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1446, 0);  mul_1446 = None
    unsqueeze_1023: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 2);  unsqueeze_1022 = None
    unsqueeze_1024: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 3);  unsqueeze_1023 = None
    mul_1447: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_424, primals_425);  primals_425 = None
    unsqueeze_1025: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1447, 0);  mul_1447 = None
    unsqueeze_1026: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 2);  unsqueeze_1025 = None
    unsqueeze_1027: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 3);  unsqueeze_1026 = None
    mul_1448: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_1024);  sub_282 = unsqueeze_1024 = None
    sub_284: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_1448);  where_27 = mul_1448 = None
    sub_285: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_284, unsqueeze_1021);  sub_284 = unsqueeze_1021 = None
    mul_1449: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_1027);  sub_285 = unsqueeze_1027 = None
    mul_1450: "f32[104]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_424);  sum_59 = squeeze_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_1449, add_780, primals_424, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1449 = add_780 = primals_424 = None
    getitem_1086: "f32[8, 104, 14, 14]" = convolution_backward_28[0]
    getitem_1087: "f32[104, 104, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_955: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_21, getitem_1086);  slice_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_28: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_28, full_default, add_955);  le_28 = add_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_60: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_286: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_1030);  convolution_140 = unsqueeze_1030 = None
    mul_1451: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_286)
    sum_61: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1451, [0, 2, 3]);  mul_1451 = None
    mul_1452: "f32[104]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_1031: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1452, 0);  mul_1452 = None
    unsqueeze_1032: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 2);  unsqueeze_1031 = None
    unsqueeze_1033: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 3);  unsqueeze_1032 = None
    mul_1453: "f32[104]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_1454: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_421, squeeze_421)
    mul_1455: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1453, mul_1454);  mul_1453 = mul_1454 = None
    unsqueeze_1034: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1455, 0);  mul_1455 = None
    unsqueeze_1035: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 2);  unsqueeze_1034 = None
    unsqueeze_1036: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 3);  unsqueeze_1035 = None
    mul_1456: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_421, primals_422);  primals_422 = None
    unsqueeze_1037: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1456, 0);  mul_1456 = None
    unsqueeze_1038: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 2);  unsqueeze_1037 = None
    unsqueeze_1039: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 3);  unsqueeze_1038 = None
    mul_1457: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_1036);  sub_286 = unsqueeze_1036 = None
    sub_288: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_28, mul_1457);  where_28 = mul_1457 = None
    sub_289: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_288, unsqueeze_1033);  sub_288 = unsqueeze_1033 = None
    mul_1458: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_1039);  sub_289 = unsqueeze_1039 = None
    mul_1459: "f32[104]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_421);  sum_61 = squeeze_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_1458, getitem_826, primals_421, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1458 = getitem_826 = primals_421 = None
    getitem_1089: "f32[8, 104, 14, 14]" = convolution_backward_29[0]
    getitem_1090: "f32[104, 104, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_38: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1089, getitem_1086, getitem_1083, slice_24], 1);  getitem_1089 = getitem_1086 = getitem_1083 = slice_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_29: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_29, full_default, cat_38);  le_29 = cat_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_62: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_290: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_1042);  convolution_139 = unsqueeze_1042 = None
    mul_1460: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_29, sub_290)
    sum_63: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1460, [0, 2, 3]);  mul_1460 = None
    mul_1461: "f32[416]" = torch.ops.aten.mul.Tensor(sum_62, 0.0006377551020408163)
    unsqueeze_1043: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1461, 0);  mul_1461 = None
    unsqueeze_1044: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 2);  unsqueeze_1043 = None
    unsqueeze_1045: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, 3);  unsqueeze_1044 = None
    mul_1462: "f32[416]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    mul_1463: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_418, squeeze_418)
    mul_1464: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1462, mul_1463);  mul_1462 = mul_1463 = None
    unsqueeze_1046: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1464, 0);  mul_1464 = None
    unsqueeze_1047: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 2);  unsqueeze_1046 = None
    unsqueeze_1048: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 3);  unsqueeze_1047 = None
    mul_1465: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_418, primals_419);  primals_419 = None
    unsqueeze_1049: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1465, 0);  mul_1465 = None
    unsqueeze_1050: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 2);  unsqueeze_1049 = None
    unsqueeze_1051: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 3);  unsqueeze_1050 = None
    mul_1466: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_1048);  sub_290 = unsqueeze_1048 = None
    sub_292: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_29, mul_1466);  where_29 = mul_1466 = None
    sub_293: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_292, unsqueeze_1045);  sub_292 = unsqueeze_1045 = None
    mul_1467: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_1051);  sub_293 = unsqueeze_1051 = None
    mul_1468: "f32[416]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_418);  sum_63 = squeeze_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_1467, relu_135, primals_418, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1467 = primals_418 = None
    getitem_1092: "f32[8, 1024, 14, 14]" = convolution_backward_30[0]
    getitem_1093: "f32[416, 1024, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_956: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_25, getitem_1092);  where_25 = getitem_1092 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_30: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_135, 0);  relu_135 = None
    where_30: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_30, full_default, add_956);  le_30 = add_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_64: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_294: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_1054);  convolution_138 = unsqueeze_1054 = None
    mul_1469: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_294)
    sum_65: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1469, [0, 2, 3]);  mul_1469 = None
    mul_1470: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    unsqueeze_1055: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1470, 0);  mul_1470 = None
    unsqueeze_1056: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 2);  unsqueeze_1055 = None
    unsqueeze_1057: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, 3);  unsqueeze_1056 = None
    mul_1471: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    mul_1472: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_415, squeeze_415)
    mul_1473: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1471, mul_1472);  mul_1471 = mul_1472 = None
    unsqueeze_1058: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1473, 0);  mul_1473 = None
    unsqueeze_1059: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 2);  unsqueeze_1058 = None
    unsqueeze_1060: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 3);  unsqueeze_1059 = None
    mul_1474: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_415, primals_416);  primals_416 = None
    unsqueeze_1061: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1474, 0);  mul_1474 = None
    unsqueeze_1062: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 2);  unsqueeze_1061 = None
    unsqueeze_1063: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 3);  unsqueeze_1062 = None
    mul_1475: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_1060);  sub_294 = unsqueeze_1060 = None
    sub_296: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_30, mul_1475);  mul_1475 = None
    sub_297: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_296, unsqueeze_1057);  sub_296 = unsqueeze_1057 = None
    mul_1476: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_1063);  sub_297 = unsqueeze_1063 = None
    mul_1477: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_415);  sum_65 = squeeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_1476, cat_26, primals_415, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1476 = cat_26 = primals_415 = None
    getitem_1095: "f32[8, 416, 14, 14]" = convolution_backward_31[0]
    getitem_1096: "f32[1024, 416, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_25: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1095, 1, 0, 104)
    slice_26: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1095, 1, 104, 208)
    slice_27: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1095, 1, 208, 312)
    slice_28: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1095, 1, 312, 416);  getitem_1095 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_31: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_31, full_default, slice_27);  le_31 = slice_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_66: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_298: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_1066);  convolution_137 = unsqueeze_1066 = None
    mul_1478: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_298)
    sum_67: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1478, [0, 2, 3]);  mul_1478 = None
    mul_1479: "f32[104]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    unsqueeze_1067: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1479, 0);  mul_1479 = None
    unsqueeze_1068: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 2);  unsqueeze_1067 = None
    unsqueeze_1069: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, 3);  unsqueeze_1068 = None
    mul_1480: "f32[104]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    mul_1481: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_412, squeeze_412)
    mul_1482: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1480, mul_1481);  mul_1480 = mul_1481 = None
    unsqueeze_1070: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1482, 0);  mul_1482 = None
    unsqueeze_1071: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 2);  unsqueeze_1070 = None
    unsqueeze_1072: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 3);  unsqueeze_1071 = None
    mul_1483: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_412, primals_413);  primals_413 = None
    unsqueeze_1073: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1483, 0);  mul_1483 = None
    unsqueeze_1074: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 2);  unsqueeze_1073 = None
    unsqueeze_1075: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 3);  unsqueeze_1074 = None
    mul_1484: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_1072);  sub_298 = unsqueeze_1072 = None
    sub_300: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_1484);  where_31 = mul_1484 = None
    sub_301: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_300, unsqueeze_1069);  sub_300 = unsqueeze_1069 = None
    mul_1485: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_1075);  sub_301 = unsqueeze_1075 = None
    mul_1486: "f32[104]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_412);  sum_67 = squeeze_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_1485, add_758, primals_412, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1485 = add_758 = primals_412 = None
    getitem_1098: "f32[8, 104, 14, 14]" = convolution_backward_32[0]
    getitem_1099: "f32[104, 104, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_957: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_26, getitem_1098);  slice_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_32: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_32, full_default, add_957);  le_32 = add_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_68: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_302: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_1078);  convolution_136 = unsqueeze_1078 = None
    mul_1487: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_302)
    sum_69: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1487, [0, 2, 3]);  mul_1487 = None
    mul_1488: "f32[104]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    unsqueeze_1079: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1488, 0);  mul_1488 = None
    unsqueeze_1080: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1079, 2);  unsqueeze_1079 = None
    unsqueeze_1081: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, 3);  unsqueeze_1080 = None
    mul_1489: "f32[104]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    mul_1490: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_409, squeeze_409)
    mul_1491: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1489, mul_1490);  mul_1489 = mul_1490 = None
    unsqueeze_1082: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1491, 0);  mul_1491 = None
    unsqueeze_1083: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 2);  unsqueeze_1082 = None
    unsqueeze_1084: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 3);  unsqueeze_1083 = None
    mul_1492: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_409, primals_410);  primals_410 = None
    unsqueeze_1085: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1492, 0);  mul_1492 = None
    unsqueeze_1086: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 2);  unsqueeze_1085 = None
    unsqueeze_1087: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 3);  unsqueeze_1086 = None
    mul_1493: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_1084);  sub_302 = unsqueeze_1084 = None
    sub_304: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_32, mul_1493);  where_32 = mul_1493 = None
    sub_305: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_304, unsqueeze_1081);  sub_304 = unsqueeze_1081 = None
    mul_1494: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_1087);  sub_305 = unsqueeze_1087 = None
    mul_1495: "f32[104]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_409);  sum_69 = squeeze_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_1494, add_752, primals_409, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1494 = add_752 = primals_409 = None
    getitem_1101: "f32[8, 104, 14, 14]" = convolution_backward_33[0]
    getitem_1102: "f32[104, 104, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_958: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_25, getitem_1101);  slice_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_33: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_33, full_default, add_958);  le_33 = add_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_70: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_306: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_1090);  convolution_135 = unsqueeze_1090 = None
    mul_1496: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_306)
    sum_71: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1496, [0, 2, 3]);  mul_1496 = None
    mul_1497: "f32[104]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_1091: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1497, 0);  mul_1497 = None
    unsqueeze_1092: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1091, 2);  unsqueeze_1091 = None
    unsqueeze_1093: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, 3);  unsqueeze_1092 = None
    mul_1498: "f32[104]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_1499: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_406, squeeze_406)
    mul_1500: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1498, mul_1499);  mul_1498 = mul_1499 = None
    unsqueeze_1094: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1500, 0);  mul_1500 = None
    unsqueeze_1095: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 2);  unsqueeze_1094 = None
    unsqueeze_1096: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 3);  unsqueeze_1095 = None
    mul_1501: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_406, primals_407);  primals_407 = None
    unsqueeze_1097: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1501, 0);  mul_1501 = None
    unsqueeze_1098: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 2);  unsqueeze_1097 = None
    unsqueeze_1099: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 3);  unsqueeze_1098 = None
    mul_1502: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_1096);  sub_306 = unsqueeze_1096 = None
    sub_308: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_33, mul_1502);  where_33 = mul_1502 = None
    sub_309: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_308, unsqueeze_1093);  sub_308 = unsqueeze_1093 = None
    mul_1503: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_1099);  sub_309 = unsqueeze_1099 = None
    mul_1504: "f32[104]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_406);  sum_71 = squeeze_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_1503, getitem_796, primals_406, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1503 = getitem_796 = primals_406 = None
    getitem_1104: "f32[8, 104, 14, 14]" = convolution_backward_34[0]
    getitem_1105: "f32[104, 104, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_39: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1104, getitem_1101, getitem_1098, slice_28], 1);  getitem_1104 = getitem_1101 = getitem_1098 = slice_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_34: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_34, full_default, cat_39);  le_34 = cat_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_72: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_310: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_1102);  convolution_134 = unsqueeze_1102 = None
    mul_1505: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_310)
    sum_73: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1505, [0, 2, 3]);  mul_1505 = None
    mul_1506: "f32[416]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_1103: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1506, 0);  mul_1506 = None
    unsqueeze_1104: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1103, 2);  unsqueeze_1103 = None
    unsqueeze_1105: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, 3);  unsqueeze_1104 = None
    mul_1507: "f32[416]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_1508: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_403, squeeze_403)
    mul_1509: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1507, mul_1508);  mul_1507 = mul_1508 = None
    unsqueeze_1106: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1509, 0);  mul_1509 = None
    unsqueeze_1107: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 2);  unsqueeze_1106 = None
    unsqueeze_1108: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 3);  unsqueeze_1107 = None
    mul_1510: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_403, primals_404);  primals_404 = None
    unsqueeze_1109: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1510, 0);  mul_1510 = None
    unsqueeze_1110: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 2);  unsqueeze_1109 = None
    unsqueeze_1111: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 3);  unsqueeze_1110 = None
    mul_1511: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_1108);  sub_310 = unsqueeze_1108 = None
    sub_312: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_34, mul_1511);  where_34 = mul_1511 = None
    sub_313: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_312, unsqueeze_1105);  sub_312 = unsqueeze_1105 = None
    mul_1512: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_1111);  sub_313 = unsqueeze_1111 = None
    mul_1513: "f32[416]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_403);  sum_73 = squeeze_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_1512, relu_130, primals_403, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1512 = primals_403 = None
    getitem_1107: "f32[8, 1024, 14, 14]" = convolution_backward_35[0]
    getitem_1108: "f32[416, 1024, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_959: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_30, getitem_1107);  where_30 = getitem_1107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_35: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_130, 0);  relu_130 = None
    where_35: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_35, full_default, add_959);  le_35 = add_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_74: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_314: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_1114);  convolution_133 = unsqueeze_1114 = None
    mul_1514: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_314)
    sum_75: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1514, [0, 2, 3]);  mul_1514 = None
    mul_1515: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_1115: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1515, 0);  mul_1515 = None
    unsqueeze_1116: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1115, 2);  unsqueeze_1115 = None
    unsqueeze_1117: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, 3);  unsqueeze_1116 = None
    mul_1516: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_1517: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_400, squeeze_400)
    mul_1518: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1516, mul_1517);  mul_1516 = mul_1517 = None
    unsqueeze_1118: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1518, 0);  mul_1518 = None
    unsqueeze_1119: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 2);  unsqueeze_1118 = None
    unsqueeze_1120: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 3);  unsqueeze_1119 = None
    mul_1519: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_400, primals_401);  primals_401 = None
    unsqueeze_1121: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1519, 0);  mul_1519 = None
    unsqueeze_1122: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 2);  unsqueeze_1121 = None
    unsqueeze_1123: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 3);  unsqueeze_1122 = None
    mul_1520: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_1120);  sub_314 = unsqueeze_1120 = None
    sub_316: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_1520);  mul_1520 = None
    sub_317: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_316, unsqueeze_1117);  sub_316 = unsqueeze_1117 = None
    mul_1521: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_1123);  sub_317 = unsqueeze_1123 = None
    mul_1522: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_400);  sum_75 = squeeze_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_1521, cat_25, primals_400, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1521 = cat_25 = primals_400 = None
    getitem_1110: "f32[8, 416, 14, 14]" = convolution_backward_36[0]
    getitem_1111: "f32[1024, 416, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_29: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1110, 1, 0, 104)
    slice_30: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1110, 1, 104, 208)
    slice_31: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1110, 1, 208, 312)
    slice_32: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1110, 1, 312, 416);  getitem_1110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_36: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_36, full_default, slice_31);  le_36 = slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_76: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_318: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_1126);  convolution_132 = unsqueeze_1126 = None
    mul_1523: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_318)
    sum_77: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1523, [0, 2, 3]);  mul_1523 = None
    mul_1524: "f32[104]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    unsqueeze_1127: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1524, 0);  mul_1524 = None
    unsqueeze_1128: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1127, 2);  unsqueeze_1127 = None
    unsqueeze_1129: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, 3);  unsqueeze_1128 = None
    mul_1525: "f32[104]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    mul_1526: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_397, squeeze_397)
    mul_1527: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1525, mul_1526);  mul_1525 = mul_1526 = None
    unsqueeze_1130: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1527, 0);  mul_1527 = None
    unsqueeze_1131: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 2);  unsqueeze_1130 = None
    unsqueeze_1132: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 3);  unsqueeze_1131 = None
    mul_1528: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_397, primals_398);  primals_398 = None
    unsqueeze_1133: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1528, 0);  mul_1528 = None
    unsqueeze_1134: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 2);  unsqueeze_1133 = None
    unsqueeze_1135: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 3);  unsqueeze_1134 = None
    mul_1529: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_1132);  sub_318 = unsqueeze_1132 = None
    sub_320: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_36, mul_1529);  where_36 = mul_1529 = None
    sub_321: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_320, unsqueeze_1129);  sub_320 = unsqueeze_1129 = None
    mul_1530: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_1135);  sub_321 = unsqueeze_1135 = None
    mul_1531: "f32[104]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_397);  sum_77 = squeeze_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_1530, add_730, primals_397, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1530 = add_730 = primals_397 = None
    getitem_1113: "f32[8, 104, 14, 14]" = convolution_backward_37[0]
    getitem_1114: "f32[104, 104, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_960: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_30, getitem_1113);  slice_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_37: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_37, full_default, add_960);  le_37 = add_960 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_78: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_322: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_1138);  convolution_131 = unsqueeze_1138 = None
    mul_1532: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_322)
    sum_79: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1532, [0, 2, 3]);  mul_1532 = None
    mul_1533: "f32[104]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    unsqueeze_1139: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1533, 0);  mul_1533 = None
    unsqueeze_1140: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1139, 2);  unsqueeze_1139 = None
    unsqueeze_1141: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, 3);  unsqueeze_1140 = None
    mul_1534: "f32[104]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    mul_1535: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_394, squeeze_394)
    mul_1536: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1534, mul_1535);  mul_1534 = mul_1535 = None
    unsqueeze_1142: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1536, 0);  mul_1536 = None
    unsqueeze_1143: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 2);  unsqueeze_1142 = None
    unsqueeze_1144: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 3);  unsqueeze_1143 = None
    mul_1537: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_394, primals_395);  primals_395 = None
    unsqueeze_1145: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1537, 0);  mul_1537 = None
    unsqueeze_1146: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 2);  unsqueeze_1145 = None
    unsqueeze_1147: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 3);  unsqueeze_1146 = None
    mul_1538: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_1144);  sub_322 = unsqueeze_1144 = None
    sub_324: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_37, mul_1538);  where_37 = mul_1538 = None
    sub_325: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_324, unsqueeze_1141);  sub_324 = unsqueeze_1141 = None
    mul_1539: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_1147);  sub_325 = unsqueeze_1147 = None
    mul_1540: "f32[104]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_394);  sum_79 = squeeze_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_1539, add_724, primals_394, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1539 = add_724 = primals_394 = None
    getitem_1116: "f32[8, 104, 14, 14]" = convolution_backward_38[0]
    getitem_1117: "f32[104, 104, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_961: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_29, getitem_1116);  slice_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_38: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_38, full_default, add_961);  le_38 = add_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_80: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_326: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_1150);  convolution_130 = unsqueeze_1150 = None
    mul_1541: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_326)
    sum_81: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1541, [0, 2, 3]);  mul_1541 = None
    mul_1542: "f32[104]" = torch.ops.aten.mul.Tensor(sum_80, 0.0006377551020408163)
    unsqueeze_1151: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1542, 0);  mul_1542 = None
    unsqueeze_1152: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1151, 2);  unsqueeze_1151 = None
    unsqueeze_1153: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, 3);  unsqueeze_1152 = None
    mul_1543: "f32[104]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    mul_1544: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_391, squeeze_391)
    mul_1545: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1543, mul_1544);  mul_1543 = mul_1544 = None
    unsqueeze_1154: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1545, 0);  mul_1545 = None
    unsqueeze_1155: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 2);  unsqueeze_1154 = None
    unsqueeze_1156: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 3);  unsqueeze_1155 = None
    mul_1546: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_391, primals_392);  primals_392 = None
    unsqueeze_1157: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1546, 0);  mul_1546 = None
    unsqueeze_1158: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 2);  unsqueeze_1157 = None
    unsqueeze_1159: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 3);  unsqueeze_1158 = None
    mul_1547: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_1156);  sub_326 = unsqueeze_1156 = None
    sub_328: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_38, mul_1547);  where_38 = mul_1547 = None
    sub_329: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_328, unsqueeze_1153);  sub_328 = unsqueeze_1153 = None
    mul_1548: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_1159);  sub_329 = unsqueeze_1159 = None
    mul_1549: "f32[104]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_391);  sum_81 = squeeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_1548, getitem_766, primals_391, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1548 = getitem_766 = primals_391 = None
    getitem_1119: "f32[8, 104, 14, 14]" = convolution_backward_39[0]
    getitem_1120: "f32[104, 104, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_40: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1119, getitem_1116, getitem_1113, slice_32], 1);  getitem_1119 = getitem_1116 = getitem_1113 = slice_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_39: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_39, full_default, cat_40);  le_39 = cat_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_82: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_330: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_1162);  convolution_129 = unsqueeze_1162 = None
    mul_1550: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_330)
    sum_83: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1550, [0, 2, 3]);  mul_1550 = None
    mul_1551: "f32[416]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    unsqueeze_1163: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1551, 0);  mul_1551 = None
    unsqueeze_1164: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1163, 2);  unsqueeze_1163 = None
    unsqueeze_1165: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, 3);  unsqueeze_1164 = None
    mul_1552: "f32[416]" = torch.ops.aten.mul.Tensor(sum_83, 0.0006377551020408163)
    mul_1553: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_388, squeeze_388)
    mul_1554: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1552, mul_1553);  mul_1552 = mul_1553 = None
    unsqueeze_1166: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1554, 0);  mul_1554 = None
    unsqueeze_1167: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 2);  unsqueeze_1166 = None
    unsqueeze_1168: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 3);  unsqueeze_1167 = None
    mul_1555: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_388, primals_389);  primals_389 = None
    unsqueeze_1169: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1555, 0);  mul_1555 = None
    unsqueeze_1170: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 2);  unsqueeze_1169 = None
    unsqueeze_1171: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 3);  unsqueeze_1170 = None
    mul_1556: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_1168);  sub_330 = unsqueeze_1168 = None
    sub_332: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_39, mul_1556);  where_39 = mul_1556 = None
    sub_333: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_332, unsqueeze_1165);  sub_332 = unsqueeze_1165 = None
    mul_1557: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_1171);  sub_333 = unsqueeze_1171 = None
    mul_1558: "f32[416]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_388);  sum_83 = squeeze_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_1557, relu_125, primals_388, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1557 = primals_388 = None
    getitem_1122: "f32[8, 1024, 14, 14]" = convolution_backward_40[0]
    getitem_1123: "f32[416, 1024, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_962: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_35, getitem_1122);  where_35 = getitem_1122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_40: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_125, 0);  relu_125 = None
    where_40: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_40, full_default, add_962);  le_40 = add_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_84: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_334: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_1174);  convolution_128 = unsqueeze_1174 = None
    mul_1559: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_334)
    sum_85: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1559, [0, 2, 3]);  mul_1559 = None
    mul_1560: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    unsqueeze_1175: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1560, 0);  mul_1560 = None
    unsqueeze_1176: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1175, 2);  unsqueeze_1175 = None
    unsqueeze_1177: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, 3);  unsqueeze_1176 = None
    mul_1561: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    mul_1562: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_385, squeeze_385)
    mul_1563: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1561, mul_1562);  mul_1561 = mul_1562 = None
    unsqueeze_1178: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1563, 0);  mul_1563 = None
    unsqueeze_1179: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 2);  unsqueeze_1178 = None
    unsqueeze_1180: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 3);  unsqueeze_1179 = None
    mul_1564: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_385, primals_386);  primals_386 = None
    unsqueeze_1181: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1564, 0);  mul_1564 = None
    unsqueeze_1182: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 2);  unsqueeze_1181 = None
    unsqueeze_1183: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 3);  unsqueeze_1182 = None
    mul_1565: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_1180);  sub_334 = unsqueeze_1180 = None
    sub_336: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_1565);  mul_1565 = None
    sub_337: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_336, unsqueeze_1177);  sub_336 = unsqueeze_1177 = None
    mul_1566: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_1183);  sub_337 = unsqueeze_1183 = None
    mul_1567: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_385);  sum_85 = squeeze_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_1566, cat_24, primals_385, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1566 = cat_24 = primals_385 = None
    getitem_1125: "f32[8, 416, 14, 14]" = convolution_backward_41[0]
    getitem_1126: "f32[1024, 416, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_33: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1125, 1, 0, 104)
    slice_34: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1125, 1, 104, 208)
    slice_35: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1125, 1, 208, 312)
    slice_36: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1125, 1, 312, 416);  getitem_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_41: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_41, full_default, slice_35);  le_41 = slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_86: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_338: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_1186);  convolution_127 = unsqueeze_1186 = None
    mul_1568: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, sub_338)
    sum_87: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1568, [0, 2, 3]);  mul_1568 = None
    mul_1569: "f32[104]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006377551020408163)
    unsqueeze_1187: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1569, 0);  mul_1569 = None
    unsqueeze_1188: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1187, 2);  unsqueeze_1187 = None
    unsqueeze_1189: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, 3);  unsqueeze_1188 = None
    mul_1570: "f32[104]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006377551020408163)
    mul_1571: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_382, squeeze_382)
    mul_1572: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1570, mul_1571);  mul_1570 = mul_1571 = None
    unsqueeze_1190: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1572, 0);  mul_1572 = None
    unsqueeze_1191: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 2);  unsqueeze_1190 = None
    unsqueeze_1192: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 3);  unsqueeze_1191 = None
    mul_1573: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_382, primals_383);  primals_383 = None
    unsqueeze_1193: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1573, 0);  mul_1573 = None
    unsqueeze_1194: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 2);  unsqueeze_1193 = None
    unsqueeze_1195: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 3);  unsqueeze_1194 = None
    mul_1574: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_1192);  sub_338 = unsqueeze_1192 = None
    sub_340: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_41, mul_1574);  where_41 = mul_1574 = None
    sub_341: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_340, unsqueeze_1189);  sub_340 = unsqueeze_1189 = None
    mul_1575: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_1195);  sub_341 = unsqueeze_1195 = None
    mul_1576: "f32[104]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_382);  sum_87 = squeeze_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_1575, add_702, primals_382, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1575 = add_702 = primals_382 = None
    getitem_1128: "f32[8, 104, 14, 14]" = convolution_backward_42[0]
    getitem_1129: "f32[104, 104, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_963: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_34, getitem_1128);  slice_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_42: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_42, full_default, add_963);  le_42 = add_963 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_88: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_342: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_1198);  convolution_126 = unsqueeze_1198 = None
    mul_1577: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_342)
    sum_89: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1577, [0, 2, 3]);  mul_1577 = None
    mul_1578: "f32[104]" = torch.ops.aten.mul.Tensor(sum_88, 0.0006377551020408163)
    unsqueeze_1199: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1578, 0);  mul_1578 = None
    unsqueeze_1200: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1199, 2);  unsqueeze_1199 = None
    unsqueeze_1201: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, 3);  unsqueeze_1200 = None
    mul_1579: "f32[104]" = torch.ops.aten.mul.Tensor(sum_89, 0.0006377551020408163)
    mul_1580: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_379, squeeze_379)
    mul_1581: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1579, mul_1580);  mul_1579 = mul_1580 = None
    unsqueeze_1202: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1581, 0);  mul_1581 = None
    unsqueeze_1203: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 2);  unsqueeze_1202 = None
    unsqueeze_1204: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 3);  unsqueeze_1203 = None
    mul_1582: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_379, primals_380);  primals_380 = None
    unsqueeze_1205: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1582, 0);  mul_1582 = None
    unsqueeze_1206: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 2);  unsqueeze_1205 = None
    unsqueeze_1207: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 3);  unsqueeze_1206 = None
    mul_1583: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_1204);  sub_342 = unsqueeze_1204 = None
    sub_344: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_42, mul_1583);  where_42 = mul_1583 = None
    sub_345: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_344, unsqueeze_1201);  sub_344 = unsqueeze_1201 = None
    mul_1584: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_1207);  sub_345 = unsqueeze_1207 = None
    mul_1585: "f32[104]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_379);  sum_89 = squeeze_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1584, add_696, primals_379, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1584 = add_696 = primals_379 = None
    getitem_1131: "f32[8, 104, 14, 14]" = convolution_backward_43[0]
    getitem_1132: "f32[104, 104, 3, 3]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_964: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_33, getitem_1131);  slice_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_43: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_43, full_default, add_964);  le_43 = add_964 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_90: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_346: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_1210);  convolution_125 = unsqueeze_1210 = None
    mul_1586: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_346)
    sum_91: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1586, [0, 2, 3]);  mul_1586 = None
    mul_1587: "f32[104]" = torch.ops.aten.mul.Tensor(sum_90, 0.0006377551020408163)
    unsqueeze_1211: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1587, 0);  mul_1587 = None
    unsqueeze_1212: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1211, 2);  unsqueeze_1211 = None
    unsqueeze_1213: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, 3);  unsqueeze_1212 = None
    mul_1588: "f32[104]" = torch.ops.aten.mul.Tensor(sum_91, 0.0006377551020408163)
    mul_1589: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_376, squeeze_376)
    mul_1590: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1588, mul_1589);  mul_1588 = mul_1589 = None
    unsqueeze_1214: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1590, 0);  mul_1590 = None
    unsqueeze_1215: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 2);  unsqueeze_1214 = None
    unsqueeze_1216: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 3);  unsqueeze_1215 = None
    mul_1591: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_376, primals_377);  primals_377 = None
    unsqueeze_1217: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1591, 0);  mul_1591 = None
    unsqueeze_1218: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 2);  unsqueeze_1217 = None
    unsqueeze_1219: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 3);  unsqueeze_1218 = None
    mul_1592: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_1216);  sub_346 = unsqueeze_1216 = None
    sub_348: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_43, mul_1592);  where_43 = mul_1592 = None
    sub_349: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_348, unsqueeze_1213);  sub_348 = unsqueeze_1213 = None
    mul_1593: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_1219);  sub_349 = unsqueeze_1219 = None
    mul_1594: "f32[104]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_376);  sum_91 = squeeze_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1593, getitem_736, primals_376, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1593 = getitem_736 = primals_376 = None
    getitem_1134: "f32[8, 104, 14, 14]" = convolution_backward_44[0]
    getitem_1135: "f32[104, 104, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_41: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1134, getitem_1131, getitem_1128, slice_36], 1);  getitem_1134 = getitem_1131 = getitem_1128 = slice_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_44: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_44, full_default, cat_41);  le_44 = cat_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_92: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_350: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_1222);  convolution_124 = unsqueeze_1222 = None
    mul_1595: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_350)
    sum_93: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1595, [0, 2, 3]);  mul_1595 = None
    mul_1596: "f32[416]" = torch.ops.aten.mul.Tensor(sum_92, 0.0006377551020408163)
    unsqueeze_1223: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1596, 0);  mul_1596 = None
    unsqueeze_1224: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1223, 2);  unsqueeze_1223 = None
    unsqueeze_1225: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, 3);  unsqueeze_1224 = None
    mul_1597: "f32[416]" = torch.ops.aten.mul.Tensor(sum_93, 0.0006377551020408163)
    mul_1598: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_373, squeeze_373)
    mul_1599: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1597, mul_1598);  mul_1597 = mul_1598 = None
    unsqueeze_1226: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1599, 0);  mul_1599 = None
    unsqueeze_1227: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 2);  unsqueeze_1226 = None
    unsqueeze_1228: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 3);  unsqueeze_1227 = None
    mul_1600: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_373, primals_374);  primals_374 = None
    unsqueeze_1229: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1600, 0);  mul_1600 = None
    unsqueeze_1230: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 2);  unsqueeze_1229 = None
    unsqueeze_1231: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 3);  unsqueeze_1230 = None
    mul_1601: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_350, unsqueeze_1228);  sub_350 = unsqueeze_1228 = None
    sub_352: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_44, mul_1601);  where_44 = mul_1601 = None
    sub_353: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_352, unsqueeze_1225);  sub_352 = unsqueeze_1225 = None
    mul_1602: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_353, unsqueeze_1231);  sub_353 = unsqueeze_1231 = None
    mul_1603: "f32[416]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_373);  sum_93 = squeeze_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1602, relu_120, primals_373, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1602 = primals_373 = None
    getitem_1137: "f32[8, 1024, 14, 14]" = convolution_backward_45[0]
    getitem_1138: "f32[416, 1024, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_965: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_40, getitem_1137);  where_40 = getitem_1137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_45: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_120, 0);  relu_120 = None
    where_45: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_45, full_default, add_965);  le_45 = add_965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_94: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_354: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_1234);  convolution_123 = unsqueeze_1234 = None
    mul_1604: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, sub_354)
    sum_95: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1604, [0, 2, 3]);  mul_1604 = None
    mul_1605: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_94, 0.0006377551020408163)
    unsqueeze_1235: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1605, 0);  mul_1605 = None
    unsqueeze_1236: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1235, 2);  unsqueeze_1235 = None
    unsqueeze_1237: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, 3);  unsqueeze_1236 = None
    mul_1606: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_95, 0.0006377551020408163)
    mul_1607: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_370, squeeze_370)
    mul_1608: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1606, mul_1607);  mul_1606 = mul_1607 = None
    unsqueeze_1238: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1608, 0);  mul_1608 = None
    unsqueeze_1239: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 2);  unsqueeze_1238 = None
    unsqueeze_1240: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 3);  unsqueeze_1239 = None
    mul_1609: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_370, primals_371);  primals_371 = None
    unsqueeze_1241: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1609, 0);  mul_1609 = None
    unsqueeze_1242: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 2);  unsqueeze_1241 = None
    unsqueeze_1243: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 3);  unsqueeze_1242 = None
    mul_1610: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_354, unsqueeze_1240);  sub_354 = unsqueeze_1240 = None
    sub_356: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_45, mul_1610);  mul_1610 = None
    sub_357: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_356, unsqueeze_1237);  sub_356 = unsqueeze_1237 = None
    mul_1611: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_357, unsqueeze_1243);  sub_357 = unsqueeze_1243 = None
    mul_1612: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_370);  sum_95 = squeeze_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1611, cat_23, primals_370, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1611 = cat_23 = primals_370 = None
    getitem_1140: "f32[8, 416, 14, 14]" = convolution_backward_46[0]
    getitem_1141: "f32[1024, 416, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_37: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1140, 1, 0, 104)
    slice_38: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1140, 1, 104, 208)
    slice_39: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1140, 1, 208, 312)
    slice_40: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1140, 1, 312, 416);  getitem_1140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_46: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_46, full_default, slice_39);  le_46 = slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_96: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_358: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_1246);  convolution_122 = unsqueeze_1246 = None
    mul_1613: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, sub_358)
    sum_97: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1613, [0, 2, 3]);  mul_1613 = None
    mul_1614: "f32[104]" = torch.ops.aten.mul.Tensor(sum_96, 0.0006377551020408163)
    unsqueeze_1247: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1614, 0);  mul_1614 = None
    unsqueeze_1248: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1247, 2);  unsqueeze_1247 = None
    unsqueeze_1249: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, 3);  unsqueeze_1248 = None
    mul_1615: "f32[104]" = torch.ops.aten.mul.Tensor(sum_97, 0.0006377551020408163)
    mul_1616: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_367, squeeze_367)
    mul_1617: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1615, mul_1616);  mul_1615 = mul_1616 = None
    unsqueeze_1250: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1617, 0);  mul_1617 = None
    unsqueeze_1251: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, 2);  unsqueeze_1250 = None
    unsqueeze_1252: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 3);  unsqueeze_1251 = None
    mul_1618: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_367, primals_368);  primals_368 = None
    unsqueeze_1253: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1618, 0);  mul_1618 = None
    unsqueeze_1254: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 2);  unsqueeze_1253 = None
    unsqueeze_1255: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 3);  unsqueeze_1254 = None
    mul_1619: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_358, unsqueeze_1252);  sub_358 = unsqueeze_1252 = None
    sub_360: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_46, mul_1619);  where_46 = mul_1619 = None
    sub_361: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_360, unsqueeze_1249);  sub_360 = unsqueeze_1249 = None
    mul_1620: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_361, unsqueeze_1255);  sub_361 = unsqueeze_1255 = None
    mul_1621: "f32[104]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_367);  sum_97 = squeeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1620, add_674, primals_367, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1620 = add_674 = primals_367 = None
    getitem_1143: "f32[8, 104, 14, 14]" = convolution_backward_47[0]
    getitem_1144: "f32[104, 104, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_966: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_38, getitem_1143);  slice_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_47: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_47, full_default, add_966);  le_47 = add_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_98: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_362: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_1258);  convolution_121 = unsqueeze_1258 = None
    mul_1622: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, sub_362)
    sum_99: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1622, [0, 2, 3]);  mul_1622 = None
    mul_1623: "f32[104]" = torch.ops.aten.mul.Tensor(sum_98, 0.0006377551020408163)
    unsqueeze_1259: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1623, 0);  mul_1623 = None
    unsqueeze_1260: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1259, 2);  unsqueeze_1259 = None
    unsqueeze_1261: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, 3);  unsqueeze_1260 = None
    mul_1624: "f32[104]" = torch.ops.aten.mul.Tensor(sum_99, 0.0006377551020408163)
    mul_1625: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_364, squeeze_364)
    mul_1626: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1624, mul_1625);  mul_1624 = mul_1625 = None
    unsqueeze_1262: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1626, 0);  mul_1626 = None
    unsqueeze_1263: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, 2);  unsqueeze_1262 = None
    unsqueeze_1264: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 3);  unsqueeze_1263 = None
    mul_1627: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_364, primals_365);  primals_365 = None
    unsqueeze_1265: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1627, 0);  mul_1627 = None
    unsqueeze_1266: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 2);  unsqueeze_1265 = None
    unsqueeze_1267: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 3);  unsqueeze_1266 = None
    mul_1628: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_362, unsqueeze_1264);  sub_362 = unsqueeze_1264 = None
    sub_364: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_47, mul_1628);  where_47 = mul_1628 = None
    sub_365: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_364, unsqueeze_1261);  sub_364 = unsqueeze_1261 = None
    mul_1629: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_365, unsqueeze_1267);  sub_365 = unsqueeze_1267 = None
    mul_1630: "f32[104]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_364);  sum_99 = squeeze_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1629, add_668, primals_364, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1629 = add_668 = primals_364 = None
    getitem_1146: "f32[8, 104, 14, 14]" = convolution_backward_48[0]
    getitem_1147: "f32[104, 104, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_967: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_37, getitem_1146);  slice_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_48: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_48, full_default, add_967);  le_48 = add_967 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_100: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_366: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_1270);  convolution_120 = unsqueeze_1270 = None
    mul_1631: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_48, sub_366)
    sum_101: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1631, [0, 2, 3]);  mul_1631 = None
    mul_1632: "f32[104]" = torch.ops.aten.mul.Tensor(sum_100, 0.0006377551020408163)
    unsqueeze_1271: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1632, 0);  mul_1632 = None
    unsqueeze_1272: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1271, 2);  unsqueeze_1271 = None
    unsqueeze_1273: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, 3);  unsqueeze_1272 = None
    mul_1633: "f32[104]" = torch.ops.aten.mul.Tensor(sum_101, 0.0006377551020408163)
    mul_1634: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_361, squeeze_361)
    mul_1635: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1633, mul_1634);  mul_1633 = mul_1634 = None
    unsqueeze_1274: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1635, 0);  mul_1635 = None
    unsqueeze_1275: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, 2);  unsqueeze_1274 = None
    unsqueeze_1276: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 3);  unsqueeze_1275 = None
    mul_1636: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_361, primals_362);  primals_362 = None
    unsqueeze_1277: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1636, 0);  mul_1636 = None
    unsqueeze_1278: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 2);  unsqueeze_1277 = None
    unsqueeze_1279: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 3);  unsqueeze_1278 = None
    mul_1637: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_366, unsqueeze_1276);  sub_366 = unsqueeze_1276 = None
    sub_368: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_48, mul_1637);  where_48 = mul_1637 = None
    sub_369: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_368, unsqueeze_1273);  sub_368 = unsqueeze_1273 = None
    mul_1638: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_369, unsqueeze_1279);  sub_369 = unsqueeze_1279 = None
    mul_1639: "f32[104]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_361);  sum_101 = squeeze_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1638, getitem_706, primals_361, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1638 = getitem_706 = primals_361 = None
    getitem_1149: "f32[8, 104, 14, 14]" = convolution_backward_49[0]
    getitem_1150: "f32[104, 104, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_42: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1149, getitem_1146, getitem_1143, slice_40], 1);  getitem_1149 = getitem_1146 = getitem_1143 = slice_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_49: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_49, full_default, cat_42);  le_49 = cat_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_102: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_370: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_1282);  convolution_119 = unsqueeze_1282 = None
    mul_1640: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_49, sub_370)
    sum_103: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1640, [0, 2, 3]);  mul_1640 = None
    mul_1641: "f32[416]" = torch.ops.aten.mul.Tensor(sum_102, 0.0006377551020408163)
    unsqueeze_1283: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1641, 0);  mul_1641 = None
    unsqueeze_1284: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1283, 2);  unsqueeze_1283 = None
    unsqueeze_1285: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, 3);  unsqueeze_1284 = None
    mul_1642: "f32[416]" = torch.ops.aten.mul.Tensor(sum_103, 0.0006377551020408163)
    mul_1643: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_358, squeeze_358)
    mul_1644: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1642, mul_1643);  mul_1642 = mul_1643 = None
    unsqueeze_1286: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1644, 0);  mul_1644 = None
    unsqueeze_1287: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, 2);  unsqueeze_1286 = None
    unsqueeze_1288: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1287, 3);  unsqueeze_1287 = None
    mul_1645: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_358, primals_359);  primals_359 = None
    unsqueeze_1289: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1645, 0);  mul_1645 = None
    unsqueeze_1290: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1289, 2);  unsqueeze_1289 = None
    unsqueeze_1291: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 3);  unsqueeze_1290 = None
    mul_1646: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_370, unsqueeze_1288);  sub_370 = unsqueeze_1288 = None
    sub_372: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_49, mul_1646);  where_49 = mul_1646 = None
    sub_373: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_372, unsqueeze_1285);  sub_372 = unsqueeze_1285 = None
    mul_1647: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_373, unsqueeze_1291);  sub_373 = unsqueeze_1291 = None
    mul_1648: "f32[416]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_358);  sum_103 = squeeze_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1647, relu_115, primals_358, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1647 = primals_358 = None
    getitem_1152: "f32[8, 1024, 14, 14]" = convolution_backward_50[0]
    getitem_1153: "f32[416, 1024, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_968: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_45, getitem_1152);  where_45 = getitem_1152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_50: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_115, 0);  relu_115 = None
    where_50: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_50, full_default, add_968);  le_50 = add_968 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_104: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_374: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_1294);  convolution_118 = unsqueeze_1294 = None
    mul_1649: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_50, sub_374)
    sum_105: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1649, [0, 2, 3]);  mul_1649 = None
    mul_1650: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_104, 0.0006377551020408163)
    unsqueeze_1295: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1650, 0);  mul_1650 = None
    unsqueeze_1296: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1295, 2);  unsqueeze_1295 = None
    unsqueeze_1297: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, 3);  unsqueeze_1296 = None
    mul_1651: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_105, 0.0006377551020408163)
    mul_1652: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_355, squeeze_355)
    mul_1653: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1651, mul_1652);  mul_1651 = mul_1652 = None
    unsqueeze_1298: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1653, 0);  mul_1653 = None
    unsqueeze_1299: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, 2);  unsqueeze_1298 = None
    unsqueeze_1300: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1299, 3);  unsqueeze_1299 = None
    mul_1654: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_355, primals_356);  primals_356 = None
    unsqueeze_1301: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1654, 0);  mul_1654 = None
    unsqueeze_1302: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1301, 2);  unsqueeze_1301 = None
    unsqueeze_1303: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 3);  unsqueeze_1302 = None
    mul_1655: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_374, unsqueeze_1300);  sub_374 = unsqueeze_1300 = None
    sub_376: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_50, mul_1655);  mul_1655 = None
    sub_377: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_376, unsqueeze_1297);  sub_376 = unsqueeze_1297 = None
    mul_1656: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_377, unsqueeze_1303);  sub_377 = unsqueeze_1303 = None
    mul_1657: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_355);  sum_105 = squeeze_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1656, cat_22, primals_355, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1656 = cat_22 = primals_355 = None
    getitem_1155: "f32[8, 416, 14, 14]" = convolution_backward_51[0]
    getitem_1156: "f32[1024, 416, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_41: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1155, 1, 0, 104)
    slice_42: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1155, 1, 104, 208)
    slice_43: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1155, 1, 208, 312)
    slice_44: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1155, 1, 312, 416);  getitem_1155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_51: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_51, full_default, slice_43);  le_51 = slice_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_106: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_378: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_1306);  convolution_117 = unsqueeze_1306 = None
    mul_1658: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_51, sub_378)
    sum_107: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1658, [0, 2, 3]);  mul_1658 = None
    mul_1659: "f32[104]" = torch.ops.aten.mul.Tensor(sum_106, 0.0006377551020408163)
    unsqueeze_1307: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1659, 0);  mul_1659 = None
    unsqueeze_1308: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1307, 2);  unsqueeze_1307 = None
    unsqueeze_1309: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, 3);  unsqueeze_1308 = None
    mul_1660: "f32[104]" = torch.ops.aten.mul.Tensor(sum_107, 0.0006377551020408163)
    mul_1661: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_352, squeeze_352)
    mul_1662: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1660, mul_1661);  mul_1660 = mul_1661 = None
    unsqueeze_1310: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1662, 0);  mul_1662 = None
    unsqueeze_1311: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, 2);  unsqueeze_1310 = None
    unsqueeze_1312: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1311, 3);  unsqueeze_1311 = None
    mul_1663: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_352, primals_353);  primals_353 = None
    unsqueeze_1313: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1663, 0);  mul_1663 = None
    unsqueeze_1314: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1313, 2);  unsqueeze_1313 = None
    unsqueeze_1315: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 3);  unsqueeze_1314 = None
    mul_1664: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_378, unsqueeze_1312);  sub_378 = unsqueeze_1312 = None
    sub_380: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_51, mul_1664);  where_51 = mul_1664 = None
    sub_381: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_380, unsqueeze_1309);  sub_380 = unsqueeze_1309 = None
    mul_1665: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_381, unsqueeze_1315);  sub_381 = unsqueeze_1315 = None
    mul_1666: "f32[104]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_352);  sum_107 = squeeze_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1665, add_646, primals_352, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1665 = add_646 = primals_352 = None
    getitem_1158: "f32[8, 104, 14, 14]" = convolution_backward_52[0]
    getitem_1159: "f32[104, 104, 3, 3]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_969: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_42, getitem_1158);  slice_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_52: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_52, full_default, add_969);  le_52 = add_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_108: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_382: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_1318);  convolution_116 = unsqueeze_1318 = None
    mul_1667: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_52, sub_382)
    sum_109: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1667, [0, 2, 3]);  mul_1667 = None
    mul_1668: "f32[104]" = torch.ops.aten.mul.Tensor(sum_108, 0.0006377551020408163)
    unsqueeze_1319: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1668, 0);  mul_1668 = None
    unsqueeze_1320: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1319, 2);  unsqueeze_1319 = None
    unsqueeze_1321: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, 3);  unsqueeze_1320 = None
    mul_1669: "f32[104]" = torch.ops.aten.mul.Tensor(sum_109, 0.0006377551020408163)
    mul_1670: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_349, squeeze_349)
    mul_1671: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1669, mul_1670);  mul_1669 = mul_1670 = None
    unsqueeze_1322: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1671, 0);  mul_1671 = None
    unsqueeze_1323: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, 2);  unsqueeze_1322 = None
    unsqueeze_1324: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1323, 3);  unsqueeze_1323 = None
    mul_1672: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_349, primals_350);  primals_350 = None
    unsqueeze_1325: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1672, 0);  mul_1672 = None
    unsqueeze_1326: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1325, 2);  unsqueeze_1325 = None
    unsqueeze_1327: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 3);  unsqueeze_1326 = None
    mul_1673: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_382, unsqueeze_1324);  sub_382 = unsqueeze_1324 = None
    sub_384: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_52, mul_1673);  where_52 = mul_1673 = None
    sub_385: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_384, unsqueeze_1321);  sub_384 = unsqueeze_1321 = None
    mul_1674: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_385, unsqueeze_1327);  sub_385 = unsqueeze_1327 = None
    mul_1675: "f32[104]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_349);  sum_109 = squeeze_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1674, add_640, primals_349, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1674 = add_640 = primals_349 = None
    getitem_1161: "f32[8, 104, 14, 14]" = convolution_backward_53[0]
    getitem_1162: "f32[104, 104, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_970: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_41, getitem_1161);  slice_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_53: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_53, full_default, add_970);  le_53 = add_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_110: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_386: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_1330);  convolution_115 = unsqueeze_1330 = None
    mul_1676: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_53, sub_386)
    sum_111: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1676, [0, 2, 3]);  mul_1676 = None
    mul_1677: "f32[104]" = torch.ops.aten.mul.Tensor(sum_110, 0.0006377551020408163)
    unsqueeze_1331: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1677, 0);  mul_1677 = None
    unsqueeze_1332: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1331, 2);  unsqueeze_1331 = None
    unsqueeze_1333: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, 3);  unsqueeze_1332 = None
    mul_1678: "f32[104]" = torch.ops.aten.mul.Tensor(sum_111, 0.0006377551020408163)
    mul_1679: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_346, squeeze_346)
    mul_1680: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1678, mul_1679);  mul_1678 = mul_1679 = None
    unsqueeze_1334: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1680, 0);  mul_1680 = None
    unsqueeze_1335: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, 2);  unsqueeze_1334 = None
    unsqueeze_1336: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1335, 3);  unsqueeze_1335 = None
    mul_1681: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_346, primals_347);  primals_347 = None
    unsqueeze_1337: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1681, 0);  mul_1681 = None
    unsqueeze_1338: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1337, 2);  unsqueeze_1337 = None
    unsqueeze_1339: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 3);  unsqueeze_1338 = None
    mul_1682: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_386, unsqueeze_1336);  sub_386 = unsqueeze_1336 = None
    sub_388: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_53, mul_1682);  where_53 = mul_1682 = None
    sub_389: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_388, unsqueeze_1333);  sub_388 = unsqueeze_1333 = None
    mul_1683: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_389, unsqueeze_1339);  sub_389 = unsqueeze_1339 = None
    mul_1684: "f32[104]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_346);  sum_111 = squeeze_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1683, getitem_676, primals_346, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1683 = getitem_676 = primals_346 = None
    getitem_1164: "f32[8, 104, 14, 14]" = convolution_backward_54[0]
    getitem_1165: "f32[104, 104, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_43: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1164, getitem_1161, getitem_1158, slice_44], 1);  getitem_1164 = getitem_1161 = getitem_1158 = slice_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_54: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_54, full_default, cat_43);  le_54 = cat_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_112: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_390: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_1342);  convolution_114 = unsqueeze_1342 = None
    mul_1685: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_54, sub_390)
    sum_113: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1685, [0, 2, 3]);  mul_1685 = None
    mul_1686: "f32[416]" = torch.ops.aten.mul.Tensor(sum_112, 0.0006377551020408163)
    unsqueeze_1343: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1686, 0);  mul_1686 = None
    unsqueeze_1344: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1343, 2);  unsqueeze_1343 = None
    unsqueeze_1345: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, 3);  unsqueeze_1344 = None
    mul_1687: "f32[416]" = torch.ops.aten.mul.Tensor(sum_113, 0.0006377551020408163)
    mul_1688: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_343, squeeze_343)
    mul_1689: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1687, mul_1688);  mul_1687 = mul_1688 = None
    unsqueeze_1346: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1689, 0);  mul_1689 = None
    unsqueeze_1347: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, 2);  unsqueeze_1346 = None
    unsqueeze_1348: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1347, 3);  unsqueeze_1347 = None
    mul_1690: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_343, primals_344);  primals_344 = None
    unsqueeze_1349: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1690, 0);  mul_1690 = None
    unsqueeze_1350: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1349, 2);  unsqueeze_1349 = None
    unsqueeze_1351: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 3);  unsqueeze_1350 = None
    mul_1691: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_390, unsqueeze_1348);  sub_390 = unsqueeze_1348 = None
    sub_392: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_54, mul_1691);  where_54 = mul_1691 = None
    sub_393: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_392, unsqueeze_1345);  sub_392 = unsqueeze_1345 = None
    mul_1692: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_393, unsqueeze_1351);  sub_393 = unsqueeze_1351 = None
    mul_1693: "f32[416]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_343);  sum_113 = squeeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1692, relu_110, primals_343, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1692 = primals_343 = None
    getitem_1167: "f32[8, 1024, 14, 14]" = convolution_backward_55[0]
    getitem_1168: "f32[416, 1024, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_971: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_50, getitem_1167);  where_50 = getitem_1167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_55: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_110, 0);  relu_110 = None
    where_55: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_55, full_default, add_971);  le_55 = add_971 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_114: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_394: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_1354);  convolution_113 = unsqueeze_1354 = None
    mul_1694: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_55, sub_394)
    sum_115: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1694, [0, 2, 3]);  mul_1694 = None
    mul_1695: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_114, 0.0006377551020408163)
    unsqueeze_1355: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1695, 0);  mul_1695 = None
    unsqueeze_1356: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1355, 2);  unsqueeze_1355 = None
    unsqueeze_1357: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, 3);  unsqueeze_1356 = None
    mul_1696: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_115, 0.0006377551020408163)
    mul_1697: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_340, squeeze_340)
    mul_1698: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1696, mul_1697);  mul_1696 = mul_1697 = None
    unsqueeze_1358: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1698, 0);  mul_1698 = None
    unsqueeze_1359: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, 2);  unsqueeze_1358 = None
    unsqueeze_1360: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1359, 3);  unsqueeze_1359 = None
    mul_1699: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_340, primals_341);  primals_341 = None
    unsqueeze_1361: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1699, 0);  mul_1699 = None
    unsqueeze_1362: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1361, 2);  unsqueeze_1361 = None
    unsqueeze_1363: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 3);  unsqueeze_1362 = None
    mul_1700: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_394, unsqueeze_1360);  sub_394 = unsqueeze_1360 = None
    sub_396: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_55, mul_1700);  mul_1700 = None
    sub_397: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_396, unsqueeze_1357);  sub_396 = unsqueeze_1357 = None
    mul_1701: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_397, unsqueeze_1363);  sub_397 = unsqueeze_1363 = None
    mul_1702: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_340);  sum_115 = squeeze_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1701, cat_21, primals_340, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1701 = cat_21 = primals_340 = None
    getitem_1170: "f32[8, 416, 14, 14]" = convolution_backward_56[0]
    getitem_1171: "f32[1024, 416, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_45: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1170, 1, 0, 104)
    slice_46: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1170, 1, 104, 208)
    slice_47: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1170, 1, 208, 312)
    slice_48: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1170, 1, 312, 416);  getitem_1170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_56: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_56, full_default, slice_47);  le_56 = slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_116: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_398: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_1366);  convolution_112 = unsqueeze_1366 = None
    mul_1703: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_56, sub_398)
    sum_117: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1703, [0, 2, 3]);  mul_1703 = None
    mul_1704: "f32[104]" = torch.ops.aten.mul.Tensor(sum_116, 0.0006377551020408163)
    unsqueeze_1367: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1704, 0);  mul_1704 = None
    unsqueeze_1368: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1367, 2);  unsqueeze_1367 = None
    unsqueeze_1369: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, 3);  unsqueeze_1368 = None
    mul_1705: "f32[104]" = torch.ops.aten.mul.Tensor(sum_117, 0.0006377551020408163)
    mul_1706: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_337, squeeze_337)
    mul_1707: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1705, mul_1706);  mul_1705 = mul_1706 = None
    unsqueeze_1370: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1707, 0);  mul_1707 = None
    unsqueeze_1371: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, 2);  unsqueeze_1370 = None
    unsqueeze_1372: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1371, 3);  unsqueeze_1371 = None
    mul_1708: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_337, primals_338);  primals_338 = None
    unsqueeze_1373: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1708, 0);  mul_1708 = None
    unsqueeze_1374: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1373, 2);  unsqueeze_1373 = None
    unsqueeze_1375: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 3);  unsqueeze_1374 = None
    mul_1709: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_398, unsqueeze_1372);  sub_398 = unsqueeze_1372 = None
    sub_400: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_56, mul_1709);  where_56 = mul_1709 = None
    sub_401: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_400, unsqueeze_1369);  sub_400 = unsqueeze_1369 = None
    mul_1710: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_401, unsqueeze_1375);  sub_401 = unsqueeze_1375 = None
    mul_1711: "f32[104]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_337);  sum_117 = squeeze_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1710, add_618, primals_337, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1710 = add_618 = primals_337 = None
    getitem_1173: "f32[8, 104, 14, 14]" = convolution_backward_57[0]
    getitem_1174: "f32[104, 104, 3, 3]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_972: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_46, getitem_1173);  slice_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_57: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_57, full_default, add_972);  le_57 = add_972 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_118: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_402: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_1378);  convolution_111 = unsqueeze_1378 = None
    mul_1712: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_57, sub_402)
    sum_119: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1712, [0, 2, 3]);  mul_1712 = None
    mul_1713: "f32[104]" = torch.ops.aten.mul.Tensor(sum_118, 0.0006377551020408163)
    unsqueeze_1379: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1713, 0);  mul_1713 = None
    unsqueeze_1380: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1379, 2);  unsqueeze_1379 = None
    unsqueeze_1381: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, 3);  unsqueeze_1380 = None
    mul_1714: "f32[104]" = torch.ops.aten.mul.Tensor(sum_119, 0.0006377551020408163)
    mul_1715: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_334, squeeze_334)
    mul_1716: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1714, mul_1715);  mul_1714 = mul_1715 = None
    unsqueeze_1382: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1716, 0);  mul_1716 = None
    unsqueeze_1383: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, 2);  unsqueeze_1382 = None
    unsqueeze_1384: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1383, 3);  unsqueeze_1383 = None
    mul_1717: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_334, primals_335);  primals_335 = None
    unsqueeze_1385: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1717, 0);  mul_1717 = None
    unsqueeze_1386: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1385, 2);  unsqueeze_1385 = None
    unsqueeze_1387: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 3);  unsqueeze_1386 = None
    mul_1718: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_402, unsqueeze_1384);  sub_402 = unsqueeze_1384 = None
    sub_404: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_57, mul_1718);  where_57 = mul_1718 = None
    sub_405: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_404, unsqueeze_1381);  sub_404 = unsqueeze_1381 = None
    mul_1719: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_405, unsqueeze_1387);  sub_405 = unsqueeze_1387 = None
    mul_1720: "f32[104]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_334);  sum_119 = squeeze_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1719, add_612, primals_334, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1719 = add_612 = primals_334 = None
    getitem_1176: "f32[8, 104, 14, 14]" = convolution_backward_58[0]
    getitem_1177: "f32[104, 104, 3, 3]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_973: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_45, getitem_1176);  slice_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_58: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_58, full_default, add_973);  le_58 = add_973 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_120: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_406: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_1390);  convolution_110 = unsqueeze_1390 = None
    mul_1721: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_58, sub_406)
    sum_121: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1721, [0, 2, 3]);  mul_1721 = None
    mul_1722: "f32[104]" = torch.ops.aten.mul.Tensor(sum_120, 0.0006377551020408163)
    unsqueeze_1391: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1722, 0);  mul_1722 = None
    unsqueeze_1392: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1391, 2);  unsqueeze_1391 = None
    unsqueeze_1393: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1392, 3);  unsqueeze_1392 = None
    mul_1723: "f32[104]" = torch.ops.aten.mul.Tensor(sum_121, 0.0006377551020408163)
    mul_1724: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_331, squeeze_331)
    mul_1725: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1723, mul_1724);  mul_1723 = mul_1724 = None
    unsqueeze_1394: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1725, 0);  mul_1725 = None
    unsqueeze_1395: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, 2);  unsqueeze_1394 = None
    unsqueeze_1396: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1395, 3);  unsqueeze_1395 = None
    mul_1726: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_331, primals_332);  primals_332 = None
    unsqueeze_1397: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1726, 0);  mul_1726 = None
    unsqueeze_1398: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1397, 2);  unsqueeze_1397 = None
    unsqueeze_1399: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, 3);  unsqueeze_1398 = None
    mul_1727: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_406, unsqueeze_1396);  sub_406 = unsqueeze_1396 = None
    sub_408: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_58, mul_1727);  where_58 = mul_1727 = None
    sub_409: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_408, unsqueeze_1393);  sub_408 = unsqueeze_1393 = None
    mul_1728: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_409, unsqueeze_1399);  sub_409 = unsqueeze_1399 = None
    mul_1729: "f32[104]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_331);  sum_121 = squeeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1728, getitem_646, primals_331, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1728 = getitem_646 = primals_331 = None
    getitem_1179: "f32[8, 104, 14, 14]" = convolution_backward_59[0]
    getitem_1180: "f32[104, 104, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_44: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1179, getitem_1176, getitem_1173, slice_48], 1);  getitem_1179 = getitem_1176 = getitem_1173 = slice_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_59: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_59, full_default, cat_44);  le_59 = cat_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_122: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_410: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_1402);  convolution_109 = unsqueeze_1402 = None
    mul_1730: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_59, sub_410)
    sum_123: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1730, [0, 2, 3]);  mul_1730 = None
    mul_1731: "f32[416]" = torch.ops.aten.mul.Tensor(sum_122, 0.0006377551020408163)
    unsqueeze_1403: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1731, 0);  mul_1731 = None
    unsqueeze_1404: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1403, 2);  unsqueeze_1403 = None
    unsqueeze_1405: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1404, 3);  unsqueeze_1404 = None
    mul_1732: "f32[416]" = torch.ops.aten.mul.Tensor(sum_123, 0.0006377551020408163)
    mul_1733: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_328, squeeze_328)
    mul_1734: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1732, mul_1733);  mul_1732 = mul_1733 = None
    unsqueeze_1406: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1734, 0);  mul_1734 = None
    unsqueeze_1407: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, 2);  unsqueeze_1406 = None
    unsqueeze_1408: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1407, 3);  unsqueeze_1407 = None
    mul_1735: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_328, primals_329);  primals_329 = None
    unsqueeze_1409: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1735, 0);  mul_1735 = None
    unsqueeze_1410: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1409, 2);  unsqueeze_1409 = None
    unsqueeze_1411: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, 3);  unsqueeze_1410 = None
    mul_1736: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_410, unsqueeze_1408);  sub_410 = unsqueeze_1408 = None
    sub_412: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_59, mul_1736);  where_59 = mul_1736 = None
    sub_413: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_412, unsqueeze_1405);  sub_412 = unsqueeze_1405 = None
    mul_1737: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_413, unsqueeze_1411);  sub_413 = unsqueeze_1411 = None
    mul_1738: "f32[416]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_328);  sum_123 = squeeze_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1737, relu_105, primals_328, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1737 = primals_328 = None
    getitem_1182: "f32[8, 1024, 14, 14]" = convolution_backward_60[0]
    getitem_1183: "f32[416, 1024, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_974: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_55, getitem_1182);  where_55 = getitem_1182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_60: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_105, 0);  relu_105 = None
    where_60: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_60, full_default, add_974);  le_60 = add_974 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_124: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_414: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_1414);  convolution_108 = unsqueeze_1414 = None
    mul_1739: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_60, sub_414)
    sum_125: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1739, [0, 2, 3]);  mul_1739 = None
    mul_1740: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_124, 0.0006377551020408163)
    unsqueeze_1415: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1740, 0);  mul_1740 = None
    unsqueeze_1416: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1415, 2);  unsqueeze_1415 = None
    unsqueeze_1417: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1416, 3);  unsqueeze_1416 = None
    mul_1741: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_125, 0.0006377551020408163)
    mul_1742: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_325, squeeze_325)
    mul_1743: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1741, mul_1742);  mul_1741 = mul_1742 = None
    unsqueeze_1418: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1743, 0);  mul_1743 = None
    unsqueeze_1419: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, 2);  unsqueeze_1418 = None
    unsqueeze_1420: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1419, 3);  unsqueeze_1419 = None
    mul_1744: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_325, primals_326);  primals_326 = None
    unsqueeze_1421: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1744, 0);  mul_1744 = None
    unsqueeze_1422: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1421, 2);  unsqueeze_1421 = None
    unsqueeze_1423: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, 3);  unsqueeze_1422 = None
    mul_1745: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_414, unsqueeze_1420);  sub_414 = unsqueeze_1420 = None
    sub_416: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_60, mul_1745);  mul_1745 = None
    sub_417: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_416, unsqueeze_1417);  sub_416 = unsqueeze_1417 = None
    mul_1746: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_417, unsqueeze_1423);  sub_417 = unsqueeze_1423 = None
    mul_1747: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_325);  sum_125 = squeeze_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1746, cat_20, primals_325, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1746 = cat_20 = primals_325 = None
    getitem_1185: "f32[8, 416, 14, 14]" = convolution_backward_61[0]
    getitem_1186: "f32[1024, 416, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_49: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1185, 1, 0, 104)
    slice_50: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1185, 1, 104, 208)
    slice_51: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1185, 1, 208, 312)
    slice_52: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1185, 1, 312, 416);  getitem_1185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_61: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_61, full_default, slice_51);  le_61 = slice_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_126: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_418: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_1426);  convolution_107 = unsqueeze_1426 = None
    mul_1748: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_61, sub_418)
    sum_127: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1748, [0, 2, 3]);  mul_1748 = None
    mul_1749: "f32[104]" = torch.ops.aten.mul.Tensor(sum_126, 0.0006377551020408163)
    unsqueeze_1427: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1749, 0);  mul_1749 = None
    unsqueeze_1428: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1427, 2);  unsqueeze_1427 = None
    unsqueeze_1429: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1428, 3);  unsqueeze_1428 = None
    mul_1750: "f32[104]" = torch.ops.aten.mul.Tensor(sum_127, 0.0006377551020408163)
    mul_1751: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_322, squeeze_322)
    mul_1752: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1750, mul_1751);  mul_1750 = mul_1751 = None
    unsqueeze_1430: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1752, 0);  mul_1752 = None
    unsqueeze_1431: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1430, 2);  unsqueeze_1430 = None
    unsqueeze_1432: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1431, 3);  unsqueeze_1431 = None
    mul_1753: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_322, primals_323);  primals_323 = None
    unsqueeze_1433: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1753, 0);  mul_1753 = None
    unsqueeze_1434: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1433, 2);  unsqueeze_1433 = None
    unsqueeze_1435: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, 3);  unsqueeze_1434 = None
    mul_1754: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_418, unsqueeze_1432);  sub_418 = unsqueeze_1432 = None
    sub_420: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_61, mul_1754);  where_61 = mul_1754 = None
    sub_421: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_420, unsqueeze_1429);  sub_420 = unsqueeze_1429 = None
    mul_1755: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_421, unsqueeze_1435);  sub_421 = unsqueeze_1435 = None
    mul_1756: "f32[104]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_322);  sum_127 = squeeze_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1755, add_590, primals_322, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1755 = add_590 = primals_322 = None
    getitem_1188: "f32[8, 104, 14, 14]" = convolution_backward_62[0]
    getitem_1189: "f32[104, 104, 3, 3]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_975: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_50, getitem_1188);  slice_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_62: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_62, full_default, add_975);  le_62 = add_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_128: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_422: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_1438);  convolution_106 = unsqueeze_1438 = None
    mul_1757: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_62, sub_422)
    sum_129: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1757, [0, 2, 3]);  mul_1757 = None
    mul_1758: "f32[104]" = torch.ops.aten.mul.Tensor(sum_128, 0.0006377551020408163)
    unsqueeze_1439: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1758, 0);  mul_1758 = None
    unsqueeze_1440: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1439, 2);  unsqueeze_1439 = None
    unsqueeze_1441: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1440, 3);  unsqueeze_1440 = None
    mul_1759: "f32[104]" = torch.ops.aten.mul.Tensor(sum_129, 0.0006377551020408163)
    mul_1760: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_319, squeeze_319)
    mul_1761: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1759, mul_1760);  mul_1759 = mul_1760 = None
    unsqueeze_1442: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1761, 0);  mul_1761 = None
    unsqueeze_1443: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1442, 2);  unsqueeze_1442 = None
    unsqueeze_1444: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1443, 3);  unsqueeze_1443 = None
    mul_1762: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_319, primals_320);  primals_320 = None
    unsqueeze_1445: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1762, 0);  mul_1762 = None
    unsqueeze_1446: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1445, 2);  unsqueeze_1445 = None
    unsqueeze_1447: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, 3);  unsqueeze_1446 = None
    mul_1763: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_422, unsqueeze_1444);  sub_422 = unsqueeze_1444 = None
    sub_424: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_62, mul_1763);  where_62 = mul_1763 = None
    sub_425: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_424, unsqueeze_1441);  sub_424 = unsqueeze_1441 = None
    mul_1764: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_425, unsqueeze_1447);  sub_425 = unsqueeze_1447 = None
    mul_1765: "f32[104]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_319);  sum_129 = squeeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1764, add_584, primals_319, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1764 = add_584 = primals_319 = None
    getitem_1191: "f32[8, 104, 14, 14]" = convolution_backward_63[0]
    getitem_1192: "f32[104, 104, 3, 3]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_976: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_49, getitem_1191);  slice_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_63: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_63, full_default, add_976);  le_63 = add_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_130: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_426: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_1450);  convolution_105 = unsqueeze_1450 = None
    mul_1766: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_63, sub_426)
    sum_131: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1766, [0, 2, 3]);  mul_1766 = None
    mul_1767: "f32[104]" = torch.ops.aten.mul.Tensor(sum_130, 0.0006377551020408163)
    unsqueeze_1451: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1767, 0);  mul_1767 = None
    unsqueeze_1452: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1451, 2);  unsqueeze_1451 = None
    unsqueeze_1453: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1452, 3);  unsqueeze_1452 = None
    mul_1768: "f32[104]" = torch.ops.aten.mul.Tensor(sum_131, 0.0006377551020408163)
    mul_1769: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_316, squeeze_316)
    mul_1770: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1768, mul_1769);  mul_1768 = mul_1769 = None
    unsqueeze_1454: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1770, 0);  mul_1770 = None
    unsqueeze_1455: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, 2);  unsqueeze_1454 = None
    unsqueeze_1456: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1455, 3);  unsqueeze_1455 = None
    mul_1771: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_316, primals_317);  primals_317 = None
    unsqueeze_1457: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1771, 0);  mul_1771 = None
    unsqueeze_1458: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1457, 2);  unsqueeze_1457 = None
    unsqueeze_1459: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, 3);  unsqueeze_1458 = None
    mul_1772: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_426, unsqueeze_1456);  sub_426 = unsqueeze_1456 = None
    sub_428: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_63, mul_1772);  where_63 = mul_1772 = None
    sub_429: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_428, unsqueeze_1453);  sub_428 = unsqueeze_1453 = None
    mul_1773: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_429, unsqueeze_1459);  sub_429 = unsqueeze_1459 = None
    mul_1774: "f32[104]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_316);  sum_131 = squeeze_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1773, getitem_616, primals_316, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1773 = getitem_616 = primals_316 = None
    getitem_1194: "f32[8, 104, 14, 14]" = convolution_backward_64[0]
    getitem_1195: "f32[104, 104, 3, 3]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_45: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1194, getitem_1191, getitem_1188, slice_52], 1);  getitem_1194 = getitem_1191 = getitem_1188 = slice_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_64: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_64, full_default, cat_45);  le_64 = cat_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_132: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_430: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_1462);  convolution_104 = unsqueeze_1462 = None
    mul_1775: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_64, sub_430)
    sum_133: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1775, [0, 2, 3]);  mul_1775 = None
    mul_1776: "f32[416]" = torch.ops.aten.mul.Tensor(sum_132, 0.0006377551020408163)
    unsqueeze_1463: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1776, 0);  mul_1776 = None
    unsqueeze_1464: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1463, 2);  unsqueeze_1463 = None
    unsqueeze_1465: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1464, 3);  unsqueeze_1464 = None
    mul_1777: "f32[416]" = torch.ops.aten.mul.Tensor(sum_133, 0.0006377551020408163)
    mul_1778: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_313, squeeze_313)
    mul_1779: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1777, mul_1778);  mul_1777 = mul_1778 = None
    unsqueeze_1466: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1779, 0);  mul_1779 = None
    unsqueeze_1467: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, 2);  unsqueeze_1466 = None
    unsqueeze_1468: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1467, 3);  unsqueeze_1467 = None
    mul_1780: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_313, primals_314);  primals_314 = None
    unsqueeze_1469: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1780, 0);  mul_1780 = None
    unsqueeze_1470: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1469, 2);  unsqueeze_1469 = None
    unsqueeze_1471: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, 3);  unsqueeze_1470 = None
    mul_1781: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_430, unsqueeze_1468);  sub_430 = unsqueeze_1468 = None
    sub_432: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_64, mul_1781);  where_64 = mul_1781 = None
    sub_433: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_432, unsqueeze_1465);  sub_432 = unsqueeze_1465 = None
    mul_1782: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_433, unsqueeze_1471);  sub_433 = unsqueeze_1471 = None
    mul_1783: "f32[416]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_313);  sum_133 = squeeze_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1782, relu_100, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1782 = primals_313 = None
    getitem_1197: "f32[8, 1024, 14, 14]" = convolution_backward_65[0]
    getitem_1198: "f32[416, 1024, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_977: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_60, getitem_1197);  where_60 = getitem_1197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_65: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_100, 0);  relu_100 = None
    where_65: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_65, full_default, add_977);  le_65 = add_977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_134: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_434: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_1474);  convolution_103 = unsqueeze_1474 = None
    mul_1784: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_65, sub_434)
    sum_135: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1784, [0, 2, 3]);  mul_1784 = None
    mul_1785: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_134, 0.0006377551020408163)
    unsqueeze_1475: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1785, 0);  mul_1785 = None
    unsqueeze_1476: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1475, 2);  unsqueeze_1475 = None
    unsqueeze_1477: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1476, 3);  unsqueeze_1476 = None
    mul_1786: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_135, 0.0006377551020408163)
    mul_1787: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_310, squeeze_310)
    mul_1788: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1786, mul_1787);  mul_1786 = mul_1787 = None
    unsqueeze_1478: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1788, 0);  mul_1788 = None
    unsqueeze_1479: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, 2);  unsqueeze_1478 = None
    unsqueeze_1480: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1479, 3);  unsqueeze_1479 = None
    mul_1789: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_310, primals_311);  primals_311 = None
    unsqueeze_1481: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1789, 0);  mul_1789 = None
    unsqueeze_1482: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1481, 2);  unsqueeze_1481 = None
    unsqueeze_1483: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, 3);  unsqueeze_1482 = None
    mul_1790: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_434, unsqueeze_1480);  sub_434 = unsqueeze_1480 = None
    sub_436: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_65, mul_1790);  mul_1790 = None
    sub_437: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_436, unsqueeze_1477);  sub_436 = unsqueeze_1477 = None
    mul_1791: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_437, unsqueeze_1483);  sub_437 = unsqueeze_1483 = None
    mul_1792: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_310);  sum_135 = squeeze_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1791, cat_19, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1791 = cat_19 = primals_310 = None
    getitem_1200: "f32[8, 416, 14, 14]" = convolution_backward_66[0]
    getitem_1201: "f32[1024, 416, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_53: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1200, 1, 0, 104)
    slice_54: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1200, 1, 104, 208)
    slice_55: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1200, 1, 208, 312)
    slice_56: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1200, 1, 312, 416);  getitem_1200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_66: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_66, full_default, slice_55);  le_66 = slice_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_136: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_438: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_1486);  convolution_102 = unsqueeze_1486 = None
    mul_1793: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_66, sub_438)
    sum_137: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1793, [0, 2, 3]);  mul_1793 = None
    mul_1794: "f32[104]" = torch.ops.aten.mul.Tensor(sum_136, 0.0006377551020408163)
    unsqueeze_1487: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1794, 0);  mul_1794 = None
    unsqueeze_1488: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1487, 2);  unsqueeze_1487 = None
    unsqueeze_1489: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1488, 3);  unsqueeze_1488 = None
    mul_1795: "f32[104]" = torch.ops.aten.mul.Tensor(sum_137, 0.0006377551020408163)
    mul_1796: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_307, squeeze_307)
    mul_1797: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1795, mul_1796);  mul_1795 = mul_1796 = None
    unsqueeze_1490: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1797, 0);  mul_1797 = None
    unsqueeze_1491: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, 2);  unsqueeze_1490 = None
    unsqueeze_1492: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1491, 3);  unsqueeze_1491 = None
    mul_1798: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_307, primals_308);  primals_308 = None
    unsqueeze_1493: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1798, 0);  mul_1798 = None
    unsqueeze_1494: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1493, 2);  unsqueeze_1493 = None
    unsqueeze_1495: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, 3);  unsqueeze_1494 = None
    mul_1799: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_438, unsqueeze_1492);  sub_438 = unsqueeze_1492 = None
    sub_440: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_66, mul_1799);  where_66 = mul_1799 = None
    sub_441: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_440, unsqueeze_1489);  sub_440 = unsqueeze_1489 = None
    mul_1800: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_441, unsqueeze_1495);  sub_441 = unsqueeze_1495 = None
    mul_1801: "f32[104]" = torch.ops.aten.mul.Tensor(sum_137, squeeze_307);  sum_137 = squeeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1800, add_562, primals_307, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1800 = add_562 = primals_307 = None
    getitem_1203: "f32[8, 104, 14, 14]" = convolution_backward_67[0]
    getitem_1204: "f32[104, 104, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_978: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_54, getitem_1203);  slice_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_67: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_67, full_default, add_978);  le_67 = add_978 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_138: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_442: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_1498);  convolution_101 = unsqueeze_1498 = None
    mul_1802: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_67, sub_442)
    sum_139: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1802, [0, 2, 3]);  mul_1802 = None
    mul_1803: "f32[104]" = torch.ops.aten.mul.Tensor(sum_138, 0.0006377551020408163)
    unsqueeze_1499: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1803, 0);  mul_1803 = None
    unsqueeze_1500: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1499, 2);  unsqueeze_1499 = None
    unsqueeze_1501: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1500, 3);  unsqueeze_1500 = None
    mul_1804: "f32[104]" = torch.ops.aten.mul.Tensor(sum_139, 0.0006377551020408163)
    mul_1805: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_304, squeeze_304)
    mul_1806: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1804, mul_1805);  mul_1804 = mul_1805 = None
    unsqueeze_1502: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1806, 0);  mul_1806 = None
    unsqueeze_1503: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1502, 2);  unsqueeze_1502 = None
    unsqueeze_1504: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1503, 3);  unsqueeze_1503 = None
    mul_1807: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_304, primals_305);  primals_305 = None
    unsqueeze_1505: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1807, 0);  mul_1807 = None
    unsqueeze_1506: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1505, 2);  unsqueeze_1505 = None
    unsqueeze_1507: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, 3);  unsqueeze_1506 = None
    mul_1808: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_442, unsqueeze_1504);  sub_442 = unsqueeze_1504 = None
    sub_444: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_67, mul_1808);  where_67 = mul_1808 = None
    sub_445: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_444, unsqueeze_1501);  sub_444 = unsqueeze_1501 = None
    mul_1809: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_445, unsqueeze_1507);  sub_445 = unsqueeze_1507 = None
    mul_1810: "f32[104]" = torch.ops.aten.mul.Tensor(sum_139, squeeze_304);  sum_139 = squeeze_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1809, add_556, primals_304, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1809 = add_556 = primals_304 = None
    getitem_1206: "f32[8, 104, 14, 14]" = convolution_backward_68[0]
    getitem_1207: "f32[104, 104, 3, 3]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_979: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_53, getitem_1206);  slice_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_68: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_68, full_default, add_979);  le_68 = add_979 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_140: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_446: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_1510);  convolution_100 = unsqueeze_1510 = None
    mul_1811: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_68, sub_446)
    sum_141: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1811, [0, 2, 3]);  mul_1811 = None
    mul_1812: "f32[104]" = torch.ops.aten.mul.Tensor(sum_140, 0.0006377551020408163)
    unsqueeze_1511: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1812, 0);  mul_1812 = None
    unsqueeze_1512: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1511, 2);  unsqueeze_1511 = None
    unsqueeze_1513: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1512, 3);  unsqueeze_1512 = None
    mul_1813: "f32[104]" = torch.ops.aten.mul.Tensor(sum_141, 0.0006377551020408163)
    mul_1814: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_301, squeeze_301)
    mul_1815: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1813, mul_1814);  mul_1813 = mul_1814 = None
    unsqueeze_1514: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1815, 0);  mul_1815 = None
    unsqueeze_1515: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1514, 2);  unsqueeze_1514 = None
    unsqueeze_1516: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1515, 3);  unsqueeze_1515 = None
    mul_1816: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_301, primals_302);  primals_302 = None
    unsqueeze_1517: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1816, 0);  mul_1816 = None
    unsqueeze_1518: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1517, 2);  unsqueeze_1517 = None
    unsqueeze_1519: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, 3);  unsqueeze_1518 = None
    mul_1817: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_446, unsqueeze_1516);  sub_446 = unsqueeze_1516 = None
    sub_448: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_68, mul_1817);  where_68 = mul_1817 = None
    sub_449: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_448, unsqueeze_1513);  sub_448 = unsqueeze_1513 = None
    mul_1818: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_449, unsqueeze_1519);  sub_449 = unsqueeze_1519 = None
    mul_1819: "f32[104]" = torch.ops.aten.mul.Tensor(sum_141, squeeze_301);  sum_141 = squeeze_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1818, getitem_586, primals_301, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1818 = getitem_586 = primals_301 = None
    getitem_1209: "f32[8, 104, 14, 14]" = convolution_backward_69[0]
    getitem_1210: "f32[104, 104, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_46: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1209, getitem_1206, getitem_1203, slice_56], 1);  getitem_1209 = getitem_1206 = getitem_1203 = slice_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_69: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_69, full_default, cat_46);  le_69 = cat_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_142: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_450: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_1522);  convolution_99 = unsqueeze_1522 = None
    mul_1820: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_69, sub_450)
    sum_143: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1820, [0, 2, 3]);  mul_1820 = None
    mul_1821: "f32[416]" = torch.ops.aten.mul.Tensor(sum_142, 0.0006377551020408163)
    unsqueeze_1523: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1821, 0);  mul_1821 = None
    unsqueeze_1524: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1523, 2);  unsqueeze_1523 = None
    unsqueeze_1525: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1524, 3);  unsqueeze_1524 = None
    mul_1822: "f32[416]" = torch.ops.aten.mul.Tensor(sum_143, 0.0006377551020408163)
    mul_1823: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_298, squeeze_298)
    mul_1824: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1822, mul_1823);  mul_1822 = mul_1823 = None
    unsqueeze_1526: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1824, 0);  mul_1824 = None
    unsqueeze_1527: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1526, 2);  unsqueeze_1526 = None
    unsqueeze_1528: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1527, 3);  unsqueeze_1527 = None
    mul_1825: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_298, primals_299);  primals_299 = None
    unsqueeze_1529: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1825, 0);  mul_1825 = None
    unsqueeze_1530: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1529, 2);  unsqueeze_1529 = None
    unsqueeze_1531: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, 3);  unsqueeze_1530 = None
    mul_1826: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_450, unsqueeze_1528);  sub_450 = unsqueeze_1528 = None
    sub_452: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_69, mul_1826);  where_69 = mul_1826 = None
    sub_453: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_452, unsqueeze_1525);  sub_452 = unsqueeze_1525 = None
    mul_1827: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_453, unsqueeze_1531);  sub_453 = unsqueeze_1531 = None
    mul_1828: "f32[416]" = torch.ops.aten.mul.Tensor(sum_143, squeeze_298);  sum_143 = squeeze_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1827, relu_95, primals_298, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1827 = primals_298 = None
    getitem_1212: "f32[8, 1024, 14, 14]" = convolution_backward_70[0]
    getitem_1213: "f32[416, 1024, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_980: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_65, getitem_1212);  where_65 = getitem_1212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_70: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_95, 0);  relu_95 = None
    where_70: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_70, full_default, add_980);  le_70 = add_980 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_144: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_454: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_1534);  convolution_98 = unsqueeze_1534 = None
    mul_1829: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_70, sub_454)
    sum_145: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1829, [0, 2, 3]);  mul_1829 = None
    mul_1830: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_144, 0.0006377551020408163)
    unsqueeze_1535: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1830, 0);  mul_1830 = None
    unsqueeze_1536: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1535, 2);  unsqueeze_1535 = None
    unsqueeze_1537: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1536, 3);  unsqueeze_1536 = None
    mul_1831: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_145, 0.0006377551020408163)
    mul_1832: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_295, squeeze_295)
    mul_1833: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1831, mul_1832);  mul_1831 = mul_1832 = None
    unsqueeze_1538: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1833, 0);  mul_1833 = None
    unsqueeze_1539: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1538, 2);  unsqueeze_1538 = None
    unsqueeze_1540: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1539, 3);  unsqueeze_1539 = None
    mul_1834: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_295, primals_296);  primals_296 = None
    unsqueeze_1541: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1834, 0);  mul_1834 = None
    unsqueeze_1542: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1541, 2);  unsqueeze_1541 = None
    unsqueeze_1543: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, 3);  unsqueeze_1542 = None
    mul_1835: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_454, unsqueeze_1540);  sub_454 = unsqueeze_1540 = None
    sub_456: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_70, mul_1835);  mul_1835 = None
    sub_457: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_456, unsqueeze_1537);  sub_456 = unsqueeze_1537 = None
    mul_1836: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_457, unsqueeze_1543);  sub_457 = unsqueeze_1543 = None
    mul_1837: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_295);  sum_145 = squeeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1836, cat_18, primals_295, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1836 = cat_18 = primals_295 = None
    getitem_1215: "f32[8, 416, 14, 14]" = convolution_backward_71[0]
    getitem_1216: "f32[1024, 416, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_57: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1215, 1, 0, 104)
    slice_58: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1215, 1, 104, 208)
    slice_59: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1215, 1, 208, 312)
    slice_60: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1215, 1, 312, 416);  getitem_1215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_71: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_71, full_default, slice_59);  le_71 = slice_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_146: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_458: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_1546);  convolution_97 = unsqueeze_1546 = None
    mul_1838: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_71, sub_458)
    sum_147: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1838, [0, 2, 3]);  mul_1838 = None
    mul_1839: "f32[104]" = torch.ops.aten.mul.Tensor(sum_146, 0.0006377551020408163)
    unsqueeze_1547: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1839, 0);  mul_1839 = None
    unsqueeze_1548: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1547, 2);  unsqueeze_1547 = None
    unsqueeze_1549: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1548, 3);  unsqueeze_1548 = None
    mul_1840: "f32[104]" = torch.ops.aten.mul.Tensor(sum_147, 0.0006377551020408163)
    mul_1841: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_292, squeeze_292)
    mul_1842: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1840, mul_1841);  mul_1840 = mul_1841 = None
    unsqueeze_1550: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1842, 0);  mul_1842 = None
    unsqueeze_1551: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1550, 2);  unsqueeze_1550 = None
    unsqueeze_1552: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1551, 3);  unsqueeze_1551 = None
    mul_1843: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_292, primals_293);  primals_293 = None
    unsqueeze_1553: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1843, 0);  mul_1843 = None
    unsqueeze_1554: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1553, 2);  unsqueeze_1553 = None
    unsqueeze_1555: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, 3);  unsqueeze_1554 = None
    mul_1844: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_458, unsqueeze_1552);  sub_458 = unsqueeze_1552 = None
    sub_460: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_71, mul_1844);  where_71 = mul_1844 = None
    sub_461: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_460, unsqueeze_1549);  sub_460 = unsqueeze_1549 = None
    mul_1845: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_461, unsqueeze_1555);  sub_461 = unsqueeze_1555 = None
    mul_1846: "f32[104]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_292);  sum_147 = squeeze_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1845, add_534, primals_292, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1845 = add_534 = primals_292 = None
    getitem_1218: "f32[8, 104, 14, 14]" = convolution_backward_72[0]
    getitem_1219: "f32[104, 104, 3, 3]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_981: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_58, getitem_1218);  slice_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_72: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_72, full_default, add_981);  le_72 = add_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_148: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_462: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_1558);  convolution_96 = unsqueeze_1558 = None
    mul_1847: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_72, sub_462)
    sum_149: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1847, [0, 2, 3]);  mul_1847 = None
    mul_1848: "f32[104]" = torch.ops.aten.mul.Tensor(sum_148, 0.0006377551020408163)
    unsqueeze_1559: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1848, 0);  mul_1848 = None
    unsqueeze_1560: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1559, 2);  unsqueeze_1559 = None
    unsqueeze_1561: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1560, 3);  unsqueeze_1560 = None
    mul_1849: "f32[104]" = torch.ops.aten.mul.Tensor(sum_149, 0.0006377551020408163)
    mul_1850: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_289, squeeze_289)
    mul_1851: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1849, mul_1850);  mul_1849 = mul_1850 = None
    unsqueeze_1562: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1851, 0);  mul_1851 = None
    unsqueeze_1563: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1562, 2);  unsqueeze_1562 = None
    unsqueeze_1564: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1563, 3);  unsqueeze_1563 = None
    mul_1852: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_289, primals_290);  primals_290 = None
    unsqueeze_1565: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1852, 0);  mul_1852 = None
    unsqueeze_1566: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1565, 2);  unsqueeze_1565 = None
    unsqueeze_1567: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, 3);  unsqueeze_1566 = None
    mul_1853: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_462, unsqueeze_1564);  sub_462 = unsqueeze_1564 = None
    sub_464: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_72, mul_1853);  where_72 = mul_1853 = None
    sub_465: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_464, unsqueeze_1561);  sub_464 = unsqueeze_1561 = None
    mul_1854: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_465, unsqueeze_1567);  sub_465 = unsqueeze_1567 = None
    mul_1855: "f32[104]" = torch.ops.aten.mul.Tensor(sum_149, squeeze_289);  sum_149 = squeeze_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1854, add_528, primals_289, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1854 = add_528 = primals_289 = None
    getitem_1221: "f32[8, 104, 14, 14]" = convolution_backward_73[0]
    getitem_1222: "f32[104, 104, 3, 3]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_982: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_57, getitem_1221);  slice_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_73: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_73, full_default, add_982);  le_73 = add_982 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_150: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_466: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_1570);  convolution_95 = unsqueeze_1570 = None
    mul_1856: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_73, sub_466)
    sum_151: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1856, [0, 2, 3]);  mul_1856 = None
    mul_1857: "f32[104]" = torch.ops.aten.mul.Tensor(sum_150, 0.0006377551020408163)
    unsqueeze_1571: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1857, 0);  mul_1857 = None
    unsqueeze_1572: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1571, 2);  unsqueeze_1571 = None
    unsqueeze_1573: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1572, 3);  unsqueeze_1572 = None
    mul_1858: "f32[104]" = torch.ops.aten.mul.Tensor(sum_151, 0.0006377551020408163)
    mul_1859: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
    mul_1860: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1858, mul_1859);  mul_1858 = mul_1859 = None
    unsqueeze_1574: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1860, 0);  mul_1860 = None
    unsqueeze_1575: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1574, 2);  unsqueeze_1574 = None
    unsqueeze_1576: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1575, 3);  unsqueeze_1575 = None
    mul_1861: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_286, primals_287);  primals_287 = None
    unsqueeze_1577: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1861, 0);  mul_1861 = None
    unsqueeze_1578: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1577, 2);  unsqueeze_1577 = None
    unsqueeze_1579: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, 3);  unsqueeze_1578 = None
    mul_1862: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_466, unsqueeze_1576);  sub_466 = unsqueeze_1576 = None
    sub_468: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_73, mul_1862);  where_73 = mul_1862 = None
    sub_469: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_468, unsqueeze_1573);  sub_468 = unsqueeze_1573 = None
    mul_1863: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_469, unsqueeze_1579);  sub_469 = unsqueeze_1579 = None
    mul_1864: "f32[104]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_286);  sum_151 = squeeze_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1863, getitem_556, primals_286, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1863 = getitem_556 = primals_286 = None
    getitem_1224: "f32[8, 104, 14, 14]" = convolution_backward_74[0]
    getitem_1225: "f32[104, 104, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_47: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1224, getitem_1221, getitem_1218, slice_60], 1);  getitem_1224 = getitem_1221 = getitem_1218 = slice_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_74: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_74, full_default, cat_47);  le_74 = cat_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_152: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_470: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_1582);  convolution_94 = unsqueeze_1582 = None
    mul_1865: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_74, sub_470)
    sum_153: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1865, [0, 2, 3]);  mul_1865 = None
    mul_1866: "f32[416]" = torch.ops.aten.mul.Tensor(sum_152, 0.0006377551020408163)
    unsqueeze_1583: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1866, 0);  mul_1866 = None
    unsqueeze_1584: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1583, 2);  unsqueeze_1583 = None
    unsqueeze_1585: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1584, 3);  unsqueeze_1584 = None
    mul_1867: "f32[416]" = torch.ops.aten.mul.Tensor(sum_153, 0.0006377551020408163)
    mul_1868: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
    mul_1869: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1867, mul_1868);  mul_1867 = mul_1868 = None
    unsqueeze_1586: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1869, 0);  mul_1869 = None
    unsqueeze_1587: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1586, 2);  unsqueeze_1586 = None
    unsqueeze_1588: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1587, 3);  unsqueeze_1587 = None
    mul_1870: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_283, primals_284);  primals_284 = None
    unsqueeze_1589: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1870, 0);  mul_1870 = None
    unsqueeze_1590: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1589, 2);  unsqueeze_1589 = None
    unsqueeze_1591: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, 3);  unsqueeze_1590 = None
    mul_1871: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_470, unsqueeze_1588);  sub_470 = unsqueeze_1588 = None
    sub_472: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_74, mul_1871);  where_74 = mul_1871 = None
    sub_473: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_472, unsqueeze_1585);  sub_472 = unsqueeze_1585 = None
    mul_1872: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_473, unsqueeze_1591);  sub_473 = unsqueeze_1591 = None
    mul_1873: "f32[416]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_283);  sum_153 = squeeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1872, relu_90, primals_283, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1872 = primals_283 = None
    getitem_1227: "f32[8, 1024, 14, 14]" = convolution_backward_75[0]
    getitem_1228: "f32[416, 1024, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_983: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_70, getitem_1227);  where_70 = getitem_1227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_75: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_90, 0);  relu_90 = None
    where_75: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_75, full_default, add_983);  le_75 = add_983 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_154: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_474: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_1594);  convolution_93 = unsqueeze_1594 = None
    mul_1874: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_75, sub_474)
    sum_155: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1874, [0, 2, 3]);  mul_1874 = None
    mul_1875: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_154, 0.0006377551020408163)
    unsqueeze_1595: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1875, 0);  mul_1875 = None
    unsqueeze_1596: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1595, 2);  unsqueeze_1595 = None
    unsqueeze_1597: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1596, 3);  unsqueeze_1596 = None
    mul_1876: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_155, 0.0006377551020408163)
    mul_1877: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
    mul_1878: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1876, mul_1877);  mul_1876 = mul_1877 = None
    unsqueeze_1598: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1878, 0);  mul_1878 = None
    unsqueeze_1599: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1598, 2);  unsqueeze_1598 = None
    unsqueeze_1600: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1599, 3);  unsqueeze_1599 = None
    mul_1879: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_280, primals_281);  primals_281 = None
    unsqueeze_1601: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1879, 0);  mul_1879 = None
    unsqueeze_1602: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1601, 2);  unsqueeze_1601 = None
    unsqueeze_1603: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, 3);  unsqueeze_1602 = None
    mul_1880: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_474, unsqueeze_1600);  sub_474 = unsqueeze_1600 = None
    sub_476: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_75, mul_1880);  mul_1880 = None
    sub_477: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_476, unsqueeze_1597);  sub_476 = unsqueeze_1597 = None
    mul_1881: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_477, unsqueeze_1603);  sub_477 = unsqueeze_1603 = None
    mul_1882: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_280);  sum_155 = squeeze_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1881, cat_17, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1881 = cat_17 = primals_280 = None
    getitem_1230: "f32[8, 416, 14, 14]" = convolution_backward_76[0]
    getitem_1231: "f32[1024, 416, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_61: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1230, 1, 0, 104)
    slice_62: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1230, 1, 104, 208)
    slice_63: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1230, 1, 208, 312)
    slice_64: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1230, 1, 312, 416);  getitem_1230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_76: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_76, full_default, slice_63);  le_76 = slice_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_156: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_478: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_1606);  convolution_92 = unsqueeze_1606 = None
    mul_1883: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_76, sub_478)
    sum_157: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1883, [0, 2, 3]);  mul_1883 = None
    mul_1884: "f32[104]" = torch.ops.aten.mul.Tensor(sum_156, 0.0006377551020408163)
    unsqueeze_1607: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1884, 0);  mul_1884 = None
    unsqueeze_1608: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1607, 2);  unsqueeze_1607 = None
    unsqueeze_1609: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1608, 3);  unsqueeze_1608 = None
    mul_1885: "f32[104]" = torch.ops.aten.mul.Tensor(sum_157, 0.0006377551020408163)
    mul_1886: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
    mul_1887: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1885, mul_1886);  mul_1885 = mul_1886 = None
    unsqueeze_1610: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1887, 0);  mul_1887 = None
    unsqueeze_1611: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1610, 2);  unsqueeze_1610 = None
    unsqueeze_1612: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1611, 3);  unsqueeze_1611 = None
    mul_1888: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_277, primals_278);  primals_278 = None
    unsqueeze_1613: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1888, 0);  mul_1888 = None
    unsqueeze_1614: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1613, 2);  unsqueeze_1613 = None
    unsqueeze_1615: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, 3);  unsqueeze_1614 = None
    mul_1889: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_478, unsqueeze_1612);  sub_478 = unsqueeze_1612 = None
    sub_480: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_76, mul_1889);  where_76 = mul_1889 = None
    sub_481: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_480, unsqueeze_1609);  sub_480 = unsqueeze_1609 = None
    mul_1890: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_481, unsqueeze_1615);  sub_481 = unsqueeze_1615 = None
    mul_1891: "f32[104]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_277);  sum_157 = squeeze_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1890, add_506, primals_277, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1890 = add_506 = primals_277 = None
    getitem_1233: "f32[8, 104, 14, 14]" = convolution_backward_77[0]
    getitem_1234: "f32[104, 104, 3, 3]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_984: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_62, getitem_1233);  slice_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_77: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_77, full_default, add_984);  le_77 = add_984 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_158: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_482: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_1618);  convolution_91 = unsqueeze_1618 = None
    mul_1892: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_77, sub_482)
    sum_159: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1892, [0, 2, 3]);  mul_1892 = None
    mul_1893: "f32[104]" = torch.ops.aten.mul.Tensor(sum_158, 0.0006377551020408163)
    unsqueeze_1619: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1893, 0);  mul_1893 = None
    unsqueeze_1620: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1619, 2);  unsqueeze_1619 = None
    unsqueeze_1621: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1620, 3);  unsqueeze_1620 = None
    mul_1894: "f32[104]" = torch.ops.aten.mul.Tensor(sum_159, 0.0006377551020408163)
    mul_1895: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
    mul_1896: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1894, mul_1895);  mul_1894 = mul_1895 = None
    unsqueeze_1622: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1896, 0);  mul_1896 = None
    unsqueeze_1623: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1622, 2);  unsqueeze_1622 = None
    unsqueeze_1624: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1623, 3);  unsqueeze_1623 = None
    mul_1897: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_274, primals_275);  primals_275 = None
    unsqueeze_1625: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1897, 0);  mul_1897 = None
    unsqueeze_1626: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1625, 2);  unsqueeze_1625 = None
    unsqueeze_1627: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, 3);  unsqueeze_1626 = None
    mul_1898: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_482, unsqueeze_1624);  sub_482 = unsqueeze_1624 = None
    sub_484: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_77, mul_1898);  where_77 = mul_1898 = None
    sub_485: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_484, unsqueeze_1621);  sub_484 = unsqueeze_1621 = None
    mul_1899: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_485, unsqueeze_1627);  sub_485 = unsqueeze_1627 = None
    mul_1900: "f32[104]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_274);  sum_159 = squeeze_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1899, add_500, primals_274, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1899 = add_500 = primals_274 = None
    getitem_1236: "f32[8, 104, 14, 14]" = convolution_backward_78[0]
    getitem_1237: "f32[104, 104, 3, 3]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_985: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_61, getitem_1236);  slice_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_78: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_78, full_default, add_985);  le_78 = add_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_160: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_486: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_1630);  convolution_90 = unsqueeze_1630 = None
    mul_1901: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_78, sub_486)
    sum_161: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1901, [0, 2, 3]);  mul_1901 = None
    mul_1902: "f32[104]" = torch.ops.aten.mul.Tensor(sum_160, 0.0006377551020408163)
    unsqueeze_1631: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1902, 0);  mul_1902 = None
    unsqueeze_1632: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1631, 2);  unsqueeze_1631 = None
    unsqueeze_1633: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1632, 3);  unsqueeze_1632 = None
    mul_1903: "f32[104]" = torch.ops.aten.mul.Tensor(sum_161, 0.0006377551020408163)
    mul_1904: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
    mul_1905: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1903, mul_1904);  mul_1903 = mul_1904 = None
    unsqueeze_1634: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1905, 0);  mul_1905 = None
    unsqueeze_1635: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1634, 2);  unsqueeze_1634 = None
    unsqueeze_1636: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1635, 3);  unsqueeze_1635 = None
    mul_1906: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_271, primals_272);  primals_272 = None
    unsqueeze_1637: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1906, 0);  mul_1906 = None
    unsqueeze_1638: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1637, 2);  unsqueeze_1637 = None
    unsqueeze_1639: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, 3);  unsqueeze_1638 = None
    mul_1907: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_486, unsqueeze_1636);  sub_486 = unsqueeze_1636 = None
    sub_488: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_78, mul_1907);  where_78 = mul_1907 = None
    sub_489: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_488, unsqueeze_1633);  sub_488 = unsqueeze_1633 = None
    mul_1908: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_489, unsqueeze_1639);  sub_489 = unsqueeze_1639 = None
    mul_1909: "f32[104]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_271);  sum_161 = squeeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1908, getitem_526, primals_271, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1908 = getitem_526 = primals_271 = None
    getitem_1239: "f32[8, 104, 14, 14]" = convolution_backward_79[0]
    getitem_1240: "f32[104, 104, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_48: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1239, getitem_1236, getitem_1233, slice_64], 1);  getitem_1239 = getitem_1236 = getitem_1233 = slice_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_79: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_79, full_default, cat_48);  le_79 = cat_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_162: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_490: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_1642);  convolution_89 = unsqueeze_1642 = None
    mul_1910: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_79, sub_490)
    sum_163: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1910, [0, 2, 3]);  mul_1910 = None
    mul_1911: "f32[416]" = torch.ops.aten.mul.Tensor(sum_162, 0.0006377551020408163)
    unsqueeze_1643: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1911, 0);  mul_1911 = None
    unsqueeze_1644: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1643, 2);  unsqueeze_1643 = None
    unsqueeze_1645: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1644, 3);  unsqueeze_1644 = None
    mul_1912: "f32[416]" = torch.ops.aten.mul.Tensor(sum_163, 0.0006377551020408163)
    mul_1913: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
    mul_1914: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1912, mul_1913);  mul_1912 = mul_1913 = None
    unsqueeze_1646: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1914, 0);  mul_1914 = None
    unsqueeze_1647: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1646, 2);  unsqueeze_1646 = None
    unsqueeze_1648: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1647, 3);  unsqueeze_1647 = None
    mul_1915: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_268, primals_269);  primals_269 = None
    unsqueeze_1649: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1915, 0);  mul_1915 = None
    unsqueeze_1650: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1649, 2);  unsqueeze_1649 = None
    unsqueeze_1651: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, 3);  unsqueeze_1650 = None
    mul_1916: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_490, unsqueeze_1648);  sub_490 = unsqueeze_1648 = None
    sub_492: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_79, mul_1916);  where_79 = mul_1916 = None
    sub_493: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_492, unsqueeze_1645);  sub_492 = unsqueeze_1645 = None
    mul_1917: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_493, unsqueeze_1651);  sub_493 = unsqueeze_1651 = None
    mul_1918: "f32[416]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_268);  sum_163 = squeeze_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1917, relu_85, primals_268, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1917 = primals_268 = None
    getitem_1242: "f32[8, 1024, 14, 14]" = convolution_backward_80[0]
    getitem_1243: "f32[416, 1024, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_986: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_75, getitem_1242);  where_75 = getitem_1242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_80: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_85, 0);  relu_85 = None
    where_80: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_80, full_default, add_986);  le_80 = add_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_164: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_494: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_1654);  convolution_88 = unsqueeze_1654 = None
    mul_1919: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_80, sub_494)
    sum_165: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1919, [0, 2, 3]);  mul_1919 = None
    mul_1920: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_164, 0.0006377551020408163)
    unsqueeze_1655: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1920, 0);  mul_1920 = None
    unsqueeze_1656: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1655, 2);  unsqueeze_1655 = None
    unsqueeze_1657: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1656, 3);  unsqueeze_1656 = None
    mul_1921: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_165, 0.0006377551020408163)
    mul_1922: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
    mul_1923: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1921, mul_1922);  mul_1921 = mul_1922 = None
    unsqueeze_1658: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1923, 0);  mul_1923 = None
    unsqueeze_1659: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1658, 2);  unsqueeze_1658 = None
    unsqueeze_1660: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1659, 3);  unsqueeze_1659 = None
    mul_1924: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_265, primals_266);  primals_266 = None
    unsqueeze_1661: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1924, 0);  mul_1924 = None
    unsqueeze_1662: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1661, 2);  unsqueeze_1661 = None
    unsqueeze_1663: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, 3);  unsqueeze_1662 = None
    mul_1925: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_494, unsqueeze_1660);  sub_494 = unsqueeze_1660 = None
    sub_496: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_80, mul_1925);  mul_1925 = None
    sub_497: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_496, unsqueeze_1657);  sub_496 = unsqueeze_1657 = None
    mul_1926: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_497, unsqueeze_1663);  sub_497 = unsqueeze_1663 = None
    mul_1927: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_265);  sum_165 = squeeze_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1926, cat_16, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1926 = cat_16 = primals_265 = None
    getitem_1245: "f32[8, 416, 14, 14]" = convolution_backward_81[0]
    getitem_1246: "f32[1024, 416, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_65: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1245, 1, 0, 104)
    slice_66: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1245, 1, 104, 208)
    slice_67: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1245, 1, 208, 312)
    slice_68: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1245, 1, 312, 416);  getitem_1245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_81: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_81, full_default, slice_67);  le_81 = slice_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_166: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
    sub_498: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_1666);  convolution_87 = unsqueeze_1666 = None
    mul_1928: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_81, sub_498)
    sum_167: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1928, [0, 2, 3]);  mul_1928 = None
    mul_1929: "f32[104]" = torch.ops.aten.mul.Tensor(sum_166, 0.0006377551020408163)
    unsqueeze_1667: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1929, 0);  mul_1929 = None
    unsqueeze_1668: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1667, 2);  unsqueeze_1667 = None
    unsqueeze_1669: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1668, 3);  unsqueeze_1668 = None
    mul_1930: "f32[104]" = torch.ops.aten.mul.Tensor(sum_167, 0.0006377551020408163)
    mul_1931: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
    mul_1932: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1930, mul_1931);  mul_1930 = mul_1931 = None
    unsqueeze_1670: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1932, 0);  mul_1932 = None
    unsqueeze_1671: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1670, 2);  unsqueeze_1670 = None
    unsqueeze_1672: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1671, 3);  unsqueeze_1671 = None
    mul_1933: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_262, primals_263);  primals_263 = None
    unsqueeze_1673: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1933, 0);  mul_1933 = None
    unsqueeze_1674: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1673, 2);  unsqueeze_1673 = None
    unsqueeze_1675: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1674, 3);  unsqueeze_1674 = None
    mul_1934: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_498, unsqueeze_1672);  sub_498 = unsqueeze_1672 = None
    sub_500: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_81, mul_1934);  where_81 = mul_1934 = None
    sub_501: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_500, unsqueeze_1669);  sub_500 = unsqueeze_1669 = None
    mul_1935: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_501, unsqueeze_1675);  sub_501 = unsqueeze_1675 = None
    mul_1936: "f32[104]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_262);  sum_167 = squeeze_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1935, add_478, primals_262, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1935 = add_478 = primals_262 = None
    getitem_1248: "f32[8, 104, 14, 14]" = convolution_backward_82[0]
    getitem_1249: "f32[104, 104, 3, 3]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_987: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_66, getitem_1248);  slice_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_82: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_82, full_default, add_987);  le_82 = add_987 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_168: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
    sub_502: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_1678);  convolution_86 = unsqueeze_1678 = None
    mul_1937: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_82, sub_502)
    sum_169: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1937, [0, 2, 3]);  mul_1937 = None
    mul_1938: "f32[104]" = torch.ops.aten.mul.Tensor(sum_168, 0.0006377551020408163)
    unsqueeze_1679: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1938, 0);  mul_1938 = None
    unsqueeze_1680: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1679, 2);  unsqueeze_1679 = None
    unsqueeze_1681: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1680, 3);  unsqueeze_1680 = None
    mul_1939: "f32[104]" = torch.ops.aten.mul.Tensor(sum_169, 0.0006377551020408163)
    mul_1940: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
    mul_1941: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1939, mul_1940);  mul_1939 = mul_1940 = None
    unsqueeze_1682: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1941, 0);  mul_1941 = None
    unsqueeze_1683: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1682, 2);  unsqueeze_1682 = None
    unsqueeze_1684: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1683, 3);  unsqueeze_1683 = None
    mul_1942: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_259, primals_260);  primals_260 = None
    unsqueeze_1685: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1942, 0);  mul_1942 = None
    unsqueeze_1686: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1685, 2);  unsqueeze_1685 = None
    unsqueeze_1687: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1686, 3);  unsqueeze_1686 = None
    mul_1943: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_502, unsqueeze_1684);  sub_502 = unsqueeze_1684 = None
    sub_504: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_82, mul_1943);  where_82 = mul_1943 = None
    sub_505: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_504, unsqueeze_1681);  sub_504 = unsqueeze_1681 = None
    mul_1944: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_505, unsqueeze_1687);  sub_505 = unsqueeze_1687 = None
    mul_1945: "f32[104]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_259);  sum_169 = squeeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1944, add_472, primals_259, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1944 = add_472 = primals_259 = None
    getitem_1251: "f32[8, 104, 14, 14]" = convolution_backward_83[0]
    getitem_1252: "f32[104, 104, 3, 3]" = convolution_backward_83[1];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_988: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_65, getitem_1251);  slice_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_83: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_83, full_default, add_988);  le_83 = add_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_170: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
    sub_506: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_1690);  convolution_85 = unsqueeze_1690 = None
    mul_1946: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_83, sub_506)
    sum_171: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1946, [0, 2, 3]);  mul_1946 = None
    mul_1947: "f32[104]" = torch.ops.aten.mul.Tensor(sum_170, 0.0006377551020408163)
    unsqueeze_1691: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1947, 0);  mul_1947 = None
    unsqueeze_1692: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1691, 2);  unsqueeze_1691 = None
    unsqueeze_1693: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1692, 3);  unsqueeze_1692 = None
    mul_1948: "f32[104]" = torch.ops.aten.mul.Tensor(sum_171, 0.0006377551020408163)
    mul_1949: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
    mul_1950: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1948, mul_1949);  mul_1948 = mul_1949 = None
    unsqueeze_1694: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1950, 0);  mul_1950 = None
    unsqueeze_1695: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1694, 2);  unsqueeze_1694 = None
    unsqueeze_1696: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1695, 3);  unsqueeze_1695 = None
    mul_1951: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_256, primals_257);  primals_257 = None
    unsqueeze_1697: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1951, 0);  mul_1951 = None
    unsqueeze_1698: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1697, 2);  unsqueeze_1697 = None
    unsqueeze_1699: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1698, 3);  unsqueeze_1698 = None
    mul_1952: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_506, unsqueeze_1696);  sub_506 = unsqueeze_1696 = None
    sub_508: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_83, mul_1952);  where_83 = mul_1952 = None
    sub_509: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_508, unsqueeze_1693);  sub_508 = unsqueeze_1693 = None
    mul_1953: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_509, unsqueeze_1699);  sub_509 = unsqueeze_1699 = None
    mul_1954: "f32[104]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_256);  sum_171 = squeeze_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1953, getitem_496, primals_256, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1953 = getitem_496 = primals_256 = None
    getitem_1254: "f32[8, 104, 14, 14]" = convolution_backward_84[0]
    getitem_1255: "f32[104, 104, 3, 3]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_49: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1254, getitem_1251, getitem_1248, slice_68], 1);  getitem_1254 = getitem_1251 = getitem_1248 = slice_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_84: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_84, full_default, cat_49);  le_84 = cat_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_172: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
    sub_510: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_1702);  convolution_84 = unsqueeze_1702 = None
    mul_1955: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_84, sub_510)
    sum_173: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1955, [0, 2, 3]);  mul_1955 = None
    mul_1956: "f32[416]" = torch.ops.aten.mul.Tensor(sum_172, 0.0006377551020408163)
    unsqueeze_1703: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1956, 0);  mul_1956 = None
    unsqueeze_1704: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1703, 2);  unsqueeze_1703 = None
    unsqueeze_1705: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1704, 3);  unsqueeze_1704 = None
    mul_1957: "f32[416]" = torch.ops.aten.mul.Tensor(sum_173, 0.0006377551020408163)
    mul_1958: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
    mul_1959: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1957, mul_1958);  mul_1957 = mul_1958 = None
    unsqueeze_1706: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1959, 0);  mul_1959 = None
    unsqueeze_1707: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1706, 2);  unsqueeze_1706 = None
    unsqueeze_1708: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1707, 3);  unsqueeze_1707 = None
    mul_1960: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_254);  primals_254 = None
    unsqueeze_1709: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1960, 0);  mul_1960 = None
    unsqueeze_1710: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1709, 2);  unsqueeze_1709 = None
    unsqueeze_1711: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1710, 3);  unsqueeze_1710 = None
    mul_1961: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_510, unsqueeze_1708);  sub_510 = unsqueeze_1708 = None
    sub_512: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_84, mul_1961);  where_84 = mul_1961 = None
    sub_513: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_512, unsqueeze_1705);  sub_512 = unsqueeze_1705 = None
    mul_1962: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_513, unsqueeze_1711);  sub_513 = unsqueeze_1711 = None
    mul_1963: "f32[416]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_253);  sum_173 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1962, relu_80, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1962 = primals_253 = None
    getitem_1257: "f32[8, 1024, 14, 14]" = convolution_backward_85[0]
    getitem_1258: "f32[416, 1024, 1, 1]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_989: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_80, getitem_1257);  where_80 = getitem_1257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_85: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_80, 0);  relu_80 = None
    where_85: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_85, full_default, add_989);  le_85 = add_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_174: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_514: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_1714);  convolution_83 = unsqueeze_1714 = None
    mul_1964: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_85, sub_514)
    sum_175: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1964, [0, 2, 3]);  mul_1964 = None
    mul_1965: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_174, 0.0006377551020408163)
    unsqueeze_1715: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1965, 0);  mul_1965 = None
    unsqueeze_1716: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1715, 2);  unsqueeze_1715 = None
    unsqueeze_1717: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1716, 3);  unsqueeze_1716 = None
    mul_1966: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_175, 0.0006377551020408163)
    mul_1967: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
    mul_1968: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1966, mul_1967);  mul_1966 = mul_1967 = None
    unsqueeze_1718: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1968, 0);  mul_1968 = None
    unsqueeze_1719: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1718, 2);  unsqueeze_1718 = None
    unsqueeze_1720: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1719, 3);  unsqueeze_1719 = None
    mul_1969: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_251);  primals_251 = None
    unsqueeze_1721: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1969, 0);  mul_1969 = None
    unsqueeze_1722: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1721, 2);  unsqueeze_1721 = None
    unsqueeze_1723: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1722, 3);  unsqueeze_1722 = None
    mul_1970: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_514, unsqueeze_1720);  sub_514 = unsqueeze_1720 = None
    sub_516: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_85, mul_1970);  mul_1970 = None
    sub_517: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_516, unsqueeze_1717);  sub_516 = unsqueeze_1717 = None
    mul_1971: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_517, unsqueeze_1723);  sub_517 = unsqueeze_1723 = None
    mul_1972: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_250);  sum_175 = squeeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1971, cat_15, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1971 = cat_15 = primals_250 = None
    getitem_1260: "f32[8, 416, 14, 14]" = convolution_backward_86[0]
    getitem_1261: "f32[1024, 416, 1, 1]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_69: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1260, 1, 0, 104)
    slice_70: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1260, 1, 104, 208)
    slice_71: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1260, 1, 208, 312)
    slice_72: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1260, 1, 312, 416);  getitem_1260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_86: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_86, full_default, slice_71);  le_86 = slice_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_176: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
    sub_518: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_1726);  convolution_82 = unsqueeze_1726 = None
    mul_1973: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_86, sub_518)
    sum_177: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1973, [0, 2, 3]);  mul_1973 = None
    mul_1974: "f32[104]" = torch.ops.aten.mul.Tensor(sum_176, 0.0006377551020408163)
    unsqueeze_1727: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1974, 0);  mul_1974 = None
    unsqueeze_1728: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1727, 2);  unsqueeze_1727 = None
    unsqueeze_1729: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1728, 3);  unsqueeze_1728 = None
    mul_1975: "f32[104]" = torch.ops.aten.mul.Tensor(sum_177, 0.0006377551020408163)
    mul_1976: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
    mul_1977: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1975, mul_1976);  mul_1975 = mul_1976 = None
    unsqueeze_1730: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1977, 0);  mul_1977 = None
    unsqueeze_1731: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1730, 2);  unsqueeze_1730 = None
    unsqueeze_1732: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1731, 3);  unsqueeze_1731 = None
    mul_1978: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_248);  primals_248 = None
    unsqueeze_1733: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1978, 0);  mul_1978 = None
    unsqueeze_1734: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1733, 2);  unsqueeze_1733 = None
    unsqueeze_1735: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1734, 3);  unsqueeze_1734 = None
    mul_1979: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_518, unsqueeze_1732);  sub_518 = unsqueeze_1732 = None
    sub_520: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_86, mul_1979);  where_86 = mul_1979 = None
    sub_521: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_520, unsqueeze_1729);  sub_520 = unsqueeze_1729 = None
    mul_1980: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_521, unsqueeze_1735);  sub_521 = unsqueeze_1735 = None
    mul_1981: "f32[104]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_247);  sum_177 = squeeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1980, add_450, primals_247, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1980 = add_450 = primals_247 = None
    getitem_1263: "f32[8, 104, 14, 14]" = convolution_backward_87[0]
    getitem_1264: "f32[104, 104, 3, 3]" = convolution_backward_87[1];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_990: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_70, getitem_1263);  slice_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_87: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_87, full_default, add_990);  le_87 = add_990 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_178: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_522: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_1738);  convolution_81 = unsqueeze_1738 = None
    mul_1982: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_87, sub_522)
    sum_179: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1982, [0, 2, 3]);  mul_1982 = None
    mul_1983: "f32[104]" = torch.ops.aten.mul.Tensor(sum_178, 0.0006377551020408163)
    unsqueeze_1739: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1983, 0);  mul_1983 = None
    unsqueeze_1740: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1739, 2);  unsqueeze_1739 = None
    unsqueeze_1741: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1740, 3);  unsqueeze_1740 = None
    mul_1984: "f32[104]" = torch.ops.aten.mul.Tensor(sum_179, 0.0006377551020408163)
    mul_1985: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
    mul_1986: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1984, mul_1985);  mul_1984 = mul_1985 = None
    unsqueeze_1742: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1986, 0);  mul_1986 = None
    unsqueeze_1743: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1742, 2);  unsqueeze_1742 = None
    unsqueeze_1744: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1743, 3);  unsqueeze_1743 = None
    mul_1987: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_245);  primals_245 = None
    unsqueeze_1745: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1987, 0);  mul_1987 = None
    unsqueeze_1746: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1745, 2);  unsqueeze_1745 = None
    unsqueeze_1747: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1746, 3);  unsqueeze_1746 = None
    mul_1988: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_522, unsqueeze_1744);  sub_522 = unsqueeze_1744 = None
    sub_524: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_87, mul_1988);  where_87 = mul_1988 = None
    sub_525: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_524, unsqueeze_1741);  sub_524 = unsqueeze_1741 = None
    mul_1989: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_525, unsqueeze_1747);  sub_525 = unsqueeze_1747 = None
    mul_1990: "f32[104]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_244);  sum_179 = squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1989, add_444, primals_244, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1989 = add_444 = primals_244 = None
    getitem_1266: "f32[8, 104, 14, 14]" = convolution_backward_88[0]
    getitem_1267: "f32[104, 104, 3, 3]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_991: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_69, getitem_1266);  slice_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_88: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_88, full_default, add_991);  le_88 = add_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_180: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_526: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_1750);  convolution_80 = unsqueeze_1750 = None
    mul_1991: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_88, sub_526)
    sum_181: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_1991, [0, 2, 3]);  mul_1991 = None
    mul_1992: "f32[104]" = torch.ops.aten.mul.Tensor(sum_180, 0.0006377551020408163)
    unsqueeze_1751: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1992, 0);  mul_1992 = None
    unsqueeze_1752: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1751, 2);  unsqueeze_1751 = None
    unsqueeze_1753: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1752, 3);  unsqueeze_1752 = None
    mul_1993: "f32[104]" = torch.ops.aten.mul.Tensor(sum_181, 0.0006377551020408163)
    mul_1994: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
    mul_1995: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1993, mul_1994);  mul_1993 = mul_1994 = None
    unsqueeze_1754: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1995, 0);  mul_1995 = None
    unsqueeze_1755: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1754, 2);  unsqueeze_1754 = None
    unsqueeze_1756: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1755, 3);  unsqueeze_1755 = None
    mul_1996: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_242);  primals_242 = None
    unsqueeze_1757: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_1996, 0);  mul_1996 = None
    unsqueeze_1758: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1757, 2);  unsqueeze_1757 = None
    unsqueeze_1759: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1758, 3);  unsqueeze_1758 = None
    mul_1997: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_526, unsqueeze_1756);  sub_526 = unsqueeze_1756 = None
    sub_528: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_88, mul_1997);  where_88 = mul_1997 = None
    sub_529: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_528, unsqueeze_1753);  sub_528 = unsqueeze_1753 = None
    mul_1998: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_529, unsqueeze_1759);  sub_529 = unsqueeze_1759 = None
    mul_1999: "f32[104]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_241);  sum_181 = squeeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1998, getitem_466, primals_241, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1998 = getitem_466 = primals_241 = None
    getitem_1269: "f32[8, 104, 14, 14]" = convolution_backward_89[0]
    getitem_1270: "f32[104, 104, 3, 3]" = convolution_backward_89[1];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_50: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1269, getitem_1266, getitem_1263, slice_72], 1);  getitem_1269 = getitem_1266 = getitem_1263 = slice_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_89: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_89, full_default, cat_50);  le_89 = cat_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_182: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
    sub_530: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_1762);  convolution_79 = unsqueeze_1762 = None
    mul_2000: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_89, sub_530)
    sum_183: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_2000, [0, 2, 3]);  mul_2000 = None
    mul_2001: "f32[416]" = torch.ops.aten.mul.Tensor(sum_182, 0.0006377551020408163)
    unsqueeze_1763: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2001, 0);  mul_2001 = None
    unsqueeze_1764: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1763, 2);  unsqueeze_1763 = None
    unsqueeze_1765: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1764, 3);  unsqueeze_1764 = None
    mul_2002: "f32[416]" = torch.ops.aten.mul.Tensor(sum_183, 0.0006377551020408163)
    mul_2003: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_2004: "f32[416]" = torch.ops.aten.mul.Tensor(mul_2002, mul_2003);  mul_2002 = mul_2003 = None
    unsqueeze_1766: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2004, 0);  mul_2004 = None
    unsqueeze_1767: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1766, 2);  unsqueeze_1766 = None
    unsqueeze_1768: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1767, 3);  unsqueeze_1767 = None
    mul_2005: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_239);  primals_239 = None
    unsqueeze_1769: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2005, 0);  mul_2005 = None
    unsqueeze_1770: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1769, 2);  unsqueeze_1769 = None
    unsqueeze_1771: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1770, 3);  unsqueeze_1770 = None
    mul_2006: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_530, unsqueeze_1768);  sub_530 = unsqueeze_1768 = None
    sub_532: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_89, mul_2006);  where_89 = mul_2006 = None
    sub_533: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_532, unsqueeze_1765);  sub_532 = unsqueeze_1765 = None
    mul_2007: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_533, unsqueeze_1771);  sub_533 = unsqueeze_1771 = None
    mul_2008: "f32[416]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_238);  sum_183 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_2007, relu_75, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2007 = primals_238 = None
    getitem_1272: "f32[8, 1024, 14, 14]" = convolution_backward_90[0]
    getitem_1273: "f32[416, 1024, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_992: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_85, getitem_1272);  where_85 = getitem_1272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_90: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
    where_90: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_90, full_default, add_992);  le_90 = add_992 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_184: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
    sub_534: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_1774);  convolution_78 = unsqueeze_1774 = None
    mul_2009: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_90, sub_534)
    sum_185: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_2009, [0, 2, 3]);  mul_2009 = None
    mul_2010: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_184, 0.0006377551020408163)
    unsqueeze_1775: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2010, 0);  mul_2010 = None
    unsqueeze_1776: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1775, 2);  unsqueeze_1775 = None
    unsqueeze_1777: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1776, 3);  unsqueeze_1776 = None
    mul_2011: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_185, 0.0006377551020408163)
    mul_2012: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_2013: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_2011, mul_2012);  mul_2011 = mul_2012 = None
    unsqueeze_1778: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2013, 0);  mul_2013 = None
    unsqueeze_1779: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1778, 2);  unsqueeze_1778 = None
    unsqueeze_1780: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1779, 3);  unsqueeze_1779 = None
    mul_2014: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_236);  primals_236 = None
    unsqueeze_1781: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2014, 0);  mul_2014 = None
    unsqueeze_1782: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1781, 2);  unsqueeze_1781 = None
    unsqueeze_1783: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1782, 3);  unsqueeze_1782 = None
    mul_2015: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_534, unsqueeze_1780);  sub_534 = unsqueeze_1780 = None
    sub_536: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_90, mul_2015);  mul_2015 = None
    sub_537: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_536, unsqueeze_1777);  sub_536 = unsqueeze_1777 = None
    mul_2016: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_537, unsqueeze_1783);  sub_537 = unsqueeze_1783 = None
    mul_2017: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_185, squeeze_235);  sum_185 = squeeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_2016, cat_14, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2016 = cat_14 = primals_235 = None
    getitem_1275: "f32[8, 416, 14, 14]" = convolution_backward_91[0]
    getitem_1276: "f32[1024, 416, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_73: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1275, 1, 0, 104)
    slice_74: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1275, 1, 104, 208)
    slice_75: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1275, 1, 208, 312)
    slice_76: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1275, 1, 312, 416);  getitem_1275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_91: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_91, full_default, slice_75);  le_91 = slice_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_186: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
    sub_538: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_1786);  convolution_77 = unsqueeze_1786 = None
    mul_2018: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_91, sub_538)
    sum_187: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2018, [0, 2, 3]);  mul_2018 = None
    mul_2019: "f32[104]" = torch.ops.aten.mul.Tensor(sum_186, 0.0006377551020408163)
    unsqueeze_1787: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2019, 0);  mul_2019 = None
    unsqueeze_1788: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1787, 2);  unsqueeze_1787 = None
    unsqueeze_1789: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1788, 3);  unsqueeze_1788 = None
    mul_2020: "f32[104]" = torch.ops.aten.mul.Tensor(sum_187, 0.0006377551020408163)
    mul_2021: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_2022: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2020, mul_2021);  mul_2020 = mul_2021 = None
    unsqueeze_1790: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2022, 0);  mul_2022 = None
    unsqueeze_1791: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1790, 2);  unsqueeze_1790 = None
    unsqueeze_1792: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1791, 3);  unsqueeze_1791 = None
    mul_2023: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_233);  primals_233 = None
    unsqueeze_1793: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2023, 0);  mul_2023 = None
    unsqueeze_1794: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1793, 2);  unsqueeze_1793 = None
    unsqueeze_1795: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1794, 3);  unsqueeze_1794 = None
    mul_2024: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_538, unsqueeze_1792);  sub_538 = unsqueeze_1792 = None
    sub_540: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_91, mul_2024);  where_91 = mul_2024 = None
    sub_541: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_540, unsqueeze_1789);  sub_540 = unsqueeze_1789 = None
    mul_2025: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_541, unsqueeze_1795);  sub_541 = unsqueeze_1795 = None
    mul_2026: "f32[104]" = torch.ops.aten.mul.Tensor(sum_187, squeeze_232);  sum_187 = squeeze_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_2025, add_422, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2025 = add_422 = primals_232 = None
    getitem_1278: "f32[8, 104, 14, 14]" = convolution_backward_92[0]
    getitem_1279: "f32[104, 104, 3, 3]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_993: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_74, getitem_1278);  slice_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_92: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_92, full_default, add_993);  le_92 = add_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_188: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_542: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_1798);  convolution_76 = unsqueeze_1798 = None
    mul_2027: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_92, sub_542)
    sum_189: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2027, [0, 2, 3]);  mul_2027 = None
    mul_2028: "f32[104]" = torch.ops.aten.mul.Tensor(sum_188, 0.0006377551020408163)
    unsqueeze_1799: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2028, 0);  mul_2028 = None
    unsqueeze_1800: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1799, 2);  unsqueeze_1799 = None
    unsqueeze_1801: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1800, 3);  unsqueeze_1800 = None
    mul_2029: "f32[104]" = torch.ops.aten.mul.Tensor(sum_189, 0.0006377551020408163)
    mul_2030: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_2031: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2029, mul_2030);  mul_2029 = mul_2030 = None
    unsqueeze_1802: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2031, 0);  mul_2031 = None
    unsqueeze_1803: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1802, 2);  unsqueeze_1802 = None
    unsqueeze_1804: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1803, 3);  unsqueeze_1803 = None
    mul_2032: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_230);  primals_230 = None
    unsqueeze_1805: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2032, 0);  mul_2032 = None
    unsqueeze_1806: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1805, 2);  unsqueeze_1805 = None
    unsqueeze_1807: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1806, 3);  unsqueeze_1806 = None
    mul_2033: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_542, unsqueeze_1804);  sub_542 = unsqueeze_1804 = None
    sub_544: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_92, mul_2033);  where_92 = mul_2033 = None
    sub_545: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_544, unsqueeze_1801);  sub_544 = unsqueeze_1801 = None
    mul_2034: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_545, unsqueeze_1807);  sub_545 = unsqueeze_1807 = None
    mul_2035: "f32[104]" = torch.ops.aten.mul.Tensor(sum_189, squeeze_229);  sum_189 = squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_2034, add_416, primals_229, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2034 = add_416 = primals_229 = None
    getitem_1281: "f32[8, 104, 14, 14]" = convolution_backward_93[0]
    getitem_1282: "f32[104, 104, 3, 3]" = convolution_backward_93[1];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_994: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_73, getitem_1281);  slice_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_93: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_93, full_default, add_994);  le_93 = add_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_190: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
    sub_546: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_1810);  convolution_75 = unsqueeze_1810 = None
    mul_2036: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_93, sub_546)
    sum_191: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2036, [0, 2, 3]);  mul_2036 = None
    mul_2037: "f32[104]" = torch.ops.aten.mul.Tensor(sum_190, 0.0006377551020408163)
    unsqueeze_1811: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2037, 0);  mul_2037 = None
    unsqueeze_1812: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1811, 2);  unsqueeze_1811 = None
    unsqueeze_1813: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1812, 3);  unsqueeze_1812 = None
    mul_2038: "f32[104]" = torch.ops.aten.mul.Tensor(sum_191, 0.0006377551020408163)
    mul_2039: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_2040: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2038, mul_2039);  mul_2038 = mul_2039 = None
    unsqueeze_1814: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2040, 0);  mul_2040 = None
    unsqueeze_1815: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1814, 2);  unsqueeze_1814 = None
    unsqueeze_1816: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1815, 3);  unsqueeze_1815 = None
    mul_2041: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_227);  primals_227 = None
    unsqueeze_1817: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2041, 0);  mul_2041 = None
    unsqueeze_1818: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1817, 2);  unsqueeze_1817 = None
    unsqueeze_1819: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1818, 3);  unsqueeze_1818 = None
    mul_2042: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_546, unsqueeze_1816);  sub_546 = unsqueeze_1816 = None
    sub_548: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_93, mul_2042);  where_93 = mul_2042 = None
    sub_549: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_548, unsqueeze_1813);  sub_548 = unsqueeze_1813 = None
    mul_2043: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_549, unsqueeze_1819);  sub_549 = unsqueeze_1819 = None
    mul_2044: "f32[104]" = torch.ops.aten.mul.Tensor(sum_191, squeeze_226);  sum_191 = squeeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_2043, getitem_436, primals_226, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2043 = getitem_436 = primals_226 = None
    getitem_1284: "f32[8, 104, 14, 14]" = convolution_backward_94[0]
    getitem_1285: "f32[104, 104, 3, 3]" = convolution_backward_94[1];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_51: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1284, getitem_1281, getitem_1278, slice_76], 1);  getitem_1284 = getitem_1281 = getitem_1278 = slice_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_94: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_94, full_default, cat_51);  le_94 = cat_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_192: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_550: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_1822);  convolution_74 = unsqueeze_1822 = None
    mul_2045: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_94, sub_550)
    sum_193: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_2045, [0, 2, 3]);  mul_2045 = None
    mul_2046: "f32[416]" = torch.ops.aten.mul.Tensor(sum_192, 0.0006377551020408163)
    unsqueeze_1823: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2046, 0);  mul_2046 = None
    unsqueeze_1824: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1823, 2);  unsqueeze_1823 = None
    unsqueeze_1825: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1824, 3);  unsqueeze_1824 = None
    mul_2047: "f32[416]" = torch.ops.aten.mul.Tensor(sum_193, 0.0006377551020408163)
    mul_2048: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_2049: "f32[416]" = torch.ops.aten.mul.Tensor(mul_2047, mul_2048);  mul_2047 = mul_2048 = None
    unsqueeze_1826: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2049, 0);  mul_2049 = None
    unsqueeze_1827: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1826, 2);  unsqueeze_1826 = None
    unsqueeze_1828: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1827, 3);  unsqueeze_1827 = None
    mul_2050: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_224);  primals_224 = None
    unsqueeze_1829: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2050, 0);  mul_2050 = None
    unsqueeze_1830: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1829, 2);  unsqueeze_1829 = None
    unsqueeze_1831: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1830, 3);  unsqueeze_1830 = None
    mul_2051: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_550, unsqueeze_1828);  sub_550 = unsqueeze_1828 = None
    sub_552: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_94, mul_2051);  where_94 = mul_2051 = None
    sub_553: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_552, unsqueeze_1825);  sub_552 = unsqueeze_1825 = None
    mul_2052: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_553, unsqueeze_1831);  sub_553 = unsqueeze_1831 = None
    mul_2053: "f32[416]" = torch.ops.aten.mul.Tensor(sum_193, squeeze_223);  sum_193 = squeeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_2052, relu_70, primals_223, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2052 = primals_223 = None
    getitem_1287: "f32[8, 1024, 14, 14]" = convolution_backward_95[0]
    getitem_1288: "f32[416, 1024, 1, 1]" = convolution_backward_95[1];  convolution_backward_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_995: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_90, getitem_1287);  where_90 = getitem_1287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_95: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
    where_95: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_95, full_default, add_995);  le_95 = add_995 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_194: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
    sub_554: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_1834);  convolution_73 = unsqueeze_1834 = None
    mul_2054: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_95, sub_554)
    sum_195: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_2054, [0, 2, 3]);  mul_2054 = None
    mul_2055: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_194, 0.0006377551020408163)
    unsqueeze_1835: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2055, 0);  mul_2055 = None
    unsqueeze_1836: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1835, 2);  unsqueeze_1835 = None
    unsqueeze_1837: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1836, 3);  unsqueeze_1836 = None
    mul_2056: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_195, 0.0006377551020408163)
    mul_2057: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_2058: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_2056, mul_2057);  mul_2056 = mul_2057 = None
    unsqueeze_1838: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2058, 0);  mul_2058 = None
    unsqueeze_1839: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1838, 2);  unsqueeze_1838 = None
    unsqueeze_1840: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1839, 3);  unsqueeze_1839 = None
    mul_2059: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_221);  primals_221 = None
    unsqueeze_1841: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2059, 0);  mul_2059 = None
    unsqueeze_1842: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1841, 2);  unsqueeze_1841 = None
    unsqueeze_1843: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1842, 3);  unsqueeze_1842 = None
    mul_2060: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_554, unsqueeze_1840);  sub_554 = unsqueeze_1840 = None
    sub_556: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_95, mul_2060);  mul_2060 = None
    sub_557: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_556, unsqueeze_1837);  sub_556 = unsqueeze_1837 = None
    mul_2061: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_557, unsqueeze_1843);  sub_557 = unsqueeze_1843 = None
    mul_2062: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_195, squeeze_220);  sum_195 = squeeze_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(mul_2061, cat_13, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2061 = cat_13 = primals_220 = None
    getitem_1290: "f32[8, 416, 14, 14]" = convolution_backward_96[0]
    getitem_1291: "f32[1024, 416, 1, 1]" = convolution_backward_96[1];  convolution_backward_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_77: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1290, 1, 0, 104)
    slice_78: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1290, 1, 104, 208)
    slice_79: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1290, 1, 208, 312)
    slice_80: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1290, 1, 312, 416);  getitem_1290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_96: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_96, full_default, slice_79);  le_96 = slice_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_196: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_558: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_1846);  convolution_72 = unsqueeze_1846 = None
    mul_2063: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_96, sub_558)
    sum_197: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2063, [0, 2, 3]);  mul_2063 = None
    mul_2064: "f32[104]" = torch.ops.aten.mul.Tensor(sum_196, 0.0006377551020408163)
    unsqueeze_1847: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2064, 0);  mul_2064 = None
    unsqueeze_1848: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1847, 2);  unsqueeze_1847 = None
    unsqueeze_1849: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1848, 3);  unsqueeze_1848 = None
    mul_2065: "f32[104]" = torch.ops.aten.mul.Tensor(sum_197, 0.0006377551020408163)
    mul_2066: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_2067: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2065, mul_2066);  mul_2065 = mul_2066 = None
    unsqueeze_1850: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2067, 0);  mul_2067 = None
    unsqueeze_1851: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1850, 2);  unsqueeze_1850 = None
    unsqueeze_1852: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1851, 3);  unsqueeze_1851 = None
    mul_2068: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_218);  primals_218 = None
    unsqueeze_1853: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2068, 0);  mul_2068 = None
    unsqueeze_1854: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1853, 2);  unsqueeze_1853 = None
    unsqueeze_1855: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1854, 3);  unsqueeze_1854 = None
    mul_2069: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_558, unsqueeze_1852);  sub_558 = unsqueeze_1852 = None
    sub_560: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_96, mul_2069);  where_96 = mul_2069 = None
    sub_561: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_560, unsqueeze_1849);  sub_560 = unsqueeze_1849 = None
    mul_2070: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_561, unsqueeze_1855);  sub_561 = unsqueeze_1855 = None
    mul_2071: "f32[104]" = torch.ops.aten.mul.Tensor(sum_197, squeeze_217);  sum_197 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(mul_2070, add_394, primals_217, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2070 = add_394 = primals_217 = None
    getitem_1293: "f32[8, 104, 14, 14]" = convolution_backward_97[0]
    getitem_1294: "f32[104, 104, 3, 3]" = convolution_backward_97[1];  convolution_backward_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_996: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_78, getitem_1293);  slice_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_97: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_97, full_default, add_996);  le_97 = add_996 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_198: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_97, [0, 2, 3])
    sub_562: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_1858);  convolution_71 = unsqueeze_1858 = None
    mul_2072: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_97, sub_562)
    sum_199: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2072, [0, 2, 3]);  mul_2072 = None
    mul_2073: "f32[104]" = torch.ops.aten.mul.Tensor(sum_198, 0.0006377551020408163)
    unsqueeze_1859: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2073, 0);  mul_2073 = None
    unsqueeze_1860: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1859, 2);  unsqueeze_1859 = None
    unsqueeze_1861: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1860, 3);  unsqueeze_1860 = None
    mul_2074: "f32[104]" = torch.ops.aten.mul.Tensor(sum_199, 0.0006377551020408163)
    mul_2075: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_2076: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2074, mul_2075);  mul_2074 = mul_2075 = None
    unsqueeze_1862: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2076, 0);  mul_2076 = None
    unsqueeze_1863: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1862, 2);  unsqueeze_1862 = None
    unsqueeze_1864: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1863, 3);  unsqueeze_1863 = None
    mul_2077: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_215);  primals_215 = None
    unsqueeze_1865: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2077, 0);  mul_2077 = None
    unsqueeze_1866: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1865, 2);  unsqueeze_1865 = None
    unsqueeze_1867: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1866, 3);  unsqueeze_1866 = None
    mul_2078: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_562, unsqueeze_1864);  sub_562 = unsqueeze_1864 = None
    sub_564: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_97, mul_2078);  where_97 = mul_2078 = None
    sub_565: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_564, unsqueeze_1861);  sub_564 = unsqueeze_1861 = None
    mul_2079: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_565, unsqueeze_1867);  sub_565 = unsqueeze_1867 = None
    mul_2080: "f32[104]" = torch.ops.aten.mul.Tensor(sum_199, squeeze_214);  sum_199 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(mul_2079, add_388, primals_214, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2079 = add_388 = primals_214 = None
    getitem_1296: "f32[8, 104, 14, 14]" = convolution_backward_98[0]
    getitem_1297: "f32[104, 104, 3, 3]" = convolution_backward_98[1];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_997: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_77, getitem_1296);  slice_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_98: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_98, full_default, add_997);  le_98 = add_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_200: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_98, [0, 2, 3])
    sub_566: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_1870);  convolution_70 = unsqueeze_1870 = None
    mul_2081: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_98, sub_566)
    sum_201: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2081, [0, 2, 3]);  mul_2081 = None
    mul_2082: "f32[104]" = torch.ops.aten.mul.Tensor(sum_200, 0.0006377551020408163)
    unsqueeze_1871: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2082, 0);  mul_2082 = None
    unsqueeze_1872: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1871, 2);  unsqueeze_1871 = None
    unsqueeze_1873: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1872, 3);  unsqueeze_1872 = None
    mul_2083: "f32[104]" = torch.ops.aten.mul.Tensor(sum_201, 0.0006377551020408163)
    mul_2084: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_2085: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2083, mul_2084);  mul_2083 = mul_2084 = None
    unsqueeze_1874: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2085, 0);  mul_2085 = None
    unsqueeze_1875: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1874, 2);  unsqueeze_1874 = None
    unsqueeze_1876: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1875, 3);  unsqueeze_1875 = None
    mul_2086: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_212);  primals_212 = None
    unsqueeze_1877: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2086, 0);  mul_2086 = None
    unsqueeze_1878: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1877, 2);  unsqueeze_1877 = None
    unsqueeze_1879: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1878, 3);  unsqueeze_1878 = None
    mul_2087: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_566, unsqueeze_1876);  sub_566 = unsqueeze_1876 = None
    sub_568: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_98, mul_2087);  where_98 = mul_2087 = None
    sub_569: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_568, unsqueeze_1873);  sub_568 = unsqueeze_1873 = None
    mul_2088: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_569, unsqueeze_1879);  sub_569 = unsqueeze_1879 = None
    mul_2089: "f32[104]" = torch.ops.aten.mul.Tensor(sum_201, squeeze_211);  sum_201 = squeeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(mul_2088, getitem_406, primals_211, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2088 = getitem_406 = primals_211 = None
    getitem_1299: "f32[8, 104, 14, 14]" = convolution_backward_99[0]
    getitem_1300: "f32[104, 104, 3, 3]" = convolution_backward_99[1];  convolution_backward_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_52: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1299, getitem_1296, getitem_1293, slice_80], 1);  getitem_1299 = getitem_1296 = getitem_1293 = slice_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_99: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_99, full_default, cat_52);  le_99 = cat_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_202: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_99, [0, 2, 3])
    sub_570: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_1882);  convolution_69 = unsqueeze_1882 = None
    mul_2090: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_99, sub_570)
    sum_203: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_2090, [0, 2, 3]);  mul_2090 = None
    mul_2091: "f32[416]" = torch.ops.aten.mul.Tensor(sum_202, 0.0006377551020408163)
    unsqueeze_1883: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2091, 0);  mul_2091 = None
    unsqueeze_1884: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1883, 2);  unsqueeze_1883 = None
    unsqueeze_1885: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1884, 3);  unsqueeze_1884 = None
    mul_2092: "f32[416]" = torch.ops.aten.mul.Tensor(sum_203, 0.0006377551020408163)
    mul_2093: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_2094: "f32[416]" = torch.ops.aten.mul.Tensor(mul_2092, mul_2093);  mul_2092 = mul_2093 = None
    unsqueeze_1886: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2094, 0);  mul_2094 = None
    unsqueeze_1887: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1886, 2);  unsqueeze_1886 = None
    unsqueeze_1888: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1887, 3);  unsqueeze_1887 = None
    mul_2095: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_209);  primals_209 = None
    unsqueeze_1889: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2095, 0);  mul_2095 = None
    unsqueeze_1890: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1889, 2);  unsqueeze_1889 = None
    unsqueeze_1891: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1890, 3);  unsqueeze_1890 = None
    mul_2096: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_570, unsqueeze_1888);  sub_570 = unsqueeze_1888 = None
    sub_572: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_99, mul_2096);  where_99 = mul_2096 = None
    sub_573: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_572, unsqueeze_1885);  sub_572 = unsqueeze_1885 = None
    mul_2097: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_573, unsqueeze_1891);  sub_573 = unsqueeze_1891 = None
    mul_2098: "f32[416]" = torch.ops.aten.mul.Tensor(sum_203, squeeze_208);  sum_203 = squeeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(mul_2097, relu_65, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2097 = primals_208 = None
    getitem_1302: "f32[8, 1024, 14, 14]" = convolution_backward_100[0]
    getitem_1303: "f32[416, 1024, 1, 1]" = convolution_backward_100[1];  convolution_backward_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_998: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_95, getitem_1302);  where_95 = getitem_1302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_100: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
    where_100: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_100, full_default, add_998);  le_100 = add_998 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_204: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_100, [0, 2, 3])
    sub_574: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_1894);  convolution_68 = unsqueeze_1894 = None
    mul_2099: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_100, sub_574)
    sum_205: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_2099, [0, 2, 3]);  mul_2099 = None
    mul_2100: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_204, 0.0006377551020408163)
    unsqueeze_1895: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2100, 0);  mul_2100 = None
    unsqueeze_1896: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1895, 2);  unsqueeze_1895 = None
    unsqueeze_1897: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1896, 3);  unsqueeze_1896 = None
    mul_2101: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_205, 0.0006377551020408163)
    mul_2102: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_2103: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_2101, mul_2102);  mul_2101 = mul_2102 = None
    unsqueeze_1898: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2103, 0);  mul_2103 = None
    unsqueeze_1899: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1898, 2);  unsqueeze_1898 = None
    unsqueeze_1900: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1899, 3);  unsqueeze_1899 = None
    mul_2104: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_206);  primals_206 = None
    unsqueeze_1901: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2104, 0);  mul_2104 = None
    unsqueeze_1902: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1901, 2);  unsqueeze_1901 = None
    unsqueeze_1903: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1902, 3);  unsqueeze_1902 = None
    mul_2105: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_574, unsqueeze_1900);  sub_574 = unsqueeze_1900 = None
    sub_576: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_100, mul_2105);  mul_2105 = None
    sub_577: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_576, unsqueeze_1897);  sub_576 = unsqueeze_1897 = None
    mul_2106: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_577, unsqueeze_1903);  sub_577 = unsqueeze_1903 = None
    mul_2107: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_205, squeeze_205);  sum_205 = squeeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(mul_2106, cat_12, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2106 = cat_12 = primals_205 = None
    getitem_1305: "f32[8, 416, 14, 14]" = convolution_backward_101[0]
    getitem_1306: "f32[1024, 416, 1, 1]" = convolution_backward_101[1];  convolution_backward_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_81: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1305, 1, 0, 104)
    slice_82: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1305, 1, 104, 208)
    slice_83: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1305, 1, 208, 312)
    slice_84: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1305, 1, 312, 416);  getitem_1305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_101: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_101, full_default, slice_83);  le_101 = slice_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_206: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_101, [0, 2, 3])
    sub_578: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_1906);  convolution_67 = unsqueeze_1906 = None
    mul_2108: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_101, sub_578)
    sum_207: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2108, [0, 2, 3]);  mul_2108 = None
    mul_2109: "f32[104]" = torch.ops.aten.mul.Tensor(sum_206, 0.0006377551020408163)
    unsqueeze_1907: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2109, 0);  mul_2109 = None
    unsqueeze_1908: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1907, 2);  unsqueeze_1907 = None
    unsqueeze_1909: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1908, 3);  unsqueeze_1908 = None
    mul_2110: "f32[104]" = torch.ops.aten.mul.Tensor(sum_207, 0.0006377551020408163)
    mul_2111: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_2112: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2110, mul_2111);  mul_2110 = mul_2111 = None
    unsqueeze_1910: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2112, 0);  mul_2112 = None
    unsqueeze_1911: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1910, 2);  unsqueeze_1910 = None
    unsqueeze_1912: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1911, 3);  unsqueeze_1911 = None
    mul_2113: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_203);  primals_203 = None
    unsqueeze_1913: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2113, 0);  mul_2113 = None
    unsqueeze_1914: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1913, 2);  unsqueeze_1913 = None
    unsqueeze_1915: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1914, 3);  unsqueeze_1914 = None
    mul_2114: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_578, unsqueeze_1912);  sub_578 = unsqueeze_1912 = None
    sub_580: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_101, mul_2114);  where_101 = mul_2114 = None
    sub_581: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_580, unsqueeze_1909);  sub_580 = unsqueeze_1909 = None
    mul_2115: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_581, unsqueeze_1915);  sub_581 = unsqueeze_1915 = None
    mul_2116: "f32[104]" = torch.ops.aten.mul.Tensor(sum_207, squeeze_202);  sum_207 = squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(mul_2115, add_366, primals_202, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2115 = add_366 = primals_202 = None
    getitem_1308: "f32[8, 104, 14, 14]" = convolution_backward_102[0]
    getitem_1309: "f32[104, 104, 3, 3]" = convolution_backward_102[1];  convolution_backward_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_999: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_82, getitem_1308);  slice_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_102: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_102, full_default, add_999);  le_102 = add_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_208: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_102, [0, 2, 3])
    sub_582: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_1918);  convolution_66 = unsqueeze_1918 = None
    mul_2117: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_102, sub_582)
    sum_209: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2117, [0, 2, 3]);  mul_2117 = None
    mul_2118: "f32[104]" = torch.ops.aten.mul.Tensor(sum_208, 0.0006377551020408163)
    unsqueeze_1919: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2118, 0);  mul_2118 = None
    unsqueeze_1920: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1919, 2);  unsqueeze_1919 = None
    unsqueeze_1921: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1920, 3);  unsqueeze_1920 = None
    mul_2119: "f32[104]" = torch.ops.aten.mul.Tensor(sum_209, 0.0006377551020408163)
    mul_2120: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_2121: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2119, mul_2120);  mul_2119 = mul_2120 = None
    unsqueeze_1922: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2121, 0);  mul_2121 = None
    unsqueeze_1923: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1922, 2);  unsqueeze_1922 = None
    unsqueeze_1924: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1923, 3);  unsqueeze_1923 = None
    mul_2122: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_200);  primals_200 = None
    unsqueeze_1925: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2122, 0);  mul_2122 = None
    unsqueeze_1926: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1925, 2);  unsqueeze_1925 = None
    unsqueeze_1927: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1926, 3);  unsqueeze_1926 = None
    mul_2123: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_582, unsqueeze_1924);  sub_582 = unsqueeze_1924 = None
    sub_584: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_102, mul_2123);  where_102 = mul_2123 = None
    sub_585: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_584, unsqueeze_1921);  sub_584 = unsqueeze_1921 = None
    mul_2124: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_585, unsqueeze_1927);  sub_585 = unsqueeze_1927 = None
    mul_2125: "f32[104]" = torch.ops.aten.mul.Tensor(sum_209, squeeze_199);  sum_209 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(mul_2124, add_360, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2124 = add_360 = primals_199 = None
    getitem_1311: "f32[8, 104, 14, 14]" = convolution_backward_103[0]
    getitem_1312: "f32[104, 104, 3, 3]" = convolution_backward_103[1];  convolution_backward_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1000: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_81, getitem_1311);  slice_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_103: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_103, full_default, add_1000);  le_103 = add_1000 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_210: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_103, [0, 2, 3])
    sub_586: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_1930);  convolution_65 = unsqueeze_1930 = None
    mul_2126: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_103, sub_586)
    sum_211: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2126, [0, 2, 3]);  mul_2126 = None
    mul_2127: "f32[104]" = torch.ops.aten.mul.Tensor(sum_210, 0.0006377551020408163)
    unsqueeze_1931: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2127, 0);  mul_2127 = None
    unsqueeze_1932: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1931, 2);  unsqueeze_1931 = None
    unsqueeze_1933: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1932, 3);  unsqueeze_1932 = None
    mul_2128: "f32[104]" = torch.ops.aten.mul.Tensor(sum_211, 0.0006377551020408163)
    mul_2129: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_2130: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2128, mul_2129);  mul_2128 = mul_2129 = None
    unsqueeze_1934: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2130, 0);  mul_2130 = None
    unsqueeze_1935: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1934, 2);  unsqueeze_1934 = None
    unsqueeze_1936: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1935, 3);  unsqueeze_1935 = None
    mul_2131: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_197);  primals_197 = None
    unsqueeze_1937: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2131, 0);  mul_2131 = None
    unsqueeze_1938: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1937, 2);  unsqueeze_1937 = None
    unsqueeze_1939: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1938, 3);  unsqueeze_1938 = None
    mul_2132: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_586, unsqueeze_1936);  sub_586 = unsqueeze_1936 = None
    sub_588: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_103, mul_2132);  where_103 = mul_2132 = None
    sub_589: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_588, unsqueeze_1933);  sub_588 = unsqueeze_1933 = None
    mul_2133: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_589, unsqueeze_1939);  sub_589 = unsqueeze_1939 = None
    mul_2134: "f32[104]" = torch.ops.aten.mul.Tensor(sum_211, squeeze_196);  sum_211 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_104 = torch.ops.aten.convolution_backward.default(mul_2133, getitem_376, primals_196, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2133 = getitem_376 = primals_196 = None
    getitem_1314: "f32[8, 104, 14, 14]" = convolution_backward_104[0]
    getitem_1315: "f32[104, 104, 3, 3]" = convolution_backward_104[1];  convolution_backward_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_53: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1314, getitem_1311, getitem_1308, slice_84], 1);  getitem_1314 = getitem_1311 = getitem_1308 = slice_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_104: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_104, full_default, cat_53);  le_104 = cat_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_212: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_104, [0, 2, 3])
    sub_590: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_1942);  convolution_64 = unsqueeze_1942 = None
    mul_2135: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_104, sub_590)
    sum_213: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_2135, [0, 2, 3]);  mul_2135 = None
    mul_2136: "f32[416]" = torch.ops.aten.mul.Tensor(sum_212, 0.0006377551020408163)
    unsqueeze_1943: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2136, 0);  mul_2136 = None
    unsqueeze_1944: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1943, 2);  unsqueeze_1943 = None
    unsqueeze_1945: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1944, 3);  unsqueeze_1944 = None
    mul_2137: "f32[416]" = torch.ops.aten.mul.Tensor(sum_213, 0.0006377551020408163)
    mul_2138: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_2139: "f32[416]" = torch.ops.aten.mul.Tensor(mul_2137, mul_2138);  mul_2137 = mul_2138 = None
    unsqueeze_1946: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2139, 0);  mul_2139 = None
    unsqueeze_1947: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1946, 2);  unsqueeze_1946 = None
    unsqueeze_1948: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1947, 3);  unsqueeze_1947 = None
    mul_2140: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_194);  primals_194 = None
    unsqueeze_1949: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2140, 0);  mul_2140 = None
    unsqueeze_1950: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1949, 2);  unsqueeze_1949 = None
    unsqueeze_1951: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1950, 3);  unsqueeze_1950 = None
    mul_2141: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_590, unsqueeze_1948);  sub_590 = unsqueeze_1948 = None
    sub_592: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_104, mul_2141);  where_104 = mul_2141 = None
    sub_593: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_592, unsqueeze_1945);  sub_592 = unsqueeze_1945 = None
    mul_2142: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_593, unsqueeze_1951);  sub_593 = unsqueeze_1951 = None
    mul_2143: "f32[416]" = torch.ops.aten.mul.Tensor(sum_213, squeeze_193);  sum_213 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_105 = torch.ops.aten.convolution_backward.default(mul_2142, relu_60, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2142 = primals_193 = None
    getitem_1317: "f32[8, 1024, 14, 14]" = convolution_backward_105[0]
    getitem_1318: "f32[416, 1024, 1, 1]" = convolution_backward_105[1];  convolution_backward_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_1001: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_100, getitem_1317);  where_100 = getitem_1317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_105: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
    where_105: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_105, full_default, add_1001);  le_105 = add_1001 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_214: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_105, [0, 2, 3])
    sub_594: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_1954);  convolution_63 = unsqueeze_1954 = None
    mul_2144: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_105, sub_594)
    sum_215: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_2144, [0, 2, 3]);  mul_2144 = None
    mul_2145: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_214, 0.0006377551020408163)
    unsqueeze_1955: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2145, 0);  mul_2145 = None
    unsqueeze_1956: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1955, 2);  unsqueeze_1955 = None
    unsqueeze_1957: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1956, 3);  unsqueeze_1956 = None
    mul_2146: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_215, 0.0006377551020408163)
    mul_2147: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_2148: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_2146, mul_2147);  mul_2146 = mul_2147 = None
    unsqueeze_1958: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2148, 0);  mul_2148 = None
    unsqueeze_1959: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1958, 2);  unsqueeze_1958 = None
    unsqueeze_1960: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1959, 3);  unsqueeze_1959 = None
    mul_2149: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_191);  primals_191 = None
    unsqueeze_1961: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2149, 0);  mul_2149 = None
    unsqueeze_1962: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1961, 2);  unsqueeze_1961 = None
    unsqueeze_1963: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1962, 3);  unsqueeze_1962 = None
    mul_2150: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_594, unsqueeze_1960);  sub_594 = unsqueeze_1960 = None
    sub_596: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_105, mul_2150);  mul_2150 = None
    sub_597: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_596, unsqueeze_1957);  sub_596 = unsqueeze_1957 = None
    mul_2151: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_597, unsqueeze_1963);  sub_597 = unsqueeze_1963 = None
    mul_2152: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_215, squeeze_190);  sum_215 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_106 = torch.ops.aten.convolution_backward.default(mul_2151, cat_11, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2151 = cat_11 = primals_190 = None
    getitem_1320: "f32[8, 416, 14, 14]" = convolution_backward_106[0]
    getitem_1321: "f32[1024, 416, 1, 1]" = convolution_backward_106[1];  convolution_backward_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_85: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1320, 1, 0, 104)
    slice_86: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1320, 1, 104, 208)
    slice_87: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1320, 1, 208, 312)
    slice_88: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1320, 1, 312, 416);  getitem_1320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_106: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_106, full_default, slice_87);  le_106 = slice_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_216: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_106, [0, 2, 3])
    sub_598: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_1966);  convolution_62 = unsqueeze_1966 = None
    mul_2153: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_106, sub_598)
    sum_217: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2153, [0, 2, 3]);  mul_2153 = None
    mul_2154: "f32[104]" = torch.ops.aten.mul.Tensor(sum_216, 0.0006377551020408163)
    unsqueeze_1967: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2154, 0);  mul_2154 = None
    unsqueeze_1968: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1967, 2);  unsqueeze_1967 = None
    unsqueeze_1969: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1968, 3);  unsqueeze_1968 = None
    mul_2155: "f32[104]" = torch.ops.aten.mul.Tensor(sum_217, 0.0006377551020408163)
    mul_2156: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_2157: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2155, mul_2156);  mul_2155 = mul_2156 = None
    unsqueeze_1970: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2157, 0);  mul_2157 = None
    unsqueeze_1971: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1970, 2);  unsqueeze_1970 = None
    unsqueeze_1972: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1971, 3);  unsqueeze_1971 = None
    mul_2158: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_188);  primals_188 = None
    unsqueeze_1973: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2158, 0);  mul_2158 = None
    unsqueeze_1974: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1973, 2);  unsqueeze_1973 = None
    unsqueeze_1975: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1974, 3);  unsqueeze_1974 = None
    mul_2159: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_598, unsqueeze_1972);  sub_598 = unsqueeze_1972 = None
    sub_600: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_106, mul_2159);  where_106 = mul_2159 = None
    sub_601: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_600, unsqueeze_1969);  sub_600 = unsqueeze_1969 = None
    mul_2160: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_601, unsqueeze_1975);  sub_601 = unsqueeze_1975 = None
    mul_2161: "f32[104]" = torch.ops.aten.mul.Tensor(sum_217, squeeze_187);  sum_217 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_107 = torch.ops.aten.convolution_backward.default(mul_2160, add_338, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2160 = add_338 = primals_187 = None
    getitem_1323: "f32[8, 104, 14, 14]" = convolution_backward_107[0]
    getitem_1324: "f32[104, 104, 3, 3]" = convolution_backward_107[1];  convolution_backward_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1002: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_86, getitem_1323);  slice_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_107: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_107, full_default, add_1002);  le_107 = add_1002 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_218: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_107, [0, 2, 3])
    sub_602: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_1978);  convolution_61 = unsqueeze_1978 = None
    mul_2162: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_107, sub_602)
    sum_219: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2162, [0, 2, 3]);  mul_2162 = None
    mul_2163: "f32[104]" = torch.ops.aten.mul.Tensor(sum_218, 0.0006377551020408163)
    unsqueeze_1979: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2163, 0);  mul_2163 = None
    unsqueeze_1980: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1979, 2);  unsqueeze_1979 = None
    unsqueeze_1981: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1980, 3);  unsqueeze_1980 = None
    mul_2164: "f32[104]" = torch.ops.aten.mul.Tensor(sum_219, 0.0006377551020408163)
    mul_2165: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_2166: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2164, mul_2165);  mul_2164 = mul_2165 = None
    unsqueeze_1982: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2166, 0);  mul_2166 = None
    unsqueeze_1983: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1982, 2);  unsqueeze_1982 = None
    unsqueeze_1984: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1983, 3);  unsqueeze_1983 = None
    mul_2167: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_185);  primals_185 = None
    unsqueeze_1985: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2167, 0);  mul_2167 = None
    unsqueeze_1986: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1985, 2);  unsqueeze_1985 = None
    unsqueeze_1987: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1986, 3);  unsqueeze_1986 = None
    mul_2168: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_602, unsqueeze_1984);  sub_602 = unsqueeze_1984 = None
    sub_604: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_107, mul_2168);  where_107 = mul_2168 = None
    sub_605: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_604, unsqueeze_1981);  sub_604 = unsqueeze_1981 = None
    mul_2169: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_605, unsqueeze_1987);  sub_605 = unsqueeze_1987 = None
    mul_2170: "f32[104]" = torch.ops.aten.mul.Tensor(sum_219, squeeze_184);  sum_219 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_108 = torch.ops.aten.convolution_backward.default(mul_2169, add_332, primals_184, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2169 = add_332 = primals_184 = None
    getitem_1326: "f32[8, 104, 14, 14]" = convolution_backward_108[0]
    getitem_1327: "f32[104, 104, 3, 3]" = convolution_backward_108[1];  convolution_backward_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1003: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_85, getitem_1326);  slice_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_108: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_108, full_default, add_1003);  le_108 = add_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_220: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_108, [0, 2, 3])
    sub_606: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_1990);  convolution_60 = unsqueeze_1990 = None
    mul_2171: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_108, sub_606)
    sum_221: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2171, [0, 2, 3]);  mul_2171 = None
    mul_2172: "f32[104]" = torch.ops.aten.mul.Tensor(sum_220, 0.0006377551020408163)
    unsqueeze_1991: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2172, 0);  mul_2172 = None
    unsqueeze_1992: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1991, 2);  unsqueeze_1991 = None
    unsqueeze_1993: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1992, 3);  unsqueeze_1992 = None
    mul_2173: "f32[104]" = torch.ops.aten.mul.Tensor(sum_221, 0.0006377551020408163)
    mul_2174: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_2175: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2173, mul_2174);  mul_2173 = mul_2174 = None
    unsqueeze_1994: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2175, 0);  mul_2175 = None
    unsqueeze_1995: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1994, 2);  unsqueeze_1994 = None
    unsqueeze_1996: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1995, 3);  unsqueeze_1995 = None
    mul_2176: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_182);  primals_182 = None
    unsqueeze_1997: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2176, 0);  mul_2176 = None
    unsqueeze_1998: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1997, 2);  unsqueeze_1997 = None
    unsqueeze_1999: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1998, 3);  unsqueeze_1998 = None
    mul_2177: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_606, unsqueeze_1996);  sub_606 = unsqueeze_1996 = None
    sub_608: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_108, mul_2177);  where_108 = mul_2177 = None
    sub_609: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_608, unsqueeze_1993);  sub_608 = unsqueeze_1993 = None
    mul_2178: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_609, unsqueeze_1999);  sub_609 = unsqueeze_1999 = None
    mul_2179: "f32[104]" = torch.ops.aten.mul.Tensor(sum_221, squeeze_181);  sum_221 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_109 = torch.ops.aten.convolution_backward.default(mul_2178, getitem_346, primals_181, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2178 = getitem_346 = primals_181 = None
    getitem_1329: "f32[8, 104, 14, 14]" = convolution_backward_109[0]
    getitem_1330: "f32[104, 104, 3, 3]" = convolution_backward_109[1];  convolution_backward_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_54: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1329, getitem_1326, getitem_1323, slice_88], 1);  getitem_1329 = getitem_1326 = getitem_1323 = slice_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_109: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_109, full_default, cat_54);  le_109 = cat_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_222: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_109, [0, 2, 3])
    sub_610: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_2002);  convolution_59 = unsqueeze_2002 = None
    mul_2180: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_109, sub_610)
    sum_223: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_2180, [0, 2, 3]);  mul_2180 = None
    mul_2181: "f32[416]" = torch.ops.aten.mul.Tensor(sum_222, 0.0006377551020408163)
    unsqueeze_2003: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2181, 0);  mul_2181 = None
    unsqueeze_2004: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2003, 2);  unsqueeze_2003 = None
    unsqueeze_2005: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2004, 3);  unsqueeze_2004 = None
    mul_2182: "f32[416]" = torch.ops.aten.mul.Tensor(sum_223, 0.0006377551020408163)
    mul_2183: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_2184: "f32[416]" = torch.ops.aten.mul.Tensor(mul_2182, mul_2183);  mul_2182 = mul_2183 = None
    unsqueeze_2006: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2184, 0);  mul_2184 = None
    unsqueeze_2007: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2006, 2);  unsqueeze_2006 = None
    unsqueeze_2008: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2007, 3);  unsqueeze_2007 = None
    mul_2185: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_179);  primals_179 = None
    unsqueeze_2009: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2185, 0);  mul_2185 = None
    unsqueeze_2010: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2009, 2);  unsqueeze_2009 = None
    unsqueeze_2011: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2010, 3);  unsqueeze_2010 = None
    mul_2186: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_610, unsqueeze_2008);  sub_610 = unsqueeze_2008 = None
    sub_612: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_109, mul_2186);  where_109 = mul_2186 = None
    sub_613: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_612, unsqueeze_2005);  sub_612 = unsqueeze_2005 = None
    mul_2187: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_613, unsqueeze_2011);  sub_613 = unsqueeze_2011 = None
    mul_2188: "f32[416]" = torch.ops.aten.mul.Tensor(sum_223, squeeze_178);  sum_223 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_110 = torch.ops.aten.convolution_backward.default(mul_2187, relu_55, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2187 = primals_178 = None
    getitem_1332: "f32[8, 1024, 14, 14]" = convolution_backward_110[0]
    getitem_1333: "f32[416, 1024, 1, 1]" = convolution_backward_110[1];  convolution_backward_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_1004: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_105, getitem_1332);  where_105 = getitem_1332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_110: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
    where_110: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_110, full_default, add_1004);  le_110 = add_1004 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_224: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_110, [0, 2, 3])
    sub_614: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_2014);  convolution_58 = unsqueeze_2014 = None
    mul_2189: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_110, sub_614)
    sum_225: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_2189, [0, 2, 3]);  mul_2189 = None
    mul_2190: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_224, 0.0006377551020408163)
    unsqueeze_2015: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2190, 0);  mul_2190 = None
    unsqueeze_2016: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2015, 2);  unsqueeze_2015 = None
    unsqueeze_2017: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2016, 3);  unsqueeze_2016 = None
    mul_2191: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_225, 0.0006377551020408163)
    mul_2192: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_2193: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_2191, mul_2192);  mul_2191 = mul_2192 = None
    unsqueeze_2018: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2193, 0);  mul_2193 = None
    unsqueeze_2019: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2018, 2);  unsqueeze_2018 = None
    unsqueeze_2020: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2019, 3);  unsqueeze_2019 = None
    mul_2194: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_176);  primals_176 = None
    unsqueeze_2021: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2194, 0);  mul_2194 = None
    unsqueeze_2022: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2021, 2);  unsqueeze_2021 = None
    unsqueeze_2023: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2022, 3);  unsqueeze_2022 = None
    mul_2195: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_614, unsqueeze_2020);  sub_614 = unsqueeze_2020 = None
    sub_616: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_110, mul_2195);  mul_2195 = None
    sub_617: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_616, unsqueeze_2017);  sub_616 = unsqueeze_2017 = None
    mul_2196: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_617, unsqueeze_2023);  sub_617 = unsqueeze_2023 = None
    mul_2197: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_225, squeeze_175);  sum_225 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_111 = torch.ops.aten.convolution_backward.default(mul_2196, cat_10, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2196 = cat_10 = primals_175 = None
    getitem_1335: "f32[8, 416, 14, 14]" = convolution_backward_111[0]
    getitem_1336: "f32[1024, 416, 1, 1]" = convolution_backward_111[1];  convolution_backward_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_89: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1335, 1, 0, 104)
    slice_90: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1335, 1, 104, 208)
    slice_91: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1335, 1, 208, 312)
    slice_92: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1335, 1, 312, 416);  getitem_1335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_111: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_111, full_default, slice_91);  le_111 = slice_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_226: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_111, [0, 2, 3])
    sub_618: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_2026);  convolution_57 = unsqueeze_2026 = None
    mul_2198: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_111, sub_618)
    sum_227: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2198, [0, 2, 3]);  mul_2198 = None
    mul_2199: "f32[104]" = torch.ops.aten.mul.Tensor(sum_226, 0.0006377551020408163)
    unsqueeze_2027: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2199, 0);  mul_2199 = None
    unsqueeze_2028: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2027, 2);  unsqueeze_2027 = None
    unsqueeze_2029: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2028, 3);  unsqueeze_2028 = None
    mul_2200: "f32[104]" = torch.ops.aten.mul.Tensor(sum_227, 0.0006377551020408163)
    mul_2201: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_2202: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2200, mul_2201);  mul_2200 = mul_2201 = None
    unsqueeze_2030: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2202, 0);  mul_2202 = None
    unsqueeze_2031: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2030, 2);  unsqueeze_2030 = None
    unsqueeze_2032: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2031, 3);  unsqueeze_2031 = None
    mul_2203: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_173);  primals_173 = None
    unsqueeze_2033: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2203, 0);  mul_2203 = None
    unsqueeze_2034: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2033, 2);  unsqueeze_2033 = None
    unsqueeze_2035: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2034, 3);  unsqueeze_2034 = None
    mul_2204: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_618, unsqueeze_2032);  sub_618 = unsqueeze_2032 = None
    sub_620: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_111, mul_2204);  where_111 = mul_2204 = None
    sub_621: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_620, unsqueeze_2029);  sub_620 = unsqueeze_2029 = None
    mul_2205: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_621, unsqueeze_2035);  sub_621 = unsqueeze_2035 = None
    mul_2206: "f32[104]" = torch.ops.aten.mul.Tensor(sum_227, squeeze_172);  sum_227 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_112 = torch.ops.aten.convolution_backward.default(mul_2205, add_310, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2205 = add_310 = primals_172 = None
    getitem_1338: "f32[8, 104, 14, 14]" = convolution_backward_112[0]
    getitem_1339: "f32[104, 104, 3, 3]" = convolution_backward_112[1];  convolution_backward_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1005: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_90, getitem_1338);  slice_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_112: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_112, full_default, add_1005);  le_112 = add_1005 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_228: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_112, [0, 2, 3])
    sub_622: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_2038);  convolution_56 = unsqueeze_2038 = None
    mul_2207: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_112, sub_622)
    sum_229: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2207, [0, 2, 3]);  mul_2207 = None
    mul_2208: "f32[104]" = torch.ops.aten.mul.Tensor(sum_228, 0.0006377551020408163)
    unsqueeze_2039: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2208, 0);  mul_2208 = None
    unsqueeze_2040: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2039, 2);  unsqueeze_2039 = None
    unsqueeze_2041: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2040, 3);  unsqueeze_2040 = None
    mul_2209: "f32[104]" = torch.ops.aten.mul.Tensor(sum_229, 0.0006377551020408163)
    mul_2210: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_2211: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2209, mul_2210);  mul_2209 = mul_2210 = None
    unsqueeze_2042: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2211, 0);  mul_2211 = None
    unsqueeze_2043: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2042, 2);  unsqueeze_2042 = None
    unsqueeze_2044: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2043, 3);  unsqueeze_2043 = None
    mul_2212: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_170);  primals_170 = None
    unsqueeze_2045: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2212, 0);  mul_2212 = None
    unsqueeze_2046: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2045, 2);  unsqueeze_2045 = None
    unsqueeze_2047: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2046, 3);  unsqueeze_2046 = None
    mul_2213: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_622, unsqueeze_2044);  sub_622 = unsqueeze_2044 = None
    sub_624: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_112, mul_2213);  where_112 = mul_2213 = None
    sub_625: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_624, unsqueeze_2041);  sub_624 = unsqueeze_2041 = None
    mul_2214: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_625, unsqueeze_2047);  sub_625 = unsqueeze_2047 = None
    mul_2215: "f32[104]" = torch.ops.aten.mul.Tensor(sum_229, squeeze_169);  sum_229 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_113 = torch.ops.aten.convolution_backward.default(mul_2214, add_304, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2214 = add_304 = primals_169 = None
    getitem_1341: "f32[8, 104, 14, 14]" = convolution_backward_113[0]
    getitem_1342: "f32[104, 104, 3, 3]" = convolution_backward_113[1];  convolution_backward_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1006: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_89, getitem_1341);  slice_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_113: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_113, full_default, add_1006);  le_113 = add_1006 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_230: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_113, [0, 2, 3])
    sub_626: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_2050);  convolution_55 = unsqueeze_2050 = None
    mul_2216: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_113, sub_626)
    sum_231: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2216, [0, 2, 3]);  mul_2216 = None
    mul_2217: "f32[104]" = torch.ops.aten.mul.Tensor(sum_230, 0.0006377551020408163)
    unsqueeze_2051: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2217, 0);  mul_2217 = None
    unsqueeze_2052: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2051, 2);  unsqueeze_2051 = None
    unsqueeze_2053: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2052, 3);  unsqueeze_2052 = None
    mul_2218: "f32[104]" = torch.ops.aten.mul.Tensor(sum_231, 0.0006377551020408163)
    mul_2219: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_2220: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2218, mul_2219);  mul_2218 = mul_2219 = None
    unsqueeze_2054: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2220, 0);  mul_2220 = None
    unsqueeze_2055: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2054, 2);  unsqueeze_2054 = None
    unsqueeze_2056: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2055, 3);  unsqueeze_2055 = None
    mul_2221: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_167);  primals_167 = None
    unsqueeze_2057: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2221, 0);  mul_2221 = None
    unsqueeze_2058: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2057, 2);  unsqueeze_2057 = None
    unsqueeze_2059: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2058, 3);  unsqueeze_2058 = None
    mul_2222: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_626, unsqueeze_2056);  sub_626 = unsqueeze_2056 = None
    sub_628: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_113, mul_2222);  where_113 = mul_2222 = None
    sub_629: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_628, unsqueeze_2053);  sub_628 = unsqueeze_2053 = None
    mul_2223: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_629, unsqueeze_2059);  sub_629 = unsqueeze_2059 = None
    mul_2224: "f32[104]" = torch.ops.aten.mul.Tensor(sum_231, squeeze_166);  sum_231 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_114 = torch.ops.aten.convolution_backward.default(mul_2223, getitem_316, primals_166, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2223 = getitem_316 = primals_166 = None
    getitem_1344: "f32[8, 104, 14, 14]" = convolution_backward_114[0]
    getitem_1345: "f32[104, 104, 3, 3]" = convolution_backward_114[1];  convolution_backward_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_55: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1344, getitem_1341, getitem_1338, slice_92], 1);  getitem_1344 = getitem_1341 = getitem_1338 = slice_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_114: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_114, full_default, cat_55);  le_114 = cat_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_232: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_114, [0, 2, 3])
    sub_630: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_2062);  convolution_54 = unsqueeze_2062 = None
    mul_2225: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_114, sub_630)
    sum_233: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_2225, [0, 2, 3]);  mul_2225 = None
    mul_2226: "f32[416]" = torch.ops.aten.mul.Tensor(sum_232, 0.0006377551020408163)
    unsqueeze_2063: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2226, 0);  mul_2226 = None
    unsqueeze_2064: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2063, 2);  unsqueeze_2063 = None
    unsqueeze_2065: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2064, 3);  unsqueeze_2064 = None
    mul_2227: "f32[416]" = torch.ops.aten.mul.Tensor(sum_233, 0.0006377551020408163)
    mul_2228: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_2229: "f32[416]" = torch.ops.aten.mul.Tensor(mul_2227, mul_2228);  mul_2227 = mul_2228 = None
    unsqueeze_2066: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2229, 0);  mul_2229 = None
    unsqueeze_2067: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2066, 2);  unsqueeze_2066 = None
    unsqueeze_2068: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2067, 3);  unsqueeze_2067 = None
    mul_2230: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_164);  primals_164 = None
    unsqueeze_2069: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2230, 0);  mul_2230 = None
    unsqueeze_2070: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2069, 2);  unsqueeze_2069 = None
    unsqueeze_2071: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2070, 3);  unsqueeze_2070 = None
    mul_2231: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_630, unsqueeze_2068);  sub_630 = unsqueeze_2068 = None
    sub_632: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_114, mul_2231);  where_114 = mul_2231 = None
    sub_633: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_632, unsqueeze_2065);  sub_632 = unsqueeze_2065 = None
    mul_2232: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_633, unsqueeze_2071);  sub_633 = unsqueeze_2071 = None
    mul_2233: "f32[416]" = torch.ops.aten.mul.Tensor(sum_233, squeeze_163);  sum_233 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_115 = torch.ops.aten.convolution_backward.default(mul_2232, relu_50, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2232 = primals_163 = None
    getitem_1347: "f32[8, 1024, 14, 14]" = convolution_backward_115[0]
    getitem_1348: "f32[416, 1024, 1, 1]" = convolution_backward_115[1];  convolution_backward_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_1007: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_110, getitem_1347);  where_110 = getitem_1347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_115: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
    where_115: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_115, full_default, add_1007);  le_115 = add_1007 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_234: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_115, [0, 2, 3])
    sub_634: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_2074);  convolution_53 = unsqueeze_2074 = None
    mul_2234: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_115, sub_634)
    sum_235: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_2234, [0, 2, 3]);  mul_2234 = None
    mul_2235: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_234, 0.0006377551020408163)
    unsqueeze_2075: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2235, 0);  mul_2235 = None
    unsqueeze_2076: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2075, 2);  unsqueeze_2075 = None
    unsqueeze_2077: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2076, 3);  unsqueeze_2076 = None
    mul_2236: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_235, 0.0006377551020408163)
    mul_2237: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_2238: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_2236, mul_2237);  mul_2236 = mul_2237 = None
    unsqueeze_2078: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2238, 0);  mul_2238 = None
    unsqueeze_2079: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2078, 2);  unsqueeze_2078 = None
    unsqueeze_2080: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2079, 3);  unsqueeze_2079 = None
    mul_2239: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_161);  primals_161 = None
    unsqueeze_2081: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2239, 0);  mul_2239 = None
    unsqueeze_2082: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2081, 2);  unsqueeze_2081 = None
    unsqueeze_2083: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2082, 3);  unsqueeze_2082 = None
    mul_2240: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_634, unsqueeze_2080);  sub_634 = unsqueeze_2080 = None
    sub_636: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_115, mul_2240);  mul_2240 = None
    sub_637: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_636, unsqueeze_2077);  sub_636 = unsqueeze_2077 = None
    mul_2241: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_637, unsqueeze_2083);  sub_637 = unsqueeze_2083 = None
    mul_2242: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_235, squeeze_160);  sum_235 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_116 = torch.ops.aten.convolution_backward.default(mul_2241, cat_9, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2241 = cat_9 = primals_160 = None
    getitem_1350: "f32[8, 416, 14, 14]" = convolution_backward_116[0]
    getitem_1351: "f32[1024, 416, 1, 1]" = convolution_backward_116[1];  convolution_backward_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_93: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1350, 1, 0, 104)
    slice_94: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1350, 1, 104, 208)
    slice_95: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1350, 1, 208, 312)
    slice_96: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1350, 1, 312, 416);  getitem_1350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_116: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_116, full_default, slice_95);  le_116 = slice_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_236: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_116, [0, 2, 3])
    sub_638: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_2086);  convolution_52 = unsqueeze_2086 = None
    mul_2243: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_116, sub_638)
    sum_237: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2243, [0, 2, 3]);  mul_2243 = None
    mul_2244: "f32[104]" = torch.ops.aten.mul.Tensor(sum_236, 0.0006377551020408163)
    unsqueeze_2087: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2244, 0);  mul_2244 = None
    unsqueeze_2088: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2087, 2);  unsqueeze_2087 = None
    unsqueeze_2089: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2088, 3);  unsqueeze_2088 = None
    mul_2245: "f32[104]" = torch.ops.aten.mul.Tensor(sum_237, 0.0006377551020408163)
    mul_2246: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_2247: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2245, mul_2246);  mul_2245 = mul_2246 = None
    unsqueeze_2090: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2247, 0);  mul_2247 = None
    unsqueeze_2091: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2090, 2);  unsqueeze_2090 = None
    unsqueeze_2092: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2091, 3);  unsqueeze_2091 = None
    mul_2248: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_158);  primals_158 = None
    unsqueeze_2093: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2248, 0);  mul_2248 = None
    unsqueeze_2094: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2093, 2);  unsqueeze_2093 = None
    unsqueeze_2095: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2094, 3);  unsqueeze_2094 = None
    mul_2249: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_638, unsqueeze_2092);  sub_638 = unsqueeze_2092 = None
    sub_640: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_116, mul_2249);  where_116 = mul_2249 = None
    sub_641: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_640, unsqueeze_2089);  sub_640 = unsqueeze_2089 = None
    mul_2250: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_641, unsqueeze_2095);  sub_641 = unsqueeze_2095 = None
    mul_2251: "f32[104]" = torch.ops.aten.mul.Tensor(sum_237, squeeze_157);  sum_237 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_117 = torch.ops.aten.convolution_backward.default(mul_2250, add_282, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2250 = add_282 = primals_157 = None
    getitem_1353: "f32[8, 104, 14, 14]" = convolution_backward_117[0]
    getitem_1354: "f32[104, 104, 3, 3]" = convolution_backward_117[1];  convolution_backward_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1008: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_94, getitem_1353);  slice_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_117: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_117, full_default, add_1008);  le_117 = add_1008 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_238: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_117, [0, 2, 3])
    sub_642: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_2098);  convolution_51 = unsqueeze_2098 = None
    mul_2252: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_117, sub_642)
    sum_239: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2252, [0, 2, 3]);  mul_2252 = None
    mul_2253: "f32[104]" = torch.ops.aten.mul.Tensor(sum_238, 0.0006377551020408163)
    unsqueeze_2099: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2253, 0);  mul_2253 = None
    unsqueeze_2100: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2099, 2);  unsqueeze_2099 = None
    unsqueeze_2101: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2100, 3);  unsqueeze_2100 = None
    mul_2254: "f32[104]" = torch.ops.aten.mul.Tensor(sum_239, 0.0006377551020408163)
    mul_2255: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_2256: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2254, mul_2255);  mul_2254 = mul_2255 = None
    unsqueeze_2102: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2256, 0);  mul_2256 = None
    unsqueeze_2103: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2102, 2);  unsqueeze_2102 = None
    unsqueeze_2104: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2103, 3);  unsqueeze_2103 = None
    mul_2257: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_155);  primals_155 = None
    unsqueeze_2105: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2257, 0);  mul_2257 = None
    unsqueeze_2106: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2105, 2);  unsqueeze_2105 = None
    unsqueeze_2107: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2106, 3);  unsqueeze_2106 = None
    mul_2258: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_642, unsqueeze_2104);  sub_642 = unsqueeze_2104 = None
    sub_644: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_117, mul_2258);  where_117 = mul_2258 = None
    sub_645: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_644, unsqueeze_2101);  sub_644 = unsqueeze_2101 = None
    mul_2259: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_645, unsqueeze_2107);  sub_645 = unsqueeze_2107 = None
    mul_2260: "f32[104]" = torch.ops.aten.mul.Tensor(sum_239, squeeze_154);  sum_239 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_118 = torch.ops.aten.convolution_backward.default(mul_2259, add_276, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2259 = add_276 = primals_154 = None
    getitem_1356: "f32[8, 104, 14, 14]" = convolution_backward_118[0]
    getitem_1357: "f32[104, 104, 3, 3]" = convolution_backward_118[1];  convolution_backward_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1009: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_93, getitem_1356);  slice_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_118: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_118, full_default, add_1009);  le_118 = add_1009 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_240: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_118, [0, 2, 3])
    sub_646: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_2110);  convolution_50 = unsqueeze_2110 = None
    mul_2261: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_118, sub_646)
    sum_241: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2261, [0, 2, 3]);  mul_2261 = None
    mul_2262: "f32[104]" = torch.ops.aten.mul.Tensor(sum_240, 0.0006377551020408163)
    unsqueeze_2111: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2262, 0);  mul_2262 = None
    unsqueeze_2112: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2111, 2);  unsqueeze_2111 = None
    unsqueeze_2113: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2112, 3);  unsqueeze_2112 = None
    mul_2263: "f32[104]" = torch.ops.aten.mul.Tensor(sum_241, 0.0006377551020408163)
    mul_2264: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_2265: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2263, mul_2264);  mul_2263 = mul_2264 = None
    unsqueeze_2114: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2265, 0);  mul_2265 = None
    unsqueeze_2115: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2114, 2);  unsqueeze_2114 = None
    unsqueeze_2116: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2115, 3);  unsqueeze_2115 = None
    mul_2266: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_152);  primals_152 = None
    unsqueeze_2117: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2266, 0);  mul_2266 = None
    unsqueeze_2118: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2117, 2);  unsqueeze_2117 = None
    unsqueeze_2119: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2118, 3);  unsqueeze_2118 = None
    mul_2267: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_646, unsqueeze_2116);  sub_646 = unsqueeze_2116 = None
    sub_648: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_118, mul_2267);  where_118 = mul_2267 = None
    sub_649: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_648, unsqueeze_2113);  sub_648 = unsqueeze_2113 = None
    mul_2268: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_649, unsqueeze_2119);  sub_649 = unsqueeze_2119 = None
    mul_2269: "f32[104]" = torch.ops.aten.mul.Tensor(sum_241, squeeze_151);  sum_241 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_119 = torch.ops.aten.convolution_backward.default(mul_2268, getitem_286, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2268 = getitem_286 = primals_151 = None
    getitem_1359: "f32[8, 104, 14, 14]" = convolution_backward_119[0]
    getitem_1360: "f32[104, 104, 3, 3]" = convolution_backward_119[1];  convolution_backward_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_56: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1359, getitem_1356, getitem_1353, slice_96], 1);  getitem_1359 = getitem_1356 = getitem_1353 = slice_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_119: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_119, full_default, cat_56);  le_119 = cat_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_242: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_119, [0, 2, 3])
    sub_650: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_2122);  convolution_49 = unsqueeze_2122 = None
    mul_2270: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_119, sub_650)
    sum_243: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_2270, [0, 2, 3]);  mul_2270 = None
    mul_2271: "f32[416]" = torch.ops.aten.mul.Tensor(sum_242, 0.0006377551020408163)
    unsqueeze_2123: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2271, 0);  mul_2271 = None
    unsqueeze_2124: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2123, 2);  unsqueeze_2123 = None
    unsqueeze_2125: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2124, 3);  unsqueeze_2124 = None
    mul_2272: "f32[416]" = torch.ops.aten.mul.Tensor(sum_243, 0.0006377551020408163)
    mul_2273: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_2274: "f32[416]" = torch.ops.aten.mul.Tensor(mul_2272, mul_2273);  mul_2272 = mul_2273 = None
    unsqueeze_2126: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2274, 0);  mul_2274 = None
    unsqueeze_2127: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2126, 2);  unsqueeze_2126 = None
    unsqueeze_2128: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2127, 3);  unsqueeze_2127 = None
    mul_2275: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_149);  primals_149 = None
    unsqueeze_2129: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2275, 0);  mul_2275 = None
    unsqueeze_2130: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2129, 2);  unsqueeze_2129 = None
    unsqueeze_2131: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2130, 3);  unsqueeze_2130 = None
    mul_2276: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_650, unsqueeze_2128);  sub_650 = unsqueeze_2128 = None
    sub_652: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_119, mul_2276);  where_119 = mul_2276 = None
    sub_653: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_652, unsqueeze_2125);  sub_652 = unsqueeze_2125 = None
    mul_2277: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_653, unsqueeze_2131);  sub_653 = unsqueeze_2131 = None
    mul_2278: "f32[416]" = torch.ops.aten.mul.Tensor(sum_243, squeeze_148);  sum_243 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_120 = torch.ops.aten.convolution_backward.default(mul_2277, relu_45, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2277 = primals_148 = None
    getitem_1362: "f32[8, 1024, 14, 14]" = convolution_backward_120[0]
    getitem_1363: "f32[416, 1024, 1, 1]" = convolution_backward_120[1];  convolution_backward_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_1010: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_115, getitem_1362);  where_115 = getitem_1362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_120: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
    where_120: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_120, full_default, add_1010);  le_120 = add_1010 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_244: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_120, [0, 2, 3])
    sub_654: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_2134);  convolution_48 = unsqueeze_2134 = None
    mul_2279: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_120, sub_654)
    sum_245: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_2279, [0, 2, 3]);  mul_2279 = None
    mul_2280: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_244, 0.0006377551020408163)
    unsqueeze_2135: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2280, 0);  mul_2280 = None
    unsqueeze_2136: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2135, 2);  unsqueeze_2135 = None
    unsqueeze_2137: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2136, 3);  unsqueeze_2136 = None
    mul_2281: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_245, 0.0006377551020408163)
    mul_2282: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_2283: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_2281, mul_2282);  mul_2281 = mul_2282 = None
    unsqueeze_2138: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2283, 0);  mul_2283 = None
    unsqueeze_2139: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2138, 2);  unsqueeze_2138 = None
    unsqueeze_2140: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2139, 3);  unsqueeze_2139 = None
    mul_2284: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_146);  primals_146 = None
    unsqueeze_2141: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2284, 0);  mul_2284 = None
    unsqueeze_2142: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2141, 2);  unsqueeze_2141 = None
    unsqueeze_2143: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2142, 3);  unsqueeze_2142 = None
    mul_2285: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_654, unsqueeze_2140);  sub_654 = unsqueeze_2140 = None
    sub_656: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_120, mul_2285);  mul_2285 = None
    sub_657: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_656, unsqueeze_2137);  sub_656 = unsqueeze_2137 = None
    mul_2286: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_657, unsqueeze_2143);  sub_657 = unsqueeze_2143 = None
    mul_2287: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_245, squeeze_145);  sum_245 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_121 = torch.ops.aten.convolution_backward.default(mul_2286, cat_8, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2286 = cat_8 = primals_145 = None
    getitem_1365: "f32[8, 416, 14, 14]" = convolution_backward_121[0]
    getitem_1366: "f32[1024, 416, 1, 1]" = convolution_backward_121[1];  convolution_backward_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_97: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1365, 1, 0, 104)
    slice_98: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1365, 1, 104, 208)
    slice_99: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1365, 1, 208, 312)
    slice_100: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1365, 1, 312, 416);  getitem_1365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_121: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_121, full_default, slice_99);  le_121 = slice_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_246: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_121, [0, 2, 3])
    sub_658: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_2146);  convolution_47 = unsqueeze_2146 = None
    mul_2288: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_121, sub_658)
    sum_247: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2288, [0, 2, 3]);  mul_2288 = None
    mul_2289: "f32[104]" = torch.ops.aten.mul.Tensor(sum_246, 0.0006377551020408163)
    unsqueeze_2147: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2289, 0);  mul_2289 = None
    unsqueeze_2148: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2147, 2);  unsqueeze_2147 = None
    unsqueeze_2149: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2148, 3);  unsqueeze_2148 = None
    mul_2290: "f32[104]" = torch.ops.aten.mul.Tensor(sum_247, 0.0006377551020408163)
    mul_2291: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_2292: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2290, mul_2291);  mul_2290 = mul_2291 = None
    unsqueeze_2150: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2292, 0);  mul_2292 = None
    unsqueeze_2151: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2150, 2);  unsqueeze_2150 = None
    unsqueeze_2152: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2151, 3);  unsqueeze_2151 = None
    mul_2293: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_143);  primals_143 = None
    unsqueeze_2153: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2293, 0);  mul_2293 = None
    unsqueeze_2154: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2153, 2);  unsqueeze_2153 = None
    unsqueeze_2155: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2154, 3);  unsqueeze_2154 = None
    mul_2294: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_658, unsqueeze_2152);  sub_658 = unsqueeze_2152 = None
    sub_660: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_121, mul_2294);  where_121 = mul_2294 = None
    sub_661: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_660, unsqueeze_2149);  sub_660 = unsqueeze_2149 = None
    mul_2295: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_661, unsqueeze_2155);  sub_661 = unsqueeze_2155 = None
    mul_2296: "f32[104]" = torch.ops.aten.mul.Tensor(sum_247, squeeze_142);  sum_247 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_122 = torch.ops.aten.convolution_backward.default(mul_2295, add_254, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2295 = add_254 = primals_142 = None
    getitem_1368: "f32[8, 104, 14, 14]" = convolution_backward_122[0]
    getitem_1369: "f32[104, 104, 3, 3]" = convolution_backward_122[1];  convolution_backward_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1011: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_98, getitem_1368);  slice_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_122: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_122, full_default, add_1011);  le_122 = add_1011 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_248: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_122, [0, 2, 3])
    sub_662: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_2158);  convolution_46 = unsqueeze_2158 = None
    mul_2297: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_122, sub_662)
    sum_249: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2297, [0, 2, 3]);  mul_2297 = None
    mul_2298: "f32[104]" = torch.ops.aten.mul.Tensor(sum_248, 0.0006377551020408163)
    unsqueeze_2159: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2298, 0);  mul_2298 = None
    unsqueeze_2160: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2159, 2);  unsqueeze_2159 = None
    unsqueeze_2161: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2160, 3);  unsqueeze_2160 = None
    mul_2299: "f32[104]" = torch.ops.aten.mul.Tensor(sum_249, 0.0006377551020408163)
    mul_2300: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_2301: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2299, mul_2300);  mul_2299 = mul_2300 = None
    unsqueeze_2162: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2301, 0);  mul_2301 = None
    unsqueeze_2163: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2162, 2);  unsqueeze_2162 = None
    unsqueeze_2164: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2163, 3);  unsqueeze_2163 = None
    mul_2302: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_140);  primals_140 = None
    unsqueeze_2165: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2302, 0);  mul_2302 = None
    unsqueeze_2166: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2165, 2);  unsqueeze_2165 = None
    unsqueeze_2167: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2166, 3);  unsqueeze_2166 = None
    mul_2303: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_662, unsqueeze_2164);  sub_662 = unsqueeze_2164 = None
    sub_664: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_122, mul_2303);  where_122 = mul_2303 = None
    sub_665: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_664, unsqueeze_2161);  sub_664 = unsqueeze_2161 = None
    mul_2304: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_665, unsqueeze_2167);  sub_665 = unsqueeze_2167 = None
    mul_2305: "f32[104]" = torch.ops.aten.mul.Tensor(sum_249, squeeze_139);  sum_249 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_123 = torch.ops.aten.convolution_backward.default(mul_2304, add_248, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2304 = add_248 = primals_139 = None
    getitem_1371: "f32[8, 104, 14, 14]" = convolution_backward_123[0]
    getitem_1372: "f32[104, 104, 3, 3]" = convolution_backward_123[1];  convolution_backward_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1012: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(slice_97, getitem_1371);  slice_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_123: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_123, full_default, add_1012);  le_123 = add_1012 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_250: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_123, [0, 2, 3])
    sub_666: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_2170);  convolution_45 = unsqueeze_2170 = None
    mul_2306: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_123, sub_666)
    sum_251: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2306, [0, 2, 3]);  mul_2306 = None
    mul_2307: "f32[104]" = torch.ops.aten.mul.Tensor(sum_250, 0.0006377551020408163)
    unsqueeze_2171: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2307, 0);  mul_2307 = None
    unsqueeze_2172: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2171, 2);  unsqueeze_2171 = None
    unsqueeze_2173: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2172, 3);  unsqueeze_2172 = None
    mul_2308: "f32[104]" = torch.ops.aten.mul.Tensor(sum_251, 0.0006377551020408163)
    mul_2309: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_2310: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2308, mul_2309);  mul_2308 = mul_2309 = None
    unsqueeze_2174: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2310, 0);  mul_2310 = None
    unsqueeze_2175: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2174, 2);  unsqueeze_2174 = None
    unsqueeze_2176: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2175, 3);  unsqueeze_2175 = None
    mul_2311: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_137);  primals_137 = None
    unsqueeze_2177: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2311, 0);  mul_2311 = None
    unsqueeze_2178: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2177, 2);  unsqueeze_2177 = None
    unsqueeze_2179: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2178, 3);  unsqueeze_2178 = None
    mul_2312: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_666, unsqueeze_2176);  sub_666 = unsqueeze_2176 = None
    sub_668: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_123, mul_2312);  where_123 = mul_2312 = None
    sub_669: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_668, unsqueeze_2173);  sub_668 = unsqueeze_2173 = None
    mul_2313: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_669, unsqueeze_2179);  sub_669 = unsqueeze_2179 = None
    mul_2314: "f32[104]" = torch.ops.aten.mul.Tensor(sum_251, squeeze_136);  sum_251 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_124 = torch.ops.aten.convolution_backward.default(mul_2313, getitem_256, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2313 = getitem_256 = primals_136 = None
    getitem_1374: "f32[8, 104, 14, 14]" = convolution_backward_124[0]
    getitem_1375: "f32[104, 104, 3, 3]" = convolution_backward_124[1];  convolution_backward_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_57: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([getitem_1374, getitem_1371, getitem_1368, slice_100], 1);  getitem_1374 = getitem_1371 = getitem_1368 = slice_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_124: "f32[8, 416, 14, 14]" = torch.ops.aten.where.self(le_124, full_default, cat_57);  le_124 = cat_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_252: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_124, [0, 2, 3])
    sub_670: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_2182);  convolution_44 = unsqueeze_2182 = None
    mul_2315: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_124, sub_670)
    sum_253: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_2315, [0, 2, 3]);  mul_2315 = None
    mul_2316: "f32[416]" = torch.ops.aten.mul.Tensor(sum_252, 0.0006377551020408163)
    unsqueeze_2183: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2316, 0);  mul_2316 = None
    unsqueeze_2184: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2183, 2);  unsqueeze_2183 = None
    unsqueeze_2185: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2184, 3);  unsqueeze_2184 = None
    mul_2317: "f32[416]" = torch.ops.aten.mul.Tensor(sum_253, 0.0006377551020408163)
    mul_2318: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_2319: "f32[416]" = torch.ops.aten.mul.Tensor(mul_2317, mul_2318);  mul_2317 = mul_2318 = None
    unsqueeze_2186: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2319, 0);  mul_2319 = None
    unsqueeze_2187: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2186, 2);  unsqueeze_2186 = None
    unsqueeze_2188: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2187, 3);  unsqueeze_2187 = None
    mul_2320: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_134);  primals_134 = None
    unsqueeze_2189: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2320, 0);  mul_2320 = None
    unsqueeze_2190: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2189, 2);  unsqueeze_2189 = None
    unsqueeze_2191: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2190, 3);  unsqueeze_2190 = None
    mul_2321: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_670, unsqueeze_2188);  sub_670 = unsqueeze_2188 = None
    sub_672: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(where_124, mul_2321);  where_124 = mul_2321 = None
    sub_673: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(sub_672, unsqueeze_2185);  sub_672 = unsqueeze_2185 = None
    mul_2322: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_673, unsqueeze_2191);  sub_673 = unsqueeze_2191 = None
    mul_2323: "f32[416]" = torch.ops.aten.mul.Tensor(sum_253, squeeze_133);  sum_253 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_125 = torch.ops.aten.convolution_backward.default(mul_2322, relu_40, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2322 = primals_133 = None
    getitem_1377: "f32[8, 1024, 14, 14]" = convolution_backward_125[0]
    getitem_1378: "f32[416, 1024, 1, 1]" = convolution_backward_125[1];  convolution_backward_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_1013: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_120, getitem_1377);  where_120 = getitem_1377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_125: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
    where_125: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_125, full_default, add_1013);  le_125 = add_1013 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    sum_254: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_125, [0, 2, 3])
    sub_674: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_2194);  convolution_43 = unsqueeze_2194 = None
    mul_2324: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_125, sub_674)
    sum_255: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_2324, [0, 2, 3]);  mul_2324 = None
    mul_2325: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_254, 0.0006377551020408163)
    unsqueeze_2195: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2325, 0);  mul_2325 = None
    unsqueeze_2196: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2195, 2);  unsqueeze_2195 = None
    unsqueeze_2197: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2196, 3);  unsqueeze_2196 = None
    mul_2326: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_255, 0.0006377551020408163)
    mul_2327: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_2328: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_2326, mul_2327);  mul_2326 = mul_2327 = None
    unsqueeze_2198: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2328, 0);  mul_2328 = None
    unsqueeze_2199: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2198, 2);  unsqueeze_2198 = None
    unsqueeze_2200: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2199, 3);  unsqueeze_2199 = None
    mul_2329: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_131);  primals_131 = None
    unsqueeze_2201: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2329, 0);  mul_2329 = None
    unsqueeze_2202: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2201, 2);  unsqueeze_2201 = None
    unsqueeze_2203: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2202, 3);  unsqueeze_2202 = None
    mul_2330: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_674, unsqueeze_2200);  sub_674 = unsqueeze_2200 = None
    sub_676: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_125, mul_2330);  mul_2330 = None
    sub_677: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_676, unsqueeze_2197);  sub_676 = None
    mul_2331: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_677, unsqueeze_2203);  sub_677 = unsqueeze_2203 = None
    mul_2332: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_255, squeeze_130);  sum_255 = squeeze_130 = None
    convolution_backward_126 = torch.ops.aten.convolution_backward.default(mul_2331, relu_35, primals_130, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2331 = primals_130 = None
    getitem_1380: "f32[8, 512, 28, 28]" = convolution_backward_126[0]
    getitem_1381: "f32[1024, 512, 1, 1]" = convolution_backward_126[1];  convolution_backward_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sub_678: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_2206);  convolution_42 = unsqueeze_2206 = None
    mul_2333: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_125, sub_678)
    sum_257: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_2333, [0, 2, 3]);  mul_2333 = None
    mul_2335: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_257, 0.0006377551020408163)
    mul_2336: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_2337: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_2335, mul_2336);  mul_2335 = mul_2336 = None
    unsqueeze_2210: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2337, 0);  mul_2337 = None
    unsqueeze_2211: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2210, 2);  unsqueeze_2210 = None
    unsqueeze_2212: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2211, 3);  unsqueeze_2211 = None
    mul_2338: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_128);  primals_128 = None
    unsqueeze_2213: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_2338, 0);  mul_2338 = None
    unsqueeze_2214: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2213, 2);  unsqueeze_2213 = None
    unsqueeze_2215: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2214, 3);  unsqueeze_2214 = None
    mul_2339: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_678, unsqueeze_2212);  sub_678 = unsqueeze_2212 = None
    sub_680: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_125, mul_2339);  where_125 = mul_2339 = None
    sub_681: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_680, unsqueeze_2197);  sub_680 = unsqueeze_2197 = None
    mul_2340: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_681, unsqueeze_2215);  sub_681 = unsqueeze_2215 = None
    mul_2341: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_257, squeeze_127);  sum_257 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_127 = torch.ops.aten.convolution_backward.default(mul_2340, cat_7, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2340 = cat_7 = primals_127 = None
    getitem_1383: "f32[8, 416, 14, 14]" = convolution_backward_127[0]
    getitem_1384: "f32[1024, 416, 1, 1]" = convolution_backward_127[1];  convolution_backward_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_101: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1383, 1, 0, 104)
    slice_102: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1383, 1, 104, 208)
    slice_103: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1383, 1, 208, 312)
    slice_104: "f32[8, 104, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1383, 1, 312, 416);  getitem_1383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_backward_1: "f32[8, 104, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(slice_104, getitem_245, [3, 3], [2, 2], [1, 1], False, True, None);  slice_104 = getitem_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_126: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_126, full_default, slice_103);  le_126 = slice_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_258: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_126, [0, 2, 3])
    sub_682: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_2218);  convolution_41 = unsqueeze_2218 = None
    mul_2342: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_126, sub_682)
    sum_259: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2342, [0, 2, 3]);  mul_2342 = None
    mul_2343: "f32[104]" = torch.ops.aten.mul.Tensor(sum_258, 0.0006377551020408163)
    unsqueeze_2219: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2343, 0);  mul_2343 = None
    unsqueeze_2220: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2219, 2);  unsqueeze_2219 = None
    unsqueeze_2221: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2220, 3);  unsqueeze_2220 = None
    mul_2344: "f32[104]" = torch.ops.aten.mul.Tensor(sum_259, 0.0006377551020408163)
    mul_2345: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_2346: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2344, mul_2345);  mul_2344 = mul_2345 = None
    unsqueeze_2222: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2346, 0);  mul_2346 = None
    unsqueeze_2223: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2222, 2);  unsqueeze_2222 = None
    unsqueeze_2224: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2223, 3);  unsqueeze_2223 = None
    mul_2347: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_125);  primals_125 = None
    unsqueeze_2225: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2347, 0);  mul_2347 = None
    unsqueeze_2226: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2225, 2);  unsqueeze_2225 = None
    unsqueeze_2227: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2226, 3);  unsqueeze_2226 = None
    mul_2348: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_682, unsqueeze_2224);  sub_682 = unsqueeze_2224 = None
    sub_684: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_126, mul_2348);  where_126 = mul_2348 = None
    sub_685: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_684, unsqueeze_2221);  sub_684 = unsqueeze_2221 = None
    mul_2349: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_685, unsqueeze_2227);  sub_685 = unsqueeze_2227 = None
    mul_2350: "f32[104]" = torch.ops.aten.mul.Tensor(sum_259, squeeze_124);  sum_259 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_128 = torch.ops.aten.convolution_backward.default(mul_2349, getitem_238, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2349 = getitem_238 = primals_124 = None
    getitem_1386: "f32[8, 104, 28, 28]" = convolution_backward_128[0]
    getitem_1387: "f32[104, 104, 3, 3]" = convolution_backward_128[1];  convolution_backward_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_127: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_127, full_default, slice_102);  le_127 = slice_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_260: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_127, [0, 2, 3])
    sub_686: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_2230);  convolution_40 = unsqueeze_2230 = None
    mul_2351: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_127, sub_686)
    sum_261: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2351, [0, 2, 3]);  mul_2351 = None
    mul_2352: "f32[104]" = torch.ops.aten.mul.Tensor(sum_260, 0.0006377551020408163)
    unsqueeze_2231: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2352, 0);  mul_2352 = None
    unsqueeze_2232: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2231, 2);  unsqueeze_2231 = None
    unsqueeze_2233: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2232, 3);  unsqueeze_2232 = None
    mul_2353: "f32[104]" = torch.ops.aten.mul.Tensor(sum_261, 0.0006377551020408163)
    mul_2354: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_2355: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2353, mul_2354);  mul_2353 = mul_2354 = None
    unsqueeze_2234: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2355, 0);  mul_2355 = None
    unsqueeze_2235: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2234, 2);  unsqueeze_2234 = None
    unsqueeze_2236: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2235, 3);  unsqueeze_2235 = None
    mul_2356: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_122);  primals_122 = None
    unsqueeze_2237: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2356, 0);  mul_2356 = None
    unsqueeze_2238: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2237, 2);  unsqueeze_2237 = None
    unsqueeze_2239: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2238, 3);  unsqueeze_2238 = None
    mul_2357: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_686, unsqueeze_2236);  sub_686 = unsqueeze_2236 = None
    sub_688: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_127, mul_2357);  where_127 = mul_2357 = None
    sub_689: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_688, unsqueeze_2233);  sub_688 = unsqueeze_2233 = None
    mul_2358: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_689, unsqueeze_2239);  sub_689 = unsqueeze_2239 = None
    mul_2359: "f32[104]" = torch.ops.aten.mul.Tensor(sum_261, squeeze_121);  sum_261 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_129 = torch.ops.aten.convolution_backward.default(mul_2358, getitem_231, primals_121, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2358 = getitem_231 = primals_121 = None
    getitem_1389: "f32[8, 104, 28, 28]" = convolution_backward_129[0]
    getitem_1390: "f32[104, 104, 3, 3]" = convolution_backward_129[1];  convolution_backward_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_128: "f32[8, 104, 14, 14]" = torch.ops.aten.where.self(le_128, full_default, slice_101);  le_128 = slice_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_262: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_128, [0, 2, 3])
    sub_690: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_2242);  convolution_39 = unsqueeze_2242 = None
    mul_2360: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(where_128, sub_690)
    sum_263: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2360, [0, 2, 3]);  mul_2360 = None
    mul_2361: "f32[104]" = torch.ops.aten.mul.Tensor(sum_262, 0.0006377551020408163)
    unsqueeze_2243: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2361, 0);  mul_2361 = None
    unsqueeze_2244: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2243, 2);  unsqueeze_2243 = None
    unsqueeze_2245: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2244, 3);  unsqueeze_2244 = None
    mul_2362: "f32[104]" = torch.ops.aten.mul.Tensor(sum_263, 0.0006377551020408163)
    mul_2363: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_2364: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2362, mul_2363);  mul_2362 = mul_2363 = None
    unsqueeze_2246: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2364, 0);  mul_2364 = None
    unsqueeze_2247: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2246, 2);  unsqueeze_2246 = None
    unsqueeze_2248: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2247, 3);  unsqueeze_2247 = None
    mul_2365: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_119);  primals_119 = None
    unsqueeze_2249: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2365, 0);  mul_2365 = None
    unsqueeze_2250: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2249, 2);  unsqueeze_2249 = None
    unsqueeze_2251: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2250, 3);  unsqueeze_2250 = None
    mul_2366: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_690, unsqueeze_2248);  sub_690 = unsqueeze_2248 = None
    sub_692: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(where_128, mul_2366);  where_128 = mul_2366 = None
    sub_693: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_692, unsqueeze_2245);  sub_692 = unsqueeze_2245 = None
    mul_2367: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_693, unsqueeze_2251);  sub_693 = unsqueeze_2251 = None
    mul_2368: "f32[104]" = torch.ops.aten.mul.Tensor(sum_263, squeeze_118);  sum_263 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_130 = torch.ops.aten.convolution_backward.default(mul_2367, getitem_224, primals_118, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2367 = getitem_224 = primals_118 = None
    getitem_1392: "f32[8, 104, 28, 28]" = convolution_backward_130[0]
    getitem_1393: "f32[104, 104, 3, 3]" = convolution_backward_130[1];  convolution_backward_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_58: "f32[8, 416, 28, 28]" = torch.ops.aten.cat.default([getitem_1392, getitem_1389, getitem_1386, avg_pool2d_backward_1], 1);  getitem_1392 = getitem_1389 = getitem_1386 = avg_pool2d_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_129: "f32[8, 416, 28, 28]" = torch.ops.aten.where.self(le_129, full_default, cat_58);  le_129 = cat_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_264: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_129, [0, 2, 3])
    sub_694: "f32[8, 416, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_2254);  convolution_38 = unsqueeze_2254 = None
    mul_2369: "f32[8, 416, 28, 28]" = torch.ops.aten.mul.Tensor(where_129, sub_694)
    sum_265: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_2369, [0, 2, 3]);  mul_2369 = None
    mul_2370: "f32[416]" = torch.ops.aten.mul.Tensor(sum_264, 0.00015943877551020407)
    unsqueeze_2255: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2370, 0);  mul_2370 = None
    unsqueeze_2256: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2255, 2);  unsqueeze_2255 = None
    unsqueeze_2257: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2256, 3);  unsqueeze_2256 = None
    mul_2371: "f32[416]" = torch.ops.aten.mul.Tensor(sum_265, 0.00015943877551020407)
    mul_2372: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_2373: "f32[416]" = torch.ops.aten.mul.Tensor(mul_2371, mul_2372);  mul_2371 = mul_2372 = None
    unsqueeze_2258: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2373, 0);  mul_2373 = None
    unsqueeze_2259: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2258, 2);  unsqueeze_2258 = None
    unsqueeze_2260: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2259, 3);  unsqueeze_2259 = None
    mul_2374: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_116);  primals_116 = None
    unsqueeze_2261: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_2374, 0);  mul_2374 = None
    unsqueeze_2262: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2261, 2);  unsqueeze_2261 = None
    unsqueeze_2263: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2262, 3);  unsqueeze_2262 = None
    mul_2375: "f32[8, 416, 28, 28]" = torch.ops.aten.mul.Tensor(sub_694, unsqueeze_2260);  sub_694 = unsqueeze_2260 = None
    sub_696: "f32[8, 416, 28, 28]" = torch.ops.aten.sub.Tensor(where_129, mul_2375);  where_129 = mul_2375 = None
    sub_697: "f32[8, 416, 28, 28]" = torch.ops.aten.sub.Tensor(sub_696, unsqueeze_2257);  sub_696 = unsqueeze_2257 = None
    mul_2376: "f32[8, 416, 28, 28]" = torch.ops.aten.mul.Tensor(sub_697, unsqueeze_2263);  sub_697 = unsqueeze_2263 = None
    mul_2377: "f32[416]" = torch.ops.aten.mul.Tensor(sum_265, squeeze_115);  sum_265 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_131 = torch.ops.aten.convolution_backward.default(mul_2376, relu_35, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2376 = primals_115 = None
    getitem_1395: "f32[8, 512, 28, 28]" = convolution_backward_131[0]
    getitem_1396: "f32[416, 512, 1, 1]" = convolution_backward_131[1];  convolution_backward_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_1014: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(getitem_1380, getitem_1395);  getitem_1380 = getitem_1395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_130: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
    where_130: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_130, full_default, add_1014);  le_130 = add_1014 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_266: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_130, [0, 2, 3])
    sub_698: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_2266);  convolution_37 = unsqueeze_2266 = None
    mul_2378: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_130, sub_698)
    sum_267: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_2378, [0, 2, 3]);  mul_2378 = None
    mul_2379: "f32[512]" = torch.ops.aten.mul.Tensor(sum_266, 0.00015943877551020407)
    unsqueeze_2267: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2379, 0);  mul_2379 = None
    unsqueeze_2268: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2267, 2);  unsqueeze_2267 = None
    unsqueeze_2269: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2268, 3);  unsqueeze_2268 = None
    mul_2380: "f32[512]" = torch.ops.aten.mul.Tensor(sum_267, 0.00015943877551020407)
    mul_2381: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_2382: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2380, mul_2381);  mul_2380 = mul_2381 = None
    unsqueeze_2270: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2382, 0);  mul_2382 = None
    unsqueeze_2271: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2270, 2);  unsqueeze_2270 = None
    unsqueeze_2272: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2271, 3);  unsqueeze_2271 = None
    mul_2383: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_113);  primals_113 = None
    unsqueeze_2273: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2383, 0);  mul_2383 = None
    unsqueeze_2274: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2273, 2);  unsqueeze_2273 = None
    unsqueeze_2275: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2274, 3);  unsqueeze_2274 = None
    mul_2384: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_698, unsqueeze_2272);  sub_698 = unsqueeze_2272 = None
    sub_700: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_130, mul_2384);  mul_2384 = None
    sub_701: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_700, unsqueeze_2269);  sub_700 = unsqueeze_2269 = None
    mul_2385: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_701, unsqueeze_2275);  sub_701 = unsqueeze_2275 = None
    mul_2386: "f32[512]" = torch.ops.aten.mul.Tensor(sum_267, squeeze_112);  sum_267 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_132 = torch.ops.aten.convolution_backward.default(mul_2385, cat_6, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2385 = cat_6 = primals_112 = None
    getitem_1398: "f32[8, 208, 28, 28]" = convolution_backward_132[0]
    getitem_1399: "f32[512, 208, 1, 1]" = convolution_backward_132[1];  convolution_backward_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_105: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1398, 1, 0, 52)
    slice_106: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1398, 1, 52, 104)
    slice_107: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1398, 1, 104, 156)
    slice_108: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1398, 1, 156, 208);  getitem_1398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_131: "f32[8, 52, 28, 28]" = torch.ops.aten.where.self(le_131, full_default, slice_107);  le_131 = slice_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_268: "f32[52]" = torch.ops.aten.sum.dim_IntList(where_131, [0, 2, 3])
    sub_702: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_2278);  convolution_36 = unsqueeze_2278 = None
    mul_2387: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(where_131, sub_702)
    sum_269: "f32[52]" = torch.ops.aten.sum.dim_IntList(mul_2387, [0, 2, 3]);  mul_2387 = None
    mul_2388: "f32[52]" = torch.ops.aten.mul.Tensor(sum_268, 0.00015943877551020407)
    unsqueeze_2279: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2388, 0);  mul_2388 = None
    unsqueeze_2280: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2279, 2);  unsqueeze_2279 = None
    unsqueeze_2281: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2280, 3);  unsqueeze_2280 = None
    mul_2389: "f32[52]" = torch.ops.aten.mul.Tensor(sum_269, 0.00015943877551020407)
    mul_2390: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_2391: "f32[52]" = torch.ops.aten.mul.Tensor(mul_2389, mul_2390);  mul_2389 = mul_2390 = None
    unsqueeze_2282: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2391, 0);  mul_2391 = None
    unsqueeze_2283: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2282, 2);  unsqueeze_2282 = None
    unsqueeze_2284: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2283, 3);  unsqueeze_2283 = None
    mul_2392: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_110);  primals_110 = None
    unsqueeze_2285: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2392, 0);  mul_2392 = None
    unsqueeze_2286: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2285, 2);  unsqueeze_2285 = None
    unsqueeze_2287: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2286, 3);  unsqueeze_2286 = None
    mul_2393: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_702, unsqueeze_2284);  sub_702 = unsqueeze_2284 = None
    sub_704: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(where_131, mul_2393);  where_131 = mul_2393 = None
    sub_705: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(sub_704, unsqueeze_2281);  sub_704 = unsqueeze_2281 = None
    mul_2394: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_705, unsqueeze_2287);  sub_705 = unsqueeze_2287 = None
    mul_2395: "f32[52]" = torch.ops.aten.mul.Tensor(sum_269, squeeze_109);  sum_269 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_133 = torch.ops.aten.convolution_backward.default(mul_2394, add_195, primals_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2394 = add_195 = primals_109 = None
    getitem_1401: "f32[8, 52, 28, 28]" = convolution_backward_133[0]
    getitem_1402: "f32[52, 52, 3, 3]" = convolution_backward_133[1];  convolution_backward_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1015: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(slice_106, getitem_1401);  slice_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_132: "f32[8, 52, 28, 28]" = torch.ops.aten.where.self(le_132, full_default, add_1015);  le_132 = add_1015 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_270: "f32[52]" = torch.ops.aten.sum.dim_IntList(where_132, [0, 2, 3])
    sub_706: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_2290);  convolution_35 = unsqueeze_2290 = None
    mul_2396: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(where_132, sub_706)
    sum_271: "f32[52]" = torch.ops.aten.sum.dim_IntList(mul_2396, [0, 2, 3]);  mul_2396 = None
    mul_2397: "f32[52]" = torch.ops.aten.mul.Tensor(sum_270, 0.00015943877551020407)
    unsqueeze_2291: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2397, 0);  mul_2397 = None
    unsqueeze_2292: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2291, 2);  unsqueeze_2291 = None
    unsqueeze_2293: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2292, 3);  unsqueeze_2292 = None
    mul_2398: "f32[52]" = torch.ops.aten.mul.Tensor(sum_271, 0.00015943877551020407)
    mul_2399: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_2400: "f32[52]" = torch.ops.aten.mul.Tensor(mul_2398, mul_2399);  mul_2398 = mul_2399 = None
    unsqueeze_2294: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2400, 0);  mul_2400 = None
    unsqueeze_2295: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2294, 2);  unsqueeze_2294 = None
    unsqueeze_2296: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2295, 3);  unsqueeze_2295 = None
    mul_2401: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_107);  primals_107 = None
    unsqueeze_2297: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2401, 0);  mul_2401 = None
    unsqueeze_2298: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2297, 2);  unsqueeze_2297 = None
    unsqueeze_2299: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2298, 3);  unsqueeze_2298 = None
    mul_2402: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_706, unsqueeze_2296);  sub_706 = unsqueeze_2296 = None
    sub_708: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(where_132, mul_2402);  where_132 = mul_2402 = None
    sub_709: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(sub_708, unsqueeze_2293);  sub_708 = unsqueeze_2293 = None
    mul_2403: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_709, unsqueeze_2299);  sub_709 = unsqueeze_2299 = None
    mul_2404: "f32[52]" = torch.ops.aten.mul.Tensor(sum_271, squeeze_106);  sum_271 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_134 = torch.ops.aten.convolution_backward.default(mul_2403, add_189, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2403 = add_189 = primals_106 = None
    getitem_1404: "f32[8, 52, 28, 28]" = convolution_backward_134[0]
    getitem_1405: "f32[52, 52, 3, 3]" = convolution_backward_134[1];  convolution_backward_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1016: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(slice_105, getitem_1404);  slice_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_133: "f32[8, 52, 28, 28]" = torch.ops.aten.where.self(le_133, full_default, add_1016);  le_133 = add_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_272: "f32[52]" = torch.ops.aten.sum.dim_IntList(where_133, [0, 2, 3])
    sub_710: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_2302);  convolution_34 = unsqueeze_2302 = None
    mul_2405: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(where_133, sub_710)
    sum_273: "f32[52]" = torch.ops.aten.sum.dim_IntList(mul_2405, [0, 2, 3]);  mul_2405 = None
    mul_2406: "f32[52]" = torch.ops.aten.mul.Tensor(sum_272, 0.00015943877551020407)
    unsqueeze_2303: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2406, 0);  mul_2406 = None
    unsqueeze_2304: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2303, 2);  unsqueeze_2303 = None
    unsqueeze_2305: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2304, 3);  unsqueeze_2304 = None
    mul_2407: "f32[52]" = torch.ops.aten.mul.Tensor(sum_273, 0.00015943877551020407)
    mul_2408: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_2409: "f32[52]" = torch.ops.aten.mul.Tensor(mul_2407, mul_2408);  mul_2407 = mul_2408 = None
    unsqueeze_2306: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2409, 0);  mul_2409 = None
    unsqueeze_2307: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2306, 2);  unsqueeze_2306 = None
    unsqueeze_2308: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2307, 3);  unsqueeze_2307 = None
    mul_2410: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_104);  primals_104 = None
    unsqueeze_2309: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2410, 0);  mul_2410 = None
    unsqueeze_2310: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2309, 2);  unsqueeze_2309 = None
    unsqueeze_2311: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2310, 3);  unsqueeze_2310 = None
    mul_2411: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_710, unsqueeze_2308);  sub_710 = unsqueeze_2308 = None
    sub_712: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(where_133, mul_2411);  where_133 = mul_2411 = None
    sub_713: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(sub_712, unsqueeze_2305);  sub_712 = unsqueeze_2305 = None
    mul_2412: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_713, unsqueeze_2311);  sub_713 = unsqueeze_2311 = None
    mul_2413: "f32[52]" = torch.ops.aten.mul.Tensor(sum_273, squeeze_103);  sum_273 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_135 = torch.ops.aten.convolution_backward.default(mul_2412, getitem_194, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2412 = getitem_194 = primals_103 = None
    getitem_1407: "f32[8, 52, 28, 28]" = convolution_backward_135[0]
    getitem_1408: "f32[52, 52, 3, 3]" = convolution_backward_135[1];  convolution_backward_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_59: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([getitem_1407, getitem_1404, getitem_1401, slice_108], 1);  getitem_1407 = getitem_1404 = getitem_1401 = slice_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_134: "f32[8, 208, 28, 28]" = torch.ops.aten.where.self(le_134, full_default, cat_59);  le_134 = cat_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_274: "f32[208]" = torch.ops.aten.sum.dim_IntList(where_134, [0, 2, 3])
    sub_714: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_2314);  convolution_33 = unsqueeze_2314 = None
    mul_2414: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(where_134, sub_714)
    sum_275: "f32[208]" = torch.ops.aten.sum.dim_IntList(mul_2414, [0, 2, 3]);  mul_2414 = None
    mul_2415: "f32[208]" = torch.ops.aten.mul.Tensor(sum_274, 0.00015943877551020407)
    unsqueeze_2315: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_2415, 0);  mul_2415 = None
    unsqueeze_2316: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2315, 2);  unsqueeze_2315 = None
    unsqueeze_2317: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2316, 3);  unsqueeze_2316 = None
    mul_2416: "f32[208]" = torch.ops.aten.mul.Tensor(sum_275, 0.00015943877551020407)
    mul_2417: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_2418: "f32[208]" = torch.ops.aten.mul.Tensor(mul_2416, mul_2417);  mul_2416 = mul_2417 = None
    unsqueeze_2318: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_2418, 0);  mul_2418 = None
    unsqueeze_2319: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2318, 2);  unsqueeze_2318 = None
    unsqueeze_2320: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2319, 3);  unsqueeze_2319 = None
    mul_2419: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_101);  primals_101 = None
    unsqueeze_2321: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_2419, 0);  mul_2419 = None
    unsqueeze_2322: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2321, 2);  unsqueeze_2321 = None
    unsqueeze_2323: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2322, 3);  unsqueeze_2322 = None
    mul_2420: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_714, unsqueeze_2320);  sub_714 = unsqueeze_2320 = None
    sub_716: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(where_134, mul_2420);  where_134 = mul_2420 = None
    sub_717: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(sub_716, unsqueeze_2317);  sub_716 = unsqueeze_2317 = None
    mul_2421: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_717, unsqueeze_2323);  sub_717 = unsqueeze_2323 = None
    mul_2422: "f32[208]" = torch.ops.aten.mul.Tensor(sum_275, squeeze_100);  sum_275 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_136 = torch.ops.aten.convolution_backward.default(mul_2421, relu_30, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2421 = primals_100 = None
    getitem_1410: "f32[8, 512, 28, 28]" = convolution_backward_136[0]
    getitem_1411: "f32[208, 512, 1, 1]" = convolution_backward_136[1];  convolution_backward_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_1017: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_130, getitem_1410);  where_130 = getitem_1410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_135: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
    where_135: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_135, full_default, add_1017);  le_135 = add_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_276: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_135, [0, 2, 3])
    sub_718: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_2326);  convolution_32 = unsqueeze_2326 = None
    mul_2423: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_135, sub_718)
    sum_277: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_2423, [0, 2, 3]);  mul_2423 = None
    mul_2424: "f32[512]" = torch.ops.aten.mul.Tensor(sum_276, 0.00015943877551020407)
    unsqueeze_2327: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2424, 0);  mul_2424 = None
    unsqueeze_2328: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2327, 2);  unsqueeze_2327 = None
    unsqueeze_2329: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2328, 3);  unsqueeze_2328 = None
    mul_2425: "f32[512]" = torch.ops.aten.mul.Tensor(sum_277, 0.00015943877551020407)
    mul_2426: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_2427: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2425, mul_2426);  mul_2425 = mul_2426 = None
    unsqueeze_2330: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2427, 0);  mul_2427 = None
    unsqueeze_2331: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2330, 2);  unsqueeze_2330 = None
    unsqueeze_2332: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2331, 3);  unsqueeze_2331 = None
    mul_2428: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_98);  primals_98 = None
    unsqueeze_2333: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2428, 0);  mul_2428 = None
    unsqueeze_2334: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2333, 2);  unsqueeze_2333 = None
    unsqueeze_2335: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2334, 3);  unsqueeze_2334 = None
    mul_2429: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_718, unsqueeze_2332);  sub_718 = unsqueeze_2332 = None
    sub_720: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_135, mul_2429);  mul_2429 = None
    sub_721: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_720, unsqueeze_2329);  sub_720 = unsqueeze_2329 = None
    mul_2430: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_721, unsqueeze_2335);  sub_721 = unsqueeze_2335 = None
    mul_2431: "f32[512]" = torch.ops.aten.mul.Tensor(sum_277, squeeze_97);  sum_277 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_137 = torch.ops.aten.convolution_backward.default(mul_2430, cat_5, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2430 = cat_5 = primals_97 = None
    getitem_1413: "f32[8, 208, 28, 28]" = convolution_backward_137[0]
    getitem_1414: "f32[512, 208, 1, 1]" = convolution_backward_137[1];  convolution_backward_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_109: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1413, 1, 0, 52)
    slice_110: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1413, 1, 52, 104)
    slice_111: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1413, 1, 104, 156)
    slice_112: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1413, 1, 156, 208);  getitem_1413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_136: "f32[8, 52, 28, 28]" = torch.ops.aten.where.self(le_136, full_default, slice_111);  le_136 = slice_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_278: "f32[52]" = torch.ops.aten.sum.dim_IntList(where_136, [0, 2, 3])
    sub_722: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_2338);  convolution_31 = unsqueeze_2338 = None
    mul_2432: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(where_136, sub_722)
    sum_279: "f32[52]" = torch.ops.aten.sum.dim_IntList(mul_2432, [0, 2, 3]);  mul_2432 = None
    mul_2433: "f32[52]" = torch.ops.aten.mul.Tensor(sum_278, 0.00015943877551020407)
    unsqueeze_2339: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2433, 0);  mul_2433 = None
    unsqueeze_2340: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2339, 2);  unsqueeze_2339 = None
    unsqueeze_2341: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2340, 3);  unsqueeze_2340 = None
    mul_2434: "f32[52]" = torch.ops.aten.mul.Tensor(sum_279, 0.00015943877551020407)
    mul_2435: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_2436: "f32[52]" = torch.ops.aten.mul.Tensor(mul_2434, mul_2435);  mul_2434 = mul_2435 = None
    unsqueeze_2342: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2436, 0);  mul_2436 = None
    unsqueeze_2343: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2342, 2);  unsqueeze_2342 = None
    unsqueeze_2344: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2343, 3);  unsqueeze_2343 = None
    mul_2437: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_95);  primals_95 = None
    unsqueeze_2345: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2437, 0);  mul_2437 = None
    unsqueeze_2346: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2345, 2);  unsqueeze_2345 = None
    unsqueeze_2347: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2346, 3);  unsqueeze_2346 = None
    mul_2438: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_722, unsqueeze_2344);  sub_722 = unsqueeze_2344 = None
    sub_724: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(where_136, mul_2438);  where_136 = mul_2438 = None
    sub_725: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(sub_724, unsqueeze_2341);  sub_724 = unsqueeze_2341 = None
    mul_2439: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_725, unsqueeze_2347);  sub_725 = unsqueeze_2347 = None
    mul_2440: "f32[52]" = torch.ops.aten.mul.Tensor(sum_279, squeeze_94);  sum_279 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_138 = torch.ops.aten.convolution_backward.default(mul_2439, add_167, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2439 = add_167 = primals_94 = None
    getitem_1416: "f32[8, 52, 28, 28]" = convolution_backward_138[0]
    getitem_1417: "f32[52, 52, 3, 3]" = convolution_backward_138[1];  convolution_backward_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1018: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(slice_110, getitem_1416);  slice_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_137: "f32[8, 52, 28, 28]" = torch.ops.aten.where.self(le_137, full_default, add_1018);  le_137 = add_1018 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_280: "f32[52]" = torch.ops.aten.sum.dim_IntList(where_137, [0, 2, 3])
    sub_726: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_2350);  convolution_30 = unsqueeze_2350 = None
    mul_2441: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(where_137, sub_726)
    sum_281: "f32[52]" = torch.ops.aten.sum.dim_IntList(mul_2441, [0, 2, 3]);  mul_2441 = None
    mul_2442: "f32[52]" = torch.ops.aten.mul.Tensor(sum_280, 0.00015943877551020407)
    unsqueeze_2351: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2442, 0);  mul_2442 = None
    unsqueeze_2352: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2351, 2);  unsqueeze_2351 = None
    unsqueeze_2353: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2352, 3);  unsqueeze_2352 = None
    mul_2443: "f32[52]" = torch.ops.aten.mul.Tensor(sum_281, 0.00015943877551020407)
    mul_2444: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_2445: "f32[52]" = torch.ops.aten.mul.Tensor(mul_2443, mul_2444);  mul_2443 = mul_2444 = None
    unsqueeze_2354: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2445, 0);  mul_2445 = None
    unsqueeze_2355: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2354, 2);  unsqueeze_2354 = None
    unsqueeze_2356: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2355, 3);  unsqueeze_2355 = None
    mul_2446: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_92);  primals_92 = None
    unsqueeze_2357: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2446, 0);  mul_2446 = None
    unsqueeze_2358: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2357, 2);  unsqueeze_2357 = None
    unsqueeze_2359: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2358, 3);  unsqueeze_2358 = None
    mul_2447: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_726, unsqueeze_2356);  sub_726 = unsqueeze_2356 = None
    sub_728: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(where_137, mul_2447);  where_137 = mul_2447 = None
    sub_729: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(sub_728, unsqueeze_2353);  sub_728 = unsqueeze_2353 = None
    mul_2448: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_729, unsqueeze_2359);  sub_729 = unsqueeze_2359 = None
    mul_2449: "f32[52]" = torch.ops.aten.mul.Tensor(sum_281, squeeze_91);  sum_281 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_139 = torch.ops.aten.convolution_backward.default(mul_2448, add_161, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2448 = add_161 = primals_91 = None
    getitem_1419: "f32[8, 52, 28, 28]" = convolution_backward_139[0]
    getitem_1420: "f32[52, 52, 3, 3]" = convolution_backward_139[1];  convolution_backward_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1019: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(slice_109, getitem_1419);  slice_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_138: "f32[8, 52, 28, 28]" = torch.ops.aten.where.self(le_138, full_default, add_1019);  le_138 = add_1019 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_282: "f32[52]" = torch.ops.aten.sum.dim_IntList(where_138, [0, 2, 3])
    sub_730: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_2362);  convolution_29 = unsqueeze_2362 = None
    mul_2450: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(where_138, sub_730)
    sum_283: "f32[52]" = torch.ops.aten.sum.dim_IntList(mul_2450, [0, 2, 3]);  mul_2450 = None
    mul_2451: "f32[52]" = torch.ops.aten.mul.Tensor(sum_282, 0.00015943877551020407)
    unsqueeze_2363: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2451, 0);  mul_2451 = None
    unsqueeze_2364: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2363, 2);  unsqueeze_2363 = None
    unsqueeze_2365: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2364, 3);  unsqueeze_2364 = None
    mul_2452: "f32[52]" = torch.ops.aten.mul.Tensor(sum_283, 0.00015943877551020407)
    mul_2453: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_2454: "f32[52]" = torch.ops.aten.mul.Tensor(mul_2452, mul_2453);  mul_2452 = mul_2453 = None
    unsqueeze_2366: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2454, 0);  mul_2454 = None
    unsqueeze_2367: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2366, 2);  unsqueeze_2366 = None
    unsqueeze_2368: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2367, 3);  unsqueeze_2367 = None
    mul_2455: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_89);  primals_89 = None
    unsqueeze_2369: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2455, 0);  mul_2455 = None
    unsqueeze_2370: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2369, 2);  unsqueeze_2369 = None
    unsqueeze_2371: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2370, 3);  unsqueeze_2370 = None
    mul_2456: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_730, unsqueeze_2368);  sub_730 = unsqueeze_2368 = None
    sub_732: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(where_138, mul_2456);  where_138 = mul_2456 = None
    sub_733: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(sub_732, unsqueeze_2365);  sub_732 = unsqueeze_2365 = None
    mul_2457: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_733, unsqueeze_2371);  sub_733 = unsqueeze_2371 = None
    mul_2458: "f32[52]" = torch.ops.aten.mul.Tensor(sum_283, squeeze_88);  sum_283 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_140 = torch.ops.aten.convolution_backward.default(mul_2457, getitem_164, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2457 = getitem_164 = primals_88 = None
    getitem_1422: "f32[8, 52, 28, 28]" = convolution_backward_140[0]
    getitem_1423: "f32[52, 52, 3, 3]" = convolution_backward_140[1];  convolution_backward_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_60: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([getitem_1422, getitem_1419, getitem_1416, slice_112], 1);  getitem_1422 = getitem_1419 = getitem_1416 = slice_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_139: "f32[8, 208, 28, 28]" = torch.ops.aten.where.self(le_139, full_default, cat_60);  le_139 = cat_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_284: "f32[208]" = torch.ops.aten.sum.dim_IntList(where_139, [0, 2, 3])
    sub_734: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_2374);  convolution_28 = unsqueeze_2374 = None
    mul_2459: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(where_139, sub_734)
    sum_285: "f32[208]" = torch.ops.aten.sum.dim_IntList(mul_2459, [0, 2, 3]);  mul_2459 = None
    mul_2460: "f32[208]" = torch.ops.aten.mul.Tensor(sum_284, 0.00015943877551020407)
    unsqueeze_2375: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_2460, 0);  mul_2460 = None
    unsqueeze_2376: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2375, 2);  unsqueeze_2375 = None
    unsqueeze_2377: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2376, 3);  unsqueeze_2376 = None
    mul_2461: "f32[208]" = torch.ops.aten.mul.Tensor(sum_285, 0.00015943877551020407)
    mul_2462: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_2463: "f32[208]" = torch.ops.aten.mul.Tensor(mul_2461, mul_2462);  mul_2461 = mul_2462 = None
    unsqueeze_2378: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_2463, 0);  mul_2463 = None
    unsqueeze_2379: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2378, 2);  unsqueeze_2378 = None
    unsqueeze_2380: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2379, 3);  unsqueeze_2379 = None
    mul_2464: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_86);  primals_86 = None
    unsqueeze_2381: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_2464, 0);  mul_2464 = None
    unsqueeze_2382: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2381, 2);  unsqueeze_2381 = None
    unsqueeze_2383: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2382, 3);  unsqueeze_2382 = None
    mul_2465: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_734, unsqueeze_2380);  sub_734 = unsqueeze_2380 = None
    sub_736: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(where_139, mul_2465);  where_139 = mul_2465 = None
    sub_737: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(sub_736, unsqueeze_2377);  sub_736 = unsqueeze_2377 = None
    mul_2466: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_737, unsqueeze_2383);  sub_737 = unsqueeze_2383 = None
    mul_2467: "f32[208]" = torch.ops.aten.mul.Tensor(sum_285, squeeze_85);  sum_285 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_141 = torch.ops.aten.convolution_backward.default(mul_2466, relu_25, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2466 = primals_85 = None
    getitem_1425: "f32[8, 512, 28, 28]" = convolution_backward_141[0]
    getitem_1426: "f32[208, 512, 1, 1]" = convolution_backward_141[1];  convolution_backward_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_1020: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_135, getitem_1425);  where_135 = getitem_1425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_140: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    where_140: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_140, full_default, add_1020);  le_140 = add_1020 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_286: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_140, [0, 2, 3])
    sub_738: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_2386);  convolution_27 = unsqueeze_2386 = None
    mul_2468: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_140, sub_738)
    sum_287: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_2468, [0, 2, 3]);  mul_2468 = None
    mul_2469: "f32[512]" = torch.ops.aten.mul.Tensor(sum_286, 0.00015943877551020407)
    unsqueeze_2387: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2469, 0);  mul_2469 = None
    unsqueeze_2388: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2387, 2);  unsqueeze_2387 = None
    unsqueeze_2389: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2388, 3);  unsqueeze_2388 = None
    mul_2470: "f32[512]" = torch.ops.aten.mul.Tensor(sum_287, 0.00015943877551020407)
    mul_2471: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_2472: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2470, mul_2471);  mul_2470 = mul_2471 = None
    unsqueeze_2390: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2472, 0);  mul_2472 = None
    unsqueeze_2391: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2390, 2);  unsqueeze_2390 = None
    unsqueeze_2392: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2391, 3);  unsqueeze_2391 = None
    mul_2473: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_83);  primals_83 = None
    unsqueeze_2393: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2473, 0);  mul_2473 = None
    unsqueeze_2394: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2393, 2);  unsqueeze_2393 = None
    unsqueeze_2395: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2394, 3);  unsqueeze_2394 = None
    mul_2474: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_738, unsqueeze_2392);  sub_738 = unsqueeze_2392 = None
    sub_740: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_140, mul_2474);  mul_2474 = None
    sub_741: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_740, unsqueeze_2389);  sub_740 = unsqueeze_2389 = None
    mul_2475: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_741, unsqueeze_2395);  sub_741 = unsqueeze_2395 = None
    mul_2476: "f32[512]" = torch.ops.aten.mul.Tensor(sum_287, squeeze_82);  sum_287 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_142 = torch.ops.aten.convolution_backward.default(mul_2475, cat_4, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2475 = cat_4 = primals_82 = None
    getitem_1428: "f32[8, 208, 28, 28]" = convolution_backward_142[0]
    getitem_1429: "f32[512, 208, 1, 1]" = convolution_backward_142[1];  convolution_backward_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_113: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1428, 1, 0, 52)
    slice_114: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1428, 1, 52, 104)
    slice_115: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1428, 1, 104, 156)
    slice_116: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1428, 1, 156, 208);  getitem_1428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_141: "f32[8, 52, 28, 28]" = torch.ops.aten.where.self(le_141, full_default, slice_115);  le_141 = slice_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_288: "f32[52]" = torch.ops.aten.sum.dim_IntList(where_141, [0, 2, 3])
    sub_742: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_2398);  convolution_26 = unsqueeze_2398 = None
    mul_2477: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(where_141, sub_742)
    sum_289: "f32[52]" = torch.ops.aten.sum.dim_IntList(mul_2477, [0, 2, 3]);  mul_2477 = None
    mul_2478: "f32[52]" = torch.ops.aten.mul.Tensor(sum_288, 0.00015943877551020407)
    unsqueeze_2399: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2478, 0);  mul_2478 = None
    unsqueeze_2400: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2399, 2);  unsqueeze_2399 = None
    unsqueeze_2401: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2400, 3);  unsqueeze_2400 = None
    mul_2479: "f32[52]" = torch.ops.aten.mul.Tensor(sum_289, 0.00015943877551020407)
    mul_2480: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_2481: "f32[52]" = torch.ops.aten.mul.Tensor(mul_2479, mul_2480);  mul_2479 = mul_2480 = None
    unsqueeze_2402: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2481, 0);  mul_2481 = None
    unsqueeze_2403: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2402, 2);  unsqueeze_2402 = None
    unsqueeze_2404: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2403, 3);  unsqueeze_2403 = None
    mul_2482: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_80);  primals_80 = None
    unsqueeze_2405: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2482, 0);  mul_2482 = None
    unsqueeze_2406: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2405, 2);  unsqueeze_2405 = None
    unsqueeze_2407: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2406, 3);  unsqueeze_2406 = None
    mul_2483: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_742, unsqueeze_2404);  sub_742 = unsqueeze_2404 = None
    sub_744: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(where_141, mul_2483);  where_141 = mul_2483 = None
    sub_745: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(sub_744, unsqueeze_2401);  sub_744 = unsqueeze_2401 = None
    mul_2484: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_745, unsqueeze_2407);  sub_745 = unsqueeze_2407 = None
    mul_2485: "f32[52]" = torch.ops.aten.mul.Tensor(sum_289, squeeze_79);  sum_289 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_143 = torch.ops.aten.convolution_backward.default(mul_2484, add_139, primals_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2484 = add_139 = primals_79 = None
    getitem_1431: "f32[8, 52, 28, 28]" = convolution_backward_143[0]
    getitem_1432: "f32[52, 52, 3, 3]" = convolution_backward_143[1];  convolution_backward_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1021: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(slice_114, getitem_1431);  slice_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_142: "f32[8, 52, 28, 28]" = torch.ops.aten.where.self(le_142, full_default, add_1021);  le_142 = add_1021 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_290: "f32[52]" = torch.ops.aten.sum.dim_IntList(where_142, [0, 2, 3])
    sub_746: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_2410);  convolution_25 = unsqueeze_2410 = None
    mul_2486: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(where_142, sub_746)
    sum_291: "f32[52]" = torch.ops.aten.sum.dim_IntList(mul_2486, [0, 2, 3]);  mul_2486 = None
    mul_2487: "f32[52]" = torch.ops.aten.mul.Tensor(sum_290, 0.00015943877551020407)
    unsqueeze_2411: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2487, 0);  mul_2487 = None
    unsqueeze_2412: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2411, 2);  unsqueeze_2411 = None
    unsqueeze_2413: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2412, 3);  unsqueeze_2412 = None
    mul_2488: "f32[52]" = torch.ops.aten.mul.Tensor(sum_291, 0.00015943877551020407)
    mul_2489: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_2490: "f32[52]" = torch.ops.aten.mul.Tensor(mul_2488, mul_2489);  mul_2488 = mul_2489 = None
    unsqueeze_2414: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2490, 0);  mul_2490 = None
    unsqueeze_2415: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2414, 2);  unsqueeze_2414 = None
    unsqueeze_2416: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2415, 3);  unsqueeze_2415 = None
    mul_2491: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_77);  primals_77 = None
    unsqueeze_2417: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2491, 0);  mul_2491 = None
    unsqueeze_2418: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2417, 2);  unsqueeze_2417 = None
    unsqueeze_2419: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2418, 3);  unsqueeze_2418 = None
    mul_2492: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_746, unsqueeze_2416);  sub_746 = unsqueeze_2416 = None
    sub_748: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(where_142, mul_2492);  where_142 = mul_2492 = None
    sub_749: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(sub_748, unsqueeze_2413);  sub_748 = unsqueeze_2413 = None
    mul_2493: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_749, unsqueeze_2419);  sub_749 = unsqueeze_2419 = None
    mul_2494: "f32[52]" = torch.ops.aten.mul.Tensor(sum_291, squeeze_76);  sum_291 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_144 = torch.ops.aten.convolution_backward.default(mul_2493, add_133, primals_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2493 = add_133 = primals_76 = None
    getitem_1434: "f32[8, 52, 28, 28]" = convolution_backward_144[0]
    getitem_1435: "f32[52, 52, 3, 3]" = convolution_backward_144[1];  convolution_backward_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1022: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(slice_113, getitem_1434);  slice_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_143: "f32[8, 52, 28, 28]" = torch.ops.aten.where.self(le_143, full_default, add_1022);  le_143 = add_1022 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_292: "f32[52]" = torch.ops.aten.sum.dim_IntList(where_143, [0, 2, 3])
    sub_750: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_2422);  convolution_24 = unsqueeze_2422 = None
    mul_2495: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(where_143, sub_750)
    sum_293: "f32[52]" = torch.ops.aten.sum.dim_IntList(mul_2495, [0, 2, 3]);  mul_2495 = None
    mul_2496: "f32[52]" = torch.ops.aten.mul.Tensor(sum_292, 0.00015943877551020407)
    unsqueeze_2423: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2496, 0);  mul_2496 = None
    unsqueeze_2424: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2423, 2);  unsqueeze_2423 = None
    unsqueeze_2425: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2424, 3);  unsqueeze_2424 = None
    mul_2497: "f32[52]" = torch.ops.aten.mul.Tensor(sum_293, 0.00015943877551020407)
    mul_2498: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_2499: "f32[52]" = torch.ops.aten.mul.Tensor(mul_2497, mul_2498);  mul_2497 = mul_2498 = None
    unsqueeze_2426: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2499, 0);  mul_2499 = None
    unsqueeze_2427: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2426, 2);  unsqueeze_2426 = None
    unsqueeze_2428: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2427, 3);  unsqueeze_2427 = None
    mul_2500: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_74);  primals_74 = None
    unsqueeze_2429: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2500, 0);  mul_2500 = None
    unsqueeze_2430: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2429, 2);  unsqueeze_2429 = None
    unsqueeze_2431: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2430, 3);  unsqueeze_2430 = None
    mul_2501: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_750, unsqueeze_2428);  sub_750 = unsqueeze_2428 = None
    sub_752: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(where_143, mul_2501);  where_143 = mul_2501 = None
    sub_753: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(sub_752, unsqueeze_2425);  sub_752 = unsqueeze_2425 = None
    mul_2502: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_753, unsqueeze_2431);  sub_753 = unsqueeze_2431 = None
    mul_2503: "f32[52]" = torch.ops.aten.mul.Tensor(sum_293, squeeze_73);  sum_293 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_145 = torch.ops.aten.convolution_backward.default(mul_2502, getitem_134, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2502 = getitem_134 = primals_73 = None
    getitem_1437: "f32[8, 52, 28, 28]" = convolution_backward_145[0]
    getitem_1438: "f32[52, 52, 3, 3]" = convolution_backward_145[1];  convolution_backward_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_61: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([getitem_1437, getitem_1434, getitem_1431, slice_116], 1);  getitem_1437 = getitem_1434 = getitem_1431 = slice_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_144: "f32[8, 208, 28, 28]" = torch.ops.aten.where.self(le_144, full_default, cat_61);  le_144 = cat_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_294: "f32[208]" = torch.ops.aten.sum.dim_IntList(where_144, [0, 2, 3])
    sub_754: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_2434);  convolution_23 = unsqueeze_2434 = None
    mul_2504: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(where_144, sub_754)
    sum_295: "f32[208]" = torch.ops.aten.sum.dim_IntList(mul_2504, [0, 2, 3]);  mul_2504 = None
    mul_2505: "f32[208]" = torch.ops.aten.mul.Tensor(sum_294, 0.00015943877551020407)
    unsqueeze_2435: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_2505, 0);  mul_2505 = None
    unsqueeze_2436: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2435, 2);  unsqueeze_2435 = None
    unsqueeze_2437: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2436, 3);  unsqueeze_2436 = None
    mul_2506: "f32[208]" = torch.ops.aten.mul.Tensor(sum_295, 0.00015943877551020407)
    mul_2507: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_2508: "f32[208]" = torch.ops.aten.mul.Tensor(mul_2506, mul_2507);  mul_2506 = mul_2507 = None
    unsqueeze_2438: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_2508, 0);  mul_2508 = None
    unsqueeze_2439: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2438, 2);  unsqueeze_2438 = None
    unsqueeze_2440: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2439, 3);  unsqueeze_2439 = None
    mul_2509: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_71);  primals_71 = None
    unsqueeze_2441: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_2509, 0);  mul_2509 = None
    unsqueeze_2442: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2441, 2);  unsqueeze_2441 = None
    unsqueeze_2443: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2442, 3);  unsqueeze_2442 = None
    mul_2510: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_754, unsqueeze_2440);  sub_754 = unsqueeze_2440 = None
    sub_756: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(where_144, mul_2510);  where_144 = mul_2510 = None
    sub_757: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(sub_756, unsqueeze_2437);  sub_756 = unsqueeze_2437 = None
    mul_2511: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_757, unsqueeze_2443);  sub_757 = unsqueeze_2443 = None
    mul_2512: "f32[208]" = torch.ops.aten.mul.Tensor(sum_295, squeeze_70);  sum_295 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_146 = torch.ops.aten.convolution_backward.default(mul_2511, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2511 = primals_70 = None
    getitem_1440: "f32[8, 512, 28, 28]" = convolution_backward_146[0]
    getitem_1441: "f32[208, 512, 1, 1]" = convolution_backward_146[1];  convolution_backward_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_1023: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_140, getitem_1440);  where_140 = getitem_1440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_145: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    where_145: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_145, full_default, add_1023);  le_145 = add_1023 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    sum_296: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_145, [0, 2, 3])
    sub_758: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_2446);  convolution_22 = unsqueeze_2446 = None
    mul_2513: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_145, sub_758)
    sum_297: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_2513, [0, 2, 3]);  mul_2513 = None
    mul_2514: "f32[512]" = torch.ops.aten.mul.Tensor(sum_296, 0.00015943877551020407)
    unsqueeze_2447: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2514, 0);  mul_2514 = None
    unsqueeze_2448: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2447, 2);  unsqueeze_2447 = None
    unsqueeze_2449: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2448, 3);  unsqueeze_2448 = None
    mul_2515: "f32[512]" = torch.ops.aten.mul.Tensor(sum_297, 0.00015943877551020407)
    mul_2516: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_2517: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2515, mul_2516);  mul_2515 = mul_2516 = None
    unsqueeze_2450: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2517, 0);  mul_2517 = None
    unsqueeze_2451: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2450, 2);  unsqueeze_2450 = None
    unsqueeze_2452: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2451, 3);  unsqueeze_2451 = None
    mul_2518: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_68);  primals_68 = None
    unsqueeze_2453: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2518, 0);  mul_2518 = None
    unsqueeze_2454: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2453, 2);  unsqueeze_2453 = None
    unsqueeze_2455: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2454, 3);  unsqueeze_2454 = None
    mul_2519: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_758, unsqueeze_2452);  sub_758 = unsqueeze_2452 = None
    sub_760: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_145, mul_2519);  mul_2519 = None
    sub_761: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_760, unsqueeze_2449);  sub_760 = None
    mul_2520: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_761, unsqueeze_2455);  sub_761 = unsqueeze_2455 = None
    mul_2521: "f32[512]" = torch.ops.aten.mul.Tensor(sum_297, squeeze_67);  sum_297 = squeeze_67 = None
    convolution_backward_147 = torch.ops.aten.convolution_backward.default(mul_2520, relu_15, primals_67, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2520 = primals_67 = None
    getitem_1443: "f32[8, 256, 56, 56]" = convolution_backward_147[0]
    getitem_1444: "f32[512, 256, 1, 1]" = convolution_backward_147[1];  convolution_backward_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sub_762: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_2458);  convolution_21 = unsqueeze_2458 = None
    mul_2522: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_145, sub_762)
    sum_299: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_2522, [0, 2, 3]);  mul_2522 = None
    mul_2524: "f32[512]" = torch.ops.aten.mul.Tensor(sum_299, 0.00015943877551020407)
    mul_2525: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_2526: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2524, mul_2525);  mul_2524 = mul_2525 = None
    unsqueeze_2462: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2526, 0);  mul_2526 = None
    unsqueeze_2463: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2462, 2);  unsqueeze_2462 = None
    unsqueeze_2464: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2463, 3);  unsqueeze_2463 = None
    mul_2527: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_65);  primals_65 = None
    unsqueeze_2465: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2527, 0);  mul_2527 = None
    unsqueeze_2466: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2465, 2);  unsqueeze_2465 = None
    unsqueeze_2467: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2466, 3);  unsqueeze_2466 = None
    mul_2528: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_762, unsqueeze_2464);  sub_762 = unsqueeze_2464 = None
    sub_764: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_145, mul_2528);  where_145 = mul_2528 = None
    sub_765: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_764, unsqueeze_2449);  sub_764 = unsqueeze_2449 = None
    mul_2529: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_765, unsqueeze_2467);  sub_765 = unsqueeze_2467 = None
    mul_2530: "f32[512]" = torch.ops.aten.mul.Tensor(sum_299, squeeze_64);  sum_299 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_148 = torch.ops.aten.convolution_backward.default(mul_2529, cat_3, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2529 = cat_3 = primals_64 = None
    getitem_1446: "f32[8, 208, 28, 28]" = convolution_backward_148[0]
    getitem_1447: "f32[512, 208, 1, 1]" = convolution_backward_148[1];  convolution_backward_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_117: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1446, 1, 0, 52)
    slice_118: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1446, 1, 52, 104)
    slice_119: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1446, 1, 104, 156)
    slice_120: "f32[8, 52, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1446, 1, 156, 208);  getitem_1446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_backward_2: "f32[8, 52, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(slice_120, getitem_123, [3, 3], [2, 2], [1, 1], False, True, None);  slice_120 = getitem_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_146: "f32[8, 52, 28, 28]" = torch.ops.aten.where.self(le_146, full_default, slice_119);  le_146 = slice_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_300: "f32[52]" = torch.ops.aten.sum.dim_IntList(where_146, [0, 2, 3])
    sub_766: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_2470);  convolution_20 = unsqueeze_2470 = None
    mul_2531: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(where_146, sub_766)
    sum_301: "f32[52]" = torch.ops.aten.sum.dim_IntList(mul_2531, [0, 2, 3]);  mul_2531 = None
    mul_2532: "f32[52]" = torch.ops.aten.mul.Tensor(sum_300, 0.00015943877551020407)
    unsqueeze_2471: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2532, 0);  mul_2532 = None
    unsqueeze_2472: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2471, 2);  unsqueeze_2471 = None
    unsqueeze_2473: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2472, 3);  unsqueeze_2472 = None
    mul_2533: "f32[52]" = torch.ops.aten.mul.Tensor(sum_301, 0.00015943877551020407)
    mul_2534: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_2535: "f32[52]" = torch.ops.aten.mul.Tensor(mul_2533, mul_2534);  mul_2533 = mul_2534 = None
    unsqueeze_2474: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2535, 0);  mul_2535 = None
    unsqueeze_2475: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2474, 2);  unsqueeze_2474 = None
    unsqueeze_2476: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2475, 3);  unsqueeze_2475 = None
    mul_2536: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_62);  primals_62 = None
    unsqueeze_2477: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2536, 0);  mul_2536 = None
    unsqueeze_2478: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2477, 2);  unsqueeze_2477 = None
    unsqueeze_2479: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2478, 3);  unsqueeze_2478 = None
    mul_2537: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_766, unsqueeze_2476);  sub_766 = unsqueeze_2476 = None
    sub_768: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(where_146, mul_2537);  where_146 = mul_2537 = None
    sub_769: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(sub_768, unsqueeze_2473);  sub_768 = unsqueeze_2473 = None
    mul_2538: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_769, unsqueeze_2479);  sub_769 = unsqueeze_2479 = None
    mul_2539: "f32[52]" = torch.ops.aten.mul.Tensor(sum_301, squeeze_61);  sum_301 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_149 = torch.ops.aten.convolution_backward.default(mul_2538, getitem_116, primals_61, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2538 = getitem_116 = primals_61 = None
    getitem_1449: "f32[8, 52, 56, 56]" = convolution_backward_149[0]
    getitem_1450: "f32[52, 52, 3, 3]" = convolution_backward_149[1];  convolution_backward_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_147: "f32[8, 52, 28, 28]" = torch.ops.aten.where.self(le_147, full_default, slice_118);  le_147 = slice_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_302: "f32[52]" = torch.ops.aten.sum.dim_IntList(where_147, [0, 2, 3])
    sub_770: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_2482);  convolution_19 = unsqueeze_2482 = None
    mul_2540: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(where_147, sub_770)
    sum_303: "f32[52]" = torch.ops.aten.sum.dim_IntList(mul_2540, [0, 2, 3]);  mul_2540 = None
    mul_2541: "f32[52]" = torch.ops.aten.mul.Tensor(sum_302, 0.00015943877551020407)
    unsqueeze_2483: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2541, 0);  mul_2541 = None
    unsqueeze_2484: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2483, 2);  unsqueeze_2483 = None
    unsqueeze_2485: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2484, 3);  unsqueeze_2484 = None
    mul_2542: "f32[52]" = torch.ops.aten.mul.Tensor(sum_303, 0.00015943877551020407)
    mul_2543: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_2544: "f32[52]" = torch.ops.aten.mul.Tensor(mul_2542, mul_2543);  mul_2542 = mul_2543 = None
    unsqueeze_2486: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2544, 0);  mul_2544 = None
    unsqueeze_2487: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2486, 2);  unsqueeze_2486 = None
    unsqueeze_2488: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2487, 3);  unsqueeze_2487 = None
    mul_2545: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_59);  primals_59 = None
    unsqueeze_2489: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2545, 0);  mul_2545 = None
    unsqueeze_2490: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2489, 2);  unsqueeze_2489 = None
    unsqueeze_2491: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2490, 3);  unsqueeze_2490 = None
    mul_2546: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_770, unsqueeze_2488);  sub_770 = unsqueeze_2488 = None
    sub_772: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(where_147, mul_2546);  where_147 = mul_2546 = None
    sub_773: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(sub_772, unsqueeze_2485);  sub_772 = unsqueeze_2485 = None
    mul_2547: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_773, unsqueeze_2491);  sub_773 = unsqueeze_2491 = None
    mul_2548: "f32[52]" = torch.ops.aten.mul.Tensor(sum_303, squeeze_58);  sum_303 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_150 = torch.ops.aten.convolution_backward.default(mul_2547, getitem_109, primals_58, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2547 = getitem_109 = primals_58 = None
    getitem_1452: "f32[8, 52, 56, 56]" = convolution_backward_150[0]
    getitem_1453: "f32[52, 52, 3, 3]" = convolution_backward_150[1];  convolution_backward_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_148: "f32[8, 52, 28, 28]" = torch.ops.aten.where.self(le_148, full_default, slice_117);  le_148 = slice_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_304: "f32[52]" = torch.ops.aten.sum.dim_IntList(where_148, [0, 2, 3])
    sub_774: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_2494);  convolution_18 = unsqueeze_2494 = None
    mul_2549: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(where_148, sub_774)
    sum_305: "f32[52]" = torch.ops.aten.sum.dim_IntList(mul_2549, [0, 2, 3]);  mul_2549 = None
    mul_2550: "f32[52]" = torch.ops.aten.mul.Tensor(sum_304, 0.00015943877551020407)
    unsqueeze_2495: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2550, 0);  mul_2550 = None
    unsqueeze_2496: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2495, 2);  unsqueeze_2495 = None
    unsqueeze_2497: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2496, 3);  unsqueeze_2496 = None
    mul_2551: "f32[52]" = torch.ops.aten.mul.Tensor(sum_305, 0.00015943877551020407)
    mul_2552: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_2553: "f32[52]" = torch.ops.aten.mul.Tensor(mul_2551, mul_2552);  mul_2551 = mul_2552 = None
    unsqueeze_2498: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2553, 0);  mul_2553 = None
    unsqueeze_2499: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2498, 2);  unsqueeze_2498 = None
    unsqueeze_2500: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2499, 3);  unsqueeze_2499 = None
    mul_2554: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_56);  primals_56 = None
    unsqueeze_2501: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(mul_2554, 0);  mul_2554 = None
    unsqueeze_2502: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2501, 2);  unsqueeze_2501 = None
    unsqueeze_2503: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2502, 3);  unsqueeze_2502 = None
    mul_2555: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_774, unsqueeze_2500);  sub_774 = unsqueeze_2500 = None
    sub_776: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(where_148, mul_2555);  where_148 = mul_2555 = None
    sub_777: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(sub_776, unsqueeze_2497);  sub_776 = unsqueeze_2497 = None
    mul_2556: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_777, unsqueeze_2503);  sub_777 = unsqueeze_2503 = None
    mul_2557: "f32[52]" = torch.ops.aten.mul.Tensor(sum_305, squeeze_55);  sum_305 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_151 = torch.ops.aten.convolution_backward.default(mul_2556, getitem_102, primals_55, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2556 = getitem_102 = primals_55 = None
    getitem_1455: "f32[8, 52, 56, 56]" = convolution_backward_151[0]
    getitem_1456: "f32[52, 52, 3, 3]" = convolution_backward_151[1];  convolution_backward_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_62: "f32[8, 208, 56, 56]" = torch.ops.aten.cat.default([getitem_1455, getitem_1452, getitem_1449, avg_pool2d_backward_2], 1);  getitem_1455 = getitem_1452 = getitem_1449 = avg_pool2d_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_149: "f32[8, 208, 56, 56]" = torch.ops.aten.where.self(le_149, full_default, cat_62);  le_149 = cat_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_306: "f32[208]" = torch.ops.aten.sum.dim_IntList(where_149, [0, 2, 3])
    sub_778: "f32[8, 208, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_2506);  convolution_17 = unsqueeze_2506 = None
    mul_2558: "f32[8, 208, 56, 56]" = torch.ops.aten.mul.Tensor(where_149, sub_778)
    sum_307: "f32[208]" = torch.ops.aten.sum.dim_IntList(mul_2558, [0, 2, 3]);  mul_2558 = None
    mul_2559: "f32[208]" = torch.ops.aten.mul.Tensor(sum_306, 3.985969387755102e-05)
    unsqueeze_2507: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_2559, 0);  mul_2559 = None
    unsqueeze_2508: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2507, 2);  unsqueeze_2507 = None
    unsqueeze_2509: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2508, 3);  unsqueeze_2508 = None
    mul_2560: "f32[208]" = torch.ops.aten.mul.Tensor(sum_307, 3.985969387755102e-05)
    mul_2561: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_2562: "f32[208]" = torch.ops.aten.mul.Tensor(mul_2560, mul_2561);  mul_2560 = mul_2561 = None
    unsqueeze_2510: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_2562, 0);  mul_2562 = None
    unsqueeze_2511: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2510, 2);  unsqueeze_2510 = None
    unsqueeze_2512: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2511, 3);  unsqueeze_2511 = None
    mul_2563: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_53);  primals_53 = None
    unsqueeze_2513: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(mul_2563, 0);  mul_2563 = None
    unsqueeze_2514: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2513, 2);  unsqueeze_2513 = None
    unsqueeze_2515: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2514, 3);  unsqueeze_2514 = None
    mul_2564: "f32[8, 208, 56, 56]" = torch.ops.aten.mul.Tensor(sub_778, unsqueeze_2512);  sub_778 = unsqueeze_2512 = None
    sub_780: "f32[8, 208, 56, 56]" = torch.ops.aten.sub.Tensor(where_149, mul_2564);  where_149 = mul_2564 = None
    sub_781: "f32[8, 208, 56, 56]" = torch.ops.aten.sub.Tensor(sub_780, unsqueeze_2509);  sub_780 = unsqueeze_2509 = None
    mul_2565: "f32[8, 208, 56, 56]" = torch.ops.aten.mul.Tensor(sub_781, unsqueeze_2515);  sub_781 = unsqueeze_2515 = None
    mul_2566: "f32[208]" = torch.ops.aten.mul.Tensor(sum_307, squeeze_52);  sum_307 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_152 = torch.ops.aten.convolution_backward.default(mul_2565, relu_15, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2565 = primals_52 = None
    getitem_1458: "f32[8, 256, 56, 56]" = convolution_backward_152[0]
    getitem_1459: "f32[208, 256, 1, 1]" = convolution_backward_152[1];  convolution_backward_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_1024: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(getitem_1443, getitem_1458);  getitem_1443 = getitem_1458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_150: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_150: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_150, full_default, add_1024);  le_150 = add_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_308: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_150, [0, 2, 3])
    sub_782: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_2518);  convolution_16 = unsqueeze_2518 = None
    mul_2567: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_150, sub_782)
    sum_309: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2567, [0, 2, 3]);  mul_2567 = None
    mul_2568: "f32[256]" = torch.ops.aten.mul.Tensor(sum_308, 3.985969387755102e-05)
    unsqueeze_2519: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2568, 0);  mul_2568 = None
    unsqueeze_2520: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2519, 2);  unsqueeze_2519 = None
    unsqueeze_2521: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2520, 3);  unsqueeze_2520 = None
    mul_2569: "f32[256]" = torch.ops.aten.mul.Tensor(sum_309, 3.985969387755102e-05)
    mul_2570: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_2571: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2569, mul_2570);  mul_2569 = mul_2570 = None
    unsqueeze_2522: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2571, 0);  mul_2571 = None
    unsqueeze_2523: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2522, 2);  unsqueeze_2522 = None
    unsqueeze_2524: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2523, 3);  unsqueeze_2523 = None
    mul_2572: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_50);  primals_50 = None
    unsqueeze_2525: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2572, 0);  mul_2572 = None
    unsqueeze_2526: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2525, 2);  unsqueeze_2525 = None
    unsqueeze_2527: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2526, 3);  unsqueeze_2526 = None
    mul_2573: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_782, unsqueeze_2524);  sub_782 = unsqueeze_2524 = None
    sub_784: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_150, mul_2573);  mul_2573 = None
    sub_785: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_784, unsqueeze_2521);  sub_784 = unsqueeze_2521 = None
    mul_2574: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_785, unsqueeze_2527);  sub_785 = unsqueeze_2527 = None
    mul_2575: "f32[256]" = torch.ops.aten.mul.Tensor(sum_309, squeeze_49);  sum_309 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_153 = torch.ops.aten.convolution_backward.default(mul_2574, cat_2, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2574 = cat_2 = primals_49 = None
    getitem_1461: "f32[8, 104, 56, 56]" = convolution_backward_153[0]
    getitem_1462: "f32[256, 104, 1, 1]" = convolution_backward_153[1];  convolution_backward_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_121: "f32[8, 26, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1461, 1, 0, 26)
    slice_122: "f32[8, 26, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1461, 1, 26, 52)
    slice_123: "f32[8, 26, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1461, 1, 52, 78)
    slice_124: "f32[8, 26, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1461, 1, 78, 104);  getitem_1461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_151: "f32[8, 26, 56, 56]" = torch.ops.aten.where.self(le_151, full_default, slice_123);  le_151 = slice_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_310: "f32[26]" = torch.ops.aten.sum.dim_IntList(where_151, [0, 2, 3])
    sub_786: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_2530);  convolution_15 = unsqueeze_2530 = None
    mul_2576: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(where_151, sub_786)
    sum_311: "f32[26]" = torch.ops.aten.sum.dim_IntList(mul_2576, [0, 2, 3]);  mul_2576 = None
    mul_2577: "f32[26]" = torch.ops.aten.mul.Tensor(sum_310, 3.985969387755102e-05)
    unsqueeze_2531: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2577, 0);  mul_2577 = None
    unsqueeze_2532: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2531, 2);  unsqueeze_2531 = None
    unsqueeze_2533: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2532, 3);  unsqueeze_2532 = None
    mul_2578: "f32[26]" = torch.ops.aten.mul.Tensor(sum_311, 3.985969387755102e-05)
    mul_2579: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_2580: "f32[26]" = torch.ops.aten.mul.Tensor(mul_2578, mul_2579);  mul_2578 = mul_2579 = None
    unsqueeze_2534: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2580, 0);  mul_2580 = None
    unsqueeze_2535: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2534, 2);  unsqueeze_2534 = None
    unsqueeze_2536: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2535, 3);  unsqueeze_2535 = None
    mul_2581: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_47);  primals_47 = None
    unsqueeze_2537: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2581, 0);  mul_2581 = None
    unsqueeze_2538: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2537, 2);  unsqueeze_2537 = None
    unsqueeze_2539: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2538, 3);  unsqueeze_2538 = None
    mul_2582: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_786, unsqueeze_2536);  sub_786 = unsqueeze_2536 = None
    sub_788: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(where_151, mul_2582);  where_151 = mul_2582 = None
    sub_789: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(sub_788, unsqueeze_2533);  sub_788 = unsqueeze_2533 = None
    mul_2583: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_789, unsqueeze_2539);  sub_789 = unsqueeze_2539 = None
    mul_2584: "f32[26]" = torch.ops.aten.mul.Tensor(sum_311, squeeze_46);  sum_311 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_154 = torch.ops.aten.convolution_backward.default(mul_2583, add_80, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2583 = add_80 = primals_46 = None
    getitem_1464: "f32[8, 26, 56, 56]" = convolution_backward_154[0]
    getitem_1465: "f32[26, 26, 3, 3]" = convolution_backward_154[1];  convolution_backward_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1025: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(slice_122, getitem_1464);  slice_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_152: "f32[8, 26, 56, 56]" = torch.ops.aten.where.self(le_152, full_default, add_1025);  le_152 = add_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_312: "f32[26]" = torch.ops.aten.sum.dim_IntList(where_152, [0, 2, 3])
    sub_790: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_2542);  convolution_14 = unsqueeze_2542 = None
    mul_2585: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(where_152, sub_790)
    sum_313: "f32[26]" = torch.ops.aten.sum.dim_IntList(mul_2585, [0, 2, 3]);  mul_2585 = None
    mul_2586: "f32[26]" = torch.ops.aten.mul.Tensor(sum_312, 3.985969387755102e-05)
    unsqueeze_2543: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2586, 0);  mul_2586 = None
    unsqueeze_2544: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2543, 2);  unsqueeze_2543 = None
    unsqueeze_2545: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2544, 3);  unsqueeze_2544 = None
    mul_2587: "f32[26]" = torch.ops.aten.mul.Tensor(sum_313, 3.985969387755102e-05)
    mul_2588: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_2589: "f32[26]" = torch.ops.aten.mul.Tensor(mul_2587, mul_2588);  mul_2587 = mul_2588 = None
    unsqueeze_2546: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2589, 0);  mul_2589 = None
    unsqueeze_2547: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2546, 2);  unsqueeze_2546 = None
    unsqueeze_2548: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2547, 3);  unsqueeze_2547 = None
    mul_2590: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_44);  primals_44 = None
    unsqueeze_2549: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2590, 0);  mul_2590 = None
    unsqueeze_2550: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2549, 2);  unsqueeze_2549 = None
    unsqueeze_2551: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2550, 3);  unsqueeze_2550 = None
    mul_2591: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_790, unsqueeze_2548);  sub_790 = unsqueeze_2548 = None
    sub_792: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(where_152, mul_2591);  where_152 = mul_2591 = None
    sub_793: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(sub_792, unsqueeze_2545);  sub_792 = unsqueeze_2545 = None
    mul_2592: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_793, unsqueeze_2551);  sub_793 = unsqueeze_2551 = None
    mul_2593: "f32[26]" = torch.ops.aten.mul.Tensor(sum_313, squeeze_43);  sum_313 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_155 = torch.ops.aten.convolution_backward.default(mul_2592, add_74, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2592 = add_74 = primals_43 = None
    getitem_1467: "f32[8, 26, 56, 56]" = convolution_backward_155[0]
    getitem_1468: "f32[26, 26, 3, 3]" = convolution_backward_155[1];  convolution_backward_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1026: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(slice_121, getitem_1467);  slice_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_153: "f32[8, 26, 56, 56]" = torch.ops.aten.where.self(le_153, full_default, add_1026);  le_153 = add_1026 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_314: "f32[26]" = torch.ops.aten.sum.dim_IntList(where_153, [0, 2, 3])
    sub_794: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_2554);  convolution_13 = unsqueeze_2554 = None
    mul_2594: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(where_153, sub_794)
    sum_315: "f32[26]" = torch.ops.aten.sum.dim_IntList(mul_2594, [0, 2, 3]);  mul_2594 = None
    mul_2595: "f32[26]" = torch.ops.aten.mul.Tensor(sum_314, 3.985969387755102e-05)
    unsqueeze_2555: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2595, 0);  mul_2595 = None
    unsqueeze_2556: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2555, 2);  unsqueeze_2555 = None
    unsqueeze_2557: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2556, 3);  unsqueeze_2556 = None
    mul_2596: "f32[26]" = torch.ops.aten.mul.Tensor(sum_315, 3.985969387755102e-05)
    mul_2597: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_2598: "f32[26]" = torch.ops.aten.mul.Tensor(mul_2596, mul_2597);  mul_2596 = mul_2597 = None
    unsqueeze_2558: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2598, 0);  mul_2598 = None
    unsqueeze_2559: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2558, 2);  unsqueeze_2558 = None
    unsqueeze_2560: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2559, 3);  unsqueeze_2559 = None
    mul_2599: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_41);  primals_41 = None
    unsqueeze_2561: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2599, 0);  mul_2599 = None
    unsqueeze_2562: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2561, 2);  unsqueeze_2561 = None
    unsqueeze_2563: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2562, 3);  unsqueeze_2562 = None
    mul_2600: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_794, unsqueeze_2560);  sub_794 = unsqueeze_2560 = None
    sub_796: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(where_153, mul_2600);  where_153 = mul_2600 = None
    sub_797: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(sub_796, unsqueeze_2557);  sub_796 = unsqueeze_2557 = None
    mul_2601: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_797, unsqueeze_2563);  sub_797 = unsqueeze_2563 = None
    mul_2602: "f32[26]" = torch.ops.aten.mul.Tensor(sum_315, squeeze_40);  sum_315 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_156 = torch.ops.aten.convolution_backward.default(mul_2601, getitem_72, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2601 = getitem_72 = primals_40 = None
    getitem_1470: "f32[8, 26, 56, 56]" = convolution_backward_156[0]
    getitem_1471: "f32[26, 26, 3, 3]" = convolution_backward_156[1];  convolution_backward_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_63: "f32[8, 104, 56, 56]" = torch.ops.aten.cat.default([getitem_1470, getitem_1467, getitem_1464, slice_124], 1);  getitem_1470 = getitem_1467 = getitem_1464 = slice_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_154: "f32[8, 104, 56, 56]" = torch.ops.aten.where.self(le_154, full_default, cat_63);  le_154 = cat_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_316: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_154, [0, 2, 3])
    sub_798: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_2566);  convolution_12 = unsqueeze_2566 = None
    mul_2603: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(where_154, sub_798)
    sum_317: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2603, [0, 2, 3]);  mul_2603 = None
    mul_2604: "f32[104]" = torch.ops.aten.mul.Tensor(sum_316, 3.985969387755102e-05)
    unsqueeze_2567: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2604, 0);  mul_2604 = None
    unsqueeze_2568: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2567, 2);  unsqueeze_2567 = None
    unsqueeze_2569: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2568, 3);  unsqueeze_2568 = None
    mul_2605: "f32[104]" = torch.ops.aten.mul.Tensor(sum_317, 3.985969387755102e-05)
    mul_2606: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_2607: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2605, mul_2606);  mul_2605 = mul_2606 = None
    unsqueeze_2570: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2607, 0);  mul_2607 = None
    unsqueeze_2571: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2570, 2);  unsqueeze_2570 = None
    unsqueeze_2572: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2571, 3);  unsqueeze_2571 = None
    mul_2608: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_38);  primals_38 = None
    unsqueeze_2573: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2608, 0);  mul_2608 = None
    unsqueeze_2574: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2573, 2);  unsqueeze_2573 = None
    unsqueeze_2575: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2574, 3);  unsqueeze_2574 = None
    mul_2609: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_798, unsqueeze_2572);  sub_798 = unsqueeze_2572 = None
    sub_800: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(where_154, mul_2609);  where_154 = mul_2609 = None
    sub_801: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(sub_800, unsqueeze_2569);  sub_800 = unsqueeze_2569 = None
    mul_2610: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_801, unsqueeze_2575);  sub_801 = unsqueeze_2575 = None
    mul_2611: "f32[104]" = torch.ops.aten.mul.Tensor(sum_317, squeeze_37);  sum_317 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_157 = torch.ops.aten.convolution_backward.default(mul_2610, relu_10, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2610 = primals_37 = None
    getitem_1473: "f32[8, 256, 56, 56]" = convolution_backward_157[0]
    getitem_1474: "f32[104, 256, 1, 1]" = convolution_backward_157[1];  convolution_backward_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_1027: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_150, getitem_1473);  where_150 = getitem_1473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_155: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_155: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_155, full_default, add_1027);  le_155 = add_1027 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_318: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_155, [0, 2, 3])
    sub_802: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_2578);  convolution_11 = unsqueeze_2578 = None
    mul_2612: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_155, sub_802)
    sum_319: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2612, [0, 2, 3]);  mul_2612 = None
    mul_2613: "f32[256]" = torch.ops.aten.mul.Tensor(sum_318, 3.985969387755102e-05)
    unsqueeze_2579: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2613, 0);  mul_2613 = None
    unsqueeze_2580: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2579, 2);  unsqueeze_2579 = None
    unsqueeze_2581: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2580, 3);  unsqueeze_2580 = None
    mul_2614: "f32[256]" = torch.ops.aten.mul.Tensor(sum_319, 3.985969387755102e-05)
    mul_2615: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_2616: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2614, mul_2615);  mul_2614 = mul_2615 = None
    unsqueeze_2582: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2616, 0);  mul_2616 = None
    unsqueeze_2583: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2582, 2);  unsqueeze_2582 = None
    unsqueeze_2584: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2583, 3);  unsqueeze_2583 = None
    mul_2617: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_35);  primals_35 = None
    unsqueeze_2585: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2617, 0);  mul_2617 = None
    unsqueeze_2586: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2585, 2);  unsqueeze_2585 = None
    unsqueeze_2587: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2586, 3);  unsqueeze_2586 = None
    mul_2618: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_802, unsqueeze_2584);  sub_802 = unsqueeze_2584 = None
    sub_804: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_155, mul_2618);  mul_2618 = None
    sub_805: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_804, unsqueeze_2581);  sub_804 = unsqueeze_2581 = None
    mul_2619: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_805, unsqueeze_2587);  sub_805 = unsqueeze_2587 = None
    mul_2620: "f32[256]" = torch.ops.aten.mul.Tensor(sum_319, squeeze_34);  sum_319 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_158 = torch.ops.aten.convolution_backward.default(mul_2619, cat_1, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2619 = cat_1 = primals_34 = None
    getitem_1476: "f32[8, 104, 56, 56]" = convolution_backward_158[0]
    getitem_1477: "f32[256, 104, 1, 1]" = convolution_backward_158[1];  convolution_backward_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_125: "f32[8, 26, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1476, 1, 0, 26)
    slice_126: "f32[8, 26, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1476, 1, 26, 52)
    slice_127: "f32[8, 26, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1476, 1, 52, 78)
    slice_128: "f32[8, 26, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1476, 1, 78, 104);  getitem_1476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_156: "f32[8, 26, 56, 56]" = torch.ops.aten.where.self(le_156, full_default, slice_127);  le_156 = slice_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_320: "f32[26]" = torch.ops.aten.sum.dim_IntList(where_156, [0, 2, 3])
    sub_806: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_2590);  convolution_10 = unsqueeze_2590 = None
    mul_2621: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(where_156, sub_806)
    sum_321: "f32[26]" = torch.ops.aten.sum.dim_IntList(mul_2621, [0, 2, 3]);  mul_2621 = None
    mul_2622: "f32[26]" = torch.ops.aten.mul.Tensor(sum_320, 3.985969387755102e-05)
    unsqueeze_2591: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2622, 0);  mul_2622 = None
    unsqueeze_2592: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2591, 2);  unsqueeze_2591 = None
    unsqueeze_2593: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2592, 3);  unsqueeze_2592 = None
    mul_2623: "f32[26]" = torch.ops.aten.mul.Tensor(sum_321, 3.985969387755102e-05)
    mul_2624: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_2625: "f32[26]" = torch.ops.aten.mul.Tensor(mul_2623, mul_2624);  mul_2623 = mul_2624 = None
    unsqueeze_2594: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2625, 0);  mul_2625 = None
    unsqueeze_2595: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2594, 2);  unsqueeze_2594 = None
    unsqueeze_2596: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2595, 3);  unsqueeze_2595 = None
    mul_2626: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_32);  primals_32 = None
    unsqueeze_2597: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2626, 0);  mul_2626 = None
    unsqueeze_2598: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2597, 2);  unsqueeze_2597 = None
    unsqueeze_2599: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2598, 3);  unsqueeze_2598 = None
    mul_2627: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_806, unsqueeze_2596);  sub_806 = unsqueeze_2596 = None
    sub_808: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(where_156, mul_2627);  where_156 = mul_2627 = None
    sub_809: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(sub_808, unsqueeze_2593);  sub_808 = unsqueeze_2593 = None
    mul_2628: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_809, unsqueeze_2599);  sub_809 = unsqueeze_2599 = None
    mul_2629: "f32[26]" = torch.ops.aten.mul.Tensor(sum_321, squeeze_31);  sum_321 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_159 = torch.ops.aten.convolution_backward.default(mul_2628, add_52, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2628 = add_52 = primals_31 = None
    getitem_1479: "f32[8, 26, 56, 56]" = convolution_backward_159[0]
    getitem_1480: "f32[26, 26, 3, 3]" = convolution_backward_159[1];  convolution_backward_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1028: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(slice_126, getitem_1479);  slice_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_157: "f32[8, 26, 56, 56]" = torch.ops.aten.where.self(le_157, full_default, add_1028);  le_157 = add_1028 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_322: "f32[26]" = torch.ops.aten.sum.dim_IntList(where_157, [0, 2, 3])
    sub_810: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_2602);  convolution_9 = unsqueeze_2602 = None
    mul_2630: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(where_157, sub_810)
    sum_323: "f32[26]" = torch.ops.aten.sum.dim_IntList(mul_2630, [0, 2, 3]);  mul_2630 = None
    mul_2631: "f32[26]" = torch.ops.aten.mul.Tensor(sum_322, 3.985969387755102e-05)
    unsqueeze_2603: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2631, 0);  mul_2631 = None
    unsqueeze_2604: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2603, 2);  unsqueeze_2603 = None
    unsqueeze_2605: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2604, 3);  unsqueeze_2604 = None
    mul_2632: "f32[26]" = torch.ops.aten.mul.Tensor(sum_323, 3.985969387755102e-05)
    mul_2633: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_2634: "f32[26]" = torch.ops.aten.mul.Tensor(mul_2632, mul_2633);  mul_2632 = mul_2633 = None
    unsqueeze_2606: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2634, 0);  mul_2634 = None
    unsqueeze_2607: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2606, 2);  unsqueeze_2606 = None
    unsqueeze_2608: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2607, 3);  unsqueeze_2607 = None
    mul_2635: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_29);  primals_29 = None
    unsqueeze_2609: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2635, 0);  mul_2635 = None
    unsqueeze_2610: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2609, 2);  unsqueeze_2609 = None
    unsqueeze_2611: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2610, 3);  unsqueeze_2610 = None
    mul_2636: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_810, unsqueeze_2608);  sub_810 = unsqueeze_2608 = None
    sub_812: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(where_157, mul_2636);  where_157 = mul_2636 = None
    sub_813: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(sub_812, unsqueeze_2605);  sub_812 = unsqueeze_2605 = None
    mul_2637: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_813, unsqueeze_2611);  sub_813 = unsqueeze_2611 = None
    mul_2638: "f32[26]" = torch.ops.aten.mul.Tensor(sum_323, squeeze_28);  sum_323 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_160 = torch.ops.aten.convolution_backward.default(mul_2637, add_46, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2637 = add_46 = primals_28 = None
    getitem_1482: "f32[8, 26, 56, 56]" = convolution_backward_160[0]
    getitem_1483: "f32[26, 26, 3, 3]" = convolution_backward_160[1];  convolution_backward_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_1029: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(slice_125, getitem_1482);  slice_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_158: "f32[8, 26, 56, 56]" = torch.ops.aten.where.self(le_158, full_default, add_1029);  le_158 = add_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_324: "f32[26]" = torch.ops.aten.sum.dim_IntList(where_158, [0, 2, 3])
    sub_814: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_2614);  convolution_8 = unsqueeze_2614 = None
    mul_2639: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(where_158, sub_814)
    sum_325: "f32[26]" = torch.ops.aten.sum.dim_IntList(mul_2639, [0, 2, 3]);  mul_2639 = None
    mul_2640: "f32[26]" = torch.ops.aten.mul.Tensor(sum_324, 3.985969387755102e-05)
    unsqueeze_2615: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2640, 0);  mul_2640 = None
    unsqueeze_2616: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2615, 2);  unsqueeze_2615 = None
    unsqueeze_2617: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2616, 3);  unsqueeze_2616 = None
    mul_2641: "f32[26]" = torch.ops.aten.mul.Tensor(sum_325, 3.985969387755102e-05)
    mul_2642: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_2643: "f32[26]" = torch.ops.aten.mul.Tensor(mul_2641, mul_2642);  mul_2641 = mul_2642 = None
    unsqueeze_2618: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2643, 0);  mul_2643 = None
    unsqueeze_2619: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2618, 2);  unsqueeze_2618 = None
    unsqueeze_2620: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2619, 3);  unsqueeze_2619 = None
    mul_2644: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_26);  primals_26 = None
    unsqueeze_2621: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2644, 0);  mul_2644 = None
    unsqueeze_2622: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2621, 2);  unsqueeze_2621 = None
    unsqueeze_2623: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2622, 3);  unsqueeze_2622 = None
    mul_2645: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_814, unsqueeze_2620);  sub_814 = unsqueeze_2620 = None
    sub_816: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(where_158, mul_2645);  where_158 = mul_2645 = None
    sub_817: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(sub_816, unsqueeze_2617);  sub_816 = unsqueeze_2617 = None
    mul_2646: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_817, unsqueeze_2623);  sub_817 = unsqueeze_2623 = None
    mul_2647: "f32[26]" = torch.ops.aten.mul.Tensor(sum_325, squeeze_25);  sum_325 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_161 = torch.ops.aten.convolution_backward.default(mul_2646, getitem_42, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2646 = getitem_42 = primals_25 = None
    getitem_1485: "f32[8, 26, 56, 56]" = convolution_backward_161[0]
    getitem_1486: "f32[26, 26, 3, 3]" = convolution_backward_161[1];  convolution_backward_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_64: "f32[8, 104, 56, 56]" = torch.ops.aten.cat.default([getitem_1485, getitem_1482, getitem_1479, slice_128], 1);  getitem_1485 = getitem_1482 = getitem_1479 = slice_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_159: "f32[8, 104, 56, 56]" = torch.ops.aten.where.self(le_159, full_default, cat_64);  le_159 = cat_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_326: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_159, [0, 2, 3])
    sub_818: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_2626);  convolution_7 = unsqueeze_2626 = None
    mul_2648: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(where_159, sub_818)
    sum_327: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2648, [0, 2, 3]);  mul_2648 = None
    mul_2649: "f32[104]" = torch.ops.aten.mul.Tensor(sum_326, 3.985969387755102e-05)
    unsqueeze_2627: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2649, 0);  mul_2649 = None
    unsqueeze_2628: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2627, 2);  unsqueeze_2627 = None
    unsqueeze_2629: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2628, 3);  unsqueeze_2628 = None
    mul_2650: "f32[104]" = torch.ops.aten.mul.Tensor(sum_327, 3.985969387755102e-05)
    mul_2651: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_2652: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2650, mul_2651);  mul_2650 = mul_2651 = None
    unsqueeze_2630: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2652, 0);  mul_2652 = None
    unsqueeze_2631: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2630, 2);  unsqueeze_2630 = None
    unsqueeze_2632: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2631, 3);  unsqueeze_2631 = None
    mul_2653: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_23);  primals_23 = None
    unsqueeze_2633: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2653, 0);  mul_2653 = None
    unsqueeze_2634: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2633, 2);  unsqueeze_2633 = None
    unsqueeze_2635: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2634, 3);  unsqueeze_2634 = None
    mul_2654: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_818, unsqueeze_2632);  sub_818 = unsqueeze_2632 = None
    sub_820: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(where_159, mul_2654);  where_159 = mul_2654 = None
    sub_821: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(sub_820, unsqueeze_2629);  sub_820 = unsqueeze_2629 = None
    mul_2655: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_821, unsqueeze_2635);  sub_821 = unsqueeze_2635 = None
    mul_2656: "f32[104]" = torch.ops.aten.mul.Tensor(sum_327, squeeze_22);  sum_327 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_162 = torch.ops.aten.convolution_backward.default(mul_2655, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2655 = primals_22 = None
    getitem_1488: "f32[8, 256, 56, 56]" = convolution_backward_162[0]
    getitem_1489: "f32[104, 256, 1, 1]" = convolution_backward_162[1];  convolution_backward_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_1030: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_155, getitem_1488);  where_155 = getitem_1488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    le_160: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_160: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_160, full_default, add_1030);  le_160 = add_1030 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    sum_328: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_160, [0, 2, 3])
    sub_822: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_2638);  convolution_6 = unsqueeze_2638 = None
    mul_2657: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_160, sub_822)
    sum_329: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2657, [0, 2, 3]);  mul_2657 = None
    mul_2658: "f32[256]" = torch.ops.aten.mul.Tensor(sum_328, 3.985969387755102e-05)
    unsqueeze_2639: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2658, 0);  mul_2658 = None
    unsqueeze_2640: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2639, 2);  unsqueeze_2639 = None
    unsqueeze_2641: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2640, 3);  unsqueeze_2640 = None
    mul_2659: "f32[256]" = torch.ops.aten.mul.Tensor(sum_329, 3.985969387755102e-05)
    mul_2660: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_2661: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2659, mul_2660);  mul_2659 = mul_2660 = None
    unsqueeze_2642: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2661, 0);  mul_2661 = None
    unsqueeze_2643: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2642, 2);  unsqueeze_2642 = None
    unsqueeze_2644: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2643, 3);  unsqueeze_2643 = None
    mul_2662: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_20);  primals_20 = None
    unsqueeze_2645: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2662, 0);  mul_2662 = None
    unsqueeze_2646: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2645, 2);  unsqueeze_2645 = None
    unsqueeze_2647: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2646, 3);  unsqueeze_2646 = None
    mul_2663: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_822, unsqueeze_2644);  sub_822 = unsqueeze_2644 = None
    sub_824: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_160, mul_2663);  mul_2663 = None
    sub_825: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_824, unsqueeze_2641);  sub_824 = None
    mul_2664: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_825, unsqueeze_2647);  sub_825 = unsqueeze_2647 = None
    mul_2665: "f32[256]" = torch.ops.aten.mul.Tensor(sum_329, squeeze_19);  sum_329 = squeeze_19 = None
    convolution_backward_163 = torch.ops.aten.convolution_backward.default(mul_2664, getitem_2, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2664 = primals_19 = None
    getitem_1491: "f32[8, 64, 56, 56]" = convolution_backward_163[0]
    getitem_1492: "f32[256, 64, 1, 1]" = convolution_backward_163[1];  convolution_backward_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sub_826: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_2650);  convolution_5 = unsqueeze_2650 = None
    mul_2666: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_160, sub_826)
    sum_331: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2666, [0, 2, 3]);  mul_2666 = None
    mul_2668: "f32[256]" = torch.ops.aten.mul.Tensor(sum_331, 3.985969387755102e-05)
    mul_2669: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_2670: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2668, mul_2669);  mul_2668 = mul_2669 = None
    unsqueeze_2654: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2670, 0);  mul_2670 = None
    unsqueeze_2655: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2654, 2);  unsqueeze_2654 = None
    unsqueeze_2656: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2655, 3);  unsqueeze_2655 = None
    mul_2671: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_17);  primals_17 = None
    unsqueeze_2657: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2671, 0);  mul_2671 = None
    unsqueeze_2658: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2657, 2);  unsqueeze_2657 = None
    unsqueeze_2659: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2658, 3);  unsqueeze_2658 = None
    mul_2672: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_826, unsqueeze_2656);  sub_826 = unsqueeze_2656 = None
    sub_828: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_160, mul_2672);  where_160 = mul_2672 = None
    sub_829: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_828, unsqueeze_2641);  sub_828 = unsqueeze_2641 = None
    mul_2673: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_829, unsqueeze_2659);  sub_829 = unsqueeze_2659 = None
    mul_2674: "f32[256]" = torch.ops.aten.mul.Tensor(sum_331, squeeze_16);  sum_331 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_164 = torch.ops.aten.convolution_backward.default(mul_2673, cat, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2673 = cat = primals_16 = None
    getitem_1494: "f32[8, 104, 56, 56]" = convolution_backward_164[0]
    getitem_1495: "f32[256, 104, 1, 1]" = convolution_backward_164[1];  convolution_backward_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_129: "f32[8, 26, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1494, 1, 0, 26)
    slice_130: "f32[8, 26, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1494, 1, 26, 52)
    slice_131: "f32[8, 26, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1494, 1, 52, 78)
    slice_132: "f32[8, 26, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1494, 1, 78, 104);  getitem_1494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_backward_3: "f32[8, 26, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(slice_132, getitem_31, [3, 3], [1, 1], [1, 1], False, True, None);  slice_132 = getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_161: "f32[8, 26, 56, 56]" = torch.ops.aten.where.self(le_161, full_default, slice_131);  le_161 = slice_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_332: "f32[26]" = torch.ops.aten.sum.dim_IntList(where_161, [0, 2, 3])
    sub_830: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_2662);  convolution_4 = unsqueeze_2662 = None
    mul_2675: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(where_161, sub_830)
    sum_333: "f32[26]" = torch.ops.aten.sum.dim_IntList(mul_2675, [0, 2, 3]);  mul_2675 = None
    mul_2676: "f32[26]" = torch.ops.aten.mul.Tensor(sum_332, 3.985969387755102e-05)
    unsqueeze_2663: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2676, 0);  mul_2676 = None
    unsqueeze_2664: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2663, 2);  unsqueeze_2663 = None
    unsqueeze_2665: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2664, 3);  unsqueeze_2664 = None
    mul_2677: "f32[26]" = torch.ops.aten.mul.Tensor(sum_333, 3.985969387755102e-05)
    mul_2678: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_2679: "f32[26]" = torch.ops.aten.mul.Tensor(mul_2677, mul_2678);  mul_2677 = mul_2678 = None
    unsqueeze_2666: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2679, 0);  mul_2679 = None
    unsqueeze_2667: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2666, 2);  unsqueeze_2666 = None
    unsqueeze_2668: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2667, 3);  unsqueeze_2667 = None
    mul_2680: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
    unsqueeze_2669: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2680, 0);  mul_2680 = None
    unsqueeze_2670: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2669, 2);  unsqueeze_2669 = None
    unsqueeze_2671: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2670, 3);  unsqueeze_2670 = None
    mul_2681: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_830, unsqueeze_2668);  sub_830 = unsqueeze_2668 = None
    sub_832: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(where_161, mul_2681);  where_161 = mul_2681 = None
    sub_833: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(sub_832, unsqueeze_2665);  sub_832 = unsqueeze_2665 = None
    mul_2682: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_833, unsqueeze_2671);  sub_833 = unsqueeze_2671 = None
    mul_2683: "f32[26]" = torch.ops.aten.mul.Tensor(sum_333, squeeze_13);  sum_333 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_165 = torch.ops.aten.convolution_backward.default(mul_2682, getitem_24, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2682 = getitem_24 = primals_13 = None
    getitem_1497: "f32[8, 26, 56, 56]" = convolution_backward_165[0]
    getitem_1498: "f32[26, 26, 3, 3]" = convolution_backward_165[1];  convolution_backward_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_162: "f32[8, 26, 56, 56]" = torch.ops.aten.where.self(le_162, full_default, slice_130);  le_162 = slice_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_334: "f32[26]" = torch.ops.aten.sum.dim_IntList(where_162, [0, 2, 3])
    sub_834: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_2674);  convolution_3 = unsqueeze_2674 = None
    mul_2684: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(where_162, sub_834)
    sum_335: "f32[26]" = torch.ops.aten.sum.dim_IntList(mul_2684, [0, 2, 3]);  mul_2684 = None
    mul_2685: "f32[26]" = torch.ops.aten.mul.Tensor(sum_334, 3.985969387755102e-05)
    unsqueeze_2675: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2685, 0);  mul_2685 = None
    unsqueeze_2676: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2675, 2);  unsqueeze_2675 = None
    unsqueeze_2677: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2676, 3);  unsqueeze_2676 = None
    mul_2686: "f32[26]" = torch.ops.aten.mul.Tensor(sum_335, 3.985969387755102e-05)
    mul_2687: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_2688: "f32[26]" = torch.ops.aten.mul.Tensor(mul_2686, mul_2687);  mul_2686 = mul_2687 = None
    unsqueeze_2678: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2688, 0);  mul_2688 = None
    unsqueeze_2679: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2678, 2);  unsqueeze_2678 = None
    unsqueeze_2680: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2679, 3);  unsqueeze_2679 = None
    mul_2689: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
    unsqueeze_2681: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2689, 0);  mul_2689 = None
    unsqueeze_2682: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2681, 2);  unsqueeze_2681 = None
    unsqueeze_2683: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2682, 3);  unsqueeze_2682 = None
    mul_2690: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_834, unsqueeze_2680);  sub_834 = unsqueeze_2680 = None
    sub_836: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(where_162, mul_2690);  where_162 = mul_2690 = None
    sub_837: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(sub_836, unsqueeze_2677);  sub_836 = unsqueeze_2677 = None
    mul_2691: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_837, unsqueeze_2683);  sub_837 = unsqueeze_2683 = None
    mul_2692: "f32[26]" = torch.ops.aten.mul.Tensor(sum_335, squeeze_10);  sum_335 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_166 = torch.ops.aten.convolution_backward.default(mul_2691, getitem_17, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2691 = getitem_17 = primals_10 = None
    getitem_1500: "f32[8, 26, 56, 56]" = convolution_backward_166[0]
    getitem_1501: "f32[26, 26, 3, 3]" = convolution_backward_166[1];  convolution_backward_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_163: "f32[8, 26, 56, 56]" = torch.ops.aten.where.self(le_163, full_default, slice_129);  le_163 = slice_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_336: "f32[26]" = torch.ops.aten.sum.dim_IntList(where_163, [0, 2, 3])
    sub_838: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_2686);  convolution_2 = unsqueeze_2686 = None
    mul_2693: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(where_163, sub_838)
    sum_337: "f32[26]" = torch.ops.aten.sum.dim_IntList(mul_2693, [0, 2, 3]);  mul_2693 = None
    mul_2694: "f32[26]" = torch.ops.aten.mul.Tensor(sum_336, 3.985969387755102e-05)
    unsqueeze_2687: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2694, 0);  mul_2694 = None
    unsqueeze_2688: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2687, 2);  unsqueeze_2687 = None
    unsqueeze_2689: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2688, 3);  unsqueeze_2688 = None
    mul_2695: "f32[26]" = torch.ops.aten.mul.Tensor(sum_337, 3.985969387755102e-05)
    mul_2696: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_2697: "f32[26]" = torch.ops.aten.mul.Tensor(mul_2695, mul_2696);  mul_2695 = mul_2696 = None
    unsqueeze_2690: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2697, 0);  mul_2697 = None
    unsqueeze_2691: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2690, 2);  unsqueeze_2690 = None
    unsqueeze_2692: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2691, 3);  unsqueeze_2691 = None
    mul_2698: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
    unsqueeze_2693: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(mul_2698, 0);  mul_2698 = None
    unsqueeze_2694: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2693, 2);  unsqueeze_2693 = None
    unsqueeze_2695: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2694, 3);  unsqueeze_2694 = None
    mul_2699: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_838, unsqueeze_2692);  sub_838 = unsqueeze_2692 = None
    sub_840: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(where_163, mul_2699);  where_163 = mul_2699 = None
    sub_841: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(sub_840, unsqueeze_2689);  sub_840 = unsqueeze_2689 = None
    mul_2700: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_841, unsqueeze_2695);  sub_841 = unsqueeze_2695 = None
    mul_2701: "f32[26]" = torch.ops.aten.mul.Tensor(sum_337, squeeze_7);  sum_337 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_167 = torch.ops.aten.convolution_backward.default(mul_2700, getitem_10, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2700 = getitem_10 = primals_7 = None
    getitem_1503: "f32[8, 26, 56, 56]" = convolution_backward_167[0]
    getitem_1504: "f32[26, 26, 3, 3]" = convolution_backward_167[1];  convolution_backward_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_65: "f32[8, 104, 56, 56]" = torch.ops.aten.cat.default([getitem_1503, getitem_1500, getitem_1497, avg_pool2d_backward_3], 1);  getitem_1503 = getitem_1500 = getitem_1497 = avg_pool2d_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_164: "f32[8, 104, 56, 56]" = torch.ops.aten.where.self(le_164, full_default, cat_65);  le_164 = cat_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_338: "f32[104]" = torch.ops.aten.sum.dim_IntList(where_164, [0, 2, 3])
    sub_842: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_2698);  convolution_1 = unsqueeze_2698 = None
    mul_2702: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(where_164, sub_842)
    sum_339: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_2702, [0, 2, 3]);  mul_2702 = None
    mul_2703: "f32[104]" = torch.ops.aten.mul.Tensor(sum_338, 3.985969387755102e-05)
    unsqueeze_2699: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2703, 0);  mul_2703 = None
    unsqueeze_2700: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2699, 2);  unsqueeze_2699 = None
    unsqueeze_2701: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2700, 3);  unsqueeze_2700 = None
    mul_2704: "f32[104]" = torch.ops.aten.mul.Tensor(sum_339, 3.985969387755102e-05)
    mul_2705: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_2706: "f32[104]" = torch.ops.aten.mul.Tensor(mul_2704, mul_2705);  mul_2704 = mul_2705 = None
    unsqueeze_2702: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2706, 0);  mul_2706 = None
    unsqueeze_2703: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2702, 2);  unsqueeze_2702 = None
    unsqueeze_2704: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2703, 3);  unsqueeze_2703 = None
    mul_2707: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
    unsqueeze_2705: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_2707, 0);  mul_2707 = None
    unsqueeze_2706: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2705, 2);  unsqueeze_2705 = None
    unsqueeze_2707: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2706, 3);  unsqueeze_2706 = None
    mul_2708: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_842, unsqueeze_2704);  sub_842 = unsqueeze_2704 = None
    sub_844: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(where_164, mul_2708);  where_164 = mul_2708 = None
    sub_845: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(sub_844, unsqueeze_2701);  sub_844 = unsqueeze_2701 = None
    mul_2709: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_845, unsqueeze_2707);  sub_845 = unsqueeze_2707 = None
    mul_2710: "f32[104]" = torch.ops.aten.mul.Tensor(sum_339, squeeze_4);  sum_339 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_168 = torch.ops.aten.convolution_backward.default(mul_2709, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2709 = getitem_2 = primals_4 = None
    getitem_1506: "f32[8, 64, 56, 56]" = convolution_backward_168[0]
    getitem_1507: "f32[104, 64, 1, 1]" = convolution_backward_168[1];  convolution_backward_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_1031: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_1491, getitem_1506);  getitem_1491 = getitem_1506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    max_pool2d_with_indices_backward: "f32[8, 64, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_1031, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_3);  add_1031 = getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    le_165: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_165: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_165, full_default, max_pool2d_with_indices_backward);  le_165 = full_default = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    sum_340: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_165, [0, 2, 3])
    sub_846: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_2710);  convolution = unsqueeze_2710 = None
    mul_2711: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_165, sub_846)
    sum_341: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_2711, [0, 2, 3]);  mul_2711 = None
    mul_2712: "f32[64]" = torch.ops.aten.mul.Tensor(sum_340, 9.964923469387754e-06)
    unsqueeze_2711: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2712, 0);  mul_2712 = None
    unsqueeze_2712: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2711, 2);  unsqueeze_2711 = None
    unsqueeze_2713: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2712, 3);  unsqueeze_2712 = None
    mul_2713: "f32[64]" = torch.ops.aten.mul.Tensor(sum_341, 9.964923469387754e-06)
    mul_2714: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_2715: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2713, mul_2714);  mul_2713 = mul_2714 = None
    unsqueeze_2714: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2715, 0);  mul_2715 = None
    unsqueeze_2715: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2714, 2);  unsqueeze_2714 = None
    unsqueeze_2716: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2715, 3);  unsqueeze_2715 = None
    mul_2716: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
    unsqueeze_2717: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2716, 0);  mul_2716 = None
    unsqueeze_2718: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2717, 2);  unsqueeze_2717 = None
    unsqueeze_2719: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2718, 3);  unsqueeze_2718 = None
    mul_2717: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_846, unsqueeze_2716);  sub_846 = unsqueeze_2716 = None
    sub_848: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_165, mul_2717);  where_165 = mul_2717 = None
    sub_849: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_848, unsqueeze_2713);  sub_848 = unsqueeze_2713 = None
    mul_2718: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_849, unsqueeze_2719);  sub_849 = unsqueeze_2719 = None
    mul_2719: "f32[64]" = torch.ops.aten.mul.Tensor(sum_341, squeeze_1);  sum_341 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution_backward_169 = torch.ops.aten.convolution_backward.default(mul_2718, primals_1023, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_2718 = primals_1023 = primals_1 = None
    getitem_1510: "f32[64, 3, 7, 7]" = convolution_backward_169[1];  convolution_backward_169 = None
    return [getitem_1510, mul_2719, sum_340, getitem_1507, mul_2710, sum_338, getitem_1504, mul_2701, sum_336, getitem_1501, mul_2692, sum_334, getitem_1498, mul_2683, sum_332, getitem_1495, mul_2674, sum_328, getitem_1492, mul_2665, sum_328, getitem_1489, mul_2656, sum_326, getitem_1486, mul_2647, sum_324, getitem_1483, mul_2638, sum_322, getitem_1480, mul_2629, sum_320, getitem_1477, mul_2620, sum_318, getitem_1474, mul_2611, sum_316, getitem_1471, mul_2602, sum_314, getitem_1468, mul_2593, sum_312, getitem_1465, mul_2584, sum_310, getitem_1462, mul_2575, sum_308, getitem_1459, mul_2566, sum_306, getitem_1456, mul_2557, sum_304, getitem_1453, mul_2548, sum_302, getitem_1450, mul_2539, sum_300, getitem_1447, mul_2530, sum_296, getitem_1444, mul_2521, sum_296, getitem_1441, mul_2512, sum_294, getitem_1438, mul_2503, sum_292, getitem_1435, mul_2494, sum_290, getitem_1432, mul_2485, sum_288, getitem_1429, mul_2476, sum_286, getitem_1426, mul_2467, sum_284, getitem_1423, mul_2458, sum_282, getitem_1420, mul_2449, sum_280, getitem_1417, mul_2440, sum_278, getitem_1414, mul_2431, sum_276, getitem_1411, mul_2422, sum_274, getitem_1408, mul_2413, sum_272, getitem_1405, mul_2404, sum_270, getitem_1402, mul_2395, sum_268, getitem_1399, mul_2386, sum_266, getitem_1396, mul_2377, sum_264, getitem_1393, mul_2368, sum_262, getitem_1390, mul_2359, sum_260, getitem_1387, mul_2350, sum_258, getitem_1384, mul_2341, sum_254, getitem_1381, mul_2332, sum_254, getitem_1378, mul_2323, sum_252, getitem_1375, mul_2314, sum_250, getitem_1372, mul_2305, sum_248, getitem_1369, mul_2296, sum_246, getitem_1366, mul_2287, sum_244, getitem_1363, mul_2278, sum_242, getitem_1360, mul_2269, sum_240, getitem_1357, mul_2260, sum_238, getitem_1354, mul_2251, sum_236, getitem_1351, mul_2242, sum_234, getitem_1348, mul_2233, sum_232, getitem_1345, mul_2224, sum_230, getitem_1342, mul_2215, sum_228, getitem_1339, mul_2206, sum_226, getitem_1336, mul_2197, sum_224, getitem_1333, mul_2188, sum_222, getitem_1330, mul_2179, sum_220, getitem_1327, mul_2170, sum_218, getitem_1324, mul_2161, sum_216, getitem_1321, mul_2152, sum_214, getitem_1318, mul_2143, sum_212, getitem_1315, mul_2134, sum_210, getitem_1312, mul_2125, sum_208, getitem_1309, mul_2116, sum_206, getitem_1306, mul_2107, sum_204, getitem_1303, mul_2098, sum_202, getitem_1300, mul_2089, sum_200, getitem_1297, mul_2080, sum_198, getitem_1294, mul_2071, sum_196, getitem_1291, mul_2062, sum_194, getitem_1288, mul_2053, sum_192, getitem_1285, mul_2044, sum_190, getitem_1282, mul_2035, sum_188, getitem_1279, mul_2026, sum_186, getitem_1276, mul_2017, sum_184, getitem_1273, mul_2008, sum_182, getitem_1270, mul_1999, sum_180, getitem_1267, mul_1990, sum_178, getitem_1264, mul_1981, sum_176, getitem_1261, mul_1972, sum_174, getitem_1258, mul_1963, sum_172, getitem_1255, mul_1954, sum_170, getitem_1252, mul_1945, sum_168, getitem_1249, mul_1936, sum_166, getitem_1246, mul_1927, sum_164, getitem_1243, mul_1918, sum_162, getitem_1240, mul_1909, sum_160, getitem_1237, mul_1900, sum_158, getitem_1234, mul_1891, sum_156, getitem_1231, mul_1882, sum_154, getitem_1228, mul_1873, sum_152, getitem_1225, mul_1864, sum_150, getitem_1222, mul_1855, sum_148, getitem_1219, mul_1846, sum_146, getitem_1216, mul_1837, sum_144, getitem_1213, mul_1828, sum_142, getitem_1210, mul_1819, sum_140, getitem_1207, mul_1810, sum_138, getitem_1204, mul_1801, sum_136, getitem_1201, mul_1792, sum_134, getitem_1198, mul_1783, sum_132, getitem_1195, mul_1774, sum_130, getitem_1192, mul_1765, sum_128, getitem_1189, mul_1756, sum_126, getitem_1186, mul_1747, sum_124, getitem_1183, mul_1738, sum_122, getitem_1180, mul_1729, sum_120, getitem_1177, mul_1720, sum_118, getitem_1174, mul_1711, sum_116, getitem_1171, mul_1702, sum_114, getitem_1168, mul_1693, sum_112, getitem_1165, mul_1684, sum_110, getitem_1162, mul_1675, sum_108, getitem_1159, mul_1666, sum_106, getitem_1156, mul_1657, sum_104, getitem_1153, mul_1648, sum_102, getitem_1150, mul_1639, sum_100, getitem_1147, mul_1630, sum_98, getitem_1144, mul_1621, sum_96, getitem_1141, mul_1612, sum_94, getitem_1138, mul_1603, sum_92, getitem_1135, mul_1594, sum_90, getitem_1132, mul_1585, sum_88, getitem_1129, mul_1576, sum_86, getitem_1126, mul_1567, sum_84, getitem_1123, mul_1558, sum_82, getitem_1120, mul_1549, sum_80, getitem_1117, mul_1540, sum_78, getitem_1114, mul_1531, sum_76, getitem_1111, mul_1522, sum_74, getitem_1108, mul_1513, sum_72, getitem_1105, mul_1504, sum_70, getitem_1102, mul_1495, sum_68, getitem_1099, mul_1486, sum_66, getitem_1096, mul_1477, sum_64, getitem_1093, mul_1468, sum_62, getitem_1090, mul_1459, sum_60, getitem_1087, mul_1450, sum_58, getitem_1084, mul_1441, sum_56, getitem_1081, mul_1432, sum_54, getitem_1078, mul_1423, sum_52, getitem_1075, mul_1414, sum_50, getitem_1072, mul_1405, sum_48, getitem_1069, mul_1396, sum_46, getitem_1066, mul_1387, sum_44, getitem_1063, mul_1378, sum_42, getitem_1060, mul_1369, sum_40, getitem_1057, mul_1360, sum_38, getitem_1054, mul_1351, sum_36, getitem_1051, mul_1342, sum_34, getitem_1048, mul_1333, sum_32, getitem_1045, mul_1324, sum_30, getitem_1042, mul_1315, sum_28, getitem_1039, mul_1306, sum_26, getitem_1036, mul_1297, sum_22, getitem_1033, mul_1288, sum_22, getitem_1030, mul_1279, sum_20, getitem_1027, mul_1270, sum_18, getitem_1024, mul_1261, sum_16, getitem_1021, mul_1252, sum_14, getitem_1018, mul_1243, sum_12, getitem_1015, mul_1234, sum_10, getitem_1012, mul_1225, sum_8, getitem_1009, mul_1216, sum_6, getitem_1006, mul_1207, sum_4, getitem_1003, mul_1198, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    