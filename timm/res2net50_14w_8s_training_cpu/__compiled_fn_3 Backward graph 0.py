from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[64]", primals_4: "f32[112, 64, 1, 1]", primals_5: "f32[112]", primals_7: "f32[14, 14, 3, 3]", primals_8: "f32[14]", primals_10: "f32[14, 14, 3, 3]", primals_11: "f32[14]", primals_13: "f32[14, 14, 3, 3]", primals_14: "f32[14]", primals_16: "f32[14, 14, 3, 3]", primals_17: "f32[14]", primals_19: "f32[14, 14, 3, 3]", primals_20: "f32[14]", primals_22: "f32[14, 14, 3, 3]", primals_23: "f32[14]", primals_25: "f32[14, 14, 3, 3]", primals_26: "f32[14]", primals_28: "f32[256, 112, 1, 1]", primals_29: "f32[256]", primals_31: "f32[256, 64, 1, 1]", primals_32: "f32[256]", primals_34: "f32[112, 256, 1, 1]", primals_35: "f32[112]", primals_37: "f32[14, 14, 3, 3]", primals_38: "f32[14]", primals_40: "f32[14, 14, 3, 3]", primals_41: "f32[14]", primals_43: "f32[14, 14, 3, 3]", primals_44: "f32[14]", primals_46: "f32[14, 14, 3, 3]", primals_47: "f32[14]", primals_49: "f32[14, 14, 3, 3]", primals_50: "f32[14]", primals_52: "f32[14, 14, 3, 3]", primals_53: "f32[14]", primals_55: "f32[14, 14, 3, 3]", primals_56: "f32[14]", primals_58: "f32[256, 112, 1, 1]", primals_59: "f32[256]", primals_61: "f32[112, 256, 1, 1]", primals_62: "f32[112]", primals_64: "f32[14, 14, 3, 3]", primals_65: "f32[14]", primals_67: "f32[14, 14, 3, 3]", primals_68: "f32[14]", primals_70: "f32[14, 14, 3, 3]", primals_71: "f32[14]", primals_73: "f32[14, 14, 3, 3]", primals_74: "f32[14]", primals_76: "f32[14, 14, 3, 3]", primals_77: "f32[14]", primals_79: "f32[14, 14, 3, 3]", primals_80: "f32[14]", primals_82: "f32[14, 14, 3, 3]", primals_83: "f32[14]", primals_85: "f32[256, 112, 1, 1]", primals_86: "f32[256]", primals_88: "f32[224, 256, 1, 1]", primals_89: "f32[224]", primals_91: "f32[28, 28, 3, 3]", primals_92: "f32[28]", primals_94: "f32[28, 28, 3, 3]", primals_95: "f32[28]", primals_97: "f32[28, 28, 3, 3]", primals_98: "f32[28]", primals_100: "f32[28, 28, 3, 3]", primals_101: "f32[28]", primals_103: "f32[28, 28, 3, 3]", primals_104: "f32[28]", primals_106: "f32[28, 28, 3, 3]", primals_107: "f32[28]", primals_109: "f32[28, 28, 3, 3]", primals_110: "f32[28]", primals_112: "f32[512, 224, 1, 1]", primals_113: "f32[512]", primals_115: "f32[512, 256, 1, 1]", primals_116: "f32[512]", primals_118: "f32[224, 512, 1, 1]", primals_119: "f32[224]", primals_121: "f32[28, 28, 3, 3]", primals_122: "f32[28]", primals_124: "f32[28, 28, 3, 3]", primals_125: "f32[28]", primals_127: "f32[28, 28, 3, 3]", primals_128: "f32[28]", primals_130: "f32[28, 28, 3, 3]", primals_131: "f32[28]", primals_133: "f32[28, 28, 3, 3]", primals_134: "f32[28]", primals_136: "f32[28, 28, 3, 3]", primals_137: "f32[28]", primals_139: "f32[28, 28, 3, 3]", primals_140: "f32[28]", primals_142: "f32[512, 224, 1, 1]", primals_143: "f32[512]", primals_145: "f32[224, 512, 1, 1]", primals_146: "f32[224]", primals_148: "f32[28, 28, 3, 3]", primals_149: "f32[28]", primals_151: "f32[28, 28, 3, 3]", primals_152: "f32[28]", primals_154: "f32[28, 28, 3, 3]", primals_155: "f32[28]", primals_157: "f32[28, 28, 3, 3]", primals_158: "f32[28]", primals_160: "f32[28, 28, 3, 3]", primals_161: "f32[28]", primals_163: "f32[28, 28, 3, 3]", primals_164: "f32[28]", primals_166: "f32[28, 28, 3, 3]", primals_167: "f32[28]", primals_169: "f32[512, 224, 1, 1]", primals_170: "f32[512]", primals_172: "f32[224, 512, 1, 1]", primals_173: "f32[224]", primals_175: "f32[28, 28, 3, 3]", primals_176: "f32[28]", primals_178: "f32[28, 28, 3, 3]", primals_179: "f32[28]", primals_181: "f32[28, 28, 3, 3]", primals_182: "f32[28]", primals_184: "f32[28, 28, 3, 3]", primals_185: "f32[28]", primals_187: "f32[28, 28, 3, 3]", primals_188: "f32[28]", primals_190: "f32[28, 28, 3, 3]", primals_191: "f32[28]", primals_193: "f32[28, 28, 3, 3]", primals_194: "f32[28]", primals_196: "f32[512, 224, 1, 1]", primals_197: "f32[512]", primals_199: "f32[448, 512, 1, 1]", primals_200: "f32[448]", primals_202: "f32[56, 56, 3, 3]", primals_203: "f32[56]", primals_205: "f32[56, 56, 3, 3]", primals_206: "f32[56]", primals_208: "f32[56, 56, 3, 3]", primals_209: "f32[56]", primals_211: "f32[56, 56, 3, 3]", primals_212: "f32[56]", primals_214: "f32[56, 56, 3, 3]", primals_215: "f32[56]", primals_217: "f32[56, 56, 3, 3]", primals_218: "f32[56]", primals_220: "f32[56, 56, 3, 3]", primals_221: "f32[56]", primals_223: "f32[1024, 448, 1, 1]", primals_224: "f32[1024]", primals_226: "f32[1024, 512, 1, 1]", primals_227: "f32[1024]", primals_229: "f32[448, 1024, 1, 1]", primals_230: "f32[448]", primals_232: "f32[56, 56, 3, 3]", primals_233: "f32[56]", primals_235: "f32[56, 56, 3, 3]", primals_236: "f32[56]", primals_238: "f32[56, 56, 3, 3]", primals_239: "f32[56]", primals_241: "f32[56, 56, 3, 3]", primals_242: "f32[56]", primals_244: "f32[56, 56, 3, 3]", primals_245: "f32[56]", primals_247: "f32[56, 56, 3, 3]", primals_248: "f32[56]", primals_250: "f32[56, 56, 3, 3]", primals_251: "f32[56]", primals_253: "f32[1024, 448, 1, 1]", primals_254: "f32[1024]", primals_256: "f32[448, 1024, 1, 1]", primals_257: "f32[448]", primals_259: "f32[56, 56, 3, 3]", primals_260: "f32[56]", primals_262: "f32[56, 56, 3, 3]", primals_263: "f32[56]", primals_265: "f32[56, 56, 3, 3]", primals_266: "f32[56]", primals_268: "f32[56, 56, 3, 3]", primals_269: "f32[56]", primals_271: "f32[56, 56, 3, 3]", primals_272: "f32[56]", primals_274: "f32[56, 56, 3, 3]", primals_275: "f32[56]", primals_277: "f32[56, 56, 3, 3]", primals_278: "f32[56]", primals_280: "f32[1024, 448, 1, 1]", primals_281: "f32[1024]", primals_283: "f32[448, 1024, 1, 1]", primals_284: "f32[448]", primals_286: "f32[56, 56, 3, 3]", primals_287: "f32[56]", primals_289: "f32[56, 56, 3, 3]", primals_290: "f32[56]", primals_292: "f32[56, 56, 3, 3]", primals_293: "f32[56]", primals_295: "f32[56, 56, 3, 3]", primals_296: "f32[56]", primals_298: "f32[56, 56, 3, 3]", primals_299: "f32[56]", primals_301: "f32[56, 56, 3, 3]", primals_302: "f32[56]", primals_304: "f32[56, 56, 3, 3]", primals_305: "f32[56]", primals_307: "f32[1024, 448, 1, 1]", primals_308: "f32[1024]", primals_310: "f32[448, 1024, 1, 1]", primals_311: "f32[448]", primals_313: "f32[56, 56, 3, 3]", primals_314: "f32[56]", primals_316: "f32[56, 56, 3, 3]", primals_317: "f32[56]", primals_319: "f32[56, 56, 3, 3]", primals_320: "f32[56]", primals_322: "f32[56, 56, 3, 3]", primals_323: "f32[56]", primals_325: "f32[56, 56, 3, 3]", primals_326: "f32[56]", primals_328: "f32[56, 56, 3, 3]", primals_329: "f32[56]", primals_331: "f32[56, 56, 3, 3]", primals_332: "f32[56]", primals_334: "f32[1024, 448, 1, 1]", primals_335: "f32[1024]", primals_337: "f32[448, 1024, 1, 1]", primals_338: "f32[448]", primals_340: "f32[56, 56, 3, 3]", primals_341: "f32[56]", primals_343: "f32[56, 56, 3, 3]", primals_344: "f32[56]", primals_346: "f32[56, 56, 3, 3]", primals_347: "f32[56]", primals_349: "f32[56, 56, 3, 3]", primals_350: "f32[56]", primals_352: "f32[56, 56, 3, 3]", primals_353: "f32[56]", primals_355: "f32[56, 56, 3, 3]", primals_356: "f32[56]", primals_358: "f32[56, 56, 3, 3]", primals_359: "f32[56]", primals_361: "f32[1024, 448, 1, 1]", primals_362: "f32[1024]", primals_364: "f32[896, 1024, 1, 1]", primals_365: "f32[896]", primals_367: "f32[112, 112, 3, 3]", primals_368: "f32[112]", primals_370: "f32[112, 112, 3, 3]", primals_371: "f32[112]", primals_373: "f32[112, 112, 3, 3]", primals_374: "f32[112]", primals_376: "f32[112, 112, 3, 3]", primals_377: "f32[112]", primals_379: "f32[112, 112, 3, 3]", primals_380: "f32[112]", primals_382: "f32[112, 112, 3, 3]", primals_383: "f32[112]", primals_385: "f32[112, 112, 3, 3]", primals_386: "f32[112]", primals_388: "f32[2048, 896, 1, 1]", primals_389: "f32[2048]", primals_391: "f32[2048, 1024, 1, 1]", primals_392: "f32[2048]", primals_394: "f32[896, 2048, 1, 1]", primals_395: "f32[896]", primals_397: "f32[112, 112, 3, 3]", primals_398: "f32[112]", primals_400: "f32[112, 112, 3, 3]", primals_401: "f32[112]", primals_403: "f32[112, 112, 3, 3]", primals_404: "f32[112]", primals_406: "f32[112, 112, 3, 3]", primals_407: "f32[112]", primals_409: "f32[112, 112, 3, 3]", primals_410: "f32[112]", primals_412: "f32[112, 112, 3, 3]", primals_413: "f32[112]", primals_415: "f32[112, 112, 3, 3]", primals_416: "f32[112]", primals_418: "f32[2048, 896, 1, 1]", primals_419: "f32[2048]", primals_421: "f32[896, 2048, 1, 1]", primals_422: "f32[896]", primals_424: "f32[112, 112, 3, 3]", primals_425: "f32[112]", primals_427: "f32[112, 112, 3, 3]", primals_428: "f32[112]", primals_430: "f32[112, 112, 3, 3]", primals_431: "f32[112]", primals_433: "f32[112, 112, 3, 3]", primals_434: "f32[112]", primals_436: "f32[112, 112, 3, 3]", primals_437: "f32[112]", primals_439: "f32[112, 112, 3, 3]", primals_440: "f32[112]", primals_442: "f32[112, 112, 3, 3]", primals_443: "f32[112]", primals_445: "f32[2048, 896, 1, 1]", primals_446: "f32[2048]", primals_897: "f32[8, 3, 224, 224]", convolution: "f32[8, 64, 112, 112]", squeeze_1: "f32[64]", relu: "f32[8, 64, 112, 112]", getitem_2: "f32[8, 64, 56, 56]", getitem_3: "i64[8, 64, 56, 56]", convolution_1: "f32[8, 112, 56, 56]", squeeze_4: "f32[112]", getitem_14: "f32[8, 14, 56, 56]", convolution_2: "f32[8, 14, 56, 56]", squeeze_7: "f32[14]", getitem_25: "f32[8, 14, 56, 56]", convolution_3: "f32[8, 14, 56, 56]", squeeze_10: "f32[14]", getitem_36: "f32[8, 14, 56, 56]", convolution_4: "f32[8, 14, 56, 56]", squeeze_13: "f32[14]", getitem_47: "f32[8, 14, 56, 56]", convolution_5: "f32[8, 14, 56, 56]", squeeze_16: "f32[14]", getitem_58: "f32[8, 14, 56, 56]", convolution_6: "f32[8, 14, 56, 56]", squeeze_19: "f32[14]", getitem_69: "f32[8, 14, 56, 56]", convolution_7: "f32[8, 14, 56, 56]", squeeze_22: "f32[14]", getitem_80: "f32[8, 14, 56, 56]", convolution_8: "f32[8, 14, 56, 56]", squeeze_25: "f32[14]", getitem_91: "f32[8, 14, 56, 56]", cat: "f32[8, 112, 56, 56]", convolution_9: "f32[8, 256, 56, 56]", squeeze_28: "f32[256]", convolution_10: "f32[8, 256, 56, 56]", squeeze_31: "f32[256]", relu_9: "f32[8, 256, 56, 56]", convolution_11: "f32[8, 112, 56, 56]", squeeze_34: "f32[112]", getitem_106: "f32[8, 14, 56, 56]", convolution_12: "f32[8, 14, 56, 56]", squeeze_37: "f32[14]", add_66: "f32[8, 14, 56, 56]", convolution_13: "f32[8, 14, 56, 56]", squeeze_40: "f32[14]", add_72: "f32[8, 14, 56, 56]", convolution_14: "f32[8, 14, 56, 56]", squeeze_43: "f32[14]", add_78: "f32[8, 14, 56, 56]", convolution_15: "f32[8, 14, 56, 56]", squeeze_46: "f32[14]", add_84: "f32[8, 14, 56, 56]", convolution_16: "f32[8, 14, 56, 56]", squeeze_49: "f32[14]", add_90: "f32[8, 14, 56, 56]", convolution_17: "f32[8, 14, 56, 56]", squeeze_52: "f32[14]", add_96: "f32[8, 14, 56, 56]", convolution_18: "f32[8, 14, 56, 56]", squeeze_55: "f32[14]", cat_1: "f32[8, 112, 56, 56]", convolution_19: "f32[8, 256, 56, 56]", squeeze_58: "f32[256]", relu_18: "f32[8, 256, 56, 56]", convolution_20: "f32[8, 112, 56, 56]", squeeze_61: "f32[112]", getitem_196: "f32[8, 14, 56, 56]", convolution_21: "f32[8, 14, 56, 56]", squeeze_64: "f32[14]", add_118: "f32[8, 14, 56, 56]", convolution_22: "f32[8, 14, 56, 56]", squeeze_67: "f32[14]", add_124: "f32[8, 14, 56, 56]", convolution_23: "f32[8, 14, 56, 56]", squeeze_70: "f32[14]", add_130: "f32[8, 14, 56, 56]", convolution_24: "f32[8, 14, 56, 56]", squeeze_73: "f32[14]", add_136: "f32[8, 14, 56, 56]", convolution_25: "f32[8, 14, 56, 56]", squeeze_76: "f32[14]", add_142: "f32[8, 14, 56, 56]", convolution_26: "f32[8, 14, 56, 56]", squeeze_79: "f32[14]", add_148: "f32[8, 14, 56, 56]", convolution_27: "f32[8, 14, 56, 56]", squeeze_82: "f32[14]", cat_2: "f32[8, 112, 56, 56]", convolution_28: "f32[8, 256, 56, 56]", squeeze_85: "f32[256]", relu_27: "f32[8, 256, 56, 56]", convolution_29: "f32[8, 224, 56, 56]", squeeze_88: "f32[224]", getitem_286: "f32[8, 28, 56, 56]", convolution_30: "f32[8, 28, 28, 28]", squeeze_91: "f32[28]", getitem_297: "f32[8, 28, 56, 56]", convolution_31: "f32[8, 28, 28, 28]", squeeze_94: "f32[28]", getitem_308: "f32[8, 28, 56, 56]", convolution_32: "f32[8, 28, 28, 28]", squeeze_97: "f32[28]", getitem_319: "f32[8, 28, 56, 56]", convolution_33: "f32[8, 28, 28, 28]", squeeze_100: "f32[28]", getitem_330: "f32[8, 28, 56, 56]", convolution_34: "f32[8, 28, 28, 28]", squeeze_103: "f32[28]", getitem_341: "f32[8, 28, 56, 56]", convolution_35: "f32[8, 28, 28, 28]", squeeze_106: "f32[28]", getitem_352: "f32[8, 28, 56, 56]", convolution_36: "f32[8, 28, 28, 28]", squeeze_109: "f32[28]", getitem_363: "f32[8, 28, 56, 56]", cat_3: "f32[8, 224, 28, 28]", convolution_37: "f32[8, 512, 28, 28]", squeeze_112: "f32[512]", convolution_38: "f32[8, 512, 28, 28]", squeeze_115: "f32[512]", relu_36: "f32[8, 512, 28, 28]", convolution_39: "f32[8, 224, 28, 28]", squeeze_118: "f32[224]", getitem_378: "f32[8, 28, 28, 28]", convolution_40: "f32[8, 28, 28, 28]", squeeze_121: "f32[28]", add_221: "f32[8, 28, 28, 28]", convolution_41: "f32[8, 28, 28, 28]", squeeze_124: "f32[28]", add_227: "f32[8, 28, 28, 28]", convolution_42: "f32[8, 28, 28, 28]", squeeze_127: "f32[28]", add_233: "f32[8, 28, 28, 28]", convolution_43: "f32[8, 28, 28, 28]", squeeze_130: "f32[28]", add_239: "f32[8, 28, 28, 28]", convolution_44: "f32[8, 28, 28, 28]", squeeze_133: "f32[28]", add_245: "f32[8, 28, 28, 28]", convolution_45: "f32[8, 28, 28, 28]", squeeze_136: "f32[28]", add_251: "f32[8, 28, 28, 28]", convolution_46: "f32[8, 28, 28, 28]", squeeze_139: "f32[28]", cat_4: "f32[8, 224, 28, 28]", convolution_47: "f32[8, 512, 28, 28]", squeeze_142: "f32[512]", relu_45: "f32[8, 512, 28, 28]", convolution_48: "f32[8, 224, 28, 28]", squeeze_145: "f32[224]", getitem_468: "f32[8, 28, 28, 28]", convolution_49: "f32[8, 28, 28, 28]", squeeze_148: "f32[28]", add_273: "f32[8, 28, 28, 28]", convolution_50: "f32[8, 28, 28, 28]", squeeze_151: "f32[28]", add_279: "f32[8, 28, 28, 28]", convolution_51: "f32[8, 28, 28, 28]", squeeze_154: "f32[28]", add_285: "f32[8, 28, 28, 28]", convolution_52: "f32[8, 28, 28, 28]", squeeze_157: "f32[28]", add_291: "f32[8, 28, 28, 28]", convolution_53: "f32[8, 28, 28, 28]", squeeze_160: "f32[28]", add_297: "f32[8, 28, 28, 28]", convolution_54: "f32[8, 28, 28, 28]", squeeze_163: "f32[28]", add_303: "f32[8, 28, 28, 28]", convolution_55: "f32[8, 28, 28, 28]", squeeze_166: "f32[28]", cat_5: "f32[8, 224, 28, 28]", convolution_56: "f32[8, 512, 28, 28]", squeeze_169: "f32[512]", relu_54: "f32[8, 512, 28, 28]", convolution_57: "f32[8, 224, 28, 28]", squeeze_172: "f32[224]", getitem_558: "f32[8, 28, 28, 28]", convolution_58: "f32[8, 28, 28, 28]", squeeze_175: "f32[28]", add_325: "f32[8, 28, 28, 28]", convolution_59: "f32[8, 28, 28, 28]", squeeze_178: "f32[28]", add_331: "f32[8, 28, 28, 28]", convolution_60: "f32[8, 28, 28, 28]", squeeze_181: "f32[28]", add_337: "f32[8, 28, 28, 28]", convolution_61: "f32[8, 28, 28, 28]", squeeze_184: "f32[28]", add_343: "f32[8, 28, 28, 28]", convolution_62: "f32[8, 28, 28, 28]", squeeze_187: "f32[28]", add_349: "f32[8, 28, 28, 28]", convolution_63: "f32[8, 28, 28, 28]", squeeze_190: "f32[28]", add_355: "f32[8, 28, 28, 28]", convolution_64: "f32[8, 28, 28, 28]", squeeze_193: "f32[28]", cat_6: "f32[8, 224, 28, 28]", convolution_65: "f32[8, 512, 28, 28]", squeeze_196: "f32[512]", relu_63: "f32[8, 512, 28, 28]", convolution_66: "f32[8, 448, 28, 28]", squeeze_199: "f32[448]", getitem_648: "f32[8, 56, 28, 28]", convolution_67: "f32[8, 56, 14, 14]", squeeze_202: "f32[56]", getitem_659: "f32[8, 56, 28, 28]", convolution_68: "f32[8, 56, 14, 14]", squeeze_205: "f32[56]", getitem_670: "f32[8, 56, 28, 28]", convolution_69: "f32[8, 56, 14, 14]", squeeze_208: "f32[56]", getitem_681: "f32[8, 56, 28, 28]", convolution_70: "f32[8, 56, 14, 14]", squeeze_211: "f32[56]", getitem_692: "f32[8, 56, 28, 28]", convolution_71: "f32[8, 56, 14, 14]", squeeze_214: "f32[56]", getitem_703: "f32[8, 56, 28, 28]", convolution_72: "f32[8, 56, 14, 14]", squeeze_217: "f32[56]", getitem_714: "f32[8, 56, 28, 28]", convolution_73: "f32[8, 56, 14, 14]", squeeze_220: "f32[56]", getitem_725: "f32[8, 56, 28, 28]", cat_7: "f32[8, 448, 14, 14]", convolution_74: "f32[8, 1024, 14, 14]", squeeze_223: "f32[1024]", convolution_75: "f32[8, 1024, 14, 14]", squeeze_226: "f32[1024]", relu_72: "f32[8, 1024, 14, 14]", convolution_76: "f32[8, 448, 14, 14]", squeeze_229: "f32[448]", getitem_740: "f32[8, 56, 14, 14]", convolution_77: "f32[8, 56, 14, 14]", squeeze_232: "f32[56]", add_428: "f32[8, 56, 14, 14]", convolution_78: "f32[8, 56, 14, 14]", squeeze_235: "f32[56]", add_434: "f32[8, 56, 14, 14]", convolution_79: "f32[8, 56, 14, 14]", squeeze_238: "f32[56]", add_440: "f32[8, 56, 14, 14]", convolution_80: "f32[8, 56, 14, 14]", squeeze_241: "f32[56]", add_446: "f32[8, 56, 14, 14]", convolution_81: "f32[8, 56, 14, 14]", squeeze_244: "f32[56]", add_452: "f32[8, 56, 14, 14]", convolution_82: "f32[8, 56, 14, 14]", squeeze_247: "f32[56]", add_458: "f32[8, 56, 14, 14]", convolution_83: "f32[8, 56, 14, 14]", squeeze_250: "f32[56]", cat_8: "f32[8, 448, 14, 14]", convolution_84: "f32[8, 1024, 14, 14]", squeeze_253: "f32[1024]", relu_81: "f32[8, 1024, 14, 14]", convolution_85: "f32[8, 448, 14, 14]", squeeze_256: "f32[448]", getitem_830: "f32[8, 56, 14, 14]", convolution_86: "f32[8, 56, 14, 14]", squeeze_259: "f32[56]", add_480: "f32[8, 56, 14, 14]", convolution_87: "f32[8, 56, 14, 14]", squeeze_262: "f32[56]", add_486: "f32[8, 56, 14, 14]", convolution_88: "f32[8, 56, 14, 14]", squeeze_265: "f32[56]", add_492: "f32[8, 56, 14, 14]", convolution_89: "f32[8, 56, 14, 14]", squeeze_268: "f32[56]", add_498: "f32[8, 56, 14, 14]", convolution_90: "f32[8, 56, 14, 14]", squeeze_271: "f32[56]", add_504: "f32[8, 56, 14, 14]", convolution_91: "f32[8, 56, 14, 14]", squeeze_274: "f32[56]", add_510: "f32[8, 56, 14, 14]", convolution_92: "f32[8, 56, 14, 14]", squeeze_277: "f32[56]", cat_9: "f32[8, 448, 14, 14]", convolution_93: "f32[8, 1024, 14, 14]", squeeze_280: "f32[1024]", relu_90: "f32[8, 1024, 14, 14]", convolution_94: "f32[8, 448, 14, 14]", squeeze_283: "f32[448]", getitem_920: "f32[8, 56, 14, 14]", convolution_95: "f32[8, 56, 14, 14]", squeeze_286: "f32[56]", add_532: "f32[8, 56, 14, 14]", convolution_96: "f32[8, 56, 14, 14]", squeeze_289: "f32[56]", add_538: "f32[8, 56, 14, 14]", convolution_97: "f32[8, 56, 14, 14]", squeeze_292: "f32[56]", add_544: "f32[8, 56, 14, 14]", convolution_98: "f32[8, 56, 14, 14]", squeeze_295: "f32[56]", add_550: "f32[8, 56, 14, 14]", convolution_99: "f32[8, 56, 14, 14]", squeeze_298: "f32[56]", add_556: "f32[8, 56, 14, 14]", convolution_100: "f32[8, 56, 14, 14]", squeeze_301: "f32[56]", add_562: "f32[8, 56, 14, 14]", convolution_101: "f32[8, 56, 14, 14]", squeeze_304: "f32[56]", cat_10: "f32[8, 448, 14, 14]", convolution_102: "f32[8, 1024, 14, 14]", squeeze_307: "f32[1024]", relu_99: "f32[8, 1024, 14, 14]", convolution_103: "f32[8, 448, 14, 14]", squeeze_310: "f32[448]", getitem_1010: "f32[8, 56, 14, 14]", convolution_104: "f32[8, 56, 14, 14]", squeeze_313: "f32[56]", add_584: "f32[8, 56, 14, 14]", convolution_105: "f32[8, 56, 14, 14]", squeeze_316: "f32[56]", add_590: "f32[8, 56, 14, 14]", convolution_106: "f32[8, 56, 14, 14]", squeeze_319: "f32[56]", add_596: "f32[8, 56, 14, 14]", convolution_107: "f32[8, 56, 14, 14]", squeeze_322: "f32[56]", add_602: "f32[8, 56, 14, 14]", convolution_108: "f32[8, 56, 14, 14]", squeeze_325: "f32[56]", add_608: "f32[8, 56, 14, 14]", convolution_109: "f32[8, 56, 14, 14]", squeeze_328: "f32[56]", add_614: "f32[8, 56, 14, 14]", convolution_110: "f32[8, 56, 14, 14]", squeeze_331: "f32[56]", cat_11: "f32[8, 448, 14, 14]", convolution_111: "f32[8, 1024, 14, 14]", squeeze_334: "f32[1024]", relu_108: "f32[8, 1024, 14, 14]", convolution_112: "f32[8, 448, 14, 14]", squeeze_337: "f32[448]", getitem_1100: "f32[8, 56, 14, 14]", convolution_113: "f32[8, 56, 14, 14]", squeeze_340: "f32[56]", add_636: "f32[8, 56, 14, 14]", convolution_114: "f32[8, 56, 14, 14]", squeeze_343: "f32[56]", add_642: "f32[8, 56, 14, 14]", convolution_115: "f32[8, 56, 14, 14]", squeeze_346: "f32[56]", add_648: "f32[8, 56, 14, 14]", convolution_116: "f32[8, 56, 14, 14]", squeeze_349: "f32[56]", add_654: "f32[8, 56, 14, 14]", convolution_117: "f32[8, 56, 14, 14]", squeeze_352: "f32[56]", add_660: "f32[8, 56, 14, 14]", convolution_118: "f32[8, 56, 14, 14]", squeeze_355: "f32[56]", add_666: "f32[8, 56, 14, 14]", convolution_119: "f32[8, 56, 14, 14]", squeeze_358: "f32[56]", cat_12: "f32[8, 448, 14, 14]", convolution_120: "f32[8, 1024, 14, 14]", squeeze_361: "f32[1024]", relu_117: "f32[8, 1024, 14, 14]", convolution_121: "f32[8, 896, 14, 14]", squeeze_364: "f32[896]", getitem_1190: "f32[8, 112, 14, 14]", convolution_122: "f32[8, 112, 7, 7]", squeeze_367: "f32[112]", getitem_1201: "f32[8, 112, 14, 14]", convolution_123: "f32[8, 112, 7, 7]", squeeze_370: "f32[112]", getitem_1212: "f32[8, 112, 14, 14]", convolution_124: "f32[8, 112, 7, 7]", squeeze_373: "f32[112]", getitem_1223: "f32[8, 112, 14, 14]", convolution_125: "f32[8, 112, 7, 7]", squeeze_376: "f32[112]", getitem_1234: "f32[8, 112, 14, 14]", convolution_126: "f32[8, 112, 7, 7]", squeeze_379: "f32[112]", getitem_1245: "f32[8, 112, 14, 14]", convolution_127: "f32[8, 112, 7, 7]", squeeze_382: "f32[112]", getitem_1256: "f32[8, 112, 14, 14]", convolution_128: "f32[8, 112, 7, 7]", squeeze_385: "f32[112]", getitem_1267: "f32[8, 112, 14, 14]", cat_13: "f32[8, 896, 7, 7]", convolution_129: "f32[8, 2048, 7, 7]", squeeze_388: "f32[2048]", convolution_130: "f32[8, 2048, 7, 7]", squeeze_391: "f32[2048]", relu_126: "f32[8, 2048, 7, 7]", convolution_131: "f32[8, 896, 7, 7]", squeeze_394: "f32[896]", getitem_1282: "f32[8, 112, 7, 7]", convolution_132: "f32[8, 112, 7, 7]", squeeze_397: "f32[112]", add_739: "f32[8, 112, 7, 7]", convolution_133: "f32[8, 112, 7, 7]", squeeze_400: "f32[112]", add_745: "f32[8, 112, 7, 7]", convolution_134: "f32[8, 112, 7, 7]", squeeze_403: "f32[112]", add_751: "f32[8, 112, 7, 7]", convolution_135: "f32[8, 112, 7, 7]", squeeze_406: "f32[112]", add_757: "f32[8, 112, 7, 7]", convolution_136: "f32[8, 112, 7, 7]", squeeze_409: "f32[112]", add_763: "f32[8, 112, 7, 7]", convolution_137: "f32[8, 112, 7, 7]", squeeze_412: "f32[112]", add_769: "f32[8, 112, 7, 7]", convolution_138: "f32[8, 112, 7, 7]", squeeze_415: "f32[112]", cat_14: "f32[8, 896, 7, 7]", convolution_139: "f32[8, 2048, 7, 7]", squeeze_418: "f32[2048]", relu_135: "f32[8, 2048, 7, 7]", convolution_140: "f32[8, 896, 7, 7]", squeeze_421: "f32[896]", getitem_1372: "f32[8, 112, 7, 7]", convolution_141: "f32[8, 112, 7, 7]", squeeze_424: "f32[112]", add_791: "f32[8, 112, 7, 7]", convolution_142: "f32[8, 112, 7, 7]", squeeze_427: "f32[112]", add_797: "f32[8, 112, 7, 7]", convolution_143: "f32[8, 112, 7, 7]", squeeze_430: "f32[112]", add_803: "f32[8, 112, 7, 7]", convolution_144: "f32[8, 112, 7, 7]", squeeze_433: "f32[112]", add_809: "f32[8, 112, 7, 7]", convolution_145: "f32[8, 112, 7, 7]", squeeze_436: "f32[112]", add_815: "f32[8, 112, 7, 7]", convolution_146: "f32[8, 112, 7, 7]", squeeze_439: "f32[112]", add_821: "f32[8, 112, 7, 7]", convolution_147: "f32[8, 112, 7, 7]", squeeze_442: "f32[112]", cat_15: "f32[8, 896, 7, 7]", convolution_148: "f32[8, 2048, 7, 7]", squeeze_445: "f32[2048]", view: "f32[8, 2048]", permute_1: "f32[1000, 2048]", le: "b8[8, 2048, 7, 7]", unsqueeze_598: "f32[1, 2048, 1, 1]", le_1: "b8[8, 112, 7, 7]", unsqueeze_610: "f32[1, 112, 1, 1]", le_2: "b8[8, 112, 7, 7]", unsqueeze_622: "f32[1, 112, 1, 1]", le_3: "b8[8, 112, 7, 7]", unsqueeze_634: "f32[1, 112, 1, 1]", le_4: "b8[8, 112, 7, 7]", unsqueeze_646: "f32[1, 112, 1, 1]", le_5: "b8[8, 112, 7, 7]", unsqueeze_658: "f32[1, 112, 1, 1]", le_6: "b8[8, 112, 7, 7]", unsqueeze_670: "f32[1, 112, 1, 1]", le_7: "b8[8, 112, 7, 7]", unsqueeze_682: "f32[1, 112, 1, 1]", le_8: "b8[8, 896, 7, 7]", unsqueeze_694: "f32[1, 896, 1, 1]", unsqueeze_706: "f32[1, 2048, 1, 1]", le_10: "b8[8, 112, 7, 7]", unsqueeze_718: "f32[1, 112, 1, 1]", le_11: "b8[8, 112, 7, 7]", unsqueeze_730: "f32[1, 112, 1, 1]", le_12: "b8[8, 112, 7, 7]", unsqueeze_742: "f32[1, 112, 1, 1]", le_13: "b8[8, 112, 7, 7]", unsqueeze_754: "f32[1, 112, 1, 1]", le_14: "b8[8, 112, 7, 7]", unsqueeze_766: "f32[1, 112, 1, 1]", le_15: "b8[8, 112, 7, 7]", unsqueeze_778: "f32[1, 112, 1, 1]", le_16: "b8[8, 112, 7, 7]", unsqueeze_790: "f32[1, 112, 1, 1]", le_17: "b8[8, 896, 7, 7]", unsqueeze_802: "f32[1, 896, 1, 1]", unsqueeze_814: "f32[1, 2048, 1, 1]", unsqueeze_826: "f32[1, 2048, 1, 1]", le_19: "b8[8, 112, 7, 7]", unsqueeze_838: "f32[1, 112, 1, 1]", le_20: "b8[8, 112, 7, 7]", unsqueeze_850: "f32[1, 112, 1, 1]", le_21: "b8[8, 112, 7, 7]", unsqueeze_862: "f32[1, 112, 1, 1]", le_22: "b8[8, 112, 7, 7]", unsqueeze_874: "f32[1, 112, 1, 1]", le_23: "b8[8, 112, 7, 7]", unsqueeze_886: "f32[1, 112, 1, 1]", le_24: "b8[8, 112, 7, 7]", unsqueeze_898: "f32[1, 112, 1, 1]", le_25: "b8[8, 112, 7, 7]", unsqueeze_910: "f32[1, 112, 1, 1]", le_26: "b8[8, 896, 14, 14]", unsqueeze_922: "f32[1, 896, 1, 1]", unsqueeze_934: "f32[1, 1024, 1, 1]", le_28: "b8[8, 56, 14, 14]", unsqueeze_946: "f32[1, 56, 1, 1]", le_29: "b8[8, 56, 14, 14]", unsqueeze_958: "f32[1, 56, 1, 1]", le_30: "b8[8, 56, 14, 14]", unsqueeze_970: "f32[1, 56, 1, 1]", le_31: "b8[8, 56, 14, 14]", unsqueeze_982: "f32[1, 56, 1, 1]", le_32: "b8[8, 56, 14, 14]", unsqueeze_994: "f32[1, 56, 1, 1]", le_33: "b8[8, 56, 14, 14]", unsqueeze_1006: "f32[1, 56, 1, 1]", le_34: "b8[8, 56, 14, 14]", unsqueeze_1018: "f32[1, 56, 1, 1]", le_35: "b8[8, 448, 14, 14]", unsqueeze_1030: "f32[1, 448, 1, 1]", unsqueeze_1042: "f32[1, 1024, 1, 1]", le_37: "b8[8, 56, 14, 14]", unsqueeze_1054: "f32[1, 56, 1, 1]", le_38: "b8[8, 56, 14, 14]", unsqueeze_1066: "f32[1, 56, 1, 1]", le_39: "b8[8, 56, 14, 14]", unsqueeze_1078: "f32[1, 56, 1, 1]", le_40: "b8[8, 56, 14, 14]", unsqueeze_1090: "f32[1, 56, 1, 1]", le_41: "b8[8, 56, 14, 14]", unsqueeze_1102: "f32[1, 56, 1, 1]", le_42: "b8[8, 56, 14, 14]", unsqueeze_1114: "f32[1, 56, 1, 1]", le_43: "b8[8, 56, 14, 14]", unsqueeze_1126: "f32[1, 56, 1, 1]", le_44: "b8[8, 448, 14, 14]", unsqueeze_1138: "f32[1, 448, 1, 1]", unsqueeze_1150: "f32[1, 1024, 1, 1]", le_46: "b8[8, 56, 14, 14]", unsqueeze_1162: "f32[1, 56, 1, 1]", le_47: "b8[8, 56, 14, 14]", unsqueeze_1174: "f32[1, 56, 1, 1]", le_48: "b8[8, 56, 14, 14]", unsqueeze_1186: "f32[1, 56, 1, 1]", le_49: "b8[8, 56, 14, 14]", unsqueeze_1198: "f32[1, 56, 1, 1]", le_50: "b8[8, 56, 14, 14]", unsqueeze_1210: "f32[1, 56, 1, 1]", le_51: "b8[8, 56, 14, 14]", unsqueeze_1222: "f32[1, 56, 1, 1]", le_52: "b8[8, 56, 14, 14]", unsqueeze_1234: "f32[1, 56, 1, 1]", le_53: "b8[8, 448, 14, 14]", unsqueeze_1246: "f32[1, 448, 1, 1]", unsqueeze_1258: "f32[1, 1024, 1, 1]", le_55: "b8[8, 56, 14, 14]", unsqueeze_1270: "f32[1, 56, 1, 1]", le_56: "b8[8, 56, 14, 14]", unsqueeze_1282: "f32[1, 56, 1, 1]", le_57: "b8[8, 56, 14, 14]", unsqueeze_1294: "f32[1, 56, 1, 1]", le_58: "b8[8, 56, 14, 14]", unsqueeze_1306: "f32[1, 56, 1, 1]", le_59: "b8[8, 56, 14, 14]", unsqueeze_1318: "f32[1, 56, 1, 1]", le_60: "b8[8, 56, 14, 14]", unsqueeze_1330: "f32[1, 56, 1, 1]", le_61: "b8[8, 56, 14, 14]", unsqueeze_1342: "f32[1, 56, 1, 1]", le_62: "b8[8, 448, 14, 14]", unsqueeze_1354: "f32[1, 448, 1, 1]", unsqueeze_1366: "f32[1, 1024, 1, 1]", le_64: "b8[8, 56, 14, 14]", unsqueeze_1378: "f32[1, 56, 1, 1]", le_65: "b8[8, 56, 14, 14]", unsqueeze_1390: "f32[1, 56, 1, 1]", le_66: "b8[8, 56, 14, 14]", unsqueeze_1402: "f32[1, 56, 1, 1]", le_67: "b8[8, 56, 14, 14]", unsqueeze_1414: "f32[1, 56, 1, 1]", le_68: "b8[8, 56, 14, 14]", unsqueeze_1426: "f32[1, 56, 1, 1]", le_69: "b8[8, 56, 14, 14]", unsqueeze_1438: "f32[1, 56, 1, 1]", le_70: "b8[8, 56, 14, 14]", unsqueeze_1450: "f32[1, 56, 1, 1]", le_71: "b8[8, 448, 14, 14]", unsqueeze_1462: "f32[1, 448, 1, 1]", unsqueeze_1474: "f32[1, 1024, 1, 1]", unsqueeze_1486: "f32[1, 1024, 1, 1]", le_73: "b8[8, 56, 14, 14]", unsqueeze_1498: "f32[1, 56, 1, 1]", le_74: "b8[8, 56, 14, 14]", unsqueeze_1510: "f32[1, 56, 1, 1]", le_75: "b8[8, 56, 14, 14]", unsqueeze_1522: "f32[1, 56, 1, 1]", le_76: "b8[8, 56, 14, 14]", unsqueeze_1534: "f32[1, 56, 1, 1]", le_77: "b8[8, 56, 14, 14]", unsqueeze_1546: "f32[1, 56, 1, 1]", le_78: "b8[8, 56, 14, 14]", unsqueeze_1558: "f32[1, 56, 1, 1]", le_79: "b8[8, 56, 14, 14]", unsqueeze_1570: "f32[1, 56, 1, 1]", le_80: "b8[8, 448, 28, 28]", unsqueeze_1582: "f32[1, 448, 1, 1]", unsqueeze_1594: "f32[1, 512, 1, 1]", le_82: "b8[8, 28, 28, 28]", unsqueeze_1606: "f32[1, 28, 1, 1]", le_83: "b8[8, 28, 28, 28]", unsqueeze_1618: "f32[1, 28, 1, 1]", le_84: "b8[8, 28, 28, 28]", unsqueeze_1630: "f32[1, 28, 1, 1]", le_85: "b8[8, 28, 28, 28]", unsqueeze_1642: "f32[1, 28, 1, 1]", le_86: "b8[8, 28, 28, 28]", unsqueeze_1654: "f32[1, 28, 1, 1]", le_87: "b8[8, 28, 28, 28]", unsqueeze_1666: "f32[1, 28, 1, 1]", le_88: "b8[8, 28, 28, 28]", unsqueeze_1678: "f32[1, 28, 1, 1]", le_89: "b8[8, 224, 28, 28]", unsqueeze_1690: "f32[1, 224, 1, 1]", unsqueeze_1702: "f32[1, 512, 1, 1]", le_91: "b8[8, 28, 28, 28]", unsqueeze_1714: "f32[1, 28, 1, 1]", le_92: "b8[8, 28, 28, 28]", unsqueeze_1726: "f32[1, 28, 1, 1]", le_93: "b8[8, 28, 28, 28]", unsqueeze_1738: "f32[1, 28, 1, 1]", le_94: "b8[8, 28, 28, 28]", unsqueeze_1750: "f32[1, 28, 1, 1]", le_95: "b8[8, 28, 28, 28]", unsqueeze_1762: "f32[1, 28, 1, 1]", le_96: "b8[8, 28, 28, 28]", unsqueeze_1774: "f32[1, 28, 1, 1]", le_97: "b8[8, 28, 28, 28]", unsqueeze_1786: "f32[1, 28, 1, 1]", le_98: "b8[8, 224, 28, 28]", unsqueeze_1798: "f32[1, 224, 1, 1]", unsqueeze_1810: "f32[1, 512, 1, 1]", le_100: "b8[8, 28, 28, 28]", unsqueeze_1822: "f32[1, 28, 1, 1]", le_101: "b8[8, 28, 28, 28]", unsqueeze_1834: "f32[1, 28, 1, 1]", le_102: "b8[8, 28, 28, 28]", unsqueeze_1846: "f32[1, 28, 1, 1]", le_103: "b8[8, 28, 28, 28]", unsqueeze_1858: "f32[1, 28, 1, 1]", le_104: "b8[8, 28, 28, 28]", unsqueeze_1870: "f32[1, 28, 1, 1]", le_105: "b8[8, 28, 28, 28]", unsqueeze_1882: "f32[1, 28, 1, 1]", le_106: "b8[8, 28, 28, 28]", unsqueeze_1894: "f32[1, 28, 1, 1]", le_107: "b8[8, 224, 28, 28]", unsqueeze_1906: "f32[1, 224, 1, 1]", unsqueeze_1918: "f32[1, 512, 1, 1]", unsqueeze_1930: "f32[1, 512, 1, 1]", le_109: "b8[8, 28, 28, 28]", unsqueeze_1942: "f32[1, 28, 1, 1]", le_110: "b8[8, 28, 28, 28]", unsqueeze_1954: "f32[1, 28, 1, 1]", le_111: "b8[8, 28, 28, 28]", unsqueeze_1966: "f32[1, 28, 1, 1]", le_112: "b8[8, 28, 28, 28]", unsqueeze_1978: "f32[1, 28, 1, 1]", le_113: "b8[8, 28, 28, 28]", unsqueeze_1990: "f32[1, 28, 1, 1]", le_114: "b8[8, 28, 28, 28]", unsqueeze_2002: "f32[1, 28, 1, 1]", le_115: "b8[8, 28, 28, 28]", unsqueeze_2014: "f32[1, 28, 1, 1]", le_116: "b8[8, 224, 56, 56]", unsqueeze_2026: "f32[1, 224, 1, 1]", unsqueeze_2038: "f32[1, 256, 1, 1]", le_118: "b8[8, 14, 56, 56]", unsqueeze_2050: "f32[1, 14, 1, 1]", le_119: "b8[8, 14, 56, 56]", unsqueeze_2062: "f32[1, 14, 1, 1]", le_120: "b8[8, 14, 56, 56]", unsqueeze_2074: "f32[1, 14, 1, 1]", le_121: "b8[8, 14, 56, 56]", unsqueeze_2086: "f32[1, 14, 1, 1]", le_122: "b8[8, 14, 56, 56]", unsqueeze_2098: "f32[1, 14, 1, 1]", le_123: "b8[8, 14, 56, 56]", unsqueeze_2110: "f32[1, 14, 1, 1]", le_124: "b8[8, 14, 56, 56]", unsqueeze_2122: "f32[1, 14, 1, 1]", le_125: "b8[8, 112, 56, 56]", unsqueeze_2134: "f32[1, 112, 1, 1]", unsqueeze_2146: "f32[1, 256, 1, 1]", le_127: "b8[8, 14, 56, 56]", unsqueeze_2158: "f32[1, 14, 1, 1]", le_128: "b8[8, 14, 56, 56]", unsqueeze_2170: "f32[1, 14, 1, 1]", le_129: "b8[8, 14, 56, 56]", unsqueeze_2182: "f32[1, 14, 1, 1]", le_130: "b8[8, 14, 56, 56]", unsqueeze_2194: "f32[1, 14, 1, 1]", le_131: "b8[8, 14, 56, 56]", unsqueeze_2206: "f32[1, 14, 1, 1]", le_132: "b8[8, 14, 56, 56]", unsqueeze_2218: "f32[1, 14, 1, 1]", le_133: "b8[8, 14, 56, 56]", unsqueeze_2230: "f32[1, 14, 1, 1]", le_134: "b8[8, 112, 56, 56]", unsqueeze_2242: "f32[1, 112, 1, 1]", unsqueeze_2254: "f32[1, 256, 1, 1]", unsqueeze_2266: "f32[1, 256, 1, 1]", le_136: "b8[8, 14, 56, 56]", unsqueeze_2278: "f32[1, 14, 1, 1]", le_137: "b8[8, 14, 56, 56]", unsqueeze_2290: "f32[1, 14, 1, 1]", le_138: "b8[8, 14, 56, 56]", unsqueeze_2302: "f32[1, 14, 1, 1]", le_139: "b8[8, 14, 56, 56]", unsqueeze_2314: "f32[1, 14, 1, 1]", le_140: "b8[8, 14, 56, 56]", unsqueeze_2326: "f32[1, 14, 1, 1]", le_141: "b8[8, 14, 56, 56]", unsqueeze_2338: "f32[1, 14, 1, 1]", le_142: "b8[8, 14, 56, 56]", unsqueeze_2350: "f32[1, 14, 1, 1]", le_143: "b8[8, 112, 56, 56]", unsqueeze_2362: "f32[1, 112, 1, 1]", unsqueeze_2374: "f32[1, 64, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
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
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_2: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_149: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_148, unsqueeze_598);  convolution_148 = unsqueeze_598 = None
    mul_1043: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_149)
    sum_3: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1043, [0, 2, 3]);  mul_1043 = None
    mul_1044: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_599: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1044, 0);  mul_1044 = None
    unsqueeze_600: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_1045: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_1046: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_445, squeeze_445)
    mul_1047: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1045, mul_1046);  mul_1045 = mul_1046 = None
    unsqueeze_602: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1047, 0);  mul_1047 = None
    unsqueeze_603: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_1048: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_445, primals_446);  primals_446 = None
    unsqueeze_605: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1048, 0);  mul_1048 = None
    unsqueeze_606: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_1049: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_604);  sub_149 = unsqueeze_604 = None
    sub_151: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_1049);  mul_1049 = None
    sub_152: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_601);  sub_151 = unsqueeze_601 = None
    mul_1050: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_607);  sub_152 = unsqueeze_607 = None
    mul_1051: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_445);  sum_3 = squeeze_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_1050, cat_15, primals_445, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1050 = cat_15 = primals_445 = None
    getitem_1452: "f32[8, 896, 7, 7]" = convolution_backward[0]
    getitem_1453: "f32[2048, 896, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_1: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1452, 1, 0, 112)
    slice_2: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1452, 1, 112, 224)
    slice_3: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1452, 1, 224, 336)
    slice_4: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1452, 1, 336, 448)
    slice_5: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1452, 1, 448, 560)
    slice_6: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1452, 1, 560, 672)
    slice_7: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1452, 1, 672, 784)
    slice_8: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1452, 1, 784, 896);  getitem_1452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_1: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, slice_7);  le_1 = slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_4: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_153: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_610);  convolution_147 = unsqueeze_610 = None
    mul_1052: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_153)
    sum_5: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1052, [0, 2, 3]);  mul_1052 = None
    mul_1053: "f32[112]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_611: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1053, 0);  mul_1053 = None
    unsqueeze_612: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
    unsqueeze_613: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
    mul_1054: "f32[112]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_1055: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_442, squeeze_442)
    mul_1056: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1054, mul_1055);  mul_1054 = mul_1055 = None
    unsqueeze_614: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1056, 0);  mul_1056 = None
    unsqueeze_615: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
    unsqueeze_616: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
    mul_1057: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_442, primals_443);  primals_443 = None
    unsqueeze_617: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1057, 0);  mul_1057 = None
    unsqueeze_618: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    mul_1058: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_616);  sub_153 = unsqueeze_616 = None
    sub_155: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_1058);  where_1 = mul_1058 = None
    sub_156: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_613);  sub_155 = unsqueeze_613 = None
    mul_1059: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_619);  sub_156 = unsqueeze_619 = None
    mul_1060: "f32[112]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_442);  sum_5 = squeeze_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_1059, add_821, primals_442, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1059 = add_821 = primals_442 = None
    getitem_1455: "f32[8, 112, 7, 7]" = convolution_backward_1[0]
    getitem_1456: "f32[112, 112, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_833: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(slice_6, getitem_1455);  slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_2: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, add_833);  le_2 = add_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_6: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_157: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_622);  convolution_146 = unsqueeze_622 = None
    mul_1061: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_157)
    sum_7: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1061, [0, 2, 3]);  mul_1061 = None
    mul_1062: "f32[112]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_623: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1062, 0);  mul_1062 = None
    unsqueeze_624: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
    unsqueeze_625: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
    mul_1063: "f32[112]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_1064: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_439, squeeze_439)
    mul_1065: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1063, mul_1064);  mul_1063 = mul_1064 = None
    unsqueeze_626: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1065, 0);  mul_1065 = None
    unsqueeze_627: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
    unsqueeze_628: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
    mul_1066: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_439, primals_440);  primals_440 = None
    unsqueeze_629: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1066, 0);  mul_1066 = None
    unsqueeze_630: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    mul_1067: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_628);  sub_157 = unsqueeze_628 = None
    sub_159: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_1067);  where_2 = mul_1067 = None
    sub_160: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_625);  sub_159 = unsqueeze_625 = None
    mul_1068: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_631);  sub_160 = unsqueeze_631 = None
    mul_1069: "f32[112]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_439);  sum_7 = squeeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_1068, add_815, primals_439, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1068 = add_815 = primals_439 = None
    getitem_1458: "f32[8, 112, 7, 7]" = convolution_backward_2[0]
    getitem_1459: "f32[112, 112, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_834: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(slice_5, getitem_1458);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_3: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, add_834);  le_3 = add_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_8: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_161: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_634);  convolution_145 = unsqueeze_634 = None
    mul_1070: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_161)
    sum_9: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1070, [0, 2, 3]);  mul_1070 = None
    mul_1071: "f32[112]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_635: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1071, 0);  mul_1071 = None
    unsqueeze_636: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
    unsqueeze_637: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
    mul_1072: "f32[112]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_1073: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_436, squeeze_436)
    mul_1074: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1072, mul_1073);  mul_1072 = mul_1073 = None
    unsqueeze_638: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1074, 0);  mul_1074 = None
    unsqueeze_639: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
    unsqueeze_640: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
    mul_1075: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_436, primals_437);  primals_437 = None
    unsqueeze_641: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1075, 0);  mul_1075 = None
    unsqueeze_642: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_1076: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_640);  sub_161 = unsqueeze_640 = None
    sub_163: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_1076);  where_3 = mul_1076 = None
    sub_164: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_637);  sub_163 = unsqueeze_637 = None
    mul_1077: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_643);  sub_164 = unsqueeze_643 = None
    mul_1078: "f32[112]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_436);  sum_9 = squeeze_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_1077, add_809, primals_436, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1077 = add_809 = primals_436 = None
    getitem_1461: "f32[8, 112, 7, 7]" = convolution_backward_3[0]
    getitem_1462: "f32[112, 112, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_835: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(slice_4, getitem_1461);  slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_4: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, add_835);  le_4 = add_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_10: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_165: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_144, unsqueeze_646);  convolution_144 = unsqueeze_646 = None
    mul_1079: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_165)
    sum_11: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1079, [0, 2, 3]);  mul_1079 = None
    mul_1080: "f32[112]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_647: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1080, 0);  mul_1080 = None
    unsqueeze_648: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
    unsqueeze_649: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
    mul_1081: "f32[112]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_1082: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_433, squeeze_433)
    mul_1083: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1081, mul_1082);  mul_1081 = mul_1082 = None
    unsqueeze_650: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1083, 0);  mul_1083 = None
    unsqueeze_651: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
    unsqueeze_652: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
    mul_1084: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_433, primals_434);  primals_434 = None
    unsqueeze_653: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1084, 0);  mul_1084 = None
    unsqueeze_654: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_1085: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_652);  sub_165 = unsqueeze_652 = None
    sub_167: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_1085);  where_4 = mul_1085 = None
    sub_168: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_649);  sub_167 = unsqueeze_649 = None
    mul_1086: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_655);  sub_168 = unsqueeze_655 = None
    mul_1087: "f32[112]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_433);  sum_11 = squeeze_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_1086, add_803, primals_433, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1086 = add_803 = primals_433 = None
    getitem_1464: "f32[8, 112, 7, 7]" = convolution_backward_4[0]
    getitem_1465: "f32[112, 112, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_836: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(slice_3, getitem_1464);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_5: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, add_836);  le_5 = add_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_12: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_169: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_143, unsqueeze_658);  convolution_143 = unsqueeze_658 = None
    mul_1088: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_169)
    sum_13: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1088, [0, 2, 3]);  mul_1088 = None
    mul_1089: "f32[112]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_659: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1089, 0);  mul_1089 = None
    unsqueeze_660: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
    unsqueeze_661: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
    mul_1090: "f32[112]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_1091: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_430, squeeze_430)
    mul_1092: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1090, mul_1091);  mul_1090 = mul_1091 = None
    unsqueeze_662: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1092, 0);  mul_1092 = None
    unsqueeze_663: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
    unsqueeze_664: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
    mul_1093: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_430, primals_431);  primals_431 = None
    unsqueeze_665: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1093, 0);  mul_1093 = None
    unsqueeze_666: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    mul_1094: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_664);  sub_169 = unsqueeze_664 = None
    sub_171: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_1094);  where_5 = mul_1094 = None
    sub_172: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_661);  sub_171 = unsqueeze_661 = None
    mul_1095: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_667);  sub_172 = unsqueeze_667 = None
    mul_1096: "f32[112]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_430);  sum_13 = squeeze_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_1095, add_797, primals_430, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1095 = add_797 = primals_430 = None
    getitem_1467: "f32[8, 112, 7, 7]" = convolution_backward_5[0]
    getitem_1468: "f32[112, 112, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_837: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(slice_2, getitem_1467);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_6: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, add_837);  le_6 = add_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_14: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_173: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_670);  convolution_142 = unsqueeze_670 = None
    mul_1097: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_173)
    sum_15: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1097, [0, 2, 3]);  mul_1097 = None
    mul_1098: "f32[112]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_671: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1098, 0);  mul_1098 = None
    unsqueeze_672: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
    unsqueeze_673: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
    mul_1099: "f32[112]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_1100: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_427, squeeze_427)
    mul_1101: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1099, mul_1100);  mul_1099 = mul_1100 = None
    unsqueeze_674: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1101, 0);  mul_1101 = None
    unsqueeze_675: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
    unsqueeze_676: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
    mul_1102: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_427, primals_428);  primals_428 = None
    unsqueeze_677: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1102, 0);  mul_1102 = None
    unsqueeze_678: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    mul_1103: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_676);  sub_173 = unsqueeze_676 = None
    sub_175: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_1103);  where_6 = mul_1103 = None
    sub_176: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_673);  sub_175 = unsqueeze_673 = None
    mul_1104: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_679);  sub_176 = unsqueeze_679 = None
    mul_1105: "f32[112]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_427);  sum_15 = squeeze_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_1104, add_791, primals_427, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1104 = add_791 = primals_427 = None
    getitem_1470: "f32[8, 112, 7, 7]" = convolution_backward_6[0]
    getitem_1471: "f32[112, 112, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_838: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(slice_1, getitem_1470);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_7: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, add_838);  le_7 = add_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_16: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_177: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_682);  convolution_141 = unsqueeze_682 = None
    mul_1106: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_177)
    sum_17: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1106, [0, 2, 3]);  mul_1106 = None
    mul_1107: "f32[112]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_683: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1107, 0);  mul_1107 = None
    unsqueeze_684: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
    unsqueeze_685: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
    mul_1108: "f32[112]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_1109: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_424, squeeze_424)
    mul_1110: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1108, mul_1109);  mul_1108 = mul_1109 = None
    unsqueeze_686: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1110, 0);  mul_1110 = None
    unsqueeze_687: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
    unsqueeze_688: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
    mul_1111: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_424, primals_425);  primals_425 = None
    unsqueeze_689: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1111, 0);  mul_1111 = None
    unsqueeze_690: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    mul_1112: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_688);  sub_177 = unsqueeze_688 = None
    sub_179: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_1112);  where_7 = mul_1112 = None
    sub_180: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_685);  sub_179 = unsqueeze_685 = None
    mul_1113: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_691);  sub_180 = unsqueeze_691 = None
    mul_1114: "f32[112]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_424);  sum_17 = squeeze_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_1113, getitem_1372, primals_424, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1113 = getitem_1372 = primals_424 = None
    getitem_1473: "f32[8, 112, 7, 7]" = convolution_backward_7[0]
    getitem_1474: "f32[112, 112, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_16: "f32[8, 896, 7, 7]" = torch.ops.aten.cat.default([getitem_1473, getitem_1470, getitem_1467, getitem_1464, getitem_1461, getitem_1458, getitem_1455, slice_8], 1);  getitem_1473 = getitem_1470 = getitem_1467 = getitem_1464 = getitem_1461 = getitem_1458 = getitem_1455 = slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_8: "f32[8, 896, 7, 7]" = torch.ops.aten.where.self(le_8, full_default, cat_16);  le_8 = cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_18: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_181: "f32[8, 896, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_694);  convolution_140 = unsqueeze_694 = None
    mul_1115: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_181)
    sum_19: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_1115, [0, 2, 3]);  mul_1115 = None
    mul_1116: "f32[896]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_695: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_1116, 0);  mul_1116 = None
    unsqueeze_696: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
    unsqueeze_697: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
    mul_1117: "f32[896]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_1118: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_421, squeeze_421)
    mul_1119: "f32[896]" = torch.ops.aten.mul.Tensor(mul_1117, mul_1118);  mul_1117 = mul_1118 = None
    unsqueeze_698: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_1119, 0);  mul_1119 = None
    unsqueeze_699: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
    unsqueeze_700: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
    mul_1120: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_421, primals_422);  primals_422 = None
    unsqueeze_701: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_1120, 0);  mul_1120 = None
    unsqueeze_702: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    mul_1121: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_700);  sub_181 = unsqueeze_700 = None
    sub_183: "f32[8, 896, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_1121);  where_8 = mul_1121 = None
    sub_184: "f32[8, 896, 7, 7]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_697);  sub_183 = unsqueeze_697 = None
    mul_1122: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_703);  sub_184 = unsqueeze_703 = None
    mul_1123: "f32[896]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_421);  sum_19 = squeeze_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_1122, relu_135, primals_421, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1122 = primals_421 = None
    getitem_1476: "f32[8, 2048, 7, 7]" = convolution_backward_8[0]
    getitem_1477: "f32[896, 2048, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_839: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where, getitem_1476);  where = getitem_1476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_173: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_135);  relu_135 = None
    alias_174: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_173);  alias_173 = None
    le_9: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_174, 0);  alias_174 = None
    where_9: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le_9, full_default, add_839);  le_9 = add_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_20: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_185: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_706);  convolution_139 = unsqueeze_706 = None
    mul_1124: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_185)
    sum_21: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1124, [0, 2, 3]);  mul_1124 = None
    mul_1125: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    unsqueeze_707: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1125, 0);  mul_1125 = None
    unsqueeze_708: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
    unsqueeze_709: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
    mul_1126: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    mul_1127: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_418, squeeze_418)
    mul_1128: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1126, mul_1127);  mul_1126 = mul_1127 = None
    unsqueeze_710: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1128, 0);  mul_1128 = None
    unsqueeze_711: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
    unsqueeze_712: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
    mul_1129: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_418, primals_419);  primals_419 = None
    unsqueeze_713: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1129, 0);  mul_1129 = None
    unsqueeze_714: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    mul_1130: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_712);  sub_185 = unsqueeze_712 = None
    sub_187: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_1130);  mul_1130 = None
    sub_188: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_709);  sub_187 = unsqueeze_709 = None
    mul_1131: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_715);  sub_188 = unsqueeze_715 = None
    mul_1132: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_418);  sum_21 = squeeze_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_1131, cat_14, primals_418, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1131 = cat_14 = primals_418 = None
    getitem_1479: "f32[8, 896, 7, 7]" = convolution_backward_9[0]
    getitem_1480: "f32[2048, 896, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_9: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1479, 1, 0, 112)
    slice_10: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1479, 1, 112, 224)
    slice_11: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1479, 1, 224, 336)
    slice_12: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1479, 1, 336, 448)
    slice_13: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1479, 1, 448, 560)
    slice_14: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1479, 1, 560, 672)
    slice_15: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1479, 1, 672, 784)
    slice_16: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1479, 1, 784, 896);  getitem_1479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_10: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_10, full_default, slice_15);  le_10 = slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_22: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_189: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_718);  convolution_138 = unsqueeze_718 = None
    mul_1133: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_189)
    sum_23: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1133, [0, 2, 3]);  mul_1133 = None
    mul_1134: "f32[112]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    unsqueeze_719: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1134, 0);  mul_1134 = None
    unsqueeze_720: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
    unsqueeze_721: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
    mul_1135: "f32[112]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    mul_1136: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_415, squeeze_415)
    mul_1137: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1135, mul_1136);  mul_1135 = mul_1136 = None
    unsqueeze_722: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1137, 0);  mul_1137 = None
    unsqueeze_723: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
    unsqueeze_724: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
    mul_1138: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_415, primals_416);  primals_416 = None
    unsqueeze_725: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1138, 0);  mul_1138 = None
    unsqueeze_726: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    mul_1139: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_724);  sub_189 = unsqueeze_724 = None
    sub_191: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_10, mul_1139);  where_10 = mul_1139 = None
    sub_192: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_721);  sub_191 = unsqueeze_721 = None
    mul_1140: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_727);  sub_192 = unsqueeze_727 = None
    mul_1141: "f32[112]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_415);  sum_23 = squeeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_1140, add_769, primals_415, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1140 = add_769 = primals_415 = None
    getitem_1482: "f32[8, 112, 7, 7]" = convolution_backward_10[0]
    getitem_1483: "f32[112, 112, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_840: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(slice_14, getitem_1482);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_11: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_11, full_default, add_840);  le_11 = add_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_24: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_193: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_730);  convolution_137 = unsqueeze_730 = None
    mul_1142: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_11, sub_193)
    sum_25: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1142, [0, 2, 3]);  mul_1142 = None
    mul_1143: "f32[112]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    unsqueeze_731: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1143, 0);  mul_1143 = None
    unsqueeze_732: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
    unsqueeze_733: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
    mul_1144: "f32[112]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    mul_1145: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_412, squeeze_412)
    mul_1146: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1144, mul_1145);  mul_1144 = mul_1145 = None
    unsqueeze_734: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1146, 0);  mul_1146 = None
    unsqueeze_735: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
    unsqueeze_736: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
    mul_1147: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_412, primals_413);  primals_413 = None
    unsqueeze_737: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1147, 0);  mul_1147 = None
    unsqueeze_738: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    mul_1148: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_736);  sub_193 = unsqueeze_736 = None
    sub_195: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_11, mul_1148);  where_11 = mul_1148 = None
    sub_196: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_733);  sub_195 = unsqueeze_733 = None
    mul_1149: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_739);  sub_196 = unsqueeze_739 = None
    mul_1150: "f32[112]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_412);  sum_25 = squeeze_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_1149, add_763, primals_412, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1149 = add_763 = primals_412 = None
    getitem_1485: "f32[8, 112, 7, 7]" = convolution_backward_11[0]
    getitem_1486: "f32[112, 112, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_841: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(slice_13, getitem_1485);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_12: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_12, full_default, add_841);  le_12 = add_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_26: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_197: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_742);  convolution_136 = unsqueeze_742 = None
    mul_1151: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_12, sub_197)
    sum_27: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1151, [0, 2, 3]);  mul_1151 = None
    mul_1152: "f32[112]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    unsqueeze_743: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1152, 0);  mul_1152 = None
    unsqueeze_744: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
    unsqueeze_745: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
    mul_1153: "f32[112]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    mul_1154: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_409, squeeze_409)
    mul_1155: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1153, mul_1154);  mul_1153 = mul_1154 = None
    unsqueeze_746: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1155, 0);  mul_1155 = None
    unsqueeze_747: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
    unsqueeze_748: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
    mul_1156: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_409, primals_410);  primals_410 = None
    unsqueeze_749: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1156, 0);  mul_1156 = None
    unsqueeze_750: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    mul_1157: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_748);  sub_197 = unsqueeze_748 = None
    sub_199: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_12, mul_1157);  where_12 = mul_1157 = None
    sub_200: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_745);  sub_199 = unsqueeze_745 = None
    mul_1158: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_751);  sub_200 = unsqueeze_751 = None
    mul_1159: "f32[112]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_409);  sum_27 = squeeze_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_1158, add_757, primals_409, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1158 = add_757 = primals_409 = None
    getitem_1488: "f32[8, 112, 7, 7]" = convolution_backward_12[0]
    getitem_1489: "f32[112, 112, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_842: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(slice_12, getitem_1488);  slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_13: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_13, full_default, add_842);  le_13 = add_842 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_28: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_201: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_754);  convolution_135 = unsqueeze_754 = None
    mul_1160: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, sub_201)
    sum_29: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1160, [0, 2, 3]);  mul_1160 = None
    mul_1161: "f32[112]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_755: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1161, 0);  mul_1161 = None
    unsqueeze_756: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
    unsqueeze_757: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
    mul_1162: "f32[112]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_1163: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_406, squeeze_406)
    mul_1164: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1162, mul_1163);  mul_1162 = mul_1163 = None
    unsqueeze_758: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1164, 0);  mul_1164 = None
    unsqueeze_759: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
    unsqueeze_760: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
    mul_1165: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_406, primals_407);  primals_407 = None
    unsqueeze_761: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1165, 0);  mul_1165 = None
    unsqueeze_762: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    mul_1166: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_760);  sub_201 = unsqueeze_760 = None
    sub_203: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_13, mul_1166);  where_13 = mul_1166 = None
    sub_204: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_757);  sub_203 = unsqueeze_757 = None
    mul_1167: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_763);  sub_204 = unsqueeze_763 = None
    mul_1168: "f32[112]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_406);  sum_29 = squeeze_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_1167, add_751, primals_406, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1167 = add_751 = primals_406 = None
    getitem_1491: "f32[8, 112, 7, 7]" = convolution_backward_13[0]
    getitem_1492: "f32[112, 112, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_843: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(slice_11, getitem_1491);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_14: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_14, full_default, add_843);  le_14 = add_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_30: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_205: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_766);  convolution_134 = unsqueeze_766 = None
    mul_1169: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_14, sub_205)
    sum_31: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1169, [0, 2, 3]);  mul_1169 = None
    mul_1170: "f32[112]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_767: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1170, 0);  mul_1170 = None
    unsqueeze_768: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
    unsqueeze_769: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
    mul_1171: "f32[112]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_1172: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_403, squeeze_403)
    mul_1173: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1171, mul_1172);  mul_1171 = mul_1172 = None
    unsqueeze_770: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1173, 0);  mul_1173 = None
    unsqueeze_771: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
    unsqueeze_772: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
    mul_1174: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_403, primals_404);  primals_404 = None
    unsqueeze_773: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1174, 0);  mul_1174 = None
    unsqueeze_774: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    mul_1175: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_772);  sub_205 = unsqueeze_772 = None
    sub_207: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_14, mul_1175);  where_14 = mul_1175 = None
    sub_208: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_769);  sub_207 = unsqueeze_769 = None
    mul_1176: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_775);  sub_208 = unsqueeze_775 = None
    mul_1177: "f32[112]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_403);  sum_31 = squeeze_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_1176, add_745, primals_403, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1176 = add_745 = primals_403 = None
    getitem_1494: "f32[8, 112, 7, 7]" = convolution_backward_14[0]
    getitem_1495: "f32[112, 112, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_844: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(slice_10, getitem_1494);  slice_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_15: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_15, full_default, add_844);  le_15 = add_844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_32: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_209: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_778);  convolution_133 = unsqueeze_778 = None
    mul_1178: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_15, sub_209)
    sum_33: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1178, [0, 2, 3]);  mul_1178 = None
    mul_1179: "f32[112]" = torch.ops.aten.mul.Tensor(sum_32, 0.002551020408163265)
    unsqueeze_779: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1179, 0);  mul_1179 = None
    unsqueeze_780: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
    unsqueeze_781: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
    mul_1180: "f32[112]" = torch.ops.aten.mul.Tensor(sum_33, 0.002551020408163265)
    mul_1181: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_400, squeeze_400)
    mul_1182: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1180, mul_1181);  mul_1180 = mul_1181 = None
    unsqueeze_782: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1182, 0);  mul_1182 = None
    unsqueeze_783: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
    unsqueeze_784: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
    mul_1183: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_400, primals_401);  primals_401 = None
    unsqueeze_785: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1183, 0);  mul_1183 = None
    unsqueeze_786: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    mul_1184: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_784);  sub_209 = unsqueeze_784 = None
    sub_211: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_15, mul_1184);  where_15 = mul_1184 = None
    sub_212: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_781);  sub_211 = unsqueeze_781 = None
    mul_1185: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_787);  sub_212 = unsqueeze_787 = None
    mul_1186: "f32[112]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_400);  sum_33 = squeeze_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_1185, add_739, primals_400, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1185 = add_739 = primals_400 = None
    getitem_1497: "f32[8, 112, 7, 7]" = convolution_backward_15[0]
    getitem_1498: "f32[112, 112, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_845: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(slice_9, getitem_1497);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_16: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_16, full_default, add_845);  le_16 = add_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_34: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_213: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_790);  convolution_132 = unsqueeze_790 = None
    mul_1187: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_16, sub_213)
    sum_35: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1187, [0, 2, 3]);  mul_1187 = None
    mul_1188: "f32[112]" = torch.ops.aten.mul.Tensor(sum_34, 0.002551020408163265)
    unsqueeze_791: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1188, 0);  mul_1188 = None
    unsqueeze_792: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
    unsqueeze_793: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
    mul_1189: "f32[112]" = torch.ops.aten.mul.Tensor(sum_35, 0.002551020408163265)
    mul_1190: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_397, squeeze_397)
    mul_1191: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1189, mul_1190);  mul_1189 = mul_1190 = None
    unsqueeze_794: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1191, 0);  mul_1191 = None
    unsqueeze_795: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
    unsqueeze_796: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
    mul_1192: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_397, primals_398);  primals_398 = None
    unsqueeze_797: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1192, 0);  mul_1192 = None
    unsqueeze_798: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    mul_1193: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_796);  sub_213 = unsqueeze_796 = None
    sub_215: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_16, mul_1193);  where_16 = mul_1193 = None
    sub_216: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_793);  sub_215 = unsqueeze_793 = None
    mul_1194: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_799);  sub_216 = unsqueeze_799 = None
    mul_1195: "f32[112]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_397);  sum_35 = squeeze_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_1194, getitem_1282, primals_397, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1194 = getitem_1282 = primals_397 = None
    getitem_1500: "f32[8, 112, 7, 7]" = convolution_backward_16[0]
    getitem_1501: "f32[112, 112, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_17: "f32[8, 896, 7, 7]" = torch.ops.aten.cat.default([getitem_1500, getitem_1497, getitem_1494, getitem_1491, getitem_1488, getitem_1485, getitem_1482, slice_16], 1);  getitem_1500 = getitem_1497 = getitem_1494 = getitem_1491 = getitem_1488 = getitem_1485 = getitem_1482 = slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_17: "f32[8, 896, 7, 7]" = torch.ops.aten.where.self(le_17, full_default, cat_17);  le_17 = cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_36: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_217: "f32[8, 896, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_802);  convolution_131 = unsqueeze_802 = None
    mul_1196: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(where_17, sub_217)
    sum_37: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_1196, [0, 2, 3]);  mul_1196 = None
    mul_1197: "f32[896]" = torch.ops.aten.mul.Tensor(sum_36, 0.002551020408163265)
    unsqueeze_803: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_1197, 0);  mul_1197 = None
    unsqueeze_804: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
    unsqueeze_805: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
    mul_1198: "f32[896]" = torch.ops.aten.mul.Tensor(sum_37, 0.002551020408163265)
    mul_1199: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_394, squeeze_394)
    mul_1200: "f32[896]" = torch.ops.aten.mul.Tensor(mul_1198, mul_1199);  mul_1198 = mul_1199 = None
    unsqueeze_806: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_1200, 0);  mul_1200 = None
    unsqueeze_807: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
    unsqueeze_808: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
    mul_1201: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_394, primals_395);  primals_395 = None
    unsqueeze_809: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_1201, 0);  mul_1201 = None
    unsqueeze_810: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    mul_1202: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_808);  sub_217 = unsqueeze_808 = None
    sub_219: "f32[8, 896, 7, 7]" = torch.ops.aten.sub.Tensor(where_17, mul_1202);  where_17 = mul_1202 = None
    sub_220: "f32[8, 896, 7, 7]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_805);  sub_219 = unsqueeze_805 = None
    mul_1203: "f32[8, 896, 7, 7]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_811);  sub_220 = unsqueeze_811 = None
    mul_1204: "f32[896]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_394);  sum_37 = squeeze_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_1203, relu_126, primals_394, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1203 = primals_394 = None
    getitem_1503: "f32[8, 2048, 7, 7]" = convolution_backward_17[0]
    getitem_1504: "f32[896, 2048, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_846: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where_9, getitem_1503);  where_9 = getitem_1503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_200: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_126);  relu_126 = None
    alias_201: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_200);  alias_200 = None
    le_18: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_201, 0);  alias_201 = None
    where_18: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le_18, full_default, add_846);  le_18 = add_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    sum_38: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_221: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_814);  convolution_130 = unsqueeze_814 = None
    mul_1205: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_18, sub_221)
    sum_39: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1205, [0, 2, 3]);  mul_1205 = None
    mul_1206: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_38, 0.002551020408163265)
    unsqueeze_815: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1206, 0);  mul_1206 = None
    unsqueeze_816: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
    unsqueeze_817: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
    mul_1207: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_39, 0.002551020408163265)
    mul_1208: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_391, squeeze_391)
    mul_1209: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1207, mul_1208);  mul_1207 = mul_1208 = None
    unsqueeze_818: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1209, 0);  mul_1209 = None
    unsqueeze_819: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
    unsqueeze_820: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
    mul_1210: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_391, primals_392);  primals_392 = None
    unsqueeze_821: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1210, 0);  mul_1210 = None
    unsqueeze_822: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    mul_1211: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_820);  sub_221 = unsqueeze_820 = None
    sub_223: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_18, mul_1211);  mul_1211 = None
    sub_224: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_817);  sub_223 = None
    mul_1212: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_823);  sub_224 = unsqueeze_823 = None
    mul_1213: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_391);  sum_39 = squeeze_391 = None
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_1212, relu_117, primals_391, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1212 = primals_391 = None
    getitem_1506: "f32[8, 1024, 14, 14]" = convolution_backward_18[0]
    getitem_1507: "f32[2048, 1024, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sub_225: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_826);  convolution_129 = unsqueeze_826 = None
    mul_1214: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_18, sub_225)
    sum_41: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1214, [0, 2, 3]);  mul_1214 = None
    mul_1216: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_41, 0.002551020408163265)
    mul_1217: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_388, squeeze_388)
    mul_1218: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1216, mul_1217);  mul_1216 = mul_1217 = None
    unsqueeze_830: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1218, 0);  mul_1218 = None
    unsqueeze_831: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
    unsqueeze_832: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
    mul_1219: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_388, primals_389);  primals_389 = None
    unsqueeze_833: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1219, 0);  mul_1219 = None
    unsqueeze_834: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    mul_1220: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_832);  sub_225 = unsqueeze_832 = None
    sub_227: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_18, mul_1220);  where_18 = mul_1220 = None
    sub_228: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_817);  sub_227 = unsqueeze_817 = None
    mul_1221: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_835);  sub_228 = unsqueeze_835 = None
    mul_1222: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_388);  sum_41 = squeeze_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_1221, cat_13, primals_388, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1221 = cat_13 = primals_388 = None
    getitem_1509: "f32[8, 896, 7, 7]" = convolution_backward_19[0]
    getitem_1510: "f32[2048, 896, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_17: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1509, 1, 0, 112)
    slice_18: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1509, 1, 112, 224)
    slice_19: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1509, 1, 224, 336)
    slice_20: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1509, 1, 336, 448)
    slice_21: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1509, 1, 448, 560)
    slice_22: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1509, 1, 560, 672)
    slice_23: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1509, 1, 672, 784)
    slice_24: "f32[8, 112, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_1509, 1, 784, 896);  getitem_1509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_backward: "f32[8, 112, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(slice_24, getitem_1267, [3, 3], [2, 2], [1, 1], False, True, None);  slice_24 = getitem_1267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_19: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_19, full_default, slice_23);  le_19 = slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_42: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_229: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_838);  convolution_128 = unsqueeze_838 = None
    mul_1223: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_19, sub_229)
    sum_43: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1223, [0, 2, 3]);  mul_1223 = None
    mul_1224: "f32[112]" = torch.ops.aten.mul.Tensor(sum_42, 0.002551020408163265)
    unsqueeze_839: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1224, 0);  mul_1224 = None
    unsqueeze_840: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
    unsqueeze_841: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
    mul_1225: "f32[112]" = torch.ops.aten.mul.Tensor(sum_43, 0.002551020408163265)
    mul_1226: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_385, squeeze_385)
    mul_1227: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1225, mul_1226);  mul_1225 = mul_1226 = None
    unsqueeze_842: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1227, 0);  mul_1227 = None
    unsqueeze_843: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
    unsqueeze_844: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
    mul_1228: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_385, primals_386);  primals_386 = None
    unsqueeze_845: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1228, 0);  mul_1228 = None
    unsqueeze_846: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    mul_1229: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_844);  sub_229 = unsqueeze_844 = None
    sub_231: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_19, mul_1229);  where_19 = mul_1229 = None
    sub_232: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_841);  sub_231 = unsqueeze_841 = None
    mul_1230: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_847);  sub_232 = unsqueeze_847 = None
    mul_1231: "f32[112]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_385);  sum_43 = squeeze_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_1230, getitem_1256, primals_385, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1230 = getitem_1256 = primals_385 = None
    getitem_1512: "f32[8, 112, 14, 14]" = convolution_backward_20[0]
    getitem_1513: "f32[112, 112, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_20: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_20, full_default, slice_22);  le_20 = slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_44: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_233: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_850);  convolution_127 = unsqueeze_850 = None
    mul_1232: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_20, sub_233)
    sum_45: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1232, [0, 2, 3]);  mul_1232 = None
    mul_1233: "f32[112]" = torch.ops.aten.mul.Tensor(sum_44, 0.002551020408163265)
    unsqueeze_851: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1233, 0);  mul_1233 = None
    unsqueeze_852: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 2);  unsqueeze_851 = None
    unsqueeze_853: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 3);  unsqueeze_852 = None
    mul_1234: "f32[112]" = torch.ops.aten.mul.Tensor(sum_45, 0.002551020408163265)
    mul_1235: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_382, squeeze_382)
    mul_1236: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1234, mul_1235);  mul_1234 = mul_1235 = None
    unsqueeze_854: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1236, 0);  mul_1236 = None
    unsqueeze_855: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 2);  unsqueeze_854 = None
    unsqueeze_856: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 3);  unsqueeze_855 = None
    mul_1237: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_382, primals_383);  primals_383 = None
    unsqueeze_857: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1237, 0);  mul_1237 = None
    unsqueeze_858: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    mul_1238: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_856);  sub_233 = unsqueeze_856 = None
    sub_235: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_20, mul_1238);  where_20 = mul_1238 = None
    sub_236: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_853);  sub_235 = unsqueeze_853 = None
    mul_1239: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_859);  sub_236 = unsqueeze_859 = None
    mul_1240: "f32[112]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_382);  sum_45 = squeeze_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_1239, getitem_1245, primals_382, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1239 = getitem_1245 = primals_382 = None
    getitem_1515: "f32[8, 112, 14, 14]" = convolution_backward_21[0]
    getitem_1516: "f32[112, 112, 3, 3]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_21: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_21, full_default, slice_21);  le_21 = slice_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_46: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_237: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_862);  convolution_126 = unsqueeze_862 = None
    mul_1241: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_21, sub_237)
    sum_47: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1241, [0, 2, 3]);  mul_1241 = None
    mul_1242: "f32[112]" = torch.ops.aten.mul.Tensor(sum_46, 0.002551020408163265)
    unsqueeze_863: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1242, 0);  mul_1242 = None
    unsqueeze_864: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 2);  unsqueeze_863 = None
    unsqueeze_865: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 3);  unsqueeze_864 = None
    mul_1243: "f32[112]" = torch.ops.aten.mul.Tensor(sum_47, 0.002551020408163265)
    mul_1244: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_379, squeeze_379)
    mul_1245: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1243, mul_1244);  mul_1243 = mul_1244 = None
    unsqueeze_866: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1245, 0);  mul_1245 = None
    unsqueeze_867: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 2);  unsqueeze_866 = None
    unsqueeze_868: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 3);  unsqueeze_867 = None
    mul_1246: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_379, primals_380);  primals_380 = None
    unsqueeze_869: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1246, 0);  mul_1246 = None
    unsqueeze_870: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    mul_1247: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_868);  sub_237 = unsqueeze_868 = None
    sub_239: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_21, mul_1247);  where_21 = mul_1247 = None
    sub_240: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_865);  sub_239 = unsqueeze_865 = None
    mul_1248: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_871);  sub_240 = unsqueeze_871 = None
    mul_1249: "f32[112]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_379);  sum_47 = squeeze_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_1248, getitem_1234, primals_379, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1248 = getitem_1234 = primals_379 = None
    getitem_1518: "f32[8, 112, 14, 14]" = convolution_backward_22[0]
    getitem_1519: "f32[112, 112, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_22: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_22, full_default, slice_20);  le_22 = slice_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_48: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_241: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_874);  convolution_125 = unsqueeze_874 = None
    mul_1250: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_22, sub_241)
    sum_49: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1250, [0, 2, 3]);  mul_1250 = None
    mul_1251: "f32[112]" = torch.ops.aten.mul.Tensor(sum_48, 0.002551020408163265)
    unsqueeze_875: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1251, 0);  mul_1251 = None
    unsqueeze_876: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 2);  unsqueeze_875 = None
    unsqueeze_877: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 3);  unsqueeze_876 = None
    mul_1252: "f32[112]" = torch.ops.aten.mul.Tensor(sum_49, 0.002551020408163265)
    mul_1253: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_376, squeeze_376)
    mul_1254: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1252, mul_1253);  mul_1252 = mul_1253 = None
    unsqueeze_878: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1254, 0);  mul_1254 = None
    unsqueeze_879: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 2);  unsqueeze_878 = None
    unsqueeze_880: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 3);  unsqueeze_879 = None
    mul_1255: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_376, primals_377);  primals_377 = None
    unsqueeze_881: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1255, 0);  mul_1255 = None
    unsqueeze_882: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    mul_1256: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_880);  sub_241 = unsqueeze_880 = None
    sub_243: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_22, mul_1256);  where_22 = mul_1256 = None
    sub_244: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_877);  sub_243 = unsqueeze_877 = None
    mul_1257: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_883);  sub_244 = unsqueeze_883 = None
    mul_1258: "f32[112]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_376);  sum_49 = squeeze_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_1257, getitem_1223, primals_376, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1257 = getitem_1223 = primals_376 = None
    getitem_1521: "f32[8, 112, 14, 14]" = convolution_backward_23[0]
    getitem_1522: "f32[112, 112, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_23: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_23, full_default, slice_19);  le_23 = slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_50: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_245: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_886);  convolution_124 = unsqueeze_886 = None
    mul_1259: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_23, sub_245)
    sum_51: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1259, [0, 2, 3]);  mul_1259 = None
    mul_1260: "f32[112]" = torch.ops.aten.mul.Tensor(sum_50, 0.002551020408163265)
    unsqueeze_887: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1260, 0);  mul_1260 = None
    unsqueeze_888: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 2);  unsqueeze_887 = None
    unsqueeze_889: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 3);  unsqueeze_888 = None
    mul_1261: "f32[112]" = torch.ops.aten.mul.Tensor(sum_51, 0.002551020408163265)
    mul_1262: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_373, squeeze_373)
    mul_1263: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1261, mul_1262);  mul_1261 = mul_1262 = None
    unsqueeze_890: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1263, 0);  mul_1263 = None
    unsqueeze_891: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 2);  unsqueeze_890 = None
    unsqueeze_892: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 3);  unsqueeze_891 = None
    mul_1264: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_373, primals_374);  primals_374 = None
    unsqueeze_893: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1264, 0);  mul_1264 = None
    unsqueeze_894: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    mul_1265: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_892);  sub_245 = unsqueeze_892 = None
    sub_247: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_23, mul_1265);  where_23 = mul_1265 = None
    sub_248: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_889);  sub_247 = unsqueeze_889 = None
    mul_1266: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_895);  sub_248 = unsqueeze_895 = None
    mul_1267: "f32[112]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_373);  sum_51 = squeeze_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_1266, getitem_1212, primals_373, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1266 = getitem_1212 = primals_373 = None
    getitem_1524: "f32[8, 112, 14, 14]" = convolution_backward_24[0]
    getitem_1525: "f32[112, 112, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_24: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_24, full_default, slice_18);  le_24 = slice_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_52: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_249: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_898);  convolution_123 = unsqueeze_898 = None
    mul_1268: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_24, sub_249)
    sum_53: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1268, [0, 2, 3]);  mul_1268 = None
    mul_1269: "f32[112]" = torch.ops.aten.mul.Tensor(sum_52, 0.002551020408163265)
    unsqueeze_899: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1269, 0);  mul_1269 = None
    unsqueeze_900: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 2);  unsqueeze_899 = None
    unsqueeze_901: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 3);  unsqueeze_900 = None
    mul_1270: "f32[112]" = torch.ops.aten.mul.Tensor(sum_53, 0.002551020408163265)
    mul_1271: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_370, squeeze_370)
    mul_1272: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1270, mul_1271);  mul_1270 = mul_1271 = None
    unsqueeze_902: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1272, 0);  mul_1272 = None
    unsqueeze_903: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 2);  unsqueeze_902 = None
    unsqueeze_904: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 3);  unsqueeze_903 = None
    mul_1273: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_370, primals_371);  primals_371 = None
    unsqueeze_905: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1273, 0);  mul_1273 = None
    unsqueeze_906: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    mul_1274: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_904);  sub_249 = unsqueeze_904 = None
    sub_251: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_24, mul_1274);  where_24 = mul_1274 = None
    sub_252: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_901);  sub_251 = unsqueeze_901 = None
    mul_1275: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_907);  sub_252 = unsqueeze_907 = None
    mul_1276: "f32[112]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_370);  sum_53 = squeeze_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_1275, getitem_1201, primals_370, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1275 = getitem_1201 = primals_370 = None
    getitem_1527: "f32[8, 112, 14, 14]" = convolution_backward_25[0]
    getitem_1528: "f32[112, 112, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_25: "f32[8, 112, 7, 7]" = torch.ops.aten.where.self(le_25, full_default, slice_17);  le_25 = slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_54: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_253: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_910);  convolution_122 = unsqueeze_910 = None
    mul_1277: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(where_25, sub_253)
    sum_55: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_1277, [0, 2, 3]);  mul_1277 = None
    mul_1278: "f32[112]" = torch.ops.aten.mul.Tensor(sum_54, 0.002551020408163265)
    unsqueeze_911: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1278, 0);  mul_1278 = None
    unsqueeze_912: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 2);  unsqueeze_911 = None
    unsqueeze_913: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 3);  unsqueeze_912 = None
    mul_1279: "f32[112]" = torch.ops.aten.mul.Tensor(sum_55, 0.002551020408163265)
    mul_1280: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_367, squeeze_367)
    mul_1281: "f32[112]" = torch.ops.aten.mul.Tensor(mul_1279, mul_1280);  mul_1279 = mul_1280 = None
    unsqueeze_914: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1281, 0);  mul_1281 = None
    unsqueeze_915: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 2);  unsqueeze_914 = None
    unsqueeze_916: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 3);  unsqueeze_915 = None
    mul_1282: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_367, primals_368);  primals_368 = None
    unsqueeze_917: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1282, 0);  mul_1282 = None
    unsqueeze_918: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    mul_1283: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_916);  sub_253 = unsqueeze_916 = None
    sub_255: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(where_25, mul_1283);  where_25 = mul_1283 = None
    sub_256: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_913);  sub_255 = unsqueeze_913 = None
    mul_1284: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_919);  sub_256 = unsqueeze_919 = None
    mul_1285: "f32[112]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_367);  sum_55 = squeeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_1284, getitem_1190, primals_367, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1284 = getitem_1190 = primals_367 = None
    getitem_1530: "f32[8, 112, 14, 14]" = convolution_backward_26[0]
    getitem_1531: "f32[112, 112, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_18: "f32[8, 896, 14, 14]" = torch.ops.aten.cat.default([getitem_1530, getitem_1527, getitem_1524, getitem_1521, getitem_1518, getitem_1515, getitem_1512, avg_pool2d_backward], 1);  getitem_1530 = getitem_1527 = getitem_1524 = getitem_1521 = getitem_1518 = getitem_1515 = getitem_1512 = avg_pool2d_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_26: "f32[8, 896, 14, 14]" = torch.ops.aten.where.self(le_26, full_default, cat_18);  le_26 = cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_56: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_257: "f32[8, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_922);  convolution_121 = unsqueeze_922 = None
    mul_1286: "f32[8, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_257)
    sum_57: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_1286, [0, 2, 3]);  mul_1286 = None
    mul_1287: "f32[896]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_923: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_1287, 0);  mul_1287 = None
    unsqueeze_924: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 2);  unsqueeze_923 = None
    unsqueeze_925: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 3);  unsqueeze_924 = None
    mul_1288: "f32[896]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_1289: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_364, squeeze_364)
    mul_1290: "f32[896]" = torch.ops.aten.mul.Tensor(mul_1288, mul_1289);  mul_1288 = mul_1289 = None
    unsqueeze_926: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_1290, 0);  mul_1290 = None
    unsqueeze_927: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 2);  unsqueeze_926 = None
    unsqueeze_928: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 3);  unsqueeze_927 = None
    mul_1291: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_364, primals_365);  primals_365 = None
    unsqueeze_929: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_1291, 0);  mul_1291 = None
    unsqueeze_930: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
    unsqueeze_931: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
    mul_1292: "f32[8, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_928);  sub_257 = unsqueeze_928 = None
    sub_259: "f32[8, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_26, mul_1292);  where_26 = mul_1292 = None
    sub_260: "f32[8, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_925);  sub_259 = unsqueeze_925 = None
    mul_1293: "f32[8, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_931);  sub_260 = unsqueeze_931 = None
    mul_1294: "f32[896]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_364);  sum_57 = squeeze_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_1293, relu_117, primals_364, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1293 = primals_364 = None
    getitem_1533: "f32[8, 1024, 14, 14]" = convolution_backward_27[0]
    getitem_1534: "f32[896, 1024, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_847: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(getitem_1506, getitem_1533);  getitem_1506 = getitem_1533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_227: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_117);  relu_117 = None
    alias_228: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_227);  alias_227 = None
    le_27: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_228, 0);  alias_228 = None
    where_27: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_27, full_default, add_847);  le_27 = add_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_58: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_261: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_934);  convolution_120 = unsqueeze_934 = None
    mul_1295: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_261)
    sum_59: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1295, [0, 2, 3]);  mul_1295 = None
    mul_1296: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_935: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1296, 0);  mul_1296 = None
    unsqueeze_936: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 2);  unsqueeze_935 = None
    unsqueeze_937: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 3);  unsqueeze_936 = None
    mul_1297: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_1298: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_361, squeeze_361)
    mul_1299: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1297, mul_1298);  mul_1297 = mul_1298 = None
    unsqueeze_938: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1299, 0);  mul_1299 = None
    unsqueeze_939: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 2);  unsqueeze_938 = None
    unsqueeze_940: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 3);  unsqueeze_939 = None
    mul_1300: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_361, primals_362);  primals_362 = None
    unsqueeze_941: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1300, 0);  mul_1300 = None
    unsqueeze_942: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
    unsqueeze_943: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
    mul_1301: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_940);  sub_261 = unsqueeze_940 = None
    sub_263: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_1301);  mul_1301 = None
    sub_264: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_937);  sub_263 = unsqueeze_937 = None
    mul_1302: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_943);  sub_264 = unsqueeze_943 = None
    mul_1303: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_361);  sum_59 = squeeze_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_1302, cat_12, primals_361, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1302 = cat_12 = primals_361 = None
    getitem_1536: "f32[8, 448, 14, 14]" = convolution_backward_28[0]
    getitem_1537: "f32[1024, 448, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_25: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1536, 1, 0, 56)
    slice_26: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1536, 1, 56, 112)
    slice_27: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1536, 1, 112, 168)
    slice_28: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1536, 1, 168, 224)
    slice_29: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1536, 1, 224, 280)
    slice_30: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1536, 1, 280, 336)
    slice_31: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1536, 1, 336, 392)
    slice_32: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1536, 1, 392, 448);  getitem_1536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_28: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_28, full_default, slice_31);  le_28 = slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_60: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_265: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_946);  convolution_119 = unsqueeze_946 = None
    mul_1304: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_265)
    sum_61: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1304, [0, 2, 3]);  mul_1304 = None
    mul_1305: "f32[56]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_947: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1305, 0);  mul_1305 = None
    unsqueeze_948: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 2);  unsqueeze_947 = None
    unsqueeze_949: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 3);  unsqueeze_948 = None
    mul_1306: "f32[56]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_1307: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_358, squeeze_358)
    mul_1308: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1306, mul_1307);  mul_1306 = mul_1307 = None
    unsqueeze_950: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1308, 0);  mul_1308 = None
    unsqueeze_951: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 2);  unsqueeze_950 = None
    unsqueeze_952: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 3);  unsqueeze_951 = None
    mul_1309: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_358, primals_359);  primals_359 = None
    unsqueeze_953: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1309, 0);  mul_1309 = None
    unsqueeze_954: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
    unsqueeze_955: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
    mul_1310: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_952);  sub_265 = unsqueeze_952 = None
    sub_267: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_28, mul_1310);  where_28 = mul_1310 = None
    sub_268: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_949);  sub_267 = unsqueeze_949 = None
    mul_1311: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_955);  sub_268 = unsqueeze_955 = None
    mul_1312: "f32[56]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_358);  sum_61 = squeeze_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_1311, add_666, primals_358, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1311 = add_666 = primals_358 = None
    getitem_1539: "f32[8, 56, 14, 14]" = convolution_backward_29[0]
    getitem_1540: "f32[56, 56, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_848: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_30, getitem_1539);  slice_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_29: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_29, full_default, add_848);  le_29 = add_848 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_62: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_269: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_958);  convolution_118 = unsqueeze_958 = None
    mul_1313: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_29, sub_269)
    sum_63: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1313, [0, 2, 3]);  mul_1313 = None
    mul_1314: "f32[56]" = torch.ops.aten.mul.Tensor(sum_62, 0.0006377551020408163)
    unsqueeze_959: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1314, 0);  mul_1314 = None
    unsqueeze_960: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 2);  unsqueeze_959 = None
    unsqueeze_961: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 3);  unsqueeze_960 = None
    mul_1315: "f32[56]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    mul_1316: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_355, squeeze_355)
    mul_1317: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1315, mul_1316);  mul_1315 = mul_1316 = None
    unsqueeze_962: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1317, 0);  mul_1317 = None
    unsqueeze_963: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 2);  unsqueeze_962 = None
    unsqueeze_964: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 3);  unsqueeze_963 = None
    mul_1318: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_355, primals_356);  primals_356 = None
    unsqueeze_965: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1318, 0);  mul_1318 = None
    unsqueeze_966: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
    unsqueeze_967: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
    mul_1319: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_964);  sub_269 = unsqueeze_964 = None
    sub_271: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_29, mul_1319);  where_29 = mul_1319 = None
    sub_272: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_961);  sub_271 = unsqueeze_961 = None
    mul_1320: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_967);  sub_272 = unsqueeze_967 = None
    mul_1321: "f32[56]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_355);  sum_63 = squeeze_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_1320, add_660, primals_355, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1320 = add_660 = primals_355 = None
    getitem_1542: "f32[8, 56, 14, 14]" = convolution_backward_30[0]
    getitem_1543: "f32[56, 56, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_849: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_29, getitem_1542);  slice_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_30: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_30, full_default, add_849);  le_30 = add_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_64: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_273: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_970);  convolution_117 = unsqueeze_970 = None
    mul_1322: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_273)
    sum_65: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1322, [0, 2, 3]);  mul_1322 = None
    mul_1323: "f32[56]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    unsqueeze_971: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1323, 0);  mul_1323 = None
    unsqueeze_972: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 2);  unsqueeze_971 = None
    unsqueeze_973: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 3);  unsqueeze_972 = None
    mul_1324: "f32[56]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    mul_1325: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_352, squeeze_352)
    mul_1326: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1324, mul_1325);  mul_1324 = mul_1325 = None
    unsqueeze_974: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1326, 0);  mul_1326 = None
    unsqueeze_975: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 2);  unsqueeze_974 = None
    unsqueeze_976: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 3);  unsqueeze_975 = None
    mul_1327: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_352, primals_353);  primals_353 = None
    unsqueeze_977: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1327, 0);  mul_1327 = None
    unsqueeze_978: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    mul_1328: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_976);  sub_273 = unsqueeze_976 = None
    sub_275: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_30, mul_1328);  where_30 = mul_1328 = None
    sub_276: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_973);  sub_275 = unsqueeze_973 = None
    mul_1329: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_979);  sub_276 = unsqueeze_979 = None
    mul_1330: "f32[56]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_352);  sum_65 = squeeze_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_1329, add_654, primals_352, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1329 = add_654 = primals_352 = None
    getitem_1545: "f32[8, 56, 14, 14]" = convolution_backward_31[0]
    getitem_1546: "f32[56, 56, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_850: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_28, getitem_1545);  slice_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_31: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_31, full_default, add_850);  le_31 = add_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_66: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_277: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_982);  convolution_116 = unsqueeze_982 = None
    mul_1331: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_277)
    sum_67: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1331, [0, 2, 3]);  mul_1331 = None
    mul_1332: "f32[56]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    unsqueeze_983: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1332, 0);  mul_1332 = None
    unsqueeze_984: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 2);  unsqueeze_983 = None
    unsqueeze_985: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 3);  unsqueeze_984 = None
    mul_1333: "f32[56]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    mul_1334: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_349, squeeze_349)
    mul_1335: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1333, mul_1334);  mul_1333 = mul_1334 = None
    unsqueeze_986: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1335, 0);  mul_1335 = None
    unsqueeze_987: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 2);  unsqueeze_986 = None
    unsqueeze_988: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 3);  unsqueeze_987 = None
    mul_1336: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_349, primals_350);  primals_350 = None
    unsqueeze_989: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1336, 0);  mul_1336 = None
    unsqueeze_990: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
    unsqueeze_991: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
    mul_1337: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_988);  sub_277 = unsqueeze_988 = None
    sub_279: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_1337);  where_31 = mul_1337 = None
    sub_280: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_985);  sub_279 = unsqueeze_985 = None
    mul_1338: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_991);  sub_280 = unsqueeze_991 = None
    mul_1339: "f32[56]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_349);  sum_67 = squeeze_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_1338, add_648, primals_349, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1338 = add_648 = primals_349 = None
    getitem_1548: "f32[8, 56, 14, 14]" = convolution_backward_32[0]
    getitem_1549: "f32[56, 56, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_851: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_27, getitem_1548);  slice_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_32: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_32, full_default, add_851);  le_32 = add_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_68: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_281: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_994);  convolution_115 = unsqueeze_994 = None
    mul_1340: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_281)
    sum_69: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1340, [0, 2, 3]);  mul_1340 = None
    mul_1341: "f32[56]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    unsqueeze_995: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1341, 0);  mul_1341 = None
    unsqueeze_996: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 2);  unsqueeze_995 = None
    unsqueeze_997: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 3);  unsqueeze_996 = None
    mul_1342: "f32[56]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    mul_1343: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_346, squeeze_346)
    mul_1344: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1342, mul_1343);  mul_1342 = mul_1343 = None
    unsqueeze_998: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1344, 0);  mul_1344 = None
    unsqueeze_999: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 2);  unsqueeze_998 = None
    unsqueeze_1000: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 3);  unsqueeze_999 = None
    mul_1345: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_346, primals_347);  primals_347 = None
    unsqueeze_1001: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1345, 0);  mul_1345 = None
    unsqueeze_1002: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 2);  unsqueeze_1001 = None
    unsqueeze_1003: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 3);  unsqueeze_1002 = None
    mul_1346: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_1000);  sub_281 = unsqueeze_1000 = None
    sub_283: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_32, mul_1346);  where_32 = mul_1346 = None
    sub_284: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_997);  sub_283 = unsqueeze_997 = None
    mul_1347: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_1003);  sub_284 = unsqueeze_1003 = None
    mul_1348: "f32[56]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_346);  sum_69 = squeeze_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_1347, add_642, primals_346, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1347 = add_642 = primals_346 = None
    getitem_1551: "f32[8, 56, 14, 14]" = convolution_backward_33[0]
    getitem_1552: "f32[56, 56, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_852: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_26, getitem_1551);  slice_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_33: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_33, full_default, add_852);  le_33 = add_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_70: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_285: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_1006);  convolution_114 = unsqueeze_1006 = None
    mul_1349: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_285)
    sum_71: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1349, [0, 2, 3]);  mul_1349 = None
    mul_1350: "f32[56]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_1007: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1350, 0);  mul_1350 = None
    unsqueeze_1008: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 2);  unsqueeze_1007 = None
    unsqueeze_1009: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 3);  unsqueeze_1008 = None
    mul_1351: "f32[56]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_1352: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_343, squeeze_343)
    mul_1353: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1351, mul_1352);  mul_1351 = mul_1352 = None
    unsqueeze_1010: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1353, 0);  mul_1353 = None
    unsqueeze_1011: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 2);  unsqueeze_1010 = None
    unsqueeze_1012: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 3);  unsqueeze_1011 = None
    mul_1354: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_343, primals_344);  primals_344 = None
    unsqueeze_1013: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1354, 0);  mul_1354 = None
    unsqueeze_1014: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 2);  unsqueeze_1013 = None
    unsqueeze_1015: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 3);  unsqueeze_1014 = None
    mul_1355: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_1012);  sub_285 = unsqueeze_1012 = None
    sub_287: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_33, mul_1355);  where_33 = mul_1355 = None
    sub_288: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_1009);  sub_287 = unsqueeze_1009 = None
    mul_1356: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_1015);  sub_288 = unsqueeze_1015 = None
    mul_1357: "f32[56]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_343);  sum_71 = squeeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_1356, add_636, primals_343, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1356 = add_636 = primals_343 = None
    getitem_1554: "f32[8, 56, 14, 14]" = convolution_backward_34[0]
    getitem_1555: "f32[56, 56, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_853: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_25, getitem_1554);  slice_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_34: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_34, full_default, add_853);  le_34 = add_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_72: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_289: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_1018);  convolution_113 = unsqueeze_1018 = None
    mul_1358: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_289)
    sum_73: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1358, [0, 2, 3]);  mul_1358 = None
    mul_1359: "f32[56]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_1019: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1359, 0);  mul_1359 = None
    unsqueeze_1020: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 2);  unsqueeze_1019 = None
    unsqueeze_1021: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 3);  unsqueeze_1020 = None
    mul_1360: "f32[56]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_1361: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_340, squeeze_340)
    mul_1362: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1360, mul_1361);  mul_1360 = mul_1361 = None
    unsqueeze_1022: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1362, 0);  mul_1362 = None
    unsqueeze_1023: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 2);  unsqueeze_1022 = None
    unsqueeze_1024: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 3);  unsqueeze_1023 = None
    mul_1363: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_340, primals_341);  primals_341 = None
    unsqueeze_1025: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1363, 0);  mul_1363 = None
    unsqueeze_1026: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 2);  unsqueeze_1025 = None
    unsqueeze_1027: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 3);  unsqueeze_1026 = None
    mul_1364: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_1024);  sub_289 = unsqueeze_1024 = None
    sub_291: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_34, mul_1364);  where_34 = mul_1364 = None
    sub_292: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_1021);  sub_291 = unsqueeze_1021 = None
    mul_1365: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_1027);  sub_292 = unsqueeze_1027 = None
    mul_1366: "f32[56]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_340);  sum_73 = squeeze_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_1365, getitem_1100, primals_340, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1365 = getitem_1100 = primals_340 = None
    getitem_1557: "f32[8, 56, 14, 14]" = convolution_backward_35[0]
    getitem_1558: "f32[56, 56, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_19: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([getitem_1557, getitem_1554, getitem_1551, getitem_1548, getitem_1545, getitem_1542, getitem_1539, slice_32], 1);  getitem_1557 = getitem_1554 = getitem_1551 = getitem_1548 = getitem_1545 = getitem_1542 = getitem_1539 = slice_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_35: "f32[8, 448, 14, 14]" = torch.ops.aten.where.self(le_35, full_default, cat_19);  le_35 = cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_74: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_293: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_1030);  convolution_112 = unsqueeze_1030 = None
    mul_1367: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_293)
    sum_75: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_1367, [0, 2, 3]);  mul_1367 = None
    mul_1368: "f32[448]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_1031: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1368, 0);  mul_1368 = None
    unsqueeze_1032: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 2);  unsqueeze_1031 = None
    unsqueeze_1033: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 3);  unsqueeze_1032 = None
    mul_1369: "f32[448]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_1370: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_337, squeeze_337)
    mul_1371: "f32[448]" = torch.ops.aten.mul.Tensor(mul_1369, mul_1370);  mul_1369 = mul_1370 = None
    unsqueeze_1034: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1371, 0);  mul_1371 = None
    unsqueeze_1035: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 2);  unsqueeze_1034 = None
    unsqueeze_1036: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 3);  unsqueeze_1035 = None
    mul_1372: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_337, primals_338);  primals_338 = None
    unsqueeze_1037: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1372, 0);  mul_1372 = None
    unsqueeze_1038: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 2);  unsqueeze_1037 = None
    unsqueeze_1039: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 3);  unsqueeze_1038 = None
    mul_1373: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_1036);  sub_293 = unsqueeze_1036 = None
    sub_295: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_1373);  where_35 = mul_1373 = None
    sub_296: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_1033);  sub_295 = unsqueeze_1033 = None
    mul_1374: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_1039);  sub_296 = unsqueeze_1039 = None
    mul_1375: "f32[448]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_337);  sum_75 = squeeze_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_1374, relu_108, primals_337, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1374 = primals_337 = None
    getitem_1560: "f32[8, 1024, 14, 14]" = convolution_backward_36[0]
    getitem_1561: "f32[448, 1024, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_854: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_27, getitem_1560);  where_27 = getitem_1560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_254: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_108);  relu_108 = None
    alias_255: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_254);  alias_254 = None
    le_36: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_255, 0);  alias_255 = None
    where_36: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_36, full_default, add_854);  le_36 = add_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_76: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_297: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_1042);  convolution_111 = unsqueeze_1042 = None
    mul_1376: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_297)
    sum_77: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1376, [0, 2, 3]);  mul_1376 = None
    mul_1377: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    unsqueeze_1043: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1377, 0);  mul_1377 = None
    unsqueeze_1044: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 2);  unsqueeze_1043 = None
    unsqueeze_1045: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, 3);  unsqueeze_1044 = None
    mul_1378: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    mul_1379: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_334, squeeze_334)
    mul_1380: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1378, mul_1379);  mul_1378 = mul_1379 = None
    unsqueeze_1046: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1380, 0);  mul_1380 = None
    unsqueeze_1047: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 2);  unsqueeze_1046 = None
    unsqueeze_1048: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 3);  unsqueeze_1047 = None
    mul_1381: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_334, primals_335);  primals_335 = None
    unsqueeze_1049: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1381, 0);  mul_1381 = None
    unsqueeze_1050: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 2);  unsqueeze_1049 = None
    unsqueeze_1051: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 3);  unsqueeze_1050 = None
    mul_1382: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_1048);  sub_297 = unsqueeze_1048 = None
    sub_299: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_36, mul_1382);  mul_1382 = None
    sub_300: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_299, unsqueeze_1045);  sub_299 = unsqueeze_1045 = None
    mul_1383: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_1051);  sub_300 = unsqueeze_1051 = None
    mul_1384: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_334);  sum_77 = squeeze_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_1383, cat_11, primals_334, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1383 = cat_11 = primals_334 = None
    getitem_1563: "f32[8, 448, 14, 14]" = convolution_backward_37[0]
    getitem_1564: "f32[1024, 448, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_33: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1563, 1, 0, 56)
    slice_34: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1563, 1, 56, 112)
    slice_35: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1563, 1, 112, 168)
    slice_36: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1563, 1, 168, 224)
    slice_37: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1563, 1, 224, 280)
    slice_38: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1563, 1, 280, 336)
    slice_39: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1563, 1, 336, 392)
    slice_40: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1563, 1, 392, 448);  getitem_1563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_37: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_37, full_default, slice_39);  le_37 = slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_78: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_301: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_1054);  convolution_110 = unsqueeze_1054 = None
    mul_1385: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_301)
    sum_79: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1385, [0, 2, 3]);  mul_1385 = None
    mul_1386: "f32[56]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    unsqueeze_1055: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1386, 0);  mul_1386 = None
    unsqueeze_1056: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 2);  unsqueeze_1055 = None
    unsqueeze_1057: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, 3);  unsqueeze_1056 = None
    mul_1387: "f32[56]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    mul_1388: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_331, squeeze_331)
    mul_1389: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1387, mul_1388);  mul_1387 = mul_1388 = None
    unsqueeze_1058: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1389, 0);  mul_1389 = None
    unsqueeze_1059: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 2);  unsqueeze_1058 = None
    unsqueeze_1060: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 3);  unsqueeze_1059 = None
    mul_1390: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_331, primals_332);  primals_332 = None
    unsqueeze_1061: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1390, 0);  mul_1390 = None
    unsqueeze_1062: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 2);  unsqueeze_1061 = None
    unsqueeze_1063: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 3);  unsqueeze_1062 = None
    mul_1391: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_1060);  sub_301 = unsqueeze_1060 = None
    sub_303: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_37, mul_1391);  where_37 = mul_1391 = None
    sub_304: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_1057);  sub_303 = unsqueeze_1057 = None
    mul_1392: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_1063);  sub_304 = unsqueeze_1063 = None
    mul_1393: "f32[56]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_331);  sum_79 = squeeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_1392, add_614, primals_331, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1392 = add_614 = primals_331 = None
    getitem_1566: "f32[8, 56, 14, 14]" = convolution_backward_38[0]
    getitem_1567: "f32[56, 56, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_855: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_38, getitem_1566);  slice_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_38: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_38, full_default, add_855);  le_38 = add_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_80: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_305: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_1066);  convolution_109 = unsqueeze_1066 = None
    mul_1394: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_305)
    sum_81: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1394, [0, 2, 3]);  mul_1394 = None
    mul_1395: "f32[56]" = torch.ops.aten.mul.Tensor(sum_80, 0.0006377551020408163)
    unsqueeze_1067: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1395, 0);  mul_1395 = None
    unsqueeze_1068: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 2);  unsqueeze_1067 = None
    unsqueeze_1069: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, 3);  unsqueeze_1068 = None
    mul_1396: "f32[56]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    mul_1397: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_328, squeeze_328)
    mul_1398: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1396, mul_1397);  mul_1396 = mul_1397 = None
    unsqueeze_1070: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1398, 0);  mul_1398 = None
    unsqueeze_1071: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 2);  unsqueeze_1070 = None
    unsqueeze_1072: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 3);  unsqueeze_1071 = None
    mul_1399: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_328, primals_329);  primals_329 = None
    unsqueeze_1073: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1399, 0);  mul_1399 = None
    unsqueeze_1074: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 2);  unsqueeze_1073 = None
    unsqueeze_1075: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 3);  unsqueeze_1074 = None
    mul_1400: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_1072);  sub_305 = unsqueeze_1072 = None
    sub_307: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_38, mul_1400);  where_38 = mul_1400 = None
    sub_308: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_307, unsqueeze_1069);  sub_307 = unsqueeze_1069 = None
    mul_1401: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1075);  sub_308 = unsqueeze_1075 = None
    mul_1402: "f32[56]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_328);  sum_81 = squeeze_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_1401, add_608, primals_328, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1401 = add_608 = primals_328 = None
    getitem_1569: "f32[8, 56, 14, 14]" = convolution_backward_39[0]
    getitem_1570: "f32[56, 56, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_856: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_37, getitem_1569);  slice_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_39: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_39, full_default, add_856);  le_39 = add_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_82: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_309: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_1078);  convolution_108 = unsqueeze_1078 = None
    mul_1403: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_309)
    sum_83: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1403, [0, 2, 3]);  mul_1403 = None
    mul_1404: "f32[56]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    unsqueeze_1079: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1404, 0);  mul_1404 = None
    unsqueeze_1080: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1079, 2);  unsqueeze_1079 = None
    unsqueeze_1081: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, 3);  unsqueeze_1080 = None
    mul_1405: "f32[56]" = torch.ops.aten.mul.Tensor(sum_83, 0.0006377551020408163)
    mul_1406: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_325, squeeze_325)
    mul_1407: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1405, mul_1406);  mul_1405 = mul_1406 = None
    unsqueeze_1082: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1407, 0);  mul_1407 = None
    unsqueeze_1083: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 2);  unsqueeze_1082 = None
    unsqueeze_1084: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 3);  unsqueeze_1083 = None
    mul_1408: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_325, primals_326);  primals_326 = None
    unsqueeze_1085: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1408, 0);  mul_1408 = None
    unsqueeze_1086: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 2);  unsqueeze_1085 = None
    unsqueeze_1087: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 3);  unsqueeze_1086 = None
    mul_1409: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_1084);  sub_309 = unsqueeze_1084 = None
    sub_311: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_39, mul_1409);  where_39 = mul_1409 = None
    sub_312: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_311, unsqueeze_1081);  sub_311 = unsqueeze_1081 = None
    mul_1410: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1087);  sub_312 = unsqueeze_1087 = None
    mul_1411: "f32[56]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_325);  sum_83 = squeeze_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_1410, add_602, primals_325, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1410 = add_602 = primals_325 = None
    getitem_1572: "f32[8, 56, 14, 14]" = convolution_backward_40[0]
    getitem_1573: "f32[56, 56, 3, 3]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_857: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_36, getitem_1572);  slice_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_40: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_40, full_default, add_857);  le_40 = add_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_84: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_313: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_1090);  convolution_107 = unsqueeze_1090 = None
    mul_1412: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_313)
    sum_85: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1412, [0, 2, 3]);  mul_1412 = None
    mul_1413: "f32[56]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    unsqueeze_1091: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1413, 0);  mul_1413 = None
    unsqueeze_1092: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1091, 2);  unsqueeze_1091 = None
    unsqueeze_1093: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, 3);  unsqueeze_1092 = None
    mul_1414: "f32[56]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    mul_1415: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_322, squeeze_322)
    mul_1416: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1414, mul_1415);  mul_1414 = mul_1415 = None
    unsqueeze_1094: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1416, 0);  mul_1416 = None
    unsqueeze_1095: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 2);  unsqueeze_1094 = None
    unsqueeze_1096: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 3);  unsqueeze_1095 = None
    mul_1417: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_322, primals_323);  primals_323 = None
    unsqueeze_1097: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1417, 0);  mul_1417 = None
    unsqueeze_1098: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 2);  unsqueeze_1097 = None
    unsqueeze_1099: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 3);  unsqueeze_1098 = None
    mul_1418: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_1096);  sub_313 = unsqueeze_1096 = None
    sub_315: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_1418);  where_40 = mul_1418 = None
    sub_316: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_315, unsqueeze_1093);  sub_315 = unsqueeze_1093 = None
    mul_1419: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1099);  sub_316 = unsqueeze_1099 = None
    mul_1420: "f32[56]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_322);  sum_85 = squeeze_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_1419, add_596, primals_322, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1419 = add_596 = primals_322 = None
    getitem_1575: "f32[8, 56, 14, 14]" = convolution_backward_41[0]
    getitem_1576: "f32[56, 56, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_858: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_35, getitem_1575);  slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_41: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_41, full_default, add_858);  le_41 = add_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_86: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_317: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_1102);  convolution_106 = unsqueeze_1102 = None
    mul_1421: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, sub_317)
    sum_87: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1421, [0, 2, 3]);  mul_1421 = None
    mul_1422: "f32[56]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006377551020408163)
    unsqueeze_1103: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1422, 0);  mul_1422 = None
    unsqueeze_1104: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1103, 2);  unsqueeze_1103 = None
    unsqueeze_1105: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, 3);  unsqueeze_1104 = None
    mul_1423: "f32[56]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006377551020408163)
    mul_1424: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_319, squeeze_319)
    mul_1425: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1423, mul_1424);  mul_1423 = mul_1424 = None
    unsqueeze_1106: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1425, 0);  mul_1425 = None
    unsqueeze_1107: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 2);  unsqueeze_1106 = None
    unsqueeze_1108: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 3);  unsqueeze_1107 = None
    mul_1426: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_319, primals_320);  primals_320 = None
    unsqueeze_1109: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1426, 0);  mul_1426 = None
    unsqueeze_1110: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 2);  unsqueeze_1109 = None
    unsqueeze_1111: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 3);  unsqueeze_1110 = None
    mul_1427: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_1108);  sub_317 = unsqueeze_1108 = None
    sub_319: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_41, mul_1427);  where_41 = mul_1427 = None
    sub_320: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_319, unsqueeze_1105);  sub_319 = unsqueeze_1105 = None
    mul_1428: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1111);  sub_320 = unsqueeze_1111 = None
    mul_1429: "f32[56]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_319);  sum_87 = squeeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_1428, add_590, primals_319, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1428 = add_590 = primals_319 = None
    getitem_1578: "f32[8, 56, 14, 14]" = convolution_backward_42[0]
    getitem_1579: "f32[56, 56, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_859: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_34, getitem_1578);  slice_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_42: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_42, full_default, add_859);  le_42 = add_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_88: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_321: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_1114);  convolution_105 = unsqueeze_1114 = None
    mul_1430: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_321)
    sum_89: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1430, [0, 2, 3]);  mul_1430 = None
    mul_1431: "f32[56]" = torch.ops.aten.mul.Tensor(sum_88, 0.0006377551020408163)
    unsqueeze_1115: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1431, 0);  mul_1431 = None
    unsqueeze_1116: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1115, 2);  unsqueeze_1115 = None
    unsqueeze_1117: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, 3);  unsqueeze_1116 = None
    mul_1432: "f32[56]" = torch.ops.aten.mul.Tensor(sum_89, 0.0006377551020408163)
    mul_1433: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_316, squeeze_316)
    mul_1434: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1432, mul_1433);  mul_1432 = mul_1433 = None
    unsqueeze_1118: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1434, 0);  mul_1434 = None
    unsqueeze_1119: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 2);  unsqueeze_1118 = None
    unsqueeze_1120: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 3);  unsqueeze_1119 = None
    mul_1435: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_316, primals_317);  primals_317 = None
    unsqueeze_1121: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1435, 0);  mul_1435 = None
    unsqueeze_1122: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 2);  unsqueeze_1121 = None
    unsqueeze_1123: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 3);  unsqueeze_1122 = None
    mul_1436: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_1120);  sub_321 = unsqueeze_1120 = None
    sub_323: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_42, mul_1436);  where_42 = mul_1436 = None
    sub_324: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_1117);  sub_323 = unsqueeze_1117 = None
    mul_1437: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1123);  sub_324 = unsqueeze_1123 = None
    mul_1438: "f32[56]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_316);  sum_89 = squeeze_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1437, add_584, primals_316, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1437 = add_584 = primals_316 = None
    getitem_1581: "f32[8, 56, 14, 14]" = convolution_backward_43[0]
    getitem_1582: "f32[56, 56, 3, 3]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_860: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_33, getitem_1581);  slice_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_43: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_43, full_default, add_860);  le_43 = add_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_90: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_325: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_1126);  convolution_104 = unsqueeze_1126 = None
    mul_1439: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_325)
    sum_91: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1439, [0, 2, 3]);  mul_1439 = None
    mul_1440: "f32[56]" = torch.ops.aten.mul.Tensor(sum_90, 0.0006377551020408163)
    unsqueeze_1127: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1440, 0);  mul_1440 = None
    unsqueeze_1128: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1127, 2);  unsqueeze_1127 = None
    unsqueeze_1129: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, 3);  unsqueeze_1128 = None
    mul_1441: "f32[56]" = torch.ops.aten.mul.Tensor(sum_91, 0.0006377551020408163)
    mul_1442: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_313, squeeze_313)
    mul_1443: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1441, mul_1442);  mul_1441 = mul_1442 = None
    unsqueeze_1130: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1443, 0);  mul_1443 = None
    unsqueeze_1131: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 2);  unsqueeze_1130 = None
    unsqueeze_1132: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 3);  unsqueeze_1131 = None
    mul_1444: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_313, primals_314);  primals_314 = None
    unsqueeze_1133: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1444, 0);  mul_1444 = None
    unsqueeze_1134: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 2);  unsqueeze_1133 = None
    unsqueeze_1135: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 3);  unsqueeze_1134 = None
    mul_1445: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_1132);  sub_325 = unsqueeze_1132 = None
    sub_327: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_43, mul_1445);  where_43 = mul_1445 = None
    sub_328: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_327, unsqueeze_1129);  sub_327 = unsqueeze_1129 = None
    mul_1446: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1135);  sub_328 = unsqueeze_1135 = None
    mul_1447: "f32[56]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_313);  sum_91 = squeeze_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1446, getitem_1010, primals_313, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1446 = getitem_1010 = primals_313 = None
    getitem_1584: "f32[8, 56, 14, 14]" = convolution_backward_44[0]
    getitem_1585: "f32[56, 56, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_20: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([getitem_1584, getitem_1581, getitem_1578, getitem_1575, getitem_1572, getitem_1569, getitem_1566, slice_40], 1);  getitem_1584 = getitem_1581 = getitem_1578 = getitem_1575 = getitem_1572 = getitem_1569 = getitem_1566 = slice_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_44: "f32[8, 448, 14, 14]" = torch.ops.aten.where.self(le_44, full_default, cat_20);  le_44 = cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_92: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_329: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_1138);  convolution_103 = unsqueeze_1138 = None
    mul_1448: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_329)
    sum_93: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_1448, [0, 2, 3]);  mul_1448 = None
    mul_1449: "f32[448]" = torch.ops.aten.mul.Tensor(sum_92, 0.0006377551020408163)
    unsqueeze_1139: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1449, 0);  mul_1449 = None
    unsqueeze_1140: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1139, 2);  unsqueeze_1139 = None
    unsqueeze_1141: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, 3);  unsqueeze_1140 = None
    mul_1450: "f32[448]" = torch.ops.aten.mul.Tensor(sum_93, 0.0006377551020408163)
    mul_1451: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_310, squeeze_310)
    mul_1452: "f32[448]" = torch.ops.aten.mul.Tensor(mul_1450, mul_1451);  mul_1450 = mul_1451 = None
    unsqueeze_1142: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1452, 0);  mul_1452 = None
    unsqueeze_1143: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 2);  unsqueeze_1142 = None
    unsqueeze_1144: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 3);  unsqueeze_1143 = None
    mul_1453: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_310, primals_311);  primals_311 = None
    unsqueeze_1145: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1453, 0);  mul_1453 = None
    unsqueeze_1146: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 2);  unsqueeze_1145 = None
    unsqueeze_1147: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 3);  unsqueeze_1146 = None
    mul_1454: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_1144);  sub_329 = unsqueeze_1144 = None
    sub_331: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(where_44, mul_1454);  where_44 = mul_1454 = None
    sub_332: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(sub_331, unsqueeze_1141);  sub_331 = unsqueeze_1141 = None
    mul_1455: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1147);  sub_332 = unsqueeze_1147 = None
    mul_1456: "f32[448]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_310);  sum_93 = squeeze_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1455, relu_99, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1455 = primals_310 = None
    getitem_1587: "f32[8, 1024, 14, 14]" = convolution_backward_45[0]
    getitem_1588: "f32[448, 1024, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_861: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_36, getitem_1587);  where_36 = getitem_1587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_281: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_99);  relu_99 = None
    alias_282: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_281);  alias_281 = None
    le_45: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_282, 0);  alias_282 = None
    where_45: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_45, full_default, add_861);  le_45 = add_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_94: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_333: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_1150);  convolution_102 = unsqueeze_1150 = None
    mul_1457: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, sub_333)
    sum_95: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1457, [0, 2, 3]);  mul_1457 = None
    mul_1458: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_94, 0.0006377551020408163)
    unsqueeze_1151: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1458, 0);  mul_1458 = None
    unsqueeze_1152: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1151, 2);  unsqueeze_1151 = None
    unsqueeze_1153: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, 3);  unsqueeze_1152 = None
    mul_1459: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_95, 0.0006377551020408163)
    mul_1460: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_307, squeeze_307)
    mul_1461: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1459, mul_1460);  mul_1459 = mul_1460 = None
    unsqueeze_1154: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1461, 0);  mul_1461 = None
    unsqueeze_1155: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 2);  unsqueeze_1154 = None
    unsqueeze_1156: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 3);  unsqueeze_1155 = None
    mul_1462: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_307, primals_308);  primals_308 = None
    unsqueeze_1157: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1462, 0);  mul_1462 = None
    unsqueeze_1158: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 2);  unsqueeze_1157 = None
    unsqueeze_1159: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 3);  unsqueeze_1158 = None
    mul_1463: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_1156);  sub_333 = unsqueeze_1156 = None
    sub_335: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_45, mul_1463);  mul_1463 = None
    sub_336: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_335, unsqueeze_1153);  sub_335 = unsqueeze_1153 = None
    mul_1464: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_1159);  sub_336 = unsqueeze_1159 = None
    mul_1465: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_307);  sum_95 = squeeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1464, cat_10, primals_307, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1464 = cat_10 = primals_307 = None
    getitem_1590: "f32[8, 448, 14, 14]" = convolution_backward_46[0]
    getitem_1591: "f32[1024, 448, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_41: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1590, 1, 0, 56)
    slice_42: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1590, 1, 56, 112)
    slice_43: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1590, 1, 112, 168)
    slice_44: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1590, 1, 168, 224)
    slice_45: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1590, 1, 224, 280)
    slice_46: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1590, 1, 280, 336)
    slice_47: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1590, 1, 336, 392)
    slice_48: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1590, 1, 392, 448);  getitem_1590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_46: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_46, full_default, slice_47);  le_46 = slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_96: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_337: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_1162);  convolution_101 = unsqueeze_1162 = None
    mul_1466: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, sub_337)
    sum_97: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1466, [0, 2, 3]);  mul_1466 = None
    mul_1467: "f32[56]" = torch.ops.aten.mul.Tensor(sum_96, 0.0006377551020408163)
    unsqueeze_1163: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1467, 0);  mul_1467 = None
    unsqueeze_1164: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1163, 2);  unsqueeze_1163 = None
    unsqueeze_1165: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, 3);  unsqueeze_1164 = None
    mul_1468: "f32[56]" = torch.ops.aten.mul.Tensor(sum_97, 0.0006377551020408163)
    mul_1469: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_304, squeeze_304)
    mul_1470: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1468, mul_1469);  mul_1468 = mul_1469 = None
    unsqueeze_1166: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1470, 0);  mul_1470 = None
    unsqueeze_1167: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 2);  unsqueeze_1166 = None
    unsqueeze_1168: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 3);  unsqueeze_1167 = None
    mul_1471: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_304, primals_305);  primals_305 = None
    unsqueeze_1169: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1471, 0);  mul_1471 = None
    unsqueeze_1170: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 2);  unsqueeze_1169 = None
    unsqueeze_1171: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 3);  unsqueeze_1170 = None
    mul_1472: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_1168);  sub_337 = unsqueeze_1168 = None
    sub_339: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_46, mul_1472);  where_46 = mul_1472 = None
    sub_340: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_339, unsqueeze_1165);  sub_339 = unsqueeze_1165 = None
    mul_1473: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_1171);  sub_340 = unsqueeze_1171 = None
    mul_1474: "f32[56]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_304);  sum_97 = squeeze_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1473, add_562, primals_304, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1473 = add_562 = primals_304 = None
    getitem_1593: "f32[8, 56, 14, 14]" = convolution_backward_47[0]
    getitem_1594: "f32[56, 56, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_862: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_46, getitem_1593);  slice_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_47: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_47, full_default, add_862);  le_47 = add_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_98: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_341: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_1174);  convolution_100 = unsqueeze_1174 = None
    mul_1475: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, sub_341)
    sum_99: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1475, [0, 2, 3]);  mul_1475 = None
    mul_1476: "f32[56]" = torch.ops.aten.mul.Tensor(sum_98, 0.0006377551020408163)
    unsqueeze_1175: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1476, 0);  mul_1476 = None
    unsqueeze_1176: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1175, 2);  unsqueeze_1175 = None
    unsqueeze_1177: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, 3);  unsqueeze_1176 = None
    mul_1477: "f32[56]" = torch.ops.aten.mul.Tensor(sum_99, 0.0006377551020408163)
    mul_1478: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_301, squeeze_301)
    mul_1479: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1477, mul_1478);  mul_1477 = mul_1478 = None
    unsqueeze_1178: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1479, 0);  mul_1479 = None
    unsqueeze_1179: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 2);  unsqueeze_1178 = None
    unsqueeze_1180: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 3);  unsqueeze_1179 = None
    mul_1480: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_301, primals_302);  primals_302 = None
    unsqueeze_1181: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1480, 0);  mul_1480 = None
    unsqueeze_1182: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 2);  unsqueeze_1181 = None
    unsqueeze_1183: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 3);  unsqueeze_1182 = None
    mul_1481: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_1180);  sub_341 = unsqueeze_1180 = None
    sub_343: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_47, mul_1481);  where_47 = mul_1481 = None
    sub_344: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_343, unsqueeze_1177);  sub_343 = unsqueeze_1177 = None
    mul_1482: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_1183);  sub_344 = unsqueeze_1183 = None
    mul_1483: "f32[56]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_301);  sum_99 = squeeze_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1482, add_556, primals_301, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1482 = add_556 = primals_301 = None
    getitem_1596: "f32[8, 56, 14, 14]" = convolution_backward_48[0]
    getitem_1597: "f32[56, 56, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_863: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_45, getitem_1596);  slice_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_48: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_48, full_default, add_863);  le_48 = add_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_100: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_345: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_1186);  convolution_99 = unsqueeze_1186 = None
    mul_1484: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_48, sub_345)
    sum_101: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1484, [0, 2, 3]);  mul_1484 = None
    mul_1485: "f32[56]" = torch.ops.aten.mul.Tensor(sum_100, 0.0006377551020408163)
    unsqueeze_1187: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1485, 0);  mul_1485 = None
    unsqueeze_1188: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1187, 2);  unsqueeze_1187 = None
    unsqueeze_1189: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, 3);  unsqueeze_1188 = None
    mul_1486: "f32[56]" = torch.ops.aten.mul.Tensor(sum_101, 0.0006377551020408163)
    mul_1487: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_298, squeeze_298)
    mul_1488: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1486, mul_1487);  mul_1486 = mul_1487 = None
    unsqueeze_1190: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1488, 0);  mul_1488 = None
    unsqueeze_1191: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 2);  unsqueeze_1190 = None
    unsqueeze_1192: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 3);  unsqueeze_1191 = None
    mul_1489: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_298, primals_299);  primals_299 = None
    unsqueeze_1193: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1489, 0);  mul_1489 = None
    unsqueeze_1194: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 2);  unsqueeze_1193 = None
    unsqueeze_1195: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 3);  unsqueeze_1194 = None
    mul_1490: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_1192);  sub_345 = unsqueeze_1192 = None
    sub_347: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_48, mul_1490);  where_48 = mul_1490 = None
    sub_348: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_347, unsqueeze_1189);  sub_347 = unsqueeze_1189 = None
    mul_1491: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_1195);  sub_348 = unsqueeze_1195 = None
    mul_1492: "f32[56]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_298);  sum_101 = squeeze_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1491, add_550, primals_298, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1491 = add_550 = primals_298 = None
    getitem_1599: "f32[8, 56, 14, 14]" = convolution_backward_49[0]
    getitem_1600: "f32[56, 56, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_864: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_44, getitem_1599);  slice_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_49: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_49, full_default, add_864);  le_49 = add_864 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_102: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_349: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_1198);  convolution_98 = unsqueeze_1198 = None
    mul_1493: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_49, sub_349)
    sum_103: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1493, [0, 2, 3]);  mul_1493 = None
    mul_1494: "f32[56]" = torch.ops.aten.mul.Tensor(sum_102, 0.0006377551020408163)
    unsqueeze_1199: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1494, 0);  mul_1494 = None
    unsqueeze_1200: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1199, 2);  unsqueeze_1199 = None
    unsqueeze_1201: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, 3);  unsqueeze_1200 = None
    mul_1495: "f32[56]" = torch.ops.aten.mul.Tensor(sum_103, 0.0006377551020408163)
    mul_1496: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_295, squeeze_295)
    mul_1497: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1495, mul_1496);  mul_1495 = mul_1496 = None
    unsqueeze_1202: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1497, 0);  mul_1497 = None
    unsqueeze_1203: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 2);  unsqueeze_1202 = None
    unsqueeze_1204: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 3);  unsqueeze_1203 = None
    mul_1498: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_295, primals_296);  primals_296 = None
    unsqueeze_1205: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1498, 0);  mul_1498 = None
    unsqueeze_1206: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 2);  unsqueeze_1205 = None
    unsqueeze_1207: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 3);  unsqueeze_1206 = None
    mul_1499: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_1204);  sub_349 = unsqueeze_1204 = None
    sub_351: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_49, mul_1499);  where_49 = mul_1499 = None
    sub_352: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_351, unsqueeze_1201);  sub_351 = unsqueeze_1201 = None
    mul_1500: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_1207);  sub_352 = unsqueeze_1207 = None
    mul_1501: "f32[56]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_295);  sum_103 = squeeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1500, add_544, primals_295, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1500 = add_544 = primals_295 = None
    getitem_1602: "f32[8, 56, 14, 14]" = convolution_backward_50[0]
    getitem_1603: "f32[56, 56, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_865: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_43, getitem_1602);  slice_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_50: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_50, full_default, add_865);  le_50 = add_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_104: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_353: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_1210);  convolution_97 = unsqueeze_1210 = None
    mul_1502: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_50, sub_353)
    sum_105: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1502, [0, 2, 3]);  mul_1502 = None
    mul_1503: "f32[56]" = torch.ops.aten.mul.Tensor(sum_104, 0.0006377551020408163)
    unsqueeze_1211: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1503, 0);  mul_1503 = None
    unsqueeze_1212: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1211, 2);  unsqueeze_1211 = None
    unsqueeze_1213: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, 3);  unsqueeze_1212 = None
    mul_1504: "f32[56]" = torch.ops.aten.mul.Tensor(sum_105, 0.0006377551020408163)
    mul_1505: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_292, squeeze_292)
    mul_1506: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1504, mul_1505);  mul_1504 = mul_1505 = None
    unsqueeze_1214: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1506, 0);  mul_1506 = None
    unsqueeze_1215: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 2);  unsqueeze_1214 = None
    unsqueeze_1216: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 3);  unsqueeze_1215 = None
    mul_1507: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_292, primals_293);  primals_293 = None
    unsqueeze_1217: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1507, 0);  mul_1507 = None
    unsqueeze_1218: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 2);  unsqueeze_1217 = None
    unsqueeze_1219: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 3);  unsqueeze_1218 = None
    mul_1508: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_353, unsqueeze_1216);  sub_353 = unsqueeze_1216 = None
    sub_355: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_50, mul_1508);  where_50 = mul_1508 = None
    sub_356: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_355, unsqueeze_1213);  sub_355 = unsqueeze_1213 = None
    mul_1509: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_1219);  sub_356 = unsqueeze_1219 = None
    mul_1510: "f32[56]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_292);  sum_105 = squeeze_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1509, add_538, primals_292, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1509 = add_538 = primals_292 = None
    getitem_1605: "f32[8, 56, 14, 14]" = convolution_backward_51[0]
    getitem_1606: "f32[56, 56, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_866: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_42, getitem_1605);  slice_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_51: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_51, full_default, add_866);  le_51 = add_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_106: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_357: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_1222);  convolution_96 = unsqueeze_1222 = None
    mul_1511: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_51, sub_357)
    sum_107: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1511, [0, 2, 3]);  mul_1511 = None
    mul_1512: "f32[56]" = torch.ops.aten.mul.Tensor(sum_106, 0.0006377551020408163)
    unsqueeze_1223: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1512, 0);  mul_1512 = None
    unsqueeze_1224: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1223, 2);  unsqueeze_1223 = None
    unsqueeze_1225: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, 3);  unsqueeze_1224 = None
    mul_1513: "f32[56]" = torch.ops.aten.mul.Tensor(sum_107, 0.0006377551020408163)
    mul_1514: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_289, squeeze_289)
    mul_1515: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1513, mul_1514);  mul_1513 = mul_1514 = None
    unsqueeze_1226: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1515, 0);  mul_1515 = None
    unsqueeze_1227: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 2);  unsqueeze_1226 = None
    unsqueeze_1228: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 3);  unsqueeze_1227 = None
    mul_1516: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_289, primals_290);  primals_290 = None
    unsqueeze_1229: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1516, 0);  mul_1516 = None
    unsqueeze_1230: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 2);  unsqueeze_1229 = None
    unsqueeze_1231: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 3);  unsqueeze_1230 = None
    mul_1517: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_357, unsqueeze_1228);  sub_357 = unsqueeze_1228 = None
    sub_359: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_51, mul_1517);  where_51 = mul_1517 = None
    sub_360: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_359, unsqueeze_1225);  sub_359 = unsqueeze_1225 = None
    mul_1518: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_1231);  sub_360 = unsqueeze_1231 = None
    mul_1519: "f32[56]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_289);  sum_107 = squeeze_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1518, add_532, primals_289, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1518 = add_532 = primals_289 = None
    getitem_1608: "f32[8, 56, 14, 14]" = convolution_backward_52[0]
    getitem_1609: "f32[56, 56, 3, 3]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_867: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_41, getitem_1608);  slice_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_52: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_52, full_default, add_867);  le_52 = add_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_108: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_361: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_1234);  convolution_95 = unsqueeze_1234 = None
    mul_1520: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_52, sub_361)
    sum_109: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1520, [0, 2, 3]);  mul_1520 = None
    mul_1521: "f32[56]" = torch.ops.aten.mul.Tensor(sum_108, 0.0006377551020408163)
    unsqueeze_1235: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1521, 0);  mul_1521 = None
    unsqueeze_1236: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1235, 2);  unsqueeze_1235 = None
    unsqueeze_1237: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, 3);  unsqueeze_1236 = None
    mul_1522: "f32[56]" = torch.ops.aten.mul.Tensor(sum_109, 0.0006377551020408163)
    mul_1523: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
    mul_1524: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1522, mul_1523);  mul_1522 = mul_1523 = None
    unsqueeze_1238: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1524, 0);  mul_1524 = None
    unsqueeze_1239: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 2);  unsqueeze_1238 = None
    unsqueeze_1240: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 3);  unsqueeze_1239 = None
    mul_1525: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_286, primals_287);  primals_287 = None
    unsqueeze_1241: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1525, 0);  mul_1525 = None
    unsqueeze_1242: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 2);  unsqueeze_1241 = None
    unsqueeze_1243: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 3);  unsqueeze_1242 = None
    mul_1526: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_361, unsqueeze_1240);  sub_361 = unsqueeze_1240 = None
    sub_363: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_52, mul_1526);  where_52 = mul_1526 = None
    sub_364: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_363, unsqueeze_1237);  sub_363 = unsqueeze_1237 = None
    mul_1527: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_1243);  sub_364 = unsqueeze_1243 = None
    mul_1528: "f32[56]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_286);  sum_109 = squeeze_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1527, getitem_920, primals_286, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1527 = getitem_920 = primals_286 = None
    getitem_1611: "f32[8, 56, 14, 14]" = convolution_backward_53[0]
    getitem_1612: "f32[56, 56, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_21: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([getitem_1611, getitem_1608, getitem_1605, getitem_1602, getitem_1599, getitem_1596, getitem_1593, slice_48], 1);  getitem_1611 = getitem_1608 = getitem_1605 = getitem_1602 = getitem_1599 = getitem_1596 = getitem_1593 = slice_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_53: "f32[8, 448, 14, 14]" = torch.ops.aten.where.self(le_53, full_default, cat_21);  le_53 = cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_110: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_365: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_1246);  convolution_94 = unsqueeze_1246 = None
    mul_1529: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(where_53, sub_365)
    sum_111: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_1529, [0, 2, 3]);  mul_1529 = None
    mul_1530: "f32[448]" = torch.ops.aten.mul.Tensor(sum_110, 0.0006377551020408163)
    unsqueeze_1247: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1530, 0);  mul_1530 = None
    unsqueeze_1248: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1247, 2);  unsqueeze_1247 = None
    unsqueeze_1249: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, 3);  unsqueeze_1248 = None
    mul_1531: "f32[448]" = torch.ops.aten.mul.Tensor(sum_111, 0.0006377551020408163)
    mul_1532: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
    mul_1533: "f32[448]" = torch.ops.aten.mul.Tensor(mul_1531, mul_1532);  mul_1531 = mul_1532 = None
    unsqueeze_1250: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1533, 0);  mul_1533 = None
    unsqueeze_1251: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, 2);  unsqueeze_1250 = None
    unsqueeze_1252: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 3);  unsqueeze_1251 = None
    mul_1534: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_283, primals_284);  primals_284 = None
    unsqueeze_1253: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1534, 0);  mul_1534 = None
    unsqueeze_1254: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 2);  unsqueeze_1253 = None
    unsqueeze_1255: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 3);  unsqueeze_1254 = None
    mul_1535: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_365, unsqueeze_1252);  sub_365 = unsqueeze_1252 = None
    sub_367: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(where_53, mul_1535);  where_53 = mul_1535 = None
    sub_368: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(sub_367, unsqueeze_1249);  sub_367 = unsqueeze_1249 = None
    mul_1536: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_1255);  sub_368 = unsqueeze_1255 = None
    mul_1537: "f32[448]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_283);  sum_111 = squeeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1536, relu_90, primals_283, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1536 = primals_283 = None
    getitem_1614: "f32[8, 1024, 14, 14]" = convolution_backward_54[0]
    getitem_1615: "f32[448, 1024, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_868: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_45, getitem_1614);  where_45 = getitem_1614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_308: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_90);  relu_90 = None
    alias_309: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_308);  alias_308 = None
    le_54: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_309, 0);  alias_309 = None
    where_54: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_54, full_default, add_868);  le_54 = add_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_112: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_369: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_1258);  convolution_93 = unsqueeze_1258 = None
    mul_1538: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_54, sub_369)
    sum_113: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1538, [0, 2, 3]);  mul_1538 = None
    mul_1539: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_112, 0.0006377551020408163)
    unsqueeze_1259: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1539, 0);  mul_1539 = None
    unsqueeze_1260: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1259, 2);  unsqueeze_1259 = None
    unsqueeze_1261: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, 3);  unsqueeze_1260 = None
    mul_1540: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_113, 0.0006377551020408163)
    mul_1541: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
    mul_1542: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1540, mul_1541);  mul_1540 = mul_1541 = None
    unsqueeze_1262: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1542, 0);  mul_1542 = None
    unsqueeze_1263: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, 2);  unsqueeze_1262 = None
    unsqueeze_1264: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 3);  unsqueeze_1263 = None
    mul_1543: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_280, primals_281);  primals_281 = None
    unsqueeze_1265: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1543, 0);  mul_1543 = None
    unsqueeze_1266: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 2);  unsqueeze_1265 = None
    unsqueeze_1267: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 3);  unsqueeze_1266 = None
    mul_1544: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_369, unsqueeze_1264);  sub_369 = unsqueeze_1264 = None
    sub_371: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_54, mul_1544);  mul_1544 = None
    sub_372: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_371, unsqueeze_1261);  sub_371 = unsqueeze_1261 = None
    mul_1545: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_1267);  sub_372 = unsqueeze_1267 = None
    mul_1546: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_280);  sum_113 = squeeze_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1545, cat_9, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1545 = cat_9 = primals_280 = None
    getitem_1617: "f32[8, 448, 14, 14]" = convolution_backward_55[0]
    getitem_1618: "f32[1024, 448, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_49: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1617, 1, 0, 56)
    slice_50: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1617, 1, 56, 112)
    slice_51: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1617, 1, 112, 168)
    slice_52: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1617, 1, 168, 224)
    slice_53: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1617, 1, 224, 280)
    slice_54: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1617, 1, 280, 336)
    slice_55: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1617, 1, 336, 392)
    slice_56: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1617, 1, 392, 448);  getitem_1617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_55: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_55, full_default, slice_55);  le_55 = slice_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_114: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_373: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_1270);  convolution_92 = unsqueeze_1270 = None
    mul_1547: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_55, sub_373)
    sum_115: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1547, [0, 2, 3]);  mul_1547 = None
    mul_1548: "f32[56]" = torch.ops.aten.mul.Tensor(sum_114, 0.0006377551020408163)
    unsqueeze_1271: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1548, 0);  mul_1548 = None
    unsqueeze_1272: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1271, 2);  unsqueeze_1271 = None
    unsqueeze_1273: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, 3);  unsqueeze_1272 = None
    mul_1549: "f32[56]" = torch.ops.aten.mul.Tensor(sum_115, 0.0006377551020408163)
    mul_1550: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
    mul_1551: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1549, mul_1550);  mul_1549 = mul_1550 = None
    unsqueeze_1274: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1551, 0);  mul_1551 = None
    unsqueeze_1275: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, 2);  unsqueeze_1274 = None
    unsqueeze_1276: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 3);  unsqueeze_1275 = None
    mul_1552: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_277, primals_278);  primals_278 = None
    unsqueeze_1277: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1552, 0);  mul_1552 = None
    unsqueeze_1278: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 2);  unsqueeze_1277 = None
    unsqueeze_1279: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 3);  unsqueeze_1278 = None
    mul_1553: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_373, unsqueeze_1276);  sub_373 = unsqueeze_1276 = None
    sub_375: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_55, mul_1553);  where_55 = mul_1553 = None
    sub_376: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_375, unsqueeze_1273);  sub_375 = unsqueeze_1273 = None
    mul_1554: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_1279);  sub_376 = unsqueeze_1279 = None
    mul_1555: "f32[56]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_277);  sum_115 = squeeze_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1554, add_510, primals_277, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1554 = add_510 = primals_277 = None
    getitem_1620: "f32[8, 56, 14, 14]" = convolution_backward_56[0]
    getitem_1621: "f32[56, 56, 3, 3]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_869: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_54, getitem_1620);  slice_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_56: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_56, full_default, add_869);  le_56 = add_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_116: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_377: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_1282);  convolution_91 = unsqueeze_1282 = None
    mul_1556: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_56, sub_377)
    sum_117: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1556, [0, 2, 3]);  mul_1556 = None
    mul_1557: "f32[56]" = torch.ops.aten.mul.Tensor(sum_116, 0.0006377551020408163)
    unsqueeze_1283: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1557, 0);  mul_1557 = None
    unsqueeze_1284: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1283, 2);  unsqueeze_1283 = None
    unsqueeze_1285: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, 3);  unsqueeze_1284 = None
    mul_1558: "f32[56]" = torch.ops.aten.mul.Tensor(sum_117, 0.0006377551020408163)
    mul_1559: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
    mul_1560: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1558, mul_1559);  mul_1558 = mul_1559 = None
    unsqueeze_1286: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1560, 0);  mul_1560 = None
    unsqueeze_1287: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, 2);  unsqueeze_1286 = None
    unsqueeze_1288: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1287, 3);  unsqueeze_1287 = None
    mul_1561: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_274, primals_275);  primals_275 = None
    unsqueeze_1289: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1561, 0);  mul_1561 = None
    unsqueeze_1290: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1289, 2);  unsqueeze_1289 = None
    unsqueeze_1291: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 3);  unsqueeze_1290 = None
    mul_1562: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_377, unsqueeze_1288);  sub_377 = unsqueeze_1288 = None
    sub_379: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_56, mul_1562);  where_56 = mul_1562 = None
    sub_380: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_379, unsqueeze_1285);  sub_379 = unsqueeze_1285 = None
    mul_1563: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_1291);  sub_380 = unsqueeze_1291 = None
    mul_1564: "f32[56]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_274);  sum_117 = squeeze_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1563, add_504, primals_274, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1563 = add_504 = primals_274 = None
    getitem_1623: "f32[8, 56, 14, 14]" = convolution_backward_57[0]
    getitem_1624: "f32[56, 56, 3, 3]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_870: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_53, getitem_1623);  slice_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_57: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_57, full_default, add_870);  le_57 = add_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_118: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_381: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_1294);  convolution_90 = unsqueeze_1294 = None
    mul_1565: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_57, sub_381)
    sum_119: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1565, [0, 2, 3]);  mul_1565 = None
    mul_1566: "f32[56]" = torch.ops.aten.mul.Tensor(sum_118, 0.0006377551020408163)
    unsqueeze_1295: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1566, 0);  mul_1566 = None
    unsqueeze_1296: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1295, 2);  unsqueeze_1295 = None
    unsqueeze_1297: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, 3);  unsqueeze_1296 = None
    mul_1567: "f32[56]" = torch.ops.aten.mul.Tensor(sum_119, 0.0006377551020408163)
    mul_1568: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
    mul_1569: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1567, mul_1568);  mul_1567 = mul_1568 = None
    unsqueeze_1298: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1569, 0);  mul_1569 = None
    unsqueeze_1299: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, 2);  unsqueeze_1298 = None
    unsqueeze_1300: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1299, 3);  unsqueeze_1299 = None
    mul_1570: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_271, primals_272);  primals_272 = None
    unsqueeze_1301: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1570, 0);  mul_1570 = None
    unsqueeze_1302: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1301, 2);  unsqueeze_1301 = None
    unsqueeze_1303: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 3);  unsqueeze_1302 = None
    mul_1571: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_381, unsqueeze_1300);  sub_381 = unsqueeze_1300 = None
    sub_383: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_57, mul_1571);  where_57 = mul_1571 = None
    sub_384: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_383, unsqueeze_1297);  sub_383 = unsqueeze_1297 = None
    mul_1572: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_1303);  sub_384 = unsqueeze_1303 = None
    mul_1573: "f32[56]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_271);  sum_119 = squeeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1572, add_498, primals_271, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1572 = add_498 = primals_271 = None
    getitem_1626: "f32[8, 56, 14, 14]" = convolution_backward_58[0]
    getitem_1627: "f32[56, 56, 3, 3]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_871: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_52, getitem_1626);  slice_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_58: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_58, full_default, add_871);  le_58 = add_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_120: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_385: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_1306);  convolution_89 = unsqueeze_1306 = None
    mul_1574: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_58, sub_385)
    sum_121: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1574, [0, 2, 3]);  mul_1574 = None
    mul_1575: "f32[56]" = torch.ops.aten.mul.Tensor(sum_120, 0.0006377551020408163)
    unsqueeze_1307: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1575, 0);  mul_1575 = None
    unsqueeze_1308: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1307, 2);  unsqueeze_1307 = None
    unsqueeze_1309: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, 3);  unsqueeze_1308 = None
    mul_1576: "f32[56]" = torch.ops.aten.mul.Tensor(sum_121, 0.0006377551020408163)
    mul_1577: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
    mul_1578: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1576, mul_1577);  mul_1576 = mul_1577 = None
    unsqueeze_1310: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1578, 0);  mul_1578 = None
    unsqueeze_1311: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, 2);  unsqueeze_1310 = None
    unsqueeze_1312: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1311, 3);  unsqueeze_1311 = None
    mul_1579: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_268, primals_269);  primals_269 = None
    unsqueeze_1313: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1579, 0);  mul_1579 = None
    unsqueeze_1314: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1313, 2);  unsqueeze_1313 = None
    unsqueeze_1315: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 3);  unsqueeze_1314 = None
    mul_1580: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_385, unsqueeze_1312);  sub_385 = unsqueeze_1312 = None
    sub_387: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_58, mul_1580);  where_58 = mul_1580 = None
    sub_388: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_387, unsqueeze_1309);  sub_387 = unsqueeze_1309 = None
    mul_1581: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_1315);  sub_388 = unsqueeze_1315 = None
    mul_1582: "f32[56]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_268);  sum_121 = squeeze_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1581, add_492, primals_268, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1581 = add_492 = primals_268 = None
    getitem_1629: "f32[8, 56, 14, 14]" = convolution_backward_59[0]
    getitem_1630: "f32[56, 56, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_872: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_51, getitem_1629);  slice_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_59: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_59, full_default, add_872);  le_59 = add_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_122: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_389: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_1318);  convolution_88 = unsqueeze_1318 = None
    mul_1583: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_59, sub_389)
    sum_123: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1583, [0, 2, 3]);  mul_1583 = None
    mul_1584: "f32[56]" = torch.ops.aten.mul.Tensor(sum_122, 0.0006377551020408163)
    unsqueeze_1319: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1584, 0);  mul_1584 = None
    unsqueeze_1320: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1319, 2);  unsqueeze_1319 = None
    unsqueeze_1321: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, 3);  unsqueeze_1320 = None
    mul_1585: "f32[56]" = torch.ops.aten.mul.Tensor(sum_123, 0.0006377551020408163)
    mul_1586: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
    mul_1587: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1585, mul_1586);  mul_1585 = mul_1586 = None
    unsqueeze_1322: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1587, 0);  mul_1587 = None
    unsqueeze_1323: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, 2);  unsqueeze_1322 = None
    unsqueeze_1324: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1323, 3);  unsqueeze_1323 = None
    mul_1588: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_265, primals_266);  primals_266 = None
    unsqueeze_1325: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1588, 0);  mul_1588 = None
    unsqueeze_1326: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1325, 2);  unsqueeze_1325 = None
    unsqueeze_1327: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 3);  unsqueeze_1326 = None
    mul_1589: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_389, unsqueeze_1324);  sub_389 = unsqueeze_1324 = None
    sub_391: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_59, mul_1589);  where_59 = mul_1589 = None
    sub_392: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_391, unsqueeze_1321);  sub_391 = unsqueeze_1321 = None
    mul_1590: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_1327);  sub_392 = unsqueeze_1327 = None
    mul_1591: "f32[56]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_265);  sum_123 = squeeze_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1590, add_486, primals_265, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1590 = add_486 = primals_265 = None
    getitem_1632: "f32[8, 56, 14, 14]" = convolution_backward_60[0]
    getitem_1633: "f32[56, 56, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_873: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_50, getitem_1632);  slice_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_60: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_60, full_default, add_873);  le_60 = add_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_124: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_393: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_1330);  convolution_87 = unsqueeze_1330 = None
    mul_1592: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_60, sub_393)
    sum_125: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1592, [0, 2, 3]);  mul_1592 = None
    mul_1593: "f32[56]" = torch.ops.aten.mul.Tensor(sum_124, 0.0006377551020408163)
    unsqueeze_1331: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1593, 0);  mul_1593 = None
    unsqueeze_1332: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1331, 2);  unsqueeze_1331 = None
    unsqueeze_1333: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, 3);  unsqueeze_1332 = None
    mul_1594: "f32[56]" = torch.ops.aten.mul.Tensor(sum_125, 0.0006377551020408163)
    mul_1595: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
    mul_1596: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1594, mul_1595);  mul_1594 = mul_1595 = None
    unsqueeze_1334: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1596, 0);  mul_1596 = None
    unsqueeze_1335: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, 2);  unsqueeze_1334 = None
    unsqueeze_1336: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1335, 3);  unsqueeze_1335 = None
    mul_1597: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_262, primals_263);  primals_263 = None
    unsqueeze_1337: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1597, 0);  mul_1597 = None
    unsqueeze_1338: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1337, 2);  unsqueeze_1337 = None
    unsqueeze_1339: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 3);  unsqueeze_1338 = None
    mul_1598: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_393, unsqueeze_1336);  sub_393 = unsqueeze_1336 = None
    sub_395: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_60, mul_1598);  where_60 = mul_1598 = None
    sub_396: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_395, unsqueeze_1333);  sub_395 = unsqueeze_1333 = None
    mul_1599: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_1339);  sub_396 = unsqueeze_1339 = None
    mul_1600: "f32[56]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_262);  sum_125 = squeeze_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1599, add_480, primals_262, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1599 = add_480 = primals_262 = None
    getitem_1635: "f32[8, 56, 14, 14]" = convolution_backward_61[0]
    getitem_1636: "f32[56, 56, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_874: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_49, getitem_1635);  slice_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_61: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_61, full_default, add_874);  le_61 = add_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_126: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_397: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_1342);  convolution_86 = unsqueeze_1342 = None
    mul_1601: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_61, sub_397)
    sum_127: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1601, [0, 2, 3]);  mul_1601 = None
    mul_1602: "f32[56]" = torch.ops.aten.mul.Tensor(sum_126, 0.0006377551020408163)
    unsqueeze_1343: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1602, 0);  mul_1602 = None
    unsqueeze_1344: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1343, 2);  unsqueeze_1343 = None
    unsqueeze_1345: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, 3);  unsqueeze_1344 = None
    mul_1603: "f32[56]" = torch.ops.aten.mul.Tensor(sum_127, 0.0006377551020408163)
    mul_1604: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
    mul_1605: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1603, mul_1604);  mul_1603 = mul_1604 = None
    unsqueeze_1346: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1605, 0);  mul_1605 = None
    unsqueeze_1347: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, 2);  unsqueeze_1346 = None
    unsqueeze_1348: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1347, 3);  unsqueeze_1347 = None
    mul_1606: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_259, primals_260);  primals_260 = None
    unsqueeze_1349: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1606, 0);  mul_1606 = None
    unsqueeze_1350: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1349, 2);  unsqueeze_1349 = None
    unsqueeze_1351: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 3);  unsqueeze_1350 = None
    mul_1607: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_397, unsqueeze_1348);  sub_397 = unsqueeze_1348 = None
    sub_399: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_61, mul_1607);  where_61 = mul_1607 = None
    sub_400: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_399, unsqueeze_1345);  sub_399 = unsqueeze_1345 = None
    mul_1608: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_1351);  sub_400 = unsqueeze_1351 = None
    mul_1609: "f32[56]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_259);  sum_127 = squeeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1608, getitem_830, primals_259, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1608 = getitem_830 = primals_259 = None
    getitem_1638: "f32[8, 56, 14, 14]" = convolution_backward_62[0]
    getitem_1639: "f32[56, 56, 3, 3]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_22: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([getitem_1638, getitem_1635, getitem_1632, getitem_1629, getitem_1626, getitem_1623, getitem_1620, slice_56], 1);  getitem_1638 = getitem_1635 = getitem_1632 = getitem_1629 = getitem_1626 = getitem_1623 = getitem_1620 = slice_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_62: "f32[8, 448, 14, 14]" = torch.ops.aten.where.self(le_62, full_default, cat_22);  le_62 = cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_128: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_401: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_1354);  convolution_85 = unsqueeze_1354 = None
    mul_1610: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(where_62, sub_401)
    sum_129: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_1610, [0, 2, 3]);  mul_1610 = None
    mul_1611: "f32[448]" = torch.ops.aten.mul.Tensor(sum_128, 0.0006377551020408163)
    unsqueeze_1355: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1611, 0);  mul_1611 = None
    unsqueeze_1356: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1355, 2);  unsqueeze_1355 = None
    unsqueeze_1357: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, 3);  unsqueeze_1356 = None
    mul_1612: "f32[448]" = torch.ops.aten.mul.Tensor(sum_129, 0.0006377551020408163)
    mul_1613: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
    mul_1614: "f32[448]" = torch.ops.aten.mul.Tensor(mul_1612, mul_1613);  mul_1612 = mul_1613 = None
    unsqueeze_1358: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1614, 0);  mul_1614 = None
    unsqueeze_1359: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, 2);  unsqueeze_1358 = None
    unsqueeze_1360: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1359, 3);  unsqueeze_1359 = None
    mul_1615: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_256, primals_257);  primals_257 = None
    unsqueeze_1361: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1615, 0);  mul_1615 = None
    unsqueeze_1362: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1361, 2);  unsqueeze_1361 = None
    unsqueeze_1363: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 3);  unsqueeze_1362 = None
    mul_1616: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_401, unsqueeze_1360);  sub_401 = unsqueeze_1360 = None
    sub_403: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(where_62, mul_1616);  where_62 = mul_1616 = None
    sub_404: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(sub_403, unsqueeze_1357);  sub_403 = unsqueeze_1357 = None
    mul_1617: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_404, unsqueeze_1363);  sub_404 = unsqueeze_1363 = None
    mul_1618: "f32[448]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_256);  sum_129 = squeeze_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1617, relu_81, primals_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1617 = primals_256 = None
    getitem_1641: "f32[8, 1024, 14, 14]" = convolution_backward_63[0]
    getitem_1642: "f32[448, 1024, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_875: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_54, getitem_1641);  where_54 = getitem_1641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_335: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_81);  relu_81 = None
    alias_336: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_335);  alias_335 = None
    le_63: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_336, 0);  alias_336 = None
    where_63: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_63, full_default, add_875);  le_63 = add_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_130: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_405: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_1366);  convolution_84 = unsqueeze_1366 = None
    mul_1619: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_63, sub_405)
    sum_131: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1619, [0, 2, 3]);  mul_1619 = None
    mul_1620: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_130, 0.0006377551020408163)
    unsqueeze_1367: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1620, 0);  mul_1620 = None
    unsqueeze_1368: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1367, 2);  unsqueeze_1367 = None
    unsqueeze_1369: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, 3);  unsqueeze_1368 = None
    mul_1621: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_131, 0.0006377551020408163)
    mul_1622: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
    mul_1623: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1621, mul_1622);  mul_1621 = mul_1622 = None
    unsqueeze_1370: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1623, 0);  mul_1623 = None
    unsqueeze_1371: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, 2);  unsqueeze_1370 = None
    unsqueeze_1372: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1371, 3);  unsqueeze_1371 = None
    mul_1624: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_254);  primals_254 = None
    unsqueeze_1373: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1624, 0);  mul_1624 = None
    unsqueeze_1374: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1373, 2);  unsqueeze_1373 = None
    unsqueeze_1375: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 3);  unsqueeze_1374 = None
    mul_1625: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_405, unsqueeze_1372);  sub_405 = unsqueeze_1372 = None
    sub_407: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_63, mul_1625);  mul_1625 = None
    sub_408: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_407, unsqueeze_1369);  sub_407 = unsqueeze_1369 = None
    mul_1626: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_408, unsqueeze_1375);  sub_408 = unsqueeze_1375 = None
    mul_1627: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_253);  sum_131 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1626, cat_8, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1626 = cat_8 = primals_253 = None
    getitem_1644: "f32[8, 448, 14, 14]" = convolution_backward_64[0]
    getitem_1645: "f32[1024, 448, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_57: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1644, 1, 0, 56)
    slice_58: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1644, 1, 56, 112)
    slice_59: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1644, 1, 112, 168)
    slice_60: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1644, 1, 168, 224)
    slice_61: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1644, 1, 224, 280)
    slice_62: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1644, 1, 280, 336)
    slice_63: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1644, 1, 336, 392)
    slice_64: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1644, 1, 392, 448);  getitem_1644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_64: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_64, full_default, slice_63);  le_64 = slice_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_132: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_409: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_1378);  convolution_83 = unsqueeze_1378 = None
    mul_1628: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_64, sub_409)
    sum_133: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1628, [0, 2, 3]);  mul_1628 = None
    mul_1629: "f32[56]" = torch.ops.aten.mul.Tensor(sum_132, 0.0006377551020408163)
    unsqueeze_1379: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1629, 0);  mul_1629 = None
    unsqueeze_1380: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1379, 2);  unsqueeze_1379 = None
    unsqueeze_1381: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, 3);  unsqueeze_1380 = None
    mul_1630: "f32[56]" = torch.ops.aten.mul.Tensor(sum_133, 0.0006377551020408163)
    mul_1631: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
    mul_1632: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1630, mul_1631);  mul_1630 = mul_1631 = None
    unsqueeze_1382: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1632, 0);  mul_1632 = None
    unsqueeze_1383: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, 2);  unsqueeze_1382 = None
    unsqueeze_1384: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1383, 3);  unsqueeze_1383 = None
    mul_1633: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_251);  primals_251 = None
    unsqueeze_1385: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1633, 0);  mul_1633 = None
    unsqueeze_1386: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1385, 2);  unsqueeze_1385 = None
    unsqueeze_1387: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 3);  unsqueeze_1386 = None
    mul_1634: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_409, unsqueeze_1384);  sub_409 = unsqueeze_1384 = None
    sub_411: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_64, mul_1634);  where_64 = mul_1634 = None
    sub_412: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_411, unsqueeze_1381);  sub_411 = unsqueeze_1381 = None
    mul_1635: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_412, unsqueeze_1387);  sub_412 = unsqueeze_1387 = None
    mul_1636: "f32[56]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_250);  sum_133 = squeeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1635, add_458, primals_250, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1635 = add_458 = primals_250 = None
    getitem_1647: "f32[8, 56, 14, 14]" = convolution_backward_65[0]
    getitem_1648: "f32[56, 56, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_876: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_62, getitem_1647);  slice_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_65: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_65, full_default, add_876);  le_65 = add_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_134: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_413: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_1390);  convolution_82 = unsqueeze_1390 = None
    mul_1637: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_65, sub_413)
    sum_135: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1637, [0, 2, 3]);  mul_1637 = None
    mul_1638: "f32[56]" = torch.ops.aten.mul.Tensor(sum_134, 0.0006377551020408163)
    unsqueeze_1391: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1638, 0);  mul_1638 = None
    unsqueeze_1392: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1391, 2);  unsqueeze_1391 = None
    unsqueeze_1393: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1392, 3);  unsqueeze_1392 = None
    mul_1639: "f32[56]" = torch.ops.aten.mul.Tensor(sum_135, 0.0006377551020408163)
    mul_1640: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
    mul_1641: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1639, mul_1640);  mul_1639 = mul_1640 = None
    unsqueeze_1394: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1641, 0);  mul_1641 = None
    unsqueeze_1395: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, 2);  unsqueeze_1394 = None
    unsqueeze_1396: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1395, 3);  unsqueeze_1395 = None
    mul_1642: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_248);  primals_248 = None
    unsqueeze_1397: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1642, 0);  mul_1642 = None
    unsqueeze_1398: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1397, 2);  unsqueeze_1397 = None
    unsqueeze_1399: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, 3);  unsqueeze_1398 = None
    mul_1643: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_413, unsqueeze_1396);  sub_413 = unsqueeze_1396 = None
    sub_415: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_65, mul_1643);  where_65 = mul_1643 = None
    sub_416: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_415, unsqueeze_1393);  sub_415 = unsqueeze_1393 = None
    mul_1644: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_416, unsqueeze_1399);  sub_416 = unsqueeze_1399 = None
    mul_1645: "f32[56]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_247);  sum_135 = squeeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1644, add_452, primals_247, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1644 = add_452 = primals_247 = None
    getitem_1650: "f32[8, 56, 14, 14]" = convolution_backward_66[0]
    getitem_1651: "f32[56, 56, 3, 3]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_877: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_61, getitem_1650);  slice_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_66: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_66, full_default, add_877);  le_66 = add_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_136: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_417: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_1402);  convolution_81 = unsqueeze_1402 = None
    mul_1646: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_66, sub_417)
    sum_137: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1646, [0, 2, 3]);  mul_1646 = None
    mul_1647: "f32[56]" = torch.ops.aten.mul.Tensor(sum_136, 0.0006377551020408163)
    unsqueeze_1403: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1647, 0);  mul_1647 = None
    unsqueeze_1404: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1403, 2);  unsqueeze_1403 = None
    unsqueeze_1405: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1404, 3);  unsqueeze_1404 = None
    mul_1648: "f32[56]" = torch.ops.aten.mul.Tensor(sum_137, 0.0006377551020408163)
    mul_1649: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
    mul_1650: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1648, mul_1649);  mul_1648 = mul_1649 = None
    unsqueeze_1406: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1650, 0);  mul_1650 = None
    unsqueeze_1407: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, 2);  unsqueeze_1406 = None
    unsqueeze_1408: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1407, 3);  unsqueeze_1407 = None
    mul_1651: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_245);  primals_245 = None
    unsqueeze_1409: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1651, 0);  mul_1651 = None
    unsqueeze_1410: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1409, 2);  unsqueeze_1409 = None
    unsqueeze_1411: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, 3);  unsqueeze_1410 = None
    mul_1652: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_417, unsqueeze_1408);  sub_417 = unsqueeze_1408 = None
    sub_419: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_66, mul_1652);  where_66 = mul_1652 = None
    sub_420: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_419, unsqueeze_1405);  sub_419 = unsqueeze_1405 = None
    mul_1653: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_420, unsqueeze_1411);  sub_420 = unsqueeze_1411 = None
    mul_1654: "f32[56]" = torch.ops.aten.mul.Tensor(sum_137, squeeze_244);  sum_137 = squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1653, add_446, primals_244, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1653 = add_446 = primals_244 = None
    getitem_1653: "f32[8, 56, 14, 14]" = convolution_backward_67[0]
    getitem_1654: "f32[56, 56, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_878: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_60, getitem_1653);  slice_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_67: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_67, full_default, add_878);  le_67 = add_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_138: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_421: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_1414);  convolution_80 = unsqueeze_1414 = None
    mul_1655: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_67, sub_421)
    sum_139: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1655, [0, 2, 3]);  mul_1655 = None
    mul_1656: "f32[56]" = torch.ops.aten.mul.Tensor(sum_138, 0.0006377551020408163)
    unsqueeze_1415: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1656, 0);  mul_1656 = None
    unsqueeze_1416: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1415, 2);  unsqueeze_1415 = None
    unsqueeze_1417: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1416, 3);  unsqueeze_1416 = None
    mul_1657: "f32[56]" = torch.ops.aten.mul.Tensor(sum_139, 0.0006377551020408163)
    mul_1658: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
    mul_1659: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1657, mul_1658);  mul_1657 = mul_1658 = None
    unsqueeze_1418: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1659, 0);  mul_1659 = None
    unsqueeze_1419: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, 2);  unsqueeze_1418 = None
    unsqueeze_1420: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1419, 3);  unsqueeze_1419 = None
    mul_1660: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_242);  primals_242 = None
    unsqueeze_1421: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1660, 0);  mul_1660 = None
    unsqueeze_1422: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1421, 2);  unsqueeze_1421 = None
    unsqueeze_1423: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, 3);  unsqueeze_1422 = None
    mul_1661: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_421, unsqueeze_1420);  sub_421 = unsqueeze_1420 = None
    sub_423: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_67, mul_1661);  where_67 = mul_1661 = None
    sub_424: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_423, unsqueeze_1417);  sub_423 = unsqueeze_1417 = None
    mul_1662: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_424, unsqueeze_1423);  sub_424 = unsqueeze_1423 = None
    mul_1663: "f32[56]" = torch.ops.aten.mul.Tensor(sum_139, squeeze_241);  sum_139 = squeeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1662, add_440, primals_241, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1662 = add_440 = primals_241 = None
    getitem_1656: "f32[8, 56, 14, 14]" = convolution_backward_68[0]
    getitem_1657: "f32[56, 56, 3, 3]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_879: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_59, getitem_1656);  slice_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_68: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_68, full_default, add_879);  le_68 = add_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_140: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_425: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_1426);  convolution_79 = unsqueeze_1426 = None
    mul_1664: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_68, sub_425)
    sum_141: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1664, [0, 2, 3]);  mul_1664 = None
    mul_1665: "f32[56]" = torch.ops.aten.mul.Tensor(sum_140, 0.0006377551020408163)
    unsqueeze_1427: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1665, 0);  mul_1665 = None
    unsqueeze_1428: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1427, 2);  unsqueeze_1427 = None
    unsqueeze_1429: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1428, 3);  unsqueeze_1428 = None
    mul_1666: "f32[56]" = torch.ops.aten.mul.Tensor(sum_141, 0.0006377551020408163)
    mul_1667: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_1668: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1666, mul_1667);  mul_1666 = mul_1667 = None
    unsqueeze_1430: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1668, 0);  mul_1668 = None
    unsqueeze_1431: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1430, 2);  unsqueeze_1430 = None
    unsqueeze_1432: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1431, 3);  unsqueeze_1431 = None
    mul_1669: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_239);  primals_239 = None
    unsqueeze_1433: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1669, 0);  mul_1669 = None
    unsqueeze_1434: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1433, 2);  unsqueeze_1433 = None
    unsqueeze_1435: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, 3);  unsqueeze_1434 = None
    mul_1670: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_425, unsqueeze_1432);  sub_425 = unsqueeze_1432 = None
    sub_427: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_68, mul_1670);  where_68 = mul_1670 = None
    sub_428: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_427, unsqueeze_1429);  sub_427 = unsqueeze_1429 = None
    mul_1671: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_428, unsqueeze_1435);  sub_428 = unsqueeze_1435 = None
    mul_1672: "f32[56]" = torch.ops.aten.mul.Tensor(sum_141, squeeze_238);  sum_141 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1671, add_434, primals_238, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1671 = add_434 = primals_238 = None
    getitem_1659: "f32[8, 56, 14, 14]" = convolution_backward_69[0]
    getitem_1660: "f32[56, 56, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_880: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_58, getitem_1659);  slice_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_69: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_69, full_default, add_880);  le_69 = add_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_142: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_429: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_1438);  convolution_78 = unsqueeze_1438 = None
    mul_1673: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_69, sub_429)
    sum_143: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1673, [0, 2, 3]);  mul_1673 = None
    mul_1674: "f32[56]" = torch.ops.aten.mul.Tensor(sum_142, 0.0006377551020408163)
    unsqueeze_1439: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1674, 0);  mul_1674 = None
    unsqueeze_1440: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1439, 2);  unsqueeze_1439 = None
    unsqueeze_1441: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1440, 3);  unsqueeze_1440 = None
    mul_1675: "f32[56]" = torch.ops.aten.mul.Tensor(sum_143, 0.0006377551020408163)
    mul_1676: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_1677: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1675, mul_1676);  mul_1675 = mul_1676 = None
    unsqueeze_1442: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1677, 0);  mul_1677 = None
    unsqueeze_1443: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1442, 2);  unsqueeze_1442 = None
    unsqueeze_1444: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1443, 3);  unsqueeze_1443 = None
    mul_1678: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_236);  primals_236 = None
    unsqueeze_1445: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1678, 0);  mul_1678 = None
    unsqueeze_1446: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1445, 2);  unsqueeze_1445 = None
    unsqueeze_1447: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, 3);  unsqueeze_1446 = None
    mul_1679: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_429, unsqueeze_1444);  sub_429 = unsqueeze_1444 = None
    sub_431: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_69, mul_1679);  where_69 = mul_1679 = None
    sub_432: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_431, unsqueeze_1441);  sub_431 = unsqueeze_1441 = None
    mul_1680: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_432, unsqueeze_1447);  sub_432 = unsqueeze_1447 = None
    mul_1681: "f32[56]" = torch.ops.aten.mul.Tensor(sum_143, squeeze_235);  sum_143 = squeeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1680, add_428, primals_235, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1680 = add_428 = primals_235 = None
    getitem_1662: "f32[8, 56, 14, 14]" = convolution_backward_70[0]
    getitem_1663: "f32[56, 56, 3, 3]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_881: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_57, getitem_1662);  slice_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_70: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_70, full_default, add_881);  le_70 = add_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_144: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_433: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_1450);  convolution_77 = unsqueeze_1450 = None
    mul_1682: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_70, sub_433)
    sum_145: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1682, [0, 2, 3]);  mul_1682 = None
    mul_1683: "f32[56]" = torch.ops.aten.mul.Tensor(sum_144, 0.0006377551020408163)
    unsqueeze_1451: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1683, 0);  mul_1683 = None
    unsqueeze_1452: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1451, 2);  unsqueeze_1451 = None
    unsqueeze_1453: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1452, 3);  unsqueeze_1452 = None
    mul_1684: "f32[56]" = torch.ops.aten.mul.Tensor(sum_145, 0.0006377551020408163)
    mul_1685: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_1686: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1684, mul_1685);  mul_1684 = mul_1685 = None
    unsqueeze_1454: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1686, 0);  mul_1686 = None
    unsqueeze_1455: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, 2);  unsqueeze_1454 = None
    unsqueeze_1456: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1455, 3);  unsqueeze_1455 = None
    mul_1687: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_233);  primals_233 = None
    unsqueeze_1457: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1687, 0);  mul_1687 = None
    unsqueeze_1458: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1457, 2);  unsqueeze_1457 = None
    unsqueeze_1459: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, 3);  unsqueeze_1458 = None
    mul_1688: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_433, unsqueeze_1456);  sub_433 = unsqueeze_1456 = None
    sub_435: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_70, mul_1688);  where_70 = mul_1688 = None
    sub_436: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_435, unsqueeze_1453);  sub_435 = unsqueeze_1453 = None
    mul_1689: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_436, unsqueeze_1459);  sub_436 = unsqueeze_1459 = None
    mul_1690: "f32[56]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_232);  sum_145 = squeeze_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1689, getitem_740, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1689 = getitem_740 = primals_232 = None
    getitem_1665: "f32[8, 56, 14, 14]" = convolution_backward_71[0]
    getitem_1666: "f32[56, 56, 3, 3]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_23: "f32[8, 448, 14, 14]" = torch.ops.aten.cat.default([getitem_1665, getitem_1662, getitem_1659, getitem_1656, getitem_1653, getitem_1650, getitem_1647, slice_64], 1);  getitem_1665 = getitem_1662 = getitem_1659 = getitem_1656 = getitem_1653 = getitem_1650 = getitem_1647 = slice_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_71: "f32[8, 448, 14, 14]" = torch.ops.aten.where.self(le_71, full_default, cat_23);  le_71 = cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_146: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_437: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_1462);  convolution_76 = unsqueeze_1462 = None
    mul_1691: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(where_71, sub_437)
    sum_147: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_1691, [0, 2, 3]);  mul_1691 = None
    mul_1692: "f32[448]" = torch.ops.aten.mul.Tensor(sum_146, 0.0006377551020408163)
    unsqueeze_1463: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1692, 0);  mul_1692 = None
    unsqueeze_1464: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1463, 2);  unsqueeze_1463 = None
    unsqueeze_1465: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1464, 3);  unsqueeze_1464 = None
    mul_1693: "f32[448]" = torch.ops.aten.mul.Tensor(sum_147, 0.0006377551020408163)
    mul_1694: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_1695: "f32[448]" = torch.ops.aten.mul.Tensor(mul_1693, mul_1694);  mul_1693 = mul_1694 = None
    unsqueeze_1466: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1695, 0);  mul_1695 = None
    unsqueeze_1467: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, 2);  unsqueeze_1466 = None
    unsqueeze_1468: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1467, 3);  unsqueeze_1467 = None
    mul_1696: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_230);  primals_230 = None
    unsqueeze_1469: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1696, 0);  mul_1696 = None
    unsqueeze_1470: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1469, 2);  unsqueeze_1469 = None
    unsqueeze_1471: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, 3);  unsqueeze_1470 = None
    mul_1697: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_437, unsqueeze_1468);  sub_437 = unsqueeze_1468 = None
    sub_439: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(where_71, mul_1697);  where_71 = mul_1697 = None
    sub_440: "f32[8, 448, 14, 14]" = torch.ops.aten.sub.Tensor(sub_439, unsqueeze_1465);  sub_439 = unsqueeze_1465 = None
    mul_1698: "f32[8, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_440, unsqueeze_1471);  sub_440 = unsqueeze_1471 = None
    mul_1699: "f32[448]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_229);  sum_147 = squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1698, relu_72, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1698 = primals_229 = None
    getitem_1668: "f32[8, 1024, 14, 14]" = convolution_backward_72[0]
    getitem_1669: "f32[448, 1024, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_882: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_63, getitem_1668);  where_63 = getitem_1668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_362: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_72);  relu_72 = None
    alias_363: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_362);  alias_362 = None
    le_72: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_363, 0);  alias_363 = None
    where_72: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_72, full_default, add_882);  le_72 = add_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    sum_148: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_441: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_1474);  convolution_75 = unsqueeze_1474 = None
    mul_1700: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_72, sub_441)
    sum_149: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1700, [0, 2, 3]);  mul_1700 = None
    mul_1701: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_148, 0.0006377551020408163)
    unsqueeze_1475: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1701, 0);  mul_1701 = None
    unsqueeze_1476: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1475, 2);  unsqueeze_1475 = None
    unsqueeze_1477: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1476, 3);  unsqueeze_1476 = None
    mul_1702: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_149, 0.0006377551020408163)
    mul_1703: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_1704: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1702, mul_1703);  mul_1702 = mul_1703 = None
    unsqueeze_1478: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1704, 0);  mul_1704 = None
    unsqueeze_1479: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, 2);  unsqueeze_1478 = None
    unsqueeze_1480: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1479, 3);  unsqueeze_1479 = None
    mul_1705: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_227);  primals_227 = None
    unsqueeze_1481: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1705, 0);  mul_1705 = None
    unsqueeze_1482: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1481, 2);  unsqueeze_1481 = None
    unsqueeze_1483: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, 3);  unsqueeze_1482 = None
    mul_1706: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_441, unsqueeze_1480);  sub_441 = unsqueeze_1480 = None
    sub_443: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_72, mul_1706);  mul_1706 = None
    sub_444: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_443, unsqueeze_1477);  sub_443 = None
    mul_1707: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_444, unsqueeze_1483);  sub_444 = unsqueeze_1483 = None
    mul_1708: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_149, squeeze_226);  sum_149 = squeeze_226 = None
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1707, relu_63, primals_226, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1707 = primals_226 = None
    getitem_1671: "f32[8, 512, 28, 28]" = convolution_backward_73[0]
    getitem_1672: "f32[1024, 512, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sub_445: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_1486);  convolution_74 = unsqueeze_1486 = None
    mul_1709: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_72, sub_445)
    sum_151: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1709, [0, 2, 3]);  mul_1709 = None
    mul_1711: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_151, 0.0006377551020408163)
    mul_1712: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_1713: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1711, mul_1712);  mul_1711 = mul_1712 = None
    unsqueeze_1490: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1713, 0);  mul_1713 = None
    unsqueeze_1491: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, 2);  unsqueeze_1490 = None
    unsqueeze_1492: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1491, 3);  unsqueeze_1491 = None
    mul_1714: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_224);  primals_224 = None
    unsqueeze_1493: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1714, 0);  mul_1714 = None
    unsqueeze_1494: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1493, 2);  unsqueeze_1493 = None
    unsqueeze_1495: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, 3);  unsqueeze_1494 = None
    mul_1715: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_445, unsqueeze_1492);  sub_445 = unsqueeze_1492 = None
    sub_447: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_72, mul_1715);  where_72 = mul_1715 = None
    sub_448: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_447, unsqueeze_1477);  sub_447 = unsqueeze_1477 = None
    mul_1716: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_448, unsqueeze_1495);  sub_448 = unsqueeze_1495 = None
    mul_1717: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_223);  sum_151 = squeeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1716, cat_7, primals_223, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1716 = cat_7 = primals_223 = None
    getitem_1674: "f32[8, 448, 14, 14]" = convolution_backward_74[0]
    getitem_1675: "f32[1024, 448, 1, 1]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_65: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1674, 1, 0, 56)
    slice_66: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1674, 1, 56, 112)
    slice_67: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1674, 1, 112, 168)
    slice_68: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1674, 1, 168, 224)
    slice_69: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1674, 1, 224, 280)
    slice_70: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1674, 1, 280, 336)
    slice_71: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1674, 1, 336, 392)
    slice_72: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_1674, 1, 392, 448);  getitem_1674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_backward_1: "f32[8, 56, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(slice_72, getitem_725, [3, 3], [2, 2], [1, 1], False, True, None);  slice_72 = getitem_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_73: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_73, full_default, slice_71);  le_73 = slice_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_152: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_449: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_1498);  convolution_73 = unsqueeze_1498 = None
    mul_1718: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_73, sub_449)
    sum_153: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1718, [0, 2, 3]);  mul_1718 = None
    mul_1719: "f32[56]" = torch.ops.aten.mul.Tensor(sum_152, 0.0006377551020408163)
    unsqueeze_1499: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1719, 0);  mul_1719 = None
    unsqueeze_1500: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1499, 2);  unsqueeze_1499 = None
    unsqueeze_1501: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1500, 3);  unsqueeze_1500 = None
    mul_1720: "f32[56]" = torch.ops.aten.mul.Tensor(sum_153, 0.0006377551020408163)
    mul_1721: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_1722: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1720, mul_1721);  mul_1720 = mul_1721 = None
    unsqueeze_1502: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1722, 0);  mul_1722 = None
    unsqueeze_1503: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1502, 2);  unsqueeze_1502 = None
    unsqueeze_1504: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1503, 3);  unsqueeze_1503 = None
    mul_1723: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_221);  primals_221 = None
    unsqueeze_1505: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1723, 0);  mul_1723 = None
    unsqueeze_1506: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1505, 2);  unsqueeze_1505 = None
    unsqueeze_1507: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, 3);  unsqueeze_1506 = None
    mul_1724: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_449, unsqueeze_1504);  sub_449 = unsqueeze_1504 = None
    sub_451: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_73, mul_1724);  where_73 = mul_1724 = None
    sub_452: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_451, unsqueeze_1501);  sub_451 = unsqueeze_1501 = None
    mul_1725: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_452, unsqueeze_1507);  sub_452 = unsqueeze_1507 = None
    mul_1726: "f32[56]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_220);  sum_153 = squeeze_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1725, getitem_714, primals_220, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1725 = getitem_714 = primals_220 = None
    getitem_1677: "f32[8, 56, 28, 28]" = convolution_backward_75[0]
    getitem_1678: "f32[56, 56, 3, 3]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_74: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_74, full_default, slice_70);  le_74 = slice_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_154: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_453: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_1510);  convolution_72 = unsqueeze_1510 = None
    mul_1727: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_74, sub_453)
    sum_155: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1727, [0, 2, 3]);  mul_1727 = None
    mul_1728: "f32[56]" = torch.ops.aten.mul.Tensor(sum_154, 0.0006377551020408163)
    unsqueeze_1511: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1728, 0);  mul_1728 = None
    unsqueeze_1512: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1511, 2);  unsqueeze_1511 = None
    unsqueeze_1513: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1512, 3);  unsqueeze_1512 = None
    mul_1729: "f32[56]" = torch.ops.aten.mul.Tensor(sum_155, 0.0006377551020408163)
    mul_1730: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_1731: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1729, mul_1730);  mul_1729 = mul_1730 = None
    unsqueeze_1514: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1731, 0);  mul_1731 = None
    unsqueeze_1515: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1514, 2);  unsqueeze_1514 = None
    unsqueeze_1516: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1515, 3);  unsqueeze_1515 = None
    mul_1732: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_218);  primals_218 = None
    unsqueeze_1517: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1732, 0);  mul_1732 = None
    unsqueeze_1518: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1517, 2);  unsqueeze_1517 = None
    unsqueeze_1519: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, 3);  unsqueeze_1518 = None
    mul_1733: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_453, unsqueeze_1516);  sub_453 = unsqueeze_1516 = None
    sub_455: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_74, mul_1733);  where_74 = mul_1733 = None
    sub_456: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_455, unsqueeze_1513);  sub_455 = unsqueeze_1513 = None
    mul_1734: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_456, unsqueeze_1519);  sub_456 = unsqueeze_1519 = None
    mul_1735: "f32[56]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_217);  sum_155 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1734, getitem_703, primals_217, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1734 = getitem_703 = primals_217 = None
    getitem_1680: "f32[8, 56, 28, 28]" = convolution_backward_76[0]
    getitem_1681: "f32[56, 56, 3, 3]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_75: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_75, full_default, slice_69);  le_75 = slice_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_156: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_457: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_1522);  convolution_71 = unsqueeze_1522 = None
    mul_1736: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_75, sub_457)
    sum_157: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1736, [0, 2, 3]);  mul_1736 = None
    mul_1737: "f32[56]" = torch.ops.aten.mul.Tensor(sum_156, 0.0006377551020408163)
    unsqueeze_1523: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1737, 0);  mul_1737 = None
    unsqueeze_1524: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1523, 2);  unsqueeze_1523 = None
    unsqueeze_1525: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1524, 3);  unsqueeze_1524 = None
    mul_1738: "f32[56]" = torch.ops.aten.mul.Tensor(sum_157, 0.0006377551020408163)
    mul_1739: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_1740: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1738, mul_1739);  mul_1738 = mul_1739 = None
    unsqueeze_1526: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1740, 0);  mul_1740 = None
    unsqueeze_1527: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1526, 2);  unsqueeze_1526 = None
    unsqueeze_1528: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1527, 3);  unsqueeze_1527 = None
    mul_1741: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_215);  primals_215 = None
    unsqueeze_1529: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1741, 0);  mul_1741 = None
    unsqueeze_1530: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1529, 2);  unsqueeze_1529 = None
    unsqueeze_1531: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, 3);  unsqueeze_1530 = None
    mul_1742: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_457, unsqueeze_1528);  sub_457 = unsqueeze_1528 = None
    sub_459: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_75, mul_1742);  where_75 = mul_1742 = None
    sub_460: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_459, unsqueeze_1525);  sub_459 = unsqueeze_1525 = None
    mul_1743: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_460, unsqueeze_1531);  sub_460 = unsqueeze_1531 = None
    mul_1744: "f32[56]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_214);  sum_157 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1743, getitem_692, primals_214, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1743 = getitem_692 = primals_214 = None
    getitem_1683: "f32[8, 56, 28, 28]" = convolution_backward_77[0]
    getitem_1684: "f32[56, 56, 3, 3]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_76: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_76, full_default, slice_68);  le_76 = slice_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_158: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_461: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_1534);  convolution_70 = unsqueeze_1534 = None
    mul_1745: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_76, sub_461)
    sum_159: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1745, [0, 2, 3]);  mul_1745 = None
    mul_1746: "f32[56]" = torch.ops.aten.mul.Tensor(sum_158, 0.0006377551020408163)
    unsqueeze_1535: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1746, 0);  mul_1746 = None
    unsqueeze_1536: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1535, 2);  unsqueeze_1535 = None
    unsqueeze_1537: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1536, 3);  unsqueeze_1536 = None
    mul_1747: "f32[56]" = torch.ops.aten.mul.Tensor(sum_159, 0.0006377551020408163)
    mul_1748: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_1749: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1747, mul_1748);  mul_1747 = mul_1748 = None
    unsqueeze_1538: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1749, 0);  mul_1749 = None
    unsqueeze_1539: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1538, 2);  unsqueeze_1538 = None
    unsqueeze_1540: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1539, 3);  unsqueeze_1539 = None
    mul_1750: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_212);  primals_212 = None
    unsqueeze_1541: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1750, 0);  mul_1750 = None
    unsqueeze_1542: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1541, 2);  unsqueeze_1541 = None
    unsqueeze_1543: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, 3);  unsqueeze_1542 = None
    mul_1751: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_461, unsqueeze_1540);  sub_461 = unsqueeze_1540 = None
    sub_463: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_76, mul_1751);  where_76 = mul_1751 = None
    sub_464: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_463, unsqueeze_1537);  sub_463 = unsqueeze_1537 = None
    mul_1752: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_464, unsqueeze_1543);  sub_464 = unsqueeze_1543 = None
    mul_1753: "f32[56]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_211);  sum_159 = squeeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1752, getitem_681, primals_211, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1752 = getitem_681 = primals_211 = None
    getitem_1686: "f32[8, 56, 28, 28]" = convolution_backward_78[0]
    getitem_1687: "f32[56, 56, 3, 3]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_77: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_77, full_default, slice_67);  le_77 = slice_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_160: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_465: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_1546);  convolution_69 = unsqueeze_1546 = None
    mul_1754: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_77, sub_465)
    sum_161: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1754, [0, 2, 3]);  mul_1754 = None
    mul_1755: "f32[56]" = torch.ops.aten.mul.Tensor(sum_160, 0.0006377551020408163)
    unsqueeze_1547: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1755, 0);  mul_1755 = None
    unsqueeze_1548: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1547, 2);  unsqueeze_1547 = None
    unsqueeze_1549: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1548, 3);  unsqueeze_1548 = None
    mul_1756: "f32[56]" = torch.ops.aten.mul.Tensor(sum_161, 0.0006377551020408163)
    mul_1757: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_1758: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1756, mul_1757);  mul_1756 = mul_1757 = None
    unsqueeze_1550: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1758, 0);  mul_1758 = None
    unsqueeze_1551: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1550, 2);  unsqueeze_1550 = None
    unsqueeze_1552: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1551, 3);  unsqueeze_1551 = None
    mul_1759: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_209);  primals_209 = None
    unsqueeze_1553: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1759, 0);  mul_1759 = None
    unsqueeze_1554: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1553, 2);  unsqueeze_1553 = None
    unsqueeze_1555: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, 3);  unsqueeze_1554 = None
    mul_1760: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_465, unsqueeze_1552);  sub_465 = unsqueeze_1552 = None
    sub_467: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_77, mul_1760);  where_77 = mul_1760 = None
    sub_468: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_467, unsqueeze_1549);  sub_467 = unsqueeze_1549 = None
    mul_1761: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_468, unsqueeze_1555);  sub_468 = unsqueeze_1555 = None
    mul_1762: "f32[56]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_208);  sum_161 = squeeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1761, getitem_670, primals_208, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1761 = getitem_670 = primals_208 = None
    getitem_1689: "f32[8, 56, 28, 28]" = convolution_backward_79[0]
    getitem_1690: "f32[56, 56, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_78: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_78, full_default, slice_66);  le_78 = slice_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_162: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_469: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_1558);  convolution_68 = unsqueeze_1558 = None
    mul_1763: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_78, sub_469)
    sum_163: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1763, [0, 2, 3]);  mul_1763 = None
    mul_1764: "f32[56]" = torch.ops.aten.mul.Tensor(sum_162, 0.0006377551020408163)
    unsqueeze_1559: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1764, 0);  mul_1764 = None
    unsqueeze_1560: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1559, 2);  unsqueeze_1559 = None
    unsqueeze_1561: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1560, 3);  unsqueeze_1560 = None
    mul_1765: "f32[56]" = torch.ops.aten.mul.Tensor(sum_163, 0.0006377551020408163)
    mul_1766: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_1767: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1765, mul_1766);  mul_1765 = mul_1766 = None
    unsqueeze_1562: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1767, 0);  mul_1767 = None
    unsqueeze_1563: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1562, 2);  unsqueeze_1562 = None
    unsqueeze_1564: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1563, 3);  unsqueeze_1563 = None
    mul_1768: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_206);  primals_206 = None
    unsqueeze_1565: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1768, 0);  mul_1768 = None
    unsqueeze_1566: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1565, 2);  unsqueeze_1565 = None
    unsqueeze_1567: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, 3);  unsqueeze_1566 = None
    mul_1769: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_469, unsqueeze_1564);  sub_469 = unsqueeze_1564 = None
    sub_471: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_78, mul_1769);  where_78 = mul_1769 = None
    sub_472: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_471, unsqueeze_1561);  sub_471 = unsqueeze_1561 = None
    mul_1770: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_472, unsqueeze_1567);  sub_472 = unsqueeze_1567 = None
    mul_1771: "f32[56]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_205);  sum_163 = squeeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1770, getitem_659, primals_205, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1770 = getitem_659 = primals_205 = None
    getitem_1692: "f32[8, 56, 28, 28]" = convolution_backward_80[0]
    getitem_1693: "f32[56, 56, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_79: "f32[8, 56, 14, 14]" = torch.ops.aten.where.self(le_79, full_default, slice_65);  le_79 = slice_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_164: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_473: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_1570);  convolution_67 = unsqueeze_1570 = None
    mul_1772: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(where_79, sub_473)
    sum_165: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1772, [0, 2, 3]);  mul_1772 = None
    mul_1773: "f32[56]" = torch.ops.aten.mul.Tensor(sum_164, 0.0006377551020408163)
    unsqueeze_1571: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1773, 0);  mul_1773 = None
    unsqueeze_1572: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1571, 2);  unsqueeze_1571 = None
    unsqueeze_1573: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1572, 3);  unsqueeze_1572 = None
    mul_1774: "f32[56]" = torch.ops.aten.mul.Tensor(sum_165, 0.0006377551020408163)
    mul_1775: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_1776: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1774, mul_1775);  mul_1774 = mul_1775 = None
    unsqueeze_1574: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1776, 0);  mul_1776 = None
    unsqueeze_1575: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1574, 2);  unsqueeze_1574 = None
    unsqueeze_1576: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1575, 3);  unsqueeze_1575 = None
    mul_1777: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_203);  primals_203 = None
    unsqueeze_1577: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1777, 0);  mul_1777 = None
    unsqueeze_1578: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1577, 2);  unsqueeze_1577 = None
    unsqueeze_1579: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, 3);  unsqueeze_1578 = None
    mul_1778: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_473, unsqueeze_1576);  sub_473 = unsqueeze_1576 = None
    sub_475: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(where_79, mul_1778);  where_79 = mul_1778 = None
    sub_476: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_475, unsqueeze_1573);  sub_475 = unsqueeze_1573 = None
    mul_1779: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_476, unsqueeze_1579);  sub_476 = unsqueeze_1579 = None
    mul_1780: "f32[56]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_202);  sum_165 = squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1779, getitem_648, primals_202, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1779 = getitem_648 = primals_202 = None
    getitem_1695: "f32[8, 56, 28, 28]" = convolution_backward_81[0]
    getitem_1696: "f32[56, 56, 3, 3]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_24: "f32[8, 448, 28, 28]" = torch.ops.aten.cat.default([getitem_1695, getitem_1692, getitem_1689, getitem_1686, getitem_1683, getitem_1680, getitem_1677, avg_pool2d_backward_1], 1);  getitem_1695 = getitem_1692 = getitem_1689 = getitem_1686 = getitem_1683 = getitem_1680 = getitem_1677 = avg_pool2d_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_80: "f32[8, 448, 28, 28]" = torch.ops.aten.where.self(le_80, full_default, cat_24);  le_80 = cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_166: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_477: "f32[8, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_1582);  convolution_66 = unsqueeze_1582 = None
    mul_1781: "f32[8, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_80, sub_477)
    sum_167: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_1781, [0, 2, 3]);  mul_1781 = None
    mul_1782: "f32[448]" = torch.ops.aten.mul.Tensor(sum_166, 0.00015943877551020407)
    unsqueeze_1583: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1782, 0);  mul_1782 = None
    unsqueeze_1584: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1583, 2);  unsqueeze_1583 = None
    unsqueeze_1585: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1584, 3);  unsqueeze_1584 = None
    mul_1783: "f32[448]" = torch.ops.aten.mul.Tensor(sum_167, 0.00015943877551020407)
    mul_1784: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_1785: "f32[448]" = torch.ops.aten.mul.Tensor(mul_1783, mul_1784);  mul_1783 = mul_1784 = None
    unsqueeze_1586: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1785, 0);  mul_1785 = None
    unsqueeze_1587: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1586, 2);  unsqueeze_1586 = None
    unsqueeze_1588: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1587, 3);  unsqueeze_1587 = None
    mul_1786: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_200);  primals_200 = None
    unsqueeze_1589: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1786, 0);  mul_1786 = None
    unsqueeze_1590: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1589, 2);  unsqueeze_1589 = None
    unsqueeze_1591: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, 3);  unsqueeze_1590 = None
    mul_1787: "f32[8, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_477, unsqueeze_1588);  sub_477 = unsqueeze_1588 = None
    sub_479: "f32[8, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_80, mul_1787);  where_80 = mul_1787 = None
    sub_480: "f32[8, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_479, unsqueeze_1585);  sub_479 = unsqueeze_1585 = None
    mul_1788: "f32[8, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_480, unsqueeze_1591);  sub_480 = unsqueeze_1591 = None
    mul_1789: "f32[448]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_199);  sum_167 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1788, relu_63, primals_199, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1788 = primals_199 = None
    getitem_1698: "f32[8, 512, 28, 28]" = convolution_backward_82[0]
    getitem_1699: "f32[448, 512, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_883: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(getitem_1671, getitem_1698);  getitem_1671 = getitem_1698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_389: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_63);  relu_63 = None
    alias_390: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_389);  alias_389 = None
    le_81: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_390, 0);  alias_390 = None
    where_81: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_81, full_default, add_883);  le_81 = add_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_168: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
    sub_481: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_1594);  convolution_65 = unsqueeze_1594 = None
    mul_1790: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_81, sub_481)
    sum_169: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1790, [0, 2, 3]);  mul_1790 = None
    mul_1791: "f32[512]" = torch.ops.aten.mul.Tensor(sum_168, 0.00015943877551020407)
    unsqueeze_1595: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1791, 0);  mul_1791 = None
    unsqueeze_1596: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1595, 2);  unsqueeze_1595 = None
    unsqueeze_1597: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1596, 3);  unsqueeze_1596 = None
    mul_1792: "f32[512]" = torch.ops.aten.mul.Tensor(sum_169, 0.00015943877551020407)
    mul_1793: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_1794: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1792, mul_1793);  mul_1792 = mul_1793 = None
    unsqueeze_1598: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1794, 0);  mul_1794 = None
    unsqueeze_1599: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1598, 2);  unsqueeze_1598 = None
    unsqueeze_1600: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1599, 3);  unsqueeze_1599 = None
    mul_1795: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_197);  primals_197 = None
    unsqueeze_1601: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1795, 0);  mul_1795 = None
    unsqueeze_1602: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1601, 2);  unsqueeze_1601 = None
    unsqueeze_1603: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, 3);  unsqueeze_1602 = None
    mul_1796: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_481, unsqueeze_1600);  sub_481 = unsqueeze_1600 = None
    sub_483: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_81, mul_1796);  mul_1796 = None
    sub_484: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_483, unsqueeze_1597);  sub_483 = unsqueeze_1597 = None
    mul_1797: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_484, unsqueeze_1603);  sub_484 = unsqueeze_1603 = None
    mul_1798: "f32[512]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_196);  sum_169 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1797, cat_6, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1797 = cat_6 = primals_196 = None
    getitem_1701: "f32[8, 224, 28, 28]" = convolution_backward_83[0]
    getitem_1702: "f32[512, 224, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_73: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1701, 1, 0, 28)
    slice_74: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1701, 1, 28, 56)
    slice_75: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1701, 1, 56, 84)
    slice_76: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1701, 1, 84, 112)
    slice_77: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1701, 1, 112, 140)
    slice_78: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1701, 1, 140, 168)
    slice_79: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1701, 1, 168, 196)
    slice_80: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1701, 1, 196, 224);  getitem_1701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_82: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_82, full_default, slice_79);  le_82 = slice_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_170: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
    sub_485: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_1606);  convolution_64 = unsqueeze_1606 = None
    mul_1799: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_82, sub_485)
    sum_171: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1799, [0, 2, 3]);  mul_1799 = None
    mul_1800: "f32[28]" = torch.ops.aten.mul.Tensor(sum_170, 0.00015943877551020407)
    unsqueeze_1607: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1800, 0);  mul_1800 = None
    unsqueeze_1608: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1607, 2);  unsqueeze_1607 = None
    unsqueeze_1609: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1608, 3);  unsqueeze_1608 = None
    mul_1801: "f32[28]" = torch.ops.aten.mul.Tensor(sum_171, 0.00015943877551020407)
    mul_1802: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_1803: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1801, mul_1802);  mul_1801 = mul_1802 = None
    unsqueeze_1610: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1803, 0);  mul_1803 = None
    unsqueeze_1611: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1610, 2);  unsqueeze_1610 = None
    unsqueeze_1612: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1611, 3);  unsqueeze_1611 = None
    mul_1804: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_194);  primals_194 = None
    unsqueeze_1613: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1804, 0);  mul_1804 = None
    unsqueeze_1614: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1613, 2);  unsqueeze_1613 = None
    unsqueeze_1615: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, 3);  unsqueeze_1614 = None
    mul_1805: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_485, unsqueeze_1612);  sub_485 = unsqueeze_1612 = None
    sub_487: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_82, mul_1805);  where_82 = mul_1805 = None
    sub_488: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_487, unsqueeze_1609);  sub_487 = unsqueeze_1609 = None
    mul_1806: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_488, unsqueeze_1615);  sub_488 = unsqueeze_1615 = None
    mul_1807: "f32[28]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_193);  sum_171 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1806, add_355, primals_193, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1806 = add_355 = primals_193 = None
    getitem_1704: "f32[8, 28, 28, 28]" = convolution_backward_84[0]
    getitem_1705: "f32[28, 28, 3, 3]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_884: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_78, getitem_1704);  slice_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_83: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_83, full_default, add_884);  le_83 = add_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_172: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
    sub_489: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_1618);  convolution_63 = unsqueeze_1618 = None
    mul_1808: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_83, sub_489)
    sum_173: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1808, [0, 2, 3]);  mul_1808 = None
    mul_1809: "f32[28]" = torch.ops.aten.mul.Tensor(sum_172, 0.00015943877551020407)
    unsqueeze_1619: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1809, 0);  mul_1809 = None
    unsqueeze_1620: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1619, 2);  unsqueeze_1619 = None
    unsqueeze_1621: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1620, 3);  unsqueeze_1620 = None
    mul_1810: "f32[28]" = torch.ops.aten.mul.Tensor(sum_173, 0.00015943877551020407)
    mul_1811: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_1812: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1810, mul_1811);  mul_1810 = mul_1811 = None
    unsqueeze_1622: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1812, 0);  mul_1812 = None
    unsqueeze_1623: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1622, 2);  unsqueeze_1622 = None
    unsqueeze_1624: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1623, 3);  unsqueeze_1623 = None
    mul_1813: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_191);  primals_191 = None
    unsqueeze_1625: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1813, 0);  mul_1813 = None
    unsqueeze_1626: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1625, 2);  unsqueeze_1625 = None
    unsqueeze_1627: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, 3);  unsqueeze_1626 = None
    mul_1814: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_489, unsqueeze_1624);  sub_489 = unsqueeze_1624 = None
    sub_491: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_83, mul_1814);  where_83 = mul_1814 = None
    sub_492: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_491, unsqueeze_1621);  sub_491 = unsqueeze_1621 = None
    mul_1815: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_492, unsqueeze_1627);  sub_492 = unsqueeze_1627 = None
    mul_1816: "f32[28]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_190);  sum_173 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1815, add_349, primals_190, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1815 = add_349 = primals_190 = None
    getitem_1707: "f32[8, 28, 28, 28]" = convolution_backward_85[0]
    getitem_1708: "f32[28, 28, 3, 3]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_885: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_77, getitem_1707);  slice_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_84: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_84, full_default, add_885);  le_84 = add_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_174: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
    sub_493: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_1630);  convolution_62 = unsqueeze_1630 = None
    mul_1817: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_84, sub_493)
    sum_175: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1817, [0, 2, 3]);  mul_1817 = None
    mul_1818: "f32[28]" = torch.ops.aten.mul.Tensor(sum_174, 0.00015943877551020407)
    unsqueeze_1631: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1818, 0);  mul_1818 = None
    unsqueeze_1632: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1631, 2);  unsqueeze_1631 = None
    unsqueeze_1633: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1632, 3);  unsqueeze_1632 = None
    mul_1819: "f32[28]" = torch.ops.aten.mul.Tensor(sum_175, 0.00015943877551020407)
    mul_1820: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_1821: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1819, mul_1820);  mul_1819 = mul_1820 = None
    unsqueeze_1634: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1821, 0);  mul_1821 = None
    unsqueeze_1635: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1634, 2);  unsqueeze_1634 = None
    unsqueeze_1636: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1635, 3);  unsqueeze_1635 = None
    mul_1822: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_188);  primals_188 = None
    unsqueeze_1637: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1822, 0);  mul_1822 = None
    unsqueeze_1638: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1637, 2);  unsqueeze_1637 = None
    unsqueeze_1639: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, 3);  unsqueeze_1638 = None
    mul_1823: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_493, unsqueeze_1636);  sub_493 = unsqueeze_1636 = None
    sub_495: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_84, mul_1823);  where_84 = mul_1823 = None
    sub_496: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_495, unsqueeze_1633);  sub_495 = unsqueeze_1633 = None
    mul_1824: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_496, unsqueeze_1639);  sub_496 = unsqueeze_1639 = None
    mul_1825: "f32[28]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_187);  sum_175 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1824, add_343, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1824 = add_343 = primals_187 = None
    getitem_1710: "f32[8, 28, 28, 28]" = convolution_backward_86[0]
    getitem_1711: "f32[28, 28, 3, 3]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_886: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_76, getitem_1710);  slice_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_85: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_85, full_default, add_886);  le_85 = add_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_176: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_497: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_1642);  convolution_61 = unsqueeze_1642 = None
    mul_1826: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_85, sub_497)
    sum_177: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1826, [0, 2, 3]);  mul_1826 = None
    mul_1827: "f32[28]" = torch.ops.aten.mul.Tensor(sum_176, 0.00015943877551020407)
    unsqueeze_1643: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1827, 0);  mul_1827 = None
    unsqueeze_1644: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1643, 2);  unsqueeze_1643 = None
    unsqueeze_1645: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1644, 3);  unsqueeze_1644 = None
    mul_1828: "f32[28]" = torch.ops.aten.mul.Tensor(sum_177, 0.00015943877551020407)
    mul_1829: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_1830: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1828, mul_1829);  mul_1828 = mul_1829 = None
    unsqueeze_1646: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1830, 0);  mul_1830 = None
    unsqueeze_1647: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1646, 2);  unsqueeze_1646 = None
    unsqueeze_1648: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1647, 3);  unsqueeze_1647 = None
    mul_1831: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_185);  primals_185 = None
    unsqueeze_1649: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1831, 0);  mul_1831 = None
    unsqueeze_1650: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1649, 2);  unsqueeze_1649 = None
    unsqueeze_1651: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, 3);  unsqueeze_1650 = None
    mul_1832: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_497, unsqueeze_1648);  sub_497 = unsqueeze_1648 = None
    sub_499: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_85, mul_1832);  where_85 = mul_1832 = None
    sub_500: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_499, unsqueeze_1645);  sub_499 = unsqueeze_1645 = None
    mul_1833: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_500, unsqueeze_1651);  sub_500 = unsqueeze_1651 = None
    mul_1834: "f32[28]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_184);  sum_177 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1833, add_337, primals_184, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1833 = add_337 = primals_184 = None
    getitem_1713: "f32[8, 28, 28, 28]" = convolution_backward_87[0]
    getitem_1714: "f32[28, 28, 3, 3]" = convolution_backward_87[1];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_887: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_75, getitem_1713);  slice_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_86: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_86, full_default, add_887);  le_86 = add_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_178: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
    sub_501: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_1654);  convolution_60 = unsqueeze_1654 = None
    mul_1835: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_86, sub_501)
    sum_179: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1835, [0, 2, 3]);  mul_1835 = None
    mul_1836: "f32[28]" = torch.ops.aten.mul.Tensor(sum_178, 0.00015943877551020407)
    unsqueeze_1655: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1836, 0);  mul_1836 = None
    unsqueeze_1656: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1655, 2);  unsqueeze_1655 = None
    unsqueeze_1657: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1656, 3);  unsqueeze_1656 = None
    mul_1837: "f32[28]" = torch.ops.aten.mul.Tensor(sum_179, 0.00015943877551020407)
    mul_1838: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_1839: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1837, mul_1838);  mul_1837 = mul_1838 = None
    unsqueeze_1658: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1839, 0);  mul_1839 = None
    unsqueeze_1659: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1658, 2);  unsqueeze_1658 = None
    unsqueeze_1660: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1659, 3);  unsqueeze_1659 = None
    mul_1840: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_182);  primals_182 = None
    unsqueeze_1661: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1840, 0);  mul_1840 = None
    unsqueeze_1662: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1661, 2);  unsqueeze_1661 = None
    unsqueeze_1663: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, 3);  unsqueeze_1662 = None
    mul_1841: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_501, unsqueeze_1660);  sub_501 = unsqueeze_1660 = None
    sub_503: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_86, mul_1841);  where_86 = mul_1841 = None
    sub_504: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_503, unsqueeze_1657);  sub_503 = unsqueeze_1657 = None
    mul_1842: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_504, unsqueeze_1663);  sub_504 = unsqueeze_1663 = None
    mul_1843: "f32[28]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_181);  sum_179 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1842, add_331, primals_181, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1842 = add_331 = primals_181 = None
    getitem_1716: "f32[8, 28, 28, 28]" = convolution_backward_88[0]
    getitem_1717: "f32[28, 28, 3, 3]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_888: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_74, getitem_1716);  slice_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_87: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_87, full_default, add_888);  le_87 = add_888 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_180: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_505: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_1666);  convolution_59 = unsqueeze_1666 = None
    mul_1844: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_87, sub_505)
    sum_181: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1844, [0, 2, 3]);  mul_1844 = None
    mul_1845: "f32[28]" = torch.ops.aten.mul.Tensor(sum_180, 0.00015943877551020407)
    unsqueeze_1667: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1845, 0);  mul_1845 = None
    unsqueeze_1668: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1667, 2);  unsqueeze_1667 = None
    unsqueeze_1669: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1668, 3);  unsqueeze_1668 = None
    mul_1846: "f32[28]" = torch.ops.aten.mul.Tensor(sum_181, 0.00015943877551020407)
    mul_1847: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_1848: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1846, mul_1847);  mul_1846 = mul_1847 = None
    unsqueeze_1670: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1848, 0);  mul_1848 = None
    unsqueeze_1671: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1670, 2);  unsqueeze_1670 = None
    unsqueeze_1672: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1671, 3);  unsqueeze_1671 = None
    mul_1849: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_179);  primals_179 = None
    unsqueeze_1673: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1849, 0);  mul_1849 = None
    unsqueeze_1674: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1673, 2);  unsqueeze_1673 = None
    unsqueeze_1675: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1674, 3);  unsqueeze_1674 = None
    mul_1850: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_505, unsqueeze_1672);  sub_505 = unsqueeze_1672 = None
    sub_507: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_87, mul_1850);  where_87 = mul_1850 = None
    sub_508: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_507, unsqueeze_1669);  sub_507 = unsqueeze_1669 = None
    mul_1851: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_508, unsqueeze_1675);  sub_508 = unsqueeze_1675 = None
    mul_1852: "f32[28]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_178);  sum_181 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1851, add_325, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1851 = add_325 = primals_178 = None
    getitem_1719: "f32[8, 28, 28, 28]" = convolution_backward_89[0]
    getitem_1720: "f32[28, 28, 3, 3]" = convolution_backward_89[1];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_889: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_73, getitem_1719);  slice_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_88: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_88, full_default, add_889);  le_88 = add_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_182: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_509: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_1678);  convolution_58 = unsqueeze_1678 = None
    mul_1853: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_88, sub_509)
    sum_183: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1853, [0, 2, 3]);  mul_1853 = None
    mul_1854: "f32[28]" = torch.ops.aten.mul.Tensor(sum_182, 0.00015943877551020407)
    unsqueeze_1679: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1854, 0);  mul_1854 = None
    unsqueeze_1680: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1679, 2);  unsqueeze_1679 = None
    unsqueeze_1681: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1680, 3);  unsqueeze_1680 = None
    mul_1855: "f32[28]" = torch.ops.aten.mul.Tensor(sum_183, 0.00015943877551020407)
    mul_1856: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_1857: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1855, mul_1856);  mul_1855 = mul_1856 = None
    unsqueeze_1682: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1857, 0);  mul_1857 = None
    unsqueeze_1683: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1682, 2);  unsqueeze_1682 = None
    unsqueeze_1684: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1683, 3);  unsqueeze_1683 = None
    mul_1858: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_176);  primals_176 = None
    unsqueeze_1685: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1858, 0);  mul_1858 = None
    unsqueeze_1686: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1685, 2);  unsqueeze_1685 = None
    unsqueeze_1687: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1686, 3);  unsqueeze_1686 = None
    mul_1859: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_509, unsqueeze_1684);  sub_509 = unsqueeze_1684 = None
    sub_511: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_88, mul_1859);  where_88 = mul_1859 = None
    sub_512: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_511, unsqueeze_1681);  sub_511 = unsqueeze_1681 = None
    mul_1860: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_512, unsqueeze_1687);  sub_512 = unsqueeze_1687 = None
    mul_1861: "f32[28]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_175);  sum_183 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1860, getitem_558, primals_175, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1860 = getitem_558 = primals_175 = None
    getitem_1722: "f32[8, 28, 28, 28]" = convolution_backward_90[0]
    getitem_1723: "f32[28, 28, 3, 3]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_25: "f32[8, 224, 28, 28]" = torch.ops.aten.cat.default([getitem_1722, getitem_1719, getitem_1716, getitem_1713, getitem_1710, getitem_1707, getitem_1704, slice_80], 1);  getitem_1722 = getitem_1719 = getitem_1716 = getitem_1713 = getitem_1710 = getitem_1707 = getitem_1704 = slice_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_89: "f32[8, 224, 28, 28]" = torch.ops.aten.where.self(le_89, full_default, cat_25);  le_89 = cat_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_184: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
    sub_513: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_1690);  convolution_57 = unsqueeze_1690 = None
    mul_1862: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(where_89, sub_513)
    sum_185: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_1862, [0, 2, 3]);  mul_1862 = None
    mul_1863: "f32[224]" = torch.ops.aten.mul.Tensor(sum_184, 0.00015943877551020407)
    unsqueeze_1691: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1863, 0);  mul_1863 = None
    unsqueeze_1692: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1691, 2);  unsqueeze_1691 = None
    unsqueeze_1693: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1692, 3);  unsqueeze_1692 = None
    mul_1864: "f32[224]" = torch.ops.aten.mul.Tensor(sum_185, 0.00015943877551020407)
    mul_1865: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_1866: "f32[224]" = torch.ops.aten.mul.Tensor(mul_1864, mul_1865);  mul_1864 = mul_1865 = None
    unsqueeze_1694: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1866, 0);  mul_1866 = None
    unsqueeze_1695: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1694, 2);  unsqueeze_1694 = None
    unsqueeze_1696: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1695, 3);  unsqueeze_1695 = None
    mul_1867: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_173);  primals_173 = None
    unsqueeze_1697: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1867, 0);  mul_1867 = None
    unsqueeze_1698: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1697, 2);  unsqueeze_1697 = None
    unsqueeze_1699: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1698, 3);  unsqueeze_1698 = None
    mul_1868: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_513, unsqueeze_1696);  sub_513 = unsqueeze_1696 = None
    sub_515: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(where_89, mul_1868);  where_89 = mul_1868 = None
    sub_516: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(sub_515, unsqueeze_1693);  sub_515 = unsqueeze_1693 = None
    mul_1869: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_516, unsqueeze_1699);  sub_516 = unsqueeze_1699 = None
    mul_1870: "f32[224]" = torch.ops.aten.mul.Tensor(sum_185, squeeze_172);  sum_185 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1869, relu_54, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1869 = primals_172 = None
    getitem_1725: "f32[8, 512, 28, 28]" = convolution_backward_91[0]
    getitem_1726: "f32[224, 512, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_890: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_81, getitem_1725);  where_81 = getitem_1725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_416: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_54);  relu_54 = None
    alias_417: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_416);  alias_416 = None
    le_90: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_417, 0);  alias_417 = None
    where_90: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_90, full_default, add_890);  le_90 = add_890 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_186: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
    sub_517: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_1702);  convolution_56 = unsqueeze_1702 = None
    mul_1871: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_90, sub_517)
    sum_187: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1871, [0, 2, 3]);  mul_1871 = None
    mul_1872: "f32[512]" = torch.ops.aten.mul.Tensor(sum_186, 0.00015943877551020407)
    unsqueeze_1703: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1872, 0);  mul_1872 = None
    unsqueeze_1704: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1703, 2);  unsqueeze_1703 = None
    unsqueeze_1705: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1704, 3);  unsqueeze_1704 = None
    mul_1873: "f32[512]" = torch.ops.aten.mul.Tensor(sum_187, 0.00015943877551020407)
    mul_1874: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_1875: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1873, mul_1874);  mul_1873 = mul_1874 = None
    unsqueeze_1706: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1875, 0);  mul_1875 = None
    unsqueeze_1707: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1706, 2);  unsqueeze_1706 = None
    unsqueeze_1708: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1707, 3);  unsqueeze_1707 = None
    mul_1876: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_170);  primals_170 = None
    unsqueeze_1709: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1876, 0);  mul_1876 = None
    unsqueeze_1710: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1709, 2);  unsqueeze_1709 = None
    unsqueeze_1711: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1710, 3);  unsqueeze_1710 = None
    mul_1877: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_517, unsqueeze_1708);  sub_517 = unsqueeze_1708 = None
    sub_519: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_90, mul_1877);  mul_1877 = None
    sub_520: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_519, unsqueeze_1705);  sub_519 = unsqueeze_1705 = None
    mul_1878: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_520, unsqueeze_1711);  sub_520 = unsqueeze_1711 = None
    mul_1879: "f32[512]" = torch.ops.aten.mul.Tensor(sum_187, squeeze_169);  sum_187 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1878, cat_5, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1878 = cat_5 = primals_169 = None
    getitem_1728: "f32[8, 224, 28, 28]" = convolution_backward_92[0]
    getitem_1729: "f32[512, 224, 1, 1]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_81: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1728, 1, 0, 28)
    slice_82: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1728, 1, 28, 56)
    slice_83: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1728, 1, 56, 84)
    slice_84: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1728, 1, 84, 112)
    slice_85: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1728, 1, 112, 140)
    slice_86: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1728, 1, 140, 168)
    slice_87: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1728, 1, 168, 196)
    slice_88: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1728, 1, 196, 224);  getitem_1728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_91: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_91, full_default, slice_87);  le_91 = slice_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_188: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
    sub_521: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_1714);  convolution_55 = unsqueeze_1714 = None
    mul_1880: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_91, sub_521)
    sum_189: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1880, [0, 2, 3]);  mul_1880 = None
    mul_1881: "f32[28]" = torch.ops.aten.mul.Tensor(sum_188, 0.00015943877551020407)
    unsqueeze_1715: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1881, 0);  mul_1881 = None
    unsqueeze_1716: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1715, 2);  unsqueeze_1715 = None
    unsqueeze_1717: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1716, 3);  unsqueeze_1716 = None
    mul_1882: "f32[28]" = torch.ops.aten.mul.Tensor(sum_189, 0.00015943877551020407)
    mul_1883: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_1884: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1882, mul_1883);  mul_1882 = mul_1883 = None
    unsqueeze_1718: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1884, 0);  mul_1884 = None
    unsqueeze_1719: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1718, 2);  unsqueeze_1718 = None
    unsqueeze_1720: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1719, 3);  unsqueeze_1719 = None
    mul_1885: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_167);  primals_167 = None
    unsqueeze_1721: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1885, 0);  mul_1885 = None
    unsqueeze_1722: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1721, 2);  unsqueeze_1721 = None
    unsqueeze_1723: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1722, 3);  unsqueeze_1722 = None
    mul_1886: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_521, unsqueeze_1720);  sub_521 = unsqueeze_1720 = None
    sub_523: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_91, mul_1886);  where_91 = mul_1886 = None
    sub_524: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_523, unsqueeze_1717);  sub_523 = unsqueeze_1717 = None
    mul_1887: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_524, unsqueeze_1723);  sub_524 = unsqueeze_1723 = None
    mul_1888: "f32[28]" = torch.ops.aten.mul.Tensor(sum_189, squeeze_166);  sum_189 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1887, add_303, primals_166, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1887 = add_303 = primals_166 = None
    getitem_1731: "f32[8, 28, 28, 28]" = convolution_backward_93[0]
    getitem_1732: "f32[28, 28, 3, 3]" = convolution_backward_93[1];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_891: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_86, getitem_1731);  slice_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_92: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_92, full_default, add_891);  le_92 = add_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_190: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_525: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_1726);  convolution_54 = unsqueeze_1726 = None
    mul_1889: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_92, sub_525)
    sum_191: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1889, [0, 2, 3]);  mul_1889 = None
    mul_1890: "f32[28]" = torch.ops.aten.mul.Tensor(sum_190, 0.00015943877551020407)
    unsqueeze_1727: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1890, 0);  mul_1890 = None
    unsqueeze_1728: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1727, 2);  unsqueeze_1727 = None
    unsqueeze_1729: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1728, 3);  unsqueeze_1728 = None
    mul_1891: "f32[28]" = torch.ops.aten.mul.Tensor(sum_191, 0.00015943877551020407)
    mul_1892: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_1893: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1891, mul_1892);  mul_1891 = mul_1892 = None
    unsqueeze_1730: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1893, 0);  mul_1893 = None
    unsqueeze_1731: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1730, 2);  unsqueeze_1730 = None
    unsqueeze_1732: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1731, 3);  unsqueeze_1731 = None
    mul_1894: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_164);  primals_164 = None
    unsqueeze_1733: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1894, 0);  mul_1894 = None
    unsqueeze_1734: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1733, 2);  unsqueeze_1733 = None
    unsqueeze_1735: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1734, 3);  unsqueeze_1734 = None
    mul_1895: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_525, unsqueeze_1732);  sub_525 = unsqueeze_1732 = None
    sub_527: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_92, mul_1895);  where_92 = mul_1895 = None
    sub_528: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_527, unsqueeze_1729);  sub_527 = unsqueeze_1729 = None
    mul_1896: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_528, unsqueeze_1735);  sub_528 = unsqueeze_1735 = None
    mul_1897: "f32[28]" = torch.ops.aten.mul.Tensor(sum_191, squeeze_163);  sum_191 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1896, add_297, primals_163, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1896 = add_297 = primals_163 = None
    getitem_1734: "f32[8, 28, 28, 28]" = convolution_backward_94[0]
    getitem_1735: "f32[28, 28, 3, 3]" = convolution_backward_94[1];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_892: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_85, getitem_1734);  slice_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_93: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_93, full_default, add_892);  le_93 = add_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_192: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
    sub_529: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_1738);  convolution_53 = unsqueeze_1738 = None
    mul_1898: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_93, sub_529)
    sum_193: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1898, [0, 2, 3]);  mul_1898 = None
    mul_1899: "f32[28]" = torch.ops.aten.mul.Tensor(sum_192, 0.00015943877551020407)
    unsqueeze_1739: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1899, 0);  mul_1899 = None
    unsqueeze_1740: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1739, 2);  unsqueeze_1739 = None
    unsqueeze_1741: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1740, 3);  unsqueeze_1740 = None
    mul_1900: "f32[28]" = torch.ops.aten.mul.Tensor(sum_193, 0.00015943877551020407)
    mul_1901: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_1902: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1900, mul_1901);  mul_1900 = mul_1901 = None
    unsqueeze_1742: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1902, 0);  mul_1902 = None
    unsqueeze_1743: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1742, 2);  unsqueeze_1742 = None
    unsqueeze_1744: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1743, 3);  unsqueeze_1743 = None
    mul_1903: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_161);  primals_161 = None
    unsqueeze_1745: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1903, 0);  mul_1903 = None
    unsqueeze_1746: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1745, 2);  unsqueeze_1745 = None
    unsqueeze_1747: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1746, 3);  unsqueeze_1746 = None
    mul_1904: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_529, unsqueeze_1744);  sub_529 = unsqueeze_1744 = None
    sub_531: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_93, mul_1904);  where_93 = mul_1904 = None
    sub_532: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_531, unsqueeze_1741);  sub_531 = unsqueeze_1741 = None
    mul_1905: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_532, unsqueeze_1747);  sub_532 = unsqueeze_1747 = None
    mul_1906: "f32[28]" = torch.ops.aten.mul.Tensor(sum_193, squeeze_160);  sum_193 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1905, add_291, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1905 = add_291 = primals_160 = None
    getitem_1737: "f32[8, 28, 28, 28]" = convolution_backward_95[0]
    getitem_1738: "f32[28, 28, 3, 3]" = convolution_backward_95[1];  convolution_backward_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_893: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_84, getitem_1737);  slice_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_94: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_94, full_default, add_893);  le_94 = add_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_194: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_533: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_1750);  convolution_52 = unsqueeze_1750 = None
    mul_1907: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_94, sub_533)
    sum_195: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1907, [0, 2, 3]);  mul_1907 = None
    mul_1908: "f32[28]" = torch.ops.aten.mul.Tensor(sum_194, 0.00015943877551020407)
    unsqueeze_1751: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1908, 0);  mul_1908 = None
    unsqueeze_1752: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1751, 2);  unsqueeze_1751 = None
    unsqueeze_1753: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1752, 3);  unsqueeze_1752 = None
    mul_1909: "f32[28]" = torch.ops.aten.mul.Tensor(sum_195, 0.00015943877551020407)
    mul_1910: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_1911: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1909, mul_1910);  mul_1909 = mul_1910 = None
    unsqueeze_1754: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1911, 0);  mul_1911 = None
    unsqueeze_1755: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1754, 2);  unsqueeze_1754 = None
    unsqueeze_1756: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1755, 3);  unsqueeze_1755 = None
    mul_1912: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_158);  primals_158 = None
    unsqueeze_1757: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1912, 0);  mul_1912 = None
    unsqueeze_1758: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1757, 2);  unsqueeze_1757 = None
    unsqueeze_1759: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1758, 3);  unsqueeze_1758 = None
    mul_1913: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_533, unsqueeze_1756);  sub_533 = unsqueeze_1756 = None
    sub_535: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_94, mul_1913);  where_94 = mul_1913 = None
    sub_536: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_535, unsqueeze_1753);  sub_535 = unsqueeze_1753 = None
    mul_1914: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_536, unsqueeze_1759);  sub_536 = unsqueeze_1759 = None
    mul_1915: "f32[28]" = torch.ops.aten.mul.Tensor(sum_195, squeeze_157);  sum_195 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(mul_1914, add_285, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1914 = add_285 = primals_157 = None
    getitem_1740: "f32[8, 28, 28, 28]" = convolution_backward_96[0]
    getitem_1741: "f32[28, 28, 3, 3]" = convolution_backward_96[1];  convolution_backward_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_894: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_83, getitem_1740);  slice_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_95: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_95, full_default, add_894);  le_95 = add_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_196: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
    sub_537: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_1762);  convolution_51 = unsqueeze_1762 = None
    mul_1916: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_95, sub_537)
    sum_197: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1916, [0, 2, 3]);  mul_1916 = None
    mul_1917: "f32[28]" = torch.ops.aten.mul.Tensor(sum_196, 0.00015943877551020407)
    unsqueeze_1763: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1917, 0);  mul_1917 = None
    unsqueeze_1764: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1763, 2);  unsqueeze_1763 = None
    unsqueeze_1765: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1764, 3);  unsqueeze_1764 = None
    mul_1918: "f32[28]" = torch.ops.aten.mul.Tensor(sum_197, 0.00015943877551020407)
    mul_1919: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_1920: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1918, mul_1919);  mul_1918 = mul_1919 = None
    unsqueeze_1766: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1920, 0);  mul_1920 = None
    unsqueeze_1767: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1766, 2);  unsqueeze_1766 = None
    unsqueeze_1768: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1767, 3);  unsqueeze_1767 = None
    mul_1921: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_155);  primals_155 = None
    unsqueeze_1769: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1921, 0);  mul_1921 = None
    unsqueeze_1770: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1769, 2);  unsqueeze_1769 = None
    unsqueeze_1771: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1770, 3);  unsqueeze_1770 = None
    mul_1922: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_537, unsqueeze_1768);  sub_537 = unsqueeze_1768 = None
    sub_539: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_95, mul_1922);  where_95 = mul_1922 = None
    sub_540: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_539, unsqueeze_1765);  sub_539 = unsqueeze_1765 = None
    mul_1923: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_540, unsqueeze_1771);  sub_540 = unsqueeze_1771 = None
    mul_1924: "f32[28]" = torch.ops.aten.mul.Tensor(sum_197, squeeze_154);  sum_197 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(mul_1923, add_279, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1923 = add_279 = primals_154 = None
    getitem_1743: "f32[8, 28, 28, 28]" = convolution_backward_97[0]
    getitem_1744: "f32[28, 28, 3, 3]" = convolution_backward_97[1];  convolution_backward_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_895: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_82, getitem_1743);  slice_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_96: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_96, full_default, add_895);  le_96 = add_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_198: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_541: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_1774);  convolution_50 = unsqueeze_1774 = None
    mul_1925: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_96, sub_541)
    sum_199: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1925, [0, 2, 3]);  mul_1925 = None
    mul_1926: "f32[28]" = torch.ops.aten.mul.Tensor(sum_198, 0.00015943877551020407)
    unsqueeze_1775: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1926, 0);  mul_1926 = None
    unsqueeze_1776: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1775, 2);  unsqueeze_1775 = None
    unsqueeze_1777: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1776, 3);  unsqueeze_1776 = None
    mul_1927: "f32[28]" = torch.ops.aten.mul.Tensor(sum_199, 0.00015943877551020407)
    mul_1928: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_1929: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1927, mul_1928);  mul_1927 = mul_1928 = None
    unsqueeze_1778: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1929, 0);  mul_1929 = None
    unsqueeze_1779: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1778, 2);  unsqueeze_1778 = None
    unsqueeze_1780: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1779, 3);  unsqueeze_1779 = None
    mul_1930: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_152);  primals_152 = None
    unsqueeze_1781: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1930, 0);  mul_1930 = None
    unsqueeze_1782: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1781, 2);  unsqueeze_1781 = None
    unsqueeze_1783: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1782, 3);  unsqueeze_1782 = None
    mul_1931: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_541, unsqueeze_1780);  sub_541 = unsqueeze_1780 = None
    sub_543: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_96, mul_1931);  where_96 = mul_1931 = None
    sub_544: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_543, unsqueeze_1777);  sub_543 = unsqueeze_1777 = None
    mul_1932: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_544, unsqueeze_1783);  sub_544 = unsqueeze_1783 = None
    mul_1933: "f32[28]" = torch.ops.aten.mul.Tensor(sum_199, squeeze_151);  sum_199 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(mul_1932, add_273, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1932 = add_273 = primals_151 = None
    getitem_1746: "f32[8, 28, 28, 28]" = convolution_backward_98[0]
    getitem_1747: "f32[28, 28, 3, 3]" = convolution_backward_98[1];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_896: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_81, getitem_1746);  slice_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_97: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_97, full_default, add_896);  le_97 = add_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_200: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_97, [0, 2, 3])
    sub_545: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_1786);  convolution_49 = unsqueeze_1786 = None
    mul_1934: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_97, sub_545)
    sum_201: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1934, [0, 2, 3]);  mul_1934 = None
    mul_1935: "f32[28]" = torch.ops.aten.mul.Tensor(sum_200, 0.00015943877551020407)
    unsqueeze_1787: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1935, 0);  mul_1935 = None
    unsqueeze_1788: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1787, 2);  unsqueeze_1787 = None
    unsqueeze_1789: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1788, 3);  unsqueeze_1788 = None
    mul_1936: "f32[28]" = torch.ops.aten.mul.Tensor(sum_201, 0.00015943877551020407)
    mul_1937: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_1938: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1936, mul_1937);  mul_1936 = mul_1937 = None
    unsqueeze_1790: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1938, 0);  mul_1938 = None
    unsqueeze_1791: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1790, 2);  unsqueeze_1790 = None
    unsqueeze_1792: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1791, 3);  unsqueeze_1791 = None
    mul_1939: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_149);  primals_149 = None
    unsqueeze_1793: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1939, 0);  mul_1939 = None
    unsqueeze_1794: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1793, 2);  unsqueeze_1793 = None
    unsqueeze_1795: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1794, 3);  unsqueeze_1794 = None
    mul_1940: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_545, unsqueeze_1792);  sub_545 = unsqueeze_1792 = None
    sub_547: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_97, mul_1940);  where_97 = mul_1940 = None
    sub_548: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_547, unsqueeze_1789);  sub_547 = unsqueeze_1789 = None
    mul_1941: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_548, unsqueeze_1795);  sub_548 = unsqueeze_1795 = None
    mul_1942: "f32[28]" = torch.ops.aten.mul.Tensor(sum_201, squeeze_148);  sum_201 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(mul_1941, getitem_468, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1941 = getitem_468 = primals_148 = None
    getitem_1749: "f32[8, 28, 28, 28]" = convolution_backward_99[0]
    getitem_1750: "f32[28, 28, 3, 3]" = convolution_backward_99[1];  convolution_backward_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_26: "f32[8, 224, 28, 28]" = torch.ops.aten.cat.default([getitem_1749, getitem_1746, getitem_1743, getitem_1740, getitem_1737, getitem_1734, getitem_1731, slice_88], 1);  getitem_1749 = getitem_1746 = getitem_1743 = getitem_1740 = getitem_1737 = getitem_1734 = getitem_1731 = slice_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_98: "f32[8, 224, 28, 28]" = torch.ops.aten.where.self(le_98, full_default, cat_26);  le_98 = cat_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_202: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_98, [0, 2, 3])
    sub_549: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_1798);  convolution_48 = unsqueeze_1798 = None
    mul_1943: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(where_98, sub_549)
    sum_203: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_1943, [0, 2, 3]);  mul_1943 = None
    mul_1944: "f32[224]" = torch.ops.aten.mul.Tensor(sum_202, 0.00015943877551020407)
    unsqueeze_1799: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1944, 0);  mul_1944 = None
    unsqueeze_1800: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1799, 2);  unsqueeze_1799 = None
    unsqueeze_1801: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1800, 3);  unsqueeze_1800 = None
    mul_1945: "f32[224]" = torch.ops.aten.mul.Tensor(sum_203, 0.00015943877551020407)
    mul_1946: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_1947: "f32[224]" = torch.ops.aten.mul.Tensor(mul_1945, mul_1946);  mul_1945 = mul_1946 = None
    unsqueeze_1802: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1947, 0);  mul_1947 = None
    unsqueeze_1803: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1802, 2);  unsqueeze_1802 = None
    unsqueeze_1804: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1803, 3);  unsqueeze_1803 = None
    mul_1948: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_146);  primals_146 = None
    unsqueeze_1805: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1948, 0);  mul_1948 = None
    unsqueeze_1806: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1805, 2);  unsqueeze_1805 = None
    unsqueeze_1807: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1806, 3);  unsqueeze_1806 = None
    mul_1949: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_549, unsqueeze_1804);  sub_549 = unsqueeze_1804 = None
    sub_551: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(where_98, mul_1949);  where_98 = mul_1949 = None
    sub_552: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(sub_551, unsqueeze_1801);  sub_551 = unsqueeze_1801 = None
    mul_1950: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_552, unsqueeze_1807);  sub_552 = unsqueeze_1807 = None
    mul_1951: "f32[224]" = torch.ops.aten.mul.Tensor(sum_203, squeeze_145);  sum_203 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(mul_1950, relu_45, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1950 = primals_145 = None
    getitem_1752: "f32[8, 512, 28, 28]" = convolution_backward_100[0]
    getitem_1753: "f32[224, 512, 1, 1]" = convolution_backward_100[1];  convolution_backward_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_897: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_90, getitem_1752);  where_90 = getitem_1752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_443: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_45);  relu_45 = None
    alias_444: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_443);  alias_443 = None
    le_99: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_444, 0);  alias_444 = None
    where_99: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_99, full_default, add_897);  le_99 = add_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_204: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_99, [0, 2, 3])
    sub_553: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_1810);  convolution_47 = unsqueeze_1810 = None
    mul_1952: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_99, sub_553)
    sum_205: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1952, [0, 2, 3]);  mul_1952 = None
    mul_1953: "f32[512]" = torch.ops.aten.mul.Tensor(sum_204, 0.00015943877551020407)
    unsqueeze_1811: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1953, 0);  mul_1953 = None
    unsqueeze_1812: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1811, 2);  unsqueeze_1811 = None
    unsqueeze_1813: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1812, 3);  unsqueeze_1812 = None
    mul_1954: "f32[512]" = torch.ops.aten.mul.Tensor(sum_205, 0.00015943877551020407)
    mul_1955: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_1956: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1954, mul_1955);  mul_1954 = mul_1955 = None
    unsqueeze_1814: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1956, 0);  mul_1956 = None
    unsqueeze_1815: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1814, 2);  unsqueeze_1814 = None
    unsqueeze_1816: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1815, 3);  unsqueeze_1815 = None
    mul_1957: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_143);  primals_143 = None
    unsqueeze_1817: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1957, 0);  mul_1957 = None
    unsqueeze_1818: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1817, 2);  unsqueeze_1817 = None
    unsqueeze_1819: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1818, 3);  unsqueeze_1818 = None
    mul_1958: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_553, unsqueeze_1816);  sub_553 = unsqueeze_1816 = None
    sub_555: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_99, mul_1958);  mul_1958 = None
    sub_556: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_555, unsqueeze_1813);  sub_555 = unsqueeze_1813 = None
    mul_1959: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_556, unsqueeze_1819);  sub_556 = unsqueeze_1819 = None
    mul_1960: "f32[512]" = torch.ops.aten.mul.Tensor(sum_205, squeeze_142);  sum_205 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(mul_1959, cat_4, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1959 = cat_4 = primals_142 = None
    getitem_1755: "f32[8, 224, 28, 28]" = convolution_backward_101[0]
    getitem_1756: "f32[512, 224, 1, 1]" = convolution_backward_101[1];  convolution_backward_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_89: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1755, 1, 0, 28)
    slice_90: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1755, 1, 28, 56)
    slice_91: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1755, 1, 56, 84)
    slice_92: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1755, 1, 84, 112)
    slice_93: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1755, 1, 112, 140)
    slice_94: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1755, 1, 140, 168)
    slice_95: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1755, 1, 168, 196)
    slice_96: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1755, 1, 196, 224);  getitem_1755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_100: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_100, full_default, slice_95);  le_100 = slice_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_206: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_100, [0, 2, 3])
    sub_557: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_1822);  convolution_46 = unsqueeze_1822 = None
    mul_1961: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_100, sub_557)
    sum_207: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1961, [0, 2, 3]);  mul_1961 = None
    mul_1962: "f32[28]" = torch.ops.aten.mul.Tensor(sum_206, 0.00015943877551020407)
    unsqueeze_1823: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1962, 0);  mul_1962 = None
    unsqueeze_1824: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1823, 2);  unsqueeze_1823 = None
    unsqueeze_1825: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1824, 3);  unsqueeze_1824 = None
    mul_1963: "f32[28]" = torch.ops.aten.mul.Tensor(sum_207, 0.00015943877551020407)
    mul_1964: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_1965: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1963, mul_1964);  mul_1963 = mul_1964 = None
    unsqueeze_1826: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1965, 0);  mul_1965 = None
    unsqueeze_1827: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1826, 2);  unsqueeze_1826 = None
    unsqueeze_1828: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1827, 3);  unsqueeze_1827 = None
    mul_1966: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_140);  primals_140 = None
    unsqueeze_1829: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1966, 0);  mul_1966 = None
    unsqueeze_1830: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1829, 2);  unsqueeze_1829 = None
    unsqueeze_1831: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1830, 3);  unsqueeze_1830 = None
    mul_1967: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_557, unsqueeze_1828);  sub_557 = unsqueeze_1828 = None
    sub_559: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_100, mul_1967);  where_100 = mul_1967 = None
    sub_560: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_559, unsqueeze_1825);  sub_559 = unsqueeze_1825 = None
    mul_1968: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_560, unsqueeze_1831);  sub_560 = unsqueeze_1831 = None
    mul_1969: "f32[28]" = torch.ops.aten.mul.Tensor(sum_207, squeeze_139);  sum_207 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(mul_1968, add_251, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1968 = add_251 = primals_139 = None
    getitem_1758: "f32[8, 28, 28, 28]" = convolution_backward_102[0]
    getitem_1759: "f32[28, 28, 3, 3]" = convolution_backward_102[1];  convolution_backward_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_898: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_94, getitem_1758);  slice_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_101: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_101, full_default, add_898);  le_101 = add_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_208: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_101, [0, 2, 3])
    sub_561: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_1834);  convolution_45 = unsqueeze_1834 = None
    mul_1970: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_101, sub_561)
    sum_209: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1970, [0, 2, 3]);  mul_1970 = None
    mul_1971: "f32[28]" = torch.ops.aten.mul.Tensor(sum_208, 0.00015943877551020407)
    unsqueeze_1835: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1971, 0);  mul_1971 = None
    unsqueeze_1836: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1835, 2);  unsqueeze_1835 = None
    unsqueeze_1837: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1836, 3);  unsqueeze_1836 = None
    mul_1972: "f32[28]" = torch.ops.aten.mul.Tensor(sum_209, 0.00015943877551020407)
    mul_1973: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_1974: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1972, mul_1973);  mul_1972 = mul_1973 = None
    unsqueeze_1838: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1974, 0);  mul_1974 = None
    unsqueeze_1839: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1838, 2);  unsqueeze_1838 = None
    unsqueeze_1840: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1839, 3);  unsqueeze_1839 = None
    mul_1975: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_137);  primals_137 = None
    unsqueeze_1841: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1975, 0);  mul_1975 = None
    unsqueeze_1842: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1841, 2);  unsqueeze_1841 = None
    unsqueeze_1843: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1842, 3);  unsqueeze_1842 = None
    mul_1976: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_561, unsqueeze_1840);  sub_561 = unsqueeze_1840 = None
    sub_563: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_101, mul_1976);  where_101 = mul_1976 = None
    sub_564: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_563, unsqueeze_1837);  sub_563 = unsqueeze_1837 = None
    mul_1977: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_564, unsqueeze_1843);  sub_564 = unsqueeze_1843 = None
    mul_1978: "f32[28]" = torch.ops.aten.mul.Tensor(sum_209, squeeze_136);  sum_209 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(mul_1977, add_245, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1977 = add_245 = primals_136 = None
    getitem_1761: "f32[8, 28, 28, 28]" = convolution_backward_103[0]
    getitem_1762: "f32[28, 28, 3, 3]" = convolution_backward_103[1];  convolution_backward_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_899: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_93, getitem_1761);  slice_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_102: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_102, full_default, add_899);  le_102 = add_899 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_210: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_102, [0, 2, 3])
    sub_565: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_1846);  convolution_44 = unsqueeze_1846 = None
    mul_1979: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_102, sub_565)
    sum_211: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1979, [0, 2, 3]);  mul_1979 = None
    mul_1980: "f32[28]" = torch.ops.aten.mul.Tensor(sum_210, 0.00015943877551020407)
    unsqueeze_1847: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1980, 0);  mul_1980 = None
    unsqueeze_1848: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1847, 2);  unsqueeze_1847 = None
    unsqueeze_1849: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1848, 3);  unsqueeze_1848 = None
    mul_1981: "f32[28]" = torch.ops.aten.mul.Tensor(sum_211, 0.00015943877551020407)
    mul_1982: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_1983: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1981, mul_1982);  mul_1981 = mul_1982 = None
    unsqueeze_1850: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1983, 0);  mul_1983 = None
    unsqueeze_1851: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1850, 2);  unsqueeze_1850 = None
    unsqueeze_1852: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1851, 3);  unsqueeze_1851 = None
    mul_1984: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_134);  primals_134 = None
    unsqueeze_1853: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1984, 0);  mul_1984 = None
    unsqueeze_1854: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1853, 2);  unsqueeze_1853 = None
    unsqueeze_1855: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1854, 3);  unsqueeze_1854 = None
    mul_1985: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_565, unsqueeze_1852);  sub_565 = unsqueeze_1852 = None
    sub_567: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_102, mul_1985);  where_102 = mul_1985 = None
    sub_568: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_567, unsqueeze_1849);  sub_567 = unsqueeze_1849 = None
    mul_1986: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_568, unsqueeze_1855);  sub_568 = unsqueeze_1855 = None
    mul_1987: "f32[28]" = torch.ops.aten.mul.Tensor(sum_211, squeeze_133);  sum_211 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_104 = torch.ops.aten.convolution_backward.default(mul_1986, add_239, primals_133, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1986 = add_239 = primals_133 = None
    getitem_1764: "f32[8, 28, 28, 28]" = convolution_backward_104[0]
    getitem_1765: "f32[28, 28, 3, 3]" = convolution_backward_104[1];  convolution_backward_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_900: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_92, getitem_1764);  slice_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_103: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_103, full_default, add_900);  le_103 = add_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_212: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_103, [0, 2, 3])
    sub_569: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_1858);  convolution_43 = unsqueeze_1858 = None
    mul_1988: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_103, sub_569)
    sum_213: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1988, [0, 2, 3]);  mul_1988 = None
    mul_1989: "f32[28]" = torch.ops.aten.mul.Tensor(sum_212, 0.00015943877551020407)
    unsqueeze_1859: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1989, 0);  mul_1989 = None
    unsqueeze_1860: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1859, 2);  unsqueeze_1859 = None
    unsqueeze_1861: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1860, 3);  unsqueeze_1860 = None
    mul_1990: "f32[28]" = torch.ops.aten.mul.Tensor(sum_213, 0.00015943877551020407)
    mul_1991: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_1992: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1990, mul_1991);  mul_1990 = mul_1991 = None
    unsqueeze_1862: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1992, 0);  mul_1992 = None
    unsqueeze_1863: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1862, 2);  unsqueeze_1862 = None
    unsqueeze_1864: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1863, 3);  unsqueeze_1863 = None
    mul_1993: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_131);  primals_131 = None
    unsqueeze_1865: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1993, 0);  mul_1993 = None
    unsqueeze_1866: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1865, 2);  unsqueeze_1865 = None
    unsqueeze_1867: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1866, 3);  unsqueeze_1866 = None
    mul_1994: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_569, unsqueeze_1864);  sub_569 = unsqueeze_1864 = None
    sub_571: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_103, mul_1994);  where_103 = mul_1994 = None
    sub_572: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_571, unsqueeze_1861);  sub_571 = unsqueeze_1861 = None
    mul_1995: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_572, unsqueeze_1867);  sub_572 = unsqueeze_1867 = None
    mul_1996: "f32[28]" = torch.ops.aten.mul.Tensor(sum_213, squeeze_130);  sum_213 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_105 = torch.ops.aten.convolution_backward.default(mul_1995, add_233, primals_130, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1995 = add_233 = primals_130 = None
    getitem_1767: "f32[8, 28, 28, 28]" = convolution_backward_105[0]
    getitem_1768: "f32[28, 28, 3, 3]" = convolution_backward_105[1];  convolution_backward_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_901: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_91, getitem_1767);  slice_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_104: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_104, full_default, add_901);  le_104 = add_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_214: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_104, [0, 2, 3])
    sub_573: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_1870);  convolution_42 = unsqueeze_1870 = None
    mul_1997: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_104, sub_573)
    sum_215: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_1997, [0, 2, 3]);  mul_1997 = None
    mul_1998: "f32[28]" = torch.ops.aten.mul.Tensor(sum_214, 0.00015943877551020407)
    unsqueeze_1871: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_1998, 0);  mul_1998 = None
    unsqueeze_1872: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1871, 2);  unsqueeze_1871 = None
    unsqueeze_1873: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1872, 3);  unsqueeze_1872 = None
    mul_1999: "f32[28]" = torch.ops.aten.mul.Tensor(sum_215, 0.00015943877551020407)
    mul_2000: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_2001: "f32[28]" = torch.ops.aten.mul.Tensor(mul_1999, mul_2000);  mul_1999 = mul_2000 = None
    unsqueeze_1874: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2001, 0);  mul_2001 = None
    unsqueeze_1875: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1874, 2);  unsqueeze_1874 = None
    unsqueeze_1876: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1875, 3);  unsqueeze_1875 = None
    mul_2002: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_128);  primals_128 = None
    unsqueeze_1877: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2002, 0);  mul_2002 = None
    unsqueeze_1878: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1877, 2);  unsqueeze_1877 = None
    unsqueeze_1879: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1878, 3);  unsqueeze_1878 = None
    mul_2003: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_573, unsqueeze_1876);  sub_573 = unsqueeze_1876 = None
    sub_575: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_104, mul_2003);  where_104 = mul_2003 = None
    sub_576: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_575, unsqueeze_1873);  sub_575 = unsqueeze_1873 = None
    mul_2004: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_576, unsqueeze_1879);  sub_576 = unsqueeze_1879 = None
    mul_2005: "f32[28]" = torch.ops.aten.mul.Tensor(sum_215, squeeze_127);  sum_215 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_106 = torch.ops.aten.convolution_backward.default(mul_2004, add_227, primals_127, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2004 = add_227 = primals_127 = None
    getitem_1770: "f32[8, 28, 28, 28]" = convolution_backward_106[0]
    getitem_1771: "f32[28, 28, 3, 3]" = convolution_backward_106[1];  convolution_backward_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_902: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_90, getitem_1770);  slice_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_105: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_105, full_default, add_902);  le_105 = add_902 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_216: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_105, [0, 2, 3])
    sub_577: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1882);  convolution_41 = unsqueeze_1882 = None
    mul_2006: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_105, sub_577)
    sum_217: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_2006, [0, 2, 3]);  mul_2006 = None
    mul_2007: "f32[28]" = torch.ops.aten.mul.Tensor(sum_216, 0.00015943877551020407)
    unsqueeze_1883: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2007, 0);  mul_2007 = None
    unsqueeze_1884: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1883, 2);  unsqueeze_1883 = None
    unsqueeze_1885: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1884, 3);  unsqueeze_1884 = None
    mul_2008: "f32[28]" = torch.ops.aten.mul.Tensor(sum_217, 0.00015943877551020407)
    mul_2009: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_2010: "f32[28]" = torch.ops.aten.mul.Tensor(mul_2008, mul_2009);  mul_2008 = mul_2009 = None
    unsqueeze_1886: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2010, 0);  mul_2010 = None
    unsqueeze_1887: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1886, 2);  unsqueeze_1886 = None
    unsqueeze_1888: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1887, 3);  unsqueeze_1887 = None
    mul_2011: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_125);  primals_125 = None
    unsqueeze_1889: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2011, 0);  mul_2011 = None
    unsqueeze_1890: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1889, 2);  unsqueeze_1889 = None
    unsqueeze_1891: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1890, 3);  unsqueeze_1890 = None
    mul_2012: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_577, unsqueeze_1888);  sub_577 = unsqueeze_1888 = None
    sub_579: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_105, mul_2012);  where_105 = mul_2012 = None
    sub_580: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_579, unsqueeze_1885);  sub_579 = unsqueeze_1885 = None
    mul_2013: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_580, unsqueeze_1891);  sub_580 = unsqueeze_1891 = None
    mul_2014: "f32[28]" = torch.ops.aten.mul.Tensor(sum_217, squeeze_124);  sum_217 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_107 = torch.ops.aten.convolution_backward.default(mul_2013, add_221, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2013 = add_221 = primals_124 = None
    getitem_1773: "f32[8, 28, 28, 28]" = convolution_backward_107[0]
    getitem_1774: "f32[28, 28, 3, 3]" = convolution_backward_107[1];  convolution_backward_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_903: "f32[8, 28, 28, 28]" = torch.ops.aten.add.Tensor(slice_89, getitem_1773);  slice_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_106: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_106, full_default, add_903);  le_106 = add_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_218: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_106, [0, 2, 3])
    sub_581: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_1894);  convolution_40 = unsqueeze_1894 = None
    mul_2015: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_106, sub_581)
    sum_219: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_2015, [0, 2, 3]);  mul_2015 = None
    mul_2016: "f32[28]" = torch.ops.aten.mul.Tensor(sum_218, 0.00015943877551020407)
    unsqueeze_1895: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2016, 0);  mul_2016 = None
    unsqueeze_1896: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1895, 2);  unsqueeze_1895 = None
    unsqueeze_1897: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1896, 3);  unsqueeze_1896 = None
    mul_2017: "f32[28]" = torch.ops.aten.mul.Tensor(sum_219, 0.00015943877551020407)
    mul_2018: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_2019: "f32[28]" = torch.ops.aten.mul.Tensor(mul_2017, mul_2018);  mul_2017 = mul_2018 = None
    unsqueeze_1898: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2019, 0);  mul_2019 = None
    unsqueeze_1899: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1898, 2);  unsqueeze_1898 = None
    unsqueeze_1900: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1899, 3);  unsqueeze_1899 = None
    mul_2020: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_122);  primals_122 = None
    unsqueeze_1901: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2020, 0);  mul_2020 = None
    unsqueeze_1902: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1901, 2);  unsqueeze_1901 = None
    unsqueeze_1903: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1902, 3);  unsqueeze_1902 = None
    mul_2021: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_581, unsqueeze_1900);  sub_581 = unsqueeze_1900 = None
    sub_583: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_106, mul_2021);  where_106 = mul_2021 = None
    sub_584: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_583, unsqueeze_1897);  sub_583 = unsqueeze_1897 = None
    mul_2022: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_584, unsqueeze_1903);  sub_584 = unsqueeze_1903 = None
    mul_2023: "f32[28]" = torch.ops.aten.mul.Tensor(sum_219, squeeze_121);  sum_219 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_108 = torch.ops.aten.convolution_backward.default(mul_2022, getitem_378, primals_121, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2022 = getitem_378 = primals_121 = None
    getitem_1776: "f32[8, 28, 28, 28]" = convolution_backward_108[0]
    getitem_1777: "f32[28, 28, 3, 3]" = convolution_backward_108[1];  convolution_backward_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_27: "f32[8, 224, 28, 28]" = torch.ops.aten.cat.default([getitem_1776, getitem_1773, getitem_1770, getitem_1767, getitem_1764, getitem_1761, getitem_1758, slice_96], 1);  getitem_1776 = getitem_1773 = getitem_1770 = getitem_1767 = getitem_1764 = getitem_1761 = getitem_1758 = slice_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_107: "f32[8, 224, 28, 28]" = torch.ops.aten.where.self(le_107, full_default, cat_27);  le_107 = cat_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_220: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_107, [0, 2, 3])
    sub_585: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_1906);  convolution_39 = unsqueeze_1906 = None
    mul_2024: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(where_107, sub_585)
    sum_221: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_2024, [0, 2, 3]);  mul_2024 = None
    mul_2025: "f32[224]" = torch.ops.aten.mul.Tensor(sum_220, 0.00015943877551020407)
    unsqueeze_1907: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_2025, 0);  mul_2025 = None
    unsqueeze_1908: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1907, 2);  unsqueeze_1907 = None
    unsqueeze_1909: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1908, 3);  unsqueeze_1908 = None
    mul_2026: "f32[224]" = torch.ops.aten.mul.Tensor(sum_221, 0.00015943877551020407)
    mul_2027: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_2028: "f32[224]" = torch.ops.aten.mul.Tensor(mul_2026, mul_2027);  mul_2026 = mul_2027 = None
    unsqueeze_1910: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_2028, 0);  mul_2028 = None
    unsqueeze_1911: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1910, 2);  unsqueeze_1910 = None
    unsqueeze_1912: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1911, 3);  unsqueeze_1911 = None
    mul_2029: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_119);  primals_119 = None
    unsqueeze_1913: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_2029, 0);  mul_2029 = None
    unsqueeze_1914: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1913, 2);  unsqueeze_1913 = None
    unsqueeze_1915: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1914, 3);  unsqueeze_1914 = None
    mul_2030: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_585, unsqueeze_1912);  sub_585 = unsqueeze_1912 = None
    sub_587: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(where_107, mul_2030);  where_107 = mul_2030 = None
    sub_588: "f32[8, 224, 28, 28]" = torch.ops.aten.sub.Tensor(sub_587, unsqueeze_1909);  sub_587 = unsqueeze_1909 = None
    mul_2031: "f32[8, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_588, unsqueeze_1915);  sub_588 = unsqueeze_1915 = None
    mul_2032: "f32[224]" = torch.ops.aten.mul.Tensor(sum_221, squeeze_118);  sum_221 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_109 = torch.ops.aten.convolution_backward.default(mul_2031, relu_36, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2031 = primals_118 = None
    getitem_1779: "f32[8, 512, 28, 28]" = convolution_backward_109[0]
    getitem_1780: "f32[224, 512, 1, 1]" = convolution_backward_109[1];  convolution_backward_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_904: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_99, getitem_1779);  where_99 = getitem_1779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_470: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_471: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_470);  alias_470 = None
    le_108: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_471, 0);  alias_471 = None
    where_108: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_108, full_default, add_904);  le_108 = add_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    sum_222: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_108, [0, 2, 3])
    sub_589: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_1918);  convolution_38 = unsqueeze_1918 = None
    mul_2033: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_108, sub_589)
    sum_223: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_2033, [0, 2, 3]);  mul_2033 = None
    mul_2034: "f32[512]" = torch.ops.aten.mul.Tensor(sum_222, 0.00015943877551020407)
    unsqueeze_1919: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2034, 0);  mul_2034 = None
    unsqueeze_1920: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1919, 2);  unsqueeze_1919 = None
    unsqueeze_1921: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1920, 3);  unsqueeze_1920 = None
    mul_2035: "f32[512]" = torch.ops.aten.mul.Tensor(sum_223, 0.00015943877551020407)
    mul_2036: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_2037: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2035, mul_2036);  mul_2035 = mul_2036 = None
    unsqueeze_1922: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2037, 0);  mul_2037 = None
    unsqueeze_1923: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1922, 2);  unsqueeze_1922 = None
    unsqueeze_1924: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1923, 3);  unsqueeze_1923 = None
    mul_2038: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_116);  primals_116 = None
    unsqueeze_1925: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2038, 0);  mul_2038 = None
    unsqueeze_1926: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1925, 2);  unsqueeze_1925 = None
    unsqueeze_1927: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1926, 3);  unsqueeze_1926 = None
    mul_2039: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_589, unsqueeze_1924);  sub_589 = unsqueeze_1924 = None
    sub_591: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_108, mul_2039);  mul_2039 = None
    sub_592: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_591, unsqueeze_1921);  sub_591 = None
    mul_2040: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_592, unsqueeze_1927);  sub_592 = unsqueeze_1927 = None
    mul_2041: "f32[512]" = torch.ops.aten.mul.Tensor(sum_223, squeeze_115);  sum_223 = squeeze_115 = None
    convolution_backward_110 = torch.ops.aten.convolution_backward.default(mul_2040, relu_27, primals_115, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2040 = primals_115 = None
    getitem_1782: "f32[8, 256, 56, 56]" = convolution_backward_110[0]
    getitem_1783: "f32[512, 256, 1, 1]" = convolution_backward_110[1];  convolution_backward_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sub_593: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1930);  convolution_37 = unsqueeze_1930 = None
    mul_2042: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_108, sub_593)
    sum_225: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_2042, [0, 2, 3]);  mul_2042 = None
    mul_2044: "f32[512]" = torch.ops.aten.mul.Tensor(sum_225, 0.00015943877551020407)
    mul_2045: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_2046: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2044, mul_2045);  mul_2044 = mul_2045 = None
    unsqueeze_1934: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2046, 0);  mul_2046 = None
    unsqueeze_1935: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1934, 2);  unsqueeze_1934 = None
    unsqueeze_1936: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1935, 3);  unsqueeze_1935 = None
    mul_2047: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_113);  primals_113 = None
    unsqueeze_1937: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_2047, 0);  mul_2047 = None
    unsqueeze_1938: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1937, 2);  unsqueeze_1937 = None
    unsqueeze_1939: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1938, 3);  unsqueeze_1938 = None
    mul_2048: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_593, unsqueeze_1936);  sub_593 = unsqueeze_1936 = None
    sub_595: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_108, mul_2048);  where_108 = mul_2048 = None
    sub_596: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_595, unsqueeze_1921);  sub_595 = unsqueeze_1921 = None
    mul_2049: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_596, unsqueeze_1939);  sub_596 = unsqueeze_1939 = None
    mul_2050: "f32[512]" = torch.ops.aten.mul.Tensor(sum_225, squeeze_112);  sum_225 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_111 = torch.ops.aten.convolution_backward.default(mul_2049, cat_3, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2049 = cat_3 = primals_112 = None
    getitem_1785: "f32[8, 224, 28, 28]" = convolution_backward_111[0]
    getitem_1786: "f32[512, 224, 1, 1]" = convolution_backward_111[1];  convolution_backward_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_97: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1785, 1, 0, 28)
    slice_98: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1785, 1, 28, 56)
    slice_99: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1785, 1, 56, 84)
    slice_100: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1785, 1, 84, 112)
    slice_101: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1785, 1, 112, 140)
    slice_102: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1785, 1, 140, 168)
    slice_103: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1785, 1, 168, 196)
    slice_104: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_1785, 1, 196, 224);  getitem_1785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_backward_2: "f32[8, 28, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(slice_104, getitem_363, [3, 3], [2, 2], [1, 1], False, True, None);  slice_104 = getitem_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_109: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_109, full_default, slice_103);  le_109 = slice_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_226: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_109, [0, 2, 3])
    sub_597: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1942);  convolution_36 = unsqueeze_1942 = None
    mul_2051: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_109, sub_597)
    sum_227: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_2051, [0, 2, 3]);  mul_2051 = None
    mul_2052: "f32[28]" = torch.ops.aten.mul.Tensor(sum_226, 0.00015943877551020407)
    unsqueeze_1943: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2052, 0);  mul_2052 = None
    unsqueeze_1944: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1943, 2);  unsqueeze_1943 = None
    unsqueeze_1945: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1944, 3);  unsqueeze_1944 = None
    mul_2053: "f32[28]" = torch.ops.aten.mul.Tensor(sum_227, 0.00015943877551020407)
    mul_2054: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_2055: "f32[28]" = torch.ops.aten.mul.Tensor(mul_2053, mul_2054);  mul_2053 = mul_2054 = None
    unsqueeze_1946: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2055, 0);  mul_2055 = None
    unsqueeze_1947: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1946, 2);  unsqueeze_1946 = None
    unsqueeze_1948: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1947, 3);  unsqueeze_1947 = None
    mul_2056: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_110);  primals_110 = None
    unsqueeze_1949: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2056, 0);  mul_2056 = None
    unsqueeze_1950: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1949, 2);  unsqueeze_1949 = None
    unsqueeze_1951: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1950, 3);  unsqueeze_1950 = None
    mul_2057: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_597, unsqueeze_1948);  sub_597 = unsqueeze_1948 = None
    sub_599: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_109, mul_2057);  where_109 = mul_2057 = None
    sub_600: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_599, unsqueeze_1945);  sub_599 = unsqueeze_1945 = None
    mul_2058: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_600, unsqueeze_1951);  sub_600 = unsqueeze_1951 = None
    mul_2059: "f32[28]" = torch.ops.aten.mul.Tensor(sum_227, squeeze_109);  sum_227 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_112 = torch.ops.aten.convolution_backward.default(mul_2058, getitem_352, primals_109, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2058 = getitem_352 = primals_109 = None
    getitem_1788: "f32[8, 28, 56, 56]" = convolution_backward_112[0]
    getitem_1789: "f32[28, 28, 3, 3]" = convolution_backward_112[1];  convolution_backward_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_110: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_110, full_default, slice_102);  le_110 = slice_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_228: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_110, [0, 2, 3])
    sub_601: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_1954);  convolution_35 = unsqueeze_1954 = None
    mul_2060: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_110, sub_601)
    sum_229: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_2060, [0, 2, 3]);  mul_2060 = None
    mul_2061: "f32[28]" = torch.ops.aten.mul.Tensor(sum_228, 0.00015943877551020407)
    unsqueeze_1955: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2061, 0);  mul_2061 = None
    unsqueeze_1956: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1955, 2);  unsqueeze_1955 = None
    unsqueeze_1957: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1956, 3);  unsqueeze_1956 = None
    mul_2062: "f32[28]" = torch.ops.aten.mul.Tensor(sum_229, 0.00015943877551020407)
    mul_2063: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_2064: "f32[28]" = torch.ops.aten.mul.Tensor(mul_2062, mul_2063);  mul_2062 = mul_2063 = None
    unsqueeze_1958: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2064, 0);  mul_2064 = None
    unsqueeze_1959: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1958, 2);  unsqueeze_1958 = None
    unsqueeze_1960: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1959, 3);  unsqueeze_1959 = None
    mul_2065: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_107);  primals_107 = None
    unsqueeze_1961: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2065, 0);  mul_2065 = None
    unsqueeze_1962: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1961, 2);  unsqueeze_1961 = None
    unsqueeze_1963: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1962, 3);  unsqueeze_1962 = None
    mul_2066: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_601, unsqueeze_1960);  sub_601 = unsqueeze_1960 = None
    sub_603: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_110, mul_2066);  where_110 = mul_2066 = None
    sub_604: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_603, unsqueeze_1957);  sub_603 = unsqueeze_1957 = None
    mul_2067: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_604, unsqueeze_1963);  sub_604 = unsqueeze_1963 = None
    mul_2068: "f32[28]" = torch.ops.aten.mul.Tensor(sum_229, squeeze_106);  sum_229 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_113 = torch.ops.aten.convolution_backward.default(mul_2067, getitem_341, primals_106, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2067 = getitem_341 = primals_106 = None
    getitem_1791: "f32[8, 28, 56, 56]" = convolution_backward_113[0]
    getitem_1792: "f32[28, 28, 3, 3]" = convolution_backward_113[1];  convolution_backward_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_111: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_111, full_default, slice_101);  le_111 = slice_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_230: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_111, [0, 2, 3])
    sub_605: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_1966);  convolution_34 = unsqueeze_1966 = None
    mul_2069: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_111, sub_605)
    sum_231: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_2069, [0, 2, 3]);  mul_2069 = None
    mul_2070: "f32[28]" = torch.ops.aten.mul.Tensor(sum_230, 0.00015943877551020407)
    unsqueeze_1967: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2070, 0);  mul_2070 = None
    unsqueeze_1968: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1967, 2);  unsqueeze_1967 = None
    unsqueeze_1969: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1968, 3);  unsqueeze_1968 = None
    mul_2071: "f32[28]" = torch.ops.aten.mul.Tensor(sum_231, 0.00015943877551020407)
    mul_2072: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_2073: "f32[28]" = torch.ops.aten.mul.Tensor(mul_2071, mul_2072);  mul_2071 = mul_2072 = None
    unsqueeze_1970: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2073, 0);  mul_2073 = None
    unsqueeze_1971: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1970, 2);  unsqueeze_1970 = None
    unsqueeze_1972: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1971, 3);  unsqueeze_1971 = None
    mul_2074: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_104);  primals_104 = None
    unsqueeze_1973: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2074, 0);  mul_2074 = None
    unsqueeze_1974: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1973, 2);  unsqueeze_1973 = None
    unsqueeze_1975: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1974, 3);  unsqueeze_1974 = None
    mul_2075: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_605, unsqueeze_1972);  sub_605 = unsqueeze_1972 = None
    sub_607: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_111, mul_2075);  where_111 = mul_2075 = None
    sub_608: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_607, unsqueeze_1969);  sub_607 = unsqueeze_1969 = None
    mul_2076: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_608, unsqueeze_1975);  sub_608 = unsqueeze_1975 = None
    mul_2077: "f32[28]" = torch.ops.aten.mul.Tensor(sum_231, squeeze_103);  sum_231 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_114 = torch.ops.aten.convolution_backward.default(mul_2076, getitem_330, primals_103, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2076 = getitem_330 = primals_103 = None
    getitem_1794: "f32[8, 28, 56, 56]" = convolution_backward_114[0]
    getitem_1795: "f32[28, 28, 3, 3]" = convolution_backward_114[1];  convolution_backward_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_112: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_112, full_default, slice_100);  le_112 = slice_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_232: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_112, [0, 2, 3])
    sub_609: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1978);  convolution_33 = unsqueeze_1978 = None
    mul_2078: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_112, sub_609)
    sum_233: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_2078, [0, 2, 3]);  mul_2078 = None
    mul_2079: "f32[28]" = torch.ops.aten.mul.Tensor(sum_232, 0.00015943877551020407)
    unsqueeze_1979: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2079, 0);  mul_2079 = None
    unsqueeze_1980: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1979, 2);  unsqueeze_1979 = None
    unsqueeze_1981: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1980, 3);  unsqueeze_1980 = None
    mul_2080: "f32[28]" = torch.ops.aten.mul.Tensor(sum_233, 0.00015943877551020407)
    mul_2081: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_2082: "f32[28]" = torch.ops.aten.mul.Tensor(mul_2080, mul_2081);  mul_2080 = mul_2081 = None
    unsqueeze_1982: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2082, 0);  mul_2082 = None
    unsqueeze_1983: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1982, 2);  unsqueeze_1982 = None
    unsqueeze_1984: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1983, 3);  unsqueeze_1983 = None
    mul_2083: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_101);  primals_101 = None
    unsqueeze_1985: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2083, 0);  mul_2083 = None
    unsqueeze_1986: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1985, 2);  unsqueeze_1985 = None
    unsqueeze_1987: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1986, 3);  unsqueeze_1986 = None
    mul_2084: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_609, unsqueeze_1984);  sub_609 = unsqueeze_1984 = None
    sub_611: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_112, mul_2084);  where_112 = mul_2084 = None
    sub_612: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_611, unsqueeze_1981);  sub_611 = unsqueeze_1981 = None
    mul_2085: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_612, unsqueeze_1987);  sub_612 = unsqueeze_1987 = None
    mul_2086: "f32[28]" = torch.ops.aten.mul.Tensor(sum_233, squeeze_100);  sum_233 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_115 = torch.ops.aten.convolution_backward.default(mul_2085, getitem_319, primals_100, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2085 = getitem_319 = primals_100 = None
    getitem_1797: "f32[8, 28, 56, 56]" = convolution_backward_115[0]
    getitem_1798: "f32[28, 28, 3, 3]" = convolution_backward_115[1];  convolution_backward_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_113: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_113, full_default, slice_99);  le_113 = slice_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_234: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_113, [0, 2, 3])
    sub_613: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1990);  convolution_32 = unsqueeze_1990 = None
    mul_2087: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_113, sub_613)
    sum_235: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_2087, [0, 2, 3]);  mul_2087 = None
    mul_2088: "f32[28]" = torch.ops.aten.mul.Tensor(sum_234, 0.00015943877551020407)
    unsqueeze_1991: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2088, 0);  mul_2088 = None
    unsqueeze_1992: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1991, 2);  unsqueeze_1991 = None
    unsqueeze_1993: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1992, 3);  unsqueeze_1992 = None
    mul_2089: "f32[28]" = torch.ops.aten.mul.Tensor(sum_235, 0.00015943877551020407)
    mul_2090: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_2091: "f32[28]" = torch.ops.aten.mul.Tensor(mul_2089, mul_2090);  mul_2089 = mul_2090 = None
    unsqueeze_1994: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2091, 0);  mul_2091 = None
    unsqueeze_1995: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1994, 2);  unsqueeze_1994 = None
    unsqueeze_1996: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1995, 3);  unsqueeze_1995 = None
    mul_2092: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_98);  primals_98 = None
    unsqueeze_1997: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2092, 0);  mul_2092 = None
    unsqueeze_1998: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1997, 2);  unsqueeze_1997 = None
    unsqueeze_1999: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1998, 3);  unsqueeze_1998 = None
    mul_2093: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_613, unsqueeze_1996);  sub_613 = unsqueeze_1996 = None
    sub_615: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_113, mul_2093);  where_113 = mul_2093 = None
    sub_616: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_615, unsqueeze_1993);  sub_615 = unsqueeze_1993 = None
    mul_2094: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_616, unsqueeze_1999);  sub_616 = unsqueeze_1999 = None
    mul_2095: "f32[28]" = torch.ops.aten.mul.Tensor(sum_235, squeeze_97);  sum_235 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_116 = torch.ops.aten.convolution_backward.default(mul_2094, getitem_308, primals_97, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2094 = getitem_308 = primals_97 = None
    getitem_1800: "f32[8, 28, 56, 56]" = convolution_backward_116[0]
    getitem_1801: "f32[28, 28, 3, 3]" = convolution_backward_116[1];  convolution_backward_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_114: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_114, full_default, slice_98);  le_114 = slice_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_236: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_114, [0, 2, 3])
    sub_617: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_2002);  convolution_31 = unsqueeze_2002 = None
    mul_2096: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_114, sub_617)
    sum_237: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_2096, [0, 2, 3]);  mul_2096 = None
    mul_2097: "f32[28]" = torch.ops.aten.mul.Tensor(sum_236, 0.00015943877551020407)
    unsqueeze_2003: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2097, 0);  mul_2097 = None
    unsqueeze_2004: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2003, 2);  unsqueeze_2003 = None
    unsqueeze_2005: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2004, 3);  unsqueeze_2004 = None
    mul_2098: "f32[28]" = torch.ops.aten.mul.Tensor(sum_237, 0.00015943877551020407)
    mul_2099: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_2100: "f32[28]" = torch.ops.aten.mul.Tensor(mul_2098, mul_2099);  mul_2098 = mul_2099 = None
    unsqueeze_2006: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2100, 0);  mul_2100 = None
    unsqueeze_2007: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2006, 2);  unsqueeze_2006 = None
    unsqueeze_2008: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2007, 3);  unsqueeze_2007 = None
    mul_2101: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_95);  primals_95 = None
    unsqueeze_2009: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2101, 0);  mul_2101 = None
    unsqueeze_2010: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2009, 2);  unsqueeze_2009 = None
    unsqueeze_2011: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2010, 3);  unsqueeze_2010 = None
    mul_2102: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_617, unsqueeze_2008);  sub_617 = unsqueeze_2008 = None
    sub_619: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_114, mul_2102);  where_114 = mul_2102 = None
    sub_620: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_619, unsqueeze_2005);  sub_619 = unsqueeze_2005 = None
    mul_2103: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_620, unsqueeze_2011);  sub_620 = unsqueeze_2011 = None
    mul_2104: "f32[28]" = torch.ops.aten.mul.Tensor(sum_237, squeeze_94);  sum_237 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_117 = torch.ops.aten.convolution_backward.default(mul_2103, getitem_297, primals_94, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2103 = getitem_297 = primals_94 = None
    getitem_1803: "f32[8, 28, 56, 56]" = convolution_backward_117[0]
    getitem_1804: "f32[28, 28, 3, 3]" = convolution_backward_117[1];  convolution_backward_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_115: "f32[8, 28, 28, 28]" = torch.ops.aten.where.self(le_115, full_default, slice_97);  le_115 = slice_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_238: "f32[28]" = torch.ops.aten.sum.dim_IntList(where_115, [0, 2, 3])
    sub_621: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_2014);  convolution_30 = unsqueeze_2014 = None
    mul_2105: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(where_115, sub_621)
    sum_239: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_2105, [0, 2, 3]);  mul_2105 = None
    mul_2106: "f32[28]" = torch.ops.aten.mul.Tensor(sum_238, 0.00015943877551020407)
    unsqueeze_2015: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2106, 0);  mul_2106 = None
    unsqueeze_2016: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2015, 2);  unsqueeze_2015 = None
    unsqueeze_2017: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2016, 3);  unsqueeze_2016 = None
    mul_2107: "f32[28]" = torch.ops.aten.mul.Tensor(sum_239, 0.00015943877551020407)
    mul_2108: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_2109: "f32[28]" = torch.ops.aten.mul.Tensor(mul_2107, mul_2108);  mul_2107 = mul_2108 = None
    unsqueeze_2018: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2109, 0);  mul_2109 = None
    unsqueeze_2019: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2018, 2);  unsqueeze_2018 = None
    unsqueeze_2020: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2019, 3);  unsqueeze_2019 = None
    mul_2110: "f32[28]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_92);  primals_92 = None
    unsqueeze_2021: "f32[1, 28]" = torch.ops.aten.unsqueeze.default(mul_2110, 0);  mul_2110 = None
    unsqueeze_2022: "f32[1, 28, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2021, 2);  unsqueeze_2021 = None
    unsqueeze_2023: "f32[1, 28, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2022, 3);  unsqueeze_2022 = None
    mul_2111: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_621, unsqueeze_2020);  sub_621 = unsqueeze_2020 = None
    sub_623: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(where_115, mul_2111);  where_115 = mul_2111 = None
    sub_624: "f32[8, 28, 28, 28]" = torch.ops.aten.sub.Tensor(sub_623, unsqueeze_2017);  sub_623 = unsqueeze_2017 = None
    mul_2112: "f32[8, 28, 28, 28]" = torch.ops.aten.mul.Tensor(sub_624, unsqueeze_2023);  sub_624 = unsqueeze_2023 = None
    mul_2113: "f32[28]" = torch.ops.aten.mul.Tensor(sum_239, squeeze_91);  sum_239 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_118 = torch.ops.aten.convolution_backward.default(mul_2112, getitem_286, primals_91, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2112 = getitem_286 = primals_91 = None
    getitem_1806: "f32[8, 28, 56, 56]" = convolution_backward_118[0]
    getitem_1807: "f32[28, 28, 3, 3]" = convolution_backward_118[1];  convolution_backward_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_28: "f32[8, 224, 56, 56]" = torch.ops.aten.cat.default([getitem_1806, getitem_1803, getitem_1800, getitem_1797, getitem_1794, getitem_1791, getitem_1788, avg_pool2d_backward_2], 1);  getitem_1806 = getitem_1803 = getitem_1800 = getitem_1797 = getitem_1794 = getitem_1791 = getitem_1788 = avg_pool2d_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_116: "f32[8, 224, 56, 56]" = torch.ops.aten.where.self(le_116, full_default, cat_28);  le_116 = cat_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_240: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_116, [0, 2, 3])
    sub_625: "f32[8, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_2026);  convolution_29 = unsqueeze_2026 = None
    mul_2114: "f32[8, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_116, sub_625)
    sum_241: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_2114, [0, 2, 3]);  mul_2114 = None
    mul_2115: "f32[224]" = torch.ops.aten.mul.Tensor(sum_240, 3.985969387755102e-05)
    unsqueeze_2027: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_2115, 0);  mul_2115 = None
    unsqueeze_2028: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2027, 2);  unsqueeze_2027 = None
    unsqueeze_2029: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2028, 3);  unsqueeze_2028 = None
    mul_2116: "f32[224]" = torch.ops.aten.mul.Tensor(sum_241, 3.985969387755102e-05)
    mul_2117: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_2118: "f32[224]" = torch.ops.aten.mul.Tensor(mul_2116, mul_2117);  mul_2116 = mul_2117 = None
    unsqueeze_2030: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_2118, 0);  mul_2118 = None
    unsqueeze_2031: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2030, 2);  unsqueeze_2030 = None
    unsqueeze_2032: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2031, 3);  unsqueeze_2031 = None
    mul_2119: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_89);  primals_89 = None
    unsqueeze_2033: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_2119, 0);  mul_2119 = None
    unsqueeze_2034: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2033, 2);  unsqueeze_2033 = None
    unsqueeze_2035: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2034, 3);  unsqueeze_2034 = None
    mul_2120: "f32[8, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_625, unsqueeze_2032);  sub_625 = unsqueeze_2032 = None
    sub_627: "f32[8, 224, 56, 56]" = torch.ops.aten.sub.Tensor(where_116, mul_2120);  where_116 = mul_2120 = None
    sub_628: "f32[8, 224, 56, 56]" = torch.ops.aten.sub.Tensor(sub_627, unsqueeze_2029);  sub_627 = unsqueeze_2029 = None
    mul_2121: "f32[8, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_628, unsqueeze_2035);  sub_628 = unsqueeze_2035 = None
    mul_2122: "f32[224]" = torch.ops.aten.mul.Tensor(sum_241, squeeze_88);  sum_241 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_119 = torch.ops.aten.convolution_backward.default(mul_2121, relu_27, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2121 = primals_88 = None
    getitem_1809: "f32[8, 256, 56, 56]" = convolution_backward_119[0]
    getitem_1810: "f32[224, 256, 1, 1]" = convolution_backward_119[1];  convolution_backward_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_905: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(getitem_1782, getitem_1809);  getitem_1782 = getitem_1809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_497: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_498: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_497);  alias_497 = None
    le_117: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_498, 0);  alias_498 = None
    where_117: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_117, full_default, add_905);  le_117 = add_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_242: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_117, [0, 2, 3])
    sub_629: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_2038);  convolution_28 = unsqueeze_2038 = None
    mul_2123: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_117, sub_629)
    sum_243: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2123, [0, 2, 3]);  mul_2123 = None
    mul_2124: "f32[256]" = torch.ops.aten.mul.Tensor(sum_242, 3.985969387755102e-05)
    unsqueeze_2039: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2124, 0);  mul_2124 = None
    unsqueeze_2040: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2039, 2);  unsqueeze_2039 = None
    unsqueeze_2041: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2040, 3);  unsqueeze_2040 = None
    mul_2125: "f32[256]" = torch.ops.aten.mul.Tensor(sum_243, 3.985969387755102e-05)
    mul_2126: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_2127: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2125, mul_2126);  mul_2125 = mul_2126 = None
    unsqueeze_2042: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2127, 0);  mul_2127 = None
    unsqueeze_2043: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2042, 2);  unsqueeze_2042 = None
    unsqueeze_2044: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2043, 3);  unsqueeze_2043 = None
    mul_2128: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_86);  primals_86 = None
    unsqueeze_2045: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2128, 0);  mul_2128 = None
    unsqueeze_2046: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2045, 2);  unsqueeze_2045 = None
    unsqueeze_2047: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2046, 3);  unsqueeze_2046 = None
    mul_2129: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_629, unsqueeze_2044);  sub_629 = unsqueeze_2044 = None
    sub_631: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_117, mul_2129);  mul_2129 = None
    sub_632: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_631, unsqueeze_2041);  sub_631 = unsqueeze_2041 = None
    mul_2130: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_632, unsqueeze_2047);  sub_632 = unsqueeze_2047 = None
    mul_2131: "f32[256]" = torch.ops.aten.mul.Tensor(sum_243, squeeze_85);  sum_243 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_120 = torch.ops.aten.convolution_backward.default(mul_2130, cat_2, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2130 = cat_2 = primals_85 = None
    getitem_1812: "f32[8, 112, 56, 56]" = convolution_backward_120[0]
    getitem_1813: "f32[256, 112, 1, 1]" = convolution_backward_120[1];  convolution_backward_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_105: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1812, 1, 0, 14)
    slice_106: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1812, 1, 14, 28)
    slice_107: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1812, 1, 28, 42)
    slice_108: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1812, 1, 42, 56)
    slice_109: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1812, 1, 56, 70)
    slice_110: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1812, 1, 70, 84)
    slice_111: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1812, 1, 84, 98)
    slice_112: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1812, 1, 98, 112);  getitem_1812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_118: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_118, full_default, slice_111);  le_118 = slice_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_244: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_118, [0, 2, 3])
    sub_633: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_2050);  convolution_27 = unsqueeze_2050 = None
    mul_2132: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_118, sub_633)
    sum_245: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2132, [0, 2, 3]);  mul_2132 = None
    mul_2133: "f32[14]" = torch.ops.aten.mul.Tensor(sum_244, 3.985969387755102e-05)
    unsqueeze_2051: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2133, 0);  mul_2133 = None
    unsqueeze_2052: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2051, 2);  unsqueeze_2051 = None
    unsqueeze_2053: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2052, 3);  unsqueeze_2052 = None
    mul_2134: "f32[14]" = torch.ops.aten.mul.Tensor(sum_245, 3.985969387755102e-05)
    mul_2135: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_2136: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2134, mul_2135);  mul_2134 = mul_2135 = None
    unsqueeze_2054: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2136, 0);  mul_2136 = None
    unsqueeze_2055: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2054, 2);  unsqueeze_2054 = None
    unsqueeze_2056: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2055, 3);  unsqueeze_2055 = None
    mul_2137: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_83);  primals_83 = None
    unsqueeze_2057: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2137, 0);  mul_2137 = None
    unsqueeze_2058: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2057, 2);  unsqueeze_2057 = None
    unsqueeze_2059: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2058, 3);  unsqueeze_2058 = None
    mul_2138: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_633, unsqueeze_2056);  sub_633 = unsqueeze_2056 = None
    sub_635: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_118, mul_2138);  where_118 = mul_2138 = None
    sub_636: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_635, unsqueeze_2053);  sub_635 = unsqueeze_2053 = None
    mul_2139: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_636, unsqueeze_2059);  sub_636 = unsqueeze_2059 = None
    mul_2140: "f32[14]" = torch.ops.aten.mul.Tensor(sum_245, squeeze_82);  sum_245 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_121 = torch.ops.aten.convolution_backward.default(mul_2139, add_148, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2139 = add_148 = primals_82 = None
    getitem_1815: "f32[8, 14, 56, 56]" = convolution_backward_121[0]
    getitem_1816: "f32[14, 14, 3, 3]" = convolution_backward_121[1];  convolution_backward_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_906: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(slice_110, getitem_1815);  slice_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_119: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_119, full_default, add_906);  le_119 = add_906 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_246: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_119, [0, 2, 3])
    sub_637: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_2062);  convolution_26 = unsqueeze_2062 = None
    mul_2141: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_119, sub_637)
    sum_247: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2141, [0, 2, 3]);  mul_2141 = None
    mul_2142: "f32[14]" = torch.ops.aten.mul.Tensor(sum_246, 3.985969387755102e-05)
    unsqueeze_2063: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2142, 0);  mul_2142 = None
    unsqueeze_2064: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2063, 2);  unsqueeze_2063 = None
    unsqueeze_2065: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2064, 3);  unsqueeze_2064 = None
    mul_2143: "f32[14]" = torch.ops.aten.mul.Tensor(sum_247, 3.985969387755102e-05)
    mul_2144: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_2145: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2143, mul_2144);  mul_2143 = mul_2144 = None
    unsqueeze_2066: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2145, 0);  mul_2145 = None
    unsqueeze_2067: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2066, 2);  unsqueeze_2066 = None
    unsqueeze_2068: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2067, 3);  unsqueeze_2067 = None
    mul_2146: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_80);  primals_80 = None
    unsqueeze_2069: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2146, 0);  mul_2146 = None
    unsqueeze_2070: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2069, 2);  unsqueeze_2069 = None
    unsqueeze_2071: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2070, 3);  unsqueeze_2070 = None
    mul_2147: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_637, unsqueeze_2068);  sub_637 = unsqueeze_2068 = None
    sub_639: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_119, mul_2147);  where_119 = mul_2147 = None
    sub_640: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_639, unsqueeze_2065);  sub_639 = unsqueeze_2065 = None
    mul_2148: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_640, unsqueeze_2071);  sub_640 = unsqueeze_2071 = None
    mul_2149: "f32[14]" = torch.ops.aten.mul.Tensor(sum_247, squeeze_79);  sum_247 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_122 = torch.ops.aten.convolution_backward.default(mul_2148, add_142, primals_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2148 = add_142 = primals_79 = None
    getitem_1818: "f32[8, 14, 56, 56]" = convolution_backward_122[0]
    getitem_1819: "f32[14, 14, 3, 3]" = convolution_backward_122[1];  convolution_backward_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_907: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(slice_109, getitem_1818);  slice_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_120: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_120, full_default, add_907);  le_120 = add_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_248: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_120, [0, 2, 3])
    sub_641: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_2074);  convolution_25 = unsqueeze_2074 = None
    mul_2150: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_120, sub_641)
    sum_249: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2150, [0, 2, 3]);  mul_2150 = None
    mul_2151: "f32[14]" = torch.ops.aten.mul.Tensor(sum_248, 3.985969387755102e-05)
    unsqueeze_2075: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2151, 0);  mul_2151 = None
    unsqueeze_2076: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2075, 2);  unsqueeze_2075 = None
    unsqueeze_2077: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2076, 3);  unsqueeze_2076 = None
    mul_2152: "f32[14]" = torch.ops.aten.mul.Tensor(sum_249, 3.985969387755102e-05)
    mul_2153: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_2154: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2152, mul_2153);  mul_2152 = mul_2153 = None
    unsqueeze_2078: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2154, 0);  mul_2154 = None
    unsqueeze_2079: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2078, 2);  unsqueeze_2078 = None
    unsqueeze_2080: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2079, 3);  unsqueeze_2079 = None
    mul_2155: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_77);  primals_77 = None
    unsqueeze_2081: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2155, 0);  mul_2155 = None
    unsqueeze_2082: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2081, 2);  unsqueeze_2081 = None
    unsqueeze_2083: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2082, 3);  unsqueeze_2082 = None
    mul_2156: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_641, unsqueeze_2080);  sub_641 = unsqueeze_2080 = None
    sub_643: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_120, mul_2156);  where_120 = mul_2156 = None
    sub_644: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_643, unsqueeze_2077);  sub_643 = unsqueeze_2077 = None
    mul_2157: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_644, unsqueeze_2083);  sub_644 = unsqueeze_2083 = None
    mul_2158: "f32[14]" = torch.ops.aten.mul.Tensor(sum_249, squeeze_76);  sum_249 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_123 = torch.ops.aten.convolution_backward.default(mul_2157, add_136, primals_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2157 = add_136 = primals_76 = None
    getitem_1821: "f32[8, 14, 56, 56]" = convolution_backward_123[0]
    getitem_1822: "f32[14, 14, 3, 3]" = convolution_backward_123[1];  convolution_backward_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_908: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(slice_108, getitem_1821);  slice_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_121: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_121, full_default, add_908);  le_121 = add_908 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_250: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_121, [0, 2, 3])
    sub_645: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_2086);  convolution_24 = unsqueeze_2086 = None
    mul_2159: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_121, sub_645)
    sum_251: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2159, [0, 2, 3]);  mul_2159 = None
    mul_2160: "f32[14]" = torch.ops.aten.mul.Tensor(sum_250, 3.985969387755102e-05)
    unsqueeze_2087: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2160, 0);  mul_2160 = None
    unsqueeze_2088: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2087, 2);  unsqueeze_2087 = None
    unsqueeze_2089: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2088, 3);  unsqueeze_2088 = None
    mul_2161: "f32[14]" = torch.ops.aten.mul.Tensor(sum_251, 3.985969387755102e-05)
    mul_2162: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_2163: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2161, mul_2162);  mul_2161 = mul_2162 = None
    unsqueeze_2090: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2163, 0);  mul_2163 = None
    unsqueeze_2091: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2090, 2);  unsqueeze_2090 = None
    unsqueeze_2092: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2091, 3);  unsqueeze_2091 = None
    mul_2164: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_74);  primals_74 = None
    unsqueeze_2093: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2164, 0);  mul_2164 = None
    unsqueeze_2094: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2093, 2);  unsqueeze_2093 = None
    unsqueeze_2095: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2094, 3);  unsqueeze_2094 = None
    mul_2165: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_645, unsqueeze_2092);  sub_645 = unsqueeze_2092 = None
    sub_647: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_121, mul_2165);  where_121 = mul_2165 = None
    sub_648: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_647, unsqueeze_2089);  sub_647 = unsqueeze_2089 = None
    mul_2166: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_648, unsqueeze_2095);  sub_648 = unsqueeze_2095 = None
    mul_2167: "f32[14]" = torch.ops.aten.mul.Tensor(sum_251, squeeze_73);  sum_251 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_124 = torch.ops.aten.convolution_backward.default(mul_2166, add_130, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2166 = add_130 = primals_73 = None
    getitem_1824: "f32[8, 14, 56, 56]" = convolution_backward_124[0]
    getitem_1825: "f32[14, 14, 3, 3]" = convolution_backward_124[1];  convolution_backward_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_909: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(slice_107, getitem_1824);  slice_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_122: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_122, full_default, add_909);  le_122 = add_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_252: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_122, [0, 2, 3])
    sub_649: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_2098);  convolution_23 = unsqueeze_2098 = None
    mul_2168: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_122, sub_649)
    sum_253: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2168, [0, 2, 3]);  mul_2168 = None
    mul_2169: "f32[14]" = torch.ops.aten.mul.Tensor(sum_252, 3.985969387755102e-05)
    unsqueeze_2099: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2169, 0);  mul_2169 = None
    unsqueeze_2100: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2099, 2);  unsqueeze_2099 = None
    unsqueeze_2101: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2100, 3);  unsqueeze_2100 = None
    mul_2170: "f32[14]" = torch.ops.aten.mul.Tensor(sum_253, 3.985969387755102e-05)
    mul_2171: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_2172: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2170, mul_2171);  mul_2170 = mul_2171 = None
    unsqueeze_2102: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2172, 0);  mul_2172 = None
    unsqueeze_2103: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2102, 2);  unsqueeze_2102 = None
    unsqueeze_2104: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2103, 3);  unsqueeze_2103 = None
    mul_2173: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_71);  primals_71 = None
    unsqueeze_2105: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2173, 0);  mul_2173 = None
    unsqueeze_2106: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2105, 2);  unsqueeze_2105 = None
    unsqueeze_2107: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2106, 3);  unsqueeze_2106 = None
    mul_2174: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_649, unsqueeze_2104);  sub_649 = unsqueeze_2104 = None
    sub_651: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_122, mul_2174);  where_122 = mul_2174 = None
    sub_652: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_651, unsqueeze_2101);  sub_651 = unsqueeze_2101 = None
    mul_2175: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_652, unsqueeze_2107);  sub_652 = unsqueeze_2107 = None
    mul_2176: "f32[14]" = torch.ops.aten.mul.Tensor(sum_253, squeeze_70);  sum_253 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_125 = torch.ops.aten.convolution_backward.default(mul_2175, add_124, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2175 = add_124 = primals_70 = None
    getitem_1827: "f32[8, 14, 56, 56]" = convolution_backward_125[0]
    getitem_1828: "f32[14, 14, 3, 3]" = convolution_backward_125[1];  convolution_backward_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_910: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(slice_106, getitem_1827);  slice_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_123: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_123, full_default, add_910);  le_123 = add_910 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_254: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_123, [0, 2, 3])
    sub_653: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_2110);  convolution_22 = unsqueeze_2110 = None
    mul_2177: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_123, sub_653)
    sum_255: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2177, [0, 2, 3]);  mul_2177 = None
    mul_2178: "f32[14]" = torch.ops.aten.mul.Tensor(sum_254, 3.985969387755102e-05)
    unsqueeze_2111: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2178, 0);  mul_2178 = None
    unsqueeze_2112: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2111, 2);  unsqueeze_2111 = None
    unsqueeze_2113: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2112, 3);  unsqueeze_2112 = None
    mul_2179: "f32[14]" = torch.ops.aten.mul.Tensor(sum_255, 3.985969387755102e-05)
    mul_2180: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_2181: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2179, mul_2180);  mul_2179 = mul_2180 = None
    unsqueeze_2114: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2181, 0);  mul_2181 = None
    unsqueeze_2115: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2114, 2);  unsqueeze_2114 = None
    unsqueeze_2116: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2115, 3);  unsqueeze_2115 = None
    mul_2182: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_68);  primals_68 = None
    unsqueeze_2117: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2182, 0);  mul_2182 = None
    unsqueeze_2118: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2117, 2);  unsqueeze_2117 = None
    unsqueeze_2119: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2118, 3);  unsqueeze_2118 = None
    mul_2183: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_653, unsqueeze_2116);  sub_653 = unsqueeze_2116 = None
    sub_655: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_123, mul_2183);  where_123 = mul_2183 = None
    sub_656: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_655, unsqueeze_2113);  sub_655 = unsqueeze_2113 = None
    mul_2184: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_656, unsqueeze_2119);  sub_656 = unsqueeze_2119 = None
    mul_2185: "f32[14]" = torch.ops.aten.mul.Tensor(sum_255, squeeze_67);  sum_255 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_126 = torch.ops.aten.convolution_backward.default(mul_2184, add_118, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2184 = add_118 = primals_67 = None
    getitem_1830: "f32[8, 14, 56, 56]" = convolution_backward_126[0]
    getitem_1831: "f32[14, 14, 3, 3]" = convolution_backward_126[1];  convolution_backward_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_911: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(slice_105, getitem_1830);  slice_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_124: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_124, full_default, add_911);  le_124 = add_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_256: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_124, [0, 2, 3])
    sub_657: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_2122);  convolution_21 = unsqueeze_2122 = None
    mul_2186: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_124, sub_657)
    sum_257: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2186, [0, 2, 3]);  mul_2186 = None
    mul_2187: "f32[14]" = torch.ops.aten.mul.Tensor(sum_256, 3.985969387755102e-05)
    unsqueeze_2123: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2187, 0);  mul_2187 = None
    unsqueeze_2124: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2123, 2);  unsqueeze_2123 = None
    unsqueeze_2125: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2124, 3);  unsqueeze_2124 = None
    mul_2188: "f32[14]" = torch.ops.aten.mul.Tensor(sum_257, 3.985969387755102e-05)
    mul_2189: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_2190: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2188, mul_2189);  mul_2188 = mul_2189 = None
    unsqueeze_2126: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2190, 0);  mul_2190 = None
    unsqueeze_2127: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2126, 2);  unsqueeze_2126 = None
    unsqueeze_2128: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2127, 3);  unsqueeze_2127 = None
    mul_2191: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_65);  primals_65 = None
    unsqueeze_2129: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2191, 0);  mul_2191 = None
    unsqueeze_2130: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2129, 2);  unsqueeze_2129 = None
    unsqueeze_2131: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2130, 3);  unsqueeze_2130 = None
    mul_2192: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_657, unsqueeze_2128);  sub_657 = unsqueeze_2128 = None
    sub_659: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_124, mul_2192);  where_124 = mul_2192 = None
    sub_660: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_659, unsqueeze_2125);  sub_659 = unsqueeze_2125 = None
    mul_2193: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_660, unsqueeze_2131);  sub_660 = unsqueeze_2131 = None
    mul_2194: "f32[14]" = torch.ops.aten.mul.Tensor(sum_257, squeeze_64);  sum_257 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_127 = torch.ops.aten.convolution_backward.default(mul_2193, getitem_196, primals_64, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2193 = getitem_196 = primals_64 = None
    getitem_1833: "f32[8, 14, 56, 56]" = convolution_backward_127[0]
    getitem_1834: "f32[14, 14, 3, 3]" = convolution_backward_127[1];  convolution_backward_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_29: "f32[8, 112, 56, 56]" = torch.ops.aten.cat.default([getitem_1833, getitem_1830, getitem_1827, getitem_1824, getitem_1821, getitem_1818, getitem_1815, slice_112], 1);  getitem_1833 = getitem_1830 = getitem_1827 = getitem_1824 = getitem_1821 = getitem_1818 = getitem_1815 = slice_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_125: "f32[8, 112, 56, 56]" = torch.ops.aten.where.self(le_125, full_default, cat_29);  le_125 = cat_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_258: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_125, [0, 2, 3])
    sub_661: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_2134);  convolution_20 = unsqueeze_2134 = None
    mul_2195: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(where_125, sub_661)
    sum_259: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_2195, [0, 2, 3]);  mul_2195 = None
    mul_2196: "f32[112]" = torch.ops.aten.mul.Tensor(sum_258, 3.985969387755102e-05)
    unsqueeze_2135: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_2196, 0);  mul_2196 = None
    unsqueeze_2136: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2135, 2);  unsqueeze_2135 = None
    unsqueeze_2137: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2136, 3);  unsqueeze_2136 = None
    mul_2197: "f32[112]" = torch.ops.aten.mul.Tensor(sum_259, 3.985969387755102e-05)
    mul_2198: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_2199: "f32[112]" = torch.ops.aten.mul.Tensor(mul_2197, mul_2198);  mul_2197 = mul_2198 = None
    unsqueeze_2138: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_2199, 0);  mul_2199 = None
    unsqueeze_2139: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2138, 2);  unsqueeze_2138 = None
    unsqueeze_2140: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2139, 3);  unsqueeze_2139 = None
    mul_2200: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_62);  primals_62 = None
    unsqueeze_2141: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_2200, 0);  mul_2200 = None
    unsqueeze_2142: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2141, 2);  unsqueeze_2141 = None
    unsqueeze_2143: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2142, 3);  unsqueeze_2142 = None
    mul_2201: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(sub_661, unsqueeze_2140);  sub_661 = unsqueeze_2140 = None
    sub_663: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(where_125, mul_2201);  where_125 = mul_2201 = None
    sub_664: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(sub_663, unsqueeze_2137);  sub_663 = unsqueeze_2137 = None
    mul_2202: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(sub_664, unsqueeze_2143);  sub_664 = unsqueeze_2143 = None
    mul_2203: "f32[112]" = torch.ops.aten.mul.Tensor(sum_259, squeeze_61);  sum_259 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_128 = torch.ops.aten.convolution_backward.default(mul_2202, relu_18, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2202 = primals_61 = None
    getitem_1836: "f32[8, 256, 56, 56]" = convolution_backward_128[0]
    getitem_1837: "f32[112, 256, 1, 1]" = convolution_backward_128[1];  convolution_backward_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_912: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_117, getitem_1836);  where_117 = getitem_1836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_524: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_525: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_524);  alias_524 = None
    le_126: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_525, 0);  alias_525 = None
    where_126: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_126, full_default, add_912);  le_126 = add_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_260: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_126, [0, 2, 3])
    sub_665: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_2146);  convolution_19 = unsqueeze_2146 = None
    mul_2204: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_126, sub_665)
    sum_261: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2204, [0, 2, 3]);  mul_2204 = None
    mul_2205: "f32[256]" = torch.ops.aten.mul.Tensor(sum_260, 3.985969387755102e-05)
    unsqueeze_2147: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2205, 0);  mul_2205 = None
    unsqueeze_2148: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2147, 2);  unsqueeze_2147 = None
    unsqueeze_2149: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2148, 3);  unsqueeze_2148 = None
    mul_2206: "f32[256]" = torch.ops.aten.mul.Tensor(sum_261, 3.985969387755102e-05)
    mul_2207: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_2208: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2206, mul_2207);  mul_2206 = mul_2207 = None
    unsqueeze_2150: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2208, 0);  mul_2208 = None
    unsqueeze_2151: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2150, 2);  unsqueeze_2150 = None
    unsqueeze_2152: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2151, 3);  unsqueeze_2151 = None
    mul_2209: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_59);  primals_59 = None
    unsqueeze_2153: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2209, 0);  mul_2209 = None
    unsqueeze_2154: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2153, 2);  unsqueeze_2153 = None
    unsqueeze_2155: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2154, 3);  unsqueeze_2154 = None
    mul_2210: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_665, unsqueeze_2152);  sub_665 = unsqueeze_2152 = None
    sub_667: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_126, mul_2210);  mul_2210 = None
    sub_668: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_667, unsqueeze_2149);  sub_667 = unsqueeze_2149 = None
    mul_2211: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_668, unsqueeze_2155);  sub_668 = unsqueeze_2155 = None
    mul_2212: "f32[256]" = torch.ops.aten.mul.Tensor(sum_261, squeeze_58);  sum_261 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_129 = torch.ops.aten.convolution_backward.default(mul_2211, cat_1, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2211 = cat_1 = primals_58 = None
    getitem_1839: "f32[8, 112, 56, 56]" = convolution_backward_129[0]
    getitem_1840: "f32[256, 112, 1, 1]" = convolution_backward_129[1];  convolution_backward_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_113: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1839, 1, 0, 14)
    slice_114: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1839, 1, 14, 28)
    slice_115: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1839, 1, 28, 42)
    slice_116: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1839, 1, 42, 56)
    slice_117: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1839, 1, 56, 70)
    slice_118: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1839, 1, 70, 84)
    slice_119: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1839, 1, 84, 98)
    slice_120: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1839, 1, 98, 112);  getitem_1839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_127: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_127, full_default, slice_119);  le_127 = slice_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_262: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_127, [0, 2, 3])
    sub_669: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_2158);  convolution_18 = unsqueeze_2158 = None
    mul_2213: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_127, sub_669)
    sum_263: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2213, [0, 2, 3]);  mul_2213 = None
    mul_2214: "f32[14]" = torch.ops.aten.mul.Tensor(sum_262, 3.985969387755102e-05)
    unsqueeze_2159: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2214, 0);  mul_2214 = None
    unsqueeze_2160: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2159, 2);  unsqueeze_2159 = None
    unsqueeze_2161: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2160, 3);  unsqueeze_2160 = None
    mul_2215: "f32[14]" = torch.ops.aten.mul.Tensor(sum_263, 3.985969387755102e-05)
    mul_2216: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_2217: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2215, mul_2216);  mul_2215 = mul_2216 = None
    unsqueeze_2162: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2217, 0);  mul_2217 = None
    unsqueeze_2163: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2162, 2);  unsqueeze_2162 = None
    unsqueeze_2164: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2163, 3);  unsqueeze_2163 = None
    mul_2218: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_56);  primals_56 = None
    unsqueeze_2165: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2218, 0);  mul_2218 = None
    unsqueeze_2166: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2165, 2);  unsqueeze_2165 = None
    unsqueeze_2167: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2166, 3);  unsqueeze_2166 = None
    mul_2219: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_669, unsqueeze_2164);  sub_669 = unsqueeze_2164 = None
    sub_671: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_127, mul_2219);  where_127 = mul_2219 = None
    sub_672: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_671, unsqueeze_2161);  sub_671 = unsqueeze_2161 = None
    mul_2220: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_672, unsqueeze_2167);  sub_672 = unsqueeze_2167 = None
    mul_2221: "f32[14]" = torch.ops.aten.mul.Tensor(sum_263, squeeze_55);  sum_263 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_130 = torch.ops.aten.convolution_backward.default(mul_2220, add_96, primals_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2220 = add_96 = primals_55 = None
    getitem_1842: "f32[8, 14, 56, 56]" = convolution_backward_130[0]
    getitem_1843: "f32[14, 14, 3, 3]" = convolution_backward_130[1];  convolution_backward_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_913: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(slice_118, getitem_1842);  slice_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_128: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_128, full_default, add_913);  le_128 = add_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_264: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_128, [0, 2, 3])
    sub_673: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_2170);  convolution_17 = unsqueeze_2170 = None
    mul_2222: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_128, sub_673)
    sum_265: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2222, [0, 2, 3]);  mul_2222 = None
    mul_2223: "f32[14]" = torch.ops.aten.mul.Tensor(sum_264, 3.985969387755102e-05)
    unsqueeze_2171: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2223, 0);  mul_2223 = None
    unsqueeze_2172: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2171, 2);  unsqueeze_2171 = None
    unsqueeze_2173: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2172, 3);  unsqueeze_2172 = None
    mul_2224: "f32[14]" = torch.ops.aten.mul.Tensor(sum_265, 3.985969387755102e-05)
    mul_2225: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_2226: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2224, mul_2225);  mul_2224 = mul_2225 = None
    unsqueeze_2174: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2226, 0);  mul_2226 = None
    unsqueeze_2175: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2174, 2);  unsqueeze_2174 = None
    unsqueeze_2176: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2175, 3);  unsqueeze_2175 = None
    mul_2227: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_53);  primals_53 = None
    unsqueeze_2177: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2227, 0);  mul_2227 = None
    unsqueeze_2178: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2177, 2);  unsqueeze_2177 = None
    unsqueeze_2179: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2178, 3);  unsqueeze_2178 = None
    mul_2228: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_673, unsqueeze_2176);  sub_673 = unsqueeze_2176 = None
    sub_675: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_128, mul_2228);  where_128 = mul_2228 = None
    sub_676: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_675, unsqueeze_2173);  sub_675 = unsqueeze_2173 = None
    mul_2229: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_676, unsqueeze_2179);  sub_676 = unsqueeze_2179 = None
    mul_2230: "f32[14]" = torch.ops.aten.mul.Tensor(sum_265, squeeze_52);  sum_265 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_131 = torch.ops.aten.convolution_backward.default(mul_2229, add_90, primals_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2229 = add_90 = primals_52 = None
    getitem_1845: "f32[8, 14, 56, 56]" = convolution_backward_131[0]
    getitem_1846: "f32[14, 14, 3, 3]" = convolution_backward_131[1];  convolution_backward_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_914: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(slice_117, getitem_1845);  slice_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_129: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_129, full_default, add_914);  le_129 = add_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_266: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_129, [0, 2, 3])
    sub_677: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_2182);  convolution_16 = unsqueeze_2182 = None
    mul_2231: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_129, sub_677)
    sum_267: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2231, [0, 2, 3]);  mul_2231 = None
    mul_2232: "f32[14]" = torch.ops.aten.mul.Tensor(sum_266, 3.985969387755102e-05)
    unsqueeze_2183: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2232, 0);  mul_2232 = None
    unsqueeze_2184: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2183, 2);  unsqueeze_2183 = None
    unsqueeze_2185: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2184, 3);  unsqueeze_2184 = None
    mul_2233: "f32[14]" = torch.ops.aten.mul.Tensor(sum_267, 3.985969387755102e-05)
    mul_2234: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_2235: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2233, mul_2234);  mul_2233 = mul_2234 = None
    unsqueeze_2186: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2235, 0);  mul_2235 = None
    unsqueeze_2187: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2186, 2);  unsqueeze_2186 = None
    unsqueeze_2188: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2187, 3);  unsqueeze_2187 = None
    mul_2236: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_50);  primals_50 = None
    unsqueeze_2189: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2236, 0);  mul_2236 = None
    unsqueeze_2190: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2189, 2);  unsqueeze_2189 = None
    unsqueeze_2191: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2190, 3);  unsqueeze_2190 = None
    mul_2237: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_677, unsqueeze_2188);  sub_677 = unsqueeze_2188 = None
    sub_679: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_129, mul_2237);  where_129 = mul_2237 = None
    sub_680: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_679, unsqueeze_2185);  sub_679 = unsqueeze_2185 = None
    mul_2238: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_680, unsqueeze_2191);  sub_680 = unsqueeze_2191 = None
    mul_2239: "f32[14]" = torch.ops.aten.mul.Tensor(sum_267, squeeze_49);  sum_267 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_132 = torch.ops.aten.convolution_backward.default(mul_2238, add_84, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2238 = add_84 = primals_49 = None
    getitem_1848: "f32[8, 14, 56, 56]" = convolution_backward_132[0]
    getitem_1849: "f32[14, 14, 3, 3]" = convolution_backward_132[1];  convolution_backward_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_915: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(slice_116, getitem_1848);  slice_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_130: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_130, full_default, add_915);  le_130 = add_915 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_268: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_130, [0, 2, 3])
    sub_681: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_2194);  convolution_15 = unsqueeze_2194 = None
    mul_2240: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_130, sub_681)
    sum_269: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2240, [0, 2, 3]);  mul_2240 = None
    mul_2241: "f32[14]" = torch.ops.aten.mul.Tensor(sum_268, 3.985969387755102e-05)
    unsqueeze_2195: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2241, 0);  mul_2241 = None
    unsqueeze_2196: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2195, 2);  unsqueeze_2195 = None
    unsqueeze_2197: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2196, 3);  unsqueeze_2196 = None
    mul_2242: "f32[14]" = torch.ops.aten.mul.Tensor(sum_269, 3.985969387755102e-05)
    mul_2243: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_2244: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2242, mul_2243);  mul_2242 = mul_2243 = None
    unsqueeze_2198: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2244, 0);  mul_2244 = None
    unsqueeze_2199: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2198, 2);  unsqueeze_2198 = None
    unsqueeze_2200: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2199, 3);  unsqueeze_2199 = None
    mul_2245: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_47);  primals_47 = None
    unsqueeze_2201: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2245, 0);  mul_2245 = None
    unsqueeze_2202: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2201, 2);  unsqueeze_2201 = None
    unsqueeze_2203: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2202, 3);  unsqueeze_2202 = None
    mul_2246: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_681, unsqueeze_2200);  sub_681 = unsqueeze_2200 = None
    sub_683: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_130, mul_2246);  where_130 = mul_2246 = None
    sub_684: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_683, unsqueeze_2197);  sub_683 = unsqueeze_2197 = None
    mul_2247: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_684, unsqueeze_2203);  sub_684 = unsqueeze_2203 = None
    mul_2248: "f32[14]" = torch.ops.aten.mul.Tensor(sum_269, squeeze_46);  sum_269 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_133 = torch.ops.aten.convolution_backward.default(mul_2247, add_78, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2247 = add_78 = primals_46 = None
    getitem_1851: "f32[8, 14, 56, 56]" = convolution_backward_133[0]
    getitem_1852: "f32[14, 14, 3, 3]" = convolution_backward_133[1];  convolution_backward_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_916: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(slice_115, getitem_1851);  slice_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_131: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_131, full_default, add_916);  le_131 = add_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_270: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_131, [0, 2, 3])
    sub_685: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_2206);  convolution_14 = unsqueeze_2206 = None
    mul_2249: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_131, sub_685)
    sum_271: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2249, [0, 2, 3]);  mul_2249 = None
    mul_2250: "f32[14]" = torch.ops.aten.mul.Tensor(sum_270, 3.985969387755102e-05)
    unsqueeze_2207: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2250, 0);  mul_2250 = None
    unsqueeze_2208: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2207, 2);  unsqueeze_2207 = None
    unsqueeze_2209: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2208, 3);  unsqueeze_2208 = None
    mul_2251: "f32[14]" = torch.ops.aten.mul.Tensor(sum_271, 3.985969387755102e-05)
    mul_2252: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_2253: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2251, mul_2252);  mul_2251 = mul_2252 = None
    unsqueeze_2210: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2253, 0);  mul_2253 = None
    unsqueeze_2211: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2210, 2);  unsqueeze_2210 = None
    unsqueeze_2212: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2211, 3);  unsqueeze_2211 = None
    mul_2254: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_44);  primals_44 = None
    unsqueeze_2213: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2254, 0);  mul_2254 = None
    unsqueeze_2214: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2213, 2);  unsqueeze_2213 = None
    unsqueeze_2215: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2214, 3);  unsqueeze_2214 = None
    mul_2255: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_685, unsqueeze_2212);  sub_685 = unsqueeze_2212 = None
    sub_687: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_131, mul_2255);  where_131 = mul_2255 = None
    sub_688: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_687, unsqueeze_2209);  sub_687 = unsqueeze_2209 = None
    mul_2256: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_688, unsqueeze_2215);  sub_688 = unsqueeze_2215 = None
    mul_2257: "f32[14]" = torch.ops.aten.mul.Tensor(sum_271, squeeze_43);  sum_271 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_134 = torch.ops.aten.convolution_backward.default(mul_2256, add_72, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2256 = add_72 = primals_43 = None
    getitem_1854: "f32[8, 14, 56, 56]" = convolution_backward_134[0]
    getitem_1855: "f32[14, 14, 3, 3]" = convolution_backward_134[1];  convolution_backward_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_917: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(slice_114, getitem_1854);  slice_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_132: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_132, full_default, add_917);  le_132 = add_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_272: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_132, [0, 2, 3])
    sub_689: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_2218);  convolution_13 = unsqueeze_2218 = None
    mul_2258: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_132, sub_689)
    sum_273: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2258, [0, 2, 3]);  mul_2258 = None
    mul_2259: "f32[14]" = torch.ops.aten.mul.Tensor(sum_272, 3.985969387755102e-05)
    unsqueeze_2219: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2259, 0);  mul_2259 = None
    unsqueeze_2220: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2219, 2);  unsqueeze_2219 = None
    unsqueeze_2221: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2220, 3);  unsqueeze_2220 = None
    mul_2260: "f32[14]" = torch.ops.aten.mul.Tensor(sum_273, 3.985969387755102e-05)
    mul_2261: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_2262: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2260, mul_2261);  mul_2260 = mul_2261 = None
    unsqueeze_2222: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2262, 0);  mul_2262 = None
    unsqueeze_2223: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2222, 2);  unsqueeze_2222 = None
    unsqueeze_2224: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2223, 3);  unsqueeze_2223 = None
    mul_2263: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_41);  primals_41 = None
    unsqueeze_2225: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2263, 0);  mul_2263 = None
    unsqueeze_2226: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2225, 2);  unsqueeze_2225 = None
    unsqueeze_2227: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2226, 3);  unsqueeze_2226 = None
    mul_2264: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_689, unsqueeze_2224);  sub_689 = unsqueeze_2224 = None
    sub_691: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_132, mul_2264);  where_132 = mul_2264 = None
    sub_692: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_691, unsqueeze_2221);  sub_691 = unsqueeze_2221 = None
    mul_2265: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_692, unsqueeze_2227);  sub_692 = unsqueeze_2227 = None
    mul_2266: "f32[14]" = torch.ops.aten.mul.Tensor(sum_273, squeeze_40);  sum_273 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_135 = torch.ops.aten.convolution_backward.default(mul_2265, add_66, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2265 = add_66 = primals_40 = None
    getitem_1857: "f32[8, 14, 56, 56]" = convolution_backward_135[0]
    getitem_1858: "f32[14, 14, 3, 3]" = convolution_backward_135[1];  convolution_backward_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_918: "f32[8, 14, 56, 56]" = torch.ops.aten.add.Tensor(slice_113, getitem_1857);  slice_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_133: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_133, full_default, add_918);  le_133 = add_918 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_274: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_133, [0, 2, 3])
    sub_693: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_2230);  convolution_12 = unsqueeze_2230 = None
    mul_2267: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_133, sub_693)
    sum_275: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2267, [0, 2, 3]);  mul_2267 = None
    mul_2268: "f32[14]" = torch.ops.aten.mul.Tensor(sum_274, 3.985969387755102e-05)
    unsqueeze_2231: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2268, 0);  mul_2268 = None
    unsqueeze_2232: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2231, 2);  unsqueeze_2231 = None
    unsqueeze_2233: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2232, 3);  unsqueeze_2232 = None
    mul_2269: "f32[14]" = torch.ops.aten.mul.Tensor(sum_275, 3.985969387755102e-05)
    mul_2270: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_2271: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2269, mul_2270);  mul_2269 = mul_2270 = None
    unsqueeze_2234: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2271, 0);  mul_2271 = None
    unsqueeze_2235: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2234, 2);  unsqueeze_2234 = None
    unsqueeze_2236: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2235, 3);  unsqueeze_2235 = None
    mul_2272: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_38);  primals_38 = None
    unsqueeze_2237: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2272, 0);  mul_2272 = None
    unsqueeze_2238: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2237, 2);  unsqueeze_2237 = None
    unsqueeze_2239: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2238, 3);  unsqueeze_2238 = None
    mul_2273: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_693, unsqueeze_2236);  sub_693 = unsqueeze_2236 = None
    sub_695: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_133, mul_2273);  where_133 = mul_2273 = None
    sub_696: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_695, unsqueeze_2233);  sub_695 = unsqueeze_2233 = None
    mul_2274: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_696, unsqueeze_2239);  sub_696 = unsqueeze_2239 = None
    mul_2275: "f32[14]" = torch.ops.aten.mul.Tensor(sum_275, squeeze_37);  sum_275 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_136 = torch.ops.aten.convolution_backward.default(mul_2274, getitem_106, primals_37, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2274 = getitem_106 = primals_37 = None
    getitem_1860: "f32[8, 14, 56, 56]" = convolution_backward_136[0]
    getitem_1861: "f32[14, 14, 3, 3]" = convolution_backward_136[1];  convolution_backward_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_30: "f32[8, 112, 56, 56]" = torch.ops.aten.cat.default([getitem_1860, getitem_1857, getitem_1854, getitem_1851, getitem_1848, getitem_1845, getitem_1842, slice_120], 1);  getitem_1860 = getitem_1857 = getitem_1854 = getitem_1851 = getitem_1848 = getitem_1845 = getitem_1842 = slice_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_134: "f32[8, 112, 56, 56]" = torch.ops.aten.where.self(le_134, full_default, cat_30);  le_134 = cat_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_276: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_134, [0, 2, 3])
    sub_697: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_2242);  convolution_11 = unsqueeze_2242 = None
    mul_2276: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(where_134, sub_697)
    sum_277: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_2276, [0, 2, 3]);  mul_2276 = None
    mul_2277: "f32[112]" = torch.ops.aten.mul.Tensor(sum_276, 3.985969387755102e-05)
    unsqueeze_2243: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_2277, 0);  mul_2277 = None
    unsqueeze_2244: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2243, 2);  unsqueeze_2243 = None
    unsqueeze_2245: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2244, 3);  unsqueeze_2244 = None
    mul_2278: "f32[112]" = torch.ops.aten.mul.Tensor(sum_277, 3.985969387755102e-05)
    mul_2279: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_2280: "f32[112]" = torch.ops.aten.mul.Tensor(mul_2278, mul_2279);  mul_2278 = mul_2279 = None
    unsqueeze_2246: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_2280, 0);  mul_2280 = None
    unsqueeze_2247: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2246, 2);  unsqueeze_2246 = None
    unsqueeze_2248: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2247, 3);  unsqueeze_2247 = None
    mul_2281: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_35);  primals_35 = None
    unsqueeze_2249: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_2281, 0);  mul_2281 = None
    unsqueeze_2250: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2249, 2);  unsqueeze_2249 = None
    unsqueeze_2251: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2250, 3);  unsqueeze_2250 = None
    mul_2282: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(sub_697, unsqueeze_2248);  sub_697 = unsqueeze_2248 = None
    sub_699: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(where_134, mul_2282);  where_134 = mul_2282 = None
    sub_700: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(sub_699, unsqueeze_2245);  sub_699 = unsqueeze_2245 = None
    mul_2283: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(sub_700, unsqueeze_2251);  sub_700 = unsqueeze_2251 = None
    mul_2284: "f32[112]" = torch.ops.aten.mul.Tensor(sum_277, squeeze_34);  sum_277 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_137 = torch.ops.aten.convolution_backward.default(mul_2283, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2283 = primals_34 = None
    getitem_1863: "f32[8, 256, 56, 56]" = convolution_backward_137[0]
    getitem_1864: "f32[112, 256, 1, 1]" = convolution_backward_137[1];  convolution_backward_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_919: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_126, getitem_1863);  where_126 = getitem_1863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_551: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_552: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_551);  alias_551 = None
    le_135: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_552, 0);  alias_552 = None
    where_135: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_135, full_default, add_919);  le_135 = add_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    sum_278: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_135, [0, 2, 3])
    sub_701: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_2254);  convolution_10 = unsqueeze_2254 = None
    mul_2285: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_135, sub_701)
    sum_279: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2285, [0, 2, 3]);  mul_2285 = None
    mul_2286: "f32[256]" = torch.ops.aten.mul.Tensor(sum_278, 3.985969387755102e-05)
    unsqueeze_2255: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2286, 0);  mul_2286 = None
    unsqueeze_2256: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2255, 2);  unsqueeze_2255 = None
    unsqueeze_2257: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2256, 3);  unsqueeze_2256 = None
    mul_2287: "f32[256]" = torch.ops.aten.mul.Tensor(sum_279, 3.985969387755102e-05)
    mul_2288: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_2289: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2287, mul_2288);  mul_2287 = mul_2288 = None
    unsqueeze_2258: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2289, 0);  mul_2289 = None
    unsqueeze_2259: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2258, 2);  unsqueeze_2258 = None
    unsqueeze_2260: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2259, 3);  unsqueeze_2259 = None
    mul_2290: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_32);  primals_32 = None
    unsqueeze_2261: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2290, 0);  mul_2290 = None
    unsqueeze_2262: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2261, 2);  unsqueeze_2261 = None
    unsqueeze_2263: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2262, 3);  unsqueeze_2262 = None
    mul_2291: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_701, unsqueeze_2260);  sub_701 = unsqueeze_2260 = None
    sub_703: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_135, mul_2291);  mul_2291 = None
    sub_704: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_703, unsqueeze_2257);  sub_703 = None
    mul_2292: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_704, unsqueeze_2263);  sub_704 = unsqueeze_2263 = None
    mul_2293: "f32[256]" = torch.ops.aten.mul.Tensor(sum_279, squeeze_31);  sum_279 = squeeze_31 = None
    convolution_backward_138 = torch.ops.aten.convolution_backward.default(mul_2292, getitem_2, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2292 = primals_31 = None
    getitem_1866: "f32[8, 64, 56, 56]" = convolution_backward_138[0]
    getitem_1867: "f32[256, 64, 1, 1]" = convolution_backward_138[1];  convolution_backward_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sub_705: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_2266);  convolution_9 = unsqueeze_2266 = None
    mul_2294: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_135, sub_705)
    sum_281: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_2294, [0, 2, 3]);  mul_2294 = None
    mul_2296: "f32[256]" = torch.ops.aten.mul.Tensor(sum_281, 3.985969387755102e-05)
    mul_2297: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_2298: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2296, mul_2297);  mul_2296 = mul_2297 = None
    unsqueeze_2270: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2298, 0);  mul_2298 = None
    unsqueeze_2271: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2270, 2);  unsqueeze_2270 = None
    unsqueeze_2272: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2271, 3);  unsqueeze_2271 = None
    mul_2299: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_29);  primals_29 = None
    unsqueeze_2273: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_2299, 0);  mul_2299 = None
    unsqueeze_2274: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2273, 2);  unsqueeze_2273 = None
    unsqueeze_2275: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2274, 3);  unsqueeze_2274 = None
    mul_2300: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_705, unsqueeze_2272);  sub_705 = unsqueeze_2272 = None
    sub_707: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_135, mul_2300);  where_135 = mul_2300 = None
    sub_708: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_707, unsqueeze_2257);  sub_707 = unsqueeze_2257 = None
    mul_2301: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_708, unsqueeze_2275);  sub_708 = unsqueeze_2275 = None
    mul_2302: "f32[256]" = torch.ops.aten.mul.Tensor(sum_281, squeeze_28);  sum_281 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_139 = torch.ops.aten.convolution_backward.default(mul_2301, cat, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2301 = cat = primals_28 = None
    getitem_1869: "f32[8, 112, 56, 56]" = convolution_backward_139[0]
    getitem_1870: "f32[256, 112, 1, 1]" = convolution_backward_139[1];  convolution_backward_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_121: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1869, 1, 0, 14)
    slice_122: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1869, 1, 14, 28)
    slice_123: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1869, 1, 28, 42)
    slice_124: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1869, 1, 42, 56)
    slice_125: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1869, 1, 56, 70)
    slice_126: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1869, 1, 70, 84)
    slice_127: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1869, 1, 84, 98)
    slice_128: "f32[8, 14, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_1869, 1, 98, 112);  getitem_1869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_backward_3: "f32[8, 14, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(slice_128, getitem_91, [3, 3], [1, 1], [1, 1], False, True, None);  slice_128 = getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_136: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_136, full_default, slice_127);  le_136 = slice_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_282: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_136, [0, 2, 3])
    sub_709: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_2278);  convolution_8 = unsqueeze_2278 = None
    mul_2303: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_136, sub_709)
    sum_283: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2303, [0, 2, 3]);  mul_2303 = None
    mul_2304: "f32[14]" = torch.ops.aten.mul.Tensor(sum_282, 3.985969387755102e-05)
    unsqueeze_2279: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2304, 0);  mul_2304 = None
    unsqueeze_2280: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2279, 2);  unsqueeze_2279 = None
    unsqueeze_2281: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2280, 3);  unsqueeze_2280 = None
    mul_2305: "f32[14]" = torch.ops.aten.mul.Tensor(sum_283, 3.985969387755102e-05)
    mul_2306: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_2307: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2305, mul_2306);  mul_2305 = mul_2306 = None
    unsqueeze_2282: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2307, 0);  mul_2307 = None
    unsqueeze_2283: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2282, 2);  unsqueeze_2282 = None
    unsqueeze_2284: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2283, 3);  unsqueeze_2283 = None
    mul_2308: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_26);  primals_26 = None
    unsqueeze_2285: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2308, 0);  mul_2308 = None
    unsqueeze_2286: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2285, 2);  unsqueeze_2285 = None
    unsqueeze_2287: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2286, 3);  unsqueeze_2286 = None
    mul_2309: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_709, unsqueeze_2284);  sub_709 = unsqueeze_2284 = None
    sub_711: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_136, mul_2309);  where_136 = mul_2309 = None
    sub_712: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_711, unsqueeze_2281);  sub_711 = unsqueeze_2281 = None
    mul_2310: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_712, unsqueeze_2287);  sub_712 = unsqueeze_2287 = None
    mul_2311: "f32[14]" = torch.ops.aten.mul.Tensor(sum_283, squeeze_25);  sum_283 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_140 = torch.ops.aten.convolution_backward.default(mul_2310, getitem_80, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2310 = getitem_80 = primals_25 = None
    getitem_1872: "f32[8, 14, 56, 56]" = convolution_backward_140[0]
    getitem_1873: "f32[14, 14, 3, 3]" = convolution_backward_140[1];  convolution_backward_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_137: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_137, full_default, slice_126);  le_137 = slice_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_284: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_137, [0, 2, 3])
    sub_713: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_2290);  convolution_7 = unsqueeze_2290 = None
    mul_2312: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_137, sub_713)
    sum_285: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2312, [0, 2, 3]);  mul_2312 = None
    mul_2313: "f32[14]" = torch.ops.aten.mul.Tensor(sum_284, 3.985969387755102e-05)
    unsqueeze_2291: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2313, 0);  mul_2313 = None
    unsqueeze_2292: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2291, 2);  unsqueeze_2291 = None
    unsqueeze_2293: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2292, 3);  unsqueeze_2292 = None
    mul_2314: "f32[14]" = torch.ops.aten.mul.Tensor(sum_285, 3.985969387755102e-05)
    mul_2315: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_2316: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2314, mul_2315);  mul_2314 = mul_2315 = None
    unsqueeze_2294: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2316, 0);  mul_2316 = None
    unsqueeze_2295: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2294, 2);  unsqueeze_2294 = None
    unsqueeze_2296: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2295, 3);  unsqueeze_2295 = None
    mul_2317: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_23);  primals_23 = None
    unsqueeze_2297: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2317, 0);  mul_2317 = None
    unsqueeze_2298: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2297, 2);  unsqueeze_2297 = None
    unsqueeze_2299: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2298, 3);  unsqueeze_2298 = None
    mul_2318: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_713, unsqueeze_2296);  sub_713 = unsqueeze_2296 = None
    sub_715: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_137, mul_2318);  where_137 = mul_2318 = None
    sub_716: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_715, unsqueeze_2293);  sub_715 = unsqueeze_2293 = None
    mul_2319: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_716, unsqueeze_2299);  sub_716 = unsqueeze_2299 = None
    mul_2320: "f32[14]" = torch.ops.aten.mul.Tensor(sum_285, squeeze_22);  sum_285 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_141 = torch.ops.aten.convolution_backward.default(mul_2319, getitem_69, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2319 = getitem_69 = primals_22 = None
    getitem_1875: "f32[8, 14, 56, 56]" = convolution_backward_141[0]
    getitem_1876: "f32[14, 14, 3, 3]" = convolution_backward_141[1];  convolution_backward_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_138: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_138, full_default, slice_125);  le_138 = slice_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_286: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_138, [0, 2, 3])
    sub_717: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_2302);  convolution_6 = unsqueeze_2302 = None
    mul_2321: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_138, sub_717)
    sum_287: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2321, [0, 2, 3]);  mul_2321 = None
    mul_2322: "f32[14]" = torch.ops.aten.mul.Tensor(sum_286, 3.985969387755102e-05)
    unsqueeze_2303: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2322, 0);  mul_2322 = None
    unsqueeze_2304: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2303, 2);  unsqueeze_2303 = None
    unsqueeze_2305: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2304, 3);  unsqueeze_2304 = None
    mul_2323: "f32[14]" = torch.ops.aten.mul.Tensor(sum_287, 3.985969387755102e-05)
    mul_2324: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_2325: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2323, mul_2324);  mul_2323 = mul_2324 = None
    unsqueeze_2306: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2325, 0);  mul_2325 = None
    unsqueeze_2307: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2306, 2);  unsqueeze_2306 = None
    unsqueeze_2308: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2307, 3);  unsqueeze_2307 = None
    mul_2326: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_20);  primals_20 = None
    unsqueeze_2309: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2326, 0);  mul_2326 = None
    unsqueeze_2310: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2309, 2);  unsqueeze_2309 = None
    unsqueeze_2311: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2310, 3);  unsqueeze_2310 = None
    mul_2327: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_717, unsqueeze_2308);  sub_717 = unsqueeze_2308 = None
    sub_719: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_138, mul_2327);  where_138 = mul_2327 = None
    sub_720: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_719, unsqueeze_2305);  sub_719 = unsqueeze_2305 = None
    mul_2328: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_720, unsqueeze_2311);  sub_720 = unsqueeze_2311 = None
    mul_2329: "f32[14]" = torch.ops.aten.mul.Tensor(sum_287, squeeze_19);  sum_287 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_142 = torch.ops.aten.convolution_backward.default(mul_2328, getitem_58, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2328 = getitem_58 = primals_19 = None
    getitem_1878: "f32[8, 14, 56, 56]" = convolution_backward_142[0]
    getitem_1879: "f32[14, 14, 3, 3]" = convolution_backward_142[1];  convolution_backward_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_139: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_139, full_default, slice_124);  le_139 = slice_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_288: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_139, [0, 2, 3])
    sub_721: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_2314);  convolution_5 = unsqueeze_2314 = None
    mul_2330: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_139, sub_721)
    sum_289: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2330, [0, 2, 3]);  mul_2330 = None
    mul_2331: "f32[14]" = torch.ops.aten.mul.Tensor(sum_288, 3.985969387755102e-05)
    unsqueeze_2315: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2331, 0);  mul_2331 = None
    unsqueeze_2316: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2315, 2);  unsqueeze_2315 = None
    unsqueeze_2317: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2316, 3);  unsqueeze_2316 = None
    mul_2332: "f32[14]" = torch.ops.aten.mul.Tensor(sum_289, 3.985969387755102e-05)
    mul_2333: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_2334: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2332, mul_2333);  mul_2332 = mul_2333 = None
    unsqueeze_2318: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2334, 0);  mul_2334 = None
    unsqueeze_2319: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2318, 2);  unsqueeze_2318 = None
    unsqueeze_2320: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2319, 3);  unsqueeze_2319 = None
    mul_2335: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_17);  primals_17 = None
    unsqueeze_2321: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2335, 0);  mul_2335 = None
    unsqueeze_2322: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2321, 2);  unsqueeze_2321 = None
    unsqueeze_2323: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2322, 3);  unsqueeze_2322 = None
    mul_2336: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_721, unsqueeze_2320);  sub_721 = unsqueeze_2320 = None
    sub_723: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_139, mul_2336);  where_139 = mul_2336 = None
    sub_724: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_723, unsqueeze_2317);  sub_723 = unsqueeze_2317 = None
    mul_2337: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_724, unsqueeze_2323);  sub_724 = unsqueeze_2323 = None
    mul_2338: "f32[14]" = torch.ops.aten.mul.Tensor(sum_289, squeeze_16);  sum_289 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_143 = torch.ops.aten.convolution_backward.default(mul_2337, getitem_47, primals_16, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2337 = getitem_47 = primals_16 = None
    getitem_1881: "f32[8, 14, 56, 56]" = convolution_backward_143[0]
    getitem_1882: "f32[14, 14, 3, 3]" = convolution_backward_143[1];  convolution_backward_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_140: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_140, full_default, slice_123);  le_140 = slice_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_290: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_140, [0, 2, 3])
    sub_725: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_2326);  convolution_4 = unsqueeze_2326 = None
    mul_2339: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_140, sub_725)
    sum_291: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2339, [0, 2, 3]);  mul_2339 = None
    mul_2340: "f32[14]" = torch.ops.aten.mul.Tensor(sum_290, 3.985969387755102e-05)
    unsqueeze_2327: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2340, 0);  mul_2340 = None
    unsqueeze_2328: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2327, 2);  unsqueeze_2327 = None
    unsqueeze_2329: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2328, 3);  unsqueeze_2328 = None
    mul_2341: "f32[14]" = torch.ops.aten.mul.Tensor(sum_291, 3.985969387755102e-05)
    mul_2342: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_2343: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2341, mul_2342);  mul_2341 = mul_2342 = None
    unsqueeze_2330: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2343, 0);  mul_2343 = None
    unsqueeze_2331: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2330, 2);  unsqueeze_2330 = None
    unsqueeze_2332: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2331, 3);  unsqueeze_2331 = None
    mul_2344: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
    unsqueeze_2333: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2344, 0);  mul_2344 = None
    unsqueeze_2334: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2333, 2);  unsqueeze_2333 = None
    unsqueeze_2335: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2334, 3);  unsqueeze_2334 = None
    mul_2345: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_725, unsqueeze_2332);  sub_725 = unsqueeze_2332 = None
    sub_727: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_140, mul_2345);  where_140 = mul_2345 = None
    sub_728: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_727, unsqueeze_2329);  sub_727 = unsqueeze_2329 = None
    mul_2346: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_728, unsqueeze_2335);  sub_728 = unsqueeze_2335 = None
    mul_2347: "f32[14]" = torch.ops.aten.mul.Tensor(sum_291, squeeze_13);  sum_291 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_144 = torch.ops.aten.convolution_backward.default(mul_2346, getitem_36, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2346 = getitem_36 = primals_13 = None
    getitem_1884: "f32[8, 14, 56, 56]" = convolution_backward_144[0]
    getitem_1885: "f32[14, 14, 3, 3]" = convolution_backward_144[1];  convolution_backward_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_141: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_141, full_default, slice_122);  le_141 = slice_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_292: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_141, [0, 2, 3])
    sub_729: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_2338);  convolution_3 = unsqueeze_2338 = None
    mul_2348: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_141, sub_729)
    sum_293: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2348, [0, 2, 3]);  mul_2348 = None
    mul_2349: "f32[14]" = torch.ops.aten.mul.Tensor(sum_292, 3.985969387755102e-05)
    unsqueeze_2339: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2349, 0);  mul_2349 = None
    unsqueeze_2340: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2339, 2);  unsqueeze_2339 = None
    unsqueeze_2341: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2340, 3);  unsqueeze_2340 = None
    mul_2350: "f32[14]" = torch.ops.aten.mul.Tensor(sum_293, 3.985969387755102e-05)
    mul_2351: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_2352: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2350, mul_2351);  mul_2350 = mul_2351 = None
    unsqueeze_2342: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2352, 0);  mul_2352 = None
    unsqueeze_2343: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2342, 2);  unsqueeze_2342 = None
    unsqueeze_2344: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2343, 3);  unsqueeze_2343 = None
    mul_2353: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
    unsqueeze_2345: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2353, 0);  mul_2353 = None
    unsqueeze_2346: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2345, 2);  unsqueeze_2345 = None
    unsqueeze_2347: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2346, 3);  unsqueeze_2346 = None
    mul_2354: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_729, unsqueeze_2344);  sub_729 = unsqueeze_2344 = None
    sub_731: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_141, mul_2354);  where_141 = mul_2354 = None
    sub_732: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_731, unsqueeze_2341);  sub_731 = unsqueeze_2341 = None
    mul_2355: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_732, unsqueeze_2347);  sub_732 = unsqueeze_2347 = None
    mul_2356: "f32[14]" = torch.ops.aten.mul.Tensor(sum_293, squeeze_10);  sum_293 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_145 = torch.ops.aten.convolution_backward.default(mul_2355, getitem_25, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2355 = getitem_25 = primals_10 = None
    getitem_1887: "f32[8, 14, 56, 56]" = convolution_backward_145[0]
    getitem_1888: "f32[14, 14, 3, 3]" = convolution_backward_145[1];  convolution_backward_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_142: "f32[8, 14, 56, 56]" = torch.ops.aten.where.self(le_142, full_default, slice_121);  le_142 = slice_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_294: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_142, [0, 2, 3])
    sub_733: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_2350);  convolution_2 = unsqueeze_2350 = None
    mul_2357: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(where_142, sub_733)
    sum_295: "f32[14]" = torch.ops.aten.sum.dim_IntList(mul_2357, [0, 2, 3]);  mul_2357 = None
    mul_2358: "f32[14]" = torch.ops.aten.mul.Tensor(sum_294, 3.985969387755102e-05)
    unsqueeze_2351: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2358, 0);  mul_2358 = None
    unsqueeze_2352: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2351, 2);  unsqueeze_2351 = None
    unsqueeze_2353: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2352, 3);  unsqueeze_2352 = None
    mul_2359: "f32[14]" = torch.ops.aten.mul.Tensor(sum_295, 3.985969387755102e-05)
    mul_2360: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_2361: "f32[14]" = torch.ops.aten.mul.Tensor(mul_2359, mul_2360);  mul_2359 = mul_2360 = None
    unsqueeze_2354: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2361, 0);  mul_2361 = None
    unsqueeze_2355: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2354, 2);  unsqueeze_2354 = None
    unsqueeze_2356: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2355, 3);  unsqueeze_2355 = None
    mul_2362: "f32[14]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
    unsqueeze_2357: "f32[1, 14]" = torch.ops.aten.unsqueeze.default(mul_2362, 0);  mul_2362 = None
    unsqueeze_2358: "f32[1, 14, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2357, 2);  unsqueeze_2357 = None
    unsqueeze_2359: "f32[1, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2358, 3);  unsqueeze_2358 = None
    mul_2363: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_733, unsqueeze_2356);  sub_733 = unsqueeze_2356 = None
    sub_735: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(where_142, mul_2363);  where_142 = mul_2363 = None
    sub_736: "f32[8, 14, 56, 56]" = torch.ops.aten.sub.Tensor(sub_735, unsqueeze_2353);  sub_735 = unsqueeze_2353 = None
    mul_2364: "f32[8, 14, 56, 56]" = torch.ops.aten.mul.Tensor(sub_736, unsqueeze_2359);  sub_736 = unsqueeze_2359 = None
    mul_2365: "f32[14]" = torch.ops.aten.mul.Tensor(sum_295, squeeze_7);  sum_295 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_146 = torch.ops.aten.convolution_backward.default(mul_2364, getitem_14, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2364 = getitem_14 = primals_7 = None
    getitem_1890: "f32[8, 14, 56, 56]" = convolution_backward_146[0]
    getitem_1891: "f32[14, 14, 3, 3]" = convolution_backward_146[1];  convolution_backward_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_31: "f32[8, 112, 56, 56]" = torch.ops.aten.cat.default([getitem_1890, getitem_1887, getitem_1884, getitem_1881, getitem_1878, getitem_1875, getitem_1872, avg_pool2d_backward_3], 1);  getitem_1890 = getitem_1887 = getitem_1884 = getitem_1881 = getitem_1878 = getitem_1875 = getitem_1872 = avg_pool2d_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    where_143: "f32[8, 112, 56, 56]" = torch.ops.aten.where.self(le_143, full_default, cat_31);  le_143 = cat_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_296: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_143, [0, 2, 3])
    sub_737: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_2362);  convolution_1 = unsqueeze_2362 = None
    mul_2366: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(where_143, sub_737)
    sum_297: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_2366, [0, 2, 3]);  mul_2366 = None
    mul_2367: "f32[112]" = torch.ops.aten.mul.Tensor(sum_296, 3.985969387755102e-05)
    unsqueeze_2363: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_2367, 0);  mul_2367 = None
    unsqueeze_2364: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2363, 2);  unsqueeze_2363 = None
    unsqueeze_2365: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2364, 3);  unsqueeze_2364 = None
    mul_2368: "f32[112]" = torch.ops.aten.mul.Tensor(sum_297, 3.985969387755102e-05)
    mul_2369: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_2370: "f32[112]" = torch.ops.aten.mul.Tensor(mul_2368, mul_2369);  mul_2368 = mul_2369 = None
    unsqueeze_2366: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_2370, 0);  mul_2370 = None
    unsqueeze_2367: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2366, 2);  unsqueeze_2366 = None
    unsqueeze_2368: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2367, 3);  unsqueeze_2367 = None
    mul_2371: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
    unsqueeze_2369: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_2371, 0);  mul_2371 = None
    unsqueeze_2370: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2369, 2);  unsqueeze_2369 = None
    unsqueeze_2371: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2370, 3);  unsqueeze_2370 = None
    mul_2372: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(sub_737, unsqueeze_2368);  sub_737 = unsqueeze_2368 = None
    sub_739: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(where_143, mul_2372);  where_143 = mul_2372 = None
    sub_740: "f32[8, 112, 56, 56]" = torch.ops.aten.sub.Tensor(sub_739, unsqueeze_2365);  sub_739 = unsqueeze_2365 = None
    mul_2373: "f32[8, 112, 56, 56]" = torch.ops.aten.mul.Tensor(sub_740, unsqueeze_2371);  sub_740 = unsqueeze_2371 = None
    mul_2374: "f32[112]" = torch.ops.aten.mul.Tensor(sum_297, squeeze_4);  sum_297 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_147 = torch.ops.aten.convolution_backward.default(mul_2373, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2373 = getitem_2 = primals_4 = None
    getitem_1893: "f32[8, 64, 56, 56]" = convolution_backward_147[0]
    getitem_1894: "f32[112, 64, 1, 1]" = convolution_backward_147[1];  convolution_backward_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_920: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_1866, getitem_1893);  getitem_1866 = getitem_1893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    max_pool2d_with_indices_backward: "f32[8, 64, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_920, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_3);  add_920 = getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    alias_578: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_579: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_578);  alias_578 = None
    le_144: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_579, 0);  alias_579 = None
    where_144: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_144, full_default, max_pool2d_with_indices_backward);  le_144 = full_default = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    sum_298: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_144, [0, 2, 3])
    sub_741: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_2374);  convolution = unsqueeze_2374 = None
    mul_2375: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_144, sub_741)
    sum_299: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_2375, [0, 2, 3]);  mul_2375 = None
    mul_2376: "f32[64]" = torch.ops.aten.mul.Tensor(sum_298, 9.964923469387754e-06)
    unsqueeze_2375: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2376, 0);  mul_2376 = None
    unsqueeze_2376: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2375, 2);  unsqueeze_2375 = None
    unsqueeze_2377: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2376, 3);  unsqueeze_2376 = None
    mul_2377: "f32[64]" = torch.ops.aten.mul.Tensor(sum_299, 9.964923469387754e-06)
    mul_2378: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_2379: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2377, mul_2378);  mul_2377 = mul_2378 = None
    unsqueeze_2378: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2379, 0);  mul_2379 = None
    unsqueeze_2379: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2378, 2);  unsqueeze_2378 = None
    unsqueeze_2380: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2379, 3);  unsqueeze_2379 = None
    mul_2380: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
    unsqueeze_2381: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_2380, 0);  mul_2380 = None
    unsqueeze_2382: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2381, 2);  unsqueeze_2381 = None
    unsqueeze_2383: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2382, 3);  unsqueeze_2382 = None
    mul_2381: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_741, unsqueeze_2380);  sub_741 = unsqueeze_2380 = None
    sub_743: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_144, mul_2381);  where_144 = mul_2381 = None
    sub_744: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_743, unsqueeze_2377);  sub_743 = unsqueeze_2377 = None
    mul_2382: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_744, unsqueeze_2383);  sub_744 = unsqueeze_2383 = None
    mul_2383: "f32[64]" = torch.ops.aten.mul.Tensor(sum_299, squeeze_1);  sum_299 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution_backward_148 = torch.ops.aten.convolution_backward.default(mul_2382, primals_897, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_2382 = primals_897 = primals_1 = None
    getitem_1897: "f32[64, 3, 7, 7]" = convolution_backward_148[1];  convolution_backward_148 = None
    return [getitem_1897, mul_2383, sum_298, getitem_1894, mul_2374, sum_296, getitem_1891, mul_2365, sum_294, getitem_1888, mul_2356, sum_292, getitem_1885, mul_2347, sum_290, getitem_1882, mul_2338, sum_288, getitem_1879, mul_2329, sum_286, getitem_1876, mul_2320, sum_284, getitem_1873, mul_2311, sum_282, getitem_1870, mul_2302, sum_278, getitem_1867, mul_2293, sum_278, getitem_1864, mul_2284, sum_276, getitem_1861, mul_2275, sum_274, getitem_1858, mul_2266, sum_272, getitem_1855, mul_2257, sum_270, getitem_1852, mul_2248, sum_268, getitem_1849, mul_2239, sum_266, getitem_1846, mul_2230, sum_264, getitem_1843, mul_2221, sum_262, getitem_1840, mul_2212, sum_260, getitem_1837, mul_2203, sum_258, getitem_1834, mul_2194, sum_256, getitem_1831, mul_2185, sum_254, getitem_1828, mul_2176, sum_252, getitem_1825, mul_2167, sum_250, getitem_1822, mul_2158, sum_248, getitem_1819, mul_2149, sum_246, getitem_1816, mul_2140, sum_244, getitem_1813, mul_2131, sum_242, getitem_1810, mul_2122, sum_240, getitem_1807, mul_2113, sum_238, getitem_1804, mul_2104, sum_236, getitem_1801, mul_2095, sum_234, getitem_1798, mul_2086, sum_232, getitem_1795, mul_2077, sum_230, getitem_1792, mul_2068, sum_228, getitem_1789, mul_2059, sum_226, getitem_1786, mul_2050, sum_222, getitem_1783, mul_2041, sum_222, getitem_1780, mul_2032, sum_220, getitem_1777, mul_2023, sum_218, getitem_1774, mul_2014, sum_216, getitem_1771, mul_2005, sum_214, getitem_1768, mul_1996, sum_212, getitem_1765, mul_1987, sum_210, getitem_1762, mul_1978, sum_208, getitem_1759, mul_1969, sum_206, getitem_1756, mul_1960, sum_204, getitem_1753, mul_1951, sum_202, getitem_1750, mul_1942, sum_200, getitem_1747, mul_1933, sum_198, getitem_1744, mul_1924, sum_196, getitem_1741, mul_1915, sum_194, getitem_1738, mul_1906, sum_192, getitem_1735, mul_1897, sum_190, getitem_1732, mul_1888, sum_188, getitem_1729, mul_1879, sum_186, getitem_1726, mul_1870, sum_184, getitem_1723, mul_1861, sum_182, getitem_1720, mul_1852, sum_180, getitem_1717, mul_1843, sum_178, getitem_1714, mul_1834, sum_176, getitem_1711, mul_1825, sum_174, getitem_1708, mul_1816, sum_172, getitem_1705, mul_1807, sum_170, getitem_1702, mul_1798, sum_168, getitem_1699, mul_1789, sum_166, getitem_1696, mul_1780, sum_164, getitem_1693, mul_1771, sum_162, getitem_1690, mul_1762, sum_160, getitem_1687, mul_1753, sum_158, getitem_1684, mul_1744, sum_156, getitem_1681, mul_1735, sum_154, getitem_1678, mul_1726, sum_152, getitem_1675, mul_1717, sum_148, getitem_1672, mul_1708, sum_148, getitem_1669, mul_1699, sum_146, getitem_1666, mul_1690, sum_144, getitem_1663, mul_1681, sum_142, getitem_1660, mul_1672, sum_140, getitem_1657, mul_1663, sum_138, getitem_1654, mul_1654, sum_136, getitem_1651, mul_1645, sum_134, getitem_1648, mul_1636, sum_132, getitem_1645, mul_1627, sum_130, getitem_1642, mul_1618, sum_128, getitem_1639, mul_1609, sum_126, getitem_1636, mul_1600, sum_124, getitem_1633, mul_1591, sum_122, getitem_1630, mul_1582, sum_120, getitem_1627, mul_1573, sum_118, getitem_1624, mul_1564, sum_116, getitem_1621, mul_1555, sum_114, getitem_1618, mul_1546, sum_112, getitem_1615, mul_1537, sum_110, getitem_1612, mul_1528, sum_108, getitem_1609, mul_1519, sum_106, getitem_1606, mul_1510, sum_104, getitem_1603, mul_1501, sum_102, getitem_1600, mul_1492, sum_100, getitem_1597, mul_1483, sum_98, getitem_1594, mul_1474, sum_96, getitem_1591, mul_1465, sum_94, getitem_1588, mul_1456, sum_92, getitem_1585, mul_1447, sum_90, getitem_1582, mul_1438, sum_88, getitem_1579, mul_1429, sum_86, getitem_1576, mul_1420, sum_84, getitem_1573, mul_1411, sum_82, getitem_1570, mul_1402, sum_80, getitem_1567, mul_1393, sum_78, getitem_1564, mul_1384, sum_76, getitem_1561, mul_1375, sum_74, getitem_1558, mul_1366, sum_72, getitem_1555, mul_1357, sum_70, getitem_1552, mul_1348, sum_68, getitem_1549, mul_1339, sum_66, getitem_1546, mul_1330, sum_64, getitem_1543, mul_1321, sum_62, getitem_1540, mul_1312, sum_60, getitem_1537, mul_1303, sum_58, getitem_1534, mul_1294, sum_56, getitem_1531, mul_1285, sum_54, getitem_1528, mul_1276, sum_52, getitem_1525, mul_1267, sum_50, getitem_1522, mul_1258, sum_48, getitem_1519, mul_1249, sum_46, getitem_1516, mul_1240, sum_44, getitem_1513, mul_1231, sum_42, getitem_1510, mul_1222, sum_38, getitem_1507, mul_1213, sum_38, getitem_1504, mul_1204, sum_36, getitem_1501, mul_1195, sum_34, getitem_1498, mul_1186, sum_32, getitem_1495, mul_1177, sum_30, getitem_1492, mul_1168, sum_28, getitem_1489, mul_1159, sum_26, getitem_1486, mul_1150, sum_24, getitem_1483, mul_1141, sum_22, getitem_1480, mul_1132, sum_20, getitem_1477, mul_1123, sum_18, getitem_1474, mul_1114, sum_16, getitem_1471, mul_1105, sum_14, getitem_1468, mul_1096, sum_12, getitem_1465, mul_1087, sum_10, getitem_1462, mul_1078, sum_8, getitem_1459, mul_1069, sum_6, getitem_1456, mul_1060, sum_4, getitem_1453, mul_1051, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    