from __future__ import annotations



def forward(self, primals_1: "f32[96]", primals_2: "f32[96]", primals_3: "f32[96, 1, 5, 5]", primals_4: "f32[54, 1, 7, 7]", primals_5: "f32[54, 1, 5, 5]", primals_6: "f32[54, 1, 3, 3]", primals_7: "f32[96, 1, 3, 3]", primals_8: "f32[54, 54, 1, 1]", primals_9: "f32[108, 1, 5, 5]", primals_10: "f32[108, 1, 7, 7]", primals_11: "f32[108, 1, 5, 5]", primals_12: "f32[108, 1, 3, 3]", primals_13: "f32[108, 1, 3, 3]", primals_14: "f32[108, 108, 1, 1]", primals_15: "f32[432, 1, 5, 5]", primals_16: "f32[432, 1, 7, 7]", primals_17: "f32[432, 1, 5, 5]", primals_18: "f32[432, 1, 3, 3]", primals_19: "f32[432, 1, 3, 3]", primals_20: "f32[432, 432, 1, 1]", primals_21: "f32[864, 1, 5, 5]", primals_22: "f32[864, 1, 7, 7]", primals_23: "f32[864, 1, 5, 5]", primals_24: "f32[864, 1, 3, 3]", primals_25: "f32[864, 1, 3, 3]", primals_26: "f32[864, 864, 1, 1]", primals_27: "f32[96, 3, 3, 3]", primals_28: "f32[54, 96, 1, 1]", primals_29: "f32[54]", primals_30: "f32[54]", primals_31: "f32[54, 96, 1, 1]", primals_32: "f32[54]", primals_33: "f32[54]", primals_34: "f32[54, 1, 5, 5]", primals_35: "f32[54, 54, 1, 1]", primals_36: "f32[54]", primals_37: "f32[54]", primals_38: "f32[54, 96, 1, 1]", primals_39: "f32[54]", primals_40: "f32[54]", primals_41: "f32[54, 54, 1, 1]", primals_42: "f32[54]", primals_43: "f32[54]", primals_44: "f32[54, 1, 7, 7]", primals_45: "f32[54, 54, 1, 1]", primals_46: "f32[54]", primals_47: "f32[54]", primals_48: "f32[54, 54, 1, 1]", primals_49: "f32[54]", primals_50: "f32[54]", primals_51: "f32[54, 1, 5, 5]", primals_52: "f32[54, 54, 1, 1]", primals_53: "f32[54]", primals_54: "f32[54]", primals_55: "f32[54, 54, 1, 1]", primals_56: "f32[54]", primals_57: "f32[54]", primals_58: "f32[54, 1, 3, 3]", primals_59: "f32[54, 54, 1, 1]", primals_60: "f32[54]", primals_61: "f32[54]", primals_62: "f32[54, 1, 3, 3]", primals_63: "f32[54, 54, 1, 1]", primals_64: "f32[54]", primals_65: "f32[54]", primals_66: "f32[54, 1, 3, 3]", primals_67: "f32[54, 54, 1, 1]", primals_68: "f32[54]", primals_69: "f32[54]", primals_70: "f32[54, 96, 1, 1]", primals_71: "f32[54]", primals_72: "f32[54]", primals_73: "f32[54, 1, 3, 3]", primals_74: "f32[54, 54, 1, 1]", primals_75: "f32[54]", primals_76: "f32[54]", primals_77: "f32[54]", primals_78: "f32[54]", primals_79: "f32[54, 96, 1, 1]", primals_80: "f32[54, 96, 1, 1]", primals_81: "f32[108]", primals_82: "f32[108]", primals_83: "f32[108, 270, 1, 1]", primals_84: "f32[108]", primals_85: "f32[108]", primals_86: "f32[108, 108, 1, 1]", primals_87: "f32[108]", primals_88: "f32[108]", primals_89: "f32[108, 1, 5, 5]", primals_90: "f32[108, 108, 1, 1]", primals_91: "f32[108]", primals_92: "f32[108]", primals_93: "f32[108, 108, 1, 1]", primals_94: "f32[108]", primals_95: "f32[108]", primals_96: "f32[108, 1, 7, 7]", primals_97: "f32[108, 108, 1, 1]", primals_98: "f32[108]", primals_99: "f32[108]", primals_100: "f32[108, 108, 1, 1]", primals_101: "f32[108]", primals_102: "f32[108]", primals_103: "f32[108, 1, 5, 5]", primals_104: "f32[108, 108, 1, 1]", primals_105: "f32[108]", primals_106: "f32[108]", primals_107: "f32[108, 108, 1, 1]", primals_108: "f32[108]", primals_109: "f32[108]", primals_110: "f32[108, 1, 3, 3]", primals_111: "f32[108, 108, 1, 1]", primals_112: "f32[108]", primals_113: "f32[108]", primals_114: "f32[108, 1, 3, 3]", primals_115: "f32[108, 108, 1, 1]", primals_116: "f32[108]", primals_117: "f32[108]", primals_118: "f32[108, 1, 3, 3]", primals_119: "f32[108, 108, 1, 1]", primals_120: "f32[108]", primals_121: "f32[108]", primals_122: "f32[108, 108, 1, 1]", primals_123: "f32[108]", primals_124: "f32[108]", primals_125: "f32[108, 1, 3, 3]", primals_126: "f32[108, 108, 1, 1]", primals_127: "f32[108]", primals_128: "f32[108]", primals_129: "f32[108]", primals_130: "f32[108]", primals_131: "f32[108, 270, 1, 1]", primals_132: "f32[108, 270, 1, 1]", primals_133: "f32[216]", primals_134: "f32[216]", primals_135: "f32[216, 540, 1, 1]", primals_136: "f32[216]", primals_137: "f32[216]", primals_138: "f32[216, 1, 5, 5]", primals_139: "f32[216, 216, 1, 1]", primals_140: "f32[216]", primals_141: "f32[216]", primals_142: "f32[216, 1, 5, 5]", primals_143: "f32[216, 216, 1, 1]", primals_144: "f32[216]", primals_145: "f32[216]", primals_146: "f32[216, 1, 7, 7]", primals_147: "f32[216, 216, 1, 1]", primals_148: "f32[216]", primals_149: "f32[216]", primals_150: "f32[216, 1, 7, 7]", primals_151: "f32[216, 216, 1, 1]", primals_152: "f32[216]", primals_153: "f32[216]", primals_154: "f32[216, 1, 5, 5]", primals_155: "f32[216, 216, 1, 1]", primals_156: "f32[216]", primals_157: "f32[216]", primals_158: "f32[216, 1, 5, 5]", primals_159: "f32[216, 216, 1, 1]", primals_160: "f32[216]", primals_161: "f32[216]", primals_162: "f32[216, 1, 3, 3]", primals_163: "f32[216, 216, 1, 1]", primals_164: "f32[216]", primals_165: "f32[216]", primals_166: "f32[216, 1, 3, 3]", primals_167: "f32[216, 216, 1, 1]", primals_168: "f32[216]", primals_169: "f32[216]", primals_170: "f32[216, 1, 3, 3]", primals_171: "f32[216, 216, 1, 1]", primals_172: "f32[216]", primals_173: "f32[216]", primals_174: "f32[216, 1, 3, 3]", primals_175: "f32[216, 216, 1, 1]", primals_176: "f32[216]", primals_177: "f32[216]", primals_178: "f32[216, 1, 3, 3]", primals_179: "f32[216, 216, 1, 1]", primals_180: "f32[216]", primals_181: "f32[216]", primals_182: "f32[216, 1, 3, 3]", primals_183: "f32[216, 216, 1, 1]", primals_184: "f32[216]", primals_185: "f32[216]", primals_186: "f32[216, 540, 1, 1]", primals_187: "f32[216]", primals_188: "f32[216]", primals_189: "f32[216, 1080, 1, 1]", primals_190: "f32[216]", primals_191: "f32[216]", primals_192: "f32[216, 1, 5, 5]", primals_193: "f32[216, 216, 1, 1]", primals_194: "f32[216]", primals_195: "f32[216]", primals_196: "f32[216, 1, 5, 5]", primals_197: "f32[216, 216, 1, 1]", primals_198: "f32[216]", primals_199: "f32[216]", primals_200: "f32[216, 1, 7, 7]", primals_201: "f32[216, 216, 1, 1]", primals_202: "f32[216]", primals_203: "f32[216]", primals_204: "f32[216, 1, 7, 7]", primals_205: "f32[216, 216, 1, 1]", primals_206: "f32[216]", primals_207: "f32[216]", primals_208: "f32[216, 1, 5, 5]", primals_209: "f32[216, 216, 1, 1]", primals_210: "f32[216]", primals_211: "f32[216]", primals_212: "f32[216, 1, 5, 5]", primals_213: "f32[216, 216, 1, 1]", primals_214: "f32[216]", primals_215: "f32[216]", primals_216: "f32[216, 1, 3, 3]", primals_217: "f32[216, 216, 1, 1]", primals_218: "f32[216]", primals_219: "f32[216]", primals_220: "f32[216, 1, 3, 3]", primals_221: "f32[216, 216, 1, 1]", primals_222: "f32[216]", primals_223: "f32[216]", primals_224: "f32[216, 1, 3, 3]", primals_225: "f32[216, 216, 1, 1]", primals_226: "f32[216]", primals_227: "f32[216]", primals_228: "f32[216, 1, 3, 3]", primals_229: "f32[216, 216, 1, 1]", primals_230: "f32[216]", primals_231: "f32[216]", primals_232: "f32[216, 1, 3, 3]", primals_233: "f32[216, 216, 1, 1]", primals_234: "f32[216]", primals_235: "f32[216]", primals_236: "f32[216, 1, 3, 3]", primals_237: "f32[216, 216, 1, 1]", primals_238: "f32[216]", primals_239: "f32[216]", primals_240: "f32[216, 1080, 1, 1]", primals_241: "f32[216]", primals_242: "f32[216]", primals_243: "f32[216, 1080, 1, 1]", primals_244: "f32[216]", primals_245: "f32[216]", primals_246: "f32[216, 1, 5, 5]", primals_247: "f32[216, 216, 1, 1]", primals_248: "f32[216]", primals_249: "f32[216]", primals_250: "f32[216, 1, 5, 5]", primals_251: "f32[216, 216, 1, 1]", primals_252: "f32[216]", primals_253: "f32[216]", primals_254: "f32[216, 1, 7, 7]", primals_255: "f32[216, 216, 1, 1]", primals_256: "f32[216]", primals_257: "f32[216]", primals_258: "f32[216, 1, 7, 7]", primals_259: "f32[216, 216, 1, 1]", primals_260: "f32[216]", primals_261: "f32[216]", primals_262: "f32[216, 1, 5, 5]", primals_263: "f32[216, 216, 1, 1]", primals_264: "f32[216]", primals_265: "f32[216]", primals_266: "f32[216, 1, 5, 5]", primals_267: "f32[216, 216, 1, 1]", primals_268: "f32[216]", primals_269: "f32[216]", primals_270: "f32[216, 1, 3, 3]", primals_271: "f32[216, 216, 1, 1]", primals_272: "f32[216]", primals_273: "f32[216]", primals_274: "f32[216, 1, 3, 3]", primals_275: "f32[216, 216, 1, 1]", primals_276: "f32[216]", primals_277: "f32[216]", primals_278: "f32[216, 1, 3, 3]", primals_279: "f32[216, 216, 1, 1]", primals_280: "f32[216]", primals_281: "f32[216]", primals_282: "f32[216, 1, 3, 3]", primals_283: "f32[216, 216, 1, 1]", primals_284: "f32[216]", primals_285: "f32[216]", primals_286: "f32[216, 1, 3, 3]", primals_287: "f32[216, 216, 1, 1]", primals_288: "f32[216]", primals_289: "f32[216]", primals_290: "f32[216, 1, 3, 3]", primals_291: "f32[216, 216, 1, 1]", primals_292: "f32[216]", primals_293: "f32[216]", primals_294: "f32[216, 1080, 1, 1]", primals_295: "f32[216]", primals_296: "f32[216]", primals_297: "f32[216, 1080, 1, 1]", primals_298: "f32[216]", primals_299: "f32[216]", primals_300: "f32[216, 1, 5, 5]", primals_301: "f32[216, 216, 1, 1]", primals_302: "f32[216]", primals_303: "f32[216]", primals_304: "f32[216, 1, 5, 5]", primals_305: "f32[216, 216, 1, 1]", primals_306: "f32[216]", primals_307: "f32[216]", primals_308: "f32[216, 1, 7, 7]", primals_309: "f32[216, 216, 1, 1]", primals_310: "f32[216]", primals_311: "f32[216]", primals_312: "f32[216, 1, 7, 7]", primals_313: "f32[216, 216, 1, 1]", primals_314: "f32[216]", primals_315: "f32[216]", primals_316: "f32[216, 1, 5, 5]", primals_317: "f32[216, 216, 1, 1]", primals_318: "f32[216]", primals_319: "f32[216]", primals_320: "f32[216, 1, 5, 5]", primals_321: "f32[216, 216, 1, 1]", primals_322: "f32[216]", primals_323: "f32[216]", primals_324: "f32[216, 1, 3, 3]", primals_325: "f32[216, 216, 1, 1]", primals_326: "f32[216]", primals_327: "f32[216]", primals_328: "f32[216, 1, 3, 3]", primals_329: "f32[216, 216, 1, 1]", primals_330: "f32[216]", primals_331: "f32[216]", primals_332: "f32[216, 1, 3, 3]", primals_333: "f32[216, 216, 1, 1]", primals_334: "f32[216]", primals_335: "f32[216]", primals_336: "f32[216, 1, 3, 3]", primals_337: "f32[216, 216, 1, 1]", primals_338: "f32[216]", primals_339: "f32[216]", primals_340: "f32[216, 1, 3, 3]", primals_341: "f32[216, 216, 1, 1]", primals_342: "f32[216]", primals_343: "f32[216]", primals_344: "f32[216, 1, 3, 3]", primals_345: "f32[216, 216, 1, 1]", primals_346: "f32[216]", primals_347: "f32[216]", primals_348: "f32[432, 1080, 1, 1]", primals_349: "f32[432]", primals_350: "f32[432]", primals_351: "f32[432, 1080, 1, 1]", primals_352: "f32[432]", primals_353: "f32[432]", primals_354: "f32[432, 432, 1, 1]", primals_355: "f32[432]", primals_356: "f32[432]", primals_357: "f32[432, 1, 5, 5]", primals_358: "f32[432, 432, 1, 1]", primals_359: "f32[432]", primals_360: "f32[432]", primals_361: "f32[432, 432, 1, 1]", primals_362: "f32[432]", primals_363: "f32[432]", primals_364: "f32[432, 1, 7, 7]", primals_365: "f32[432, 432, 1, 1]", primals_366: "f32[432]", primals_367: "f32[432]", primals_368: "f32[432, 432, 1, 1]", primals_369: "f32[432]", primals_370: "f32[432]", primals_371: "f32[432, 1, 5, 5]", primals_372: "f32[432, 432, 1, 1]", primals_373: "f32[432]", primals_374: "f32[432]", primals_375: "f32[432, 432, 1, 1]", primals_376: "f32[432]", primals_377: "f32[432]", primals_378: "f32[432, 1, 3, 3]", primals_379: "f32[432, 432, 1, 1]", primals_380: "f32[432]", primals_381: "f32[432]", primals_382: "f32[432, 1, 3, 3]", primals_383: "f32[432, 432, 1, 1]", primals_384: "f32[432]", primals_385: "f32[432]", primals_386: "f32[432, 1, 3, 3]", primals_387: "f32[432, 432, 1, 1]", primals_388: "f32[432]", primals_389: "f32[432]", primals_390: "f32[432, 432, 1, 1]", primals_391: "f32[432]", primals_392: "f32[432]", primals_393: "f32[432, 1, 3, 3]", primals_394: "f32[432, 432, 1, 1]", primals_395: "f32[432]", primals_396: "f32[432]", primals_397: "f32[432]", primals_398: "f32[432]", primals_399: "f32[216, 1080, 1, 1]", primals_400: "f32[216, 1080, 1, 1]", primals_401: "f32[432]", primals_402: "f32[432]", primals_403: "f32[432, 2160, 1, 1]", primals_404: "f32[432]", primals_405: "f32[432]", primals_406: "f32[432, 1, 5, 5]", primals_407: "f32[432, 432, 1, 1]", primals_408: "f32[432]", primals_409: "f32[432]", primals_410: "f32[432, 1, 5, 5]", primals_411: "f32[432, 432, 1, 1]", primals_412: "f32[432]", primals_413: "f32[432]", primals_414: "f32[432, 1, 7, 7]", primals_415: "f32[432, 432, 1, 1]", primals_416: "f32[432]", primals_417: "f32[432]", primals_418: "f32[432, 1, 7, 7]", primals_419: "f32[432, 432, 1, 1]", primals_420: "f32[432]", primals_421: "f32[432]", primals_422: "f32[432, 1, 5, 5]", primals_423: "f32[432, 432, 1, 1]", primals_424: "f32[432]", primals_425: "f32[432]", primals_426: "f32[432, 1, 5, 5]", primals_427: "f32[432, 432, 1, 1]", primals_428: "f32[432]", primals_429: "f32[432]", primals_430: "f32[432, 1, 3, 3]", primals_431: "f32[432, 432, 1, 1]", primals_432: "f32[432]", primals_433: "f32[432]", primals_434: "f32[432, 1, 3, 3]", primals_435: "f32[432, 432, 1, 1]", primals_436: "f32[432]", primals_437: "f32[432]", primals_438: "f32[432, 1, 3, 3]", primals_439: "f32[432, 432, 1, 1]", primals_440: "f32[432]", primals_441: "f32[432]", primals_442: "f32[432, 1, 3, 3]", primals_443: "f32[432, 432, 1, 1]", primals_444: "f32[432]", primals_445: "f32[432]", primals_446: "f32[432, 1, 3, 3]", primals_447: "f32[432, 432, 1, 1]", primals_448: "f32[432]", primals_449: "f32[432]", primals_450: "f32[432, 1, 3, 3]", primals_451: "f32[432, 432, 1, 1]", primals_452: "f32[432]", primals_453: "f32[432]", primals_454: "f32[432, 2160, 1, 1]", primals_455: "f32[432]", primals_456: "f32[432]", primals_457: "f32[432, 2160, 1, 1]", primals_458: "f32[432]", primals_459: "f32[432]", primals_460: "f32[432, 1, 5, 5]", primals_461: "f32[432, 432, 1, 1]", primals_462: "f32[432]", primals_463: "f32[432]", primals_464: "f32[432, 1, 5, 5]", primals_465: "f32[432, 432, 1, 1]", primals_466: "f32[432]", primals_467: "f32[432]", primals_468: "f32[432, 1, 7, 7]", primals_469: "f32[432, 432, 1, 1]", primals_470: "f32[432]", primals_471: "f32[432]", primals_472: "f32[432, 1, 7, 7]", primals_473: "f32[432, 432, 1, 1]", primals_474: "f32[432]", primals_475: "f32[432]", primals_476: "f32[432, 1, 5, 5]", primals_477: "f32[432, 432, 1, 1]", primals_478: "f32[432]", primals_479: "f32[432]", primals_480: "f32[432, 1, 5, 5]", primals_481: "f32[432, 432, 1, 1]", primals_482: "f32[432]", primals_483: "f32[432]", primals_484: "f32[432, 1, 3, 3]", primals_485: "f32[432, 432, 1, 1]", primals_486: "f32[432]", primals_487: "f32[432]", primals_488: "f32[432, 1, 3, 3]", primals_489: "f32[432, 432, 1, 1]", primals_490: "f32[432]", primals_491: "f32[432]", primals_492: "f32[432, 1, 3, 3]", primals_493: "f32[432, 432, 1, 1]", primals_494: "f32[432]", primals_495: "f32[432]", primals_496: "f32[432, 1, 3, 3]", primals_497: "f32[432, 432, 1, 1]", primals_498: "f32[432]", primals_499: "f32[432]", primals_500: "f32[432, 1, 3, 3]", primals_501: "f32[432, 432, 1, 1]", primals_502: "f32[432]", primals_503: "f32[432]", primals_504: "f32[432, 1, 3, 3]", primals_505: "f32[432, 432, 1, 1]", primals_506: "f32[432]", primals_507: "f32[432]", primals_508: "f32[432, 2160, 1, 1]", primals_509: "f32[432]", primals_510: "f32[432]", primals_511: "f32[432, 2160, 1, 1]", primals_512: "f32[432]", primals_513: "f32[432]", primals_514: "f32[432, 1, 5, 5]", primals_515: "f32[432, 432, 1, 1]", primals_516: "f32[432]", primals_517: "f32[432]", primals_518: "f32[432, 1, 5, 5]", primals_519: "f32[432, 432, 1, 1]", primals_520: "f32[432]", primals_521: "f32[432]", primals_522: "f32[432, 1, 7, 7]", primals_523: "f32[432, 432, 1, 1]", primals_524: "f32[432]", primals_525: "f32[432]", primals_526: "f32[432, 1, 7, 7]", primals_527: "f32[432, 432, 1, 1]", primals_528: "f32[432]", primals_529: "f32[432]", primals_530: "f32[432, 1, 5, 5]", primals_531: "f32[432, 432, 1, 1]", primals_532: "f32[432]", primals_533: "f32[432]", primals_534: "f32[432, 1, 5, 5]", primals_535: "f32[432, 432, 1, 1]", primals_536: "f32[432]", primals_537: "f32[432]", primals_538: "f32[432, 1, 3, 3]", primals_539: "f32[432, 432, 1, 1]", primals_540: "f32[432]", primals_541: "f32[432]", primals_542: "f32[432, 1, 3, 3]", primals_543: "f32[432, 432, 1, 1]", primals_544: "f32[432]", primals_545: "f32[432]", primals_546: "f32[432, 1, 3, 3]", primals_547: "f32[432, 432, 1, 1]", primals_548: "f32[432]", primals_549: "f32[432]", primals_550: "f32[432, 1, 3, 3]", primals_551: "f32[432, 432, 1, 1]", primals_552: "f32[432]", primals_553: "f32[432]", primals_554: "f32[432, 1, 3, 3]", primals_555: "f32[432, 432, 1, 1]", primals_556: "f32[432]", primals_557: "f32[432]", primals_558: "f32[432, 1, 3, 3]", primals_559: "f32[432, 432, 1, 1]", primals_560: "f32[432]", primals_561: "f32[432]", primals_562: "f32[864, 2160, 1, 1]", primals_563: "f32[864]", primals_564: "f32[864]", primals_565: "f32[864, 2160, 1, 1]", primals_566: "f32[864]", primals_567: "f32[864]", primals_568: "f32[864, 864, 1, 1]", primals_569: "f32[864]", primals_570: "f32[864]", primals_571: "f32[864, 1, 5, 5]", primals_572: "f32[864, 864, 1, 1]", primals_573: "f32[864]", primals_574: "f32[864]", primals_575: "f32[864, 864, 1, 1]", primals_576: "f32[864]", primals_577: "f32[864]", primals_578: "f32[864, 1, 7, 7]", primals_579: "f32[864, 864, 1, 1]", primals_580: "f32[864]", primals_581: "f32[864]", primals_582: "f32[864, 864, 1, 1]", primals_583: "f32[864]", primals_584: "f32[864]", primals_585: "f32[864, 1, 5, 5]", primals_586: "f32[864, 864, 1, 1]", primals_587: "f32[864]", primals_588: "f32[864]", primals_589: "f32[864, 864, 1, 1]", primals_590: "f32[864]", primals_591: "f32[864]", primals_592: "f32[864, 1, 3, 3]", primals_593: "f32[864, 864, 1, 1]", primals_594: "f32[864]", primals_595: "f32[864]", primals_596: "f32[864, 1, 3, 3]", primals_597: "f32[864, 864, 1, 1]", primals_598: "f32[864]", primals_599: "f32[864]", primals_600: "f32[864, 1, 3, 3]", primals_601: "f32[864, 864, 1, 1]", primals_602: "f32[864]", primals_603: "f32[864]", primals_604: "f32[864, 864, 1, 1]", primals_605: "f32[864]", primals_606: "f32[864]", primals_607: "f32[864, 1, 3, 3]", primals_608: "f32[864, 864, 1, 1]", primals_609: "f32[864]", primals_610: "f32[864]", primals_611: "f32[864]", primals_612: "f32[864]", primals_613: "f32[432, 2160, 1, 1]", primals_614: "f32[432, 2160, 1, 1]", primals_615: "f32[864]", primals_616: "f32[864]", primals_617: "f32[864, 4320, 1, 1]", primals_618: "f32[864]", primals_619: "f32[864]", primals_620: "f32[864, 1, 5, 5]", primals_621: "f32[864, 864, 1, 1]", primals_622: "f32[864]", primals_623: "f32[864]", primals_624: "f32[864, 1, 5, 5]", primals_625: "f32[864, 864, 1, 1]", primals_626: "f32[864]", primals_627: "f32[864]", primals_628: "f32[864, 1, 7, 7]", primals_629: "f32[864, 864, 1, 1]", primals_630: "f32[864]", primals_631: "f32[864]", primals_632: "f32[864, 1, 7, 7]", primals_633: "f32[864, 864, 1, 1]", primals_634: "f32[864]", primals_635: "f32[864]", primals_636: "f32[864, 1, 5, 5]", primals_637: "f32[864, 864, 1, 1]", primals_638: "f32[864]", primals_639: "f32[864]", primals_640: "f32[864, 1, 5, 5]", primals_641: "f32[864, 864, 1, 1]", primals_642: "f32[864]", primals_643: "f32[864]", primals_644: "f32[864, 1, 3, 3]", primals_645: "f32[864, 864, 1, 1]", primals_646: "f32[864]", primals_647: "f32[864]", primals_648: "f32[864, 1, 3, 3]", primals_649: "f32[864, 864, 1, 1]", primals_650: "f32[864]", primals_651: "f32[864]", primals_652: "f32[864, 1, 3, 3]", primals_653: "f32[864, 864, 1, 1]", primals_654: "f32[864]", primals_655: "f32[864]", primals_656: "f32[864, 1, 3, 3]", primals_657: "f32[864, 864, 1, 1]", primals_658: "f32[864]", primals_659: "f32[864]", primals_660: "f32[864, 1, 3, 3]", primals_661: "f32[864, 864, 1, 1]", primals_662: "f32[864]", primals_663: "f32[864]", primals_664: "f32[864, 1, 3, 3]", primals_665: "f32[864, 864, 1, 1]", primals_666: "f32[864]", primals_667: "f32[864]", primals_668: "f32[864, 4320, 1, 1]", primals_669: "f32[864]", primals_670: "f32[864]", primals_671: "f32[864, 4320, 1, 1]", primals_672: "f32[864]", primals_673: "f32[864]", primals_674: "f32[864, 1, 5, 5]", primals_675: "f32[864, 864, 1, 1]", primals_676: "f32[864]", primals_677: "f32[864]", primals_678: "f32[864, 1, 5, 5]", primals_679: "f32[864, 864, 1, 1]", primals_680: "f32[864]", primals_681: "f32[864]", primals_682: "f32[864, 1, 7, 7]", primals_683: "f32[864, 864, 1, 1]", primals_684: "f32[864]", primals_685: "f32[864]", primals_686: "f32[864, 1, 7, 7]", primals_687: "f32[864, 864, 1, 1]", primals_688: "f32[864]", primals_689: "f32[864]", primals_690: "f32[864, 1, 5, 5]", primals_691: "f32[864, 864, 1, 1]", primals_692: "f32[864]", primals_693: "f32[864]", primals_694: "f32[864, 1, 5, 5]", primals_695: "f32[864, 864, 1, 1]", primals_696: "f32[864]", primals_697: "f32[864]", primals_698: "f32[864, 1, 3, 3]", primals_699: "f32[864, 864, 1, 1]", primals_700: "f32[864]", primals_701: "f32[864]", primals_702: "f32[864, 1, 3, 3]", primals_703: "f32[864, 864, 1, 1]", primals_704: "f32[864]", primals_705: "f32[864]", primals_706: "f32[864, 1, 3, 3]", primals_707: "f32[864, 864, 1, 1]", primals_708: "f32[864]", primals_709: "f32[864]", primals_710: "f32[864, 1, 3, 3]", primals_711: "f32[864, 864, 1, 1]", primals_712: "f32[864]", primals_713: "f32[864]", primals_714: "f32[864, 1, 3, 3]", primals_715: "f32[864, 864, 1, 1]", primals_716: "f32[864]", primals_717: "f32[864]", primals_718: "f32[864, 1, 3, 3]", primals_719: "f32[864, 864, 1, 1]", primals_720: "f32[864]", primals_721: "f32[864]", primals_722: "f32[864, 4320, 1, 1]", primals_723: "f32[864]", primals_724: "f32[864]", primals_725: "f32[864, 4320, 1, 1]", primals_726: "f32[864]", primals_727: "f32[864]", primals_728: "f32[864, 1, 5, 5]", primals_729: "f32[864, 864, 1, 1]", primals_730: "f32[864]", primals_731: "f32[864]", primals_732: "f32[864, 1, 5, 5]", primals_733: "f32[864, 864, 1, 1]", primals_734: "f32[864]", primals_735: "f32[864]", primals_736: "f32[864, 1, 7, 7]", primals_737: "f32[864, 864, 1, 1]", primals_738: "f32[864]", primals_739: "f32[864]", primals_740: "f32[864, 1, 7, 7]", primals_741: "f32[864, 864, 1, 1]", primals_742: "f32[864]", primals_743: "f32[864]", primals_744: "f32[864, 1, 5, 5]", primals_745: "f32[864, 864, 1, 1]", primals_746: "f32[864]", primals_747: "f32[864]", primals_748: "f32[864, 1, 5, 5]", primals_749: "f32[864, 864, 1, 1]", primals_750: "f32[864]", primals_751: "f32[864]", primals_752: "f32[864, 1, 3, 3]", primals_753: "f32[864, 864, 1, 1]", primals_754: "f32[864]", primals_755: "f32[864]", primals_756: "f32[864, 1, 3, 3]", primals_757: "f32[864, 864, 1, 1]", primals_758: "f32[864]", primals_759: "f32[864]", primals_760: "f32[864, 1, 3, 3]", primals_761: "f32[864, 864, 1, 1]", primals_762: "f32[864]", primals_763: "f32[864]", primals_764: "f32[864, 1, 3, 3]", primals_765: "f32[864, 864, 1, 1]", primals_766: "f32[864]", primals_767: "f32[864]", primals_768: "f32[864, 1, 3, 3]", primals_769: "f32[864, 864, 1, 1]", primals_770: "f32[864]", primals_771: "f32[864]", primals_772: "f32[864, 1, 3, 3]", primals_773: "f32[864, 864, 1, 1]", primals_774: "f32[864]", primals_775: "f32[864]", primals_776: "f32[1000, 4320]", primals_777: "f32[1000]", primals_778: "i64[]", primals_779: "f32[96]", primals_780: "f32[96]", primals_781: "f32[54]", primals_782: "f32[54]", primals_783: "i64[]", primals_784: "f32[54]", primals_785: "f32[54]", primals_786: "i64[]", primals_787: "f32[54]", primals_788: "f32[54]", primals_789: "i64[]", primals_790: "f32[54]", primals_791: "f32[54]", primals_792: "i64[]", primals_793: "f32[54]", primals_794: "f32[54]", primals_795: "i64[]", primals_796: "f32[54]", primals_797: "f32[54]", primals_798: "i64[]", primals_799: "f32[54]", primals_800: "f32[54]", primals_801: "i64[]", primals_802: "f32[54]", primals_803: "f32[54]", primals_804: "i64[]", primals_805: "f32[54]", primals_806: "f32[54]", primals_807: "i64[]", primals_808: "f32[54]", primals_809: "f32[54]", primals_810: "i64[]", primals_811: "f32[54]", primals_812: "f32[54]", primals_813: "i64[]", primals_814: "f32[54]", primals_815: "f32[54]", primals_816: "i64[]", primals_817: "f32[54]", primals_818: "f32[54]", primals_819: "i64[]", primals_820: "f32[54]", primals_821: "f32[54]", primals_822: "i64[]", primals_823: "f32[54]", primals_824: "f32[54]", primals_825: "i64[]", primals_826: "f32[108]", primals_827: "f32[108]", primals_828: "i64[]", primals_829: "f32[108]", primals_830: "f32[108]", primals_831: "i64[]", primals_832: "f32[108]", primals_833: "f32[108]", primals_834: "i64[]", primals_835: "f32[108]", primals_836: "f32[108]", primals_837: "i64[]", primals_838: "f32[108]", primals_839: "f32[108]", primals_840: "i64[]", primals_841: "f32[108]", primals_842: "f32[108]", primals_843: "i64[]", primals_844: "f32[108]", primals_845: "f32[108]", primals_846: "i64[]", primals_847: "f32[108]", primals_848: "f32[108]", primals_849: "i64[]", primals_850: "f32[108]", primals_851: "f32[108]", primals_852: "i64[]", primals_853: "f32[108]", primals_854: "f32[108]", primals_855: "i64[]", primals_856: "f32[108]", primals_857: "f32[108]", primals_858: "i64[]", primals_859: "f32[108]", primals_860: "f32[108]", primals_861: "i64[]", primals_862: "f32[108]", primals_863: "f32[108]", primals_864: "i64[]", primals_865: "f32[108]", primals_866: "f32[108]", primals_867: "i64[]", primals_868: "f32[108]", primals_869: "f32[108]", primals_870: "i64[]", primals_871: "f32[216]", primals_872: "f32[216]", primals_873: "i64[]", primals_874: "f32[216]", primals_875: "f32[216]", primals_876: "i64[]", primals_877: "f32[216]", primals_878: "f32[216]", primals_879: "i64[]", primals_880: "f32[216]", primals_881: "f32[216]", primals_882: "i64[]", primals_883: "f32[216]", primals_884: "f32[216]", primals_885: "i64[]", primals_886: "f32[216]", primals_887: "f32[216]", primals_888: "i64[]", primals_889: "f32[216]", primals_890: "f32[216]", primals_891: "i64[]", primals_892: "f32[216]", primals_893: "f32[216]", primals_894: "i64[]", primals_895: "f32[216]", primals_896: "f32[216]", primals_897: "i64[]", primals_898: "f32[216]", primals_899: "f32[216]", primals_900: "i64[]", primals_901: "f32[216]", primals_902: "f32[216]", primals_903: "i64[]", primals_904: "f32[216]", primals_905: "f32[216]", primals_906: "i64[]", primals_907: "f32[216]", primals_908: "f32[216]", primals_909: "i64[]", primals_910: "f32[216]", primals_911: "f32[216]", primals_912: "i64[]", primals_913: "f32[216]", primals_914: "f32[216]", primals_915: "i64[]", primals_916: "f32[216]", primals_917: "f32[216]", primals_918: "i64[]", primals_919: "f32[216]", primals_920: "f32[216]", primals_921: "i64[]", primals_922: "f32[216]", primals_923: "f32[216]", primals_924: "i64[]", primals_925: "f32[216]", primals_926: "f32[216]", primals_927: "i64[]", primals_928: "f32[216]", primals_929: "f32[216]", primals_930: "i64[]", primals_931: "f32[216]", primals_932: "f32[216]", primals_933: "i64[]", primals_934: "f32[216]", primals_935: "f32[216]", primals_936: "i64[]", primals_937: "f32[216]", primals_938: "f32[216]", primals_939: "i64[]", primals_940: "f32[216]", primals_941: "f32[216]", primals_942: "i64[]", primals_943: "f32[216]", primals_944: "f32[216]", primals_945: "i64[]", primals_946: "f32[216]", primals_947: "f32[216]", primals_948: "i64[]", primals_949: "f32[216]", primals_950: "f32[216]", primals_951: "i64[]", primals_952: "f32[216]", primals_953: "f32[216]", primals_954: "i64[]", primals_955: "f32[216]", primals_956: "f32[216]", primals_957: "i64[]", primals_958: "f32[216]", primals_959: "f32[216]", primals_960: "i64[]", primals_961: "f32[216]", primals_962: "f32[216]", primals_963: "i64[]", primals_964: "f32[216]", primals_965: "f32[216]", primals_966: "i64[]", primals_967: "f32[216]", primals_968: "f32[216]", primals_969: "i64[]", primals_970: "f32[216]", primals_971: "f32[216]", primals_972: "i64[]", primals_973: "f32[216]", primals_974: "f32[216]", primals_975: "i64[]", primals_976: "f32[216]", primals_977: "f32[216]", primals_978: "i64[]", primals_979: "f32[216]", primals_980: "f32[216]", primals_981: "i64[]", primals_982: "f32[216]", primals_983: "f32[216]", primals_984: "i64[]", primals_985: "f32[216]", primals_986: "f32[216]", primals_987: "i64[]", primals_988: "f32[216]", primals_989: "f32[216]", primals_990: "i64[]", primals_991: "f32[216]", primals_992: "f32[216]", primals_993: "i64[]", primals_994: "f32[216]", primals_995: "f32[216]", primals_996: "i64[]", primals_997: "f32[216]", primals_998: "f32[216]", primals_999: "i64[]", primals_1000: "f32[216]", primals_1001: "f32[216]", primals_1002: "i64[]", primals_1003: "f32[216]", primals_1004: "f32[216]", primals_1005: "i64[]", primals_1006: "f32[216]", primals_1007: "f32[216]", primals_1008: "i64[]", primals_1009: "f32[216]", primals_1010: "f32[216]", primals_1011: "i64[]", primals_1012: "f32[216]", primals_1013: "f32[216]", primals_1014: "i64[]", primals_1015: "f32[216]", primals_1016: "f32[216]", primals_1017: "i64[]", primals_1018: "f32[216]", primals_1019: "f32[216]", primals_1020: "i64[]", primals_1021: "f32[216]", primals_1022: "f32[216]", primals_1023: "i64[]", primals_1024: "f32[216]", primals_1025: "f32[216]", primals_1026: "i64[]", primals_1027: "f32[216]", primals_1028: "f32[216]", primals_1029: "i64[]", primals_1030: "f32[216]", primals_1031: "f32[216]", primals_1032: "i64[]", primals_1033: "f32[216]", primals_1034: "f32[216]", primals_1035: "i64[]", primals_1036: "f32[216]", primals_1037: "f32[216]", primals_1038: "i64[]", primals_1039: "f32[432]", primals_1040: "f32[432]", primals_1041: "i64[]", primals_1042: "f32[432]", primals_1043: "f32[432]", primals_1044: "i64[]", primals_1045: "f32[432]", primals_1046: "f32[432]", primals_1047: "i64[]", primals_1048: "f32[432]", primals_1049: "f32[432]", primals_1050: "i64[]", primals_1051: "f32[432]", primals_1052: "f32[432]", primals_1053: "i64[]", primals_1054: "f32[432]", primals_1055: "f32[432]", primals_1056: "i64[]", primals_1057: "f32[432]", primals_1058: "f32[432]", primals_1059: "i64[]", primals_1060: "f32[432]", primals_1061: "f32[432]", primals_1062: "i64[]", primals_1063: "f32[432]", primals_1064: "f32[432]", primals_1065: "i64[]", primals_1066: "f32[432]", primals_1067: "f32[432]", primals_1068: "i64[]", primals_1069: "f32[432]", primals_1070: "f32[432]", primals_1071: "i64[]", primals_1072: "f32[432]", primals_1073: "f32[432]", primals_1074: "i64[]", primals_1075: "f32[432]", primals_1076: "f32[432]", primals_1077: "i64[]", primals_1078: "f32[432]", primals_1079: "f32[432]", primals_1080: "i64[]", primals_1081: "f32[432]", primals_1082: "f32[432]", primals_1083: "i64[]", primals_1084: "f32[432]", primals_1085: "f32[432]", primals_1086: "i64[]", primals_1087: "f32[432]", primals_1088: "f32[432]", primals_1089: "i64[]", primals_1090: "f32[432]", primals_1091: "f32[432]", primals_1092: "i64[]", primals_1093: "f32[432]", primals_1094: "f32[432]", primals_1095: "i64[]", primals_1096: "f32[432]", primals_1097: "f32[432]", primals_1098: "i64[]", primals_1099: "f32[432]", primals_1100: "f32[432]", primals_1101: "i64[]", primals_1102: "f32[432]", primals_1103: "f32[432]", primals_1104: "i64[]", primals_1105: "f32[432]", primals_1106: "f32[432]", primals_1107: "i64[]", primals_1108: "f32[432]", primals_1109: "f32[432]", primals_1110: "i64[]", primals_1111: "f32[432]", primals_1112: "f32[432]", primals_1113: "i64[]", primals_1114: "f32[432]", primals_1115: "f32[432]", primals_1116: "i64[]", primals_1117: "f32[432]", primals_1118: "f32[432]", primals_1119: "i64[]", primals_1120: "f32[432]", primals_1121: "f32[432]", primals_1122: "i64[]", primals_1123: "f32[432]", primals_1124: "f32[432]", primals_1125: "i64[]", primals_1126: "f32[432]", primals_1127: "f32[432]", primals_1128: "i64[]", primals_1129: "f32[432]", primals_1130: "f32[432]", primals_1131: "i64[]", primals_1132: "f32[432]", primals_1133: "f32[432]", primals_1134: "i64[]", primals_1135: "f32[432]", primals_1136: "f32[432]", primals_1137: "i64[]", primals_1138: "f32[432]", primals_1139: "f32[432]", primals_1140: "i64[]", primals_1141: "f32[432]", primals_1142: "f32[432]", primals_1143: "i64[]", primals_1144: "f32[432]", primals_1145: "f32[432]", primals_1146: "i64[]", primals_1147: "f32[432]", primals_1148: "f32[432]", primals_1149: "i64[]", primals_1150: "f32[432]", primals_1151: "f32[432]", primals_1152: "i64[]", primals_1153: "f32[432]", primals_1154: "f32[432]", primals_1155: "i64[]", primals_1156: "f32[432]", primals_1157: "f32[432]", primals_1158: "i64[]", primals_1159: "f32[432]", primals_1160: "f32[432]", primals_1161: "i64[]", primals_1162: "f32[432]", primals_1163: "f32[432]", primals_1164: "i64[]", primals_1165: "f32[432]", primals_1166: "f32[432]", primals_1167: "i64[]", primals_1168: "f32[432]", primals_1169: "f32[432]", primals_1170: "i64[]", primals_1171: "f32[432]", primals_1172: "f32[432]", primals_1173: "i64[]", primals_1174: "f32[432]", primals_1175: "f32[432]", primals_1176: "i64[]", primals_1177: "f32[432]", primals_1178: "f32[432]", primals_1179: "i64[]", primals_1180: "f32[432]", primals_1181: "f32[432]", primals_1182: "i64[]", primals_1183: "f32[432]", primals_1184: "f32[432]", primals_1185: "i64[]", primals_1186: "f32[432]", primals_1187: "f32[432]", primals_1188: "i64[]", primals_1189: "f32[432]", primals_1190: "f32[432]", primals_1191: "i64[]", primals_1192: "f32[432]", primals_1193: "f32[432]", primals_1194: "i64[]", primals_1195: "f32[432]", primals_1196: "f32[432]", primals_1197: "i64[]", primals_1198: "f32[432]", primals_1199: "f32[432]", primals_1200: "i64[]", primals_1201: "f32[432]", primals_1202: "f32[432]", primals_1203: "i64[]", primals_1204: "f32[432]", primals_1205: "f32[432]", primals_1206: "i64[]", primals_1207: "f32[432]", primals_1208: "f32[432]", primals_1209: "i64[]", primals_1210: "f32[864]", primals_1211: "f32[864]", primals_1212: "i64[]", primals_1213: "f32[864]", primals_1214: "f32[864]", primals_1215: "i64[]", primals_1216: "f32[864]", primals_1217: "f32[864]", primals_1218: "i64[]", primals_1219: "f32[864]", primals_1220: "f32[864]", primals_1221: "i64[]", primals_1222: "f32[864]", primals_1223: "f32[864]", primals_1224: "i64[]", primals_1225: "f32[864]", primals_1226: "f32[864]", primals_1227: "i64[]", primals_1228: "f32[864]", primals_1229: "f32[864]", primals_1230: "i64[]", primals_1231: "f32[864]", primals_1232: "f32[864]", primals_1233: "i64[]", primals_1234: "f32[864]", primals_1235: "f32[864]", primals_1236: "i64[]", primals_1237: "f32[864]", primals_1238: "f32[864]", primals_1239: "i64[]", primals_1240: "f32[864]", primals_1241: "f32[864]", primals_1242: "i64[]", primals_1243: "f32[864]", primals_1244: "f32[864]", primals_1245: "i64[]", primals_1246: "f32[864]", primals_1247: "f32[864]", primals_1248: "i64[]", primals_1249: "f32[864]", primals_1250: "f32[864]", primals_1251: "i64[]", primals_1252: "f32[864]", primals_1253: "f32[864]", primals_1254: "i64[]", primals_1255: "f32[864]", primals_1256: "f32[864]", primals_1257: "i64[]", primals_1258: "f32[864]", primals_1259: "f32[864]", primals_1260: "i64[]", primals_1261: "f32[864]", primals_1262: "f32[864]", primals_1263: "i64[]", primals_1264: "f32[864]", primals_1265: "f32[864]", primals_1266: "i64[]", primals_1267: "f32[864]", primals_1268: "f32[864]", primals_1269: "i64[]", primals_1270: "f32[864]", primals_1271: "f32[864]", primals_1272: "i64[]", primals_1273: "f32[864]", primals_1274: "f32[864]", primals_1275: "i64[]", primals_1276: "f32[864]", primals_1277: "f32[864]", primals_1278: "i64[]", primals_1279: "f32[864]", primals_1280: "f32[864]", primals_1281: "i64[]", primals_1282: "f32[864]", primals_1283: "f32[864]", primals_1284: "i64[]", primals_1285: "f32[864]", primals_1286: "f32[864]", primals_1287: "i64[]", primals_1288: "f32[864]", primals_1289: "f32[864]", primals_1290: "i64[]", primals_1291: "f32[864]", primals_1292: "f32[864]", primals_1293: "i64[]", primals_1294: "f32[864]", primals_1295: "f32[864]", primals_1296: "i64[]", primals_1297: "f32[864]", primals_1298: "f32[864]", primals_1299: "i64[]", primals_1300: "f32[864]", primals_1301: "f32[864]", primals_1302: "i64[]", primals_1303: "f32[864]", primals_1304: "f32[864]", primals_1305: "i64[]", primals_1306: "f32[864]", primals_1307: "f32[864]", primals_1308: "i64[]", primals_1309: "f32[864]", primals_1310: "f32[864]", primals_1311: "i64[]", primals_1312: "f32[864]", primals_1313: "f32[864]", primals_1314: "i64[]", primals_1315: "f32[864]", primals_1316: "f32[864]", primals_1317: "i64[]", primals_1318: "f32[864]", primals_1319: "f32[864]", primals_1320: "i64[]", primals_1321: "f32[864]", primals_1322: "f32[864]", primals_1323: "i64[]", primals_1324: "f32[864]", primals_1325: "f32[864]", primals_1326: "i64[]", primals_1327: "f32[864]", primals_1328: "f32[864]", primals_1329: "i64[]", primals_1330: "f32[864]", primals_1331: "f32[864]", primals_1332: "i64[]", primals_1333: "f32[864]", primals_1334: "f32[864]", primals_1335: "i64[]", primals_1336: "f32[864]", primals_1337: "f32[864]", primals_1338: "i64[]", primals_1339: "f32[864]", primals_1340: "f32[864]", primals_1341: "i64[]", primals_1342: "f32[864]", primals_1343: "f32[864]", primals_1344: "i64[]", primals_1345: "f32[864]", primals_1346: "f32[864]", primals_1347: "i64[]", primals_1348: "f32[864]", primals_1349: "f32[864]", primals_1350: "i64[]", primals_1351: "f32[864]", primals_1352: "f32[864]", primals_1353: "i64[]", primals_1354: "f32[864]", primals_1355: "f32[864]", primals_1356: "i64[]", primals_1357: "f32[864]", primals_1358: "f32[864]", primals_1359: "i64[]", primals_1360: "f32[864]", primals_1361: "f32[864]", primals_1362: "i64[]", primals_1363: "f32[864]", primals_1364: "f32[864]", primals_1365: "i64[]", primals_1366: "f32[864]", primals_1367: "f32[864]", primals_1368: "i64[]", primals_1369: "f32[864]", primals_1370: "f32[864]", primals_1371: "i64[]", primals_1372: "f32[864]", primals_1373: "f32[864]", primals_1374: "i64[]", primals_1375: "f32[864]", primals_1376: "f32[864]", primals_1377: "i64[]", primals_1378: "f32[864]", primals_1379: "f32[864]", primals_1380: "i64[]", primals_1381: "f32[8, 3, 331, 331]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 96, 165, 165]" = torch.ops.aten.convolution.default(primals_1381, primals_27, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_778, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 96, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 96, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
    rsqrt: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 96, 165, 165]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 96, 165, 165]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[96]" = torch.ops.aten.mul.Tensor(primals_779, 0.9)
    add_2: "f32[96]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[96]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000045913893085);  squeeze_2 = None
    mul_4: "f32[96]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[96]" = torch.ops.aten.mul.Tensor(primals_780, 0.9)
    add_3: "f32[96]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 96, 165, 165]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 96, 165, 165]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu: "f32[8, 96, 165, 165]" = torch.ops.aten.relu.default(add_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_1: "f32[8, 54, 165, 165]" = torch.ops.aten.convolution.default(relu, primals_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_783, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 54, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 54, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
    rsqrt_1: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 54, 165, 165]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 54, 165, 165]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[54]" = torch.ops.aten.mul.Tensor(primals_781, 0.9)
    add_7: "f32[54]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000045913893085);  squeeze_5 = None
    mul_11: "f32[54]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[54]" = torch.ops.aten.mul.Tensor(primals_782, 0.9)
    add_8: "f32[54]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_5: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 54, 165, 165]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_7: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 54, 165, 165]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd: "f32[8, 96, 169, 169]" = torch.ops.aten.constant_pad_nd.default(relu, [2, 2, 2, 2], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_2: "f32[8, 96, 83, 83]" = torch.ops.aten.convolution.default(constant_pad_nd, primals_3, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_3: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(convolution_2, primals_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_786, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 54, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 54, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 0.001)
    rsqrt_2: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_5)
    mul_14: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[54]" = torch.ops.aten.mul.Tensor(primals_784, 0.9)
    add_12: "f32[54]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000181451978734);  squeeze_8 = None
    mul_18: "f32[54]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[54]" = torch.ops.aten.mul.Tensor(primals_785, 0.9)
    add_13: "f32[54]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_9: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_11: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_2: "f32[8, 54, 83, 83]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_4: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(relu_2, primals_34, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_5: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(convolution_4, primals_35, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_789, 1)
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 54, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 54, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 0.001)
    rsqrt_3: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_7)
    mul_21: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[54]" = torch.ops.aten.mul.Tensor(primals_787, 0.9)
    add_17: "f32[54]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_24: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000181451978734);  squeeze_11 = None
    mul_25: "f32[54]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[54]" = torch.ops.aten.mul.Tensor(primals_788, 0.9)
    add_18: "f32[54]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1)
    unsqueeze_13: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
    unsqueeze_15: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_1: "f32[8, 96, 167, 167]" = torch.ops.aten.constant_pad_nd.default(add_4, [1, 1, 1, 1], -inf);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd_1, [3, 3], [2, 2])
    getitem_8: "f32[8, 96, 83, 83]" = max_pool2d_with_indices[0]
    getitem_9: "i64[8, 96, 83, 83]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    convolution_6: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(getitem_8, primals_38, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_792, 1)
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 54, 1, 1]" = var_mean_4[0]
    getitem_11: "f32[1, 54, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 0.001)
    rsqrt_4: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_11)
    mul_28: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_13: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[54]" = torch.ops.aten.mul.Tensor(primals_790, 0.9)
    add_22: "f32[54]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_31: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000181451978734);  squeeze_14 = None
    mul_32: "f32[54]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[54]" = torch.ops.aten.mul.Tensor(primals_791, 0.9)
    add_23: "f32[54]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_17: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_19: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_25: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(add_19, add_24);  add_19 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_3: "f32[8, 54, 165, 165]" = torch.ops.aten.relu.default(add_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_2: "f32[8, 54, 171, 171]" = torch.ops.aten.constant_pad_nd.default(relu_3, [3, 3, 3, 3], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_7: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(constant_pad_nd_2, primals_4, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_8: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(convolution_7, primals_41, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_795, 1)
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 54, 1, 1]" = var_mean_5[0]
    getitem_13: "f32[1, 54, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 0.001)
    rsqrt_5: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_5: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_13)
    mul_35: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_16: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[54]" = torch.ops.aten.mul.Tensor(primals_793, 0.9)
    add_28: "f32[54]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_38: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000181451978734);  squeeze_17 = None
    mul_39: "f32[54]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[54]" = torch.ops.aten.mul.Tensor(primals_794, 0.9)
    add_29: "f32[54]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1)
    unsqueeze_21: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
    unsqueeze_23: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_30: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_4: "f32[8, 54, 83, 83]" = torch.ops.aten.relu.default(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_9: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(relu_4, primals_44, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_10: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(convolution_9, primals_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_798, 1)
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 54, 1, 1]" = var_mean_6[0]
    getitem_15: "f32[1, 54, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_32: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 0.001)
    rsqrt_6: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_6: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_15)
    mul_42: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_19: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[54]" = torch.ops.aten.mul.Tensor(primals_796, 0.9)
    add_33: "f32[54]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_45: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000181451978734);  squeeze_20 = None
    mul_46: "f32[54]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[54]" = torch.ops.aten.mul.Tensor(primals_797, 0.9)
    add_34: "f32[54]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1)
    unsqueeze_25: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1);  primals_47 = None
    unsqueeze_27: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_35: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_3: "f32[8, 54, 167, 167]" = torch.ops.aten.constant_pad_nd.default(add_9, [1, 1, 1, 1], -inf);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd_3, [3, 3], [2, 2])
    getitem_16: "f32[8, 54, 83, 83]" = max_pool2d_with_indices_1[0]
    getitem_17: "i64[8, 54, 83, 83]" = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_36: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(add_35, getitem_16);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_4: "f32[8, 54, 169, 169]" = torch.ops.aten.constant_pad_nd.default(relu_3, [2, 2, 2, 2], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_11: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(constant_pad_nd_4, primals_5, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_12: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(convolution_11, primals_48, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_37: "i64[]" = torch.ops.aten.add.Tensor(primals_801, 1)
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 54, 1, 1]" = var_mean_7[0]
    getitem_19: "f32[1, 54, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_38: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 0.001)
    rsqrt_7: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_7: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_19)
    mul_49: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_22: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[54]" = torch.ops.aten.mul.Tensor(primals_799, 0.9)
    add_39: "f32[54]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_52: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000181451978734);  squeeze_23 = None
    mul_53: "f32[54]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[54]" = torch.ops.aten.mul.Tensor(primals_800, 0.9)
    add_40: "f32[54]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_29: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_31: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_41: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_6: "f32[8, 54, 83, 83]" = torch.ops.aten.relu.default(add_41);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_13: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(relu_6, primals_51, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_14: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(convolution_13, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_42: "i64[]" = torch.ops.aten.add.Tensor(primals_804, 1)
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 54, 1, 1]" = var_mean_8[0]
    getitem_21: "f32[1, 54, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_43: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 0.001)
    rsqrt_8: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_8: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_21)
    mul_56: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_25: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[54]" = torch.ops.aten.mul.Tensor(primals_802, 0.9)
    add_44: "f32[54]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_59: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000181451978734);  squeeze_26 = None
    mul_60: "f32[54]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[54]" = torch.ops.aten.mul.Tensor(primals_803, 0.9)
    add_45: "f32[54]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_33: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_35: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_46: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_5: "f32[8, 54, 167, 167]" = torch.ops.aten.constant_pad_nd.default(relu_3, [1, 1, 1, 1], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_15: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(constant_pad_nd_5, primals_6, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_16: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(convolution_15, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_807, 1)
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 54, 1, 1]" = var_mean_9[0]
    getitem_23: "f32[1, 54, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_48: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 0.001)
    rsqrt_9: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_9: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_23)
    mul_63: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_28: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[54]" = torch.ops.aten.mul.Tensor(primals_805, 0.9)
    add_49: "f32[54]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_66: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000181451978734);  squeeze_29 = None
    mul_67: "f32[54]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[54]" = torch.ops.aten.mul.Tensor(primals_806, 0.9)
    add_50: "f32[54]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_37: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_39: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_51: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_8: "f32[8, 54, 83, 83]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_17: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(relu_8, primals_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_18: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(convolution_17, primals_59, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_810, 1)
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 54, 1, 1]" = var_mean_10[0]
    getitem_25: "f32[1, 54, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_53: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 0.001)
    rsqrt_10: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_10: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_25)
    mul_70: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_31: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_71: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_72: "f32[54]" = torch.ops.aten.mul.Tensor(primals_808, 0.9)
    add_54: "f32[54]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_32: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_73: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000181451978734);  squeeze_32 = None
    mul_74: "f32[54]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[54]" = torch.ops.aten.mul.Tensor(primals_809, 0.9)
    add_55: "f32[54]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_40: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1)
    unsqueeze_41: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_76: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
    unsqueeze_42: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
    unsqueeze_43: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_56: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_57: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(add_46, add_56);  add_46 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_9: "f32[8, 54, 83, 83]" = torch.ops.aten.relu.default(add_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_19: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(relu_9, primals_62, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_20: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(convolution_19, primals_63, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_58: "i64[]" = torch.ops.aten.add.Tensor(primals_813, 1)
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 54, 1, 1]" = var_mean_11[0]
    getitem_27: "f32[1, 54, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_59: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 0.001)
    rsqrt_11: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_11: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_27)
    mul_77: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_34: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_78: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_79: "f32[54]" = torch.ops.aten.mul.Tensor(primals_811, 0.9)
    add_60: "f32[54]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_35: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_80: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000181451978734);  squeeze_35 = None
    mul_81: "f32[54]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[54]" = torch.ops.aten.mul.Tensor(primals_812, 0.9)
    add_61: "f32[54]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_44: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1)
    unsqueeze_45: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_83: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
    unsqueeze_46: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1);  primals_65 = None
    unsqueeze_47: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_62: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_10: "f32[8, 54, 83, 83]" = torch.ops.aten.relu.default(add_62);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_21: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(relu_10, primals_66, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_22: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(convolution_21, primals_67, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_63: "i64[]" = torch.ops.aten.add.Tensor(primals_816, 1)
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 54, 1, 1]" = var_mean_12[0]
    getitem_29: "f32[1, 54, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_64: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 0.001)
    rsqrt_12: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_12: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_29)
    mul_84: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_37: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_85: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_86: "f32[54]" = torch.ops.aten.mul.Tensor(primals_814, 0.9)
    add_65: "f32[54]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_38: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_87: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000181451978734);  squeeze_38 = None
    mul_88: "f32[54]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[54]" = torch.ops.aten.mul.Tensor(primals_815, 0.9)
    add_66: "f32[54]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_48: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_49: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_90: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
    unsqueeze_50: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_51: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_67: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_68: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(add_67, getitem_16);  add_67 = getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_7: "f32[8, 96, 167, 167]" = torch.ops.aten.constant_pad_nd.default(relu, [1, 1, 1, 1], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_23: "f32[8, 96, 83, 83]" = torch.ops.aten.convolution.default(constant_pad_nd_7, primals_7, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_24: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(convolution_23, primals_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_69: "i64[]" = torch.ops.aten.add.Tensor(primals_819, 1)
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 54, 1, 1]" = var_mean_13[0]
    getitem_33: "f32[1, 54, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_70: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 0.001)
    rsqrt_13: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_13: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_33)
    mul_91: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_40: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_92: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_93: "f32[54]" = torch.ops.aten.mul.Tensor(primals_817, 0.9)
    add_71: "f32[54]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    squeeze_41: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_94: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000181451978734);  squeeze_41 = None
    mul_95: "f32[54]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
    mul_96: "f32[54]" = torch.ops.aten.mul.Tensor(primals_818, 0.9)
    add_72: "f32[54]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    unsqueeze_52: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_53: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_97: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
    unsqueeze_54: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_55: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_73: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_12: "f32[8, 54, 83, 83]" = torch.ops.aten.relu.default(add_73);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_25: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(relu_12, primals_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_26: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(convolution_25, primals_74, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_74: "i64[]" = torch.ops.aten.add.Tensor(primals_822, 1)
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 54, 1, 1]" = var_mean_14[0]
    getitem_35: "f32[1, 54, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_75: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 0.001)
    rsqrt_14: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    sub_14: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_35)
    mul_98: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_43: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_99: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_100: "f32[54]" = torch.ops.aten.mul.Tensor(primals_820, 0.9)
    add_76: "f32[54]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_44: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_101: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000181451978734);  squeeze_44 = None
    mul_102: "f32[54]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[54]" = torch.ops.aten.mul.Tensor(primals_821, 0.9)
    add_77: "f32[54]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_56: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_57: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_104: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
    unsqueeze_58: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_59: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_78: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_27: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(relu_3, primals_8, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_79: "i64[]" = torch.ops.aten.add.Tensor(primals_825, 1)
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 54, 1, 1]" = var_mean_15[0]
    getitem_37: "f32[1, 54, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_80: "f32[1, 54, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 0.001)
    rsqrt_15: "f32[1, 54, 1, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_15: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_37)
    mul_105: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_46: "f32[54]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_106: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_107: "f32[54]" = torch.ops.aten.mul.Tensor(primals_823, 0.9)
    add_81: "f32[54]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_47: "f32[54]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_108: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000181451978734);  squeeze_47 = None
    mul_109: "f32[54]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[54]" = torch.ops.aten.mul.Tensor(primals_824, 0.9)
    add_82: "f32[54]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_60: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_61: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_111: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
    unsqueeze_62: "f32[54, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_63: "f32[54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_83: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_84: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(add_78, add_83);  add_78 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat: "f32[8, 270, 83, 83]" = torch.ops.aten.cat.default([add_25, add_36, add_57, add_68, add_84], 1);  add_25 = add_36 = add_57 = add_68 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    avg_pool2d: "f32[8, 96, 83, 83]" = torch.ops.aten.avg_pool2d.default(relu, [1, 1], [2, 2], [0, 0], False, False)
    convolution_28: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(avg_pool2d, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:97, code: x_path2 = self.path_2(x)
    constant_pad_nd_9: "f32[8, 96, 165, 165]" = torch.ops.aten.constant_pad_nd.default(relu, [-1, 1, -1, 1], 0.0)
    avg_pool2d_1: "f32[8, 96, 83, 83]" = torch.ops.aten.avg_pool2d.default(constant_pad_nd_9, [1, 1], [2, 2], [0, 0], False, False)
    convolution_29: "f32[8, 54, 83, 83]" = torch.ops.aten.convolution.default(avg_pool2d_1, primals_80, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    cat_1: "f32[8, 108, 83, 83]" = torch.ops.aten.cat.default([convolution_28, convolution_29], 1);  convolution_28 = convolution_29 = None
    add_85: "i64[]" = torch.ops.aten.add.Tensor(primals_828, 1)
    var_mean_16 = torch.ops.aten.var_mean.correction(cat_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 108, 1, 1]" = var_mean_16[0]
    getitem_39: "f32[1, 108, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_86: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 0.001)
    rsqrt_16: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_16: "f32[8, 108, 83, 83]" = torch.ops.aten.sub.Tensor(cat_1, getitem_39)
    mul_112: "f32[8, 108, 83, 83]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_49: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_113: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_114: "f32[108]" = torch.ops.aten.mul.Tensor(primals_826, 0.9)
    add_87: "f32[108]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_50: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_115: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000181451978734);  squeeze_50 = None
    mul_116: "f32[108]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[108]" = torch.ops.aten.mul.Tensor(primals_827, 0.9)
    add_88: "f32[108]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_64: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_65: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_118: "f32[8, 108, 83, 83]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
    unsqueeze_66: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_67: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_89: "f32[8, 108, 83, 83]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu_15: "f32[8, 270, 83, 83]" = torch.ops.aten.relu.default(cat);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_30: "f32[8, 108, 83, 83]" = torch.ops.aten.convolution.default(relu_15, primals_83, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_90: "i64[]" = torch.ops.aten.add.Tensor(primals_831, 1)
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 108, 1, 1]" = var_mean_17[0]
    getitem_41: "f32[1, 108, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_91: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 0.001)
    rsqrt_17: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_17: "f32[8, 108, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_41)
    mul_119: "f32[8, 108, 83, 83]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_52: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_120: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_121: "f32[108]" = torch.ops.aten.mul.Tensor(primals_829, 0.9)
    add_92: "f32[108]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_53: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_122: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000181451978734);  squeeze_53 = None
    mul_123: "f32[108]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[108]" = torch.ops.aten.mul.Tensor(primals_830, 0.9)
    add_93: "f32[108]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_68: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
    unsqueeze_69: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_125: "f32[8, 108, 83, 83]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
    unsqueeze_70: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
    unsqueeze_71: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_94: "f32[8, 108, 83, 83]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_16: "f32[8, 108, 83, 83]" = torch.ops.aten.relu.default(add_89)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_10: "f32[8, 108, 87, 87]" = torch.ops.aten.constant_pad_nd.default(relu_16, [2, 2, 2, 2], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_31: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(constant_pad_nd_10, primals_9, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_32: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(convolution_31, primals_86, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_95: "i64[]" = torch.ops.aten.add.Tensor(primals_834, 1)
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 108, 1, 1]" = var_mean_18[0]
    getitem_43: "f32[1, 108, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_96: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 0.001)
    rsqrt_18: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_18: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_43)
    mul_126: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_55: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_127: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_128: "f32[108]" = torch.ops.aten.mul.Tensor(primals_832, 0.9)
    add_97: "f32[108]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_56: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_129: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0000708666997378);  squeeze_56 = None
    mul_130: "f32[108]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[108]" = torch.ops.aten.mul.Tensor(primals_833, 0.9)
    add_98: "f32[108]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_72: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_73: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_132: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
    unsqueeze_74: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_75: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_99: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_17: "f32[8, 108, 42, 42]" = torch.ops.aten.relu.default(add_99);  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_33: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(relu_17, primals_89, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_34: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(convolution_33, primals_90, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_100: "i64[]" = torch.ops.aten.add.Tensor(primals_837, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 108, 1, 1]" = var_mean_19[0]
    getitem_45: "f32[1, 108, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_101: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 0.001)
    rsqrt_19: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_19: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_45)
    mul_133: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_58: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_134: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_135: "f32[108]" = torch.ops.aten.mul.Tensor(primals_835, 0.9)
    add_102: "f32[108]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_59: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_136: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0000708666997378);  squeeze_59 = None
    mul_137: "f32[108]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[108]" = torch.ops.aten.mul.Tensor(primals_836, 0.9)
    add_103: "f32[108]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_76: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_77: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_139: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
    unsqueeze_78: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_79: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_104: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_11: "f32[8, 108, 85, 85]" = torch.ops.aten.constant_pad_nd.default(add_89, [1, 1, 1, 1], -inf);  add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_3 = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd_11, [3, 3], [2, 2])
    getitem_46: "f32[8, 108, 42, 42]" = max_pool2d_with_indices_3[0]
    getitem_47: "i64[8, 108, 42, 42]" = max_pool2d_with_indices_3[1];  max_pool2d_with_indices_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_105: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(add_104, getitem_46);  add_104 = getitem_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_18: "f32[8, 108, 83, 83]" = torch.ops.aten.relu.default(add_94)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_12: "f32[8, 108, 89, 89]" = torch.ops.aten.constant_pad_nd.default(relu_18, [3, 3, 3, 3], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_35: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(constant_pad_nd_12, primals_10, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_36: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(convolution_35, primals_93, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_106: "i64[]" = torch.ops.aten.add.Tensor(primals_840, 1)
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 108, 1, 1]" = var_mean_20[0]
    getitem_49: "f32[1, 108, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_107: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 0.001)
    rsqrt_20: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
    sub_20: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_49)
    mul_140: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_61: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_141: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_142: "f32[108]" = torch.ops.aten.mul.Tensor(primals_838, 0.9)
    add_108: "f32[108]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_62: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_143: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0000708666997378);  squeeze_62 = None
    mul_144: "f32[108]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[108]" = torch.ops.aten.mul.Tensor(primals_839, 0.9)
    add_109: "f32[108]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    unsqueeze_80: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1)
    unsqueeze_81: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_146: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
    unsqueeze_82: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1);  primals_95 = None
    unsqueeze_83: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_110: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_19: "f32[8, 108, 42, 42]" = torch.ops.aten.relu.default(add_110);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_37: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(relu_19, primals_96, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_38: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(convolution_37, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_111: "i64[]" = torch.ops.aten.add.Tensor(primals_843, 1)
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 108, 1, 1]" = var_mean_21[0]
    getitem_51: "f32[1, 108, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_112: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 0.001)
    rsqrt_21: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_21: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_51)
    mul_147: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_64: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_148: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_149: "f32[108]" = torch.ops.aten.mul.Tensor(primals_841, 0.9)
    add_113: "f32[108]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_65: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_150: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0000708666997378);  squeeze_65 = None
    mul_151: "f32[108]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[108]" = torch.ops.aten.mul.Tensor(primals_842, 0.9)
    add_114: "f32[108]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_84: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_85: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_153: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
    unsqueeze_86: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_87: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_115: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_13: "f32[8, 108, 85, 85]" = torch.ops.aten.constant_pad_nd.default(add_94, [1, 1, 1, 1], -inf);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_4 = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd_13, [3, 3], [2, 2])
    getitem_52: "f32[8, 108, 42, 42]" = max_pool2d_with_indices_4[0]
    getitem_53: "i64[8, 108, 42, 42]" = max_pool2d_with_indices_4[1];  max_pool2d_with_indices_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_116: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(add_115, getitem_52);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_14: "f32[8, 108, 87, 87]" = torch.ops.aten.constant_pad_nd.default(relu_18, [2, 2, 2, 2], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_39: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(constant_pad_nd_14, primals_11, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_40: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(convolution_39, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_117: "i64[]" = torch.ops.aten.add.Tensor(primals_846, 1)
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 108, 1, 1]" = var_mean_22[0]
    getitem_55: "f32[1, 108, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_118: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 0.001)
    rsqrt_22: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_22: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_55)
    mul_154: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_67: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_155: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_156: "f32[108]" = torch.ops.aten.mul.Tensor(primals_844, 0.9)
    add_119: "f32[108]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    squeeze_68: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_157: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0000708666997378);  squeeze_68 = None
    mul_158: "f32[108]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
    mul_159: "f32[108]" = torch.ops.aten.mul.Tensor(primals_845, 0.9)
    add_120: "f32[108]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    unsqueeze_88: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_89: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_160: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
    unsqueeze_90: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_91: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_121: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_21: "f32[8, 108, 42, 42]" = torch.ops.aten.relu.default(add_121);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_41: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(relu_21, primals_103, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_42: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(convolution_41, primals_104, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_122: "i64[]" = torch.ops.aten.add.Tensor(primals_849, 1)
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 108, 1, 1]" = var_mean_23[0]
    getitem_57: "f32[1, 108, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_123: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 0.001)
    rsqrt_23: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_23: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_57)
    mul_161: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_70: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_162: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_163: "f32[108]" = torch.ops.aten.mul.Tensor(primals_847, 0.9)
    add_124: "f32[108]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    squeeze_71: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_164: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0000708666997378);  squeeze_71 = None
    mul_165: "f32[108]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
    mul_166: "f32[108]" = torch.ops.aten.mul.Tensor(primals_848, 0.9)
    add_125: "f32[108]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    unsqueeze_92: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1)
    unsqueeze_93: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_167: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
    unsqueeze_94: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1);  primals_106 = None
    unsqueeze_95: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_126: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_15: "f32[8, 108, 85, 85]" = torch.ops.aten.constant_pad_nd.default(relu_18, [1, 1, 1, 1], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_43: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(constant_pad_nd_15, primals_12, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_44: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(convolution_43, primals_107, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_127: "i64[]" = torch.ops.aten.add.Tensor(primals_852, 1)
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 108, 1, 1]" = var_mean_24[0]
    getitem_59: "f32[1, 108, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_128: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 0.001)
    rsqrt_24: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    sub_24: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_59)
    mul_168: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_73: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_169: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_170: "f32[108]" = torch.ops.aten.mul.Tensor(primals_850, 0.9)
    add_129: "f32[108]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_74: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_171: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0000708666997378);  squeeze_74 = None
    mul_172: "f32[108]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[108]" = torch.ops.aten.mul.Tensor(primals_851, 0.9)
    add_130: "f32[108]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_96: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1)
    unsqueeze_97: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_174: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
    unsqueeze_98: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
    unsqueeze_99: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_131: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_23: "f32[8, 108, 42, 42]" = torch.ops.aten.relu.default(add_131);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_45: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(relu_23, primals_110, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_46: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(convolution_45, primals_111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_132: "i64[]" = torch.ops.aten.add.Tensor(primals_855, 1)
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 108, 1, 1]" = var_mean_25[0]
    getitem_61: "f32[1, 108, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_133: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 0.001)
    rsqrt_25: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_25: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_61)
    mul_175: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_76: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_176: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_177: "f32[108]" = torch.ops.aten.mul.Tensor(primals_853, 0.9)
    add_134: "f32[108]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_77: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_178: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0000708666997378);  squeeze_77 = None
    mul_179: "f32[108]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[108]" = torch.ops.aten.mul.Tensor(primals_854, 0.9)
    add_135: "f32[108]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_100: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1)
    unsqueeze_101: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_181: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
    unsqueeze_102: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1);  primals_113 = None
    unsqueeze_103: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_136: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_137: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(add_126, add_136);  add_126 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_24: "f32[8, 108, 42, 42]" = torch.ops.aten.relu.default(add_137)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_47: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(relu_24, primals_114, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_48: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(convolution_47, primals_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_138: "i64[]" = torch.ops.aten.add.Tensor(primals_858, 1)
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 108, 1, 1]" = var_mean_26[0]
    getitem_63: "f32[1, 108, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_139: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 0.001)
    rsqrt_26: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    sub_26: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_63)
    mul_182: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_79: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_183: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_184: "f32[108]" = torch.ops.aten.mul.Tensor(primals_856, 0.9)
    add_140: "f32[108]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_80: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_185: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0000708666997378);  squeeze_80 = None
    mul_186: "f32[108]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[108]" = torch.ops.aten.mul.Tensor(primals_857, 0.9)
    add_141: "f32[108]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_104: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
    unsqueeze_105: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_188: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
    unsqueeze_106: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
    unsqueeze_107: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_142: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_25: "f32[8, 108, 42, 42]" = torch.ops.aten.relu.default(add_142);  add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_49: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(relu_25, primals_118, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_50: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(convolution_49, primals_119, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_143: "i64[]" = torch.ops.aten.add.Tensor(primals_861, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 108, 1, 1]" = var_mean_27[0]
    getitem_65: "f32[1, 108, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_144: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 0.001)
    rsqrt_27: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_27: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_65)
    mul_189: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_82: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_190: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_191: "f32[108]" = torch.ops.aten.mul.Tensor(primals_859, 0.9)
    add_145: "f32[108]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    squeeze_83: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_192: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0000708666997378);  squeeze_83 = None
    mul_193: "f32[108]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
    mul_194: "f32[108]" = torch.ops.aten.mul.Tensor(primals_860, 0.9)
    add_146: "f32[108]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    unsqueeze_108: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1)
    unsqueeze_109: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_195: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
    unsqueeze_110: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
    unsqueeze_111: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_147: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_148: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(add_147, getitem_52);  add_147 = getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_17: "f32[8, 108, 85, 85]" = torch.ops.aten.constant_pad_nd.default(relu_16, [1, 1, 1, 1], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_51: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(constant_pad_nd_17, primals_13, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_52: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(convolution_51, primals_122, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_149: "i64[]" = torch.ops.aten.add.Tensor(primals_864, 1)
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 108, 1, 1]" = var_mean_28[0]
    getitem_69: "f32[1, 108, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_150: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 0.001)
    rsqrt_28: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_28: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_69)
    mul_196: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_85: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_197: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_198: "f32[108]" = torch.ops.aten.mul.Tensor(primals_862, 0.9)
    add_151: "f32[108]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_86: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_199: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0000708666997378);  squeeze_86 = None
    mul_200: "f32[108]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[108]" = torch.ops.aten.mul.Tensor(primals_863, 0.9)
    add_152: "f32[108]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_112: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1)
    unsqueeze_113: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_202: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
    unsqueeze_114: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1);  primals_124 = None
    unsqueeze_115: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_153: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_27: "f32[8, 108, 42, 42]" = torch.ops.aten.relu.default(add_153);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_53: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(relu_27, primals_125, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_54: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(convolution_53, primals_126, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_154: "i64[]" = torch.ops.aten.add.Tensor(primals_867, 1)
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 108, 1, 1]" = var_mean_29[0]
    getitem_71: "f32[1, 108, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_155: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 0.001)
    rsqrt_29: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_29: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_71)
    mul_203: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_88: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_204: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_205: "f32[108]" = torch.ops.aten.mul.Tensor(primals_865, 0.9)
    add_156: "f32[108]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_89: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_206: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0000708666997378);  squeeze_89 = None
    mul_207: "f32[108]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[108]" = torch.ops.aten.mul.Tensor(primals_866, 0.9)
    add_157: "f32[108]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_116: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1)
    unsqueeze_117: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_209: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
    unsqueeze_118: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1);  primals_128 = None
    unsqueeze_119: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_158: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_55: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(relu_18, primals_14, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_159: "i64[]" = torch.ops.aten.add.Tensor(primals_870, 1)
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 108, 1, 1]" = var_mean_30[0]
    getitem_73: "f32[1, 108, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_160: "f32[1, 108, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 0.001)
    rsqrt_30: "f32[1, 108, 1, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    sub_30: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_73)
    mul_210: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_91: "f32[108]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_211: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_212: "f32[108]" = torch.ops.aten.mul.Tensor(primals_868, 0.9)
    add_161: "f32[108]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    squeeze_92: "f32[108]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_213: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0000708666997378);  squeeze_92 = None
    mul_214: "f32[108]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
    mul_215: "f32[108]" = torch.ops.aten.mul.Tensor(primals_869, 0.9)
    add_162: "f32[108]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    unsqueeze_120: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1)
    unsqueeze_121: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_216: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
    unsqueeze_122: "f32[108, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1);  primals_130 = None
    unsqueeze_123: "f32[108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_163: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_164: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(add_158, add_163);  add_158 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat_2: "f32[8, 540, 42, 42]" = torch.ops.aten.cat.default([add_105, add_116, add_137, add_148, add_164], 1);  add_105 = add_116 = add_137 = add_148 = add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    avg_pool2d_2: "f32[8, 270, 42, 42]" = torch.ops.aten.avg_pool2d.default(relu_15, [1, 1], [2, 2], [0, 0], False, False)
    convolution_56: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(avg_pool2d_2, primals_131, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:97, code: x_path2 = self.path_2(x)
    constant_pad_nd_19: "f32[8, 270, 83, 83]" = torch.ops.aten.constant_pad_nd.default(relu_15, [-1, 1, -1, 1], 0.0)
    avg_pool2d_3: "f32[8, 270, 42, 42]" = torch.ops.aten.avg_pool2d.default(constant_pad_nd_19, [1, 1], [2, 2], [0, 0], False, False)
    convolution_57: "f32[8, 108, 42, 42]" = torch.ops.aten.convolution.default(avg_pool2d_3, primals_132, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    cat_3: "f32[8, 216, 42, 42]" = torch.ops.aten.cat.default([convolution_56, convolution_57], 1);  convolution_56 = convolution_57 = None
    add_165: "i64[]" = torch.ops.aten.add.Tensor(primals_873, 1)
    var_mean_31 = torch.ops.aten.var_mean.correction(cat_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 216, 1, 1]" = var_mean_31[0]
    getitem_75: "f32[1, 216, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_166: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 0.001)
    rsqrt_31: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    sub_31: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(cat_3, getitem_75)
    mul_217: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_94: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_218: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_219: "f32[216]" = torch.ops.aten.mul.Tensor(primals_871, 0.9)
    add_167: "f32[216]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_95: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_220: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0000708666997378);  squeeze_95 = None
    mul_221: "f32[216]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[216]" = torch.ops.aten.mul.Tensor(primals_872, 0.9)
    add_168: "f32[216]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    unsqueeze_124: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_133, -1)
    unsqueeze_125: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_223: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
    unsqueeze_126: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1);  primals_134 = None
    unsqueeze_127: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_169: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu_30: "f32[8, 540, 42, 42]" = torch.ops.aten.relu.default(cat_2);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_58: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_30, primals_135, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_170: "i64[]" = torch.ops.aten.add.Tensor(primals_876, 1)
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 216, 1, 1]" = var_mean_32[0]
    getitem_77: "f32[1, 216, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_171: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 0.001)
    rsqrt_32: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_32: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_77)
    mul_224: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_97: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_225: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_226: "f32[216]" = torch.ops.aten.mul.Tensor(primals_874, 0.9)
    add_172: "f32[216]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_98: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_227: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0000708666997378);  squeeze_98 = None
    mul_228: "f32[216]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[216]" = torch.ops.aten.mul.Tensor(primals_875, 0.9)
    add_173: "f32[216]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_128: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_136, -1)
    unsqueeze_129: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_230: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
    unsqueeze_130: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1);  primals_137 = None
    unsqueeze_131: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_174: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_31: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_169)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_59: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_31, primals_138, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_60: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_59, primals_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_175: "i64[]" = torch.ops.aten.add.Tensor(primals_879, 1)
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 216, 1, 1]" = var_mean_33[0]
    getitem_79: "f32[1, 216, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_176: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 0.001)
    rsqrt_33: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_33: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_79)
    mul_231: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_100: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_232: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_233: "f32[216]" = torch.ops.aten.mul.Tensor(primals_877, 0.9)
    add_177: "f32[216]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_101: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_234: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0000708666997378);  squeeze_101 = None
    mul_235: "f32[216]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[216]" = torch.ops.aten.mul.Tensor(primals_878, 0.9)
    add_178: "f32[216]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_132: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
    unsqueeze_133: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_237: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
    unsqueeze_134: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
    unsqueeze_135: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_179: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_32: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_179);  add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_61: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_32, primals_142, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_62: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_61, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_882, 1)
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 216, 1, 1]" = var_mean_34[0]
    getitem_81: "f32[1, 216, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_181: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 0.001)
    rsqrt_34: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    sub_34: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_81)
    mul_238: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_103: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_239: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_240: "f32[216]" = torch.ops.aten.mul.Tensor(primals_880, 0.9)
    add_182: "f32[216]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_104: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_241: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0000708666997378);  squeeze_104 = None
    mul_242: "f32[216]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[216]" = torch.ops.aten.mul.Tensor(primals_881, 0.9)
    add_183: "f32[216]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_136: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1)
    unsqueeze_137: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_244: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
    unsqueeze_138: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
    unsqueeze_139: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_184: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_6 = torch.ops.aten.max_pool2d_with_indices.default(add_169, [3, 3], [1, 1], [1, 1])
    getitem_82: "f32[8, 216, 42, 42]" = max_pool2d_with_indices_6[0]
    getitem_83: "i64[8, 216, 42, 42]" = max_pool2d_with_indices_6[1];  max_pool2d_with_indices_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_185: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_184, getitem_82);  add_184 = getitem_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_33: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_174)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_63: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_33, primals_146, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_64: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_63, primals_147, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_186: "i64[]" = torch.ops.aten.add.Tensor(primals_885, 1)
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 216, 1, 1]" = var_mean_35[0]
    getitem_85: "f32[1, 216, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_187: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 0.001)
    rsqrt_35: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    sub_35: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_85)
    mul_245: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_106: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_246: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_247: "f32[216]" = torch.ops.aten.mul.Tensor(primals_883, 0.9)
    add_188: "f32[216]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_107: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_248: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0000708666997378);  squeeze_107 = None
    mul_249: "f32[216]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[216]" = torch.ops.aten.mul.Tensor(primals_884, 0.9)
    add_189: "f32[216]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_140: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_148, -1)
    unsqueeze_141: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_251: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
    unsqueeze_142: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1);  primals_149 = None
    unsqueeze_143: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_190: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_34: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_190);  add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_65: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_34, primals_150, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_66: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_65, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_191: "i64[]" = torch.ops.aten.add.Tensor(primals_888, 1)
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 216, 1, 1]" = var_mean_36[0]
    getitem_87: "f32[1, 216, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_192: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 0.001)
    rsqrt_36: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_36: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_87)
    mul_252: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_109: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_253: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_254: "f32[216]" = torch.ops.aten.mul.Tensor(primals_886, 0.9)
    add_193: "f32[216]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    squeeze_110: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_255: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0000708666997378);  squeeze_110 = None
    mul_256: "f32[216]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
    mul_257: "f32[216]" = torch.ops.aten.mul.Tensor(primals_887, 0.9)
    add_194: "f32[216]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    unsqueeze_144: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1)
    unsqueeze_145: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_258: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
    unsqueeze_146: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
    unsqueeze_147: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_195: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_7 = torch.ops.aten.max_pool2d_with_indices.default(add_174, [3, 3], [1, 1], [1, 1])
    getitem_88: "f32[8, 216, 42, 42]" = max_pool2d_with_indices_7[0]
    getitem_89: "i64[8, 216, 42, 42]" = max_pool2d_with_indices_7[1];  max_pool2d_with_indices_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_196: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_195, getitem_88);  add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_67: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_33, primals_154, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_68: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_67, primals_155, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_197: "i64[]" = torch.ops.aten.add.Tensor(primals_891, 1)
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 216, 1, 1]" = var_mean_37[0]
    getitem_91: "f32[1, 216, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_198: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 0.001)
    rsqrt_37: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    sub_37: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_91)
    mul_259: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_112: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_260: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_261: "f32[216]" = torch.ops.aten.mul.Tensor(primals_889, 0.9)
    add_199: "f32[216]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    squeeze_113: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_262: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0000708666997378);  squeeze_113 = None
    mul_263: "f32[216]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
    mul_264: "f32[216]" = torch.ops.aten.mul.Tensor(primals_890, 0.9)
    add_200: "f32[216]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    unsqueeze_148: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1)
    unsqueeze_149: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_265: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
    unsqueeze_150: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
    unsqueeze_151: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_201: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_36: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_201);  add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_69: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_36, primals_158, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_70: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_69, primals_159, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_202: "i64[]" = torch.ops.aten.add.Tensor(primals_894, 1)
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[1, 216, 1, 1]" = var_mean_38[0]
    getitem_93: "f32[1, 216, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_203: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 0.001)
    rsqrt_38: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    sub_38: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_93)
    mul_266: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
    squeeze_115: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_267: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_268: "f32[216]" = torch.ops.aten.mul.Tensor(primals_892, 0.9)
    add_204: "f32[216]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    squeeze_116: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
    mul_269: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0000708666997378);  squeeze_116 = None
    mul_270: "f32[216]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
    mul_271: "f32[216]" = torch.ops.aten.mul.Tensor(primals_893, 0.9)
    add_205: "f32[216]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    unsqueeze_152: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_160, -1)
    unsqueeze_153: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_272: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
    unsqueeze_154: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_161, -1);  primals_161 = None
    unsqueeze_155: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_206: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_71: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_33, primals_162, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_72: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_71, primals_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_207: "i64[]" = torch.ops.aten.add.Tensor(primals_897, 1)
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_72, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 216, 1, 1]" = var_mean_39[0]
    getitem_95: "f32[1, 216, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_208: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 0.001)
    rsqrt_39: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_39: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_72, getitem_95)
    mul_273: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_118: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_274: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_275: "f32[216]" = torch.ops.aten.mul.Tensor(primals_895, 0.9)
    add_209: "f32[216]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    squeeze_119: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_276: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0000708666997378);  squeeze_119 = None
    mul_277: "f32[216]" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
    mul_278: "f32[216]" = torch.ops.aten.mul.Tensor(primals_896, 0.9)
    add_210: "f32[216]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    unsqueeze_156: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_164, -1)
    unsqueeze_157: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_279: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
    unsqueeze_158: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_165, -1);  primals_165 = None
    unsqueeze_159: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_211: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_38: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_211);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_73: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_38, primals_166, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_74: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_73, primals_167, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_900, 1)
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_74, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 216, 1, 1]" = var_mean_40[0]
    getitem_97: "f32[1, 216, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_213: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 0.001)
    rsqrt_40: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_40: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_74, getitem_97)
    mul_280: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_121: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_281: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_282: "f32[216]" = torch.ops.aten.mul.Tensor(primals_898, 0.9)
    add_214: "f32[216]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    squeeze_122: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_283: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0000708666997378);  squeeze_122 = None
    mul_284: "f32[216]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
    mul_285: "f32[216]" = torch.ops.aten.mul.Tensor(primals_899, 0.9)
    add_215: "f32[216]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_160: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1)
    unsqueeze_161: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_286: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
    unsqueeze_162: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_169, -1);  primals_169 = None
    unsqueeze_163: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_216: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_217: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_206, add_216);  add_206 = add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_39: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_217)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_75: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_39, primals_170, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_76: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_75, primals_171, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_218: "i64[]" = torch.ops.aten.add.Tensor(primals_903, 1)
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 216, 1, 1]" = var_mean_41[0]
    getitem_99: "f32[1, 216, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_219: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 0.001)
    rsqrt_41: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
    sub_41: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_99)
    mul_287: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_124: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_288: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_289: "f32[216]" = torch.ops.aten.mul.Tensor(primals_901, 0.9)
    add_220: "f32[216]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    squeeze_125: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_290: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0000708666997378);  squeeze_125 = None
    mul_291: "f32[216]" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
    mul_292: "f32[216]" = torch.ops.aten.mul.Tensor(primals_902, 0.9)
    add_221: "f32[216]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    unsqueeze_164: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_172, -1)
    unsqueeze_165: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_293: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
    unsqueeze_166: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_173, -1);  primals_173 = None
    unsqueeze_167: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_222: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_40: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_222);  add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_77: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_40, primals_174, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_78: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_77, primals_175, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_223: "i64[]" = torch.ops.aten.add.Tensor(primals_906, 1)
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 216, 1, 1]" = var_mean_42[0]
    getitem_101: "f32[1, 216, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_224: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 0.001)
    rsqrt_42: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    sub_42: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_101)
    mul_294: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_127: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_295: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_296: "f32[216]" = torch.ops.aten.mul.Tensor(primals_904, 0.9)
    add_225: "f32[216]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_128: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_297: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0000708666997378);  squeeze_128 = None
    mul_298: "f32[216]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[216]" = torch.ops.aten.mul.Tensor(primals_905, 0.9)
    add_226: "f32[216]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_168: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_176, -1)
    unsqueeze_169: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_300: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
    unsqueeze_170: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_177, -1);  primals_177 = None
    unsqueeze_171: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_227: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_228: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_227, getitem_88);  add_227 = getitem_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_79: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_31, primals_178, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_80: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_79, primals_179, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_229: "i64[]" = torch.ops.aten.add.Tensor(primals_909, 1)
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_80, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 216, 1, 1]" = var_mean_43[0]
    getitem_105: "f32[1, 216, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_230: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 0.001)
    rsqrt_43: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_230);  add_230 = None
    sub_43: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_105)
    mul_301: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_130: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_302: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_303: "f32[216]" = torch.ops.aten.mul.Tensor(primals_907, 0.9)
    add_231: "f32[216]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    squeeze_131: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_304: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0000708666997378);  squeeze_131 = None
    mul_305: "f32[216]" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
    mul_306: "f32[216]" = torch.ops.aten.mul.Tensor(primals_908, 0.9)
    add_232: "f32[216]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_172: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_180, -1)
    unsqueeze_173: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_307: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
    unsqueeze_174: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_181, -1);  primals_181 = None
    unsqueeze_175: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_233: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_42: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_233);  add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_81: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_42, primals_182, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_82: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_81, primals_183, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_234: "i64[]" = torch.ops.aten.add.Tensor(primals_912, 1)
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_82, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 216, 1, 1]" = var_mean_44[0]
    getitem_107: "f32[1, 216, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_235: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 0.001)
    rsqrt_44: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    sub_44: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_82, getitem_107)
    mul_308: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_133: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_309: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_310: "f32[216]" = torch.ops.aten.mul.Tensor(primals_910, 0.9)
    add_236: "f32[216]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    squeeze_134: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_311: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0000708666997378);  squeeze_134 = None
    mul_312: "f32[216]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
    mul_313: "f32[216]" = torch.ops.aten.mul.Tensor(primals_911, 0.9)
    add_237: "f32[216]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_176: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_184, -1)
    unsqueeze_177: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_314: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
    unsqueeze_178: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_185, -1);  primals_185 = None
    unsqueeze_179: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_238: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_239: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_238, add_174);  add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat_4: "f32[8, 1080, 42, 42]" = torch.ops.aten.cat.default([add_185, add_196, add_217, add_228, add_239], 1);  add_185 = add_196 = add_217 = add_228 = add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_83: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_30, primals_186, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_240: "i64[]" = torch.ops.aten.add.Tensor(primals_915, 1)
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 216, 1, 1]" = var_mean_45[0]
    getitem_109: "f32[1, 216, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_241: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 0.001)
    rsqrt_45: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
    sub_45: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_109)
    mul_315: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_136: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_316: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_317: "f32[216]" = torch.ops.aten.mul.Tensor(primals_913, 0.9)
    add_242: "f32[216]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    squeeze_137: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_318: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0000708666997378);  squeeze_137 = None
    mul_319: "f32[216]" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
    mul_320: "f32[216]" = torch.ops.aten.mul.Tensor(primals_914, 0.9)
    add_243: "f32[216]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    unsqueeze_180: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_187, -1)
    unsqueeze_181: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_321: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
    unsqueeze_182: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_188, -1);  primals_188 = None
    unsqueeze_183: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_244: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu_44: "f32[8, 1080, 42, 42]" = torch.ops.aten.relu.default(cat_4);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_84: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_44, primals_189, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_245: "i64[]" = torch.ops.aten.add.Tensor(primals_918, 1)
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 216, 1, 1]" = var_mean_46[0]
    getitem_111: "f32[1, 216, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_246: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 0.001)
    rsqrt_46: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
    sub_46: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_111)
    mul_322: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_139: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_323: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_324: "f32[216]" = torch.ops.aten.mul.Tensor(primals_916, 0.9)
    add_247: "f32[216]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    squeeze_140: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_325: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0000708666997378);  squeeze_140 = None
    mul_326: "f32[216]" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
    mul_327: "f32[216]" = torch.ops.aten.mul.Tensor(primals_917, 0.9)
    add_248: "f32[216]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    unsqueeze_184: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_190, -1)
    unsqueeze_185: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_328: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
    unsqueeze_186: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_191, -1);  primals_191 = None
    unsqueeze_187: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_249: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_45: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_244)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_85: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_45, primals_192, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_86: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_85, primals_193, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_250: "i64[]" = torch.ops.aten.add.Tensor(primals_921, 1)
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_86, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 216, 1, 1]" = var_mean_47[0]
    getitem_113: "f32[1, 216, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_251: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 0.001)
    rsqrt_47: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
    sub_47: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_86, getitem_113)
    mul_329: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_142: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_330: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_331: "f32[216]" = torch.ops.aten.mul.Tensor(primals_919, 0.9)
    add_252: "f32[216]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    squeeze_143: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_332: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0000708666997378);  squeeze_143 = None
    mul_333: "f32[216]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
    mul_334: "f32[216]" = torch.ops.aten.mul.Tensor(primals_920, 0.9)
    add_253: "f32[216]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    unsqueeze_188: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_194, -1)
    unsqueeze_189: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_335: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
    unsqueeze_190: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_195, -1);  primals_195 = None
    unsqueeze_191: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_254: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_46: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_254);  add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_87: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_46, primals_196, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_88: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_87, primals_197, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_255: "i64[]" = torch.ops.aten.add.Tensor(primals_924, 1)
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 216, 1, 1]" = var_mean_48[0]
    getitem_115: "f32[1, 216, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_256: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 0.001)
    rsqrt_48: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
    sub_48: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_115)
    mul_336: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_145: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_337: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_338: "f32[216]" = torch.ops.aten.mul.Tensor(primals_922, 0.9)
    add_257: "f32[216]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    squeeze_146: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_339: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0000708666997378);  squeeze_146 = None
    mul_340: "f32[216]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
    mul_341: "f32[216]" = torch.ops.aten.mul.Tensor(primals_923, 0.9)
    add_258: "f32[216]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    unsqueeze_192: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_198, -1)
    unsqueeze_193: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_342: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
    unsqueeze_194: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_199, -1);  primals_199 = None
    unsqueeze_195: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_259: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_9 = torch.ops.aten.max_pool2d_with_indices.default(add_244, [3, 3], [1, 1], [1, 1])
    getitem_116: "f32[8, 216, 42, 42]" = max_pool2d_with_indices_9[0]
    getitem_117: "i64[8, 216, 42, 42]" = max_pool2d_with_indices_9[1];  max_pool2d_with_indices_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_260: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_259, getitem_116);  add_259 = getitem_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_47: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_249)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_89: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_47, primals_200, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_90: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_89, primals_201, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_261: "i64[]" = torch.ops.aten.add.Tensor(primals_927, 1)
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_90, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 216, 1, 1]" = var_mean_49[0]
    getitem_119: "f32[1, 216, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_262: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 0.001)
    rsqrt_49: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
    sub_49: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_90, getitem_119)
    mul_343: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_148: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_344: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_345: "f32[216]" = torch.ops.aten.mul.Tensor(primals_925, 0.9)
    add_263: "f32[216]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    squeeze_149: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_346: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0000708666997378);  squeeze_149 = None
    mul_347: "f32[216]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
    mul_348: "f32[216]" = torch.ops.aten.mul.Tensor(primals_926, 0.9)
    add_264: "f32[216]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    unsqueeze_196: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_202, -1)
    unsqueeze_197: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_349: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
    unsqueeze_198: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1);  primals_203 = None
    unsqueeze_199: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_265: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_48: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_265);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_91: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_48, primals_204, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_92: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_91, primals_205, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_266: "i64[]" = torch.ops.aten.add.Tensor(primals_930, 1)
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_92, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 216, 1, 1]" = var_mean_50[0]
    getitem_121: "f32[1, 216, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_267: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 0.001)
    rsqrt_50: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
    sub_50: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_121)
    mul_350: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_151: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_351: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_352: "f32[216]" = torch.ops.aten.mul.Tensor(primals_928, 0.9)
    add_268: "f32[216]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    squeeze_152: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_353: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0000708666997378);  squeeze_152 = None
    mul_354: "f32[216]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
    mul_355: "f32[216]" = torch.ops.aten.mul.Tensor(primals_929, 0.9)
    add_269: "f32[216]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_200: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_206, -1)
    unsqueeze_201: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_356: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
    unsqueeze_202: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_207, -1);  primals_207 = None
    unsqueeze_203: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_270: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_10 = torch.ops.aten.max_pool2d_with_indices.default(add_249, [3, 3], [1, 1], [1, 1])
    getitem_122: "f32[8, 216, 42, 42]" = max_pool2d_with_indices_10[0]
    getitem_123: "i64[8, 216, 42, 42]" = max_pool2d_with_indices_10[1];  max_pool2d_with_indices_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_271: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_270, getitem_122);  add_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_93: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_47, primals_208, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_94: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_93, primals_209, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_272: "i64[]" = torch.ops.aten.add.Tensor(primals_933, 1)
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_94, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 216, 1, 1]" = var_mean_51[0]
    getitem_125: "f32[1, 216, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_273: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 0.001)
    rsqrt_51: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_273);  add_273 = None
    sub_51: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_94, getitem_125)
    mul_357: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_154: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_358: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_359: "f32[216]" = torch.ops.aten.mul.Tensor(primals_931, 0.9)
    add_274: "f32[216]" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    squeeze_155: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_360: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0000708666997378);  squeeze_155 = None
    mul_361: "f32[216]" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
    mul_362: "f32[216]" = torch.ops.aten.mul.Tensor(primals_932, 0.9)
    add_275: "f32[216]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    unsqueeze_204: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_210, -1)
    unsqueeze_205: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_363: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
    unsqueeze_206: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_211, -1);  primals_211 = None
    unsqueeze_207: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_276: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_50: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_276);  add_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_95: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_50, primals_212, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_96: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_95, primals_213, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_277: "i64[]" = torch.ops.aten.add.Tensor(primals_936, 1)
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_96, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 216, 1, 1]" = var_mean_52[0]
    getitem_127: "f32[1, 216, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_278: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 0.001)
    rsqrt_52: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
    sub_52: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_96, getitem_127)
    mul_364: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_157: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_365: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_366: "f32[216]" = torch.ops.aten.mul.Tensor(primals_934, 0.9)
    add_279: "f32[216]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    squeeze_158: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_367: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0000708666997378);  squeeze_158 = None
    mul_368: "f32[216]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
    mul_369: "f32[216]" = torch.ops.aten.mul.Tensor(primals_935, 0.9)
    add_280: "f32[216]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    unsqueeze_208: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_214, -1)
    unsqueeze_209: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_370: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
    unsqueeze_210: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_215, -1);  primals_215 = None
    unsqueeze_211: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_281: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_97: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_47, primals_216, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_98: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_97, primals_217, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_282: "i64[]" = torch.ops.aten.add.Tensor(primals_939, 1)
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_98, [0, 2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[1, 216, 1, 1]" = var_mean_53[0]
    getitem_129: "f32[1, 216, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_283: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 0.001)
    rsqrt_53: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
    sub_53: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_98, getitem_129)
    mul_371: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
    squeeze_160: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_372: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_373: "f32[216]" = torch.ops.aten.mul.Tensor(primals_937, 0.9)
    add_284: "f32[216]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    squeeze_161: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
    mul_374: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0000708666997378);  squeeze_161 = None
    mul_375: "f32[216]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
    mul_376: "f32[216]" = torch.ops.aten.mul.Tensor(primals_938, 0.9)
    add_285: "f32[216]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_212: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_218, -1)
    unsqueeze_213: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_377: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
    unsqueeze_214: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_219, -1);  primals_219 = None
    unsqueeze_215: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_286: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_52: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_286);  add_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_99: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_52, primals_220, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_100: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_99, primals_221, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_287: "i64[]" = torch.ops.aten.add.Tensor(primals_942, 1)
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_100, [0, 2, 3], correction = 0, keepdim = True)
    getitem_130: "f32[1, 216, 1, 1]" = var_mean_54[0]
    getitem_131: "f32[1, 216, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_288: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 0.001)
    rsqrt_54: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_288);  add_288 = None
    sub_54: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_100, getitem_131)
    mul_378: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
    squeeze_163: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_379: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_380: "f32[216]" = torch.ops.aten.mul.Tensor(primals_940, 0.9)
    add_289: "f32[216]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    squeeze_164: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
    mul_381: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0000708666997378);  squeeze_164 = None
    mul_382: "f32[216]" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
    mul_383: "f32[216]" = torch.ops.aten.mul.Tensor(primals_941, 0.9)
    add_290: "f32[216]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    unsqueeze_216: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_222, -1)
    unsqueeze_217: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_384: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
    unsqueeze_218: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_223, -1);  primals_223 = None
    unsqueeze_219: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_291: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_292: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_281, add_291);  add_281 = add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_53: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_292)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_101: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_53, primals_224, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_102: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_101, primals_225, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_293: "i64[]" = torch.ops.aten.add.Tensor(primals_945, 1)
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_102, [0, 2, 3], correction = 0, keepdim = True)
    getitem_132: "f32[1, 216, 1, 1]" = var_mean_55[0]
    getitem_133: "f32[1, 216, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_294: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 0.001)
    rsqrt_55: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
    sub_55: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_102, getitem_133)
    mul_385: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
    squeeze_166: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_386: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_387: "f32[216]" = torch.ops.aten.mul.Tensor(primals_943, 0.9)
    add_295: "f32[216]" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
    squeeze_167: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
    mul_388: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0000708666997378);  squeeze_167 = None
    mul_389: "f32[216]" = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
    mul_390: "f32[216]" = torch.ops.aten.mul.Tensor(primals_944, 0.9)
    add_296: "f32[216]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    unsqueeze_220: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_226, -1)
    unsqueeze_221: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_391: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
    unsqueeze_222: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_227, -1);  primals_227 = None
    unsqueeze_223: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_297: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_54: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_297);  add_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_103: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_54, primals_228, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_104: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_103, primals_229, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_298: "i64[]" = torch.ops.aten.add.Tensor(primals_948, 1)
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_104, [0, 2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[1, 216, 1, 1]" = var_mean_56[0]
    getitem_135: "f32[1, 216, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_299: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 0.001)
    rsqrt_56: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
    sub_56: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_104, getitem_135)
    mul_392: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
    squeeze_169: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_393: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_394: "f32[216]" = torch.ops.aten.mul.Tensor(primals_946, 0.9)
    add_300: "f32[216]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    squeeze_170: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
    mul_395: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0000708666997378);  squeeze_170 = None
    mul_396: "f32[216]" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
    mul_397: "f32[216]" = torch.ops.aten.mul.Tensor(primals_947, 0.9)
    add_301: "f32[216]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    unsqueeze_224: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_230, -1)
    unsqueeze_225: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_398: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_225);  mul_392 = unsqueeze_225 = None
    unsqueeze_226: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_231, -1);  primals_231 = None
    unsqueeze_227: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_302: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_227);  mul_398 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_303: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_302, getitem_122);  add_302 = getitem_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_105: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_45, primals_232, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_106: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_105, primals_233, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_304: "i64[]" = torch.ops.aten.add.Tensor(primals_951, 1)
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_106, [0, 2, 3], correction = 0, keepdim = True)
    getitem_138: "f32[1, 216, 1, 1]" = var_mean_57[0]
    getitem_139: "f32[1, 216, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_305: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 0.001)
    rsqrt_57: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_305);  add_305 = None
    sub_57: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_106, getitem_139)
    mul_399: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
    squeeze_172: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_400: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_401: "f32[216]" = torch.ops.aten.mul.Tensor(primals_949, 0.9)
    add_306: "f32[216]" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
    squeeze_173: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
    mul_402: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0000708666997378);  squeeze_173 = None
    mul_403: "f32[216]" = torch.ops.aten.mul.Tensor(mul_402, 0.1);  mul_402 = None
    mul_404: "f32[216]" = torch.ops.aten.mul.Tensor(primals_950, 0.9)
    add_307: "f32[216]" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
    unsqueeze_228: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_234, -1)
    unsqueeze_229: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_405: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_229);  mul_399 = unsqueeze_229 = None
    unsqueeze_230: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_235, -1);  primals_235 = None
    unsqueeze_231: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_308: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_405, unsqueeze_231);  mul_405 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_56: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_308);  add_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_107: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_56, primals_236, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_108: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_107, primals_237, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_309: "i64[]" = torch.ops.aten.add.Tensor(primals_954, 1)
    var_mean_58 = torch.ops.aten.var_mean.correction(convolution_108, [0, 2, 3], correction = 0, keepdim = True)
    getitem_140: "f32[1, 216, 1, 1]" = var_mean_58[0]
    getitem_141: "f32[1, 216, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_310: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 0.001)
    rsqrt_58: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
    sub_58: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_108, getitem_141)
    mul_406: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    squeeze_174: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
    squeeze_175: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_407: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_408: "f32[216]" = torch.ops.aten.mul.Tensor(primals_952, 0.9)
    add_311: "f32[216]" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    squeeze_176: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
    mul_409: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0000708666997378);  squeeze_176 = None
    mul_410: "f32[216]" = torch.ops.aten.mul.Tensor(mul_409, 0.1);  mul_409 = None
    mul_411: "f32[216]" = torch.ops.aten.mul.Tensor(primals_953, 0.9)
    add_312: "f32[216]" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    unsqueeze_232: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_238, -1)
    unsqueeze_233: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_412: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_233);  mul_406 = unsqueeze_233 = None
    unsqueeze_234: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_239, -1);  primals_239 = None
    unsqueeze_235: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_313: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_235);  mul_412 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_314: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_313, add_249);  add_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat_5: "f32[8, 1080, 42, 42]" = torch.ops.aten.cat.default([add_260, add_271, add_292, add_303, add_314], 1);  add_260 = add_271 = add_292 = add_303 = add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_109: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_44, primals_240, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_315: "i64[]" = torch.ops.aten.add.Tensor(primals_957, 1)
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_109, [0, 2, 3], correction = 0, keepdim = True)
    getitem_142: "f32[1, 216, 1, 1]" = var_mean_59[0]
    getitem_143: "f32[1, 216, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_316: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 0.001)
    rsqrt_59: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_316);  add_316 = None
    sub_59: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_109, getitem_143)
    mul_413: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    squeeze_177: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
    squeeze_178: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    mul_414: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_415: "f32[216]" = torch.ops.aten.mul.Tensor(primals_955, 0.9)
    add_317: "f32[216]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    squeeze_179: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
    mul_416: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0000708666997378);  squeeze_179 = None
    mul_417: "f32[216]" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
    mul_418: "f32[216]" = torch.ops.aten.mul.Tensor(primals_956, 0.9)
    add_318: "f32[216]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    unsqueeze_236: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_241, -1)
    unsqueeze_237: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_419: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_237);  mul_413 = unsqueeze_237 = None
    unsqueeze_238: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_242, -1);  primals_242 = None
    unsqueeze_239: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_319: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_239);  mul_419 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu_58: "f32[8, 1080, 42, 42]" = torch.ops.aten.relu.default(cat_5);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_110: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_58, primals_243, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_320: "i64[]" = torch.ops.aten.add.Tensor(primals_960, 1)
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_110, [0, 2, 3], correction = 0, keepdim = True)
    getitem_144: "f32[1, 216, 1, 1]" = var_mean_60[0]
    getitem_145: "f32[1, 216, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_321: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 0.001)
    rsqrt_60: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_321);  add_321 = None
    sub_60: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_110, getitem_145)
    mul_420: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    squeeze_180: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
    squeeze_181: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    mul_421: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_422: "f32[216]" = torch.ops.aten.mul.Tensor(primals_958, 0.9)
    add_322: "f32[216]" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    squeeze_182: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
    mul_423: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0000708666997378);  squeeze_182 = None
    mul_424: "f32[216]" = torch.ops.aten.mul.Tensor(mul_423, 0.1);  mul_423 = None
    mul_425: "f32[216]" = torch.ops.aten.mul.Tensor(primals_959, 0.9)
    add_323: "f32[216]" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
    unsqueeze_240: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_244, -1)
    unsqueeze_241: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_426: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_241);  mul_420 = unsqueeze_241 = None
    unsqueeze_242: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_245, -1);  primals_245 = None
    unsqueeze_243: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_324: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_243);  mul_426 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_59: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_319)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_111: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_59, primals_246, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_112: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_111, primals_247, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_325: "i64[]" = torch.ops.aten.add.Tensor(primals_963, 1)
    var_mean_61 = torch.ops.aten.var_mean.correction(convolution_112, [0, 2, 3], correction = 0, keepdim = True)
    getitem_146: "f32[1, 216, 1, 1]" = var_mean_61[0]
    getitem_147: "f32[1, 216, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_326: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 0.001)
    rsqrt_61: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_326);  add_326 = None
    sub_61: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_112, getitem_147)
    mul_427: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    squeeze_183: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
    squeeze_184: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_428: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_429: "f32[216]" = torch.ops.aten.mul.Tensor(primals_961, 0.9)
    add_327: "f32[216]" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    squeeze_185: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
    mul_430: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0000708666997378);  squeeze_185 = None
    mul_431: "f32[216]" = torch.ops.aten.mul.Tensor(mul_430, 0.1);  mul_430 = None
    mul_432: "f32[216]" = torch.ops.aten.mul.Tensor(primals_962, 0.9)
    add_328: "f32[216]" = torch.ops.aten.add.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
    unsqueeze_244: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_248, -1)
    unsqueeze_245: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_433: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_245);  mul_427 = unsqueeze_245 = None
    unsqueeze_246: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_249, -1);  primals_249 = None
    unsqueeze_247: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_329: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_433, unsqueeze_247);  mul_433 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_60: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_329);  add_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_113: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_60, primals_250, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_114: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_113, primals_251, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_330: "i64[]" = torch.ops.aten.add.Tensor(primals_966, 1)
    var_mean_62 = torch.ops.aten.var_mean.correction(convolution_114, [0, 2, 3], correction = 0, keepdim = True)
    getitem_148: "f32[1, 216, 1, 1]" = var_mean_62[0]
    getitem_149: "f32[1, 216, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_331: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_148, 0.001)
    rsqrt_62: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
    sub_62: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_114, getitem_149)
    mul_434: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
    squeeze_186: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
    squeeze_187: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
    mul_435: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_436: "f32[216]" = torch.ops.aten.mul.Tensor(primals_964, 0.9)
    add_332: "f32[216]" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
    squeeze_188: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
    mul_437: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0000708666997378);  squeeze_188 = None
    mul_438: "f32[216]" = torch.ops.aten.mul.Tensor(mul_437, 0.1);  mul_437 = None
    mul_439: "f32[216]" = torch.ops.aten.mul.Tensor(primals_965, 0.9)
    add_333: "f32[216]" = torch.ops.aten.add.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_248: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_252, -1)
    unsqueeze_249: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    mul_440: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_249);  mul_434 = unsqueeze_249 = None
    unsqueeze_250: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_253, -1);  primals_253 = None
    unsqueeze_251: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    add_334: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_251);  mul_440 = unsqueeze_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_12 = torch.ops.aten.max_pool2d_with_indices.default(add_319, [3, 3], [1, 1], [1, 1])
    getitem_150: "f32[8, 216, 42, 42]" = max_pool2d_with_indices_12[0]
    getitem_151: "i64[8, 216, 42, 42]" = max_pool2d_with_indices_12[1];  max_pool2d_with_indices_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_335: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_334, getitem_150);  add_334 = getitem_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_61: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_324)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_115: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_61, primals_254, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_116: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_115, primals_255, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_336: "i64[]" = torch.ops.aten.add.Tensor(primals_969, 1)
    var_mean_63 = torch.ops.aten.var_mean.correction(convolution_116, [0, 2, 3], correction = 0, keepdim = True)
    getitem_152: "f32[1, 216, 1, 1]" = var_mean_63[0]
    getitem_153: "f32[1, 216, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_337: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_152, 0.001)
    rsqrt_63: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_337);  add_337 = None
    sub_63: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_116, getitem_153)
    mul_441: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
    squeeze_189: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
    squeeze_190: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
    mul_442: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_443: "f32[216]" = torch.ops.aten.mul.Tensor(primals_967, 0.9)
    add_338: "f32[216]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    squeeze_191: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
    mul_444: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0000708666997378);  squeeze_191 = None
    mul_445: "f32[216]" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
    mul_446: "f32[216]" = torch.ops.aten.mul.Tensor(primals_968, 0.9)
    add_339: "f32[216]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
    unsqueeze_252: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_256, -1)
    unsqueeze_253: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_447: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_253);  mul_441 = unsqueeze_253 = None
    unsqueeze_254: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_257, -1);  primals_257 = None
    unsqueeze_255: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_340: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_255);  mul_447 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_62: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_340);  add_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_117: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_62, primals_258, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_118: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_117, primals_259, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_341: "i64[]" = torch.ops.aten.add.Tensor(primals_972, 1)
    var_mean_64 = torch.ops.aten.var_mean.correction(convolution_118, [0, 2, 3], correction = 0, keepdim = True)
    getitem_154: "f32[1, 216, 1, 1]" = var_mean_64[0]
    getitem_155: "f32[1, 216, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_342: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 0.001)
    rsqrt_64: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_342);  add_342 = None
    sub_64: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_118, getitem_155)
    mul_448: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
    squeeze_192: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
    squeeze_193: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
    mul_449: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
    mul_450: "f32[216]" = torch.ops.aten.mul.Tensor(primals_970, 0.9)
    add_343: "f32[216]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    squeeze_194: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
    mul_451: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0000708666997378);  squeeze_194 = None
    mul_452: "f32[216]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
    mul_453: "f32[216]" = torch.ops.aten.mul.Tensor(primals_971, 0.9)
    add_344: "f32[216]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    unsqueeze_256: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_260, -1)
    unsqueeze_257: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    mul_454: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_257);  mul_448 = unsqueeze_257 = None
    unsqueeze_258: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_261, -1);  primals_261 = None
    unsqueeze_259: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    add_345: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_259);  mul_454 = unsqueeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_13 = torch.ops.aten.max_pool2d_with_indices.default(add_324, [3, 3], [1, 1], [1, 1])
    getitem_156: "f32[8, 216, 42, 42]" = max_pool2d_with_indices_13[0]
    getitem_157: "i64[8, 216, 42, 42]" = max_pool2d_with_indices_13[1];  max_pool2d_with_indices_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_346: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_345, getitem_156);  add_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_119: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_61, primals_262, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_120: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_119, primals_263, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_347: "i64[]" = torch.ops.aten.add.Tensor(primals_975, 1)
    var_mean_65 = torch.ops.aten.var_mean.correction(convolution_120, [0, 2, 3], correction = 0, keepdim = True)
    getitem_158: "f32[1, 216, 1, 1]" = var_mean_65[0]
    getitem_159: "f32[1, 216, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_348: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 0.001)
    rsqrt_65: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_348);  add_348 = None
    sub_65: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_120, getitem_159)
    mul_455: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
    squeeze_195: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
    squeeze_196: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
    mul_456: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
    mul_457: "f32[216]" = torch.ops.aten.mul.Tensor(primals_973, 0.9)
    add_349: "f32[216]" = torch.ops.aten.add.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    squeeze_197: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
    mul_458: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0000708666997378);  squeeze_197 = None
    mul_459: "f32[216]" = torch.ops.aten.mul.Tensor(mul_458, 0.1);  mul_458 = None
    mul_460: "f32[216]" = torch.ops.aten.mul.Tensor(primals_974, 0.9)
    add_350: "f32[216]" = torch.ops.aten.add.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
    unsqueeze_260: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_264, -1)
    unsqueeze_261: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_461: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_455, unsqueeze_261);  mul_455 = unsqueeze_261 = None
    unsqueeze_262: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_265, -1);  primals_265 = None
    unsqueeze_263: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_351: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_263);  mul_461 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_64: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_351);  add_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_121: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_64, primals_266, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_122: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_121, primals_267, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_352: "i64[]" = torch.ops.aten.add.Tensor(primals_978, 1)
    var_mean_66 = torch.ops.aten.var_mean.correction(convolution_122, [0, 2, 3], correction = 0, keepdim = True)
    getitem_160: "f32[1, 216, 1, 1]" = var_mean_66[0]
    getitem_161: "f32[1, 216, 1, 1]" = var_mean_66[1];  var_mean_66 = None
    add_353: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_160, 0.001)
    rsqrt_66: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_353);  add_353 = None
    sub_66: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_122, getitem_161)
    mul_462: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
    squeeze_198: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3]);  getitem_161 = None
    squeeze_199: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
    mul_463: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
    mul_464: "f32[216]" = torch.ops.aten.mul.Tensor(primals_976, 0.9)
    add_354: "f32[216]" = torch.ops.aten.add.Tensor(mul_463, mul_464);  mul_463 = mul_464 = None
    squeeze_200: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_160, [0, 2, 3]);  getitem_160 = None
    mul_465: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0000708666997378);  squeeze_200 = None
    mul_466: "f32[216]" = torch.ops.aten.mul.Tensor(mul_465, 0.1);  mul_465 = None
    mul_467: "f32[216]" = torch.ops.aten.mul.Tensor(primals_977, 0.9)
    add_355: "f32[216]" = torch.ops.aten.add.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
    unsqueeze_264: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_268, -1)
    unsqueeze_265: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    mul_468: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_265);  mul_462 = unsqueeze_265 = None
    unsqueeze_266: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_269, -1);  primals_269 = None
    unsqueeze_267: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    add_356: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_468, unsqueeze_267);  mul_468 = unsqueeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_123: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_61, primals_270, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_124: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_123, primals_271, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_357: "i64[]" = torch.ops.aten.add.Tensor(primals_981, 1)
    var_mean_67 = torch.ops.aten.var_mean.correction(convolution_124, [0, 2, 3], correction = 0, keepdim = True)
    getitem_162: "f32[1, 216, 1, 1]" = var_mean_67[0]
    getitem_163: "f32[1, 216, 1, 1]" = var_mean_67[1];  var_mean_67 = None
    add_358: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 0.001)
    rsqrt_67: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_358);  add_358 = None
    sub_67: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_124, getitem_163)
    mul_469: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
    squeeze_201: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
    squeeze_202: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
    mul_470: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
    mul_471: "f32[216]" = torch.ops.aten.mul.Tensor(primals_979, 0.9)
    add_359: "f32[216]" = torch.ops.aten.add.Tensor(mul_470, mul_471);  mul_470 = mul_471 = None
    squeeze_203: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
    mul_472: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_203, 1.0000708666997378);  squeeze_203 = None
    mul_473: "f32[216]" = torch.ops.aten.mul.Tensor(mul_472, 0.1);  mul_472 = None
    mul_474: "f32[216]" = torch.ops.aten.mul.Tensor(primals_980, 0.9)
    add_360: "f32[216]" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    unsqueeze_268: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_272, -1)
    unsqueeze_269: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_475: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_269);  mul_469 = unsqueeze_269 = None
    unsqueeze_270: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_273, -1);  primals_273 = None
    unsqueeze_271: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_361: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_475, unsqueeze_271);  mul_475 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_66: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_361);  add_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_125: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_66, primals_274, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_126: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_125, primals_275, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_362: "i64[]" = torch.ops.aten.add.Tensor(primals_984, 1)
    var_mean_68 = torch.ops.aten.var_mean.correction(convolution_126, [0, 2, 3], correction = 0, keepdim = True)
    getitem_164: "f32[1, 216, 1, 1]" = var_mean_68[0]
    getitem_165: "f32[1, 216, 1, 1]" = var_mean_68[1];  var_mean_68 = None
    add_363: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_164, 0.001)
    rsqrt_68: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_363);  add_363 = None
    sub_68: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_126, getitem_165)
    mul_476: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
    squeeze_204: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
    squeeze_205: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
    mul_477: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1)
    mul_478: "f32[216]" = torch.ops.aten.mul.Tensor(primals_982, 0.9)
    add_364: "f32[216]" = torch.ops.aten.add.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
    squeeze_206: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
    mul_479: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_206, 1.0000708666997378);  squeeze_206 = None
    mul_480: "f32[216]" = torch.ops.aten.mul.Tensor(mul_479, 0.1);  mul_479 = None
    mul_481: "f32[216]" = torch.ops.aten.mul.Tensor(primals_983, 0.9)
    add_365: "f32[216]" = torch.ops.aten.add.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
    unsqueeze_272: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_276, -1)
    unsqueeze_273: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    mul_482: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_273);  mul_476 = unsqueeze_273 = None
    unsqueeze_274: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_277, -1);  primals_277 = None
    unsqueeze_275: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    add_366: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_275);  mul_482 = unsqueeze_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_367: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_356, add_366);  add_356 = add_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_67: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_367)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_127: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_67, primals_278, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_128: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_127, primals_279, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_368: "i64[]" = torch.ops.aten.add.Tensor(primals_987, 1)
    var_mean_69 = torch.ops.aten.var_mean.correction(convolution_128, [0, 2, 3], correction = 0, keepdim = True)
    getitem_166: "f32[1, 216, 1, 1]" = var_mean_69[0]
    getitem_167: "f32[1, 216, 1, 1]" = var_mean_69[1];  var_mean_69 = None
    add_369: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_166, 0.001)
    rsqrt_69: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_369);  add_369 = None
    sub_69: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_128, getitem_167)
    mul_483: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
    squeeze_207: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
    squeeze_208: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
    mul_484: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1)
    mul_485: "f32[216]" = torch.ops.aten.mul.Tensor(primals_985, 0.9)
    add_370: "f32[216]" = torch.ops.aten.add.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
    squeeze_209: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
    mul_486: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_209, 1.0000708666997378);  squeeze_209 = None
    mul_487: "f32[216]" = torch.ops.aten.mul.Tensor(mul_486, 0.1);  mul_486 = None
    mul_488: "f32[216]" = torch.ops.aten.mul.Tensor(primals_986, 0.9)
    add_371: "f32[216]" = torch.ops.aten.add.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
    unsqueeze_276: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_280, -1)
    unsqueeze_277: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_489: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_483, unsqueeze_277);  mul_483 = unsqueeze_277 = None
    unsqueeze_278: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_281, -1);  primals_281 = None
    unsqueeze_279: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_372: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_489, unsqueeze_279);  mul_489 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_68: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_372);  add_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_129: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_68, primals_282, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_130: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_129, primals_283, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_373: "i64[]" = torch.ops.aten.add.Tensor(primals_990, 1)
    var_mean_70 = torch.ops.aten.var_mean.correction(convolution_130, [0, 2, 3], correction = 0, keepdim = True)
    getitem_168: "f32[1, 216, 1, 1]" = var_mean_70[0]
    getitem_169: "f32[1, 216, 1, 1]" = var_mean_70[1];  var_mean_70 = None
    add_374: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_168, 0.001)
    rsqrt_70: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_374);  add_374 = None
    sub_70: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_130, getitem_169)
    mul_490: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
    squeeze_210: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
    squeeze_211: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
    mul_491: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
    mul_492: "f32[216]" = torch.ops.aten.mul.Tensor(primals_988, 0.9)
    add_375: "f32[216]" = torch.ops.aten.add.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
    squeeze_212: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
    mul_493: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_212, 1.0000708666997378);  squeeze_212 = None
    mul_494: "f32[216]" = torch.ops.aten.mul.Tensor(mul_493, 0.1);  mul_493 = None
    mul_495: "f32[216]" = torch.ops.aten.mul.Tensor(primals_989, 0.9)
    add_376: "f32[216]" = torch.ops.aten.add.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
    unsqueeze_280: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_284, -1)
    unsqueeze_281: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    mul_496: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_281);  mul_490 = unsqueeze_281 = None
    unsqueeze_282: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_285, -1);  primals_285 = None
    unsqueeze_283: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    add_377: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_496, unsqueeze_283);  mul_496 = unsqueeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_378: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_377, getitem_156);  add_377 = getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_131: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_59, primals_286, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_132: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_131, primals_287, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_379: "i64[]" = torch.ops.aten.add.Tensor(primals_993, 1)
    var_mean_71 = torch.ops.aten.var_mean.correction(convolution_132, [0, 2, 3], correction = 0, keepdim = True)
    getitem_172: "f32[1, 216, 1, 1]" = var_mean_71[0]
    getitem_173: "f32[1, 216, 1, 1]" = var_mean_71[1];  var_mean_71 = None
    add_380: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_172, 0.001)
    rsqrt_71: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_380);  add_380 = None
    sub_71: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_132, getitem_173)
    mul_497: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
    squeeze_213: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3]);  getitem_173 = None
    squeeze_214: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
    mul_498: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1)
    mul_499: "f32[216]" = torch.ops.aten.mul.Tensor(primals_991, 0.9)
    add_381: "f32[216]" = torch.ops.aten.add.Tensor(mul_498, mul_499);  mul_498 = mul_499 = None
    squeeze_215: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_172, [0, 2, 3]);  getitem_172 = None
    mul_500: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_215, 1.0000708666997378);  squeeze_215 = None
    mul_501: "f32[216]" = torch.ops.aten.mul.Tensor(mul_500, 0.1);  mul_500 = None
    mul_502: "f32[216]" = torch.ops.aten.mul.Tensor(primals_992, 0.9)
    add_382: "f32[216]" = torch.ops.aten.add.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
    unsqueeze_284: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_288, -1)
    unsqueeze_285: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_503: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_497, unsqueeze_285);  mul_497 = unsqueeze_285 = None
    unsqueeze_286: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_289, -1);  primals_289 = None
    unsqueeze_287: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_383: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_287);  mul_503 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_70: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_383);  add_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_133: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_70, primals_290, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_134: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_133, primals_291, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_384: "i64[]" = torch.ops.aten.add.Tensor(primals_996, 1)
    var_mean_72 = torch.ops.aten.var_mean.correction(convolution_134, [0, 2, 3], correction = 0, keepdim = True)
    getitem_174: "f32[1, 216, 1, 1]" = var_mean_72[0]
    getitem_175: "f32[1, 216, 1, 1]" = var_mean_72[1];  var_mean_72 = None
    add_385: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_174, 0.001)
    rsqrt_72: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_385);  add_385 = None
    sub_72: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_134, getitem_175)
    mul_504: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
    squeeze_216: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
    squeeze_217: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
    mul_505: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1)
    mul_506: "f32[216]" = torch.ops.aten.mul.Tensor(primals_994, 0.9)
    add_386: "f32[216]" = torch.ops.aten.add.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    squeeze_218: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_174, [0, 2, 3]);  getitem_174 = None
    mul_507: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_218, 1.0000708666997378);  squeeze_218 = None
    mul_508: "f32[216]" = torch.ops.aten.mul.Tensor(mul_507, 0.1);  mul_507 = None
    mul_509: "f32[216]" = torch.ops.aten.mul.Tensor(primals_995, 0.9)
    add_387: "f32[216]" = torch.ops.aten.add.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
    unsqueeze_288: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_292, -1)
    unsqueeze_289: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    mul_510: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_504, unsqueeze_289);  mul_504 = unsqueeze_289 = None
    unsqueeze_290: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_293, -1);  primals_293 = None
    unsqueeze_291: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    add_388: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_510, unsqueeze_291);  mul_510 = unsqueeze_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_389: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_388, add_324);  add_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat_6: "f32[8, 1080, 42, 42]" = torch.ops.aten.cat.default([add_335, add_346, add_367, add_378, add_389], 1);  add_335 = add_346 = add_367 = add_378 = add_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_135: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_58, primals_294, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_390: "i64[]" = torch.ops.aten.add.Tensor(primals_999, 1)
    var_mean_73 = torch.ops.aten.var_mean.correction(convolution_135, [0, 2, 3], correction = 0, keepdim = True)
    getitem_176: "f32[1, 216, 1, 1]" = var_mean_73[0]
    getitem_177: "f32[1, 216, 1, 1]" = var_mean_73[1];  var_mean_73 = None
    add_391: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_176, 0.001)
    rsqrt_73: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_391);  add_391 = None
    sub_73: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_135, getitem_177)
    mul_511: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
    squeeze_219: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_177, [0, 2, 3]);  getitem_177 = None
    squeeze_220: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
    mul_512: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
    mul_513: "f32[216]" = torch.ops.aten.mul.Tensor(primals_997, 0.9)
    add_392: "f32[216]" = torch.ops.aten.add.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
    squeeze_221: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_176, [0, 2, 3]);  getitem_176 = None
    mul_514: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_221, 1.0000708666997378);  squeeze_221 = None
    mul_515: "f32[216]" = torch.ops.aten.mul.Tensor(mul_514, 0.1);  mul_514 = None
    mul_516: "f32[216]" = torch.ops.aten.mul.Tensor(primals_998, 0.9)
    add_393: "f32[216]" = torch.ops.aten.add.Tensor(mul_515, mul_516);  mul_515 = mul_516 = None
    unsqueeze_292: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_295, -1)
    unsqueeze_293: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_517: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_511, unsqueeze_293);  mul_511 = unsqueeze_293 = None
    unsqueeze_294: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_296, -1);  primals_296 = None
    unsqueeze_295: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_394: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_517, unsqueeze_295);  mul_517 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu_72: "f32[8, 1080, 42, 42]" = torch.ops.aten.relu.default(cat_6);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_136: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_72, primals_297, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_395: "i64[]" = torch.ops.aten.add.Tensor(primals_1002, 1)
    var_mean_74 = torch.ops.aten.var_mean.correction(convolution_136, [0, 2, 3], correction = 0, keepdim = True)
    getitem_178: "f32[1, 216, 1, 1]" = var_mean_74[0]
    getitem_179: "f32[1, 216, 1, 1]" = var_mean_74[1];  var_mean_74 = None
    add_396: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_178, 0.001)
    rsqrt_74: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_396);  add_396 = None
    sub_74: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_136, getitem_179)
    mul_518: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
    squeeze_222: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
    squeeze_223: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
    mul_519: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
    mul_520: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1000, 0.9)
    add_397: "f32[216]" = torch.ops.aten.add.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
    squeeze_224: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_178, [0, 2, 3]);  getitem_178 = None
    mul_521: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_224, 1.0000708666997378);  squeeze_224 = None
    mul_522: "f32[216]" = torch.ops.aten.mul.Tensor(mul_521, 0.1);  mul_521 = None
    mul_523: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1001, 0.9)
    add_398: "f32[216]" = torch.ops.aten.add.Tensor(mul_522, mul_523);  mul_522 = mul_523 = None
    unsqueeze_296: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_298, -1)
    unsqueeze_297: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    mul_524: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_518, unsqueeze_297);  mul_518 = unsqueeze_297 = None
    unsqueeze_298: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_299, -1);  primals_299 = None
    unsqueeze_299: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    add_399: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_524, unsqueeze_299);  mul_524 = unsqueeze_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_73: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_394)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_137: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_73, primals_300, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_138: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_137, primals_301, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_400: "i64[]" = torch.ops.aten.add.Tensor(primals_1005, 1)
    var_mean_75 = torch.ops.aten.var_mean.correction(convolution_138, [0, 2, 3], correction = 0, keepdim = True)
    getitem_180: "f32[1, 216, 1, 1]" = var_mean_75[0]
    getitem_181: "f32[1, 216, 1, 1]" = var_mean_75[1];  var_mean_75 = None
    add_401: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_180, 0.001)
    rsqrt_75: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_401);  add_401 = None
    sub_75: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_138, getitem_181)
    mul_525: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
    squeeze_225: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_181, [0, 2, 3]);  getitem_181 = None
    squeeze_226: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
    mul_526: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1)
    mul_527: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1003, 0.9)
    add_402: "f32[216]" = torch.ops.aten.add.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
    squeeze_227: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_180, [0, 2, 3]);  getitem_180 = None
    mul_528: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_227, 1.0000708666997378);  squeeze_227 = None
    mul_529: "f32[216]" = torch.ops.aten.mul.Tensor(mul_528, 0.1);  mul_528 = None
    mul_530: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1004, 0.9)
    add_403: "f32[216]" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    unsqueeze_300: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_302, -1)
    unsqueeze_301: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_531: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_301);  mul_525 = unsqueeze_301 = None
    unsqueeze_302: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_303, -1);  primals_303 = None
    unsqueeze_303: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_404: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_531, unsqueeze_303);  mul_531 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_74: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_404);  add_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_139: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_74, primals_304, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_140: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_139, primals_305, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_405: "i64[]" = torch.ops.aten.add.Tensor(primals_1008, 1)
    var_mean_76 = torch.ops.aten.var_mean.correction(convolution_140, [0, 2, 3], correction = 0, keepdim = True)
    getitem_182: "f32[1, 216, 1, 1]" = var_mean_76[0]
    getitem_183: "f32[1, 216, 1, 1]" = var_mean_76[1];  var_mean_76 = None
    add_406: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_182, 0.001)
    rsqrt_76: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_406);  add_406 = None
    sub_76: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_140, getitem_183)
    mul_532: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
    squeeze_228: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3]);  getitem_183 = None
    squeeze_229: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
    mul_533: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1)
    mul_534: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1006, 0.9)
    add_407: "f32[216]" = torch.ops.aten.add.Tensor(mul_533, mul_534);  mul_533 = mul_534 = None
    squeeze_230: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_182, [0, 2, 3]);  getitem_182 = None
    mul_535: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_230, 1.0000708666997378);  squeeze_230 = None
    mul_536: "f32[216]" = torch.ops.aten.mul.Tensor(mul_535, 0.1);  mul_535 = None
    mul_537: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1007, 0.9)
    add_408: "f32[216]" = torch.ops.aten.add.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
    unsqueeze_304: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_306, -1)
    unsqueeze_305: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    mul_538: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_305);  mul_532 = unsqueeze_305 = None
    unsqueeze_306: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_307, -1);  primals_307 = None
    unsqueeze_307: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    add_409: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_538, unsqueeze_307);  mul_538 = unsqueeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_15 = torch.ops.aten.max_pool2d_with_indices.default(add_394, [3, 3], [1, 1], [1, 1])
    getitem_184: "f32[8, 216, 42, 42]" = max_pool2d_with_indices_15[0]
    getitem_185: "i64[8, 216, 42, 42]" = max_pool2d_with_indices_15[1];  max_pool2d_with_indices_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_410: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_409, getitem_184);  add_409 = getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_75: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_399)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_141: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_75, primals_308, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_142: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_141, primals_309, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_411: "i64[]" = torch.ops.aten.add.Tensor(primals_1011, 1)
    var_mean_77 = torch.ops.aten.var_mean.correction(convolution_142, [0, 2, 3], correction = 0, keepdim = True)
    getitem_186: "f32[1, 216, 1, 1]" = var_mean_77[0]
    getitem_187: "f32[1, 216, 1, 1]" = var_mean_77[1];  var_mean_77 = None
    add_412: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_186, 0.001)
    rsqrt_77: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_412);  add_412 = None
    sub_77: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_142, getitem_187)
    mul_539: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
    squeeze_231: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
    squeeze_232: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
    mul_540: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
    mul_541: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1009, 0.9)
    add_413: "f32[216]" = torch.ops.aten.add.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
    squeeze_233: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_186, [0, 2, 3]);  getitem_186 = None
    mul_542: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_233, 1.0000708666997378);  squeeze_233 = None
    mul_543: "f32[216]" = torch.ops.aten.mul.Tensor(mul_542, 0.1);  mul_542 = None
    mul_544: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1010, 0.9)
    add_414: "f32[216]" = torch.ops.aten.add.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
    unsqueeze_308: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_310, -1)
    unsqueeze_309: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_545: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_539, unsqueeze_309);  mul_539 = unsqueeze_309 = None
    unsqueeze_310: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_311, -1);  primals_311 = None
    unsqueeze_311: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_415: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_311);  mul_545 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_76: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_415);  add_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_143: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_76, primals_312, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_144: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_143, primals_313, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_416: "i64[]" = torch.ops.aten.add.Tensor(primals_1014, 1)
    var_mean_78 = torch.ops.aten.var_mean.correction(convolution_144, [0, 2, 3], correction = 0, keepdim = True)
    getitem_188: "f32[1, 216, 1, 1]" = var_mean_78[0]
    getitem_189: "f32[1, 216, 1, 1]" = var_mean_78[1];  var_mean_78 = None
    add_417: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_188, 0.001)
    rsqrt_78: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_417);  add_417 = None
    sub_78: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_144, getitem_189)
    mul_546: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
    squeeze_234: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
    squeeze_235: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
    mul_547: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1)
    mul_548: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1012, 0.9)
    add_418: "f32[216]" = torch.ops.aten.add.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
    squeeze_236: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_188, [0, 2, 3]);  getitem_188 = None
    mul_549: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_236, 1.0000708666997378);  squeeze_236 = None
    mul_550: "f32[216]" = torch.ops.aten.mul.Tensor(mul_549, 0.1);  mul_549 = None
    mul_551: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1013, 0.9)
    add_419: "f32[216]" = torch.ops.aten.add.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
    unsqueeze_312: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_314, -1)
    unsqueeze_313: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    mul_552: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_546, unsqueeze_313);  mul_546 = unsqueeze_313 = None
    unsqueeze_314: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_315, -1);  primals_315 = None
    unsqueeze_315: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    add_420: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_552, unsqueeze_315);  mul_552 = unsqueeze_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_16 = torch.ops.aten.max_pool2d_with_indices.default(add_399, [3, 3], [1, 1], [1, 1])
    getitem_190: "f32[8, 216, 42, 42]" = max_pool2d_with_indices_16[0]
    getitem_191: "i64[8, 216, 42, 42]" = max_pool2d_with_indices_16[1];  max_pool2d_with_indices_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_421: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_420, getitem_190);  add_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_145: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_75, primals_316, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_146: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_145, primals_317, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_422: "i64[]" = torch.ops.aten.add.Tensor(primals_1017, 1)
    var_mean_79 = torch.ops.aten.var_mean.correction(convolution_146, [0, 2, 3], correction = 0, keepdim = True)
    getitem_192: "f32[1, 216, 1, 1]" = var_mean_79[0]
    getitem_193: "f32[1, 216, 1, 1]" = var_mean_79[1];  var_mean_79 = None
    add_423: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_192, 0.001)
    rsqrt_79: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_423);  add_423 = None
    sub_79: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_146, getitem_193)
    mul_553: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
    squeeze_237: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
    squeeze_238: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
    mul_554: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1)
    mul_555: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1015, 0.9)
    add_424: "f32[216]" = torch.ops.aten.add.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
    squeeze_239: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_192, [0, 2, 3]);  getitem_192 = None
    mul_556: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_239, 1.0000708666997378);  squeeze_239 = None
    mul_557: "f32[216]" = torch.ops.aten.mul.Tensor(mul_556, 0.1);  mul_556 = None
    mul_558: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1016, 0.9)
    add_425: "f32[216]" = torch.ops.aten.add.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
    unsqueeze_316: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_318, -1)
    unsqueeze_317: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_559: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_317);  mul_553 = unsqueeze_317 = None
    unsqueeze_318: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_319, -1);  primals_319 = None
    unsqueeze_319: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_426: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_559, unsqueeze_319);  mul_559 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_78: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_426);  add_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_147: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_78, primals_320, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_148: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_147, primals_321, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_427: "i64[]" = torch.ops.aten.add.Tensor(primals_1020, 1)
    var_mean_80 = torch.ops.aten.var_mean.correction(convolution_148, [0, 2, 3], correction = 0, keepdim = True)
    getitem_194: "f32[1, 216, 1, 1]" = var_mean_80[0]
    getitem_195: "f32[1, 216, 1, 1]" = var_mean_80[1];  var_mean_80 = None
    add_428: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_194, 0.001)
    rsqrt_80: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_428);  add_428 = None
    sub_80: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_148, getitem_195)
    mul_560: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = None
    squeeze_240: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3]);  getitem_195 = None
    squeeze_241: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
    mul_561: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_240, 0.1)
    mul_562: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1018, 0.9)
    add_429: "f32[216]" = torch.ops.aten.add.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    squeeze_242: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_194, [0, 2, 3]);  getitem_194 = None
    mul_563: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_242, 1.0000708666997378);  squeeze_242 = None
    mul_564: "f32[216]" = torch.ops.aten.mul.Tensor(mul_563, 0.1);  mul_563 = None
    mul_565: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1019, 0.9)
    add_430: "f32[216]" = torch.ops.aten.add.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_320: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_322, -1)
    unsqueeze_321: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    mul_566: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_321);  mul_560 = unsqueeze_321 = None
    unsqueeze_322: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_323, -1);  primals_323 = None
    unsqueeze_323: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    add_431: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_323);  mul_566 = unsqueeze_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_149: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_75, primals_324, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_150: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_149, primals_325, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_432: "i64[]" = torch.ops.aten.add.Tensor(primals_1023, 1)
    var_mean_81 = torch.ops.aten.var_mean.correction(convolution_150, [0, 2, 3], correction = 0, keepdim = True)
    getitem_196: "f32[1, 216, 1, 1]" = var_mean_81[0]
    getitem_197: "f32[1, 216, 1, 1]" = var_mean_81[1];  var_mean_81 = None
    add_433: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_196, 0.001)
    rsqrt_81: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_433);  add_433 = None
    sub_81: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_150, getitem_197)
    mul_567: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = None
    squeeze_243: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_197, [0, 2, 3]);  getitem_197 = None
    squeeze_244: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
    mul_568: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_243, 0.1)
    mul_569: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1021, 0.9)
    add_434: "f32[216]" = torch.ops.aten.add.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
    squeeze_245: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_196, [0, 2, 3]);  getitem_196 = None
    mul_570: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_245, 1.0000708666997378);  squeeze_245 = None
    mul_571: "f32[216]" = torch.ops.aten.mul.Tensor(mul_570, 0.1);  mul_570 = None
    mul_572: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1022, 0.9)
    add_435: "f32[216]" = torch.ops.aten.add.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
    unsqueeze_324: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_326, -1)
    unsqueeze_325: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_573: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_567, unsqueeze_325);  mul_567 = unsqueeze_325 = None
    unsqueeze_326: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_327, -1);  primals_327 = None
    unsqueeze_327: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_436: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_573, unsqueeze_327);  mul_573 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_80: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_436);  add_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_151: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_80, primals_328, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_152: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_151, primals_329, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_437: "i64[]" = torch.ops.aten.add.Tensor(primals_1026, 1)
    var_mean_82 = torch.ops.aten.var_mean.correction(convolution_152, [0, 2, 3], correction = 0, keepdim = True)
    getitem_198: "f32[1, 216, 1, 1]" = var_mean_82[0]
    getitem_199: "f32[1, 216, 1, 1]" = var_mean_82[1];  var_mean_82 = None
    add_438: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_198, 0.001)
    rsqrt_82: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_438);  add_438 = None
    sub_82: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_152, getitem_199)
    mul_574: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
    squeeze_246: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_199, [0, 2, 3]);  getitem_199 = None
    squeeze_247: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
    mul_575: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_246, 0.1)
    mul_576: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1024, 0.9)
    add_439: "f32[216]" = torch.ops.aten.add.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
    squeeze_248: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_198, [0, 2, 3]);  getitem_198 = None
    mul_577: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_248, 1.0000708666997378);  squeeze_248 = None
    mul_578: "f32[216]" = torch.ops.aten.mul.Tensor(mul_577, 0.1);  mul_577 = None
    mul_579: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1025, 0.9)
    add_440: "f32[216]" = torch.ops.aten.add.Tensor(mul_578, mul_579);  mul_578 = mul_579 = None
    unsqueeze_328: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_330, -1)
    unsqueeze_329: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    mul_580: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_329);  mul_574 = unsqueeze_329 = None
    unsqueeze_330: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_331, -1);  primals_331 = None
    unsqueeze_331: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    add_441: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_580, unsqueeze_331);  mul_580 = unsqueeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_442: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_431, add_441);  add_431 = add_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_81: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_442)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_153: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_81, primals_332, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_154: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_153, primals_333, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_443: "i64[]" = torch.ops.aten.add.Tensor(primals_1029, 1)
    var_mean_83 = torch.ops.aten.var_mean.correction(convolution_154, [0, 2, 3], correction = 0, keepdim = True)
    getitem_200: "f32[1, 216, 1, 1]" = var_mean_83[0]
    getitem_201: "f32[1, 216, 1, 1]" = var_mean_83[1];  var_mean_83 = None
    add_444: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_200, 0.001)
    rsqrt_83: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_444);  add_444 = None
    sub_83: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_154, getitem_201)
    mul_581: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
    squeeze_249: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_201, [0, 2, 3]);  getitem_201 = None
    squeeze_250: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
    mul_582: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_249, 0.1)
    mul_583: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1027, 0.9)
    add_445: "f32[216]" = torch.ops.aten.add.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    squeeze_251: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_200, [0, 2, 3]);  getitem_200 = None
    mul_584: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_251, 1.0000708666997378);  squeeze_251 = None
    mul_585: "f32[216]" = torch.ops.aten.mul.Tensor(mul_584, 0.1);  mul_584 = None
    mul_586: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1028, 0.9)
    add_446: "f32[216]" = torch.ops.aten.add.Tensor(mul_585, mul_586);  mul_585 = mul_586 = None
    unsqueeze_332: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_334, -1)
    unsqueeze_333: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_587: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_581, unsqueeze_333);  mul_581 = unsqueeze_333 = None
    unsqueeze_334: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_335, -1);  primals_335 = None
    unsqueeze_335: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_447: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_335);  mul_587 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_82: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_447);  add_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_155: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_82, primals_336, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_156: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_155, primals_337, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_448: "i64[]" = torch.ops.aten.add.Tensor(primals_1032, 1)
    var_mean_84 = torch.ops.aten.var_mean.correction(convolution_156, [0, 2, 3], correction = 0, keepdim = True)
    getitem_202: "f32[1, 216, 1, 1]" = var_mean_84[0]
    getitem_203: "f32[1, 216, 1, 1]" = var_mean_84[1];  var_mean_84 = None
    add_449: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_202, 0.001)
    rsqrt_84: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_449);  add_449 = None
    sub_84: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_156, getitem_203)
    mul_588: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
    squeeze_252: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_203, [0, 2, 3]);  getitem_203 = None
    squeeze_253: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
    mul_589: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_252, 0.1)
    mul_590: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1030, 0.9)
    add_450: "f32[216]" = torch.ops.aten.add.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
    squeeze_254: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_202, [0, 2, 3]);  getitem_202 = None
    mul_591: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_254, 1.0000708666997378);  squeeze_254 = None
    mul_592: "f32[216]" = torch.ops.aten.mul.Tensor(mul_591, 0.1);  mul_591 = None
    mul_593: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1031, 0.9)
    add_451: "f32[216]" = torch.ops.aten.add.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
    unsqueeze_336: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_338, -1)
    unsqueeze_337: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    mul_594: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_588, unsqueeze_337);  mul_588 = unsqueeze_337 = None
    unsqueeze_338: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_339, -1);  primals_339 = None
    unsqueeze_339: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    add_452: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_594, unsqueeze_339);  mul_594 = unsqueeze_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_453: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_452, getitem_190);  add_452 = getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_157: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_73, primals_340, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_158: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_157, primals_341, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_454: "i64[]" = torch.ops.aten.add.Tensor(primals_1035, 1)
    var_mean_85 = torch.ops.aten.var_mean.correction(convolution_158, [0, 2, 3], correction = 0, keepdim = True)
    getitem_206: "f32[1, 216, 1, 1]" = var_mean_85[0]
    getitem_207: "f32[1, 216, 1, 1]" = var_mean_85[1];  var_mean_85 = None
    add_455: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_206, 0.001)
    rsqrt_85: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_455);  add_455 = None
    sub_85: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_158, getitem_207)
    mul_595: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = None
    squeeze_255: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_207, [0, 2, 3]);  getitem_207 = None
    squeeze_256: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
    mul_596: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_255, 0.1)
    mul_597: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1033, 0.9)
    add_456: "f32[216]" = torch.ops.aten.add.Tensor(mul_596, mul_597);  mul_596 = mul_597 = None
    squeeze_257: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_206, [0, 2, 3]);  getitem_206 = None
    mul_598: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_257, 1.0000708666997378);  squeeze_257 = None
    mul_599: "f32[216]" = torch.ops.aten.mul.Tensor(mul_598, 0.1);  mul_598 = None
    mul_600: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1034, 0.9)
    add_457: "f32[216]" = torch.ops.aten.add.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
    unsqueeze_340: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_342, -1)
    unsqueeze_341: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_601: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_341);  mul_595 = unsqueeze_341 = None
    unsqueeze_342: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_343, -1);  primals_343 = None
    unsqueeze_343: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_458: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_601, unsqueeze_343);  mul_601 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_84: "f32[8, 216, 42, 42]" = torch.ops.aten.relu.default(add_458);  add_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_159: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(relu_84, primals_344, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_160: "f32[8, 216, 42, 42]" = torch.ops.aten.convolution.default(convolution_159, primals_345, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_459: "i64[]" = torch.ops.aten.add.Tensor(primals_1038, 1)
    var_mean_86 = torch.ops.aten.var_mean.correction(convolution_160, [0, 2, 3], correction = 0, keepdim = True)
    getitem_208: "f32[1, 216, 1, 1]" = var_mean_86[0]
    getitem_209: "f32[1, 216, 1, 1]" = var_mean_86[1];  var_mean_86 = None
    add_460: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_208, 0.001)
    rsqrt_86: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_460);  add_460 = None
    sub_86: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_160, getitem_209)
    mul_602: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = None
    squeeze_258: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_209, [0, 2, 3]);  getitem_209 = None
    squeeze_259: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_86, [0, 2, 3]);  rsqrt_86 = None
    mul_603: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_258, 0.1)
    mul_604: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1036, 0.9)
    add_461: "f32[216]" = torch.ops.aten.add.Tensor(mul_603, mul_604);  mul_603 = mul_604 = None
    squeeze_260: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_208, [0, 2, 3]);  getitem_208 = None
    mul_605: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_260, 1.0000708666997378);  squeeze_260 = None
    mul_606: "f32[216]" = torch.ops.aten.mul.Tensor(mul_605, 0.1);  mul_605 = None
    mul_607: "f32[216]" = torch.ops.aten.mul.Tensor(primals_1037, 0.9)
    add_462: "f32[216]" = torch.ops.aten.add.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
    unsqueeze_344: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_346, -1)
    unsqueeze_345: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    mul_608: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(mul_602, unsqueeze_345);  mul_602 = unsqueeze_345 = None
    unsqueeze_346: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_347, -1);  primals_347 = None
    unsqueeze_347: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    add_463: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(mul_608, unsqueeze_347);  mul_608 = unsqueeze_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_464: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_463, add_399);  add_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat_7: "f32[8, 1080, 42, 42]" = torch.ops.aten.cat.default([add_410, add_421, add_442, add_453, add_464], 1);  add_410 = add_421 = add_442 = add_453 = add_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_161: "f32[8, 432, 42, 42]" = torch.ops.aten.convolution.default(relu_72, primals_348, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_465: "i64[]" = torch.ops.aten.add.Tensor(primals_1041, 1)
    var_mean_87 = torch.ops.aten.var_mean.correction(convolution_161, [0, 2, 3], correction = 0, keepdim = True)
    getitem_210: "f32[1, 432, 1, 1]" = var_mean_87[0]
    getitem_211: "f32[1, 432, 1, 1]" = var_mean_87[1];  var_mean_87 = None
    add_466: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_210, 0.001)
    rsqrt_87: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_466);  add_466 = None
    sub_87: "f32[8, 432, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_161, getitem_211)
    mul_609: "f32[8, 432, 42, 42]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = None
    squeeze_261: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_211, [0, 2, 3]);  getitem_211 = None
    squeeze_262: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
    mul_610: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_261, 0.1)
    mul_611: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1039, 0.9)
    add_467: "f32[432]" = torch.ops.aten.add.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
    squeeze_263: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_210, [0, 2, 3]);  getitem_210 = None
    mul_612: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_263, 1.0000708666997378);  squeeze_263 = None
    mul_613: "f32[432]" = torch.ops.aten.mul.Tensor(mul_612, 0.1);  mul_612 = None
    mul_614: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1040, 0.9)
    add_468: "f32[432]" = torch.ops.aten.add.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
    unsqueeze_348: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_349, -1)
    unsqueeze_349: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_615: "f32[8, 432, 42, 42]" = torch.ops.aten.mul.Tensor(mul_609, unsqueeze_349);  mul_609 = unsqueeze_349 = None
    unsqueeze_350: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_350, -1);  primals_350 = None
    unsqueeze_351: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_469: "f32[8, 432, 42, 42]" = torch.ops.aten.add.Tensor(mul_615, unsqueeze_351);  mul_615 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu_86: "f32[8, 1080, 42, 42]" = torch.ops.aten.relu.default(cat_7);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_162: "f32[8, 432, 42, 42]" = torch.ops.aten.convolution.default(relu_86, primals_351, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_470: "i64[]" = torch.ops.aten.add.Tensor(primals_1044, 1)
    var_mean_88 = torch.ops.aten.var_mean.correction(convolution_162, [0, 2, 3], correction = 0, keepdim = True)
    getitem_212: "f32[1, 432, 1, 1]" = var_mean_88[0]
    getitem_213: "f32[1, 432, 1, 1]" = var_mean_88[1];  var_mean_88 = None
    add_471: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_212, 0.001)
    rsqrt_88: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_471);  add_471 = None
    sub_88: "f32[8, 432, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_162, getitem_213)
    mul_616: "f32[8, 432, 42, 42]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = None
    squeeze_264: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_213, [0, 2, 3]);  getitem_213 = None
    squeeze_265: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
    mul_617: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_264, 0.1)
    mul_618: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1042, 0.9)
    add_472: "f32[432]" = torch.ops.aten.add.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
    squeeze_266: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_212, [0, 2, 3]);  getitem_212 = None
    mul_619: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_266, 1.0000708666997378);  squeeze_266 = None
    mul_620: "f32[432]" = torch.ops.aten.mul.Tensor(mul_619, 0.1);  mul_619 = None
    mul_621: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1043, 0.9)
    add_473: "f32[432]" = torch.ops.aten.add.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
    unsqueeze_352: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_352, -1)
    unsqueeze_353: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    mul_622: "f32[8, 432, 42, 42]" = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_353);  mul_616 = unsqueeze_353 = None
    unsqueeze_354: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_353, -1);  primals_353 = None
    unsqueeze_355: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    add_474: "f32[8, 432, 42, 42]" = torch.ops.aten.add.Tensor(mul_622, unsqueeze_355);  mul_622 = unsqueeze_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_87: "f32[8, 432, 42, 42]" = torch.ops.aten.relu.default(add_469)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_20: "f32[8, 432, 45, 45]" = torch.ops.aten.constant_pad_nd.default(relu_87, [1, 2, 1, 2], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_163: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(constant_pad_nd_20, primals_15, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_164: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_163, primals_354, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_475: "i64[]" = torch.ops.aten.add.Tensor(primals_1047, 1)
    var_mean_89 = torch.ops.aten.var_mean.correction(convolution_164, [0, 2, 3], correction = 0, keepdim = True)
    getitem_214: "f32[1, 432, 1, 1]" = var_mean_89[0]
    getitem_215: "f32[1, 432, 1, 1]" = var_mean_89[1];  var_mean_89 = None
    add_476: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_214, 0.001)
    rsqrt_89: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_476);  add_476 = None
    sub_89: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_164, getitem_215)
    mul_623: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = None
    squeeze_267: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_215, [0, 2, 3]);  getitem_215 = None
    squeeze_268: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_89, [0, 2, 3]);  rsqrt_89 = None
    mul_624: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_267, 0.1)
    mul_625: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1045, 0.9)
    add_477: "f32[432]" = torch.ops.aten.add.Tensor(mul_624, mul_625);  mul_624 = mul_625 = None
    squeeze_269: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_214, [0, 2, 3]);  getitem_214 = None
    mul_626: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_269, 1.0002835270768358);  squeeze_269 = None
    mul_627: "f32[432]" = torch.ops.aten.mul.Tensor(mul_626, 0.1);  mul_626 = None
    mul_628: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1046, 0.9)
    add_478: "f32[432]" = torch.ops.aten.add.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
    unsqueeze_356: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_355, -1)
    unsqueeze_357: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_629: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_623, unsqueeze_357);  mul_623 = unsqueeze_357 = None
    unsqueeze_358: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_356, -1);  primals_356 = None
    unsqueeze_359: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_479: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_629, unsqueeze_359);  mul_629 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_88: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_479);  add_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_165: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_88, primals_357, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_166: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_165, primals_358, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_480: "i64[]" = torch.ops.aten.add.Tensor(primals_1050, 1)
    var_mean_90 = torch.ops.aten.var_mean.correction(convolution_166, [0, 2, 3], correction = 0, keepdim = True)
    getitem_216: "f32[1, 432, 1, 1]" = var_mean_90[0]
    getitem_217: "f32[1, 432, 1, 1]" = var_mean_90[1];  var_mean_90 = None
    add_481: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_216, 0.001)
    rsqrt_90: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_481);  add_481 = None
    sub_90: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_166, getitem_217)
    mul_630: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = None
    squeeze_270: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_217, [0, 2, 3]);  getitem_217 = None
    squeeze_271: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_90, [0, 2, 3]);  rsqrt_90 = None
    mul_631: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_270, 0.1)
    mul_632: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1048, 0.9)
    add_482: "f32[432]" = torch.ops.aten.add.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
    squeeze_272: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_216, [0, 2, 3]);  getitem_216 = None
    mul_633: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_272, 1.0002835270768358);  squeeze_272 = None
    mul_634: "f32[432]" = torch.ops.aten.mul.Tensor(mul_633, 0.1);  mul_633 = None
    mul_635: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1049, 0.9)
    add_483: "f32[432]" = torch.ops.aten.add.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
    unsqueeze_360: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_359, -1)
    unsqueeze_361: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    mul_636: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_630, unsqueeze_361);  mul_630 = unsqueeze_361 = None
    unsqueeze_362: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_360, -1);  primals_360 = None
    unsqueeze_363: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    add_484: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_636, unsqueeze_363);  mul_636 = unsqueeze_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_21: "f32[8, 432, 43, 43]" = torch.ops.aten.constant_pad_nd.default(add_469, [0, 1, 0, 1], -inf);  add_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_18 = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd_21, [3, 3], [2, 2])
    getitem_218: "f32[8, 432, 21, 21]" = max_pool2d_with_indices_18[0]
    getitem_219: "i64[8, 432, 21, 21]" = max_pool2d_with_indices_18[1];  max_pool2d_with_indices_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_485: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_484, getitem_218);  add_484 = getitem_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_89: "f32[8, 432, 42, 42]" = torch.ops.aten.relu.default(add_474)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_22: "f32[8, 432, 47, 47]" = torch.ops.aten.constant_pad_nd.default(relu_89, [2, 3, 2, 3], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_167: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(constant_pad_nd_22, primals_16, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_168: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_167, primals_361, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_486: "i64[]" = torch.ops.aten.add.Tensor(primals_1053, 1)
    var_mean_91 = torch.ops.aten.var_mean.correction(convolution_168, [0, 2, 3], correction = 0, keepdim = True)
    getitem_220: "f32[1, 432, 1, 1]" = var_mean_91[0]
    getitem_221: "f32[1, 432, 1, 1]" = var_mean_91[1];  var_mean_91 = None
    add_487: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_220, 0.001)
    rsqrt_91: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_487);  add_487 = None
    sub_91: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_168, getitem_221)
    mul_637: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = None
    squeeze_273: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_221, [0, 2, 3]);  getitem_221 = None
    squeeze_274: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_91, [0, 2, 3]);  rsqrt_91 = None
    mul_638: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_273, 0.1)
    mul_639: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1051, 0.9)
    add_488: "f32[432]" = torch.ops.aten.add.Tensor(mul_638, mul_639);  mul_638 = mul_639 = None
    squeeze_275: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_220, [0, 2, 3]);  getitem_220 = None
    mul_640: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_275, 1.0002835270768358);  squeeze_275 = None
    mul_641: "f32[432]" = torch.ops.aten.mul.Tensor(mul_640, 0.1);  mul_640 = None
    mul_642: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1052, 0.9)
    add_489: "f32[432]" = torch.ops.aten.add.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
    unsqueeze_364: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_362, -1)
    unsqueeze_365: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_643: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_365);  mul_637 = unsqueeze_365 = None
    unsqueeze_366: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_363, -1);  primals_363 = None
    unsqueeze_367: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_490: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_643, unsqueeze_367);  mul_643 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_90: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_490);  add_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_169: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_90, primals_364, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_170: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_169, primals_365, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_491: "i64[]" = torch.ops.aten.add.Tensor(primals_1056, 1)
    var_mean_92 = torch.ops.aten.var_mean.correction(convolution_170, [0, 2, 3], correction = 0, keepdim = True)
    getitem_222: "f32[1, 432, 1, 1]" = var_mean_92[0]
    getitem_223: "f32[1, 432, 1, 1]" = var_mean_92[1];  var_mean_92 = None
    add_492: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_222, 0.001)
    rsqrt_92: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_492);  add_492 = None
    sub_92: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_170, getitem_223)
    mul_644: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = None
    squeeze_276: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_223, [0, 2, 3]);  getitem_223 = None
    squeeze_277: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_92, [0, 2, 3]);  rsqrt_92 = None
    mul_645: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_276, 0.1)
    mul_646: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1054, 0.9)
    add_493: "f32[432]" = torch.ops.aten.add.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
    squeeze_278: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_222, [0, 2, 3]);  getitem_222 = None
    mul_647: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_278, 1.0002835270768358);  squeeze_278 = None
    mul_648: "f32[432]" = torch.ops.aten.mul.Tensor(mul_647, 0.1);  mul_647 = None
    mul_649: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1055, 0.9)
    add_494: "f32[432]" = torch.ops.aten.add.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
    unsqueeze_368: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_366, -1)
    unsqueeze_369: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    mul_650: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_644, unsqueeze_369);  mul_644 = unsqueeze_369 = None
    unsqueeze_370: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_367, -1);  primals_367 = None
    unsqueeze_371: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    add_495: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_650, unsqueeze_371);  mul_650 = unsqueeze_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_23: "f32[8, 432, 43, 43]" = torch.ops.aten.constant_pad_nd.default(add_474, [0, 1, 0, 1], -inf);  add_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_19 = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd_23, [3, 3], [2, 2])
    getitem_224: "f32[8, 432, 21, 21]" = max_pool2d_with_indices_19[0]
    getitem_225: "i64[8, 432, 21, 21]" = max_pool2d_with_indices_19[1];  max_pool2d_with_indices_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_496: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_495, getitem_224);  add_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_24: "f32[8, 432, 45, 45]" = torch.ops.aten.constant_pad_nd.default(relu_89, [1, 2, 1, 2], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_171: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(constant_pad_nd_24, primals_17, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_172: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_171, primals_368, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_497: "i64[]" = torch.ops.aten.add.Tensor(primals_1059, 1)
    var_mean_93 = torch.ops.aten.var_mean.correction(convolution_172, [0, 2, 3], correction = 0, keepdim = True)
    getitem_226: "f32[1, 432, 1, 1]" = var_mean_93[0]
    getitem_227: "f32[1, 432, 1, 1]" = var_mean_93[1];  var_mean_93 = None
    add_498: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_226, 0.001)
    rsqrt_93: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_498);  add_498 = None
    sub_93: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_172, getitem_227)
    mul_651: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = None
    squeeze_279: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_227, [0, 2, 3]);  getitem_227 = None
    squeeze_280: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
    mul_652: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_279, 0.1)
    mul_653: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1057, 0.9)
    add_499: "f32[432]" = torch.ops.aten.add.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
    squeeze_281: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_226, [0, 2, 3]);  getitem_226 = None
    mul_654: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_281, 1.0002835270768358);  squeeze_281 = None
    mul_655: "f32[432]" = torch.ops.aten.mul.Tensor(mul_654, 0.1);  mul_654 = None
    mul_656: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1058, 0.9)
    add_500: "f32[432]" = torch.ops.aten.add.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
    unsqueeze_372: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_369, -1)
    unsqueeze_373: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_657: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_651, unsqueeze_373);  mul_651 = unsqueeze_373 = None
    unsqueeze_374: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_370, -1);  primals_370 = None
    unsqueeze_375: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_501: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_657, unsqueeze_375);  mul_657 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_92: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_501);  add_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_173: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_92, primals_371, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_174: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_173, primals_372, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_502: "i64[]" = torch.ops.aten.add.Tensor(primals_1062, 1)
    var_mean_94 = torch.ops.aten.var_mean.correction(convolution_174, [0, 2, 3], correction = 0, keepdim = True)
    getitem_228: "f32[1, 432, 1, 1]" = var_mean_94[0]
    getitem_229: "f32[1, 432, 1, 1]" = var_mean_94[1];  var_mean_94 = None
    add_503: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_228, 0.001)
    rsqrt_94: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_503);  add_503 = None
    sub_94: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_174, getitem_229)
    mul_658: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = None
    squeeze_282: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_229, [0, 2, 3]);  getitem_229 = None
    squeeze_283: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_94, [0, 2, 3]);  rsqrt_94 = None
    mul_659: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_282, 0.1)
    mul_660: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1060, 0.9)
    add_504: "f32[432]" = torch.ops.aten.add.Tensor(mul_659, mul_660);  mul_659 = mul_660 = None
    squeeze_284: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_228, [0, 2, 3]);  getitem_228 = None
    mul_661: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_284, 1.0002835270768358);  squeeze_284 = None
    mul_662: "f32[432]" = torch.ops.aten.mul.Tensor(mul_661, 0.1);  mul_661 = None
    mul_663: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1061, 0.9)
    add_505: "f32[432]" = torch.ops.aten.add.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
    unsqueeze_376: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_373, -1)
    unsqueeze_377: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    mul_664: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_377);  mul_658 = unsqueeze_377 = None
    unsqueeze_378: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_374, -1);  primals_374 = None
    unsqueeze_379: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    add_506: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_664, unsqueeze_379);  mul_664 = unsqueeze_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_25: "f32[8, 432, 43, 43]" = torch.ops.aten.constant_pad_nd.default(relu_89, [0, 1, 0, 1], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_175: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(constant_pad_nd_25, primals_18, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_176: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_175, primals_375, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_507: "i64[]" = torch.ops.aten.add.Tensor(primals_1065, 1)
    var_mean_95 = torch.ops.aten.var_mean.correction(convolution_176, [0, 2, 3], correction = 0, keepdim = True)
    getitem_230: "f32[1, 432, 1, 1]" = var_mean_95[0]
    getitem_231: "f32[1, 432, 1, 1]" = var_mean_95[1];  var_mean_95 = None
    add_508: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_230, 0.001)
    rsqrt_95: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_508);  add_508 = None
    sub_95: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_176, getitem_231)
    mul_665: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = None
    squeeze_285: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_231, [0, 2, 3]);  getitem_231 = None
    squeeze_286: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_95, [0, 2, 3]);  rsqrt_95 = None
    mul_666: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_285, 0.1)
    mul_667: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1063, 0.9)
    add_509: "f32[432]" = torch.ops.aten.add.Tensor(mul_666, mul_667);  mul_666 = mul_667 = None
    squeeze_287: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_230, [0, 2, 3]);  getitem_230 = None
    mul_668: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_287, 1.0002835270768358);  squeeze_287 = None
    mul_669: "f32[432]" = torch.ops.aten.mul.Tensor(mul_668, 0.1);  mul_668 = None
    mul_670: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1064, 0.9)
    add_510: "f32[432]" = torch.ops.aten.add.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    unsqueeze_380: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_376, -1)
    unsqueeze_381: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_671: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_665, unsqueeze_381);  mul_665 = unsqueeze_381 = None
    unsqueeze_382: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_377, -1);  primals_377 = None
    unsqueeze_383: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_511: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_671, unsqueeze_383);  mul_671 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_94: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_511);  add_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_177: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_94, primals_378, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_178: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_177, primals_379, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_512: "i64[]" = torch.ops.aten.add.Tensor(primals_1068, 1)
    var_mean_96 = torch.ops.aten.var_mean.correction(convolution_178, [0, 2, 3], correction = 0, keepdim = True)
    getitem_232: "f32[1, 432, 1, 1]" = var_mean_96[0]
    getitem_233: "f32[1, 432, 1, 1]" = var_mean_96[1];  var_mean_96 = None
    add_513: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_232, 0.001)
    rsqrt_96: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_513);  add_513 = None
    sub_96: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_178, getitem_233)
    mul_672: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = None
    squeeze_288: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_233, [0, 2, 3]);  getitem_233 = None
    squeeze_289: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_96, [0, 2, 3]);  rsqrt_96 = None
    mul_673: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_288, 0.1)
    mul_674: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1066, 0.9)
    add_514: "f32[432]" = torch.ops.aten.add.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
    squeeze_290: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_232, [0, 2, 3]);  getitem_232 = None
    mul_675: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_290, 1.0002835270768358);  squeeze_290 = None
    mul_676: "f32[432]" = torch.ops.aten.mul.Tensor(mul_675, 0.1);  mul_675 = None
    mul_677: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1067, 0.9)
    add_515: "f32[432]" = torch.ops.aten.add.Tensor(mul_676, mul_677);  mul_676 = mul_677 = None
    unsqueeze_384: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_380, -1)
    unsqueeze_385: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    mul_678: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_672, unsqueeze_385);  mul_672 = unsqueeze_385 = None
    unsqueeze_386: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_381, -1);  primals_381 = None
    unsqueeze_387: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    add_516: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_678, unsqueeze_387);  mul_678 = unsqueeze_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_517: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_506, add_516);  add_506 = add_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_95: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_517)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_179: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_95, primals_382, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_180: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_179, primals_383, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_518: "i64[]" = torch.ops.aten.add.Tensor(primals_1071, 1)
    var_mean_97 = torch.ops.aten.var_mean.correction(convolution_180, [0, 2, 3], correction = 0, keepdim = True)
    getitem_234: "f32[1, 432, 1, 1]" = var_mean_97[0]
    getitem_235: "f32[1, 432, 1, 1]" = var_mean_97[1];  var_mean_97 = None
    add_519: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_234, 0.001)
    rsqrt_97: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_519);  add_519 = None
    sub_97: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_180, getitem_235)
    mul_679: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = None
    squeeze_291: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_235, [0, 2, 3]);  getitem_235 = None
    squeeze_292: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_97, [0, 2, 3]);  rsqrt_97 = None
    mul_680: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_291, 0.1)
    mul_681: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1069, 0.9)
    add_520: "f32[432]" = torch.ops.aten.add.Tensor(mul_680, mul_681);  mul_680 = mul_681 = None
    squeeze_293: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_234, [0, 2, 3]);  getitem_234 = None
    mul_682: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_293, 1.0002835270768358);  squeeze_293 = None
    mul_683: "f32[432]" = torch.ops.aten.mul.Tensor(mul_682, 0.1);  mul_682 = None
    mul_684: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1070, 0.9)
    add_521: "f32[432]" = torch.ops.aten.add.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
    unsqueeze_388: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_384, -1)
    unsqueeze_389: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_685: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_679, unsqueeze_389);  mul_679 = unsqueeze_389 = None
    unsqueeze_390: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_385, -1);  primals_385 = None
    unsqueeze_391: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_522: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_685, unsqueeze_391);  mul_685 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_96: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_522);  add_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_181: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_96, primals_386, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_182: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_181, primals_387, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_523: "i64[]" = torch.ops.aten.add.Tensor(primals_1074, 1)
    var_mean_98 = torch.ops.aten.var_mean.correction(convolution_182, [0, 2, 3], correction = 0, keepdim = True)
    getitem_236: "f32[1, 432, 1, 1]" = var_mean_98[0]
    getitem_237: "f32[1, 432, 1, 1]" = var_mean_98[1];  var_mean_98 = None
    add_524: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_236, 0.001)
    rsqrt_98: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_524);  add_524 = None
    sub_98: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_182, getitem_237)
    mul_686: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = None
    squeeze_294: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_237, [0, 2, 3]);  getitem_237 = None
    squeeze_295: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_98, [0, 2, 3]);  rsqrt_98 = None
    mul_687: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_294, 0.1)
    mul_688: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1072, 0.9)
    add_525: "f32[432]" = torch.ops.aten.add.Tensor(mul_687, mul_688);  mul_687 = mul_688 = None
    squeeze_296: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_236, [0, 2, 3]);  getitem_236 = None
    mul_689: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_296, 1.0002835270768358);  squeeze_296 = None
    mul_690: "f32[432]" = torch.ops.aten.mul.Tensor(mul_689, 0.1);  mul_689 = None
    mul_691: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1073, 0.9)
    add_526: "f32[432]" = torch.ops.aten.add.Tensor(mul_690, mul_691);  mul_690 = mul_691 = None
    unsqueeze_392: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_388, -1)
    unsqueeze_393: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    mul_692: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_686, unsqueeze_393);  mul_686 = unsqueeze_393 = None
    unsqueeze_394: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_389, -1);  primals_389 = None
    unsqueeze_395: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    add_527: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_692, unsqueeze_395);  mul_692 = unsqueeze_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_528: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_527, getitem_224);  add_527 = getitem_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_27: "f32[8, 432, 43, 43]" = torch.ops.aten.constant_pad_nd.default(relu_87, [0, 1, 0, 1], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_183: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(constant_pad_nd_27, primals_19, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_184: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_183, primals_390, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_529: "i64[]" = torch.ops.aten.add.Tensor(primals_1077, 1)
    var_mean_99 = torch.ops.aten.var_mean.correction(convolution_184, [0, 2, 3], correction = 0, keepdim = True)
    getitem_240: "f32[1, 432, 1, 1]" = var_mean_99[0]
    getitem_241: "f32[1, 432, 1, 1]" = var_mean_99[1];  var_mean_99 = None
    add_530: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_240, 0.001)
    rsqrt_99: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_530);  add_530 = None
    sub_99: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_184, getitem_241)
    mul_693: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = None
    squeeze_297: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_241, [0, 2, 3]);  getitem_241 = None
    squeeze_298: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_99, [0, 2, 3]);  rsqrt_99 = None
    mul_694: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_297, 0.1)
    mul_695: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1075, 0.9)
    add_531: "f32[432]" = torch.ops.aten.add.Tensor(mul_694, mul_695);  mul_694 = mul_695 = None
    squeeze_299: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_240, [0, 2, 3]);  getitem_240 = None
    mul_696: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_299, 1.0002835270768358);  squeeze_299 = None
    mul_697: "f32[432]" = torch.ops.aten.mul.Tensor(mul_696, 0.1);  mul_696 = None
    mul_698: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1076, 0.9)
    add_532: "f32[432]" = torch.ops.aten.add.Tensor(mul_697, mul_698);  mul_697 = mul_698 = None
    unsqueeze_396: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_391, -1)
    unsqueeze_397: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_699: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_693, unsqueeze_397);  mul_693 = unsqueeze_397 = None
    unsqueeze_398: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_392, -1);  primals_392 = None
    unsqueeze_399: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_533: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_699, unsqueeze_399);  mul_699 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_98: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_533);  add_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_185: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_98, primals_393, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_186: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_185, primals_394, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_534: "i64[]" = torch.ops.aten.add.Tensor(primals_1080, 1)
    var_mean_100 = torch.ops.aten.var_mean.correction(convolution_186, [0, 2, 3], correction = 0, keepdim = True)
    getitem_242: "f32[1, 432, 1, 1]" = var_mean_100[0]
    getitem_243: "f32[1, 432, 1, 1]" = var_mean_100[1];  var_mean_100 = None
    add_535: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_242, 0.001)
    rsqrt_100: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_535);  add_535 = None
    sub_100: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_186, getitem_243)
    mul_700: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = None
    squeeze_300: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_243, [0, 2, 3]);  getitem_243 = None
    squeeze_301: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_100, [0, 2, 3]);  rsqrt_100 = None
    mul_701: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_300, 0.1)
    mul_702: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1078, 0.9)
    add_536: "f32[432]" = torch.ops.aten.add.Tensor(mul_701, mul_702);  mul_701 = mul_702 = None
    squeeze_302: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_242, [0, 2, 3]);  getitem_242 = None
    mul_703: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_302, 1.0002835270768358);  squeeze_302 = None
    mul_704: "f32[432]" = torch.ops.aten.mul.Tensor(mul_703, 0.1);  mul_703 = None
    mul_705: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1079, 0.9)
    add_537: "f32[432]" = torch.ops.aten.add.Tensor(mul_704, mul_705);  mul_704 = mul_705 = None
    unsqueeze_400: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_395, -1)
    unsqueeze_401: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    mul_706: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_700, unsqueeze_401);  mul_700 = unsqueeze_401 = None
    unsqueeze_402: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_396, -1);  primals_396 = None
    unsqueeze_403: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    add_538: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_706, unsqueeze_403);  mul_706 = unsqueeze_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_187: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_89, primals_20, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_539: "i64[]" = torch.ops.aten.add.Tensor(primals_1083, 1)
    var_mean_101 = torch.ops.aten.var_mean.correction(convolution_187, [0, 2, 3], correction = 0, keepdim = True)
    getitem_244: "f32[1, 432, 1, 1]" = var_mean_101[0]
    getitem_245: "f32[1, 432, 1, 1]" = var_mean_101[1];  var_mean_101 = None
    add_540: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_244, 0.001)
    rsqrt_101: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_540);  add_540 = None
    sub_101: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_187, getitem_245)
    mul_707: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = None
    squeeze_303: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_245, [0, 2, 3]);  getitem_245 = None
    squeeze_304: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_101, [0, 2, 3]);  rsqrt_101 = None
    mul_708: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_303, 0.1)
    mul_709: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1081, 0.9)
    add_541: "f32[432]" = torch.ops.aten.add.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
    squeeze_305: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_244, [0, 2, 3]);  getitem_244 = None
    mul_710: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_305, 1.0002835270768358);  squeeze_305 = None
    mul_711: "f32[432]" = torch.ops.aten.mul.Tensor(mul_710, 0.1);  mul_710 = None
    mul_712: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1082, 0.9)
    add_542: "f32[432]" = torch.ops.aten.add.Tensor(mul_711, mul_712);  mul_711 = mul_712 = None
    unsqueeze_404: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_397, -1)
    unsqueeze_405: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_713: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_707, unsqueeze_405);  mul_707 = unsqueeze_405 = None
    unsqueeze_406: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_398, -1);  primals_398 = None
    unsqueeze_407: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_543: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_713, unsqueeze_407);  mul_713 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_544: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_538, add_543);  add_538 = add_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat_8: "f32[8, 2160, 21, 21]" = torch.ops.aten.cat.default([add_485, add_496, add_517, add_528, add_544], 1);  add_485 = add_496 = add_517 = add_528 = add_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    avg_pool2d_4: "f32[8, 1080, 21, 21]" = torch.ops.aten.avg_pool2d.default(relu_86, [1, 1], [2, 2], [0, 0], False, False)
    convolution_188: "f32[8, 216, 21, 21]" = torch.ops.aten.convolution.default(avg_pool2d_4, primals_399, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:97, code: x_path2 = self.path_2(x)
    constant_pad_nd_29: "f32[8, 1080, 42, 42]" = torch.ops.aten.constant_pad_nd.default(relu_86, [-1, 1, -1, 1], 0.0)
    avg_pool2d_5: "f32[8, 1080, 21, 21]" = torch.ops.aten.avg_pool2d.default(constant_pad_nd_29, [1, 1], [2, 2], [0, 0], False, False)
    convolution_189: "f32[8, 216, 21, 21]" = torch.ops.aten.convolution.default(avg_pool2d_5, primals_400, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    cat_9: "f32[8, 432, 21, 21]" = torch.ops.aten.cat.default([convolution_188, convolution_189], 1);  convolution_188 = convolution_189 = None
    add_545: "i64[]" = torch.ops.aten.add.Tensor(primals_1086, 1)
    var_mean_102 = torch.ops.aten.var_mean.correction(cat_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_246: "f32[1, 432, 1, 1]" = var_mean_102[0]
    getitem_247: "f32[1, 432, 1, 1]" = var_mean_102[1];  var_mean_102 = None
    add_546: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_246, 0.001)
    rsqrt_102: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_546);  add_546 = None
    sub_102: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(cat_9, getitem_247)
    mul_714: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = None
    squeeze_306: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_247, [0, 2, 3]);  getitem_247 = None
    squeeze_307: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_102, [0, 2, 3]);  rsqrt_102 = None
    mul_715: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_306, 0.1)
    mul_716: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1084, 0.9)
    add_547: "f32[432]" = torch.ops.aten.add.Tensor(mul_715, mul_716);  mul_715 = mul_716 = None
    squeeze_308: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_246, [0, 2, 3]);  getitem_246 = None
    mul_717: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_308, 1.0002835270768358);  squeeze_308 = None
    mul_718: "f32[432]" = torch.ops.aten.mul.Tensor(mul_717, 0.1);  mul_717 = None
    mul_719: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1085, 0.9)
    add_548: "f32[432]" = torch.ops.aten.add.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
    unsqueeze_408: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_401, -1)
    unsqueeze_409: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    mul_720: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_714, unsqueeze_409);  mul_714 = unsqueeze_409 = None
    unsqueeze_410: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_402, -1);  primals_402 = None
    unsqueeze_411: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    add_549: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_720, unsqueeze_411);  mul_720 = unsqueeze_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu_101: "f32[8, 2160, 21, 21]" = torch.ops.aten.relu.default(cat_8);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_190: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_101, primals_403, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_550: "i64[]" = torch.ops.aten.add.Tensor(primals_1089, 1)
    var_mean_103 = torch.ops.aten.var_mean.correction(convolution_190, [0, 2, 3], correction = 0, keepdim = True)
    getitem_248: "f32[1, 432, 1, 1]" = var_mean_103[0]
    getitem_249: "f32[1, 432, 1, 1]" = var_mean_103[1];  var_mean_103 = None
    add_551: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_248, 0.001)
    rsqrt_103: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_551);  add_551 = None
    sub_103: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_190, getitem_249)
    mul_721: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = None
    squeeze_309: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_249, [0, 2, 3]);  getitem_249 = None
    squeeze_310: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_103, [0, 2, 3]);  rsqrt_103 = None
    mul_722: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_309, 0.1)
    mul_723: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1087, 0.9)
    add_552: "f32[432]" = torch.ops.aten.add.Tensor(mul_722, mul_723);  mul_722 = mul_723 = None
    squeeze_311: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_248, [0, 2, 3]);  getitem_248 = None
    mul_724: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_311, 1.0002835270768358);  squeeze_311 = None
    mul_725: "f32[432]" = torch.ops.aten.mul.Tensor(mul_724, 0.1);  mul_724 = None
    mul_726: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1088, 0.9)
    add_553: "f32[432]" = torch.ops.aten.add.Tensor(mul_725, mul_726);  mul_725 = mul_726 = None
    unsqueeze_412: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_404, -1)
    unsqueeze_413: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_727: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_721, unsqueeze_413);  mul_721 = unsqueeze_413 = None
    unsqueeze_414: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_405, -1);  primals_405 = None
    unsqueeze_415: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_554: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_727, unsqueeze_415);  mul_727 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_102: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_549)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_191: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_102, primals_406, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_192: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_191, primals_407, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_555: "i64[]" = torch.ops.aten.add.Tensor(primals_1092, 1)
    var_mean_104 = torch.ops.aten.var_mean.correction(convolution_192, [0, 2, 3], correction = 0, keepdim = True)
    getitem_250: "f32[1, 432, 1, 1]" = var_mean_104[0]
    getitem_251: "f32[1, 432, 1, 1]" = var_mean_104[1];  var_mean_104 = None
    add_556: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_250, 0.001)
    rsqrt_104: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_556);  add_556 = None
    sub_104: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_192, getitem_251)
    mul_728: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_104);  sub_104 = None
    squeeze_312: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_251, [0, 2, 3]);  getitem_251 = None
    squeeze_313: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_104, [0, 2, 3]);  rsqrt_104 = None
    mul_729: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_312, 0.1)
    mul_730: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1090, 0.9)
    add_557: "f32[432]" = torch.ops.aten.add.Tensor(mul_729, mul_730);  mul_729 = mul_730 = None
    squeeze_314: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_250, [0, 2, 3]);  getitem_250 = None
    mul_731: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_314, 1.0002835270768358);  squeeze_314 = None
    mul_732: "f32[432]" = torch.ops.aten.mul.Tensor(mul_731, 0.1);  mul_731 = None
    mul_733: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1091, 0.9)
    add_558: "f32[432]" = torch.ops.aten.add.Tensor(mul_732, mul_733);  mul_732 = mul_733 = None
    unsqueeze_416: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_408, -1)
    unsqueeze_417: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    mul_734: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_728, unsqueeze_417);  mul_728 = unsqueeze_417 = None
    unsqueeze_418: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_409, -1);  primals_409 = None
    unsqueeze_419: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    add_559: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_734, unsqueeze_419);  mul_734 = unsqueeze_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_103: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_559);  add_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_193: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_103, primals_410, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_194: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_193, primals_411, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_560: "i64[]" = torch.ops.aten.add.Tensor(primals_1095, 1)
    var_mean_105 = torch.ops.aten.var_mean.correction(convolution_194, [0, 2, 3], correction = 0, keepdim = True)
    getitem_252: "f32[1, 432, 1, 1]" = var_mean_105[0]
    getitem_253: "f32[1, 432, 1, 1]" = var_mean_105[1];  var_mean_105 = None
    add_561: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_252, 0.001)
    rsqrt_105: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_561);  add_561 = None
    sub_105: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_194, getitem_253)
    mul_735: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_105);  sub_105 = None
    squeeze_315: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_253, [0, 2, 3]);  getitem_253 = None
    squeeze_316: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_105, [0, 2, 3]);  rsqrt_105 = None
    mul_736: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_315, 0.1)
    mul_737: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1093, 0.9)
    add_562: "f32[432]" = torch.ops.aten.add.Tensor(mul_736, mul_737);  mul_736 = mul_737 = None
    squeeze_317: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_252, [0, 2, 3]);  getitem_252 = None
    mul_738: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_317, 1.0002835270768358);  squeeze_317 = None
    mul_739: "f32[432]" = torch.ops.aten.mul.Tensor(mul_738, 0.1);  mul_738 = None
    mul_740: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1094, 0.9)
    add_563: "f32[432]" = torch.ops.aten.add.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
    unsqueeze_420: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_412, -1)
    unsqueeze_421: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_741: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_735, unsqueeze_421);  mul_735 = unsqueeze_421 = None
    unsqueeze_422: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_413, -1);  primals_413 = None
    unsqueeze_423: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_564: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_741, unsqueeze_423);  mul_741 = unsqueeze_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_21 = torch.ops.aten.max_pool2d_with_indices.default(add_549, [3, 3], [1, 1], [1, 1])
    getitem_254: "f32[8, 432, 21, 21]" = max_pool2d_with_indices_21[0]
    getitem_255: "i64[8, 432, 21, 21]" = max_pool2d_with_indices_21[1];  max_pool2d_with_indices_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_565: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_564, getitem_254);  add_564 = getitem_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_104: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_554)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_195: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_104, primals_414, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_196: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_195, primals_415, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_566: "i64[]" = torch.ops.aten.add.Tensor(primals_1098, 1)
    var_mean_106 = torch.ops.aten.var_mean.correction(convolution_196, [0, 2, 3], correction = 0, keepdim = True)
    getitem_256: "f32[1, 432, 1, 1]" = var_mean_106[0]
    getitem_257: "f32[1, 432, 1, 1]" = var_mean_106[1];  var_mean_106 = None
    add_567: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_256, 0.001)
    rsqrt_106: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_567);  add_567 = None
    sub_106: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_196, getitem_257)
    mul_742: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_106);  sub_106 = None
    squeeze_318: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_257, [0, 2, 3]);  getitem_257 = None
    squeeze_319: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_106, [0, 2, 3]);  rsqrt_106 = None
    mul_743: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_318, 0.1)
    mul_744: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1096, 0.9)
    add_568: "f32[432]" = torch.ops.aten.add.Tensor(mul_743, mul_744);  mul_743 = mul_744 = None
    squeeze_320: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_256, [0, 2, 3]);  getitem_256 = None
    mul_745: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_320, 1.0002835270768358);  squeeze_320 = None
    mul_746: "f32[432]" = torch.ops.aten.mul.Tensor(mul_745, 0.1);  mul_745 = None
    mul_747: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1097, 0.9)
    add_569: "f32[432]" = torch.ops.aten.add.Tensor(mul_746, mul_747);  mul_746 = mul_747 = None
    unsqueeze_424: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_416, -1)
    unsqueeze_425: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    mul_748: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_742, unsqueeze_425);  mul_742 = unsqueeze_425 = None
    unsqueeze_426: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_417, -1);  primals_417 = None
    unsqueeze_427: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    add_570: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_748, unsqueeze_427);  mul_748 = unsqueeze_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_105: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_570);  add_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_197: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_105, primals_418, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_198: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_197, primals_419, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_571: "i64[]" = torch.ops.aten.add.Tensor(primals_1101, 1)
    var_mean_107 = torch.ops.aten.var_mean.correction(convolution_198, [0, 2, 3], correction = 0, keepdim = True)
    getitem_258: "f32[1, 432, 1, 1]" = var_mean_107[0]
    getitem_259: "f32[1, 432, 1, 1]" = var_mean_107[1];  var_mean_107 = None
    add_572: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_258, 0.001)
    rsqrt_107: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_572);  add_572 = None
    sub_107: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_198, getitem_259)
    mul_749: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_107);  sub_107 = None
    squeeze_321: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_259, [0, 2, 3]);  getitem_259 = None
    squeeze_322: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_107, [0, 2, 3]);  rsqrt_107 = None
    mul_750: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_321, 0.1)
    mul_751: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1099, 0.9)
    add_573: "f32[432]" = torch.ops.aten.add.Tensor(mul_750, mul_751);  mul_750 = mul_751 = None
    squeeze_323: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_258, [0, 2, 3]);  getitem_258 = None
    mul_752: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_323, 1.0002835270768358);  squeeze_323 = None
    mul_753: "f32[432]" = torch.ops.aten.mul.Tensor(mul_752, 0.1);  mul_752 = None
    mul_754: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1100, 0.9)
    add_574: "f32[432]" = torch.ops.aten.add.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
    unsqueeze_428: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_420, -1)
    unsqueeze_429: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_755: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_749, unsqueeze_429);  mul_749 = unsqueeze_429 = None
    unsqueeze_430: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_421, -1);  primals_421 = None
    unsqueeze_431: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_575: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_755, unsqueeze_431);  mul_755 = unsqueeze_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_22 = torch.ops.aten.max_pool2d_with_indices.default(add_554, [3, 3], [1, 1], [1, 1])
    getitem_260: "f32[8, 432, 21, 21]" = max_pool2d_with_indices_22[0]
    getitem_261: "i64[8, 432, 21, 21]" = max_pool2d_with_indices_22[1];  max_pool2d_with_indices_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_576: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_575, getitem_260);  add_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_199: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_104, primals_422, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_200: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_199, primals_423, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_577: "i64[]" = torch.ops.aten.add.Tensor(primals_1104, 1)
    var_mean_108 = torch.ops.aten.var_mean.correction(convolution_200, [0, 2, 3], correction = 0, keepdim = True)
    getitem_262: "f32[1, 432, 1, 1]" = var_mean_108[0]
    getitem_263: "f32[1, 432, 1, 1]" = var_mean_108[1];  var_mean_108 = None
    add_578: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_262, 0.001)
    rsqrt_108: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_578);  add_578 = None
    sub_108: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_200, getitem_263)
    mul_756: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_108);  sub_108 = None
    squeeze_324: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_263, [0, 2, 3]);  getitem_263 = None
    squeeze_325: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_108, [0, 2, 3]);  rsqrt_108 = None
    mul_757: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_324, 0.1)
    mul_758: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1102, 0.9)
    add_579: "f32[432]" = torch.ops.aten.add.Tensor(mul_757, mul_758);  mul_757 = mul_758 = None
    squeeze_326: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_262, [0, 2, 3]);  getitem_262 = None
    mul_759: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_326, 1.0002835270768358);  squeeze_326 = None
    mul_760: "f32[432]" = torch.ops.aten.mul.Tensor(mul_759, 0.1);  mul_759 = None
    mul_761: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1103, 0.9)
    add_580: "f32[432]" = torch.ops.aten.add.Tensor(mul_760, mul_761);  mul_760 = mul_761 = None
    unsqueeze_432: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_424, -1)
    unsqueeze_433: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    mul_762: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_756, unsqueeze_433);  mul_756 = unsqueeze_433 = None
    unsqueeze_434: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_425, -1);  primals_425 = None
    unsqueeze_435: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    add_581: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_762, unsqueeze_435);  mul_762 = unsqueeze_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_107: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_581);  add_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_201: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_107, primals_426, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_202: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_201, primals_427, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_582: "i64[]" = torch.ops.aten.add.Tensor(primals_1107, 1)
    var_mean_109 = torch.ops.aten.var_mean.correction(convolution_202, [0, 2, 3], correction = 0, keepdim = True)
    getitem_264: "f32[1, 432, 1, 1]" = var_mean_109[0]
    getitem_265: "f32[1, 432, 1, 1]" = var_mean_109[1];  var_mean_109 = None
    add_583: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_264, 0.001)
    rsqrt_109: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_583);  add_583 = None
    sub_109: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_202, getitem_265)
    mul_763: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_109);  sub_109 = None
    squeeze_327: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_265, [0, 2, 3]);  getitem_265 = None
    squeeze_328: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_109, [0, 2, 3]);  rsqrt_109 = None
    mul_764: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_327, 0.1)
    mul_765: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1105, 0.9)
    add_584: "f32[432]" = torch.ops.aten.add.Tensor(mul_764, mul_765);  mul_764 = mul_765 = None
    squeeze_329: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_264, [0, 2, 3]);  getitem_264 = None
    mul_766: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_329, 1.0002835270768358);  squeeze_329 = None
    mul_767: "f32[432]" = torch.ops.aten.mul.Tensor(mul_766, 0.1);  mul_766 = None
    mul_768: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1106, 0.9)
    add_585: "f32[432]" = torch.ops.aten.add.Tensor(mul_767, mul_768);  mul_767 = mul_768 = None
    unsqueeze_436: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_428, -1)
    unsqueeze_437: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_769: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_763, unsqueeze_437);  mul_763 = unsqueeze_437 = None
    unsqueeze_438: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_429, -1);  primals_429 = None
    unsqueeze_439: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_586: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_769, unsqueeze_439);  mul_769 = unsqueeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_203: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_104, primals_430, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_204: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_203, primals_431, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_587: "i64[]" = torch.ops.aten.add.Tensor(primals_1110, 1)
    var_mean_110 = torch.ops.aten.var_mean.correction(convolution_204, [0, 2, 3], correction = 0, keepdim = True)
    getitem_266: "f32[1, 432, 1, 1]" = var_mean_110[0]
    getitem_267: "f32[1, 432, 1, 1]" = var_mean_110[1];  var_mean_110 = None
    add_588: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_266, 0.001)
    rsqrt_110: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_588);  add_588 = None
    sub_110: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_204, getitem_267)
    mul_770: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_110);  sub_110 = None
    squeeze_330: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_267, [0, 2, 3]);  getitem_267 = None
    squeeze_331: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_110, [0, 2, 3]);  rsqrt_110 = None
    mul_771: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_330, 0.1)
    mul_772: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1108, 0.9)
    add_589: "f32[432]" = torch.ops.aten.add.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
    squeeze_332: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_266, [0, 2, 3]);  getitem_266 = None
    mul_773: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_332, 1.0002835270768358);  squeeze_332 = None
    mul_774: "f32[432]" = torch.ops.aten.mul.Tensor(mul_773, 0.1);  mul_773 = None
    mul_775: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1109, 0.9)
    add_590: "f32[432]" = torch.ops.aten.add.Tensor(mul_774, mul_775);  mul_774 = mul_775 = None
    unsqueeze_440: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_432, -1)
    unsqueeze_441: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    mul_776: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_770, unsqueeze_441);  mul_770 = unsqueeze_441 = None
    unsqueeze_442: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_433, -1);  primals_433 = None
    unsqueeze_443: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    add_591: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_776, unsqueeze_443);  mul_776 = unsqueeze_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_109: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_591);  add_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_205: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_109, primals_434, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_206: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_205, primals_435, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_592: "i64[]" = torch.ops.aten.add.Tensor(primals_1113, 1)
    var_mean_111 = torch.ops.aten.var_mean.correction(convolution_206, [0, 2, 3], correction = 0, keepdim = True)
    getitem_268: "f32[1, 432, 1, 1]" = var_mean_111[0]
    getitem_269: "f32[1, 432, 1, 1]" = var_mean_111[1];  var_mean_111 = None
    add_593: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_268, 0.001)
    rsqrt_111: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_593);  add_593 = None
    sub_111: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_206, getitem_269)
    mul_777: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_111);  sub_111 = None
    squeeze_333: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_269, [0, 2, 3]);  getitem_269 = None
    squeeze_334: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_111, [0, 2, 3]);  rsqrt_111 = None
    mul_778: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_333, 0.1)
    mul_779: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1111, 0.9)
    add_594: "f32[432]" = torch.ops.aten.add.Tensor(mul_778, mul_779);  mul_778 = mul_779 = None
    squeeze_335: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_268, [0, 2, 3]);  getitem_268 = None
    mul_780: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_335, 1.0002835270768358);  squeeze_335 = None
    mul_781: "f32[432]" = torch.ops.aten.mul.Tensor(mul_780, 0.1);  mul_780 = None
    mul_782: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1112, 0.9)
    add_595: "f32[432]" = torch.ops.aten.add.Tensor(mul_781, mul_782);  mul_781 = mul_782 = None
    unsqueeze_444: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_436, -1)
    unsqueeze_445: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_783: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_777, unsqueeze_445);  mul_777 = unsqueeze_445 = None
    unsqueeze_446: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_437, -1);  primals_437 = None
    unsqueeze_447: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_596: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_783, unsqueeze_447);  mul_783 = unsqueeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_597: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_586, add_596);  add_586 = add_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_110: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_597)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_207: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_110, primals_438, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_208: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_207, primals_439, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_598: "i64[]" = torch.ops.aten.add.Tensor(primals_1116, 1)
    var_mean_112 = torch.ops.aten.var_mean.correction(convolution_208, [0, 2, 3], correction = 0, keepdim = True)
    getitem_270: "f32[1, 432, 1, 1]" = var_mean_112[0]
    getitem_271: "f32[1, 432, 1, 1]" = var_mean_112[1];  var_mean_112 = None
    add_599: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_270, 0.001)
    rsqrt_112: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_599);  add_599 = None
    sub_112: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_208, getitem_271)
    mul_784: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_112);  sub_112 = None
    squeeze_336: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_271, [0, 2, 3]);  getitem_271 = None
    squeeze_337: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_112, [0, 2, 3]);  rsqrt_112 = None
    mul_785: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_336, 0.1)
    mul_786: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1114, 0.9)
    add_600: "f32[432]" = torch.ops.aten.add.Tensor(mul_785, mul_786);  mul_785 = mul_786 = None
    squeeze_338: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_270, [0, 2, 3]);  getitem_270 = None
    mul_787: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_338, 1.0002835270768358);  squeeze_338 = None
    mul_788: "f32[432]" = torch.ops.aten.mul.Tensor(mul_787, 0.1);  mul_787 = None
    mul_789: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1115, 0.9)
    add_601: "f32[432]" = torch.ops.aten.add.Tensor(mul_788, mul_789);  mul_788 = mul_789 = None
    unsqueeze_448: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_440, -1)
    unsqueeze_449: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    mul_790: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_784, unsqueeze_449);  mul_784 = unsqueeze_449 = None
    unsqueeze_450: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_441, -1);  primals_441 = None
    unsqueeze_451: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    add_602: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_790, unsqueeze_451);  mul_790 = unsqueeze_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_111: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_602);  add_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_209: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_111, primals_442, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_210: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_209, primals_443, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_603: "i64[]" = torch.ops.aten.add.Tensor(primals_1119, 1)
    var_mean_113 = torch.ops.aten.var_mean.correction(convolution_210, [0, 2, 3], correction = 0, keepdim = True)
    getitem_272: "f32[1, 432, 1, 1]" = var_mean_113[0]
    getitem_273: "f32[1, 432, 1, 1]" = var_mean_113[1];  var_mean_113 = None
    add_604: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_272, 0.001)
    rsqrt_113: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_604);  add_604 = None
    sub_113: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_210, getitem_273)
    mul_791: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_113);  sub_113 = None
    squeeze_339: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_273, [0, 2, 3]);  getitem_273 = None
    squeeze_340: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_113, [0, 2, 3]);  rsqrt_113 = None
    mul_792: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_339, 0.1)
    mul_793: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1117, 0.9)
    add_605: "f32[432]" = torch.ops.aten.add.Tensor(mul_792, mul_793);  mul_792 = mul_793 = None
    squeeze_341: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_272, [0, 2, 3]);  getitem_272 = None
    mul_794: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_341, 1.0002835270768358);  squeeze_341 = None
    mul_795: "f32[432]" = torch.ops.aten.mul.Tensor(mul_794, 0.1);  mul_794 = None
    mul_796: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1118, 0.9)
    add_606: "f32[432]" = torch.ops.aten.add.Tensor(mul_795, mul_796);  mul_795 = mul_796 = None
    unsqueeze_452: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_444, -1)
    unsqueeze_453: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_797: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_791, unsqueeze_453);  mul_791 = unsqueeze_453 = None
    unsqueeze_454: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_445, -1);  primals_445 = None
    unsqueeze_455: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_607: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_797, unsqueeze_455);  mul_797 = unsqueeze_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_608: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_607, getitem_260);  add_607 = getitem_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_211: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_102, primals_446, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_212: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_211, primals_447, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_609: "i64[]" = torch.ops.aten.add.Tensor(primals_1122, 1)
    var_mean_114 = torch.ops.aten.var_mean.correction(convolution_212, [0, 2, 3], correction = 0, keepdim = True)
    getitem_276: "f32[1, 432, 1, 1]" = var_mean_114[0]
    getitem_277: "f32[1, 432, 1, 1]" = var_mean_114[1];  var_mean_114 = None
    add_610: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_276, 0.001)
    rsqrt_114: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_610);  add_610 = None
    sub_114: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_212, getitem_277)
    mul_798: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_114);  sub_114 = None
    squeeze_342: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_277, [0, 2, 3]);  getitem_277 = None
    squeeze_343: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_114, [0, 2, 3]);  rsqrt_114 = None
    mul_799: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_342, 0.1)
    mul_800: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1120, 0.9)
    add_611: "f32[432]" = torch.ops.aten.add.Tensor(mul_799, mul_800);  mul_799 = mul_800 = None
    squeeze_344: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_276, [0, 2, 3]);  getitem_276 = None
    mul_801: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_344, 1.0002835270768358);  squeeze_344 = None
    mul_802: "f32[432]" = torch.ops.aten.mul.Tensor(mul_801, 0.1);  mul_801 = None
    mul_803: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1121, 0.9)
    add_612: "f32[432]" = torch.ops.aten.add.Tensor(mul_802, mul_803);  mul_802 = mul_803 = None
    unsqueeze_456: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_448, -1)
    unsqueeze_457: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    mul_804: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_798, unsqueeze_457);  mul_798 = unsqueeze_457 = None
    unsqueeze_458: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_449, -1);  primals_449 = None
    unsqueeze_459: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    add_613: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_804, unsqueeze_459);  mul_804 = unsqueeze_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_113: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_613);  add_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_213: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_113, primals_450, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_214: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_213, primals_451, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_614: "i64[]" = torch.ops.aten.add.Tensor(primals_1125, 1)
    var_mean_115 = torch.ops.aten.var_mean.correction(convolution_214, [0, 2, 3], correction = 0, keepdim = True)
    getitem_278: "f32[1, 432, 1, 1]" = var_mean_115[0]
    getitem_279: "f32[1, 432, 1, 1]" = var_mean_115[1];  var_mean_115 = None
    add_615: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_278, 0.001)
    rsqrt_115: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_615);  add_615 = None
    sub_115: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_214, getitem_279)
    mul_805: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_115);  sub_115 = None
    squeeze_345: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_279, [0, 2, 3]);  getitem_279 = None
    squeeze_346: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_115, [0, 2, 3]);  rsqrt_115 = None
    mul_806: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_345, 0.1)
    mul_807: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1123, 0.9)
    add_616: "f32[432]" = torch.ops.aten.add.Tensor(mul_806, mul_807);  mul_806 = mul_807 = None
    squeeze_347: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_278, [0, 2, 3]);  getitem_278 = None
    mul_808: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_347, 1.0002835270768358);  squeeze_347 = None
    mul_809: "f32[432]" = torch.ops.aten.mul.Tensor(mul_808, 0.1);  mul_808 = None
    mul_810: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1124, 0.9)
    add_617: "f32[432]" = torch.ops.aten.add.Tensor(mul_809, mul_810);  mul_809 = mul_810 = None
    unsqueeze_460: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_452, -1)
    unsqueeze_461: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_811: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_805, unsqueeze_461);  mul_805 = unsqueeze_461 = None
    unsqueeze_462: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_453, -1);  primals_453 = None
    unsqueeze_463: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_618: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_811, unsqueeze_463);  mul_811 = unsqueeze_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_619: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_618, add_554);  add_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat_10: "f32[8, 2160, 21, 21]" = torch.ops.aten.cat.default([add_565, add_576, add_597, add_608, add_619], 1);  add_565 = add_576 = add_597 = add_608 = add_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_215: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_101, primals_454, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_620: "i64[]" = torch.ops.aten.add.Tensor(primals_1128, 1)
    var_mean_116 = torch.ops.aten.var_mean.correction(convolution_215, [0, 2, 3], correction = 0, keepdim = True)
    getitem_280: "f32[1, 432, 1, 1]" = var_mean_116[0]
    getitem_281: "f32[1, 432, 1, 1]" = var_mean_116[1];  var_mean_116 = None
    add_621: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_280, 0.001)
    rsqrt_116: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_621);  add_621 = None
    sub_116: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_215, getitem_281)
    mul_812: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_116);  sub_116 = None
    squeeze_348: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_281, [0, 2, 3]);  getitem_281 = None
    squeeze_349: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_116, [0, 2, 3]);  rsqrt_116 = None
    mul_813: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_348, 0.1)
    mul_814: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1126, 0.9)
    add_622: "f32[432]" = torch.ops.aten.add.Tensor(mul_813, mul_814);  mul_813 = mul_814 = None
    squeeze_350: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_280, [0, 2, 3]);  getitem_280 = None
    mul_815: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_350, 1.0002835270768358);  squeeze_350 = None
    mul_816: "f32[432]" = torch.ops.aten.mul.Tensor(mul_815, 0.1);  mul_815 = None
    mul_817: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1127, 0.9)
    add_623: "f32[432]" = torch.ops.aten.add.Tensor(mul_816, mul_817);  mul_816 = mul_817 = None
    unsqueeze_464: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_455, -1)
    unsqueeze_465: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
    mul_818: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_812, unsqueeze_465);  mul_812 = unsqueeze_465 = None
    unsqueeze_466: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_456, -1);  primals_456 = None
    unsqueeze_467: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    add_624: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_818, unsqueeze_467);  mul_818 = unsqueeze_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu_115: "f32[8, 2160, 21, 21]" = torch.ops.aten.relu.default(cat_10);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_216: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_115, primals_457, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_625: "i64[]" = torch.ops.aten.add.Tensor(primals_1131, 1)
    var_mean_117 = torch.ops.aten.var_mean.correction(convolution_216, [0, 2, 3], correction = 0, keepdim = True)
    getitem_282: "f32[1, 432, 1, 1]" = var_mean_117[0]
    getitem_283: "f32[1, 432, 1, 1]" = var_mean_117[1];  var_mean_117 = None
    add_626: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_282, 0.001)
    rsqrt_117: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_626);  add_626 = None
    sub_117: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_216, getitem_283)
    mul_819: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_117);  sub_117 = None
    squeeze_351: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_283, [0, 2, 3]);  getitem_283 = None
    squeeze_352: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_117, [0, 2, 3]);  rsqrt_117 = None
    mul_820: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_351, 0.1)
    mul_821: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1129, 0.9)
    add_627: "f32[432]" = torch.ops.aten.add.Tensor(mul_820, mul_821);  mul_820 = mul_821 = None
    squeeze_353: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_282, [0, 2, 3]);  getitem_282 = None
    mul_822: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_353, 1.0002835270768358);  squeeze_353 = None
    mul_823: "f32[432]" = torch.ops.aten.mul.Tensor(mul_822, 0.1);  mul_822 = None
    mul_824: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1130, 0.9)
    add_628: "f32[432]" = torch.ops.aten.add.Tensor(mul_823, mul_824);  mul_823 = mul_824 = None
    unsqueeze_468: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_458, -1)
    unsqueeze_469: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    mul_825: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_819, unsqueeze_469);  mul_819 = unsqueeze_469 = None
    unsqueeze_470: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_459, -1);  primals_459 = None
    unsqueeze_471: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
    add_629: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_825, unsqueeze_471);  mul_825 = unsqueeze_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_116: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_624)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_217: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_116, primals_460, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_218: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_217, primals_461, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_630: "i64[]" = torch.ops.aten.add.Tensor(primals_1134, 1)
    var_mean_118 = torch.ops.aten.var_mean.correction(convolution_218, [0, 2, 3], correction = 0, keepdim = True)
    getitem_284: "f32[1, 432, 1, 1]" = var_mean_118[0]
    getitem_285: "f32[1, 432, 1, 1]" = var_mean_118[1];  var_mean_118 = None
    add_631: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_284, 0.001)
    rsqrt_118: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_631);  add_631 = None
    sub_118: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_218, getitem_285)
    mul_826: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_118);  sub_118 = None
    squeeze_354: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_285, [0, 2, 3]);  getitem_285 = None
    squeeze_355: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_118, [0, 2, 3]);  rsqrt_118 = None
    mul_827: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_354, 0.1)
    mul_828: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1132, 0.9)
    add_632: "f32[432]" = torch.ops.aten.add.Tensor(mul_827, mul_828);  mul_827 = mul_828 = None
    squeeze_356: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_284, [0, 2, 3]);  getitem_284 = None
    mul_829: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_356, 1.0002835270768358);  squeeze_356 = None
    mul_830: "f32[432]" = torch.ops.aten.mul.Tensor(mul_829, 0.1);  mul_829 = None
    mul_831: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1133, 0.9)
    add_633: "f32[432]" = torch.ops.aten.add.Tensor(mul_830, mul_831);  mul_830 = mul_831 = None
    unsqueeze_472: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_462, -1)
    unsqueeze_473: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
    mul_832: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_826, unsqueeze_473);  mul_826 = unsqueeze_473 = None
    unsqueeze_474: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_463, -1);  primals_463 = None
    unsqueeze_475: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
    add_634: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_832, unsqueeze_475);  mul_832 = unsqueeze_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_117: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_634);  add_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_219: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_117, primals_464, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_220: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_219, primals_465, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_635: "i64[]" = torch.ops.aten.add.Tensor(primals_1137, 1)
    var_mean_119 = torch.ops.aten.var_mean.correction(convolution_220, [0, 2, 3], correction = 0, keepdim = True)
    getitem_286: "f32[1, 432, 1, 1]" = var_mean_119[0]
    getitem_287: "f32[1, 432, 1, 1]" = var_mean_119[1];  var_mean_119 = None
    add_636: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_286, 0.001)
    rsqrt_119: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_636);  add_636 = None
    sub_119: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_220, getitem_287)
    mul_833: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_119);  sub_119 = None
    squeeze_357: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_287, [0, 2, 3]);  getitem_287 = None
    squeeze_358: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_119, [0, 2, 3]);  rsqrt_119 = None
    mul_834: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_357, 0.1)
    mul_835: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1135, 0.9)
    add_637: "f32[432]" = torch.ops.aten.add.Tensor(mul_834, mul_835);  mul_834 = mul_835 = None
    squeeze_359: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_286, [0, 2, 3]);  getitem_286 = None
    mul_836: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_359, 1.0002835270768358);  squeeze_359 = None
    mul_837: "f32[432]" = torch.ops.aten.mul.Tensor(mul_836, 0.1);  mul_836 = None
    mul_838: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1136, 0.9)
    add_638: "f32[432]" = torch.ops.aten.add.Tensor(mul_837, mul_838);  mul_837 = mul_838 = None
    unsqueeze_476: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_466, -1)
    unsqueeze_477: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    mul_839: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_833, unsqueeze_477);  mul_833 = unsqueeze_477 = None
    unsqueeze_478: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_467, -1);  primals_467 = None
    unsqueeze_479: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    add_639: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_839, unsqueeze_479);  mul_839 = unsqueeze_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_24 = torch.ops.aten.max_pool2d_with_indices.default(add_624, [3, 3], [1, 1], [1, 1])
    getitem_288: "f32[8, 432, 21, 21]" = max_pool2d_with_indices_24[0]
    getitem_289: "i64[8, 432, 21, 21]" = max_pool2d_with_indices_24[1];  max_pool2d_with_indices_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_640: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_639, getitem_288);  add_639 = getitem_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_118: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_629)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_221: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_118, primals_468, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_222: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_221, primals_469, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_641: "i64[]" = torch.ops.aten.add.Tensor(primals_1140, 1)
    var_mean_120 = torch.ops.aten.var_mean.correction(convolution_222, [0, 2, 3], correction = 0, keepdim = True)
    getitem_290: "f32[1, 432, 1, 1]" = var_mean_120[0]
    getitem_291: "f32[1, 432, 1, 1]" = var_mean_120[1];  var_mean_120 = None
    add_642: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_290, 0.001)
    rsqrt_120: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_642);  add_642 = None
    sub_120: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_222, getitem_291)
    mul_840: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_120);  sub_120 = None
    squeeze_360: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_291, [0, 2, 3]);  getitem_291 = None
    squeeze_361: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_120, [0, 2, 3]);  rsqrt_120 = None
    mul_841: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_360, 0.1)
    mul_842: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1138, 0.9)
    add_643: "f32[432]" = torch.ops.aten.add.Tensor(mul_841, mul_842);  mul_841 = mul_842 = None
    squeeze_362: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_290, [0, 2, 3]);  getitem_290 = None
    mul_843: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_362, 1.0002835270768358);  squeeze_362 = None
    mul_844: "f32[432]" = torch.ops.aten.mul.Tensor(mul_843, 0.1);  mul_843 = None
    mul_845: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1139, 0.9)
    add_644: "f32[432]" = torch.ops.aten.add.Tensor(mul_844, mul_845);  mul_844 = mul_845 = None
    unsqueeze_480: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_470, -1)
    unsqueeze_481: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
    mul_846: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_840, unsqueeze_481);  mul_840 = unsqueeze_481 = None
    unsqueeze_482: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_471, -1);  primals_471 = None
    unsqueeze_483: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
    add_645: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_846, unsqueeze_483);  mul_846 = unsqueeze_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_119: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_645);  add_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_223: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_119, primals_472, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_224: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_223, primals_473, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_646: "i64[]" = torch.ops.aten.add.Tensor(primals_1143, 1)
    var_mean_121 = torch.ops.aten.var_mean.correction(convolution_224, [0, 2, 3], correction = 0, keepdim = True)
    getitem_292: "f32[1, 432, 1, 1]" = var_mean_121[0]
    getitem_293: "f32[1, 432, 1, 1]" = var_mean_121[1];  var_mean_121 = None
    add_647: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_292, 0.001)
    rsqrt_121: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_647);  add_647 = None
    sub_121: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_224, getitem_293)
    mul_847: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_121);  sub_121 = None
    squeeze_363: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_293, [0, 2, 3]);  getitem_293 = None
    squeeze_364: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_121, [0, 2, 3]);  rsqrt_121 = None
    mul_848: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_363, 0.1)
    mul_849: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1141, 0.9)
    add_648: "f32[432]" = torch.ops.aten.add.Tensor(mul_848, mul_849);  mul_848 = mul_849 = None
    squeeze_365: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_292, [0, 2, 3]);  getitem_292 = None
    mul_850: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_365, 1.0002835270768358);  squeeze_365 = None
    mul_851: "f32[432]" = torch.ops.aten.mul.Tensor(mul_850, 0.1);  mul_850 = None
    mul_852: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1142, 0.9)
    add_649: "f32[432]" = torch.ops.aten.add.Tensor(mul_851, mul_852);  mul_851 = mul_852 = None
    unsqueeze_484: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_474, -1)
    unsqueeze_485: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
    mul_853: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_847, unsqueeze_485);  mul_847 = unsqueeze_485 = None
    unsqueeze_486: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_475, -1);  primals_475 = None
    unsqueeze_487: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    add_650: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_853, unsqueeze_487);  mul_853 = unsqueeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_25 = torch.ops.aten.max_pool2d_with_indices.default(add_629, [3, 3], [1, 1], [1, 1])
    getitem_294: "f32[8, 432, 21, 21]" = max_pool2d_with_indices_25[0]
    getitem_295: "i64[8, 432, 21, 21]" = max_pool2d_with_indices_25[1];  max_pool2d_with_indices_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_651: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_650, getitem_294);  add_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_225: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_118, primals_476, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_226: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_225, primals_477, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_652: "i64[]" = torch.ops.aten.add.Tensor(primals_1146, 1)
    var_mean_122 = torch.ops.aten.var_mean.correction(convolution_226, [0, 2, 3], correction = 0, keepdim = True)
    getitem_296: "f32[1, 432, 1, 1]" = var_mean_122[0]
    getitem_297: "f32[1, 432, 1, 1]" = var_mean_122[1];  var_mean_122 = None
    add_653: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_296, 0.001)
    rsqrt_122: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_653);  add_653 = None
    sub_122: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_226, getitem_297)
    mul_854: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_122);  sub_122 = None
    squeeze_366: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_297, [0, 2, 3]);  getitem_297 = None
    squeeze_367: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_122, [0, 2, 3]);  rsqrt_122 = None
    mul_855: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_366, 0.1)
    mul_856: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1144, 0.9)
    add_654: "f32[432]" = torch.ops.aten.add.Tensor(mul_855, mul_856);  mul_855 = mul_856 = None
    squeeze_368: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_296, [0, 2, 3]);  getitem_296 = None
    mul_857: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_368, 1.0002835270768358);  squeeze_368 = None
    mul_858: "f32[432]" = torch.ops.aten.mul.Tensor(mul_857, 0.1);  mul_857 = None
    mul_859: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1145, 0.9)
    add_655: "f32[432]" = torch.ops.aten.add.Tensor(mul_858, mul_859);  mul_858 = mul_859 = None
    unsqueeze_488: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_478, -1)
    unsqueeze_489: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    mul_860: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_854, unsqueeze_489);  mul_854 = unsqueeze_489 = None
    unsqueeze_490: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_479, -1);  primals_479 = None
    unsqueeze_491: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
    add_656: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_860, unsqueeze_491);  mul_860 = unsqueeze_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_121: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_656);  add_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_227: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_121, primals_480, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_228: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_227, primals_481, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_657: "i64[]" = torch.ops.aten.add.Tensor(primals_1149, 1)
    var_mean_123 = torch.ops.aten.var_mean.correction(convolution_228, [0, 2, 3], correction = 0, keepdim = True)
    getitem_298: "f32[1, 432, 1, 1]" = var_mean_123[0]
    getitem_299: "f32[1, 432, 1, 1]" = var_mean_123[1];  var_mean_123 = None
    add_658: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_298, 0.001)
    rsqrt_123: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_658);  add_658 = None
    sub_123: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_228, getitem_299)
    mul_861: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_123, rsqrt_123);  sub_123 = None
    squeeze_369: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_299, [0, 2, 3]);  getitem_299 = None
    squeeze_370: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_123, [0, 2, 3]);  rsqrt_123 = None
    mul_862: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_369, 0.1)
    mul_863: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1147, 0.9)
    add_659: "f32[432]" = torch.ops.aten.add.Tensor(mul_862, mul_863);  mul_862 = mul_863 = None
    squeeze_371: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_298, [0, 2, 3]);  getitem_298 = None
    mul_864: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_371, 1.0002835270768358);  squeeze_371 = None
    mul_865: "f32[432]" = torch.ops.aten.mul.Tensor(mul_864, 0.1);  mul_864 = None
    mul_866: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1148, 0.9)
    add_660: "f32[432]" = torch.ops.aten.add.Tensor(mul_865, mul_866);  mul_865 = mul_866 = None
    unsqueeze_492: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_482, -1)
    unsqueeze_493: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
    mul_867: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_861, unsqueeze_493);  mul_861 = unsqueeze_493 = None
    unsqueeze_494: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_483, -1);  primals_483 = None
    unsqueeze_495: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
    add_661: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_867, unsqueeze_495);  mul_867 = unsqueeze_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_229: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_118, primals_484, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_230: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_229, primals_485, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_662: "i64[]" = torch.ops.aten.add.Tensor(primals_1152, 1)
    var_mean_124 = torch.ops.aten.var_mean.correction(convolution_230, [0, 2, 3], correction = 0, keepdim = True)
    getitem_300: "f32[1, 432, 1, 1]" = var_mean_124[0]
    getitem_301: "f32[1, 432, 1, 1]" = var_mean_124[1];  var_mean_124 = None
    add_663: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_300, 0.001)
    rsqrt_124: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_663);  add_663 = None
    sub_124: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_230, getitem_301)
    mul_868: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_124);  sub_124 = None
    squeeze_372: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_301, [0, 2, 3]);  getitem_301 = None
    squeeze_373: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_124, [0, 2, 3]);  rsqrt_124 = None
    mul_869: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_372, 0.1)
    mul_870: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1150, 0.9)
    add_664: "f32[432]" = torch.ops.aten.add.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
    squeeze_374: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_300, [0, 2, 3]);  getitem_300 = None
    mul_871: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_374, 1.0002835270768358);  squeeze_374 = None
    mul_872: "f32[432]" = torch.ops.aten.mul.Tensor(mul_871, 0.1);  mul_871 = None
    mul_873: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1151, 0.9)
    add_665: "f32[432]" = torch.ops.aten.add.Tensor(mul_872, mul_873);  mul_872 = mul_873 = None
    unsqueeze_496: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_486, -1)
    unsqueeze_497: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
    mul_874: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_868, unsqueeze_497);  mul_868 = unsqueeze_497 = None
    unsqueeze_498: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_487, -1);  primals_487 = None
    unsqueeze_499: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
    add_666: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_874, unsqueeze_499);  mul_874 = unsqueeze_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_123: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_666);  add_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_231: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_123, primals_488, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_232: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_231, primals_489, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_667: "i64[]" = torch.ops.aten.add.Tensor(primals_1155, 1)
    var_mean_125 = torch.ops.aten.var_mean.correction(convolution_232, [0, 2, 3], correction = 0, keepdim = True)
    getitem_302: "f32[1, 432, 1, 1]" = var_mean_125[0]
    getitem_303: "f32[1, 432, 1, 1]" = var_mean_125[1];  var_mean_125 = None
    add_668: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_302, 0.001)
    rsqrt_125: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_668);  add_668 = None
    sub_125: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_232, getitem_303)
    mul_875: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_125);  sub_125 = None
    squeeze_375: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_303, [0, 2, 3]);  getitem_303 = None
    squeeze_376: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_125, [0, 2, 3]);  rsqrt_125 = None
    mul_876: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_375, 0.1)
    mul_877: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1153, 0.9)
    add_669: "f32[432]" = torch.ops.aten.add.Tensor(mul_876, mul_877);  mul_876 = mul_877 = None
    squeeze_377: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_302, [0, 2, 3]);  getitem_302 = None
    mul_878: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_377, 1.0002835270768358);  squeeze_377 = None
    mul_879: "f32[432]" = torch.ops.aten.mul.Tensor(mul_878, 0.1);  mul_878 = None
    mul_880: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1154, 0.9)
    add_670: "f32[432]" = torch.ops.aten.add.Tensor(mul_879, mul_880);  mul_879 = mul_880 = None
    unsqueeze_500: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_490, -1)
    unsqueeze_501: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
    mul_881: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_875, unsqueeze_501);  mul_875 = unsqueeze_501 = None
    unsqueeze_502: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_491, -1);  primals_491 = None
    unsqueeze_503: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
    add_671: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_881, unsqueeze_503);  mul_881 = unsqueeze_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_672: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_661, add_671);  add_661 = add_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_124: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_233: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_124, primals_492, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_234: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_233, primals_493, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_673: "i64[]" = torch.ops.aten.add.Tensor(primals_1158, 1)
    var_mean_126 = torch.ops.aten.var_mean.correction(convolution_234, [0, 2, 3], correction = 0, keepdim = True)
    getitem_304: "f32[1, 432, 1, 1]" = var_mean_126[0]
    getitem_305: "f32[1, 432, 1, 1]" = var_mean_126[1];  var_mean_126 = None
    add_674: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_304, 0.001)
    rsqrt_126: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_674);  add_674 = None
    sub_126: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_234, getitem_305)
    mul_882: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_126);  sub_126 = None
    squeeze_378: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_305, [0, 2, 3]);  getitem_305 = None
    squeeze_379: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_126, [0, 2, 3]);  rsqrt_126 = None
    mul_883: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_378, 0.1)
    mul_884: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1156, 0.9)
    add_675: "f32[432]" = torch.ops.aten.add.Tensor(mul_883, mul_884);  mul_883 = mul_884 = None
    squeeze_380: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_304, [0, 2, 3]);  getitem_304 = None
    mul_885: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_380, 1.0002835270768358);  squeeze_380 = None
    mul_886: "f32[432]" = torch.ops.aten.mul.Tensor(mul_885, 0.1);  mul_885 = None
    mul_887: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1157, 0.9)
    add_676: "f32[432]" = torch.ops.aten.add.Tensor(mul_886, mul_887);  mul_886 = mul_887 = None
    unsqueeze_504: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_494, -1)
    unsqueeze_505: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
    mul_888: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_882, unsqueeze_505);  mul_882 = unsqueeze_505 = None
    unsqueeze_506: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_495, -1);  primals_495 = None
    unsqueeze_507: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
    add_677: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_888, unsqueeze_507);  mul_888 = unsqueeze_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_125: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_677);  add_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_235: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_125, primals_496, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_236: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_235, primals_497, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_678: "i64[]" = torch.ops.aten.add.Tensor(primals_1161, 1)
    var_mean_127 = torch.ops.aten.var_mean.correction(convolution_236, [0, 2, 3], correction = 0, keepdim = True)
    getitem_306: "f32[1, 432, 1, 1]" = var_mean_127[0]
    getitem_307: "f32[1, 432, 1, 1]" = var_mean_127[1];  var_mean_127 = None
    add_679: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_306, 0.001)
    rsqrt_127: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_679);  add_679 = None
    sub_127: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_236, getitem_307)
    mul_889: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_127);  sub_127 = None
    squeeze_381: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_307, [0, 2, 3]);  getitem_307 = None
    squeeze_382: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_127, [0, 2, 3]);  rsqrt_127 = None
    mul_890: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_381, 0.1)
    mul_891: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1159, 0.9)
    add_680: "f32[432]" = torch.ops.aten.add.Tensor(mul_890, mul_891);  mul_890 = mul_891 = None
    squeeze_383: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_306, [0, 2, 3]);  getitem_306 = None
    mul_892: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_383, 1.0002835270768358);  squeeze_383 = None
    mul_893: "f32[432]" = torch.ops.aten.mul.Tensor(mul_892, 0.1);  mul_892 = None
    mul_894: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1160, 0.9)
    add_681: "f32[432]" = torch.ops.aten.add.Tensor(mul_893, mul_894);  mul_893 = mul_894 = None
    unsqueeze_508: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_498, -1)
    unsqueeze_509: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
    mul_895: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_889, unsqueeze_509);  mul_889 = unsqueeze_509 = None
    unsqueeze_510: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_499, -1);  primals_499 = None
    unsqueeze_511: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
    add_682: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_895, unsqueeze_511);  mul_895 = unsqueeze_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_683: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_682, getitem_294);  add_682 = getitem_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_237: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_116, primals_500, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_238: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_237, primals_501, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_684: "i64[]" = torch.ops.aten.add.Tensor(primals_1164, 1)
    var_mean_128 = torch.ops.aten.var_mean.correction(convolution_238, [0, 2, 3], correction = 0, keepdim = True)
    getitem_310: "f32[1, 432, 1, 1]" = var_mean_128[0]
    getitem_311: "f32[1, 432, 1, 1]" = var_mean_128[1];  var_mean_128 = None
    add_685: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_310, 0.001)
    rsqrt_128: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_685);  add_685 = None
    sub_128: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_238, getitem_311)
    mul_896: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_128);  sub_128 = None
    squeeze_384: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_311, [0, 2, 3]);  getitem_311 = None
    squeeze_385: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_128, [0, 2, 3]);  rsqrt_128 = None
    mul_897: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_384, 0.1)
    mul_898: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1162, 0.9)
    add_686: "f32[432]" = torch.ops.aten.add.Tensor(mul_897, mul_898);  mul_897 = mul_898 = None
    squeeze_386: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_310, [0, 2, 3]);  getitem_310 = None
    mul_899: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_386, 1.0002835270768358);  squeeze_386 = None
    mul_900: "f32[432]" = torch.ops.aten.mul.Tensor(mul_899, 0.1);  mul_899 = None
    mul_901: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1163, 0.9)
    add_687: "f32[432]" = torch.ops.aten.add.Tensor(mul_900, mul_901);  mul_900 = mul_901 = None
    unsqueeze_512: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_502, -1)
    unsqueeze_513: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
    mul_902: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_896, unsqueeze_513);  mul_896 = unsqueeze_513 = None
    unsqueeze_514: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_503, -1);  primals_503 = None
    unsqueeze_515: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
    add_688: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_902, unsqueeze_515);  mul_902 = unsqueeze_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_127: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_688);  add_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_239: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_127, primals_504, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_240: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_239, primals_505, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_689: "i64[]" = torch.ops.aten.add.Tensor(primals_1167, 1)
    var_mean_129 = torch.ops.aten.var_mean.correction(convolution_240, [0, 2, 3], correction = 0, keepdim = True)
    getitem_312: "f32[1, 432, 1, 1]" = var_mean_129[0]
    getitem_313: "f32[1, 432, 1, 1]" = var_mean_129[1];  var_mean_129 = None
    add_690: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_312, 0.001)
    rsqrt_129: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_690);  add_690 = None
    sub_129: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_240, getitem_313)
    mul_903: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_129, rsqrt_129);  sub_129 = None
    squeeze_387: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_313, [0, 2, 3]);  getitem_313 = None
    squeeze_388: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_129, [0, 2, 3]);  rsqrt_129 = None
    mul_904: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_387, 0.1)
    mul_905: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1165, 0.9)
    add_691: "f32[432]" = torch.ops.aten.add.Tensor(mul_904, mul_905);  mul_904 = mul_905 = None
    squeeze_389: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_312, [0, 2, 3]);  getitem_312 = None
    mul_906: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_389, 1.0002835270768358);  squeeze_389 = None
    mul_907: "f32[432]" = torch.ops.aten.mul.Tensor(mul_906, 0.1);  mul_906 = None
    mul_908: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1166, 0.9)
    add_692: "f32[432]" = torch.ops.aten.add.Tensor(mul_907, mul_908);  mul_907 = mul_908 = None
    unsqueeze_516: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_506, -1)
    unsqueeze_517: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
    mul_909: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_903, unsqueeze_517);  mul_903 = unsqueeze_517 = None
    unsqueeze_518: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_507, -1);  primals_507 = None
    unsqueeze_519: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
    add_693: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_909, unsqueeze_519);  mul_909 = unsqueeze_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_694: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_693, add_629);  add_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat_11: "f32[8, 2160, 21, 21]" = torch.ops.aten.cat.default([add_640, add_651, add_672, add_683, add_694], 1);  add_640 = add_651 = add_672 = add_683 = add_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_241: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_115, primals_508, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_695: "i64[]" = torch.ops.aten.add.Tensor(primals_1170, 1)
    var_mean_130 = torch.ops.aten.var_mean.correction(convolution_241, [0, 2, 3], correction = 0, keepdim = True)
    getitem_314: "f32[1, 432, 1, 1]" = var_mean_130[0]
    getitem_315: "f32[1, 432, 1, 1]" = var_mean_130[1];  var_mean_130 = None
    add_696: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_314, 0.001)
    rsqrt_130: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_696);  add_696 = None
    sub_130: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_241, getitem_315)
    mul_910: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_130);  sub_130 = None
    squeeze_390: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_315, [0, 2, 3]);  getitem_315 = None
    squeeze_391: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_130, [0, 2, 3]);  rsqrt_130 = None
    mul_911: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_390, 0.1)
    mul_912: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1168, 0.9)
    add_697: "f32[432]" = torch.ops.aten.add.Tensor(mul_911, mul_912);  mul_911 = mul_912 = None
    squeeze_392: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_314, [0, 2, 3]);  getitem_314 = None
    mul_913: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_392, 1.0002835270768358);  squeeze_392 = None
    mul_914: "f32[432]" = torch.ops.aten.mul.Tensor(mul_913, 0.1);  mul_913 = None
    mul_915: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1169, 0.9)
    add_698: "f32[432]" = torch.ops.aten.add.Tensor(mul_914, mul_915);  mul_914 = mul_915 = None
    unsqueeze_520: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_509, -1)
    unsqueeze_521: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
    mul_916: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_910, unsqueeze_521);  mul_910 = unsqueeze_521 = None
    unsqueeze_522: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_510, -1);  primals_510 = None
    unsqueeze_523: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
    add_699: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_916, unsqueeze_523);  mul_916 = unsqueeze_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu_129: "f32[8, 2160, 21, 21]" = torch.ops.aten.relu.default(cat_11);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_242: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_129, primals_511, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_700: "i64[]" = torch.ops.aten.add.Tensor(primals_1173, 1)
    var_mean_131 = torch.ops.aten.var_mean.correction(convolution_242, [0, 2, 3], correction = 0, keepdim = True)
    getitem_316: "f32[1, 432, 1, 1]" = var_mean_131[0]
    getitem_317: "f32[1, 432, 1, 1]" = var_mean_131[1];  var_mean_131 = None
    add_701: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_316, 0.001)
    rsqrt_131: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_701);  add_701 = None
    sub_131: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_242, getitem_317)
    mul_917: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_131);  sub_131 = None
    squeeze_393: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_317, [0, 2, 3]);  getitem_317 = None
    squeeze_394: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_131, [0, 2, 3]);  rsqrt_131 = None
    mul_918: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_393, 0.1)
    mul_919: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1171, 0.9)
    add_702: "f32[432]" = torch.ops.aten.add.Tensor(mul_918, mul_919);  mul_918 = mul_919 = None
    squeeze_395: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_316, [0, 2, 3]);  getitem_316 = None
    mul_920: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_395, 1.0002835270768358);  squeeze_395 = None
    mul_921: "f32[432]" = torch.ops.aten.mul.Tensor(mul_920, 0.1);  mul_920 = None
    mul_922: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1172, 0.9)
    add_703: "f32[432]" = torch.ops.aten.add.Tensor(mul_921, mul_922);  mul_921 = mul_922 = None
    unsqueeze_524: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_512, -1)
    unsqueeze_525: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
    mul_923: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_917, unsqueeze_525);  mul_917 = unsqueeze_525 = None
    unsqueeze_526: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_513, -1);  primals_513 = None
    unsqueeze_527: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
    add_704: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_923, unsqueeze_527);  mul_923 = unsqueeze_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_130: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_699)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_243: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_130, primals_514, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_244: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_243, primals_515, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_705: "i64[]" = torch.ops.aten.add.Tensor(primals_1176, 1)
    var_mean_132 = torch.ops.aten.var_mean.correction(convolution_244, [0, 2, 3], correction = 0, keepdim = True)
    getitem_318: "f32[1, 432, 1, 1]" = var_mean_132[0]
    getitem_319: "f32[1, 432, 1, 1]" = var_mean_132[1];  var_mean_132 = None
    add_706: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_318, 0.001)
    rsqrt_132: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_706);  add_706 = None
    sub_132: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_244, getitem_319)
    mul_924: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_132, rsqrt_132);  sub_132 = None
    squeeze_396: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_319, [0, 2, 3]);  getitem_319 = None
    squeeze_397: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_132, [0, 2, 3]);  rsqrt_132 = None
    mul_925: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_396, 0.1)
    mul_926: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1174, 0.9)
    add_707: "f32[432]" = torch.ops.aten.add.Tensor(mul_925, mul_926);  mul_925 = mul_926 = None
    squeeze_398: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_318, [0, 2, 3]);  getitem_318 = None
    mul_927: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_398, 1.0002835270768358);  squeeze_398 = None
    mul_928: "f32[432]" = torch.ops.aten.mul.Tensor(mul_927, 0.1);  mul_927 = None
    mul_929: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1175, 0.9)
    add_708: "f32[432]" = torch.ops.aten.add.Tensor(mul_928, mul_929);  mul_928 = mul_929 = None
    unsqueeze_528: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_516, -1)
    unsqueeze_529: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
    mul_930: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_924, unsqueeze_529);  mul_924 = unsqueeze_529 = None
    unsqueeze_530: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_517, -1);  primals_517 = None
    unsqueeze_531: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
    add_709: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_930, unsqueeze_531);  mul_930 = unsqueeze_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_131: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_709);  add_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_245: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_131, primals_518, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_246: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_245, primals_519, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_710: "i64[]" = torch.ops.aten.add.Tensor(primals_1179, 1)
    var_mean_133 = torch.ops.aten.var_mean.correction(convolution_246, [0, 2, 3], correction = 0, keepdim = True)
    getitem_320: "f32[1, 432, 1, 1]" = var_mean_133[0]
    getitem_321: "f32[1, 432, 1, 1]" = var_mean_133[1];  var_mean_133 = None
    add_711: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_320, 0.001)
    rsqrt_133: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_711);  add_711 = None
    sub_133: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_246, getitem_321)
    mul_931: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_133);  sub_133 = None
    squeeze_399: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_321, [0, 2, 3]);  getitem_321 = None
    squeeze_400: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_133, [0, 2, 3]);  rsqrt_133 = None
    mul_932: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_399, 0.1)
    mul_933: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1177, 0.9)
    add_712: "f32[432]" = torch.ops.aten.add.Tensor(mul_932, mul_933);  mul_932 = mul_933 = None
    squeeze_401: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_320, [0, 2, 3]);  getitem_320 = None
    mul_934: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_401, 1.0002835270768358);  squeeze_401 = None
    mul_935: "f32[432]" = torch.ops.aten.mul.Tensor(mul_934, 0.1);  mul_934 = None
    mul_936: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1178, 0.9)
    add_713: "f32[432]" = torch.ops.aten.add.Tensor(mul_935, mul_936);  mul_935 = mul_936 = None
    unsqueeze_532: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_520, -1)
    unsqueeze_533: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
    mul_937: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_931, unsqueeze_533);  mul_931 = unsqueeze_533 = None
    unsqueeze_534: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_521, -1);  primals_521 = None
    unsqueeze_535: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
    add_714: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_937, unsqueeze_535);  mul_937 = unsqueeze_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_27 = torch.ops.aten.max_pool2d_with_indices.default(add_699, [3, 3], [1, 1], [1, 1])
    getitem_322: "f32[8, 432, 21, 21]" = max_pool2d_with_indices_27[0]
    getitem_323: "i64[8, 432, 21, 21]" = max_pool2d_with_indices_27[1];  max_pool2d_with_indices_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_715: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_714, getitem_322);  add_714 = getitem_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_132: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_704)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_247: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_132, primals_522, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_248: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_247, primals_523, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_716: "i64[]" = torch.ops.aten.add.Tensor(primals_1182, 1)
    var_mean_134 = torch.ops.aten.var_mean.correction(convolution_248, [0, 2, 3], correction = 0, keepdim = True)
    getitem_324: "f32[1, 432, 1, 1]" = var_mean_134[0]
    getitem_325: "f32[1, 432, 1, 1]" = var_mean_134[1];  var_mean_134 = None
    add_717: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_324, 0.001)
    rsqrt_134: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_717);  add_717 = None
    sub_134: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_248, getitem_325)
    mul_938: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_134);  sub_134 = None
    squeeze_402: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_325, [0, 2, 3]);  getitem_325 = None
    squeeze_403: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_134, [0, 2, 3]);  rsqrt_134 = None
    mul_939: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_402, 0.1)
    mul_940: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1180, 0.9)
    add_718: "f32[432]" = torch.ops.aten.add.Tensor(mul_939, mul_940);  mul_939 = mul_940 = None
    squeeze_404: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_324, [0, 2, 3]);  getitem_324 = None
    mul_941: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_404, 1.0002835270768358);  squeeze_404 = None
    mul_942: "f32[432]" = torch.ops.aten.mul.Tensor(mul_941, 0.1);  mul_941 = None
    mul_943: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1181, 0.9)
    add_719: "f32[432]" = torch.ops.aten.add.Tensor(mul_942, mul_943);  mul_942 = mul_943 = None
    unsqueeze_536: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_524, -1)
    unsqueeze_537: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
    mul_944: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_938, unsqueeze_537);  mul_938 = unsqueeze_537 = None
    unsqueeze_538: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_525, -1);  primals_525 = None
    unsqueeze_539: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
    add_720: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_944, unsqueeze_539);  mul_944 = unsqueeze_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_133: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_720);  add_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_249: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_133, primals_526, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_250: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_249, primals_527, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_721: "i64[]" = torch.ops.aten.add.Tensor(primals_1185, 1)
    var_mean_135 = torch.ops.aten.var_mean.correction(convolution_250, [0, 2, 3], correction = 0, keepdim = True)
    getitem_326: "f32[1, 432, 1, 1]" = var_mean_135[0]
    getitem_327: "f32[1, 432, 1, 1]" = var_mean_135[1];  var_mean_135 = None
    add_722: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_326, 0.001)
    rsqrt_135: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_722);  add_722 = None
    sub_135: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_250, getitem_327)
    mul_945: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_135, rsqrt_135);  sub_135 = None
    squeeze_405: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_327, [0, 2, 3]);  getitem_327 = None
    squeeze_406: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_135, [0, 2, 3]);  rsqrt_135 = None
    mul_946: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_405, 0.1)
    mul_947: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1183, 0.9)
    add_723: "f32[432]" = torch.ops.aten.add.Tensor(mul_946, mul_947);  mul_946 = mul_947 = None
    squeeze_407: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_326, [0, 2, 3]);  getitem_326 = None
    mul_948: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_407, 1.0002835270768358);  squeeze_407 = None
    mul_949: "f32[432]" = torch.ops.aten.mul.Tensor(mul_948, 0.1);  mul_948 = None
    mul_950: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1184, 0.9)
    add_724: "f32[432]" = torch.ops.aten.add.Tensor(mul_949, mul_950);  mul_949 = mul_950 = None
    unsqueeze_540: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_528, -1)
    unsqueeze_541: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
    mul_951: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_945, unsqueeze_541);  mul_945 = unsqueeze_541 = None
    unsqueeze_542: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_529, -1);  primals_529 = None
    unsqueeze_543: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
    add_725: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_951, unsqueeze_543);  mul_951 = unsqueeze_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_28 = torch.ops.aten.max_pool2d_with_indices.default(add_704, [3, 3], [1, 1], [1, 1])
    getitem_328: "f32[8, 432, 21, 21]" = max_pool2d_with_indices_28[0]
    getitem_329: "i64[8, 432, 21, 21]" = max_pool2d_with_indices_28[1];  max_pool2d_with_indices_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_726: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_725, getitem_328);  add_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_251: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_132, primals_530, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_252: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_251, primals_531, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_727: "i64[]" = torch.ops.aten.add.Tensor(primals_1188, 1)
    var_mean_136 = torch.ops.aten.var_mean.correction(convolution_252, [0, 2, 3], correction = 0, keepdim = True)
    getitem_330: "f32[1, 432, 1, 1]" = var_mean_136[0]
    getitem_331: "f32[1, 432, 1, 1]" = var_mean_136[1];  var_mean_136 = None
    add_728: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_330, 0.001)
    rsqrt_136: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_728);  add_728 = None
    sub_136: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_252, getitem_331)
    mul_952: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_136, rsqrt_136);  sub_136 = None
    squeeze_408: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_331, [0, 2, 3]);  getitem_331 = None
    squeeze_409: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_136, [0, 2, 3]);  rsqrt_136 = None
    mul_953: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_408, 0.1)
    mul_954: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1186, 0.9)
    add_729: "f32[432]" = torch.ops.aten.add.Tensor(mul_953, mul_954);  mul_953 = mul_954 = None
    squeeze_410: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_330, [0, 2, 3]);  getitem_330 = None
    mul_955: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_410, 1.0002835270768358);  squeeze_410 = None
    mul_956: "f32[432]" = torch.ops.aten.mul.Tensor(mul_955, 0.1);  mul_955 = None
    mul_957: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1187, 0.9)
    add_730: "f32[432]" = torch.ops.aten.add.Tensor(mul_956, mul_957);  mul_956 = mul_957 = None
    unsqueeze_544: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_532, -1)
    unsqueeze_545: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
    mul_958: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_952, unsqueeze_545);  mul_952 = unsqueeze_545 = None
    unsqueeze_546: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_533, -1);  primals_533 = None
    unsqueeze_547: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
    add_731: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_958, unsqueeze_547);  mul_958 = unsqueeze_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_135: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_731);  add_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_253: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_135, primals_534, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_254: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_253, primals_535, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_732: "i64[]" = torch.ops.aten.add.Tensor(primals_1191, 1)
    var_mean_137 = torch.ops.aten.var_mean.correction(convolution_254, [0, 2, 3], correction = 0, keepdim = True)
    getitem_332: "f32[1, 432, 1, 1]" = var_mean_137[0]
    getitem_333: "f32[1, 432, 1, 1]" = var_mean_137[1];  var_mean_137 = None
    add_733: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_332, 0.001)
    rsqrt_137: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_733);  add_733 = None
    sub_137: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_254, getitem_333)
    mul_959: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_137);  sub_137 = None
    squeeze_411: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_333, [0, 2, 3]);  getitem_333 = None
    squeeze_412: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_137, [0, 2, 3]);  rsqrt_137 = None
    mul_960: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_411, 0.1)
    mul_961: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1189, 0.9)
    add_734: "f32[432]" = torch.ops.aten.add.Tensor(mul_960, mul_961);  mul_960 = mul_961 = None
    squeeze_413: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_332, [0, 2, 3]);  getitem_332 = None
    mul_962: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_413, 1.0002835270768358);  squeeze_413 = None
    mul_963: "f32[432]" = torch.ops.aten.mul.Tensor(mul_962, 0.1);  mul_962 = None
    mul_964: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1190, 0.9)
    add_735: "f32[432]" = torch.ops.aten.add.Tensor(mul_963, mul_964);  mul_963 = mul_964 = None
    unsqueeze_548: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_536, -1)
    unsqueeze_549: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
    mul_965: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_959, unsqueeze_549);  mul_959 = unsqueeze_549 = None
    unsqueeze_550: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_537, -1);  primals_537 = None
    unsqueeze_551: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
    add_736: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_965, unsqueeze_551);  mul_965 = unsqueeze_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_255: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_132, primals_538, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_256: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_255, primals_539, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_737: "i64[]" = torch.ops.aten.add.Tensor(primals_1194, 1)
    var_mean_138 = torch.ops.aten.var_mean.correction(convolution_256, [0, 2, 3], correction = 0, keepdim = True)
    getitem_334: "f32[1, 432, 1, 1]" = var_mean_138[0]
    getitem_335: "f32[1, 432, 1, 1]" = var_mean_138[1];  var_mean_138 = None
    add_738: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_334, 0.001)
    rsqrt_138: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_738);  add_738 = None
    sub_138: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_256, getitem_335)
    mul_966: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_138, rsqrt_138);  sub_138 = None
    squeeze_414: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_335, [0, 2, 3]);  getitem_335 = None
    squeeze_415: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_138, [0, 2, 3]);  rsqrt_138 = None
    mul_967: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_414, 0.1)
    mul_968: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1192, 0.9)
    add_739: "f32[432]" = torch.ops.aten.add.Tensor(mul_967, mul_968);  mul_967 = mul_968 = None
    squeeze_416: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_334, [0, 2, 3]);  getitem_334 = None
    mul_969: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_416, 1.0002835270768358);  squeeze_416 = None
    mul_970: "f32[432]" = torch.ops.aten.mul.Tensor(mul_969, 0.1);  mul_969 = None
    mul_971: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1193, 0.9)
    add_740: "f32[432]" = torch.ops.aten.add.Tensor(mul_970, mul_971);  mul_970 = mul_971 = None
    unsqueeze_552: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_540, -1)
    unsqueeze_553: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
    mul_972: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_966, unsqueeze_553);  mul_966 = unsqueeze_553 = None
    unsqueeze_554: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_541, -1);  primals_541 = None
    unsqueeze_555: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
    add_741: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_972, unsqueeze_555);  mul_972 = unsqueeze_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_137: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_741);  add_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_257: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_137, primals_542, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_258: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_257, primals_543, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_742: "i64[]" = torch.ops.aten.add.Tensor(primals_1197, 1)
    var_mean_139 = torch.ops.aten.var_mean.correction(convolution_258, [0, 2, 3], correction = 0, keepdim = True)
    getitem_336: "f32[1, 432, 1, 1]" = var_mean_139[0]
    getitem_337: "f32[1, 432, 1, 1]" = var_mean_139[1];  var_mean_139 = None
    add_743: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_336, 0.001)
    rsqrt_139: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_743);  add_743 = None
    sub_139: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_258, getitem_337)
    mul_973: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_139, rsqrt_139);  sub_139 = None
    squeeze_417: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_337, [0, 2, 3]);  getitem_337 = None
    squeeze_418: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_139, [0, 2, 3]);  rsqrt_139 = None
    mul_974: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_417, 0.1)
    mul_975: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1195, 0.9)
    add_744: "f32[432]" = torch.ops.aten.add.Tensor(mul_974, mul_975);  mul_974 = mul_975 = None
    squeeze_419: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_336, [0, 2, 3]);  getitem_336 = None
    mul_976: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_419, 1.0002835270768358);  squeeze_419 = None
    mul_977: "f32[432]" = torch.ops.aten.mul.Tensor(mul_976, 0.1);  mul_976 = None
    mul_978: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1196, 0.9)
    add_745: "f32[432]" = torch.ops.aten.add.Tensor(mul_977, mul_978);  mul_977 = mul_978 = None
    unsqueeze_556: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_544, -1)
    unsqueeze_557: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
    mul_979: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_973, unsqueeze_557);  mul_973 = unsqueeze_557 = None
    unsqueeze_558: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_545, -1);  primals_545 = None
    unsqueeze_559: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
    add_746: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_979, unsqueeze_559);  mul_979 = unsqueeze_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_747: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_736, add_746);  add_736 = add_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_138: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_747)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_259: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_138, primals_546, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_260: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_259, primals_547, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_748: "i64[]" = torch.ops.aten.add.Tensor(primals_1200, 1)
    var_mean_140 = torch.ops.aten.var_mean.correction(convolution_260, [0, 2, 3], correction = 0, keepdim = True)
    getitem_338: "f32[1, 432, 1, 1]" = var_mean_140[0]
    getitem_339: "f32[1, 432, 1, 1]" = var_mean_140[1];  var_mean_140 = None
    add_749: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_338, 0.001)
    rsqrt_140: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_749);  add_749 = None
    sub_140: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_260, getitem_339)
    mul_980: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_140);  sub_140 = None
    squeeze_420: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_339, [0, 2, 3]);  getitem_339 = None
    squeeze_421: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_140, [0, 2, 3]);  rsqrt_140 = None
    mul_981: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_420, 0.1)
    mul_982: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1198, 0.9)
    add_750: "f32[432]" = torch.ops.aten.add.Tensor(mul_981, mul_982);  mul_981 = mul_982 = None
    squeeze_422: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_338, [0, 2, 3]);  getitem_338 = None
    mul_983: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_422, 1.0002835270768358);  squeeze_422 = None
    mul_984: "f32[432]" = torch.ops.aten.mul.Tensor(mul_983, 0.1);  mul_983 = None
    mul_985: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1199, 0.9)
    add_751: "f32[432]" = torch.ops.aten.add.Tensor(mul_984, mul_985);  mul_984 = mul_985 = None
    unsqueeze_560: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_548, -1)
    unsqueeze_561: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
    mul_986: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_980, unsqueeze_561);  mul_980 = unsqueeze_561 = None
    unsqueeze_562: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_549, -1);  primals_549 = None
    unsqueeze_563: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
    add_752: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_986, unsqueeze_563);  mul_986 = unsqueeze_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_139: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_752);  add_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_261: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_139, primals_550, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_262: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_261, primals_551, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_753: "i64[]" = torch.ops.aten.add.Tensor(primals_1203, 1)
    var_mean_141 = torch.ops.aten.var_mean.correction(convolution_262, [0, 2, 3], correction = 0, keepdim = True)
    getitem_340: "f32[1, 432, 1, 1]" = var_mean_141[0]
    getitem_341: "f32[1, 432, 1, 1]" = var_mean_141[1];  var_mean_141 = None
    add_754: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_340, 0.001)
    rsqrt_141: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_754);  add_754 = None
    sub_141: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_262, getitem_341)
    mul_987: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_141, rsqrt_141);  sub_141 = None
    squeeze_423: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_341, [0, 2, 3]);  getitem_341 = None
    squeeze_424: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_141, [0, 2, 3]);  rsqrt_141 = None
    mul_988: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_423, 0.1)
    mul_989: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1201, 0.9)
    add_755: "f32[432]" = torch.ops.aten.add.Tensor(mul_988, mul_989);  mul_988 = mul_989 = None
    squeeze_425: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_340, [0, 2, 3]);  getitem_340 = None
    mul_990: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_425, 1.0002835270768358);  squeeze_425 = None
    mul_991: "f32[432]" = torch.ops.aten.mul.Tensor(mul_990, 0.1);  mul_990 = None
    mul_992: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1202, 0.9)
    add_756: "f32[432]" = torch.ops.aten.add.Tensor(mul_991, mul_992);  mul_991 = mul_992 = None
    unsqueeze_564: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_552, -1)
    unsqueeze_565: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
    mul_993: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_987, unsqueeze_565);  mul_987 = unsqueeze_565 = None
    unsqueeze_566: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_553, -1);  primals_553 = None
    unsqueeze_567: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
    add_757: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_993, unsqueeze_567);  mul_993 = unsqueeze_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_758: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_757, getitem_328);  add_757 = getitem_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_263: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_130, primals_554, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_264: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_263, primals_555, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_759: "i64[]" = torch.ops.aten.add.Tensor(primals_1206, 1)
    var_mean_142 = torch.ops.aten.var_mean.correction(convolution_264, [0, 2, 3], correction = 0, keepdim = True)
    getitem_344: "f32[1, 432, 1, 1]" = var_mean_142[0]
    getitem_345: "f32[1, 432, 1, 1]" = var_mean_142[1];  var_mean_142 = None
    add_760: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_344, 0.001)
    rsqrt_142: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_760);  add_760 = None
    sub_142: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_264, getitem_345)
    mul_994: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_142, rsqrt_142);  sub_142 = None
    squeeze_426: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_345, [0, 2, 3]);  getitem_345 = None
    squeeze_427: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_142, [0, 2, 3]);  rsqrt_142 = None
    mul_995: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_426, 0.1)
    mul_996: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1204, 0.9)
    add_761: "f32[432]" = torch.ops.aten.add.Tensor(mul_995, mul_996);  mul_995 = mul_996 = None
    squeeze_428: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_344, [0, 2, 3]);  getitem_344 = None
    mul_997: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_428, 1.0002835270768358);  squeeze_428 = None
    mul_998: "f32[432]" = torch.ops.aten.mul.Tensor(mul_997, 0.1);  mul_997 = None
    mul_999: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1205, 0.9)
    add_762: "f32[432]" = torch.ops.aten.add.Tensor(mul_998, mul_999);  mul_998 = mul_999 = None
    unsqueeze_568: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_556, -1)
    unsqueeze_569: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
    mul_1000: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_994, unsqueeze_569);  mul_994 = unsqueeze_569 = None
    unsqueeze_570: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_557, -1);  primals_557 = None
    unsqueeze_571: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
    add_763: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_1000, unsqueeze_571);  mul_1000 = unsqueeze_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_141: "f32[8, 432, 21, 21]" = torch.ops.aten.relu.default(add_763);  add_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_265: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(relu_141, primals_558, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_266: "f32[8, 432, 21, 21]" = torch.ops.aten.convolution.default(convolution_265, primals_559, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_764: "i64[]" = torch.ops.aten.add.Tensor(primals_1209, 1)
    var_mean_143 = torch.ops.aten.var_mean.correction(convolution_266, [0, 2, 3], correction = 0, keepdim = True)
    getitem_346: "f32[1, 432, 1, 1]" = var_mean_143[0]
    getitem_347: "f32[1, 432, 1, 1]" = var_mean_143[1];  var_mean_143 = None
    add_765: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_346, 0.001)
    rsqrt_143: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_765);  add_765 = None
    sub_143: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_266, getitem_347)
    mul_1001: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_143, rsqrt_143);  sub_143 = None
    squeeze_429: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_347, [0, 2, 3]);  getitem_347 = None
    squeeze_430: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_143, [0, 2, 3]);  rsqrt_143 = None
    mul_1002: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_429, 0.1)
    mul_1003: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1207, 0.9)
    add_766: "f32[432]" = torch.ops.aten.add.Tensor(mul_1002, mul_1003);  mul_1002 = mul_1003 = None
    squeeze_431: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_346, [0, 2, 3]);  getitem_346 = None
    mul_1004: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_431, 1.0002835270768358);  squeeze_431 = None
    mul_1005: "f32[432]" = torch.ops.aten.mul.Tensor(mul_1004, 0.1);  mul_1004 = None
    mul_1006: "f32[432]" = torch.ops.aten.mul.Tensor(primals_1208, 0.9)
    add_767: "f32[432]" = torch.ops.aten.add.Tensor(mul_1005, mul_1006);  mul_1005 = mul_1006 = None
    unsqueeze_572: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_560, -1)
    unsqueeze_573: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
    mul_1007: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(mul_1001, unsqueeze_573);  mul_1001 = unsqueeze_573 = None
    unsqueeze_574: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_561, -1);  primals_561 = None
    unsqueeze_575: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
    add_768: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(mul_1007, unsqueeze_575);  mul_1007 = unsqueeze_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_769: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_768, add_704);  add_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat_12: "f32[8, 2160, 21, 21]" = torch.ops.aten.cat.default([add_715, add_726, add_747, add_758, add_769], 1);  add_715 = add_726 = add_747 = add_758 = add_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_267: "f32[8, 864, 21, 21]" = torch.ops.aten.convolution.default(relu_129, primals_562, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_770: "i64[]" = torch.ops.aten.add.Tensor(primals_1212, 1)
    var_mean_144 = torch.ops.aten.var_mean.correction(convolution_267, [0, 2, 3], correction = 0, keepdim = True)
    getitem_348: "f32[1, 864, 1, 1]" = var_mean_144[0]
    getitem_349: "f32[1, 864, 1, 1]" = var_mean_144[1];  var_mean_144 = None
    add_771: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_348, 0.001)
    rsqrt_144: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_771);  add_771 = None
    sub_144: "f32[8, 864, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_267, getitem_349)
    mul_1008: "f32[8, 864, 21, 21]" = torch.ops.aten.mul.Tensor(sub_144, rsqrt_144);  sub_144 = None
    squeeze_432: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_349, [0, 2, 3]);  getitem_349 = None
    squeeze_433: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_144, [0, 2, 3]);  rsqrt_144 = None
    mul_1009: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_432, 0.1)
    mul_1010: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1210, 0.9)
    add_772: "f32[864]" = torch.ops.aten.add.Tensor(mul_1009, mul_1010);  mul_1009 = mul_1010 = None
    squeeze_434: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_348, [0, 2, 3]);  getitem_348 = None
    mul_1011: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_434, 1.0002835270768358);  squeeze_434 = None
    mul_1012: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1011, 0.1);  mul_1011 = None
    mul_1013: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1211, 0.9)
    add_773: "f32[864]" = torch.ops.aten.add.Tensor(mul_1012, mul_1013);  mul_1012 = mul_1013 = None
    unsqueeze_576: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_563, -1)
    unsqueeze_577: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
    mul_1014: "f32[8, 864, 21, 21]" = torch.ops.aten.mul.Tensor(mul_1008, unsqueeze_577);  mul_1008 = unsqueeze_577 = None
    unsqueeze_578: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_564, -1);  primals_564 = None
    unsqueeze_579: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
    add_774: "f32[8, 864, 21, 21]" = torch.ops.aten.add.Tensor(mul_1014, unsqueeze_579);  mul_1014 = unsqueeze_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu_143: "f32[8, 2160, 21, 21]" = torch.ops.aten.relu.default(cat_12);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_268: "f32[8, 864, 21, 21]" = torch.ops.aten.convolution.default(relu_143, primals_565, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_775: "i64[]" = torch.ops.aten.add.Tensor(primals_1215, 1)
    var_mean_145 = torch.ops.aten.var_mean.correction(convolution_268, [0, 2, 3], correction = 0, keepdim = True)
    getitem_350: "f32[1, 864, 1, 1]" = var_mean_145[0]
    getitem_351: "f32[1, 864, 1, 1]" = var_mean_145[1];  var_mean_145 = None
    add_776: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_350, 0.001)
    rsqrt_145: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_776);  add_776 = None
    sub_145: "f32[8, 864, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_268, getitem_351)
    mul_1015: "f32[8, 864, 21, 21]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_145);  sub_145 = None
    squeeze_435: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_351, [0, 2, 3]);  getitem_351 = None
    squeeze_436: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_145, [0, 2, 3]);  rsqrt_145 = None
    mul_1016: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_435, 0.1)
    mul_1017: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1213, 0.9)
    add_777: "f32[864]" = torch.ops.aten.add.Tensor(mul_1016, mul_1017);  mul_1016 = mul_1017 = None
    squeeze_437: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_350, [0, 2, 3]);  getitem_350 = None
    mul_1018: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_437, 1.0002835270768358);  squeeze_437 = None
    mul_1019: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1018, 0.1);  mul_1018 = None
    mul_1020: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1214, 0.9)
    add_778: "f32[864]" = torch.ops.aten.add.Tensor(mul_1019, mul_1020);  mul_1019 = mul_1020 = None
    unsqueeze_580: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_566, -1)
    unsqueeze_581: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
    mul_1021: "f32[8, 864, 21, 21]" = torch.ops.aten.mul.Tensor(mul_1015, unsqueeze_581);  mul_1015 = unsqueeze_581 = None
    unsqueeze_582: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_567, -1);  primals_567 = None
    unsqueeze_583: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
    add_779: "f32[8, 864, 21, 21]" = torch.ops.aten.add.Tensor(mul_1021, unsqueeze_583);  mul_1021 = unsqueeze_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_144: "f32[8, 864, 21, 21]" = torch.ops.aten.relu.default(add_774)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_30: "f32[8, 864, 25, 25]" = torch.ops.aten.constant_pad_nd.default(relu_144, [2, 2, 2, 2], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_269: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(constant_pad_nd_30, primals_21, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_270: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_269, primals_568, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_780: "i64[]" = torch.ops.aten.add.Tensor(primals_1218, 1)
    var_mean_146 = torch.ops.aten.var_mean.correction(convolution_270, [0, 2, 3], correction = 0, keepdim = True)
    getitem_352: "f32[1, 864, 1, 1]" = var_mean_146[0]
    getitem_353: "f32[1, 864, 1, 1]" = var_mean_146[1];  var_mean_146 = None
    add_781: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_352, 0.001)
    rsqrt_146: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_781);  add_781 = None
    sub_146: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_270, getitem_353)
    mul_1022: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_146, rsqrt_146);  sub_146 = None
    squeeze_438: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_353, [0, 2, 3]);  getitem_353 = None
    squeeze_439: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_146, [0, 2, 3]);  rsqrt_146 = None
    mul_1023: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_438, 0.1)
    mul_1024: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1216, 0.9)
    add_782: "f32[864]" = torch.ops.aten.add.Tensor(mul_1023, mul_1024);  mul_1023 = mul_1024 = None
    squeeze_440: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_352, [0, 2, 3]);  getitem_352 = None
    mul_1025: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_440, 1.001034126163392);  squeeze_440 = None
    mul_1026: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1025, 0.1);  mul_1025 = None
    mul_1027: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1217, 0.9)
    add_783: "f32[864]" = torch.ops.aten.add.Tensor(mul_1026, mul_1027);  mul_1026 = mul_1027 = None
    unsqueeze_584: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_569, -1)
    unsqueeze_585: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
    mul_1028: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1022, unsqueeze_585);  mul_1022 = unsqueeze_585 = None
    unsqueeze_586: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_570, -1);  primals_570 = None
    unsqueeze_587: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
    add_784: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1028, unsqueeze_587);  mul_1028 = unsqueeze_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_145: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_784);  add_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_271: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_145, primals_571, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_272: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_271, primals_572, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_785: "i64[]" = torch.ops.aten.add.Tensor(primals_1221, 1)
    var_mean_147 = torch.ops.aten.var_mean.correction(convolution_272, [0, 2, 3], correction = 0, keepdim = True)
    getitem_354: "f32[1, 864, 1, 1]" = var_mean_147[0]
    getitem_355: "f32[1, 864, 1, 1]" = var_mean_147[1];  var_mean_147 = None
    add_786: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_354, 0.001)
    rsqrt_147: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_786);  add_786 = None
    sub_147: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_272, getitem_355)
    mul_1029: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_147, rsqrt_147);  sub_147 = None
    squeeze_441: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_355, [0, 2, 3]);  getitem_355 = None
    squeeze_442: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_147, [0, 2, 3]);  rsqrt_147 = None
    mul_1030: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_441, 0.1)
    mul_1031: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1219, 0.9)
    add_787: "f32[864]" = torch.ops.aten.add.Tensor(mul_1030, mul_1031);  mul_1030 = mul_1031 = None
    squeeze_443: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_354, [0, 2, 3]);  getitem_354 = None
    mul_1032: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_443, 1.001034126163392);  squeeze_443 = None
    mul_1033: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1032, 0.1);  mul_1032 = None
    mul_1034: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1220, 0.9)
    add_788: "f32[864]" = torch.ops.aten.add.Tensor(mul_1033, mul_1034);  mul_1033 = mul_1034 = None
    unsqueeze_588: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_573, -1)
    unsqueeze_589: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
    mul_1035: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1029, unsqueeze_589);  mul_1029 = unsqueeze_589 = None
    unsqueeze_590: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_574, -1);  primals_574 = None
    unsqueeze_591: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
    add_789: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1035, unsqueeze_591);  mul_1035 = unsqueeze_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_31: "f32[8, 864, 23, 23]" = torch.ops.aten.constant_pad_nd.default(add_774, [1, 1, 1, 1], -inf);  add_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_30 = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd_31, [3, 3], [2, 2])
    getitem_356: "f32[8, 864, 11, 11]" = max_pool2d_with_indices_30[0]
    getitem_357: "i64[8, 864, 11, 11]" = max_pool2d_with_indices_30[1];  max_pool2d_with_indices_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_790: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_789, getitem_356);  add_789 = getitem_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_146: "f32[8, 864, 21, 21]" = torch.ops.aten.relu.default(add_779)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_32: "f32[8, 864, 27, 27]" = torch.ops.aten.constant_pad_nd.default(relu_146, [3, 3, 3, 3], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_273: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(constant_pad_nd_32, primals_22, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_274: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_273, primals_575, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_791: "i64[]" = torch.ops.aten.add.Tensor(primals_1224, 1)
    var_mean_148 = torch.ops.aten.var_mean.correction(convolution_274, [0, 2, 3], correction = 0, keepdim = True)
    getitem_358: "f32[1, 864, 1, 1]" = var_mean_148[0]
    getitem_359: "f32[1, 864, 1, 1]" = var_mean_148[1];  var_mean_148 = None
    add_792: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_358, 0.001)
    rsqrt_148: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_792);  add_792 = None
    sub_148: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_274, getitem_359)
    mul_1036: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_148, rsqrt_148);  sub_148 = None
    squeeze_444: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_359, [0, 2, 3]);  getitem_359 = None
    squeeze_445: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_148, [0, 2, 3]);  rsqrt_148 = None
    mul_1037: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_444, 0.1)
    mul_1038: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1222, 0.9)
    add_793: "f32[864]" = torch.ops.aten.add.Tensor(mul_1037, mul_1038);  mul_1037 = mul_1038 = None
    squeeze_446: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_358, [0, 2, 3]);  getitem_358 = None
    mul_1039: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_446, 1.001034126163392);  squeeze_446 = None
    mul_1040: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1039, 0.1);  mul_1039 = None
    mul_1041: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1223, 0.9)
    add_794: "f32[864]" = torch.ops.aten.add.Tensor(mul_1040, mul_1041);  mul_1040 = mul_1041 = None
    unsqueeze_592: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_576, -1)
    unsqueeze_593: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
    mul_1042: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1036, unsqueeze_593);  mul_1036 = unsqueeze_593 = None
    unsqueeze_594: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_577, -1);  primals_577 = None
    unsqueeze_595: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
    add_795: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1042, unsqueeze_595);  mul_1042 = unsqueeze_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_147: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_795);  add_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_275: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_147, primals_578, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_276: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_275, primals_579, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_796: "i64[]" = torch.ops.aten.add.Tensor(primals_1227, 1)
    var_mean_149 = torch.ops.aten.var_mean.correction(convolution_276, [0, 2, 3], correction = 0, keepdim = True)
    getitem_360: "f32[1, 864, 1, 1]" = var_mean_149[0]
    getitem_361: "f32[1, 864, 1, 1]" = var_mean_149[1];  var_mean_149 = None
    add_797: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_360, 0.001)
    rsqrt_149: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_797);  add_797 = None
    sub_149: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_276, getitem_361)
    mul_1043: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_149, rsqrt_149);  sub_149 = None
    squeeze_447: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_361, [0, 2, 3]);  getitem_361 = None
    squeeze_448: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_149, [0, 2, 3]);  rsqrt_149 = None
    mul_1044: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_447, 0.1)
    mul_1045: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1225, 0.9)
    add_798: "f32[864]" = torch.ops.aten.add.Tensor(mul_1044, mul_1045);  mul_1044 = mul_1045 = None
    squeeze_449: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_360, [0, 2, 3]);  getitem_360 = None
    mul_1046: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_449, 1.001034126163392);  squeeze_449 = None
    mul_1047: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1046, 0.1);  mul_1046 = None
    mul_1048: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1226, 0.9)
    add_799: "f32[864]" = torch.ops.aten.add.Tensor(mul_1047, mul_1048);  mul_1047 = mul_1048 = None
    unsqueeze_596: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_580, -1)
    unsqueeze_597: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
    mul_1049: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1043, unsqueeze_597);  mul_1043 = unsqueeze_597 = None
    unsqueeze_598: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_581, -1);  primals_581 = None
    unsqueeze_599: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
    add_800: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1049, unsqueeze_599);  mul_1049 = unsqueeze_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_33: "f32[8, 864, 23, 23]" = torch.ops.aten.constant_pad_nd.default(add_779, [1, 1, 1, 1], -inf);  add_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_31 = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd_33, [3, 3], [2, 2])
    getitem_362: "f32[8, 864, 11, 11]" = max_pool2d_with_indices_31[0]
    getitem_363: "i64[8, 864, 11, 11]" = max_pool2d_with_indices_31[1];  max_pool2d_with_indices_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_801: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_800, getitem_362);  add_800 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_34: "f32[8, 864, 25, 25]" = torch.ops.aten.constant_pad_nd.default(relu_146, [2, 2, 2, 2], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_277: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(constant_pad_nd_34, primals_23, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_278: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_277, primals_582, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_802: "i64[]" = torch.ops.aten.add.Tensor(primals_1230, 1)
    var_mean_150 = torch.ops.aten.var_mean.correction(convolution_278, [0, 2, 3], correction = 0, keepdim = True)
    getitem_364: "f32[1, 864, 1, 1]" = var_mean_150[0]
    getitem_365: "f32[1, 864, 1, 1]" = var_mean_150[1];  var_mean_150 = None
    add_803: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_364, 0.001)
    rsqrt_150: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_803);  add_803 = None
    sub_150: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_278, getitem_365)
    mul_1050: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_150, rsqrt_150);  sub_150 = None
    squeeze_450: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_365, [0, 2, 3]);  getitem_365 = None
    squeeze_451: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_150, [0, 2, 3]);  rsqrt_150 = None
    mul_1051: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_450, 0.1)
    mul_1052: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1228, 0.9)
    add_804: "f32[864]" = torch.ops.aten.add.Tensor(mul_1051, mul_1052);  mul_1051 = mul_1052 = None
    squeeze_452: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_364, [0, 2, 3]);  getitem_364 = None
    mul_1053: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_452, 1.001034126163392);  squeeze_452 = None
    mul_1054: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1053, 0.1);  mul_1053 = None
    mul_1055: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1229, 0.9)
    add_805: "f32[864]" = torch.ops.aten.add.Tensor(mul_1054, mul_1055);  mul_1054 = mul_1055 = None
    unsqueeze_600: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_583, -1)
    unsqueeze_601: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
    mul_1056: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1050, unsqueeze_601);  mul_1050 = unsqueeze_601 = None
    unsqueeze_602: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_584, -1);  primals_584 = None
    unsqueeze_603: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
    add_806: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1056, unsqueeze_603);  mul_1056 = unsqueeze_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_149: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_806);  add_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_279: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_149, primals_585, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_280: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_279, primals_586, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_807: "i64[]" = torch.ops.aten.add.Tensor(primals_1233, 1)
    var_mean_151 = torch.ops.aten.var_mean.correction(convolution_280, [0, 2, 3], correction = 0, keepdim = True)
    getitem_366: "f32[1, 864, 1, 1]" = var_mean_151[0]
    getitem_367: "f32[1, 864, 1, 1]" = var_mean_151[1];  var_mean_151 = None
    add_808: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_366, 0.001)
    rsqrt_151: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_808);  add_808 = None
    sub_151: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_280, getitem_367)
    mul_1057: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_151, rsqrt_151);  sub_151 = None
    squeeze_453: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_367, [0, 2, 3]);  getitem_367 = None
    squeeze_454: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_151, [0, 2, 3]);  rsqrt_151 = None
    mul_1058: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_453, 0.1)
    mul_1059: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1231, 0.9)
    add_809: "f32[864]" = torch.ops.aten.add.Tensor(mul_1058, mul_1059);  mul_1058 = mul_1059 = None
    squeeze_455: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_366, [0, 2, 3]);  getitem_366 = None
    mul_1060: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_455, 1.001034126163392);  squeeze_455 = None
    mul_1061: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1060, 0.1);  mul_1060 = None
    mul_1062: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1232, 0.9)
    add_810: "f32[864]" = torch.ops.aten.add.Tensor(mul_1061, mul_1062);  mul_1061 = mul_1062 = None
    unsqueeze_604: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_587, -1)
    unsqueeze_605: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
    mul_1063: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1057, unsqueeze_605);  mul_1057 = unsqueeze_605 = None
    unsqueeze_606: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_588, -1);  primals_588 = None
    unsqueeze_607: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
    add_811: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1063, unsqueeze_607);  mul_1063 = unsqueeze_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_35: "f32[8, 864, 23, 23]" = torch.ops.aten.constant_pad_nd.default(relu_146, [1, 1, 1, 1], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_281: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(constant_pad_nd_35, primals_24, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_282: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_281, primals_589, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_812: "i64[]" = torch.ops.aten.add.Tensor(primals_1236, 1)
    var_mean_152 = torch.ops.aten.var_mean.correction(convolution_282, [0, 2, 3], correction = 0, keepdim = True)
    getitem_368: "f32[1, 864, 1, 1]" = var_mean_152[0]
    getitem_369: "f32[1, 864, 1, 1]" = var_mean_152[1];  var_mean_152 = None
    add_813: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_368, 0.001)
    rsqrt_152: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_813);  add_813 = None
    sub_152: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_282, getitem_369)
    mul_1064: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_152, rsqrt_152);  sub_152 = None
    squeeze_456: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_369, [0, 2, 3]);  getitem_369 = None
    squeeze_457: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_152, [0, 2, 3]);  rsqrt_152 = None
    mul_1065: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_456, 0.1)
    mul_1066: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1234, 0.9)
    add_814: "f32[864]" = torch.ops.aten.add.Tensor(mul_1065, mul_1066);  mul_1065 = mul_1066 = None
    squeeze_458: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_368, [0, 2, 3]);  getitem_368 = None
    mul_1067: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_458, 1.001034126163392);  squeeze_458 = None
    mul_1068: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1067, 0.1);  mul_1067 = None
    mul_1069: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1235, 0.9)
    add_815: "f32[864]" = torch.ops.aten.add.Tensor(mul_1068, mul_1069);  mul_1068 = mul_1069 = None
    unsqueeze_608: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_590, -1)
    unsqueeze_609: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
    mul_1070: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1064, unsqueeze_609);  mul_1064 = unsqueeze_609 = None
    unsqueeze_610: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_591, -1);  primals_591 = None
    unsqueeze_611: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
    add_816: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1070, unsqueeze_611);  mul_1070 = unsqueeze_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_151: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_816);  add_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_283: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_151, primals_592, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_284: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_283, primals_593, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_817: "i64[]" = torch.ops.aten.add.Tensor(primals_1239, 1)
    var_mean_153 = torch.ops.aten.var_mean.correction(convolution_284, [0, 2, 3], correction = 0, keepdim = True)
    getitem_370: "f32[1, 864, 1, 1]" = var_mean_153[0]
    getitem_371: "f32[1, 864, 1, 1]" = var_mean_153[1];  var_mean_153 = None
    add_818: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_370, 0.001)
    rsqrt_153: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_818);  add_818 = None
    sub_153: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_284, getitem_371)
    mul_1071: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_153, rsqrt_153);  sub_153 = None
    squeeze_459: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_371, [0, 2, 3]);  getitem_371 = None
    squeeze_460: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_153, [0, 2, 3]);  rsqrt_153 = None
    mul_1072: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_459, 0.1)
    mul_1073: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1237, 0.9)
    add_819: "f32[864]" = torch.ops.aten.add.Tensor(mul_1072, mul_1073);  mul_1072 = mul_1073 = None
    squeeze_461: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_370, [0, 2, 3]);  getitem_370 = None
    mul_1074: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_461, 1.001034126163392);  squeeze_461 = None
    mul_1075: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1074, 0.1);  mul_1074 = None
    mul_1076: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1238, 0.9)
    add_820: "f32[864]" = torch.ops.aten.add.Tensor(mul_1075, mul_1076);  mul_1075 = mul_1076 = None
    unsqueeze_612: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_594, -1)
    unsqueeze_613: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
    mul_1077: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1071, unsqueeze_613);  mul_1071 = unsqueeze_613 = None
    unsqueeze_614: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_595, -1);  primals_595 = None
    unsqueeze_615: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
    add_821: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1077, unsqueeze_615);  mul_1077 = unsqueeze_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_822: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_811, add_821);  add_811 = add_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_152: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_822)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_285: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_152, primals_596, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_286: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_285, primals_597, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_823: "i64[]" = torch.ops.aten.add.Tensor(primals_1242, 1)
    var_mean_154 = torch.ops.aten.var_mean.correction(convolution_286, [0, 2, 3], correction = 0, keepdim = True)
    getitem_372: "f32[1, 864, 1, 1]" = var_mean_154[0]
    getitem_373: "f32[1, 864, 1, 1]" = var_mean_154[1];  var_mean_154 = None
    add_824: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_372, 0.001)
    rsqrt_154: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_824);  add_824 = None
    sub_154: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_286, getitem_373)
    mul_1078: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_154, rsqrt_154);  sub_154 = None
    squeeze_462: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_373, [0, 2, 3]);  getitem_373 = None
    squeeze_463: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_154, [0, 2, 3]);  rsqrt_154 = None
    mul_1079: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_462, 0.1)
    mul_1080: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1240, 0.9)
    add_825: "f32[864]" = torch.ops.aten.add.Tensor(mul_1079, mul_1080);  mul_1079 = mul_1080 = None
    squeeze_464: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_372, [0, 2, 3]);  getitem_372 = None
    mul_1081: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_464, 1.001034126163392);  squeeze_464 = None
    mul_1082: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1081, 0.1);  mul_1081 = None
    mul_1083: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1241, 0.9)
    add_826: "f32[864]" = torch.ops.aten.add.Tensor(mul_1082, mul_1083);  mul_1082 = mul_1083 = None
    unsqueeze_616: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_598, -1)
    unsqueeze_617: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
    mul_1084: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1078, unsqueeze_617);  mul_1078 = unsqueeze_617 = None
    unsqueeze_618: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_599, -1);  primals_599 = None
    unsqueeze_619: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
    add_827: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1084, unsqueeze_619);  mul_1084 = unsqueeze_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_153: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_827);  add_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_287: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_153, primals_600, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_288: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_287, primals_601, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_828: "i64[]" = torch.ops.aten.add.Tensor(primals_1245, 1)
    var_mean_155 = torch.ops.aten.var_mean.correction(convolution_288, [0, 2, 3], correction = 0, keepdim = True)
    getitem_374: "f32[1, 864, 1, 1]" = var_mean_155[0]
    getitem_375: "f32[1, 864, 1, 1]" = var_mean_155[1];  var_mean_155 = None
    add_829: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_374, 0.001)
    rsqrt_155: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_829);  add_829 = None
    sub_155: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_288, getitem_375)
    mul_1085: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_155, rsqrt_155);  sub_155 = None
    squeeze_465: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_375, [0, 2, 3]);  getitem_375 = None
    squeeze_466: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_155, [0, 2, 3]);  rsqrt_155 = None
    mul_1086: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_465, 0.1)
    mul_1087: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1243, 0.9)
    add_830: "f32[864]" = torch.ops.aten.add.Tensor(mul_1086, mul_1087);  mul_1086 = mul_1087 = None
    squeeze_467: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_374, [0, 2, 3]);  getitem_374 = None
    mul_1088: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_467, 1.001034126163392);  squeeze_467 = None
    mul_1089: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1088, 0.1);  mul_1088 = None
    mul_1090: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1244, 0.9)
    add_831: "f32[864]" = torch.ops.aten.add.Tensor(mul_1089, mul_1090);  mul_1089 = mul_1090 = None
    unsqueeze_620: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_602, -1)
    unsqueeze_621: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
    mul_1091: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1085, unsqueeze_621);  mul_1085 = unsqueeze_621 = None
    unsqueeze_622: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_603, -1);  primals_603 = None
    unsqueeze_623: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
    add_832: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1091, unsqueeze_623);  mul_1091 = unsqueeze_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_833: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_832, getitem_362);  add_832 = getitem_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_37: "f32[8, 864, 23, 23]" = torch.ops.aten.constant_pad_nd.default(relu_144, [1, 1, 1, 1], 0.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_289: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(constant_pad_nd_37, primals_25, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_290: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_289, primals_604, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_834: "i64[]" = torch.ops.aten.add.Tensor(primals_1248, 1)
    var_mean_156 = torch.ops.aten.var_mean.correction(convolution_290, [0, 2, 3], correction = 0, keepdim = True)
    getitem_378: "f32[1, 864, 1, 1]" = var_mean_156[0]
    getitem_379: "f32[1, 864, 1, 1]" = var_mean_156[1];  var_mean_156 = None
    add_835: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_378, 0.001)
    rsqrt_156: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_835);  add_835 = None
    sub_156: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_290, getitem_379)
    mul_1092: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_156, rsqrt_156);  sub_156 = None
    squeeze_468: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_379, [0, 2, 3]);  getitem_379 = None
    squeeze_469: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_156, [0, 2, 3]);  rsqrt_156 = None
    mul_1093: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_468, 0.1)
    mul_1094: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1246, 0.9)
    add_836: "f32[864]" = torch.ops.aten.add.Tensor(mul_1093, mul_1094);  mul_1093 = mul_1094 = None
    squeeze_470: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_378, [0, 2, 3]);  getitem_378 = None
    mul_1095: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_470, 1.001034126163392);  squeeze_470 = None
    mul_1096: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1095, 0.1);  mul_1095 = None
    mul_1097: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1247, 0.9)
    add_837: "f32[864]" = torch.ops.aten.add.Tensor(mul_1096, mul_1097);  mul_1096 = mul_1097 = None
    unsqueeze_624: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_605, -1)
    unsqueeze_625: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
    mul_1098: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1092, unsqueeze_625);  mul_1092 = unsqueeze_625 = None
    unsqueeze_626: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_606, -1);  primals_606 = None
    unsqueeze_627: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
    add_838: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1098, unsqueeze_627);  mul_1098 = unsqueeze_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_155: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_838);  add_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_291: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_155, primals_607, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_292: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_291, primals_608, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_839: "i64[]" = torch.ops.aten.add.Tensor(primals_1251, 1)
    var_mean_157 = torch.ops.aten.var_mean.correction(convolution_292, [0, 2, 3], correction = 0, keepdim = True)
    getitem_380: "f32[1, 864, 1, 1]" = var_mean_157[0]
    getitem_381: "f32[1, 864, 1, 1]" = var_mean_157[1];  var_mean_157 = None
    add_840: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_380, 0.001)
    rsqrt_157: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_840);  add_840 = None
    sub_157: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_292, getitem_381)
    mul_1099: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_157, rsqrt_157);  sub_157 = None
    squeeze_471: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_381, [0, 2, 3]);  getitem_381 = None
    squeeze_472: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_157, [0, 2, 3]);  rsqrt_157 = None
    mul_1100: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_471, 0.1)
    mul_1101: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1249, 0.9)
    add_841: "f32[864]" = torch.ops.aten.add.Tensor(mul_1100, mul_1101);  mul_1100 = mul_1101 = None
    squeeze_473: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_380, [0, 2, 3]);  getitem_380 = None
    mul_1102: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_473, 1.001034126163392);  squeeze_473 = None
    mul_1103: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1102, 0.1);  mul_1102 = None
    mul_1104: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1250, 0.9)
    add_842: "f32[864]" = torch.ops.aten.add.Tensor(mul_1103, mul_1104);  mul_1103 = mul_1104 = None
    unsqueeze_628: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_609, -1)
    unsqueeze_629: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
    mul_1105: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1099, unsqueeze_629);  mul_1099 = unsqueeze_629 = None
    unsqueeze_630: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_610, -1);  primals_610 = None
    unsqueeze_631: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
    add_843: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1105, unsqueeze_631);  mul_1105 = unsqueeze_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_293: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_146, primals_26, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_844: "i64[]" = torch.ops.aten.add.Tensor(primals_1254, 1)
    var_mean_158 = torch.ops.aten.var_mean.correction(convolution_293, [0, 2, 3], correction = 0, keepdim = True)
    getitem_382: "f32[1, 864, 1, 1]" = var_mean_158[0]
    getitem_383: "f32[1, 864, 1, 1]" = var_mean_158[1];  var_mean_158 = None
    add_845: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_382, 0.001)
    rsqrt_158: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_845);  add_845 = None
    sub_158: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_293, getitem_383)
    mul_1106: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_158, rsqrt_158);  sub_158 = None
    squeeze_474: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_383, [0, 2, 3]);  getitem_383 = None
    squeeze_475: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_158, [0, 2, 3]);  rsqrt_158 = None
    mul_1107: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_474, 0.1)
    mul_1108: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1252, 0.9)
    add_846: "f32[864]" = torch.ops.aten.add.Tensor(mul_1107, mul_1108);  mul_1107 = mul_1108 = None
    squeeze_476: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_382, [0, 2, 3]);  getitem_382 = None
    mul_1109: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_476, 1.001034126163392);  squeeze_476 = None
    mul_1110: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1109, 0.1);  mul_1109 = None
    mul_1111: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1253, 0.9)
    add_847: "f32[864]" = torch.ops.aten.add.Tensor(mul_1110, mul_1111);  mul_1110 = mul_1111 = None
    unsqueeze_632: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_611, -1)
    unsqueeze_633: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
    mul_1112: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1106, unsqueeze_633);  mul_1106 = unsqueeze_633 = None
    unsqueeze_634: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_612, -1);  primals_612 = None
    unsqueeze_635: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
    add_848: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1112, unsqueeze_635);  mul_1112 = unsqueeze_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_849: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_843, add_848);  add_843 = add_848 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat_13: "f32[8, 4320, 11, 11]" = torch.ops.aten.cat.default([add_790, add_801, add_822, add_833, add_849], 1);  add_790 = add_801 = add_822 = add_833 = add_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    avg_pool2d_6: "f32[8, 2160, 11, 11]" = torch.ops.aten.avg_pool2d.default(relu_143, [1, 1], [2, 2], [0, 0], False, False)
    convolution_294: "f32[8, 432, 11, 11]" = torch.ops.aten.convolution.default(avg_pool2d_6, primals_613, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:97, code: x_path2 = self.path_2(x)
    constant_pad_nd_39: "f32[8, 2160, 21, 21]" = torch.ops.aten.constant_pad_nd.default(relu_143, [-1, 1, -1, 1], 0.0)
    avg_pool2d_7: "f32[8, 2160, 11, 11]" = torch.ops.aten.avg_pool2d.default(constant_pad_nd_39, [1, 1], [2, 2], [0, 0], False, False)
    convolution_295: "f32[8, 432, 11, 11]" = torch.ops.aten.convolution.default(avg_pool2d_7, primals_614, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    cat_14: "f32[8, 864, 11, 11]" = torch.ops.aten.cat.default([convolution_294, convolution_295], 1);  convolution_294 = convolution_295 = None
    add_850: "i64[]" = torch.ops.aten.add.Tensor(primals_1257, 1)
    var_mean_159 = torch.ops.aten.var_mean.correction(cat_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_384: "f32[1, 864, 1, 1]" = var_mean_159[0]
    getitem_385: "f32[1, 864, 1, 1]" = var_mean_159[1];  var_mean_159 = None
    add_851: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_384, 0.001)
    rsqrt_159: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_851);  add_851 = None
    sub_159: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(cat_14, getitem_385)
    mul_1113: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_159, rsqrt_159);  sub_159 = None
    squeeze_477: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_385, [0, 2, 3]);  getitem_385 = None
    squeeze_478: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_159, [0, 2, 3]);  rsqrt_159 = None
    mul_1114: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_477, 0.1)
    mul_1115: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1255, 0.9)
    add_852: "f32[864]" = torch.ops.aten.add.Tensor(mul_1114, mul_1115);  mul_1114 = mul_1115 = None
    squeeze_479: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_384, [0, 2, 3]);  getitem_384 = None
    mul_1116: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_479, 1.001034126163392);  squeeze_479 = None
    mul_1117: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1116, 0.1);  mul_1116 = None
    mul_1118: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1256, 0.9)
    add_853: "f32[864]" = torch.ops.aten.add.Tensor(mul_1117, mul_1118);  mul_1117 = mul_1118 = None
    unsqueeze_636: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_615, -1)
    unsqueeze_637: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
    mul_1119: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1113, unsqueeze_637);  mul_1113 = unsqueeze_637 = None
    unsqueeze_638: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_616, -1);  primals_616 = None
    unsqueeze_639: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
    add_854: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1119, unsqueeze_639);  mul_1119 = unsqueeze_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu_158: "f32[8, 4320, 11, 11]" = torch.ops.aten.relu.default(cat_13);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_296: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_158, primals_617, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_855: "i64[]" = torch.ops.aten.add.Tensor(primals_1260, 1)
    var_mean_160 = torch.ops.aten.var_mean.correction(convolution_296, [0, 2, 3], correction = 0, keepdim = True)
    getitem_386: "f32[1, 864, 1, 1]" = var_mean_160[0]
    getitem_387: "f32[1, 864, 1, 1]" = var_mean_160[1];  var_mean_160 = None
    add_856: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_386, 0.001)
    rsqrt_160: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_856);  add_856 = None
    sub_160: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_296, getitem_387)
    mul_1120: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_160, rsqrt_160);  sub_160 = None
    squeeze_480: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_387, [0, 2, 3]);  getitem_387 = None
    squeeze_481: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_160, [0, 2, 3]);  rsqrt_160 = None
    mul_1121: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_480, 0.1)
    mul_1122: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1258, 0.9)
    add_857: "f32[864]" = torch.ops.aten.add.Tensor(mul_1121, mul_1122);  mul_1121 = mul_1122 = None
    squeeze_482: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_386, [0, 2, 3]);  getitem_386 = None
    mul_1123: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_482, 1.001034126163392);  squeeze_482 = None
    mul_1124: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1123, 0.1);  mul_1123 = None
    mul_1125: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1259, 0.9)
    add_858: "f32[864]" = torch.ops.aten.add.Tensor(mul_1124, mul_1125);  mul_1124 = mul_1125 = None
    unsqueeze_640: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_618, -1)
    unsqueeze_641: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
    mul_1126: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1120, unsqueeze_641);  mul_1120 = unsqueeze_641 = None
    unsqueeze_642: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_619, -1);  primals_619 = None
    unsqueeze_643: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
    add_859: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1126, unsqueeze_643);  mul_1126 = unsqueeze_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_159: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_854)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_297: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_159, primals_620, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_298: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_297, primals_621, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_860: "i64[]" = torch.ops.aten.add.Tensor(primals_1263, 1)
    var_mean_161 = torch.ops.aten.var_mean.correction(convolution_298, [0, 2, 3], correction = 0, keepdim = True)
    getitem_388: "f32[1, 864, 1, 1]" = var_mean_161[0]
    getitem_389: "f32[1, 864, 1, 1]" = var_mean_161[1];  var_mean_161 = None
    add_861: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_388, 0.001)
    rsqrt_161: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_861);  add_861 = None
    sub_161: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_298, getitem_389)
    mul_1127: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_161, rsqrt_161);  sub_161 = None
    squeeze_483: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_389, [0, 2, 3]);  getitem_389 = None
    squeeze_484: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_161, [0, 2, 3]);  rsqrt_161 = None
    mul_1128: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_483, 0.1)
    mul_1129: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1261, 0.9)
    add_862: "f32[864]" = torch.ops.aten.add.Tensor(mul_1128, mul_1129);  mul_1128 = mul_1129 = None
    squeeze_485: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_388, [0, 2, 3]);  getitem_388 = None
    mul_1130: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_485, 1.001034126163392);  squeeze_485 = None
    mul_1131: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1130, 0.1);  mul_1130 = None
    mul_1132: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1262, 0.9)
    add_863: "f32[864]" = torch.ops.aten.add.Tensor(mul_1131, mul_1132);  mul_1131 = mul_1132 = None
    unsqueeze_644: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_622, -1)
    unsqueeze_645: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
    mul_1133: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1127, unsqueeze_645);  mul_1127 = unsqueeze_645 = None
    unsqueeze_646: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_623, -1);  primals_623 = None
    unsqueeze_647: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
    add_864: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1133, unsqueeze_647);  mul_1133 = unsqueeze_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_160: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_864);  add_864 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_299: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_160, primals_624, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_300: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_299, primals_625, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_865: "i64[]" = torch.ops.aten.add.Tensor(primals_1266, 1)
    var_mean_162 = torch.ops.aten.var_mean.correction(convolution_300, [0, 2, 3], correction = 0, keepdim = True)
    getitem_390: "f32[1, 864, 1, 1]" = var_mean_162[0]
    getitem_391: "f32[1, 864, 1, 1]" = var_mean_162[1];  var_mean_162 = None
    add_866: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_390, 0.001)
    rsqrt_162: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_866);  add_866 = None
    sub_162: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_300, getitem_391)
    mul_1134: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_162, rsqrt_162);  sub_162 = None
    squeeze_486: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_391, [0, 2, 3]);  getitem_391 = None
    squeeze_487: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_162, [0, 2, 3]);  rsqrt_162 = None
    mul_1135: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_486, 0.1)
    mul_1136: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1264, 0.9)
    add_867: "f32[864]" = torch.ops.aten.add.Tensor(mul_1135, mul_1136);  mul_1135 = mul_1136 = None
    squeeze_488: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_390, [0, 2, 3]);  getitem_390 = None
    mul_1137: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_488, 1.001034126163392);  squeeze_488 = None
    mul_1138: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1137, 0.1);  mul_1137 = None
    mul_1139: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1265, 0.9)
    add_868: "f32[864]" = torch.ops.aten.add.Tensor(mul_1138, mul_1139);  mul_1138 = mul_1139 = None
    unsqueeze_648: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_626, -1)
    unsqueeze_649: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
    mul_1140: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1134, unsqueeze_649);  mul_1134 = unsqueeze_649 = None
    unsqueeze_650: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_627, -1);  primals_627 = None
    unsqueeze_651: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
    add_869: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1140, unsqueeze_651);  mul_1140 = unsqueeze_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_33 = torch.ops.aten.max_pool2d_with_indices.default(add_854, [3, 3], [1, 1], [1, 1])
    getitem_392: "f32[8, 864, 11, 11]" = max_pool2d_with_indices_33[0]
    getitem_393: "i64[8, 864, 11, 11]" = max_pool2d_with_indices_33[1];  max_pool2d_with_indices_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_870: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_869, getitem_392);  add_869 = getitem_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_161: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_859)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_301: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_161, primals_628, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_302: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_301, primals_629, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_871: "i64[]" = torch.ops.aten.add.Tensor(primals_1269, 1)
    var_mean_163 = torch.ops.aten.var_mean.correction(convolution_302, [0, 2, 3], correction = 0, keepdim = True)
    getitem_394: "f32[1, 864, 1, 1]" = var_mean_163[0]
    getitem_395: "f32[1, 864, 1, 1]" = var_mean_163[1];  var_mean_163 = None
    add_872: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_394, 0.001)
    rsqrt_163: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_872);  add_872 = None
    sub_163: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_302, getitem_395)
    mul_1141: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_163, rsqrt_163);  sub_163 = None
    squeeze_489: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_395, [0, 2, 3]);  getitem_395 = None
    squeeze_490: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_163, [0, 2, 3]);  rsqrt_163 = None
    mul_1142: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_489, 0.1)
    mul_1143: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1267, 0.9)
    add_873: "f32[864]" = torch.ops.aten.add.Tensor(mul_1142, mul_1143);  mul_1142 = mul_1143 = None
    squeeze_491: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_394, [0, 2, 3]);  getitem_394 = None
    mul_1144: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_491, 1.001034126163392);  squeeze_491 = None
    mul_1145: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1144, 0.1);  mul_1144 = None
    mul_1146: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1268, 0.9)
    add_874: "f32[864]" = torch.ops.aten.add.Tensor(mul_1145, mul_1146);  mul_1145 = mul_1146 = None
    unsqueeze_652: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_630, -1)
    unsqueeze_653: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
    mul_1147: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1141, unsqueeze_653);  mul_1141 = unsqueeze_653 = None
    unsqueeze_654: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_631, -1);  primals_631 = None
    unsqueeze_655: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
    add_875: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1147, unsqueeze_655);  mul_1147 = unsqueeze_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_162: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_875);  add_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_303: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_162, primals_632, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_304: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_303, primals_633, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_876: "i64[]" = torch.ops.aten.add.Tensor(primals_1272, 1)
    var_mean_164 = torch.ops.aten.var_mean.correction(convolution_304, [0, 2, 3], correction = 0, keepdim = True)
    getitem_396: "f32[1, 864, 1, 1]" = var_mean_164[0]
    getitem_397: "f32[1, 864, 1, 1]" = var_mean_164[1];  var_mean_164 = None
    add_877: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_396, 0.001)
    rsqrt_164: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_877);  add_877 = None
    sub_164: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_304, getitem_397)
    mul_1148: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_164, rsqrt_164);  sub_164 = None
    squeeze_492: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_397, [0, 2, 3]);  getitem_397 = None
    squeeze_493: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_164, [0, 2, 3]);  rsqrt_164 = None
    mul_1149: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_492, 0.1)
    mul_1150: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1270, 0.9)
    add_878: "f32[864]" = torch.ops.aten.add.Tensor(mul_1149, mul_1150);  mul_1149 = mul_1150 = None
    squeeze_494: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_396, [0, 2, 3]);  getitem_396 = None
    mul_1151: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_494, 1.001034126163392);  squeeze_494 = None
    mul_1152: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1151, 0.1);  mul_1151 = None
    mul_1153: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1271, 0.9)
    add_879: "f32[864]" = torch.ops.aten.add.Tensor(mul_1152, mul_1153);  mul_1152 = mul_1153 = None
    unsqueeze_656: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_634, -1)
    unsqueeze_657: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
    mul_1154: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1148, unsqueeze_657);  mul_1148 = unsqueeze_657 = None
    unsqueeze_658: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_635, -1);  primals_635 = None
    unsqueeze_659: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
    add_880: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1154, unsqueeze_659);  mul_1154 = unsqueeze_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_34 = torch.ops.aten.max_pool2d_with_indices.default(add_859, [3, 3], [1, 1], [1, 1])
    getitem_398: "f32[8, 864, 11, 11]" = max_pool2d_with_indices_34[0]
    getitem_399: "i64[8, 864, 11, 11]" = max_pool2d_with_indices_34[1];  max_pool2d_with_indices_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_881: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_880, getitem_398);  add_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_305: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_161, primals_636, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_306: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_305, primals_637, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_882: "i64[]" = torch.ops.aten.add.Tensor(primals_1275, 1)
    var_mean_165 = torch.ops.aten.var_mean.correction(convolution_306, [0, 2, 3], correction = 0, keepdim = True)
    getitem_400: "f32[1, 864, 1, 1]" = var_mean_165[0]
    getitem_401: "f32[1, 864, 1, 1]" = var_mean_165[1];  var_mean_165 = None
    add_883: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_400, 0.001)
    rsqrt_165: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_883);  add_883 = None
    sub_165: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_306, getitem_401)
    mul_1155: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_165, rsqrt_165);  sub_165 = None
    squeeze_495: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_401, [0, 2, 3]);  getitem_401 = None
    squeeze_496: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_165, [0, 2, 3]);  rsqrt_165 = None
    mul_1156: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_495, 0.1)
    mul_1157: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1273, 0.9)
    add_884: "f32[864]" = torch.ops.aten.add.Tensor(mul_1156, mul_1157);  mul_1156 = mul_1157 = None
    squeeze_497: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_400, [0, 2, 3]);  getitem_400 = None
    mul_1158: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_497, 1.001034126163392);  squeeze_497 = None
    mul_1159: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1158, 0.1);  mul_1158 = None
    mul_1160: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1274, 0.9)
    add_885: "f32[864]" = torch.ops.aten.add.Tensor(mul_1159, mul_1160);  mul_1159 = mul_1160 = None
    unsqueeze_660: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_638, -1)
    unsqueeze_661: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
    mul_1161: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1155, unsqueeze_661);  mul_1155 = unsqueeze_661 = None
    unsqueeze_662: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_639, -1);  primals_639 = None
    unsqueeze_663: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
    add_886: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1161, unsqueeze_663);  mul_1161 = unsqueeze_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_164: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_886);  add_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_307: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_164, primals_640, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_308: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_307, primals_641, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_887: "i64[]" = torch.ops.aten.add.Tensor(primals_1278, 1)
    var_mean_166 = torch.ops.aten.var_mean.correction(convolution_308, [0, 2, 3], correction = 0, keepdim = True)
    getitem_402: "f32[1, 864, 1, 1]" = var_mean_166[0]
    getitem_403: "f32[1, 864, 1, 1]" = var_mean_166[1];  var_mean_166 = None
    add_888: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_402, 0.001)
    rsqrt_166: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_888);  add_888 = None
    sub_166: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_308, getitem_403)
    mul_1162: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_166, rsqrt_166);  sub_166 = None
    squeeze_498: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_403, [0, 2, 3]);  getitem_403 = None
    squeeze_499: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_166, [0, 2, 3]);  rsqrt_166 = None
    mul_1163: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_498, 0.1)
    mul_1164: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1276, 0.9)
    add_889: "f32[864]" = torch.ops.aten.add.Tensor(mul_1163, mul_1164);  mul_1163 = mul_1164 = None
    squeeze_500: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_402, [0, 2, 3]);  getitem_402 = None
    mul_1165: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_500, 1.001034126163392);  squeeze_500 = None
    mul_1166: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1165, 0.1);  mul_1165 = None
    mul_1167: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1277, 0.9)
    add_890: "f32[864]" = torch.ops.aten.add.Tensor(mul_1166, mul_1167);  mul_1166 = mul_1167 = None
    unsqueeze_664: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_642, -1)
    unsqueeze_665: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
    mul_1168: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1162, unsqueeze_665);  mul_1162 = unsqueeze_665 = None
    unsqueeze_666: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_643, -1);  primals_643 = None
    unsqueeze_667: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
    add_891: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1168, unsqueeze_667);  mul_1168 = unsqueeze_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_309: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_161, primals_644, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_310: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_309, primals_645, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_892: "i64[]" = torch.ops.aten.add.Tensor(primals_1281, 1)
    var_mean_167 = torch.ops.aten.var_mean.correction(convolution_310, [0, 2, 3], correction = 0, keepdim = True)
    getitem_404: "f32[1, 864, 1, 1]" = var_mean_167[0]
    getitem_405: "f32[1, 864, 1, 1]" = var_mean_167[1];  var_mean_167 = None
    add_893: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_404, 0.001)
    rsqrt_167: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_893);  add_893 = None
    sub_167: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_310, getitem_405)
    mul_1169: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_167, rsqrt_167);  sub_167 = None
    squeeze_501: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_405, [0, 2, 3]);  getitem_405 = None
    squeeze_502: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_167, [0, 2, 3]);  rsqrt_167 = None
    mul_1170: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_501, 0.1)
    mul_1171: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1279, 0.9)
    add_894: "f32[864]" = torch.ops.aten.add.Tensor(mul_1170, mul_1171);  mul_1170 = mul_1171 = None
    squeeze_503: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_404, [0, 2, 3]);  getitem_404 = None
    mul_1172: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_503, 1.001034126163392);  squeeze_503 = None
    mul_1173: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1172, 0.1);  mul_1172 = None
    mul_1174: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1280, 0.9)
    add_895: "f32[864]" = torch.ops.aten.add.Tensor(mul_1173, mul_1174);  mul_1173 = mul_1174 = None
    unsqueeze_668: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_646, -1)
    unsqueeze_669: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
    mul_1175: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1169, unsqueeze_669);  mul_1169 = unsqueeze_669 = None
    unsqueeze_670: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_647, -1);  primals_647 = None
    unsqueeze_671: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
    add_896: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1175, unsqueeze_671);  mul_1175 = unsqueeze_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_166: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_896);  add_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_311: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_166, primals_648, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_312: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_311, primals_649, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_897: "i64[]" = torch.ops.aten.add.Tensor(primals_1284, 1)
    var_mean_168 = torch.ops.aten.var_mean.correction(convolution_312, [0, 2, 3], correction = 0, keepdim = True)
    getitem_406: "f32[1, 864, 1, 1]" = var_mean_168[0]
    getitem_407: "f32[1, 864, 1, 1]" = var_mean_168[1];  var_mean_168 = None
    add_898: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_406, 0.001)
    rsqrt_168: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_898);  add_898 = None
    sub_168: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_312, getitem_407)
    mul_1176: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_168, rsqrt_168);  sub_168 = None
    squeeze_504: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_407, [0, 2, 3]);  getitem_407 = None
    squeeze_505: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_168, [0, 2, 3]);  rsqrt_168 = None
    mul_1177: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_504, 0.1)
    mul_1178: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1282, 0.9)
    add_899: "f32[864]" = torch.ops.aten.add.Tensor(mul_1177, mul_1178);  mul_1177 = mul_1178 = None
    squeeze_506: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_406, [0, 2, 3]);  getitem_406 = None
    mul_1179: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_506, 1.001034126163392);  squeeze_506 = None
    mul_1180: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1179, 0.1);  mul_1179 = None
    mul_1181: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1283, 0.9)
    add_900: "f32[864]" = torch.ops.aten.add.Tensor(mul_1180, mul_1181);  mul_1180 = mul_1181 = None
    unsqueeze_672: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_650, -1)
    unsqueeze_673: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
    mul_1182: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1176, unsqueeze_673);  mul_1176 = unsqueeze_673 = None
    unsqueeze_674: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_651, -1);  primals_651 = None
    unsqueeze_675: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
    add_901: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1182, unsqueeze_675);  mul_1182 = unsqueeze_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_902: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_891, add_901);  add_891 = add_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_167: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_902)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_313: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_167, primals_652, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_314: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_313, primals_653, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_903: "i64[]" = torch.ops.aten.add.Tensor(primals_1287, 1)
    var_mean_169 = torch.ops.aten.var_mean.correction(convolution_314, [0, 2, 3], correction = 0, keepdim = True)
    getitem_408: "f32[1, 864, 1, 1]" = var_mean_169[0]
    getitem_409: "f32[1, 864, 1, 1]" = var_mean_169[1];  var_mean_169 = None
    add_904: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_408, 0.001)
    rsqrt_169: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_904);  add_904 = None
    sub_169: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_314, getitem_409)
    mul_1183: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_169, rsqrt_169);  sub_169 = None
    squeeze_507: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_409, [0, 2, 3]);  getitem_409 = None
    squeeze_508: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_169, [0, 2, 3]);  rsqrt_169 = None
    mul_1184: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_507, 0.1)
    mul_1185: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1285, 0.9)
    add_905: "f32[864]" = torch.ops.aten.add.Tensor(mul_1184, mul_1185);  mul_1184 = mul_1185 = None
    squeeze_509: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_408, [0, 2, 3]);  getitem_408 = None
    mul_1186: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_509, 1.001034126163392);  squeeze_509 = None
    mul_1187: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1186, 0.1);  mul_1186 = None
    mul_1188: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1286, 0.9)
    add_906: "f32[864]" = torch.ops.aten.add.Tensor(mul_1187, mul_1188);  mul_1187 = mul_1188 = None
    unsqueeze_676: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_654, -1)
    unsqueeze_677: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
    mul_1189: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1183, unsqueeze_677);  mul_1183 = unsqueeze_677 = None
    unsqueeze_678: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_655, -1);  primals_655 = None
    unsqueeze_679: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
    add_907: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1189, unsqueeze_679);  mul_1189 = unsqueeze_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_168: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_907);  add_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_315: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_168, primals_656, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_316: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_315, primals_657, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_908: "i64[]" = torch.ops.aten.add.Tensor(primals_1290, 1)
    var_mean_170 = torch.ops.aten.var_mean.correction(convolution_316, [0, 2, 3], correction = 0, keepdim = True)
    getitem_410: "f32[1, 864, 1, 1]" = var_mean_170[0]
    getitem_411: "f32[1, 864, 1, 1]" = var_mean_170[1];  var_mean_170 = None
    add_909: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_410, 0.001)
    rsqrt_170: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_909);  add_909 = None
    sub_170: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_316, getitem_411)
    mul_1190: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_170, rsqrt_170);  sub_170 = None
    squeeze_510: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_411, [0, 2, 3]);  getitem_411 = None
    squeeze_511: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_170, [0, 2, 3]);  rsqrt_170 = None
    mul_1191: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_510, 0.1)
    mul_1192: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1288, 0.9)
    add_910: "f32[864]" = torch.ops.aten.add.Tensor(mul_1191, mul_1192);  mul_1191 = mul_1192 = None
    squeeze_512: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_410, [0, 2, 3]);  getitem_410 = None
    mul_1193: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_512, 1.001034126163392);  squeeze_512 = None
    mul_1194: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1193, 0.1);  mul_1193 = None
    mul_1195: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1289, 0.9)
    add_911: "f32[864]" = torch.ops.aten.add.Tensor(mul_1194, mul_1195);  mul_1194 = mul_1195 = None
    unsqueeze_680: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_658, -1)
    unsqueeze_681: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
    mul_1196: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1190, unsqueeze_681);  mul_1190 = unsqueeze_681 = None
    unsqueeze_682: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_659, -1);  primals_659 = None
    unsqueeze_683: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
    add_912: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1196, unsqueeze_683);  mul_1196 = unsqueeze_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_913: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_912, getitem_398);  add_912 = getitem_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_317: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_159, primals_660, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_318: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_317, primals_661, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_914: "i64[]" = torch.ops.aten.add.Tensor(primals_1293, 1)
    var_mean_171 = torch.ops.aten.var_mean.correction(convolution_318, [0, 2, 3], correction = 0, keepdim = True)
    getitem_414: "f32[1, 864, 1, 1]" = var_mean_171[0]
    getitem_415: "f32[1, 864, 1, 1]" = var_mean_171[1];  var_mean_171 = None
    add_915: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_414, 0.001)
    rsqrt_171: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_915);  add_915 = None
    sub_171: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_318, getitem_415)
    mul_1197: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_171, rsqrt_171);  sub_171 = None
    squeeze_513: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_415, [0, 2, 3]);  getitem_415 = None
    squeeze_514: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_171, [0, 2, 3]);  rsqrt_171 = None
    mul_1198: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_513, 0.1)
    mul_1199: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1291, 0.9)
    add_916: "f32[864]" = torch.ops.aten.add.Tensor(mul_1198, mul_1199);  mul_1198 = mul_1199 = None
    squeeze_515: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_414, [0, 2, 3]);  getitem_414 = None
    mul_1200: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_515, 1.001034126163392);  squeeze_515 = None
    mul_1201: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1200, 0.1);  mul_1200 = None
    mul_1202: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1292, 0.9)
    add_917: "f32[864]" = torch.ops.aten.add.Tensor(mul_1201, mul_1202);  mul_1201 = mul_1202 = None
    unsqueeze_684: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_662, -1)
    unsqueeze_685: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
    mul_1203: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1197, unsqueeze_685);  mul_1197 = unsqueeze_685 = None
    unsqueeze_686: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_663, -1);  primals_663 = None
    unsqueeze_687: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
    add_918: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1203, unsqueeze_687);  mul_1203 = unsqueeze_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_170: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_918);  add_918 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_319: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_170, primals_664, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_320: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_319, primals_665, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_919: "i64[]" = torch.ops.aten.add.Tensor(primals_1296, 1)
    var_mean_172 = torch.ops.aten.var_mean.correction(convolution_320, [0, 2, 3], correction = 0, keepdim = True)
    getitem_416: "f32[1, 864, 1, 1]" = var_mean_172[0]
    getitem_417: "f32[1, 864, 1, 1]" = var_mean_172[1];  var_mean_172 = None
    add_920: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_416, 0.001)
    rsqrt_172: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_920);  add_920 = None
    sub_172: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_320, getitem_417)
    mul_1204: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_172, rsqrt_172);  sub_172 = None
    squeeze_516: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_417, [0, 2, 3]);  getitem_417 = None
    squeeze_517: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_172, [0, 2, 3]);  rsqrt_172 = None
    mul_1205: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_516, 0.1)
    mul_1206: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1294, 0.9)
    add_921: "f32[864]" = torch.ops.aten.add.Tensor(mul_1205, mul_1206);  mul_1205 = mul_1206 = None
    squeeze_518: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_416, [0, 2, 3]);  getitem_416 = None
    mul_1207: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_518, 1.001034126163392);  squeeze_518 = None
    mul_1208: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1207, 0.1);  mul_1207 = None
    mul_1209: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1295, 0.9)
    add_922: "f32[864]" = torch.ops.aten.add.Tensor(mul_1208, mul_1209);  mul_1208 = mul_1209 = None
    unsqueeze_688: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_666, -1)
    unsqueeze_689: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
    mul_1210: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1204, unsqueeze_689);  mul_1204 = unsqueeze_689 = None
    unsqueeze_690: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_667, -1);  primals_667 = None
    unsqueeze_691: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
    add_923: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1210, unsqueeze_691);  mul_1210 = unsqueeze_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_924: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_923, add_859);  add_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat_15: "f32[8, 4320, 11, 11]" = torch.ops.aten.cat.default([add_870, add_881, add_902, add_913, add_924], 1);  add_870 = add_881 = add_902 = add_913 = add_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_321: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_158, primals_668, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_925: "i64[]" = torch.ops.aten.add.Tensor(primals_1299, 1)
    var_mean_173 = torch.ops.aten.var_mean.correction(convolution_321, [0, 2, 3], correction = 0, keepdim = True)
    getitem_418: "f32[1, 864, 1, 1]" = var_mean_173[0]
    getitem_419: "f32[1, 864, 1, 1]" = var_mean_173[1];  var_mean_173 = None
    add_926: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_418, 0.001)
    rsqrt_173: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_926);  add_926 = None
    sub_173: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_321, getitem_419)
    mul_1211: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_173, rsqrt_173);  sub_173 = None
    squeeze_519: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_419, [0, 2, 3]);  getitem_419 = None
    squeeze_520: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_173, [0, 2, 3]);  rsqrt_173 = None
    mul_1212: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_519, 0.1)
    mul_1213: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1297, 0.9)
    add_927: "f32[864]" = torch.ops.aten.add.Tensor(mul_1212, mul_1213);  mul_1212 = mul_1213 = None
    squeeze_521: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_418, [0, 2, 3]);  getitem_418 = None
    mul_1214: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_521, 1.001034126163392);  squeeze_521 = None
    mul_1215: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1214, 0.1);  mul_1214 = None
    mul_1216: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1298, 0.9)
    add_928: "f32[864]" = torch.ops.aten.add.Tensor(mul_1215, mul_1216);  mul_1215 = mul_1216 = None
    unsqueeze_692: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_669, -1)
    unsqueeze_693: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
    mul_1217: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1211, unsqueeze_693);  mul_1211 = unsqueeze_693 = None
    unsqueeze_694: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_670, -1);  primals_670 = None
    unsqueeze_695: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
    add_929: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1217, unsqueeze_695);  mul_1217 = unsqueeze_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu_172: "f32[8, 4320, 11, 11]" = torch.ops.aten.relu.default(cat_15);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_322: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_172, primals_671, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_930: "i64[]" = torch.ops.aten.add.Tensor(primals_1302, 1)
    var_mean_174 = torch.ops.aten.var_mean.correction(convolution_322, [0, 2, 3], correction = 0, keepdim = True)
    getitem_420: "f32[1, 864, 1, 1]" = var_mean_174[0]
    getitem_421: "f32[1, 864, 1, 1]" = var_mean_174[1];  var_mean_174 = None
    add_931: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_420, 0.001)
    rsqrt_174: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_931);  add_931 = None
    sub_174: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_322, getitem_421)
    mul_1218: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_174, rsqrt_174);  sub_174 = None
    squeeze_522: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_421, [0, 2, 3]);  getitem_421 = None
    squeeze_523: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_174, [0, 2, 3]);  rsqrt_174 = None
    mul_1219: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_522, 0.1)
    mul_1220: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1300, 0.9)
    add_932: "f32[864]" = torch.ops.aten.add.Tensor(mul_1219, mul_1220);  mul_1219 = mul_1220 = None
    squeeze_524: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_420, [0, 2, 3]);  getitem_420 = None
    mul_1221: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_524, 1.001034126163392);  squeeze_524 = None
    mul_1222: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1221, 0.1);  mul_1221 = None
    mul_1223: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1301, 0.9)
    add_933: "f32[864]" = torch.ops.aten.add.Tensor(mul_1222, mul_1223);  mul_1222 = mul_1223 = None
    unsqueeze_696: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_672, -1)
    unsqueeze_697: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
    mul_1224: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1218, unsqueeze_697);  mul_1218 = unsqueeze_697 = None
    unsqueeze_698: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_673, -1);  primals_673 = None
    unsqueeze_699: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
    add_934: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1224, unsqueeze_699);  mul_1224 = unsqueeze_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_173: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_929)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_323: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_173, primals_674, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_324: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_323, primals_675, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_935: "i64[]" = torch.ops.aten.add.Tensor(primals_1305, 1)
    var_mean_175 = torch.ops.aten.var_mean.correction(convolution_324, [0, 2, 3], correction = 0, keepdim = True)
    getitem_422: "f32[1, 864, 1, 1]" = var_mean_175[0]
    getitem_423: "f32[1, 864, 1, 1]" = var_mean_175[1];  var_mean_175 = None
    add_936: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_422, 0.001)
    rsqrt_175: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_936);  add_936 = None
    sub_175: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_324, getitem_423)
    mul_1225: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_175, rsqrt_175);  sub_175 = None
    squeeze_525: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_423, [0, 2, 3]);  getitem_423 = None
    squeeze_526: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_175, [0, 2, 3]);  rsqrt_175 = None
    mul_1226: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_525, 0.1)
    mul_1227: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1303, 0.9)
    add_937: "f32[864]" = torch.ops.aten.add.Tensor(mul_1226, mul_1227);  mul_1226 = mul_1227 = None
    squeeze_527: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_422, [0, 2, 3]);  getitem_422 = None
    mul_1228: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_527, 1.001034126163392);  squeeze_527 = None
    mul_1229: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1228, 0.1);  mul_1228 = None
    mul_1230: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1304, 0.9)
    add_938: "f32[864]" = torch.ops.aten.add.Tensor(mul_1229, mul_1230);  mul_1229 = mul_1230 = None
    unsqueeze_700: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_676, -1)
    unsqueeze_701: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
    mul_1231: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1225, unsqueeze_701);  mul_1225 = unsqueeze_701 = None
    unsqueeze_702: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_677, -1);  primals_677 = None
    unsqueeze_703: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
    add_939: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1231, unsqueeze_703);  mul_1231 = unsqueeze_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_174: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_939);  add_939 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_325: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_174, primals_678, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_326: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_325, primals_679, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_940: "i64[]" = torch.ops.aten.add.Tensor(primals_1308, 1)
    var_mean_176 = torch.ops.aten.var_mean.correction(convolution_326, [0, 2, 3], correction = 0, keepdim = True)
    getitem_424: "f32[1, 864, 1, 1]" = var_mean_176[0]
    getitem_425: "f32[1, 864, 1, 1]" = var_mean_176[1];  var_mean_176 = None
    add_941: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_424, 0.001)
    rsqrt_176: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_941);  add_941 = None
    sub_176: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_326, getitem_425)
    mul_1232: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_176, rsqrt_176);  sub_176 = None
    squeeze_528: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_425, [0, 2, 3]);  getitem_425 = None
    squeeze_529: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_176, [0, 2, 3]);  rsqrt_176 = None
    mul_1233: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_528, 0.1)
    mul_1234: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1306, 0.9)
    add_942: "f32[864]" = torch.ops.aten.add.Tensor(mul_1233, mul_1234);  mul_1233 = mul_1234 = None
    squeeze_530: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_424, [0, 2, 3]);  getitem_424 = None
    mul_1235: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_530, 1.001034126163392);  squeeze_530 = None
    mul_1236: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1235, 0.1);  mul_1235 = None
    mul_1237: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1307, 0.9)
    add_943: "f32[864]" = torch.ops.aten.add.Tensor(mul_1236, mul_1237);  mul_1236 = mul_1237 = None
    unsqueeze_704: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_680, -1)
    unsqueeze_705: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
    mul_1238: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1232, unsqueeze_705);  mul_1232 = unsqueeze_705 = None
    unsqueeze_706: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_681, -1);  primals_681 = None
    unsqueeze_707: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
    add_944: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1238, unsqueeze_707);  mul_1238 = unsqueeze_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_36 = torch.ops.aten.max_pool2d_with_indices.default(add_929, [3, 3], [1, 1], [1, 1])
    getitem_426: "f32[8, 864, 11, 11]" = max_pool2d_with_indices_36[0]
    getitem_427: "i64[8, 864, 11, 11]" = max_pool2d_with_indices_36[1];  max_pool2d_with_indices_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_945: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_944, getitem_426);  add_944 = getitem_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_175: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_934)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_327: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_175, primals_682, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_328: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_327, primals_683, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_946: "i64[]" = torch.ops.aten.add.Tensor(primals_1311, 1)
    var_mean_177 = torch.ops.aten.var_mean.correction(convolution_328, [0, 2, 3], correction = 0, keepdim = True)
    getitem_428: "f32[1, 864, 1, 1]" = var_mean_177[0]
    getitem_429: "f32[1, 864, 1, 1]" = var_mean_177[1];  var_mean_177 = None
    add_947: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_428, 0.001)
    rsqrt_177: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_947);  add_947 = None
    sub_177: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_328, getitem_429)
    mul_1239: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_177, rsqrt_177);  sub_177 = None
    squeeze_531: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_429, [0, 2, 3]);  getitem_429 = None
    squeeze_532: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_177, [0, 2, 3]);  rsqrt_177 = None
    mul_1240: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_531, 0.1)
    mul_1241: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1309, 0.9)
    add_948: "f32[864]" = torch.ops.aten.add.Tensor(mul_1240, mul_1241);  mul_1240 = mul_1241 = None
    squeeze_533: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_428, [0, 2, 3]);  getitem_428 = None
    mul_1242: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_533, 1.001034126163392);  squeeze_533 = None
    mul_1243: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1242, 0.1);  mul_1242 = None
    mul_1244: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1310, 0.9)
    add_949: "f32[864]" = torch.ops.aten.add.Tensor(mul_1243, mul_1244);  mul_1243 = mul_1244 = None
    unsqueeze_708: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_684, -1)
    unsqueeze_709: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
    mul_1245: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1239, unsqueeze_709);  mul_1239 = unsqueeze_709 = None
    unsqueeze_710: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_685, -1);  primals_685 = None
    unsqueeze_711: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
    add_950: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1245, unsqueeze_711);  mul_1245 = unsqueeze_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_176: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_950);  add_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_329: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_176, primals_686, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_330: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_329, primals_687, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_951: "i64[]" = torch.ops.aten.add.Tensor(primals_1314, 1)
    var_mean_178 = torch.ops.aten.var_mean.correction(convolution_330, [0, 2, 3], correction = 0, keepdim = True)
    getitem_430: "f32[1, 864, 1, 1]" = var_mean_178[0]
    getitem_431: "f32[1, 864, 1, 1]" = var_mean_178[1];  var_mean_178 = None
    add_952: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_430, 0.001)
    rsqrt_178: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_952);  add_952 = None
    sub_178: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_330, getitem_431)
    mul_1246: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_178, rsqrt_178);  sub_178 = None
    squeeze_534: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_431, [0, 2, 3]);  getitem_431 = None
    squeeze_535: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_178, [0, 2, 3]);  rsqrt_178 = None
    mul_1247: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_534, 0.1)
    mul_1248: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1312, 0.9)
    add_953: "f32[864]" = torch.ops.aten.add.Tensor(mul_1247, mul_1248);  mul_1247 = mul_1248 = None
    squeeze_536: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_430, [0, 2, 3]);  getitem_430 = None
    mul_1249: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_536, 1.001034126163392);  squeeze_536 = None
    mul_1250: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1249, 0.1);  mul_1249 = None
    mul_1251: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1313, 0.9)
    add_954: "f32[864]" = torch.ops.aten.add.Tensor(mul_1250, mul_1251);  mul_1250 = mul_1251 = None
    unsqueeze_712: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_688, -1)
    unsqueeze_713: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
    mul_1252: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1246, unsqueeze_713);  mul_1246 = unsqueeze_713 = None
    unsqueeze_714: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_689, -1);  primals_689 = None
    unsqueeze_715: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
    add_955: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1252, unsqueeze_715);  mul_1252 = unsqueeze_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_37 = torch.ops.aten.max_pool2d_with_indices.default(add_934, [3, 3], [1, 1], [1, 1])
    getitem_432: "f32[8, 864, 11, 11]" = max_pool2d_with_indices_37[0]
    getitem_433: "i64[8, 864, 11, 11]" = max_pool2d_with_indices_37[1];  max_pool2d_with_indices_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_956: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_955, getitem_432);  add_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_331: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_175, primals_690, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_332: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_331, primals_691, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_957: "i64[]" = torch.ops.aten.add.Tensor(primals_1317, 1)
    var_mean_179 = torch.ops.aten.var_mean.correction(convolution_332, [0, 2, 3], correction = 0, keepdim = True)
    getitem_434: "f32[1, 864, 1, 1]" = var_mean_179[0]
    getitem_435: "f32[1, 864, 1, 1]" = var_mean_179[1];  var_mean_179 = None
    add_958: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_434, 0.001)
    rsqrt_179: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_958);  add_958 = None
    sub_179: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_332, getitem_435)
    mul_1253: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_179, rsqrt_179);  sub_179 = None
    squeeze_537: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_435, [0, 2, 3]);  getitem_435 = None
    squeeze_538: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_179, [0, 2, 3]);  rsqrt_179 = None
    mul_1254: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_537, 0.1)
    mul_1255: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1315, 0.9)
    add_959: "f32[864]" = torch.ops.aten.add.Tensor(mul_1254, mul_1255);  mul_1254 = mul_1255 = None
    squeeze_539: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_434, [0, 2, 3]);  getitem_434 = None
    mul_1256: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_539, 1.001034126163392);  squeeze_539 = None
    mul_1257: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1256, 0.1);  mul_1256 = None
    mul_1258: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1316, 0.9)
    add_960: "f32[864]" = torch.ops.aten.add.Tensor(mul_1257, mul_1258);  mul_1257 = mul_1258 = None
    unsqueeze_716: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_692, -1)
    unsqueeze_717: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
    mul_1259: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1253, unsqueeze_717);  mul_1253 = unsqueeze_717 = None
    unsqueeze_718: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_693, -1);  primals_693 = None
    unsqueeze_719: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
    add_961: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1259, unsqueeze_719);  mul_1259 = unsqueeze_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_178: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_961);  add_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_333: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_178, primals_694, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_334: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_333, primals_695, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_962: "i64[]" = torch.ops.aten.add.Tensor(primals_1320, 1)
    var_mean_180 = torch.ops.aten.var_mean.correction(convolution_334, [0, 2, 3], correction = 0, keepdim = True)
    getitem_436: "f32[1, 864, 1, 1]" = var_mean_180[0]
    getitem_437: "f32[1, 864, 1, 1]" = var_mean_180[1];  var_mean_180 = None
    add_963: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_436, 0.001)
    rsqrt_180: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_963);  add_963 = None
    sub_180: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_334, getitem_437)
    mul_1260: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_180, rsqrt_180);  sub_180 = None
    squeeze_540: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_437, [0, 2, 3]);  getitem_437 = None
    squeeze_541: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_180, [0, 2, 3]);  rsqrt_180 = None
    mul_1261: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_540, 0.1)
    mul_1262: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1318, 0.9)
    add_964: "f32[864]" = torch.ops.aten.add.Tensor(mul_1261, mul_1262);  mul_1261 = mul_1262 = None
    squeeze_542: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_436, [0, 2, 3]);  getitem_436 = None
    mul_1263: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_542, 1.001034126163392);  squeeze_542 = None
    mul_1264: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1263, 0.1);  mul_1263 = None
    mul_1265: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1319, 0.9)
    add_965: "f32[864]" = torch.ops.aten.add.Tensor(mul_1264, mul_1265);  mul_1264 = mul_1265 = None
    unsqueeze_720: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_696, -1)
    unsqueeze_721: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
    mul_1266: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1260, unsqueeze_721);  mul_1260 = unsqueeze_721 = None
    unsqueeze_722: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_697, -1);  primals_697 = None
    unsqueeze_723: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
    add_966: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1266, unsqueeze_723);  mul_1266 = unsqueeze_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_335: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_175, primals_698, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_336: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_335, primals_699, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_967: "i64[]" = torch.ops.aten.add.Tensor(primals_1323, 1)
    var_mean_181 = torch.ops.aten.var_mean.correction(convolution_336, [0, 2, 3], correction = 0, keepdim = True)
    getitem_438: "f32[1, 864, 1, 1]" = var_mean_181[0]
    getitem_439: "f32[1, 864, 1, 1]" = var_mean_181[1];  var_mean_181 = None
    add_968: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_438, 0.001)
    rsqrt_181: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_968);  add_968 = None
    sub_181: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_336, getitem_439)
    mul_1267: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_181, rsqrt_181);  sub_181 = None
    squeeze_543: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_439, [0, 2, 3]);  getitem_439 = None
    squeeze_544: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_181, [0, 2, 3]);  rsqrt_181 = None
    mul_1268: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_543, 0.1)
    mul_1269: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1321, 0.9)
    add_969: "f32[864]" = torch.ops.aten.add.Tensor(mul_1268, mul_1269);  mul_1268 = mul_1269 = None
    squeeze_545: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_438, [0, 2, 3]);  getitem_438 = None
    mul_1270: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_545, 1.001034126163392);  squeeze_545 = None
    mul_1271: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1270, 0.1);  mul_1270 = None
    mul_1272: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1322, 0.9)
    add_970: "f32[864]" = torch.ops.aten.add.Tensor(mul_1271, mul_1272);  mul_1271 = mul_1272 = None
    unsqueeze_724: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_700, -1)
    unsqueeze_725: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
    mul_1273: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1267, unsqueeze_725);  mul_1267 = unsqueeze_725 = None
    unsqueeze_726: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_701, -1);  primals_701 = None
    unsqueeze_727: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
    add_971: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1273, unsqueeze_727);  mul_1273 = unsqueeze_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_180: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_971);  add_971 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_337: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_180, primals_702, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_338: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_337, primals_703, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_972: "i64[]" = torch.ops.aten.add.Tensor(primals_1326, 1)
    var_mean_182 = torch.ops.aten.var_mean.correction(convolution_338, [0, 2, 3], correction = 0, keepdim = True)
    getitem_440: "f32[1, 864, 1, 1]" = var_mean_182[0]
    getitem_441: "f32[1, 864, 1, 1]" = var_mean_182[1];  var_mean_182 = None
    add_973: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_440, 0.001)
    rsqrt_182: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_973);  add_973 = None
    sub_182: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_338, getitem_441)
    mul_1274: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_182, rsqrt_182);  sub_182 = None
    squeeze_546: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_441, [0, 2, 3]);  getitem_441 = None
    squeeze_547: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_182, [0, 2, 3]);  rsqrt_182 = None
    mul_1275: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_546, 0.1)
    mul_1276: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1324, 0.9)
    add_974: "f32[864]" = torch.ops.aten.add.Tensor(mul_1275, mul_1276);  mul_1275 = mul_1276 = None
    squeeze_548: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_440, [0, 2, 3]);  getitem_440 = None
    mul_1277: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_548, 1.001034126163392);  squeeze_548 = None
    mul_1278: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1277, 0.1);  mul_1277 = None
    mul_1279: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1325, 0.9)
    add_975: "f32[864]" = torch.ops.aten.add.Tensor(mul_1278, mul_1279);  mul_1278 = mul_1279 = None
    unsqueeze_728: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_704, -1)
    unsqueeze_729: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
    mul_1280: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1274, unsqueeze_729);  mul_1274 = unsqueeze_729 = None
    unsqueeze_730: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_705, -1);  primals_705 = None
    unsqueeze_731: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
    add_976: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1280, unsqueeze_731);  mul_1280 = unsqueeze_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_977: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_966, add_976);  add_966 = add_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_181: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_977)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_339: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_181, primals_706, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_340: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_339, primals_707, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_978: "i64[]" = torch.ops.aten.add.Tensor(primals_1329, 1)
    var_mean_183 = torch.ops.aten.var_mean.correction(convolution_340, [0, 2, 3], correction = 0, keepdim = True)
    getitem_442: "f32[1, 864, 1, 1]" = var_mean_183[0]
    getitem_443: "f32[1, 864, 1, 1]" = var_mean_183[1];  var_mean_183 = None
    add_979: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_442, 0.001)
    rsqrt_183: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_979);  add_979 = None
    sub_183: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_340, getitem_443)
    mul_1281: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_183, rsqrt_183);  sub_183 = None
    squeeze_549: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_443, [0, 2, 3]);  getitem_443 = None
    squeeze_550: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_183, [0, 2, 3]);  rsqrt_183 = None
    mul_1282: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_549, 0.1)
    mul_1283: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1327, 0.9)
    add_980: "f32[864]" = torch.ops.aten.add.Tensor(mul_1282, mul_1283);  mul_1282 = mul_1283 = None
    squeeze_551: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_442, [0, 2, 3]);  getitem_442 = None
    mul_1284: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_551, 1.001034126163392);  squeeze_551 = None
    mul_1285: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1284, 0.1);  mul_1284 = None
    mul_1286: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1328, 0.9)
    add_981: "f32[864]" = torch.ops.aten.add.Tensor(mul_1285, mul_1286);  mul_1285 = mul_1286 = None
    unsqueeze_732: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_708, -1)
    unsqueeze_733: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
    mul_1287: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1281, unsqueeze_733);  mul_1281 = unsqueeze_733 = None
    unsqueeze_734: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_709, -1);  primals_709 = None
    unsqueeze_735: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
    add_982: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1287, unsqueeze_735);  mul_1287 = unsqueeze_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_182: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_982);  add_982 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_341: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_182, primals_710, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_342: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_341, primals_711, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_983: "i64[]" = torch.ops.aten.add.Tensor(primals_1332, 1)
    var_mean_184 = torch.ops.aten.var_mean.correction(convolution_342, [0, 2, 3], correction = 0, keepdim = True)
    getitem_444: "f32[1, 864, 1, 1]" = var_mean_184[0]
    getitem_445: "f32[1, 864, 1, 1]" = var_mean_184[1];  var_mean_184 = None
    add_984: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_444, 0.001)
    rsqrt_184: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_984);  add_984 = None
    sub_184: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_342, getitem_445)
    mul_1288: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_184, rsqrt_184);  sub_184 = None
    squeeze_552: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_445, [0, 2, 3]);  getitem_445 = None
    squeeze_553: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_184, [0, 2, 3]);  rsqrt_184 = None
    mul_1289: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_552, 0.1)
    mul_1290: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1330, 0.9)
    add_985: "f32[864]" = torch.ops.aten.add.Tensor(mul_1289, mul_1290);  mul_1289 = mul_1290 = None
    squeeze_554: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_444, [0, 2, 3]);  getitem_444 = None
    mul_1291: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_554, 1.001034126163392);  squeeze_554 = None
    mul_1292: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1291, 0.1);  mul_1291 = None
    mul_1293: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1331, 0.9)
    add_986: "f32[864]" = torch.ops.aten.add.Tensor(mul_1292, mul_1293);  mul_1292 = mul_1293 = None
    unsqueeze_736: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_712, -1)
    unsqueeze_737: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
    mul_1294: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1288, unsqueeze_737);  mul_1288 = unsqueeze_737 = None
    unsqueeze_738: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_713, -1);  primals_713 = None
    unsqueeze_739: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
    add_987: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1294, unsqueeze_739);  mul_1294 = unsqueeze_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_988: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_987, getitem_432);  add_987 = getitem_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_343: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_173, primals_714, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_344: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_343, primals_715, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_989: "i64[]" = torch.ops.aten.add.Tensor(primals_1335, 1)
    var_mean_185 = torch.ops.aten.var_mean.correction(convolution_344, [0, 2, 3], correction = 0, keepdim = True)
    getitem_448: "f32[1, 864, 1, 1]" = var_mean_185[0]
    getitem_449: "f32[1, 864, 1, 1]" = var_mean_185[1];  var_mean_185 = None
    add_990: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_448, 0.001)
    rsqrt_185: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_990);  add_990 = None
    sub_185: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_344, getitem_449)
    mul_1295: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_185, rsqrt_185);  sub_185 = None
    squeeze_555: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_449, [0, 2, 3]);  getitem_449 = None
    squeeze_556: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_185, [0, 2, 3]);  rsqrt_185 = None
    mul_1296: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_555, 0.1)
    mul_1297: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1333, 0.9)
    add_991: "f32[864]" = torch.ops.aten.add.Tensor(mul_1296, mul_1297);  mul_1296 = mul_1297 = None
    squeeze_557: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_448, [0, 2, 3]);  getitem_448 = None
    mul_1298: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_557, 1.001034126163392);  squeeze_557 = None
    mul_1299: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1298, 0.1);  mul_1298 = None
    mul_1300: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1334, 0.9)
    add_992: "f32[864]" = torch.ops.aten.add.Tensor(mul_1299, mul_1300);  mul_1299 = mul_1300 = None
    unsqueeze_740: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_716, -1)
    unsqueeze_741: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
    mul_1301: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1295, unsqueeze_741);  mul_1295 = unsqueeze_741 = None
    unsqueeze_742: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_717, -1);  primals_717 = None
    unsqueeze_743: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
    add_993: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1301, unsqueeze_743);  mul_1301 = unsqueeze_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_184: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_993);  add_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_345: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_184, primals_718, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_346: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_345, primals_719, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_994: "i64[]" = torch.ops.aten.add.Tensor(primals_1338, 1)
    var_mean_186 = torch.ops.aten.var_mean.correction(convolution_346, [0, 2, 3], correction = 0, keepdim = True)
    getitem_450: "f32[1, 864, 1, 1]" = var_mean_186[0]
    getitem_451: "f32[1, 864, 1, 1]" = var_mean_186[1];  var_mean_186 = None
    add_995: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_450, 0.001)
    rsqrt_186: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_995);  add_995 = None
    sub_186: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_346, getitem_451)
    mul_1302: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_186, rsqrt_186);  sub_186 = None
    squeeze_558: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_451, [0, 2, 3]);  getitem_451 = None
    squeeze_559: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_186, [0, 2, 3]);  rsqrt_186 = None
    mul_1303: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_558, 0.1)
    mul_1304: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1336, 0.9)
    add_996: "f32[864]" = torch.ops.aten.add.Tensor(mul_1303, mul_1304);  mul_1303 = mul_1304 = None
    squeeze_560: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_450, [0, 2, 3]);  getitem_450 = None
    mul_1305: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_560, 1.001034126163392);  squeeze_560 = None
    mul_1306: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1305, 0.1);  mul_1305 = None
    mul_1307: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1337, 0.9)
    add_997: "f32[864]" = torch.ops.aten.add.Tensor(mul_1306, mul_1307);  mul_1306 = mul_1307 = None
    unsqueeze_744: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_720, -1)
    unsqueeze_745: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
    mul_1308: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1302, unsqueeze_745);  mul_1302 = unsqueeze_745 = None
    unsqueeze_746: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_721, -1);  primals_721 = None
    unsqueeze_747: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
    add_998: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1308, unsqueeze_747);  mul_1308 = unsqueeze_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_999: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_998, add_934);  add_998 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat_16: "f32[8, 4320, 11, 11]" = torch.ops.aten.cat.default([add_945, add_956, add_977, add_988, add_999], 1);  add_945 = add_956 = add_977 = add_988 = add_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_347: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_172, primals_722, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_1000: "i64[]" = torch.ops.aten.add.Tensor(primals_1341, 1)
    var_mean_187 = torch.ops.aten.var_mean.correction(convolution_347, [0, 2, 3], correction = 0, keepdim = True)
    getitem_452: "f32[1, 864, 1, 1]" = var_mean_187[0]
    getitem_453: "f32[1, 864, 1, 1]" = var_mean_187[1];  var_mean_187 = None
    add_1001: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_452, 0.001)
    rsqrt_187: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1001);  add_1001 = None
    sub_187: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_347, getitem_453)
    mul_1309: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_187, rsqrt_187);  sub_187 = None
    squeeze_561: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_453, [0, 2, 3]);  getitem_453 = None
    squeeze_562: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_187, [0, 2, 3]);  rsqrt_187 = None
    mul_1310: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_561, 0.1)
    mul_1311: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1339, 0.9)
    add_1002: "f32[864]" = torch.ops.aten.add.Tensor(mul_1310, mul_1311);  mul_1310 = mul_1311 = None
    squeeze_563: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_452, [0, 2, 3]);  getitem_452 = None
    mul_1312: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_563, 1.001034126163392);  squeeze_563 = None
    mul_1313: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1312, 0.1);  mul_1312 = None
    mul_1314: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1340, 0.9)
    add_1003: "f32[864]" = torch.ops.aten.add.Tensor(mul_1313, mul_1314);  mul_1313 = mul_1314 = None
    unsqueeze_748: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_723, -1)
    unsqueeze_749: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
    mul_1315: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1309, unsqueeze_749);  mul_1309 = unsqueeze_749 = None
    unsqueeze_750: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_724, -1);  primals_724 = None
    unsqueeze_751: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
    add_1004: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1315, unsqueeze_751);  mul_1315 = unsqueeze_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    relu_186: "f32[8, 4320, 11, 11]" = torch.ops.aten.relu.default(cat_16);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_348: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_186, primals_725, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    add_1005: "i64[]" = torch.ops.aten.add.Tensor(primals_1344, 1)
    var_mean_188 = torch.ops.aten.var_mean.correction(convolution_348, [0, 2, 3], correction = 0, keepdim = True)
    getitem_454: "f32[1, 864, 1, 1]" = var_mean_188[0]
    getitem_455: "f32[1, 864, 1, 1]" = var_mean_188[1];  var_mean_188 = None
    add_1006: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_454, 0.001)
    rsqrt_188: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1006);  add_1006 = None
    sub_188: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_348, getitem_455)
    mul_1316: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_188, rsqrt_188);  sub_188 = None
    squeeze_564: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_455, [0, 2, 3]);  getitem_455 = None
    squeeze_565: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_188, [0, 2, 3]);  rsqrt_188 = None
    mul_1317: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_564, 0.1)
    mul_1318: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1342, 0.9)
    add_1007: "f32[864]" = torch.ops.aten.add.Tensor(mul_1317, mul_1318);  mul_1317 = mul_1318 = None
    squeeze_566: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_454, [0, 2, 3]);  getitem_454 = None
    mul_1319: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_566, 1.001034126163392);  squeeze_566 = None
    mul_1320: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1319, 0.1);  mul_1319 = None
    mul_1321: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1343, 0.9)
    add_1008: "f32[864]" = torch.ops.aten.add.Tensor(mul_1320, mul_1321);  mul_1320 = mul_1321 = None
    unsqueeze_752: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_726, -1)
    unsqueeze_753: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
    mul_1322: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1316, unsqueeze_753);  mul_1316 = unsqueeze_753 = None
    unsqueeze_754: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_727, -1);  primals_727 = None
    unsqueeze_755: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
    add_1009: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1322, unsqueeze_755);  mul_1322 = unsqueeze_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_187: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_1004)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_349: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_187, primals_728, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_350: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_349, primals_729, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_1010: "i64[]" = torch.ops.aten.add.Tensor(primals_1347, 1)
    var_mean_189 = torch.ops.aten.var_mean.correction(convolution_350, [0, 2, 3], correction = 0, keepdim = True)
    getitem_456: "f32[1, 864, 1, 1]" = var_mean_189[0]
    getitem_457: "f32[1, 864, 1, 1]" = var_mean_189[1];  var_mean_189 = None
    add_1011: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_456, 0.001)
    rsqrt_189: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1011);  add_1011 = None
    sub_189: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_350, getitem_457)
    mul_1323: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_189, rsqrt_189);  sub_189 = None
    squeeze_567: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_457, [0, 2, 3]);  getitem_457 = None
    squeeze_568: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_189, [0, 2, 3]);  rsqrt_189 = None
    mul_1324: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_567, 0.1)
    mul_1325: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1345, 0.9)
    add_1012: "f32[864]" = torch.ops.aten.add.Tensor(mul_1324, mul_1325);  mul_1324 = mul_1325 = None
    squeeze_569: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_456, [0, 2, 3]);  getitem_456 = None
    mul_1326: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_569, 1.001034126163392);  squeeze_569 = None
    mul_1327: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1326, 0.1);  mul_1326 = None
    mul_1328: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1346, 0.9)
    add_1013: "f32[864]" = torch.ops.aten.add.Tensor(mul_1327, mul_1328);  mul_1327 = mul_1328 = None
    unsqueeze_756: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_730, -1)
    unsqueeze_757: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
    mul_1329: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1323, unsqueeze_757);  mul_1323 = unsqueeze_757 = None
    unsqueeze_758: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_731, -1);  primals_731 = None
    unsqueeze_759: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
    add_1014: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1329, unsqueeze_759);  mul_1329 = unsqueeze_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_188: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_1014);  add_1014 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_351: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_188, primals_732, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_352: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_351, primals_733, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_1015: "i64[]" = torch.ops.aten.add.Tensor(primals_1350, 1)
    var_mean_190 = torch.ops.aten.var_mean.correction(convolution_352, [0, 2, 3], correction = 0, keepdim = True)
    getitem_458: "f32[1, 864, 1, 1]" = var_mean_190[0]
    getitem_459: "f32[1, 864, 1, 1]" = var_mean_190[1];  var_mean_190 = None
    add_1016: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_458, 0.001)
    rsqrt_190: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1016);  add_1016 = None
    sub_190: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_352, getitem_459)
    mul_1330: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_190, rsqrt_190);  sub_190 = None
    squeeze_570: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_459, [0, 2, 3]);  getitem_459 = None
    squeeze_571: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_190, [0, 2, 3]);  rsqrt_190 = None
    mul_1331: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_570, 0.1)
    mul_1332: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1348, 0.9)
    add_1017: "f32[864]" = torch.ops.aten.add.Tensor(mul_1331, mul_1332);  mul_1331 = mul_1332 = None
    squeeze_572: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_458, [0, 2, 3]);  getitem_458 = None
    mul_1333: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_572, 1.001034126163392);  squeeze_572 = None
    mul_1334: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1333, 0.1);  mul_1333 = None
    mul_1335: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1349, 0.9)
    add_1018: "f32[864]" = torch.ops.aten.add.Tensor(mul_1334, mul_1335);  mul_1334 = mul_1335 = None
    unsqueeze_760: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_734, -1)
    unsqueeze_761: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
    mul_1336: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1330, unsqueeze_761);  mul_1330 = unsqueeze_761 = None
    unsqueeze_762: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_735, -1);  primals_735 = None
    unsqueeze_763: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
    add_1019: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1336, unsqueeze_763);  mul_1336 = unsqueeze_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_39 = torch.ops.aten.max_pool2d_with_indices.default(add_1004, [3, 3], [1, 1], [1, 1])
    getitem_460: "f32[8, 864, 11, 11]" = max_pool2d_with_indices_39[0]
    getitem_461: "i64[8, 864, 11, 11]" = max_pool2d_with_indices_39[1];  max_pool2d_with_indices_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    add_1020: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1019, getitem_460);  add_1019 = getitem_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_189: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_1009)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_353: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_189, primals_736, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_354: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_353, primals_737, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_1021: "i64[]" = torch.ops.aten.add.Tensor(primals_1353, 1)
    var_mean_191 = torch.ops.aten.var_mean.correction(convolution_354, [0, 2, 3], correction = 0, keepdim = True)
    getitem_462: "f32[1, 864, 1, 1]" = var_mean_191[0]
    getitem_463: "f32[1, 864, 1, 1]" = var_mean_191[1];  var_mean_191 = None
    add_1022: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_462, 0.001)
    rsqrt_191: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1022);  add_1022 = None
    sub_191: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_354, getitem_463)
    mul_1337: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_191, rsqrt_191);  sub_191 = None
    squeeze_573: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_463, [0, 2, 3]);  getitem_463 = None
    squeeze_574: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_191, [0, 2, 3]);  rsqrt_191 = None
    mul_1338: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_573, 0.1)
    mul_1339: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1351, 0.9)
    add_1023: "f32[864]" = torch.ops.aten.add.Tensor(mul_1338, mul_1339);  mul_1338 = mul_1339 = None
    squeeze_575: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_462, [0, 2, 3]);  getitem_462 = None
    mul_1340: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_575, 1.001034126163392);  squeeze_575 = None
    mul_1341: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1340, 0.1);  mul_1340 = None
    mul_1342: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1352, 0.9)
    add_1024: "f32[864]" = torch.ops.aten.add.Tensor(mul_1341, mul_1342);  mul_1341 = mul_1342 = None
    unsqueeze_764: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_738, -1)
    unsqueeze_765: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
    mul_1343: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1337, unsqueeze_765);  mul_1337 = unsqueeze_765 = None
    unsqueeze_766: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_739, -1);  primals_739 = None
    unsqueeze_767: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
    add_1025: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1343, unsqueeze_767);  mul_1343 = unsqueeze_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_190: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_1025);  add_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_355: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_190, primals_740, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_356: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_355, primals_741, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_1026: "i64[]" = torch.ops.aten.add.Tensor(primals_1356, 1)
    var_mean_192 = torch.ops.aten.var_mean.correction(convolution_356, [0, 2, 3], correction = 0, keepdim = True)
    getitem_464: "f32[1, 864, 1, 1]" = var_mean_192[0]
    getitem_465: "f32[1, 864, 1, 1]" = var_mean_192[1];  var_mean_192 = None
    add_1027: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_464, 0.001)
    rsqrt_192: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1027);  add_1027 = None
    sub_192: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_356, getitem_465)
    mul_1344: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_192, rsqrt_192);  sub_192 = None
    squeeze_576: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_465, [0, 2, 3]);  getitem_465 = None
    squeeze_577: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_192, [0, 2, 3]);  rsqrt_192 = None
    mul_1345: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_576, 0.1)
    mul_1346: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1354, 0.9)
    add_1028: "f32[864]" = torch.ops.aten.add.Tensor(mul_1345, mul_1346);  mul_1345 = mul_1346 = None
    squeeze_578: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_464, [0, 2, 3]);  getitem_464 = None
    mul_1347: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_578, 1.001034126163392);  squeeze_578 = None
    mul_1348: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1347, 0.1);  mul_1347 = None
    mul_1349: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1355, 0.9)
    add_1029: "f32[864]" = torch.ops.aten.add.Tensor(mul_1348, mul_1349);  mul_1348 = mul_1349 = None
    unsqueeze_768: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_742, -1)
    unsqueeze_769: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
    mul_1350: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1344, unsqueeze_769);  mul_1344 = unsqueeze_769 = None
    unsqueeze_770: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_743, -1);  primals_743 = None
    unsqueeze_771: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
    add_1030: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1350, unsqueeze_771);  mul_1350 = unsqueeze_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_40 = torch.ops.aten.max_pool2d_with_indices.default(add_1009, [3, 3], [1, 1], [1, 1])
    getitem_466: "f32[8, 864, 11, 11]" = max_pool2d_with_indices_40[0]
    getitem_467: "i64[8, 864, 11, 11]" = max_pool2d_with_indices_40[1];  max_pool2d_with_indices_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    add_1031: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1030, getitem_466);  add_1030 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_357: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_189, primals_744, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_358: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_357, primals_745, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_1032: "i64[]" = torch.ops.aten.add.Tensor(primals_1359, 1)
    var_mean_193 = torch.ops.aten.var_mean.correction(convolution_358, [0, 2, 3], correction = 0, keepdim = True)
    getitem_468: "f32[1, 864, 1, 1]" = var_mean_193[0]
    getitem_469: "f32[1, 864, 1, 1]" = var_mean_193[1];  var_mean_193 = None
    add_1033: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_468, 0.001)
    rsqrt_193: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1033);  add_1033 = None
    sub_193: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_358, getitem_469)
    mul_1351: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_193, rsqrt_193);  sub_193 = None
    squeeze_579: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_469, [0, 2, 3]);  getitem_469 = None
    squeeze_580: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_193, [0, 2, 3]);  rsqrt_193 = None
    mul_1352: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_579, 0.1)
    mul_1353: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1357, 0.9)
    add_1034: "f32[864]" = torch.ops.aten.add.Tensor(mul_1352, mul_1353);  mul_1352 = mul_1353 = None
    squeeze_581: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_468, [0, 2, 3]);  getitem_468 = None
    mul_1354: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_581, 1.001034126163392);  squeeze_581 = None
    mul_1355: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1354, 0.1);  mul_1354 = None
    mul_1356: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1358, 0.9)
    add_1035: "f32[864]" = torch.ops.aten.add.Tensor(mul_1355, mul_1356);  mul_1355 = mul_1356 = None
    unsqueeze_772: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_746, -1)
    unsqueeze_773: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
    mul_1357: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1351, unsqueeze_773);  mul_1351 = unsqueeze_773 = None
    unsqueeze_774: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_747, -1);  primals_747 = None
    unsqueeze_775: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
    add_1036: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1357, unsqueeze_775);  mul_1357 = unsqueeze_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_192: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_1036);  add_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_359: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_192, primals_748, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_360: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_359, primals_749, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_1037: "i64[]" = torch.ops.aten.add.Tensor(primals_1362, 1)
    var_mean_194 = torch.ops.aten.var_mean.correction(convolution_360, [0, 2, 3], correction = 0, keepdim = True)
    getitem_470: "f32[1, 864, 1, 1]" = var_mean_194[0]
    getitem_471: "f32[1, 864, 1, 1]" = var_mean_194[1];  var_mean_194 = None
    add_1038: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_470, 0.001)
    rsqrt_194: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1038);  add_1038 = None
    sub_194: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_360, getitem_471)
    mul_1358: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_194, rsqrt_194);  sub_194 = None
    squeeze_582: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_471, [0, 2, 3]);  getitem_471 = None
    squeeze_583: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_194, [0, 2, 3]);  rsqrt_194 = None
    mul_1359: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_582, 0.1)
    mul_1360: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1360, 0.9)
    add_1039: "f32[864]" = torch.ops.aten.add.Tensor(mul_1359, mul_1360);  mul_1359 = mul_1360 = None
    squeeze_584: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_470, [0, 2, 3]);  getitem_470 = None
    mul_1361: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_584, 1.001034126163392);  squeeze_584 = None
    mul_1362: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1361, 0.1);  mul_1361 = None
    mul_1363: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1361, 0.9)
    add_1040: "f32[864]" = torch.ops.aten.add.Tensor(mul_1362, mul_1363);  mul_1362 = mul_1363 = None
    unsqueeze_776: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_750, -1)
    unsqueeze_777: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
    mul_1364: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1358, unsqueeze_777);  mul_1358 = unsqueeze_777 = None
    unsqueeze_778: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_751, -1);  primals_751 = None
    unsqueeze_779: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
    add_1041: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1364, unsqueeze_779);  mul_1364 = unsqueeze_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_361: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_189, primals_752, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_362: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_361, primals_753, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_1042: "i64[]" = torch.ops.aten.add.Tensor(primals_1365, 1)
    var_mean_195 = torch.ops.aten.var_mean.correction(convolution_362, [0, 2, 3], correction = 0, keepdim = True)
    getitem_472: "f32[1, 864, 1, 1]" = var_mean_195[0]
    getitem_473: "f32[1, 864, 1, 1]" = var_mean_195[1];  var_mean_195 = None
    add_1043: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_472, 0.001)
    rsqrt_195: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1043);  add_1043 = None
    sub_195: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_362, getitem_473)
    mul_1365: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_195, rsqrt_195);  sub_195 = None
    squeeze_585: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_473, [0, 2, 3]);  getitem_473 = None
    squeeze_586: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_195, [0, 2, 3]);  rsqrt_195 = None
    mul_1366: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_585, 0.1)
    mul_1367: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1363, 0.9)
    add_1044: "f32[864]" = torch.ops.aten.add.Tensor(mul_1366, mul_1367);  mul_1366 = mul_1367 = None
    squeeze_587: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_472, [0, 2, 3]);  getitem_472 = None
    mul_1368: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_587, 1.001034126163392);  squeeze_587 = None
    mul_1369: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1368, 0.1);  mul_1368 = None
    mul_1370: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1364, 0.9)
    add_1045: "f32[864]" = torch.ops.aten.add.Tensor(mul_1369, mul_1370);  mul_1369 = mul_1370 = None
    unsqueeze_780: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_754, -1)
    unsqueeze_781: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
    mul_1371: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1365, unsqueeze_781);  mul_1365 = unsqueeze_781 = None
    unsqueeze_782: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_755, -1);  primals_755 = None
    unsqueeze_783: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
    add_1046: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1371, unsqueeze_783);  mul_1371 = unsqueeze_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_194: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_1046);  add_1046 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_363: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_194, primals_756, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_364: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_363, primals_757, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_1047: "i64[]" = torch.ops.aten.add.Tensor(primals_1368, 1)
    var_mean_196 = torch.ops.aten.var_mean.correction(convolution_364, [0, 2, 3], correction = 0, keepdim = True)
    getitem_474: "f32[1, 864, 1, 1]" = var_mean_196[0]
    getitem_475: "f32[1, 864, 1, 1]" = var_mean_196[1];  var_mean_196 = None
    add_1048: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_474, 0.001)
    rsqrt_196: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1048);  add_1048 = None
    sub_196: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_364, getitem_475)
    mul_1372: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_196, rsqrt_196);  sub_196 = None
    squeeze_588: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_475, [0, 2, 3]);  getitem_475 = None
    squeeze_589: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_196, [0, 2, 3]);  rsqrt_196 = None
    mul_1373: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_588, 0.1)
    mul_1374: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1366, 0.9)
    add_1049: "f32[864]" = torch.ops.aten.add.Tensor(mul_1373, mul_1374);  mul_1373 = mul_1374 = None
    squeeze_590: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_474, [0, 2, 3]);  getitem_474 = None
    mul_1375: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_590, 1.001034126163392);  squeeze_590 = None
    mul_1376: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1375, 0.1);  mul_1375 = None
    mul_1377: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1367, 0.9)
    add_1050: "f32[864]" = torch.ops.aten.add.Tensor(mul_1376, mul_1377);  mul_1376 = mul_1377 = None
    unsqueeze_784: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_758, -1)
    unsqueeze_785: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
    mul_1378: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1372, unsqueeze_785);  mul_1372 = unsqueeze_785 = None
    unsqueeze_786: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_759, -1);  primals_759 = None
    unsqueeze_787: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
    add_1051: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1378, unsqueeze_787);  mul_1378 = unsqueeze_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    add_1052: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1041, add_1051);  add_1041 = add_1051 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    relu_195: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_1052)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_365: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_195, primals_760, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_366: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_365, primals_761, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_1053: "i64[]" = torch.ops.aten.add.Tensor(primals_1371, 1)
    var_mean_197 = torch.ops.aten.var_mean.correction(convolution_366, [0, 2, 3], correction = 0, keepdim = True)
    getitem_476: "f32[1, 864, 1, 1]" = var_mean_197[0]
    getitem_477: "f32[1, 864, 1, 1]" = var_mean_197[1];  var_mean_197 = None
    add_1054: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_476, 0.001)
    rsqrt_197: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1054);  add_1054 = None
    sub_197: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_366, getitem_477)
    mul_1379: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_197, rsqrt_197);  sub_197 = None
    squeeze_591: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_477, [0, 2, 3]);  getitem_477 = None
    squeeze_592: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_197, [0, 2, 3]);  rsqrt_197 = None
    mul_1380: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_591, 0.1)
    mul_1381: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1369, 0.9)
    add_1055: "f32[864]" = torch.ops.aten.add.Tensor(mul_1380, mul_1381);  mul_1380 = mul_1381 = None
    squeeze_593: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_476, [0, 2, 3]);  getitem_476 = None
    mul_1382: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_593, 1.001034126163392);  squeeze_593 = None
    mul_1383: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1382, 0.1);  mul_1382 = None
    mul_1384: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1370, 0.9)
    add_1056: "f32[864]" = torch.ops.aten.add.Tensor(mul_1383, mul_1384);  mul_1383 = mul_1384 = None
    unsqueeze_788: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_762, -1)
    unsqueeze_789: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
    mul_1385: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1379, unsqueeze_789);  mul_1379 = unsqueeze_789 = None
    unsqueeze_790: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_763, -1);  primals_763 = None
    unsqueeze_791: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
    add_1057: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1385, unsqueeze_791);  mul_1385 = unsqueeze_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_196: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_1057);  add_1057 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_367: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_196, primals_764, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_368: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_367, primals_765, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_1058: "i64[]" = torch.ops.aten.add.Tensor(primals_1374, 1)
    var_mean_198 = torch.ops.aten.var_mean.correction(convolution_368, [0, 2, 3], correction = 0, keepdim = True)
    getitem_478: "f32[1, 864, 1, 1]" = var_mean_198[0]
    getitem_479: "f32[1, 864, 1, 1]" = var_mean_198[1];  var_mean_198 = None
    add_1059: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_478, 0.001)
    rsqrt_198: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1059);  add_1059 = None
    sub_198: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_368, getitem_479)
    mul_1386: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_198, rsqrt_198);  sub_198 = None
    squeeze_594: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_479, [0, 2, 3]);  getitem_479 = None
    squeeze_595: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_198, [0, 2, 3]);  rsqrt_198 = None
    mul_1387: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_594, 0.1)
    mul_1388: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1372, 0.9)
    add_1060: "f32[864]" = torch.ops.aten.add.Tensor(mul_1387, mul_1388);  mul_1387 = mul_1388 = None
    squeeze_596: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_478, [0, 2, 3]);  getitem_478 = None
    mul_1389: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_596, 1.001034126163392);  squeeze_596 = None
    mul_1390: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1389, 0.1);  mul_1389 = None
    mul_1391: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1373, 0.9)
    add_1061: "f32[864]" = torch.ops.aten.add.Tensor(mul_1390, mul_1391);  mul_1390 = mul_1391 = None
    unsqueeze_792: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_766, -1)
    unsqueeze_793: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
    mul_1392: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1386, unsqueeze_793);  mul_1386 = unsqueeze_793 = None
    unsqueeze_794: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_767, -1);  primals_767 = None
    unsqueeze_795: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
    add_1062: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1392, unsqueeze_795);  mul_1392 = unsqueeze_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    add_1063: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1062, getitem_466);  add_1062 = getitem_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_369: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_187, primals_768, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_370: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_369, primals_769, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    add_1064: "i64[]" = torch.ops.aten.add.Tensor(primals_1377, 1)
    var_mean_199 = torch.ops.aten.var_mean.correction(convolution_370, [0, 2, 3], correction = 0, keepdim = True)
    getitem_482: "f32[1, 864, 1, 1]" = var_mean_199[0]
    getitem_483: "f32[1, 864, 1, 1]" = var_mean_199[1];  var_mean_199 = None
    add_1065: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_482, 0.001)
    rsqrt_199: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1065);  add_1065 = None
    sub_199: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_370, getitem_483)
    mul_1393: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_199, rsqrt_199);  sub_199 = None
    squeeze_597: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_483, [0, 2, 3]);  getitem_483 = None
    squeeze_598: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_199, [0, 2, 3]);  rsqrt_199 = None
    mul_1394: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_597, 0.1)
    mul_1395: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1375, 0.9)
    add_1066: "f32[864]" = torch.ops.aten.add.Tensor(mul_1394, mul_1395);  mul_1394 = mul_1395 = None
    squeeze_599: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_482, [0, 2, 3]);  getitem_482 = None
    mul_1396: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_599, 1.001034126163392);  squeeze_599 = None
    mul_1397: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1396, 0.1);  mul_1396 = None
    mul_1398: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1376, 0.9)
    add_1067: "f32[864]" = torch.ops.aten.add.Tensor(mul_1397, mul_1398);  mul_1397 = mul_1398 = None
    unsqueeze_796: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_770, -1)
    unsqueeze_797: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
    mul_1399: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1393, unsqueeze_797);  mul_1393 = unsqueeze_797 = None
    unsqueeze_798: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_771, -1);  primals_771 = None
    unsqueeze_799: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
    add_1068: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1399, unsqueeze_799);  mul_1399 = unsqueeze_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    relu_198: "f32[8, 864, 11, 11]" = torch.ops.aten.relu.default(add_1068);  add_1068 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_371: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(relu_198, primals_772, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_372: "f32[8, 864, 11, 11]" = torch.ops.aten.convolution.default(convolution_371, primals_773, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    add_1069: "i64[]" = torch.ops.aten.add.Tensor(primals_1380, 1)
    var_mean_200 = torch.ops.aten.var_mean.correction(convolution_372, [0, 2, 3], correction = 0, keepdim = True)
    getitem_484: "f32[1, 864, 1, 1]" = var_mean_200[0]
    getitem_485: "f32[1, 864, 1, 1]" = var_mean_200[1];  var_mean_200 = None
    add_1070: "f32[1, 864, 1, 1]" = torch.ops.aten.add.Tensor(getitem_484, 0.001)
    rsqrt_200: "f32[1, 864, 1, 1]" = torch.ops.aten.rsqrt.default(add_1070);  add_1070 = None
    sub_200: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_372, getitem_485)
    mul_1400: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_200, rsqrt_200);  sub_200 = None
    squeeze_600: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_485, [0, 2, 3]);  getitem_485 = None
    squeeze_601: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_200, [0, 2, 3]);  rsqrt_200 = None
    mul_1401: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_600, 0.1)
    mul_1402: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1378, 0.9)
    add_1071: "f32[864]" = torch.ops.aten.add.Tensor(mul_1401, mul_1402);  mul_1401 = mul_1402 = None
    squeeze_602: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_484, [0, 2, 3]);  getitem_484 = None
    mul_1403: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_602, 1.001034126163392);  squeeze_602 = None
    mul_1404: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1403, 0.1);  mul_1403 = None
    mul_1405: "f32[864]" = torch.ops.aten.mul.Tensor(primals_1379, 0.9)
    add_1072: "f32[864]" = torch.ops.aten.add.Tensor(mul_1404, mul_1405);  mul_1404 = mul_1405 = None
    unsqueeze_800: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_774, -1)
    unsqueeze_801: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
    mul_1406: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(mul_1400, unsqueeze_801);  mul_1400 = unsqueeze_801 = None
    unsqueeze_802: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_775, -1);  primals_775 = None
    unsqueeze_803: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
    add_1073: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(mul_1406, unsqueeze_803);  mul_1406 = unsqueeze_803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    add_1074: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1073, add_1009);  add_1073 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    cat_17: "f32[8, 4320, 11, 11]" = torch.ops.aten.cat.default([add_1020, add_1031, add_1052, add_1063, add_1074], 1);  add_1020 = add_1031 = add_1052 = add_1063 = add_1074 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:331, code: x = self.act(x_cell_11)
    relu_199: "f32[8, 4320, 11, 11]" = torch.ops.aten.relu.default(cat_17);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 4320, 1, 1]" = torch.ops.aten.mean.dim(relu_199, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 4320]" = torch.ops.aten.reshape.default(mean, [8, 4320]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:337, code: return x if pre_logits else self.last_linear(x)
    permute: "f32[4320, 1000]" = torch.ops.aten.permute.default(primals_776, [1, 0]);  primals_776 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_777, view, permute);  primals_777 = None
    permute_1: "f32[1000, 4320]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:331, code: x = self.act(x_cell_11)
    le: "b8[8, 4320, 11, 11]" = torch.ops.aten.le.Scalar(relu_199, 0);  relu_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_804: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_600, 0);  squeeze_600 = None
    unsqueeze_805: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 2);  unsqueeze_804 = None
    unsqueeze_806: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 3);  unsqueeze_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_816: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_597, 0);  squeeze_597 = None
    unsqueeze_817: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 2);  unsqueeze_816 = None
    unsqueeze_818: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 3);  unsqueeze_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_828: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_594, 0);  squeeze_594 = None
    unsqueeze_829: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 2);  unsqueeze_828 = None
    unsqueeze_830: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 3);  unsqueeze_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_840: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_591, 0);  squeeze_591 = None
    unsqueeze_841: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 2);  unsqueeze_840 = None
    unsqueeze_842: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 3);  unsqueeze_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_852: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_588, 0);  squeeze_588 = None
    unsqueeze_853: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 2);  unsqueeze_852 = None
    unsqueeze_854: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 3);  unsqueeze_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_864: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_585, 0);  squeeze_585 = None
    unsqueeze_865: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 2);  unsqueeze_864 = None
    unsqueeze_866: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 3);  unsqueeze_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_876: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_582, 0);  squeeze_582 = None
    unsqueeze_877: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 2);  unsqueeze_876 = None
    unsqueeze_878: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 3);  unsqueeze_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_888: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_579, 0);  squeeze_579 = None
    unsqueeze_889: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 2);  unsqueeze_888 = None
    unsqueeze_890: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 3);  unsqueeze_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_900: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_576, 0);  squeeze_576 = None
    unsqueeze_901: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 2);  unsqueeze_900 = None
    unsqueeze_902: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 3);  unsqueeze_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_912: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_573, 0);  squeeze_573 = None
    unsqueeze_913: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 2);  unsqueeze_912 = None
    unsqueeze_914: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 3);  unsqueeze_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_924: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_570, 0);  squeeze_570 = None
    unsqueeze_925: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 2);  unsqueeze_924 = None
    unsqueeze_926: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 3);  unsqueeze_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_936: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_567, 0);  squeeze_567 = None
    unsqueeze_937: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 2);  unsqueeze_936 = None
    unsqueeze_938: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 3);  unsqueeze_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_948: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_564, 0);  squeeze_564 = None
    unsqueeze_949: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 2);  unsqueeze_948 = None
    unsqueeze_950: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 3);  unsqueeze_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_960: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_561, 0);  squeeze_561 = None
    unsqueeze_961: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 2);  unsqueeze_960 = None
    unsqueeze_962: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 3);  unsqueeze_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_972: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_558, 0);  squeeze_558 = None
    unsqueeze_973: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 2);  unsqueeze_972 = None
    unsqueeze_974: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 3);  unsqueeze_973 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_984: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_555, 0);  squeeze_555 = None
    unsqueeze_985: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 2);  unsqueeze_984 = None
    unsqueeze_986: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 3);  unsqueeze_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_996: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_552, 0);  squeeze_552 = None
    unsqueeze_997: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 2);  unsqueeze_996 = None
    unsqueeze_998: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 3);  unsqueeze_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1008: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_549, 0);  squeeze_549 = None
    unsqueeze_1009: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 2);  unsqueeze_1008 = None
    unsqueeze_1010: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 3);  unsqueeze_1009 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1020: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_546, 0);  squeeze_546 = None
    unsqueeze_1021: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 2);  unsqueeze_1020 = None
    unsqueeze_1022: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 3);  unsqueeze_1021 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1032: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_543, 0);  squeeze_543 = None
    unsqueeze_1033: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 2);  unsqueeze_1032 = None
    unsqueeze_1034: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 3);  unsqueeze_1033 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1044: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_540, 0);  squeeze_540 = None
    unsqueeze_1045: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, 2);  unsqueeze_1044 = None
    unsqueeze_1046: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 3);  unsqueeze_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1056: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_537, 0);  squeeze_537 = None
    unsqueeze_1057: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, 2);  unsqueeze_1056 = None
    unsqueeze_1058: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 3);  unsqueeze_1057 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1068: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_534, 0);  squeeze_534 = None
    unsqueeze_1069: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, 2);  unsqueeze_1068 = None
    unsqueeze_1070: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 3);  unsqueeze_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1080: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_531, 0);  squeeze_531 = None
    unsqueeze_1081: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, 2);  unsqueeze_1080 = None
    unsqueeze_1082: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1081, 3);  unsqueeze_1081 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1092: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_528, 0);  squeeze_528 = None
    unsqueeze_1093: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, 2);  unsqueeze_1092 = None
    unsqueeze_1094: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1093, 3);  unsqueeze_1093 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1104: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_525, 0);  squeeze_525 = None
    unsqueeze_1105: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, 2);  unsqueeze_1104 = None
    unsqueeze_1106: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1105, 3);  unsqueeze_1105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_1116: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_522, 0);  squeeze_522 = None
    unsqueeze_1117: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, 2);  unsqueeze_1116 = None
    unsqueeze_1118: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1117, 3);  unsqueeze_1117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_1128: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_519, 0);  squeeze_519 = None
    unsqueeze_1129: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, 2);  unsqueeze_1128 = None
    unsqueeze_1130: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1129, 3);  unsqueeze_1129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1140: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_516, 0);  squeeze_516 = None
    unsqueeze_1141: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, 2);  unsqueeze_1140 = None
    unsqueeze_1142: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1141, 3);  unsqueeze_1141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1152: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_513, 0);  squeeze_513 = None
    unsqueeze_1153: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, 2);  unsqueeze_1152 = None
    unsqueeze_1154: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1153, 3);  unsqueeze_1153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1164: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_510, 0);  squeeze_510 = None
    unsqueeze_1165: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, 2);  unsqueeze_1164 = None
    unsqueeze_1166: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1165, 3);  unsqueeze_1165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1176: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_507, 0);  squeeze_507 = None
    unsqueeze_1177: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, 2);  unsqueeze_1176 = None
    unsqueeze_1178: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1177, 3);  unsqueeze_1177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1188: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_504, 0);  squeeze_504 = None
    unsqueeze_1189: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, 2);  unsqueeze_1188 = None
    unsqueeze_1190: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1189, 3);  unsqueeze_1189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1200: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_501, 0);  squeeze_501 = None
    unsqueeze_1201: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, 2);  unsqueeze_1200 = None
    unsqueeze_1202: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1201, 3);  unsqueeze_1201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1212: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_498, 0);  squeeze_498 = None
    unsqueeze_1213: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, 2);  unsqueeze_1212 = None
    unsqueeze_1214: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1213, 3);  unsqueeze_1213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1224: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_495, 0);  squeeze_495 = None
    unsqueeze_1225: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, 2);  unsqueeze_1224 = None
    unsqueeze_1226: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1225, 3);  unsqueeze_1225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1236: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_492, 0);  squeeze_492 = None
    unsqueeze_1237: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, 2);  unsqueeze_1236 = None
    unsqueeze_1238: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1237, 3);  unsqueeze_1237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1248: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_489, 0);  squeeze_489 = None
    unsqueeze_1249: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, 2);  unsqueeze_1248 = None
    unsqueeze_1250: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1249, 3);  unsqueeze_1249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1260: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_486, 0);  squeeze_486 = None
    unsqueeze_1261: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, 2);  unsqueeze_1260 = None
    unsqueeze_1262: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1261, 3);  unsqueeze_1261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1272: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_483, 0);  squeeze_483 = None
    unsqueeze_1273: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, 2);  unsqueeze_1272 = None
    unsqueeze_1274: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1273, 3);  unsqueeze_1273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_1284: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_480, 0);  squeeze_480 = None
    unsqueeze_1285: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, 2);  unsqueeze_1284 = None
    unsqueeze_1286: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1285, 3);  unsqueeze_1285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    unsqueeze_1296: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_477, 0);  squeeze_477 = None
    unsqueeze_1297: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, 2);  unsqueeze_1296 = None
    unsqueeze_1298: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1297, 3);  unsqueeze_1297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_1308: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_474, 0);  squeeze_474 = None
    unsqueeze_1309: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, 2);  unsqueeze_1308 = None
    unsqueeze_1310: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1309, 3);  unsqueeze_1309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_43: "b8[8, 864, 21, 21]" = torch.ops.aten.le.Scalar(relu_146, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1320: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_471, 0);  squeeze_471 = None
    unsqueeze_1321: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, 2);  unsqueeze_1320 = None
    unsqueeze_1322: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1321, 3);  unsqueeze_1321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1332: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_468, 0);  squeeze_468 = None
    unsqueeze_1333: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, 2);  unsqueeze_1332 = None
    unsqueeze_1334: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1333, 3);  unsqueeze_1333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_45: "b8[8, 864, 21, 21]" = torch.ops.aten.le.Scalar(relu_144, 0);  relu_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1344: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_465, 0);  squeeze_465 = None
    unsqueeze_1345: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, 2);  unsqueeze_1344 = None
    unsqueeze_1346: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1345, 3);  unsqueeze_1345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1356: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_462, 0);  squeeze_462 = None
    unsqueeze_1357: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, 2);  unsqueeze_1356 = None
    unsqueeze_1358: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1357, 3);  unsqueeze_1357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1368: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_459, 0);  squeeze_459 = None
    unsqueeze_1369: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, 2);  unsqueeze_1368 = None
    unsqueeze_1370: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1369, 3);  unsqueeze_1369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1380: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_456, 0);  squeeze_456 = None
    unsqueeze_1381: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, 2);  unsqueeze_1380 = None
    unsqueeze_1382: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1381, 3);  unsqueeze_1381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1392: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_453, 0);  squeeze_453 = None
    unsqueeze_1393: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1392, 2);  unsqueeze_1392 = None
    unsqueeze_1394: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1393, 3);  unsqueeze_1393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1404: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_450, 0);  squeeze_450 = None
    unsqueeze_1405: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1404, 2);  unsqueeze_1404 = None
    unsqueeze_1406: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1405, 3);  unsqueeze_1405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1416: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_447, 0);  squeeze_447 = None
    unsqueeze_1417: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1416, 2);  unsqueeze_1416 = None
    unsqueeze_1418: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1417, 3);  unsqueeze_1417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1428: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_444, 0);  squeeze_444 = None
    unsqueeze_1429: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1428, 2);  unsqueeze_1428 = None
    unsqueeze_1430: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1429, 3);  unsqueeze_1429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1440: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_441, 0);  squeeze_441 = None
    unsqueeze_1441: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1440, 2);  unsqueeze_1440 = None
    unsqueeze_1442: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1441, 3);  unsqueeze_1441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1452: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_438, 0);  squeeze_438 = None
    unsqueeze_1453: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1452, 2);  unsqueeze_1452 = None
    unsqueeze_1454: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1453, 3);  unsqueeze_1453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_1464: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_435, 0);  squeeze_435 = None
    unsqueeze_1465: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1464, 2);  unsqueeze_1464 = None
    unsqueeze_1466: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1465, 3);  unsqueeze_1465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_1476: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_432, 0);  squeeze_432 = None
    unsqueeze_1477: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1476, 2);  unsqueeze_1476 = None
    unsqueeze_1478: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1477, 3);  unsqueeze_1477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1488: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_429, 0);  squeeze_429 = None
    unsqueeze_1489: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1488, 2);  unsqueeze_1488 = None
    unsqueeze_1490: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1489, 3);  unsqueeze_1489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1500: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_426, 0);  squeeze_426 = None
    unsqueeze_1501: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1500, 2);  unsqueeze_1500 = None
    unsqueeze_1502: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1501, 3);  unsqueeze_1501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1512: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_423, 0);  squeeze_423 = None
    unsqueeze_1513: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1512, 2);  unsqueeze_1512 = None
    unsqueeze_1514: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1513, 3);  unsqueeze_1513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1524: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_420, 0);  squeeze_420 = None
    unsqueeze_1525: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1524, 2);  unsqueeze_1524 = None
    unsqueeze_1526: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1525, 3);  unsqueeze_1525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1536: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_417, 0);  squeeze_417 = None
    unsqueeze_1537: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1536, 2);  unsqueeze_1536 = None
    unsqueeze_1538: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1537, 3);  unsqueeze_1537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1548: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_414, 0);  squeeze_414 = None
    unsqueeze_1549: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1548, 2);  unsqueeze_1548 = None
    unsqueeze_1550: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1549, 3);  unsqueeze_1549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1560: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_411, 0);  squeeze_411 = None
    unsqueeze_1561: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1560, 2);  unsqueeze_1560 = None
    unsqueeze_1562: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1561, 3);  unsqueeze_1561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1572: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_408, 0);  squeeze_408 = None
    unsqueeze_1573: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1572, 2);  unsqueeze_1572 = None
    unsqueeze_1574: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1573, 3);  unsqueeze_1573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1584: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_405, 0);  squeeze_405 = None
    unsqueeze_1585: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1584, 2);  unsqueeze_1584 = None
    unsqueeze_1586: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1585, 3);  unsqueeze_1585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1596: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_402, 0);  squeeze_402 = None
    unsqueeze_1597: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1596, 2);  unsqueeze_1596 = None
    unsqueeze_1598: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1597, 3);  unsqueeze_1597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1608: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_399, 0);  squeeze_399 = None
    unsqueeze_1609: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1608, 2);  unsqueeze_1608 = None
    unsqueeze_1610: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1609, 3);  unsqueeze_1609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1620: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_396, 0);  squeeze_396 = None
    unsqueeze_1621: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1620, 2);  unsqueeze_1620 = None
    unsqueeze_1622: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1621, 3);  unsqueeze_1621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_1632: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_393, 0);  squeeze_393 = None
    unsqueeze_1633: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1632, 2);  unsqueeze_1632 = None
    unsqueeze_1634: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1633, 3);  unsqueeze_1633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_1644: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_390, 0);  squeeze_390 = None
    unsqueeze_1645: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1644, 2);  unsqueeze_1644 = None
    unsqueeze_1646: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1645, 3);  unsqueeze_1645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1656: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_387, 0);  squeeze_387 = None
    unsqueeze_1657: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1656, 2);  unsqueeze_1656 = None
    unsqueeze_1658: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1657, 3);  unsqueeze_1657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1668: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_384, 0);  squeeze_384 = None
    unsqueeze_1669: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1668, 2);  unsqueeze_1668 = None
    unsqueeze_1670: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1669, 3);  unsqueeze_1669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1680: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_381, 0);  squeeze_381 = None
    unsqueeze_1681: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1680, 2);  unsqueeze_1680 = None
    unsqueeze_1682: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1681, 3);  unsqueeze_1681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1692: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_378, 0);  squeeze_378 = None
    unsqueeze_1693: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1692, 2);  unsqueeze_1692 = None
    unsqueeze_1694: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1693, 3);  unsqueeze_1693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1704: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_375, 0);  squeeze_375 = None
    unsqueeze_1705: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1704, 2);  unsqueeze_1704 = None
    unsqueeze_1706: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1705, 3);  unsqueeze_1705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1716: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_372, 0);  squeeze_372 = None
    unsqueeze_1717: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1716, 2);  unsqueeze_1716 = None
    unsqueeze_1718: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1717, 3);  unsqueeze_1717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1728: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_369, 0);  squeeze_369 = None
    unsqueeze_1729: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1728, 2);  unsqueeze_1728 = None
    unsqueeze_1730: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1729, 3);  unsqueeze_1729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1740: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_366, 0);  squeeze_366 = None
    unsqueeze_1741: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1740, 2);  unsqueeze_1740 = None
    unsqueeze_1742: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1741, 3);  unsqueeze_1741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1752: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_363, 0);  squeeze_363 = None
    unsqueeze_1753: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1752, 2);  unsqueeze_1752 = None
    unsqueeze_1754: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1753, 3);  unsqueeze_1753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1764: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_360, 0);  squeeze_360 = None
    unsqueeze_1765: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1764, 2);  unsqueeze_1764 = None
    unsqueeze_1766: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1765, 3);  unsqueeze_1765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1776: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_357, 0);  squeeze_357 = None
    unsqueeze_1777: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1776, 2);  unsqueeze_1776 = None
    unsqueeze_1778: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1777, 3);  unsqueeze_1777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1788: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_354, 0);  squeeze_354 = None
    unsqueeze_1789: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1788, 2);  unsqueeze_1788 = None
    unsqueeze_1790: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1789, 3);  unsqueeze_1789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_1800: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_351, 0);  squeeze_351 = None
    unsqueeze_1801: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1800, 2);  unsqueeze_1800 = None
    unsqueeze_1802: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1801, 3);  unsqueeze_1801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_1812: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_348, 0);  squeeze_348 = None
    unsqueeze_1813: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1812, 2);  unsqueeze_1812 = None
    unsqueeze_1814: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1813, 3);  unsqueeze_1813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1824: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_345, 0);  squeeze_345 = None
    unsqueeze_1825: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1824, 2);  unsqueeze_1824 = None
    unsqueeze_1826: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1825, 3);  unsqueeze_1825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1836: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_342, 0);  squeeze_342 = None
    unsqueeze_1837: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1836, 2);  unsqueeze_1836 = None
    unsqueeze_1838: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1837, 3);  unsqueeze_1837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1848: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_339, 0);  squeeze_339 = None
    unsqueeze_1849: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1848, 2);  unsqueeze_1848 = None
    unsqueeze_1850: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1849, 3);  unsqueeze_1849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1860: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_336, 0);  squeeze_336 = None
    unsqueeze_1861: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1860, 2);  unsqueeze_1860 = None
    unsqueeze_1862: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1861, 3);  unsqueeze_1861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1872: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_333, 0);  squeeze_333 = None
    unsqueeze_1873: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1872, 2);  unsqueeze_1872 = None
    unsqueeze_1874: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1873, 3);  unsqueeze_1873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1884: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_330, 0);  squeeze_330 = None
    unsqueeze_1885: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1884, 2);  unsqueeze_1884 = None
    unsqueeze_1886: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1885, 3);  unsqueeze_1885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1896: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_327, 0);  squeeze_327 = None
    unsqueeze_1897: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1896, 2);  unsqueeze_1896 = None
    unsqueeze_1898: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1897, 3);  unsqueeze_1897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1908: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_324, 0);  squeeze_324 = None
    unsqueeze_1909: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1908, 2);  unsqueeze_1908 = None
    unsqueeze_1910: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1909, 3);  unsqueeze_1909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1920: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_321, 0);  squeeze_321 = None
    unsqueeze_1921: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1920, 2);  unsqueeze_1920 = None
    unsqueeze_1922: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1921, 3);  unsqueeze_1921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1932: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_318, 0);  squeeze_318 = None
    unsqueeze_1933: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1932, 2);  unsqueeze_1932 = None
    unsqueeze_1934: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1933, 3);  unsqueeze_1933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_1944: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_315, 0);  squeeze_315 = None
    unsqueeze_1945: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1944, 2);  unsqueeze_1944 = None
    unsqueeze_1946: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1945, 3);  unsqueeze_1945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_1956: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_312, 0);  squeeze_312 = None
    unsqueeze_1957: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1956, 2);  unsqueeze_1956 = None
    unsqueeze_1958: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1957, 3);  unsqueeze_1957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_1968: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_309, 0);  squeeze_309 = None
    unsqueeze_1969: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1968, 2);  unsqueeze_1968 = None
    unsqueeze_1970: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1969, 3);  unsqueeze_1969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    unsqueeze_1980: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_306, 0);  squeeze_306 = None
    unsqueeze_1981: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1980, 2);  unsqueeze_1980 = None
    unsqueeze_1982: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1981, 3);  unsqueeze_1981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_1992: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_303, 0);  squeeze_303 = None
    unsqueeze_1993: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1992, 2);  unsqueeze_1992 = None
    unsqueeze_1994: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1993, 3);  unsqueeze_1993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_100: "b8[8, 432, 42, 42]" = torch.ops.aten.le.Scalar(relu_89, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2004: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_300, 0);  squeeze_300 = None
    unsqueeze_2005: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2004, 2);  unsqueeze_2004 = None
    unsqueeze_2006: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2005, 3);  unsqueeze_2005 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2016: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_297, 0);  squeeze_297 = None
    unsqueeze_2017: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2016, 2);  unsqueeze_2016 = None
    unsqueeze_2018: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2017, 3);  unsqueeze_2017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_102: "b8[8, 432, 42, 42]" = torch.ops.aten.le.Scalar(relu_87, 0);  relu_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2028: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_294, 0);  squeeze_294 = None
    unsqueeze_2029: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2028, 2);  unsqueeze_2028 = None
    unsqueeze_2030: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2029, 3);  unsqueeze_2029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2040: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_291, 0);  squeeze_291 = None
    unsqueeze_2041: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2040, 2);  unsqueeze_2040 = None
    unsqueeze_2042: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2041, 3);  unsqueeze_2041 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2052: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_288, 0);  squeeze_288 = None
    unsqueeze_2053: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2052, 2);  unsqueeze_2052 = None
    unsqueeze_2054: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2053, 3);  unsqueeze_2053 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2064: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_285, 0);  squeeze_285 = None
    unsqueeze_2065: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2064, 2);  unsqueeze_2064 = None
    unsqueeze_2066: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2065, 3);  unsqueeze_2065 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2076: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_282, 0);  squeeze_282 = None
    unsqueeze_2077: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2076, 2);  unsqueeze_2076 = None
    unsqueeze_2078: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2077, 3);  unsqueeze_2077 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2088: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_279, 0);  squeeze_279 = None
    unsqueeze_2089: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2088, 2);  unsqueeze_2088 = None
    unsqueeze_2090: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2089, 3);  unsqueeze_2089 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2100: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_276, 0);  squeeze_276 = None
    unsqueeze_2101: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2100, 2);  unsqueeze_2100 = None
    unsqueeze_2102: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2101, 3);  unsqueeze_2101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2112: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_273, 0);  squeeze_273 = None
    unsqueeze_2113: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2112, 2);  unsqueeze_2112 = None
    unsqueeze_2114: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2113, 3);  unsqueeze_2113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2124: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_270, 0);  squeeze_270 = None
    unsqueeze_2125: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2124, 2);  unsqueeze_2124 = None
    unsqueeze_2126: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2125, 3);  unsqueeze_2125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2136: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_267, 0);  squeeze_267 = None
    unsqueeze_2137: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2136, 2);  unsqueeze_2136 = None
    unsqueeze_2138: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2137, 3);  unsqueeze_2137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_2148: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_264, 0);  squeeze_264 = None
    unsqueeze_2149: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2148, 2);  unsqueeze_2148 = None
    unsqueeze_2150: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2149, 3);  unsqueeze_2149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_2160: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_261, 0);  squeeze_261 = None
    unsqueeze_2161: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2160, 2);  unsqueeze_2160 = None
    unsqueeze_2162: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2161, 3);  unsqueeze_2161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2172: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_258, 0);  squeeze_258 = None
    unsqueeze_2173: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2172, 2);  unsqueeze_2172 = None
    unsqueeze_2174: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2173, 3);  unsqueeze_2173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2184: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_255, 0);  squeeze_255 = None
    unsqueeze_2185: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2184, 2);  unsqueeze_2184 = None
    unsqueeze_2186: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2185, 3);  unsqueeze_2185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2196: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
    unsqueeze_2197: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2196, 2);  unsqueeze_2196 = None
    unsqueeze_2198: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2197, 3);  unsqueeze_2197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2208: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
    unsqueeze_2209: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2208, 2);  unsqueeze_2208 = None
    unsqueeze_2210: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2209, 3);  unsqueeze_2209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2220: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
    unsqueeze_2221: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2220, 2);  unsqueeze_2220 = None
    unsqueeze_2222: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2221, 3);  unsqueeze_2221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2232: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
    unsqueeze_2233: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2232, 2);  unsqueeze_2232 = None
    unsqueeze_2234: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2233, 3);  unsqueeze_2233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2244: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
    unsqueeze_2245: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2244, 2);  unsqueeze_2244 = None
    unsqueeze_2246: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2245, 3);  unsqueeze_2245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2256: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
    unsqueeze_2257: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2256, 2);  unsqueeze_2256 = None
    unsqueeze_2258: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2257, 3);  unsqueeze_2257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2268: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
    unsqueeze_2269: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2268, 2);  unsqueeze_2268 = None
    unsqueeze_2270: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2269, 3);  unsqueeze_2269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2280: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
    unsqueeze_2281: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2280, 2);  unsqueeze_2280 = None
    unsqueeze_2282: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2281, 3);  unsqueeze_2281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2292: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
    unsqueeze_2293: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2292, 2);  unsqueeze_2292 = None
    unsqueeze_2294: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2293, 3);  unsqueeze_2293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2304: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
    unsqueeze_2305: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2304, 2);  unsqueeze_2304 = None
    unsqueeze_2306: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2305, 3);  unsqueeze_2305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_2316: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
    unsqueeze_2317: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2316, 2);  unsqueeze_2316 = None
    unsqueeze_2318: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2317, 3);  unsqueeze_2317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_2328: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
    unsqueeze_2329: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2328, 2);  unsqueeze_2328 = None
    unsqueeze_2330: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2329, 3);  unsqueeze_2329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2340: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
    unsqueeze_2341: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2340, 2);  unsqueeze_2340 = None
    unsqueeze_2342: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2341, 3);  unsqueeze_2341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2352: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
    unsqueeze_2353: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2352, 2);  unsqueeze_2352 = None
    unsqueeze_2354: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2353, 3);  unsqueeze_2353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2364: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
    unsqueeze_2365: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2364, 2);  unsqueeze_2364 = None
    unsqueeze_2366: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2365, 3);  unsqueeze_2365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2376: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
    unsqueeze_2377: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2376, 2);  unsqueeze_2376 = None
    unsqueeze_2378: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2377, 3);  unsqueeze_2377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2388: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
    unsqueeze_2389: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2388, 2);  unsqueeze_2388 = None
    unsqueeze_2390: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2389, 3);  unsqueeze_2389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2400: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
    unsqueeze_2401: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2400, 2);  unsqueeze_2400 = None
    unsqueeze_2402: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2401, 3);  unsqueeze_2401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2412: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
    unsqueeze_2413: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2412, 2);  unsqueeze_2412 = None
    unsqueeze_2414: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2413, 3);  unsqueeze_2413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2424: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
    unsqueeze_2425: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2424, 2);  unsqueeze_2424 = None
    unsqueeze_2426: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2425, 3);  unsqueeze_2425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2436: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_2437: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2436, 2);  unsqueeze_2436 = None
    unsqueeze_2438: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2437, 3);  unsqueeze_2437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2448: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_2449: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2448, 2);  unsqueeze_2448 = None
    unsqueeze_2450: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2449, 3);  unsqueeze_2449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2460: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_2461: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2460, 2);  unsqueeze_2460 = None
    unsqueeze_2462: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2461, 3);  unsqueeze_2461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2472: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_2473: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2472, 2);  unsqueeze_2472 = None
    unsqueeze_2474: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2473, 3);  unsqueeze_2473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_2484: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_2485: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2484, 2);  unsqueeze_2484 = None
    unsqueeze_2486: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2485, 3);  unsqueeze_2485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_2496: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_2497: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2496, 2);  unsqueeze_2496 = None
    unsqueeze_2498: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2497, 3);  unsqueeze_2497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2508: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_2509: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2508, 2);  unsqueeze_2508 = None
    unsqueeze_2510: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2509, 3);  unsqueeze_2509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2520: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_2521: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2520, 2);  unsqueeze_2520 = None
    unsqueeze_2522: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2521, 3);  unsqueeze_2521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2532: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_2533: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2532, 2);  unsqueeze_2532 = None
    unsqueeze_2534: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2533, 3);  unsqueeze_2533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2544: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_2545: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2544, 2);  unsqueeze_2544 = None
    unsqueeze_2546: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2545, 3);  unsqueeze_2545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2556: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_2557: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2556, 2);  unsqueeze_2556 = None
    unsqueeze_2558: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2557, 3);  unsqueeze_2557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2568: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_2569: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2568, 2);  unsqueeze_2568 = None
    unsqueeze_2570: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2569, 3);  unsqueeze_2569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2580: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_2581: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2580, 2);  unsqueeze_2580 = None
    unsqueeze_2582: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2581, 3);  unsqueeze_2581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2592: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_2593: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2592, 2);  unsqueeze_2592 = None
    unsqueeze_2594: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2593, 3);  unsqueeze_2593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2604: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_2605: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2604, 2);  unsqueeze_2604 = None
    unsqueeze_2606: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2605, 3);  unsqueeze_2605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2616: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_2617: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2616, 2);  unsqueeze_2616 = None
    unsqueeze_2618: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2617, 3);  unsqueeze_2617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2628: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_2629: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2628, 2);  unsqueeze_2628 = None
    unsqueeze_2630: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2629, 3);  unsqueeze_2629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2640: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_2641: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2640, 2);  unsqueeze_2640 = None
    unsqueeze_2642: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2641, 3);  unsqueeze_2641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_2652: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_2653: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2652, 2);  unsqueeze_2652 = None
    unsqueeze_2654: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2653, 3);  unsqueeze_2653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_2664: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_2665: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2664, 2);  unsqueeze_2664 = None
    unsqueeze_2666: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2665, 3);  unsqueeze_2665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2676: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_2677: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2676, 2);  unsqueeze_2676 = None
    unsqueeze_2678: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2677, 3);  unsqueeze_2677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2688: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_2689: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2688, 2);  unsqueeze_2688 = None
    unsqueeze_2690: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2689, 3);  unsqueeze_2689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2700: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_2701: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2700, 2);  unsqueeze_2700 = None
    unsqueeze_2702: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2701, 3);  unsqueeze_2701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2712: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_2713: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2712, 2);  unsqueeze_2712 = None
    unsqueeze_2714: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2713, 3);  unsqueeze_2713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2724: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_2725: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2724, 2);  unsqueeze_2724 = None
    unsqueeze_2726: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2725, 3);  unsqueeze_2725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2736: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_2737: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2736, 2);  unsqueeze_2736 = None
    unsqueeze_2738: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2737, 3);  unsqueeze_2737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2748: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_2749: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2748, 2);  unsqueeze_2748 = None
    unsqueeze_2750: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2749, 3);  unsqueeze_2749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2760: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_2761: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2760, 2);  unsqueeze_2760 = None
    unsqueeze_2762: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2761, 3);  unsqueeze_2761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2772: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_2773: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2772, 2);  unsqueeze_2772 = None
    unsqueeze_2774: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2773, 3);  unsqueeze_2773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2784: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_2785: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2784, 2);  unsqueeze_2784 = None
    unsqueeze_2786: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2785, 3);  unsqueeze_2785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2796: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_2797: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2796, 2);  unsqueeze_2796 = None
    unsqueeze_2798: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2797, 3);  unsqueeze_2797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2808: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_2809: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2808, 2);  unsqueeze_2808 = None
    unsqueeze_2810: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2809, 3);  unsqueeze_2809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_2820: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_2821: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2820, 2);  unsqueeze_2820 = None
    unsqueeze_2822: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2821, 3);  unsqueeze_2821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    unsqueeze_2832: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_2833: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2832, 2);  unsqueeze_2832 = None
    unsqueeze_2834: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2833, 3);  unsqueeze_2833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_2844: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_2845: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2844, 2);  unsqueeze_2844 = None
    unsqueeze_2846: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2845, 3);  unsqueeze_2845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_171: "b8[8, 108, 83, 83]" = torch.ops.aten.le.Scalar(relu_18, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2856: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_2857: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2856, 2);  unsqueeze_2856 = None
    unsqueeze_2858: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2857, 3);  unsqueeze_2857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2868: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_2869: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2868, 2);  unsqueeze_2868 = None
    unsqueeze_2870: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2869, 3);  unsqueeze_2869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_173: "b8[8, 108, 83, 83]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2880: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_2881: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2880, 2);  unsqueeze_2880 = None
    unsqueeze_2882: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2881, 3);  unsqueeze_2881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2892: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_2893: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2892, 2);  unsqueeze_2892 = None
    unsqueeze_2894: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2893, 3);  unsqueeze_2893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2904: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_2905: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2904, 2);  unsqueeze_2904 = None
    unsqueeze_2906: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2905, 3);  unsqueeze_2905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2916: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_2917: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2916, 2);  unsqueeze_2916 = None
    unsqueeze_2918: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2917, 3);  unsqueeze_2917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2928: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_2929: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2928, 2);  unsqueeze_2928 = None
    unsqueeze_2930: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2929, 3);  unsqueeze_2929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2940: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_2941: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2940, 2);  unsqueeze_2940 = None
    unsqueeze_2942: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2941, 3);  unsqueeze_2941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2952: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_2953: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2952, 2);  unsqueeze_2952 = None
    unsqueeze_2954: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2953, 3);  unsqueeze_2953 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2964: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_2965: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2964, 2);  unsqueeze_2964 = None
    unsqueeze_2966: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2965, 3);  unsqueeze_2965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_2976: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_2977: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2976, 2);  unsqueeze_2976 = None
    unsqueeze_2978: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2977, 3);  unsqueeze_2977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_2988: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_2989: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2988, 2);  unsqueeze_2988 = None
    unsqueeze_2990: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2989, 3);  unsqueeze_2989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_3000: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_3001: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3000, 2);  unsqueeze_3000 = None
    unsqueeze_3002: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3001, 3);  unsqueeze_3001 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    unsqueeze_3012: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_3013: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3012, 2);  unsqueeze_3012 = None
    unsqueeze_3014: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3013, 3);  unsqueeze_3013 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_3024: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_3025: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3024, 2);  unsqueeze_3024 = None
    unsqueeze_3026: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3025, 3);  unsqueeze_3025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_186: "b8[8, 54, 165, 165]" = torch.ops.aten.le.Scalar(relu_3, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_3036: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_3037: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3036, 2);  unsqueeze_3036 = None
    unsqueeze_3038: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3037, 3);  unsqueeze_3037 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_3048: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_3049: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3048, 2);  unsqueeze_3048 = None
    unsqueeze_3050: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3049, 3);  unsqueeze_3049 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_3060: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_3061: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3060, 2);  unsqueeze_3060 = None
    unsqueeze_3062: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3061, 3);  unsqueeze_3061 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_3072: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_3073: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3072, 2);  unsqueeze_3072 = None
    unsqueeze_3074: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3073, 3);  unsqueeze_3073 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_3084: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_3085: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3084, 2);  unsqueeze_3084 = None
    unsqueeze_3086: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3085, 3);  unsqueeze_3085 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_3096: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_3097: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3096, 2);  unsqueeze_3096 = None
    unsqueeze_3098: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3097, 3);  unsqueeze_3097 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_3108: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_3109: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3108, 2);  unsqueeze_3108 = None
    unsqueeze_3110: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3109, 3);  unsqueeze_3109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_3120: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_3121: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3120, 2);  unsqueeze_3120 = None
    unsqueeze_3122: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3121, 3);  unsqueeze_3121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_3132: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_3133: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3132, 2);  unsqueeze_3132 = None
    unsqueeze_3134: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3133, 3);  unsqueeze_3133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_3144: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_3145: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3144, 2);  unsqueeze_3144 = None
    unsqueeze_3146: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3145, 3);  unsqueeze_3145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    unsqueeze_3156: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_3157: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3156, 2);  unsqueeze_3156 = None
    unsqueeze_3158: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3157, 3);  unsqueeze_3157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    unsqueeze_3168: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_3169: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3168, 2);  unsqueeze_3168 = None
    unsqueeze_3170: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3169, 3);  unsqueeze_3169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    unsqueeze_3180: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_3181: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3180, 2);  unsqueeze_3180 = None
    unsqueeze_3182: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3181, 3);  unsqueeze_3181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    unsqueeze_3192: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_3193: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3192, 2);  unsqueeze_3192 = None
    unsqueeze_3194: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3193, 3);  unsqueeze_3193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_3204: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_3205: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3204, 2);  unsqueeze_3204 = None
    unsqueeze_3206: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3205, 3);  unsqueeze_3205 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_778, add);  primals_778 = add = None
    copy__1: "f32[96]" = torch.ops.aten.copy_.default(primals_779, add_2);  primals_779 = add_2 = None
    copy__2: "f32[96]" = torch.ops.aten.copy_.default(primals_780, add_3);  primals_780 = add_3 = None
    copy__3: "f32[54]" = torch.ops.aten.copy_.default(primals_781, add_7);  primals_781 = add_7 = None
    copy__4: "f32[54]" = torch.ops.aten.copy_.default(primals_782, add_8);  primals_782 = add_8 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_783, add_5);  primals_783 = add_5 = None
    copy__6: "f32[54]" = torch.ops.aten.copy_.default(primals_784, add_12);  primals_784 = add_12 = None
    copy__7: "f32[54]" = torch.ops.aten.copy_.default(primals_785, add_13);  primals_785 = add_13 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_786, add_10);  primals_786 = add_10 = None
    copy__9: "f32[54]" = torch.ops.aten.copy_.default(primals_787, add_17);  primals_787 = add_17 = None
    copy__10: "f32[54]" = torch.ops.aten.copy_.default(primals_788, add_18);  primals_788 = add_18 = None
    copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_789, add_15);  primals_789 = add_15 = None
    copy__12: "f32[54]" = torch.ops.aten.copy_.default(primals_790, add_22);  primals_790 = add_22 = None
    copy__13: "f32[54]" = torch.ops.aten.copy_.default(primals_791, add_23);  primals_791 = add_23 = None
    copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_792, add_20);  primals_792 = add_20 = None
    copy__15: "f32[54]" = torch.ops.aten.copy_.default(primals_793, add_28);  primals_793 = add_28 = None
    copy__16: "f32[54]" = torch.ops.aten.copy_.default(primals_794, add_29);  primals_794 = add_29 = None
    copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_795, add_26);  primals_795 = add_26 = None
    copy__18: "f32[54]" = torch.ops.aten.copy_.default(primals_796, add_33);  primals_796 = add_33 = None
    copy__19: "f32[54]" = torch.ops.aten.copy_.default(primals_797, add_34);  primals_797 = add_34 = None
    copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_798, add_31);  primals_798 = add_31 = None
    copy__21: "f32[54]" = torch.ops.aten.copy_.default(primals_799, add_39);  primals_799 = add_39 = None
    copy__22: "f32[54]" = torch.ops.aten.copy_.default(primals_800, add_40);  primals_800 = add_40 = None
    copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_801, add_37);  primals_801 = add_37 = None
    copy__24: "f32[54]" = torch.ops.aten.copy_.default(primals_802, add_44);  primals_802 = add_44 = None
    copy__25: "f32[54]" = torch.ops.aten.copy_.default(primals_803, add_45);  primals_803 = add_45 = None
    copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_804, add_42);  primals_804 = add_42 = None
    copy__27: "f32[54]" = torch.ops.aten.copy_.default(primals_805, add_49);  primals_805 = add_49 = None
    copy__28: "f32[54]" = torch.ops.aten.copy_.default(primals_806, add_50);  primals_806 = add_50 = None
    copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_807, add_47);  primals_807 = add_47 = None
    copy__30: "f32[54]" = torch.ops.aten.copy_.default(primals_808, add_54);  primals_808 = add_54 = None
    copy__31: "f32[54]" = torch.ops.aten.copy_.default(primals_809, add_55);  primals_809 = add_55 = None
    copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_810, add_52);  primals_810 = add_52 = None
    copy__33: "f32[54]" = torch.ops.aten.copy_.default(primals_811, add_60);  primals_811 = add_60 = None
    copy__34: "f32[54]" = torch.ops.aten.copy_.default(primals_812, add_61);  primals_812 = add_61 = None
    copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_813, add_58);  primals_813 = add_58 = None
    copy__36: "f32[54]" = torch.ops.aten.copy_.default(primals_814, add_65);  primals_814 = add_65 = None
    copy__37: "f32[54]" = torch.ops.aten.copy_.default(primals_815, add_66);  primals_815 = add_66 = None
    copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_816, add_63);  primals_816 = add_63 = None
    copy__39: "f32[54]" = torch.ops.aten.copy_.default(primals_817, add_71);  primals_817 = add_71 = None
    copy__40: "f32[54]" = torch.ops.aten.copy_.default(primals_818, add_72);  primals_818 = add_72 = None
    copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_819, add_69);  primals_819 = add_69 = None
    copy__42: "f32[54]" = torch.ops.aten.copy_.default(primals_820, add_76);  primals_820 = add_76 = None
    copy__43: "f32[54]" = torch.ops.aten.copy_.default(primals_821, add_77);  primals_821 = add_77 = None
    copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_822, add_74);  primals_822 = add_74 = None
    copy__45: "f32[54]" = torch.ops.aten.copy_.default(primals_823, add_81);  primals_823 = add_81 = None
    copy__46: "f32[54]" = torch.ops.aten.copy_.default(primals_824, add_82);  primals_824 = add_82 = None
    copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_825, add_79);  primals_825 = add_79 = None
    copy__48: "f32[108]" = torch.ops.aten.copy_.default(primals_826, add_87);  primals_826 = add_87 = None
    copy__49: "f32[108]" = torch.ops.aten.copy_.default(primals_827, add_88);  primals_827 = add_88 = None
    copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_828, add_85);  primals_828 = add_85 = None
    copy__51: "f32[108]" = torch.ops.aten.copy_.default(primals_829, add_92);  primals_829 = add_92 = None
    copy__52: "f32[108]" = torch.ops.aten.copy_.default(primals_830, add_93);  primals_830 = add_93 = None
    copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_831, add_90);  primals_831 = add_90 = None
    copy__54: "f32[108]" = torch.ops.aten.copy_.default(primals_832, add_97);  primals_832 = add_97 = None
    copy__55: "f32[108]" = torch.ops.aten.copy_.default(primals_833, add_98);  primals_833 = add_98 = None
    copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_834, add_95);  primals_834 = add_95 = None
    copy__57: "f32[108]" = torch.ops.aten.copy_.default(primals_835, add_102);  primals_835 = add_102 = None
    copy__58: "f32[108]" = torch.ops.aten.copy_.default(primals_836, add_103);  primals_836 = add_103 = None
    copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_837, add_100);  primals_837 = add_100 = None
    copy__60: "f32[108]" = torch.ops.aten.copy_.default(primals_838, add_108);  primals_838 = add_108 = None
    copy__61: "f32[108]" = torch.ops.aten.copy_.default(primals_839, add_109);  primals_839 = add_109 = None
    copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_840, add_106);  primals_840 = add_106 = None
    copy__63: "f32[108]" = torch.ops.aten.copy_.default(primals_841, add_113);  primals_841 = add_113 = None
    copy__64: "f32[108]" = torch.ops.aten.copy_.default(primals_842, add_114);  primals_842 = add_114 = None
    copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_843, add_111);  primals_843 = add_111 = None
    copy__66: "f32[108]" = torch.ops.aten.copy_.default(primals_844, add_119);  primals_844 = add_119 = None
    copy__67: "f32[108]" = torch.ops.aten.copy_.default(primals_845, add_120);  primals_845 = add_120 = None
    copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_846, add_117);  primals_846 = add_117 = None
    copy__69: "f32[108]" = torch.ops.aten.copy_.default(primals_847, add_124);  primals_847 = add_124 = None
    copy__70: "f32[108]" = torch.ops.aten.copy_.default(primals_848, add_125);  primals_848 = add_125 = None
    copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_849, add_122);  primals_849 = add_122 = None
    copy__72: "f32[108]" = torch.ops.aten.copy_.default(primals_850, add_129);  primals_850 = add_129 = None
    copy__73: "f32[108]" = torch.ops.aten.copy_.default(primals_851, add_130);  primals_851 = add_130 = None
    copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_852, add_127);  primals_852 = add_127 = None
    copy__75: "f32[108]" = torch.ops.aten.copy_.default(primals_853, add_134);  primals_853 = add_134 = None
    copy__76: "f32[108]" = torch.ops.aten.copy_.default(primals_854, add_135);  primals_854 = add_135 = None
    copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_855, add_132);  primals_855 = add_132 = None
    copy__78: "f32[108]" = torch.ops.aten.copy_.default(primals_856, add_140);  primals_856 = add_140 = None
    copy__79: "f32[108]" = torch.ops.aten.copy_.default(primals_857, add_141);  primals_857 = add_141 = None
    copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_858, add_138);  primals_858 = add_138 = None
    copy__81: "f32[108]" = torch.ops.aten.copy_.default(primals_859, add_145);  primals_859 = add_145 = None
    copy__82: "f32[108]" = torch.ops.aten.copy_.default(primals_860, add_146);  primals_860 = add_146 = None
    copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_861, add_143);  primals_861 = add_143 = None
    copy__84: "f32[108]" = torch.ops.aten.copy_.default(primals_862, add_151);  primals_862 = add_151 = None
    copy__85: "f32[108]" = torch.ops.aten.copy_.default(primals_863, add_152);  primals_863 = add_152 = None
    copy__86: "i64[]" = torch.ops.aten.copy_.default(primals_864, add_149);  primals_864 = add_149 = None
    copy__87: "f32[108]" = torch.ops.aten.copy_.default(primals_865, add_156);  primals_865 = add_156 = None
    copy__88: "f32[108]" = torch.ops.aten.copy_.default(primals_866, add_157);  primals_866 = add_157 = None
    copy__89: "i64[]" = torch.ops.aten.copy_.default(primals_867, add_154);  primals_867 = add_154 = None
    copy__90: "f32[108]" = torch.ops.aten.copy_.default(primals_868, add_161);  primals_868 = add_161 = None
    copy__91: "f32[108]" = torch.ops.aten.copy_.default(primals_869, add_162);  primals_869 = add_162 = None
    copy__92: "i64[]" = torch.ops.aten.copy_.default(primals_870, add_159);  primals_870 = add_159 = None
    copy__93: "f32[216]" = torch.ops.aten.copy_.default(primals_871, add_167);  primals_871 = add_167 = None
    copy__94: "f32[216]" = torch.ops.aten.copy_.default(primals_872, add_168);  primals_872 = add_168 = None
    copy__95: "i64[]" = torch.ops.aten.copy_.default(primals_873, add_165);  primals_873 = add_165 = None
    copy__96: "f32[216]" = torch.ops.aten.copy_.default(primals_874, add_172);  primals_874 = add_172 = None
    copy__97: "f32[216]" = torch.ops.aten.copy_.default(primals_875, add_173);  primals_875 = add_173 = None
    copy__98: "i64[]" = torch.ops.aten.copy_.default(primals_876, add_170);  primals_876 = add_170 = None
    copy__99: "f32[216]" = torch.ops.aten.copy_.default(primals_877, add_177);  primals_877 = add_177 = None
    copy__100: "f32[216]" = torch.ops.aten.copy_.default(primals_878, add_178);  primals_878 = add_178 = None
    copy__101: "i64[]" = torch.ops.aten.copy_.default(primals_879, add_175);  primals_879 = add_175 = None
    copy__102: "f32[216]" = torch.ops.aten.copy_.default(primals_880, add_182);  primals_880 = add_182 = None
    copy__103: "f32[216]" = torch.ops.aten.copy_.default(primals_881, add_183);  primals_881 = add_183 = None
    copy__104: "i64[]" = torch.ops.aten.copy_.default(primals_882, add_180);  primals_882 = add_180 = None
    copy__105: "f32[216]" = torch.ops.aten.copy_.default(primals_883, add_188);  primals_883 = add_188 = None
    copy__106: "f32[216]" = torch.ops.aten.copy_.default(primals_884, add_189);  primals_884 = add_189 = None
    copy__107: "i64[]" = torch.ops.aten.copy_.default(primals_885, add_186);  primals_885 = add_186 = None
    copy__108: "f32[216]" = torch.ops.aten.copy_.default(primals_886, add_193);  primals_886 = add_193 = None
    copy__109: "f32[216]" = torch.ops.aten.copy_.default(primals_887, add_194);  primals_887 = add_194 = None
    copy__110: "i64[]" = torch.ops.aten.copy_.default(primals_888, add_191);  primals_888 = add_191 = None
    copy__111: "f32[216]" = torch.ops.aten.copy_.default(primals_889, add_199);  primals_889 = add_199 = None
    copy__112: "f32[216]" = torch.ops.aten.copy_.default(primals_890, add_200);  primals_890 = add_200 = None
    copy__113: "i64[]" = torch.ops.aten.copy_.default(primals_891, add_197);  primals_891 = add_197 = None
    copy__114: "f32[216]" = torch.ops.aten.copy_.default(primals_892, add_204);  primals_892 = add_204 = None
    copy__115: "f32[216]" = torch.ops.aten.copy_.default(primals_893, add_205);  primals_893 = add_205 = None
    copy__116: "i64[]" = torch.ops.aten.copy_.default(primals_894, add_202);  primals_894 = add_202 = None
    copy__117: "f32[216]" = torch.ops.aten.copy_.default(primals_895, add_209);  primals_895 = add_209 = None
    copy__118: "f32[216]" = torch.ops.aten.copy_.default(primals_896, add_210);  primals_896 = add_210 = None
    copy__119: "i64[]" = torch.ops.aten.copy_.default(primals_897, add_207);  primals_897 = add_207 = None
    copy__120: "f32[216]" = torch.ops.aten.copy_.default(primals_898, add_214);  primals_898 = add_214 = None
    copy__121: "f32[216]" = torch.ops.aten.copy_.default(primals_899, add_215);  primals_899 = add_215 = None
    copy__122: "i64[]" = torch.ops.aten.copy_.default(primals_900, add_212);  primals_900 = add_212 = None
    copy__123: "f32[216]" = torch.ops.aten.copy_.default(primals_901, add_220);  primals_901 = add_220 = None
    copy__124: "f32[216]" = torch.ops.aten.copy_.default(primals_902, add_221);  primals_902 = add_221 = None
    copy__125: "i64[]" = torch.ops.aten.copy_.default(primals_903, add_218);  primals_903 = add_218 = None
    copy__126: "f32[216]" = torch.ops.aten.copy_.default(primals_904, add_225);  primals_904 = add_225 = None
    copy__127: "f32[216]" = torch.ops.aten.copy_.default(primals_905, add_226);  primals_905 = add_226 = None
    copy__128: "i64[]" = torch.ops.aten.copy_.default(primals_906, add_223);  primals_906 = add_223 = None
    copy__129: "f32[216]" = torch.ops.aten.copy_.default(primals_907, add_231);  primals_907 = add_231 = None
    copy__130: "f32[216]" = torch.ops.aten.copy_.default(primals_908, add_232);  primals_908 = add_232 = None
    copy__131: "i64[]" = torch.ops.aten.copy_.default(primals_909, add_229);  primals_909 = add_229 = None
    copy__132: "f32[216]" = torch.ops.aten.copy_.default(primals_910, add_236);  primals_910 = add_236 = None
    copy__133: "f32[216]" = torch.ops.aten.copy_.default(primals_911, add_237);  primals_911 = add_237 = None
    copy__134: "i64[]" = torch.ops.aten.copy_.default(primals_912, add_234);  primals_912 = add_234 = None
    copy__135: "f32[216]" = torch.ops.aten.copy_.default(primals_913, add_242);  primals_913 = add_242 = None
    copy__136: "f32[216]" = torch.ops.aten.copy_.default(primals_914, add_243);  primals_914 = add_243 = None
    copy__137: "i64[]" = torch.ops.aten.copy_.default(primals_915, add_240);  primals_915 = add_240 = None
    copy__138: "f32[216]" = torch.ops.aten.copy_.default(primals_916, add_247);  primals_916 = add_247 = None
    copy__139: "f32[216]" = torch.ops.aten.copy_.default(primals_917, add_248);  primals_917 = add_248 = None
    copy__140: "i64[]" = torch.ops.aten.copy_.default(primals_918, add_245);  primals_918 = add_245 = None
    copy__141: "f32[216]" = torch.ops.aten.copy_.default(primals_919, add_252);  primals_919 = add_252 = None
    copy__142: "f32[216]" = torch.ops.aten.copy_.default(primals_920, add_253);  primals_920 = add_253 = None
    copy__143: "i64[]" = torch.ops.aten.copy_.default(primals_921, add_250);  primals_921 = add_250 = None
    copy__144: "f32[216]" = torch.ops.aten.copy_.default(primals_922, add_257);  primals_922 = add_257 = None
    copy__145: "f32[216]" = torch.ops.aten.copy_.default(primals_923, add_258);  primals_923 = add_258 = None
    copy__146: "i64[]" = torch.ops.aten.copy_.default(primals_924, add_255);  primals_924 = add_255 = None
    copy__147: "f32[216]" = torch.ops.aten.copy_.default(primals_925, add_263);  primals_925 = add_263 = None
    copy__148: "f32[216]" = torch.ops.aten.copy_.default(primals_926, add_264);  primals_926 = add_264 = None
    copy__149: "i64[]" = torch.ops.aten.copy_.default(primals_927, add_261);  primals_927 = add_261 = None
    copy__150: "f32[216]" = torch.ops.aten.copy_.default(primals_928, add_268);  primals_928 = add_268 = None
    copy__151: "f32[216]" = torch.ops.aten.copy_.default(primals_929, add_269);  primals_929 = add_269 = None
    copy__152: "i64[]" = torch.ops.aten.copy_.default(primals_930, add_266);  primals_930 = add_266 = None
    copy__153: "f32[216]" = torch.ops.aten.copy_.default(primals_931, add_274);  primals_931 = add_274 = None
    copy__154: "f32[216]" = torch.ops.aten.copy_.default(primals_932, add_275);  primals_932 = add_275 = None
    copy__155: "i64[]" = torch.ops.aten.copy_.default(primals_933, add_272);  primals_933 = add_272 = None
    copy__156: "f32[216]" = torch.ops.aten.copy_.default(primals_934, add_279);  primals_934 = add_279 = None
    copy__157: "f32[216]" = torch.ops.aten.copy_.default(primals_935, add_280);  primals_935 = add_280 = None
    copy__158: "i64[]" = torch.ops.aten.copy_.default(primals_936, add_277);  primals_936 = add_277 = None
    copy__159: "f32[216]" = torch.ops.aten.copy_.default(primals_937, add_284);  primals_937 = add_284 = None
    copy__160: "f32[216]" = torch.ops.aten.copy_.default(primals_938, add_285);  primals_938 = add_285 = None
    copy__161: "i64[]" = torch.ops.aten.copy_.default(primals_939, add_282);  primals_939 = add_282 = None
    copy__162: "f32[216]" = torch.ops.aten.copy_.default(primals_940, add_289);  primals_940 = add_289 = None
    copy__163: "f32[216]" = torch.ops.aten.copy_.default(primals_941, add_290);  primals_941 = add_290 = None
    copy__164: "i64[]" = torch.ops.aten.copy_.default(primals_942, add_287);  primals_942 = add_287 = None
    copy__165: "f32[216]" = torch.ops.aten.copy_.default(primals_943, add_295);  primals_943 = add_295 = None
    copy__166: "f32[216]" = torch.ops.aten.copy_.default(primals_944, add_296);  primals_944 = add_296 = None
    copy__167: "i64[]" = torch.ops.aten.copy_.default(primals_945, add_293);  primals_945 = add_293 = None
    copy__168: "f32[216]" = torch.ops.aten.copy_.default(primals_946, add_300);  primals_946 = add_300 = None
    copy__169: "f32[216]" = torch.ops.aten.copy_.default(primals_947, add_301);  primals_947 = add_301 = None
    copy__170: "i64[]" = torch.ops.aten.copy_.default(primals_948, add_298);  primals_948 = add_298 = None
    copy__171: "f32[216]" = torch.ops.aten.copy_.default(primals_949, add_306);  primals_949 = add_306 = None
    copy__172: "f32[216]" = torch.ops.aten.copy_.default(primals_950, add_307);  primals_950 = add_307 = None
    copy__173: "i64[]" = torch.ops.aten.copy_.default(primals_951, add_304);  primals_951 = add_304 = None
    copy__174: "f32[216]" = torch.ops.aten.copy_.default(primals_952, add_311);  primals_952 = add_311 = None
    copy__175: "f32[216]" = torch.ops.aten.copy_.default(primals_953, add_312);  primals_953 = add_312 = None
    copy__176: "i64[]" = torch.ops.aten.copy_.default(primals_954, add_309);  primals_954 = add_309 = None
    copy__177: "f32[216]" = torch.ops.aten.copy_.default(primals_955, add_317);  primals_955 = add_317 = None
    copy__178: "f32[216]" = torch.ops.aten.copy_.default(primals_956, add_318);  primals_956 = add_318 = None
    copy__179: "i64[]" = torch.ops.aten.copy_.default(primals_957, add_315);  primals_957 = add_315 = None
    copy__180: "f32[216]" = torch.ops.aten.copy_.default(primals_958, add_322);  primals_958 = add_322 = None
    copy__181: "f32[216]" = torch.ops.aten.copy_.default(primals_959, add_323);  primals_959 = add_323 = None
    copy__182: "i64[]" = torch.ops.aten.copy_.default(primals_960, add_320);  primals_960 = add_320 = None
    copy__183: "f32[216]" = torch.ops.aten.copy_.default(primals_961, add_327);  primals_961 = add_327 = None
    copy__184: "f32[216]" = torch.ops.aten.copy_.default(primals_962, add_328);  primals_962 = add_328 = None
    copy__185: "i64[]" = torch.ops.aten.copy_.default(primals_963, add_325);  primals_963 = add_325 = None
    copy__186: "f32[216]" = torch.ops.aten.copy_.default(primals_964, add_332);  primals_964 = add_332 = None
    copy__187: "f32[216]" = torch.ops.aten.copy_.default(primals_965, add_333);  primals_965 = add_333 = None
    copy__188: "i64[]" = torch.ops.aten.copy_.default(primals_966, add_330);  primals_966 = add_330 = None
    copy__189: "f32[216]" = torch.ops.aten.copy_.default(primals_967, add_338);  primals_967 = add_338 = None
    copy__190: "f32[216]" = torch.ops.aten.copy_.default(primals_968, add_339);  primals_968 = add_339 = None
    copy__191: "i64[]" = torch.ops.aten.copy_.default(primals_969, add_336);  primals_969 = add_336 = None
    copy__192: "f32[216]" = torch.ops.aten.copy_.default(primals_970, add_343);  primals_970 = add_343 = None
    copy__193: "f32[216]" = torch.ops.aten.copy_.default(primals_971, add_344);  primals_971 = add_344 = None
    copy__194: "i64[]" = torch.ops.aten.copy_.default(primals_972, add_341);  primals_972 = add_341 = None
    copy__195: "f32[216]" = torch.ops.aten.copy_.default(primals_973, add_349);  primals_973 = add_349 = None
    copy__196: "f32[216]" = torch.ops.aten.copy_.default(primals_974, add_350);  primals_974 = add_350 = None
    copy__197: "i64[]" = torch.ops.aten.copy_.default(primals_975, add_347);  primals_975 = add_347 = None
    copy__198: "f32[216]" = torch.ops.aten.copy_.default(primals_976, add_354);  primals_976 = add_354 = None
    copy__199: "f32[216]" = torch.ops.aten.copy_.default(primals_977, add_355);  primals_977 = add_355 = None
    copy__200: "i64[]" = torch.ops.aten.copy_.default(primals_978, add_352);  primals_978 = add_352 = None
    copy__201: "f32[216]" = torch.ops.aten.copy_.default(primals_979, add_359);  primals_979 = add_359 = None
    copy__202: "f32[216]" = torch.ops.aten.copy_.default(primals_980, add_360);  primals_980 = add_360 = None
    copy__203: "i64[]" = torch.ops.aten.copy_.default(primals_981, add_357);  primals_981 = add_357 = None
    copy__204: "f32[216]" = torch.ops.aten.copy_.default(primals_982, add_364);  primals_982 = add_364 = None
    copy__205: "f32[216]" = torch.ops.aten.copy_.default(primals_983, add_365);  primals_983 = add_365 = None
    copy__206: "i64[]" = torch.ops.aten.copy_.default(primals_984, add_362);  primals_984 = add_362 = None
    copy__207: "f32[216]" = torch.ops.aten.copy_.default(primals_985, add_370);  primals_985 = add_370 = None
    copy__208: "f32[216]" = torch.ops.aten.copy_.default(primals_986, add_371);  primals_986 = add_371 = None
    copy__209: "i64[]" = torch.ops.aten.copy_.default(primals_987, add_368);  primals_987 = add_368 = None
    copy__210: "f32[216]" = torch.ops.aten.copy_.default(primals_988, add_375);  primals_988 = add_375 = None
    copy__211: "f32[216]" = torch.ops.aten.copy_.default(primals_989, add_376);  primals_989 = add_376 = None
    copy__212: "i64[]" = torch.ops.aten.copy_.default(primals_990, add_373);  primals_990 = add_373 = None
    copy__213: "f32[216]" = torch.ops.aten.copy_.default(primals_991, add_381);  primals_991 = add_381 = None
    copy__214: "f32[216]" = torch.ops.aten.copy_.default(primals_992, add_382);  primals_992 = add_382 = None
    copy__215: "i64[]" = torch.ops.aten.copy_.default(primals_993, add_379);  primals_993 = add_379 = None
    copy__216: "f32[216]" = torch.ops.aten.copy_.default(primals_994, add_386);  primals_994 = add_386 = None
    copy__217: "f32[216]" = torch.ops.aten.copy_.default(primals_995, add_387);  primals_995 = add_387 = None
    copy__218: "i64[]" = torch.ops.aten.copy_.default(primals_996, add_384);  primals_996 = add_384 = None
    copy__219: "f32[216]" = torch.ops.aten.copy_.default(primals_997, add_392);  primals_997 = add_392 = None
    copy__220: "f32[216]" = torch.ops.aten.copy_.default(primals_998, add_393);  primals_998 = add_393 = None
    copy__221: "i64[]" = torch.ops.aten.copy_.default(primals_999, add_390);  primals_999 = add_390 = None
    copy__222: "f32[216]" = torch.ops.aten.copy_.default(primals_1000, add_397);  primals_1000 = add_397 = None
    copy__223: "f32[216]" = torch.ops.aten.copy_.default(primals_1001, add_398);  primals_1001 = add_398 = None
    copy__224: "i64[]" = torch.ops.aten.copy_.default(primals_1002, add_395);  primals_1002 = add_395 = None
    copy__225: "f32[216]" = torch.ops.aten.copy_.default(primals_1003, add_402);  primals_1003 = add_402 = None
    copy__226: "f32[216]" = torch.ops.aten.copy_.default(primals_1004, add_403);  primals_1004 = add_403 = None
    copy__227: "i64[]" = torch.ops.aten.copy_.default(primals_1005, add_400);  primals_1005 = add_400 = None
    copy__228: "f32[216]" = torch.ops.aten.copy_.default(primals_1006, add_407);  primals_1006 = add_407 = None
    copy__229: "f32[216]" = torch.ops.aten.copy_.default(primals_1007, add_408);  primals_1007 = add_408 = None
    copy__230: "i64[]" = torch.ops.aten.copy_.default(primals_1008, add_405);  primals_1008 = add_405 = None
    copy__231: "f32[216]" = torch.ops.aten.copy_.default(primals_1009, add_413);  primals_1009 = add_413 = None
    copy__232: "f32[216]" = torch.ops.aten.copy_.default(primals_1010, add_414);  primals_1010 = add_414 = None
    copy__233: "i64[]" = torch.ops.aten.copy_.default(primals_1011, add_411);  primals_1011 = add_411 = None
    copy__234: "f32[216]" = torch.ops.aten.copy_.default(primals_1012, add_418);  primals_1012 = add_418 = None
    copy__235: "f32[216]" = torch.ops.aten.copy_.default(primals_1013, add_419);  primals_1013 = add_419 = None
    copy__236: "i64[]" = torch.ops.aten.copy_.default(primals_1014, add_416);  primals_1014 = add_416 = None
    copy__237: "f32[216]" = torch.ops.aten.copy_.default(primals_1015, add_424);  primals_1015 = add_424 = None
    copy__238: "f32[216]" = torch.ops.aten.copy_.default(primals_1016, add_425);  primals_1016 = add_425 = None
    copy__239: "i64[]" = torch.ops.aten.copy_.default(primals_1017, add_422);  primals_1017 = add_422 = None
    copy__240: "f32[216]" = torch.ops.aten.copy_.default(primals_1018, add_429);  primals_1018 = add_429 = None
    copy__241: "f32[216]" = torch.ops.aten.copy_.default(primals_1019, add_430);  primals_1019 = add_430 = None
    copy__242: "i64[]" = torch.ops.aten.copy_.default(primals_1020, add_427);  primals_1020 = add_427 = None
    copy__243: "f32[216]" = torch.ops.aten.copy_.default(primals_1021, add_434);  primals_1021 = add_434 = None
    copy__244: "f32[216]" = torch.ops.aten.copy_.default(primals_1022, add_435);  primals_1022 = add_435 = None
    copy__245: "i64[]" = torch.ops.aten.copy_.default(primals_1023, add_432);  primals_1023 = add_432 = None
    copy__246: "f32[216]" = torch.ops.aten.copy_.default(primals_1024, add_439);  primals_1024 = add_439 = None
    copy__247: "f32[216]" = torch.ops.aten.copy_.default(primals_1025, add_440);  primals_1025 = add_440 = None
    copy__248: "i64[]" = torch.ops.aten.copy_.default(primals_1026, add_437);  primals_1026 = add_437 = None
    copy__249: "f32[216]" = torch.ops.aten.copy_.default(primals_1027, add_445);  primals_1027 = add_445 = None
    copy__250: "f32[216]" = torch.ops.aten.copy_.default(primals_1028, add_446);  primals_1028 = add_446 = None
    copy__251: "i64[]" = torch.ops.aten.copy_.default(primals_1029, add_443);  primals_1029 = add_443 = None
    copy__252: "f32[216]" = torch.ops.aten.copy_.default(primals_1030, add_450);  primals_1030 = add_450 = None
    copy__253: "f32[216]" = torch.ops.aten.copy_.default(primals_1031, add_451);  primals_1031 = add_451 = None
    copy__254: "i64[]" = torch.ops.aten.copy_.default(primals_1032, add_448);  primals_1032 = add_448 = None
    copy__255: "f32[216]" = torch.ops.aten.copy_.default(primals_1033, add_456);  primals_1033 = add_456 = None
    copy__256: "f32[216]" = torch.ops.aten.copy_.default(primals_1034, add_457);  primals_1034 = add_457 = None
    copy__257: "i64[]" = torch.ops.aten.copy_.default(primals_1035, add_454);  primals_1035 = add_454 = None
    copy__258: "f32[216]" = torch.ops.aten.copy_.default(primals_1036, add_461);  primals_1036 = add_461 = None
    copy__259: "f32[216]" = torch.ops.aten.copy_.default(primals_1037, add_462);  primals_1037 = add_462 = None
    copy__260: "i64[]" = torch.ops.aten.copy_.default(primals_1038, add_459);  primals_1038 = add_459 = None
    copy__261: "f32[432]" = torch.ops.aten.copy_.default(primals_1039, add_467);  primals_1039 = add_467 = None
    copy__262: "f32[432]" = torch.ops.aten.copy_.default(primals_1040, add_468);  primals_1040 = add_468 = None
    copy__263: "i64[]" = torch.ops.aten.copy_.default(primals_1041, add_465);  primals_1041 = add_465 = None
    copy__264: "f32[432]" = torch.ops.aten.copy_.default(primals_1042, add_472);  primals_1042 = add_472 = None
    copy__265: "f32[432]" = torch.ops.aten.copy_.default(primals_1043, add_473);  primals_1043 = add_473 = None
    copy__266: "i64[]" = torch.ops.aten.copy_.default(primals_1044, add_470);  primals_1044 = add_470 = None
    copy__267: "f32[432]" = torch.ops.aten.copy_.default(primals_1045, add_477);  primals_1045 = add_477 = None
    copy__268: "f32[432]" = torch.ops.aten.copy_.default(primals_1046, add_478);  primals_1046 = add_478 = None
    copy__269: "i64[]" = torch.ops.aten.copy_.default(primals_1047, add_475);  primals_1047 = add_475 = None
    copy__270: "f32[432]" = torch.ops.aten.copy_.default(primals_1048, add_482);  primals_1048 = add_482 = None
    copy__271: "f32[432]" = torch.ops.aten.copy_.default(primals_1049, add_483);  primals_1049 = add_483 = None
    copy__272: "i64[]" = torch.ops.aten.copy_.default(primals_1050, add_480);  primals_1050 = add_480 = None
    copy__273: "f32[432]" = torch.ops.aten.copy_.default(primals_1051, add_488);  primals_1051 = add_488 = None
    copy__274: "f32[432]" = torch.ops.aten.copy_.default(primals_1052, add_489);  primals_1052 = add_489 = None
    copy__275: "i64[]" = torch.ops.aten.copy_.default(primals_1053, add_486);  primals_1053 = add_486 = None
    copy__276: "f32[432]" = torch.ops.aten.copy_.default(primals_1054, add_493);  primals_1054 = add_493 = None
    copy__277: "f32[432]" = torch.ops.aten.copy_.default(primals_1055, add_494);  primals_1055 = add_494 = None
    copy__278: "i64[]" = torch.ops.aten.copy_.default(primals_1056, add_491);  primals_1056 = add_491 = None
    copy__279: "f32[432]" = torch.ops.aten.copy_.default(primals_1057, add_499);  primals_1057 = add_499 = None
    copy__280: "f32[432]" = torch.ops.aten.copy_.default(primals_1058, add_500);  primals_1058 = add_500 = None
    copy__281: "i64[]" = torch.ops.aten.copy_.default(primals_1059, add_497);  primals_1059 = add_497 = None
    copy__282: "f32[432]" = torch.ops.aten.copy_.default(primals_1060, add_504);  primals_1060 = add_504 = None
    copy__283: "f32[432]" = torch.ops.aten.copy_.default(primals_1061, add_505);  primals_1061 = add_505 = None
    copy__284: "i64[]" = torch.ops.aten.copy_.default(primals_1062, add_502);  primals_1062 = add_502 = None
    copy__285: "f32[432]" = torch.ops.aten.copy_.default(primals_1063, add_509);  primals_1063 = add_509 = None
    copy__286: "f32[432]" = torch.ops.aten.copy_.default(primals_1064, add_510);  primals_1064 = add_510 = None
    copy__287: "i64[]" = torch.ops.aten.copy_.default(primals_1065, add_507);  primals_1065 = add_507 = None
    copy__288: "f32[432]" = torch.ops.aten.copy_.default(primals_1066, add_514);  primals_1066 = add_514 = None
    copy__289: "f32[432]" = torch.ops.aten.copy_.default(primals_1067, add_515);  primals_1067 = add_515 = None
    copy__290: "i64[]" = torch.ops.aten.copy_.default(primals_1068, add_512);  primals_1068 = add_512 = None
    copy__291: "f32[432]" = torch.ops.aten.copy_.default(primals_1069, add_520);  primals_1069 = add_520 = None
    copy__292: "f32[432]" = torch.ops.aten.copy_.default(primals_1070, add_521);  primals_1070 = add_521 = None
    copy__293: "i64[]" = torch.ops.aten.copy_.default(primals_1071, add_518);  primals_1071 = add_518 = None
    copy__294: "f32[432]" = torch.ops.aten.copy_.default(primals_1072, add_525);  primals_1072 = add_525 = None
    copy__295: "f32[432]" = torch.ops.aten.copy_.default(primals_1073, add_526);  primals_1073 = add_526 = None
    copy__296: "i64[]" = torch.ops.aten.copy_.default(primals_1074, add_523);  primals_1074 = add_523 = None
    copy__297: "f32[432]" = torch.ops.aten.copy_.default(primals_1075, add_531);  primals_1075 = add_531 = None
    copy__298: "f32[432]" = torch.ops.aten.copy_.default(primals_1076, add_532);  primals_1076 = add_532 = None
    copy__299: "i64[]" = torch.ops.aten.copy_.default(primals_1077, add_529);  primals_1077 = add_529 = None
    copy__300: "f32[432]" = torch.ops.aten.copy_.default(primals_1078, add_536);  primals_1078 = add_536 = None
    copy__301: "f32[432]" = torch.ops.aten.copy_.default(primals_1079, add_537);  primals_1079 = add_537 = None
    copy__302: "i64[]" = torch.ops.aten.copy_.default(primals_1080, add_534);  primals_1080 = add_534 = None
    copy__303: "f32[432]" = torch.ops.aten.copy_.default(primals_1081, add_541);  primals_1081 = add_541 = None
    copy__304: "f32[432]" = torch.ops.aten.copy_.default(primals_1082, add_542);  primals_1082 = add_542 = None
    copy__305: "i64[]" = torch.ops.aten.copy_.default(primals_1083, add_539);  primals_1083 = add_539 = None
    copy__306: "f32[432]" = torch.ops.aten.copy_.default(primals_1084, add_547);  primals_1084 = add_547 = None
    copy__307: "f32[432]" = torch.ops.aten.copy_.default(primals_1085, add_548);  primals_1085 = add_548 = None
    copy__308: "i64[]" = torch.ops.aten.copy_.default(primals_1086, add_545);  primals_1086 = add_545 = None
    copy__309: "f32[432]" = torch.ops.aten.copy_.default(primals_1087, add_552);  primals_1087 = add_552 = None
    copy__310: "f32[432]" = torch.ops.aten.copy_.default(primals_1088, add_553);  primals_1088 = add_553 = None
    copy__311: "i64[]" = torch.ops.aten.copy_.default(primals_1089, add_550);  primals_1089 = add_550 = None
    copy__312: "f32[432]" = torch.ops.aten.copy_.default(primals_1090, add_557);  primals_1090 = add_557 = None
    copy__313: "f32[432]" = torch.ops.aten.copy_.default(primals_1091, add_558);  primals_1091 = add_558 = None
    copy__314: "i64[]" = torch.ops.aten.copy_.default(primals_1092, add_555);  primals_1092 = add_555 = None
    copy__315: "f32[432]" = torch.ops.aten.copy_.default(primals_1093, add_562);  primals_1093 = add_562 = None
    copy__316: "f32[432]" = torch.ops.aten.copy_.default(primals_1094, add_563);  primals_1094 = add_563 = None
    copy__317: "i64[]" = torch.ops.aten.copy_.default(primals_1095, add_560);  primals_1095 = add_560 = None
    copy__318: "f32[432]" = torch.ops.aten.copy_.default(primals_1096, add_568);  primals_1096 = add_568 = None
    copy__319: "f32[432]" = torch.ops.aten.copy_.default(primals_1097, add_569);  primals_1097 = add_569 = None
    copy__320: "i64[]" = torch.ops.aten.copy_.default(primals_1098, add_566);  primals_1098 = add_566 = None
    copy__321: "f32[432]" = torch.ops.aten.copy_.default(primals_1099, add_573);  primals_1099 = add_573 = None
    copy__322: "f32[432]" = torch.ops.aten.copy_.default(primals_1100, add_574);  primals_1100 = add_574 = None
    copy__323: "i64[]" = torch.ops.aten.copy_.default(primals_1101, add_571);  primals_1101 = add_571 = None
    copy__324: "f32[432]" = torch.ops.aten.copy_.default(primals_1102, add_579);  primals_1102 = add_579 = None
    copy__325: "f32[432]" = torch.ops.aten.copy_.default(primals_1103, add_580);  primals_1103 = add_580 = None
    copy__326: "i64[]" = torch.ops.aten.copy_.default(primals_1104, add_577);  primals_1104 = add_577 = None
    copy__327: "f32[432]" = torch.ops.aten.copy_.default(primals_1105, add_584);  primals_1105 = add_584 = None
    copy__328: "f32[432]" = torch.ops.aten.copy_.default(primals_1106, add_585);  primals_1106 = add_585 = None
    copy__329: "i64[]" = torch.ops.aten.copy_.default(primals_1107, add_582);  primals_1107 = add_582 = None
    copy__330: "f32[432]" = torch.ops.aten.copy_.default(primals_1108, add_589);  primals_1108 = add_589 = None
    copy__331: "f32[432]" = torch.ops.aten.copy_.default(primals_1109, add_590);  primals_1109 = add_590 = None
    copy__332: "i64[]" = torch.ops.aten.copy_.default(primals_1110, add_587);  primals_1110 = add_587 = None
    copy__333: "f32[432]" = torch.ops.aten.copy_.default(primals_1111, add_594);  primals_1111 = add_594 = None
    copy__334: "f32[432]" = torch.ops.aten.copy_.default(primals_1112, add_595);  primals_1112 = add_595 = None
    copy__335: "i64[]" = torch.ops.aten.copy_.default(primals_1113, add_592);  primals_1113 = add_592 = None
    copy__336: "f32[432]" = torch.ops.aten.copy_.default(primals_1114, add_600);  primals_1114 = add_600 = None
    copy__337: "f32[432]" = torch.ops.aten.copy_.default(primals_1115, add_601);  primals_1115 = add_601 = None
    copy__338: "i64[]" = torch.ops.aten.copy_.default(primals_1116, add_598);  primals_1116 = add_598 = None
    copy__339: "f32[432]" = torch.ops.aten.copy_.default(primals_1117, add_605);  primals_1117 = add_605 = None
    copy__340: "f32[432]" = torch.ops.aten.copy_.default(primals_1118, add_606);  primals_1118 = add_606 = None
    copy__341: "i64[]" = torch.ops.aten.copy_.default(primals_1119, add_603);  primals_1119 = add_603 = None
    copy__342: "f32[432]" = torch.ops.aten.copy_.default(primals_1120, add_611);  primals_1120 = add_611 = None
    copy__343: "f32[432]" = torch.ops.aten.copy_.default(primals_1121, add_612);  primals_1121 = add_612 = None
    copy__344: "i64[]" = torch.ops.aten.copy_.default(primals_1122, add_609);  primals_1122 = add_609 = None
    copy__345: "f32[432]" = torch.ops.aten.copy_.default(primals_1123, add_616);  primals_1123 = add_616 = None
    copy__346: "f32[432]" = torch.ops.aten.copy_.default(primals_1124, add_617);  primals_1124 = add_617 = None
    copy__347: "i64[]" = torch.ops.aten.copy_.default(primals_1125, add_614);  primals_1125 = add_614 = None
    copy__348: "f32[432]" = torch.ops.aten.copy_.default(primals_1126, add_622);  primals_1126 = add_622 = None
    copy__349: "f32[432]" = torch.ops.aten.copy_.default(primals_1127, add_623);  primals_1127 = add_623 = None
    copy__350: "i64[]" = torch.ops.aten.copy_.default(primals_1128, add_620);  primals_1128 = add_620 = None
    copy__351: "f32[432]" = torch.ops.aten.copy_.default(primals_1129, add_627);  primals_1129 = add_627 = None
    copy__352: "f32[432]" = torch.ops.aten.copy_.default(primals_1130, add_628);  primals_1130 = add_628 = None
    copy__353: "i64[]" = torch.ops.aten.copy_.default(primals_1131, add_625);  primals_1131 = add_625 = None
    copy__354: "f32[432]" = torch.ops.aten.copy_.default(primals_1132, add_632);  primals_1132 = add_632 = None
    copy__355: "f32[432]" = torch.ops.aten.copy_.default(primals_1133, add_633);  primals_1133 = add_633 = None
    copy__356: "i64[]" = torch.ops.aten.copy_.default(primals_1134, add_630);  primals_1134 = add_630 = None
    copy__357: "f32[432]" = torch.ops.aten.copy_.default(primals_1135, add_637);  primals_1135 = add_637 = None
    copy__358: "f32[432]" = torch.ops.aten.copy_.default(primals_1136, add_638);  primals_1136 = add_638 = None
    copy__359: "i64[]" = torch.ops.aten.copy_.default(primals_1137, add_635);  primals_1137 = add_635 = None
    copy__360: "f32[432]" = torch.ops.aten.copy_.default(primals_1138, add_643);  primals_1138 = add_643 = None
    copy__361: "f32[432]" = torch.ops.aten.copy_.default(primals_1139, add_644);  primals_1139 = add_644 = None
    copy__362: "i64[]" = torch.ops.aten.copy_.default(primals_1140, add_641);  primals_1140 = add_641 = None
    copy__363: "f32[432]" = torch.ops.aten.copy_.default(primals_1141, add_648);  primals_1141 = add_648 = None
    copy__364: "f32[432]" = torch.ops.aten.copy_.default(primals_1142, add_649);  primals_1142 = add_649 = None
    copy__365: "i64[]" = torch.ops.aten.copy_.default(primals_1143, add_646);  primals_1143 = add_646 = None
    copy__366: "f32[432]" = torch.ops.aten.copy_.default(primals_1144, add_654);  primals_1144 = add_654 = None
    copy__367: "f32[432]" = torch.ops.aten.copy_.default(primals_1145, add_655);  primals_1145 = add_655 = None
    copy__368: "i64[]" = torch.ops.aten.copy_.default(primals_1146, add_652);  primals_1146 = add_652 = None
    copy__369: "f32[432]" = torch.ops.aten.copy_.default(primals_1147, add_659);  primals_1147 = add_659 = None
    copy__370: "f32[432]" = torch.ops.aten.copy_.default(primals_1148, add_660);  primals_1148 = add_660 = None
    copy__371: "i64[]" = torch.ops.aten.copy_.default(primals_1149, add_657);  primals_1149 = add_657 = None
    copy__372: "f32[432]" = torch.ops.aten.copy_.default(primals_1150, add_664);  primals_1150 = add_664 = None
    copy__373: "f32[432]" = torch.ops.aten.copy_.default(primals_1151, add_665);  primals_1151 = add_665 = None
    copy__374: "i64[]" = torch.ops.aten.copy_.default(primals_1152, add_662);  primals_1152 = add_662 = None
    copy__375: "f32[432]" = torch.ops.aten.copy_.default(primals_1153, add_669);  primals_1153 = add_669 = None
    copy__376: "f32[432]" = torch.ops.aten.copy_.default(primals_1154, add_670);  primals_1154 = add_670 = None
    copy__377: "i64[]" = torch.ops.aten.copy_.default(primals_1155, add_667);  primals_1155 = add_667 = None
    copy__378: "f32[432]" = torch.ops.aten.copy_.default(primals_1156, add_675);  primals_1156 = add_675 = None
    copy__379: "f32[432]" = torch.ops.aten.copy_.default(primals_1157, add_676);  primals_1157 = add_676 = None
    copy__380: "i64[]" = torch.ops.aten.copy_.default(primals_1158, add_673);  primals_1158 = add_673 = None
    copy__381: "f32[432]" = torch.ops.aten.copy_.default(primals_1159, add_680);  primals_1159 = add_680 = None
    copy__382: "f32[432]" = torch.ops.aten.copy_.default(primals_1160, add_681);  primals_1160 = add_681 = None
    copy__383: "i64[]" = torch.ops.aten.copy_.default(primals_1161, add_678);  primals_1161 = add_678 = None
    copy__384: "f32[432]" = torch.ops.aten.copy_.default(primals_1162, add_686);  primals_1162 = add_686 = None
    copy__385: "f32[432]" = torch.ops.aten.copy_.default(primals_1163, add_687);  primals_1163 = add_687 = None
    copy__386: "i64[]" = torch.ops.aten.copy_.default(primals_1164, add_684);  primals_1164 = add_684 = None
    copy__387: "f32[432]" = torch.ops.aten.copy_.default(primals_1165, add_691);  primals_1165 = add_691 = None
    copy__388: "f32[432]" = torch.ops.aten.copy_.default(primals_1166, add_692);  primals_1166 = add_692 = None
    copy__389: "i64[]" = torch.ops.aten.copy_.default(primals_1167, add_689);  primals_1167 = add_689 = None
    copy__390: "f32[432]" = torch.ops.aten.copy_.default(primals_1168, add_697);  primals_1168 = add_697 = None
    copy__391: "f32[432]" = torch.ops.aten.copy_.default(primals_1169, add_698);  primals_1169 = add_698 = None
    copy__392: "i64[]" = torch.ops.aten.copy_.default(primals_1170, add_695);  primals_1170 = add_695 = None
    copy__393: "f32[432]" = torch.ops.aten.copy_.default(primals_1171, add_702);  primals_1171 = add_702 = None
    copy__394: "f32[432]" = torch.ops.aten.copy_.default(primals_1172, add_703);  primals_1172 = add_703 = None
    copy__395: "i64[]" = torch.ops.aten.copy_.default(primals_1173, add_700);  primals_1173 = add_700 = None
    copy__396: "f32[432]" = torch.ops.aten.copy_.default(primals_1174, add_707);  primals_1174 = add_707 = None
    copy__397: "f32[432]" = torch.ops.aten.copy_.default(primals_1175, add_708);  primals_1175 = add_708 = None
    copy__398: "i64[]" = torch.ops.aten.copy_.default(primals_1176, add_705);  primals_1176 = add_705 = None
    copy__399: "f32[432]" = torch.ops.aten.copy_.default(primals_1177, add_712);  primals_1177 = add_712 = None
    copy__400: "f32[432]" = torch.ops.aten.copy_.default(primals_1178, add_713);  primals_1178 = add_713 = None
    copy__401: "i64[]" = torch.ops.aten.copy_.default(primals_1179, add_710);  primals_1179 = add_710 = None
    copy__402: "f32[432]" = torch.ops.aten.copy_.default(primals_1180, add_718);  primals_1180 = add_718 = None
    copy__403: "f32[432]" = torch.ops.aten.copy_.default(primals_1181, add_719);  primals_1181 = add_719 = None
    copy__404: "i64[]" = torch.ops.aten.copy_.default(primals_1182, add_716);  primals_1182 = add_716 = None
    copy__405: "f32[432]" = torch.ops.aten.copy_.default(primals_1183, add_723);  primals_1183 = add_723 = None
    copy__406: "f32[432]" = torch.ops.aten.copy_.default(primals_1184, add_724);  primals_1184 = add_724 = None
    copy__407: "i64[]" = torch.ops.aten.copy_.default(primals_1185, add_721);  primals_1185 = add_721 = None
    copy__408: "f32[432]" = torch.ops.aten.copy_.default(primals_1186, add_729);  primals_1186 = add_729 = None
    copy__409: "f32[432]" = torch.ops.aten.copy_.default(primals_1187, add_730);  primals_1187 = add_730 = None
    copy__410: "i64[]" = torch.ops.aten.copy_.default(primals_1188, add_727);  primals_1188 = add_727 = None
    copy__411: "f32[432]" = torch.ops.aten.copy_.default(primals_1189, add_734);  primals_1189 = add_734 = None
    copy__412: "f32[432]" = torch.ops.aten.copy_.default(primals_1190, add_735);  primals_1190 = add_735 = None
    copy__413: "i64[]" = torch.ops.aten.copy_.default(primals_1191, add_732);  primals_1191 = add_732 = None
    copy__414: "f32[432]" = torch.ops.aten.copy_.default(primals_1192, add_739);  primals_1192 = add_739 = None
    copy__415: "f32[432]" = torch.ops.aten.copy_.default(primals_1193, add_740);  primals_1193 = add_740 = None
    copy__416: "i64[]" = torch.ops.aten.copy_.default(primals_1194, add_737);  primals_1194 = add_737 = None
    copy__417: "f32[432]" = torch.ops.aten.copy_.default(primals_1195, add_744);  primals_1195 = add_744 = None
    copy__418: "f32[432]" = torch.ops.aten.copy_.default(primals_1196, add_745);  primals_1196 = add_745 = None
    copy__419: "i64[]" = torch.ops.aten.copy_.default(primals_1197, add_742);  primals_1197 = add_742 = None
    copy__420: "f32[432]" = torch.ops.aten.copy_.default(primals_1198, add_750);  primals_1198 = add_750 = None
    copy__421: "f32[432]" = torch.ops.aten.copy_.default(primals_1199, add_751);  primals_1199 = add_751 = None
    copy__422: "i64[]" = torch.ops.aten.copy_.default(primals_1200, add_748);  primals_1200 = add_748 = None
    copy__423: "f32[432]" = torch.ops.aten.copy_.default(primals_1201, add_755);  primals_1201 = add_755 = None
    copy__424: "f32[432]" = torch.ops.aten.copy_.default(primals_1202, add_756);  primals_1202 = add_756 = None
    copy__425: "i64[]" = torch.ops.aten.copy_.default(primals_1203, add_753);  primals_1203 = add_753 = None
    copy__426: "f32[432]" = torch.ops.aten.copy_.default(primals_1204, add_761);  primals_1204 = add_761 = None
    copy__427: "f32[432]" = torch.ops.aten.copy_.default(primals_1205, add_762);  primals_1205 = add_762 = None
    copy__428: "i64[]" = torch.ops.aten.copy_.default(primals_1206, add_759);  primals_1206 = add_759 = None
    copy__429: "f32[432]" = torch.ops.aten.copy_.default(primals_1207, add_766);  primals_1207 = add_766 = None
    copy__430: "f32[432]" = torch.ops.aten.copy_.default(primals_1208, add_767);  primals_1208 = add_767 = None
    copy__431: "i64[]" = torch.ops.aten.copy_.default(primals_1209, add_764);  primals_1209 = add_764 = None
    copy__432: "f32[864]" = torch.ops.aten.copy_.default(primals_1210, add_772);  primals_1210 = add_772 = None
    copy__433: "f32[864]" = torch.ops.aten.copy_.default(primals_1211, add_773);  primals_1211 = add_773 = None
    copy__434: "i64[]" = torch.ops.aten.copy_.default(primals_1212, add_770);  primals_1212 = add_770 = None
    copy__435: "f32[864]" = torch.ops.aten.copy_.default(primals_1213, add_777);  primals_1213 = add_777 = None
    copy__436: "f32[864]" = torch.ops.aten.copy_.default(primals_1214, add_778);  primals_1214 = add_778 = None
    copy__437: "i64[]" = torch.ops.aten.copy_.default(primals_1215, add_775);  primals_1215 = add_775 = None
    copy__438: "f32[864]" = torch.ops.aten.copy_.default(primals_1216, add_782);  primals_1216 = add_782 = None
    copy__439: "f32[864]" = torch.ops.aten.copy_.default(primals_1217, add_783);  primals_1217 = add_783 = None
    copy__440: "i64[]" = torch.ops.aten.copy_.default(primals_1218, add_780);  primals_1218 = add_780 = None
    copy__441: "f32[864]" = torch.ops.aten.copy_.default(primals_1219, add_787);  primals_1219 = add_787 = None
    copy__442: "f32[864]" = torch.ops.aten.copy_.default(primals_1220, add_788);  primals_1220 = add_788 = None
    copy__443: "i64[]" = torch.ops.aten.copy_.default(primals_1221, add_785);  primals_1221 = add_785 = None
    copy__444: "f32[864]" = torch.ops.aten.copy_.default(primals_1222, add_793);  primals_1222 = add_793 = None
    copy__445: "f32[864]" = torch.ops.aten.copy_.default(primals_1223, add_794);  primals_1223 = add_794 = None
    copy__446: "i64[]" = torch.ops.aten.copy_.default(primals_1224, add_791);  primals_1224 = add_791 = None
    copy__447: "f32[864]" = torch.ops.aten.copy_.default(primals_1225, add_798);  primals_1225 = add_798 = None
    copy__448: "f32[864]" = torch.ops.aten.copy_.default(primals_1226, add_799);  primals_1226 = add_799 = None
    copy__449: "i64[]" = torch.ops.aten.copy_.default(primals_1227, add_796);  primals_1227 = add_796 = None
    copy__450: "f32[864]" = torch.ops.aten.copy_.default(primals_1228, add_804);  primals_1228 = add_804 = None
    copy__451: "f32[864]" = torch.ops.aten.copy_.default(primals_1229, add_805);  primals_1229 = add_805 = None
    copy__452: "i64[]" = torch.ops.aten.copy_.default(primals_1230, add_802);  primals_1230 = add_802 = None
    copy__453: "f32[864]" = torch.ops.aten.copy_.default(primals_1231, add_809);  primals_1231 = add_809 = None
    copy__454: "f32[864]" = torch.ops.aten.copy_.default(primals_1232, add_810);  primals_1232 = add_810 = None
    copy__455: "i64[]" = torch.ops.aten.copy_.default(primals_1233, add_807);  primals_1233 = add_807 = None
    copy__456: "f32[864]" = torch.ops.aten.copy_.default(primals_1234, add_814);  primals_1234 = add_814 = None
    copy__457: "f32[864]" = torch.ops.aten.copy_.default(primals_1235, add_815);  primals_1235 = add_815 = None
    copy__458: "i64[]" = torch.ops.aten.copy_.default(primals_1236, add_812);  primals_1236 = add_812 = None
    copy__459: "f32[864]" = torch.ops.aten.copy_.default(primals_1237, add_819);  primals_1237 = add_819 = None
    copy__460: "f32[864]" = torch.ops.aten.copy_.default(primals_1238, add_820);  primals_1238 = add_820 = None
    copy__461: "i64[]" = torch.ops.aten.copy_.default(primals_1239, add_817);  primals_1239 = add_817 = None
    copy__462: "f32[864]" = torch.ops.aten.copy_.default(primals_1240, add_825);  primals_1240 = add_825 = None
    copy__463: "f32[864]" = torch.ops.aten.copy_.default(primals_1241, add_826);  primals_1241 = add_826 = None
    copy__464: "i64[]" = torch.ops.aten.copy_.default(primals_1242, add_823);  primals_1242 = add_823 = None
    copy__465: "f32[864]" = torch.ops.aten.copy_.default(primals_1243, add_830);  primals_1243 = add_830 = None
    copy__466: "f32[864]" = torch.ops.aten.copy_.default(primals_1244, add_831);  primals_1244 = add_831 = None
    copy__467: "i64[]" = torch.ops.aten.copy_.default(primals_1245, add_828);  primals_1245 = add_828 = None
    copy__468: "f32[864]" = torch.ops.aten.copy_.default(primals_1246, add_836);  primals_1246 = add_836 = None
    copy__469: "f32[864]" = torch.ops.aten.copy_.default(primals_1247, add_837);  primals_1247 = add_837 = None
    copy__470: "i64[]" = torch.ops.aten.copy_.default(primals_1248, add_834);  primals_1248 = add_834 = None
    copy__471: "f32[864]" = torch.ops.aten.copy_.default(primals_1249, add_841);  primals_1249 = add_841 = None
    copy__472: "f32[864]" = torch.ops.aten.copy_.default(primals_1250, add_842);  primals_1250 = add_842 = None
    copy__473: "i64[]" = torch.ops.aten.copy_.default(primals_1251, add_839);  primals_1251 = add_839 = None
    copy__474: "f32[864]" = torch.ops.aten.copy_.default(primals_1252, add_846);  primals_1252 = add_846 = None
    copy__475: "f32[864]" = torch.ops.aten.copy_.default(primals_1253, add_847);  primals_1253 = add_847 = None
    copy__476: "i64[]" = torch.ops.aten.copy_.default(primals_1254, add_844);  primals_1254 = add_844 = None
    copy__477: "f32[864]" = torch.ops.aten.copy_.default(primals_1255, add_852);  primals_1255 = add_852 = None
    copy__478: "f32[864]" = torch.ops.aten.copy_.default(primals_1256, add_853);  primals_1256 = add_853 = None
    copy__479: "i64[]" = torch.ops.aten.copy_.default(primals_1257, add_850);  primals_1257 = add_850 = None
    copy__480: "f32[864]" = torch.ops.aten.copy_.default(primals_1258, add_857);  primals_1258 = add_857 = None
    copy__481: "f32[864]" = torch.ops.aten.copy_.default(primals_1259, add_858);  primals_1259 = add_858 = None
    copy__482: "i64[]" = torch.ops.aten.copy_.default(primals_1260, add_855);  primals_1260 = add_855 = None
    copy__483: "f32[864]" = torch.ops.aten.copy_.default(primals_1261, add_862);  primals_1261 = add_862 = None
    copy__484: "f32[864]" = torch.ops.aten.copy_.default(primals_1262, add_863);  primals_1262 = add_863 = None
    copy__485: "i64[]" = torch.ops.aten.copy_.default(primals_1263, add_860);  primals_1263 = add_860 = None
    copy__486: "f32[864]" = torch.ops.aten.copy_.default(primals_1264, add_867);  primals_1264 = add_867 = None
    copy__487: "f32[864]" = torch.ops.aten.copy_.default(primals_1265, add_868);  primals_1265 = add_868 = None
    copy__488: "i64[]" = torch.ops.aten.copy_.default(primals_1266, add_865);  primals_1266 = add_865 = None
    copy__489: "f32[864]" = torch.ops.aten.copy_.default(primals_1267, add_873);  primals_1267 = add_873 = None
    copy__490: "f32[864]" = torch.ops.aten.copy_.default(primals_1268, add_874);  primals_1268 = add_874 = None
    copy__491: "i64[]" = torch.ops.aten.copy_.default(primals_1269, add_871);  primals_1269 = add_871 = None
    copy__492: "f32[864]" = torch.ops.aten.copy_.default(primals_1270, add_878);  primals_1270 = add_878 = None
    copy__493: "f32[864]" = torch.ops.aten.copy_.default(primals_1271, add_879);  primals_1271 = add_879 = None
    copy__494: "i64[]" = torch.ops.aten.copy_.default(primals_1272, add_876);  primals_1272 = add_876 = None
    copy__495: "f32[864]" = torch.ops.aten.copy_.default(primals_1273, add_884);  primals_1273 = add_884 = None
    copy__496: "f32[864]" = torch.ops.aten.copy_.default(primals_1274, add_885);  primals_1274 = add_885 = None
    copy__497: "i64[]" = torch.ops.aten.copy_.default(primals_1275, add_882);  primals_1275 = add_882 = None
    copy__498: "f32[864]" = torch.ops.aten.copy_.default(primals_1276, add_889);  primals_1276 = add_889 = None
    copy__499: "f32[864]" = torch.ops.aten.copy_.default(primals_1277, add_890);  primals_1277 = add_890 = None
    copy__500: "i64[]" = torch.ops.aten.copy_.default(primals_1278, add_887);  primals_1278 = add_887 = None
    copy__501: "f32[864]" = torch.ops.aten.copy_.default(primals_1279, add_894);  primals_1279 = add_894 = None
    copy__502: "f32[864]" = torch.ops.aten.copy_.default(primals_1280, add_895);  primals_1280 = add_895 = None
    copy__503: "i64[]" = torch.ops.aten.copy_.default(primals_1281, add_892);  primals_1281 = add_892 = None
    copy__504: "f32[864]" = torch.ops.aten.copy_.default(primals_1282, add_899);  primals_1282 = add_899 = None
    copy__505: "f32[864]" = torch.ops.aten.copy_.default(primals_1283, add_900);  primals_1283 = add_900 = None
    copy__506: "i64[]" = torch.ops.aten.copy_.default(primals_1284, add_897);  primals_1284 = add_897 = None
    copy__507: "f32[864]" = torch.ops.aten.copy_.default(primals_1285, add_905);  primals_1285 = add_905 = None
    copy__508: "f32[864]" = torch.ops.aten.copy_.default(primals_1286, add_906);  primals_1286 = add_906 = None
    copy__509: "i64[]" = torch.ops.aten.copy_.default(primals_1287, add_903);  primals_1287 = add_903 = None
    copy__510: "f32[864]" = torch.ops.aten.copy_.default(primals_1288, add_910);  primals_1288 = add_910 = None
    copy__511: "f32[864]" = torch.ops.aten.copy_.default(primals_1289, add_911);  primals_1289 = add_911 = None
    copy__512: "i64[]" = torch.ops.aten.copy_.default(primals_1290, add_908);  primals_1290 = add_908 = None
    copy__513: "f32[864]" = torch.ops.aten.copy_.default(primals_1291, add_916);  primals_1291 = add_916 = None
    copy__514: "f32[864]" = torch.ops.aten.copy_.default(primals_1292, add_917);  primals_1292 = add_917 = None
    copy__515: "i64[]" = torch.ops.aten.copy_.default(primals_1293, add_914);  primals_1293 = add_914 = None
    copy__516: "f32[864]" = torch.ops.aten.copy_.default(primals_1294, add_921);  primals_1294 = add_921 = None
    copy__517: "f32[864]" = torch.ops.aten.copy_.default(primals_1295, add_922);  primals_1295 = add_922 = None
    copy__518: "i64[]" = torch.ops.aten.copy_.default(primals_1296, add_919);  primals_1296 = add_919 = None
    copy__519: "f32[864]" = torch.ops.aten.copy_.default(primals_1297, add_927);  primals_1297 = add_927 = None
    copy__520: "f32[864]" = torch.ops.aten.copy_.default(primals_1298, add_928);  primals_1298 = add_928 = None
    copy__521: "i64[]" = torch.ops.aten.copy_.default(primals_1299, add_925);  primals_1299 = add_925 = None
    copy__522: "f32[864]" = torch.ops.aten.copy_.default(primals_1300, add_932);  primals_1300 = add_932 = None
    copy__523: "f32[864]" = torch.ops.aten.copy_.default(primals_1301, add_933);  primals_1301 = add_933 = None
    copy__524: "i64[]" = torch.ops.aten.copy_.default(primals_1302, add_930);  primals_1302 = add_930 = None
    copy__525: "f32[864]" = torch.ops.aten.copy_.default(primals_1303, add_937);  primals_1303 = add_937 = None
    copy__526: "f32[864]" = torch.ops.aten.copy_.default(primals_1304, add_938);  primals_1304 = add_938 = None
    copy__527: "i64[]" = torch.ops.aten.copy_.default(primals_1305, add_935);  primals_1305 = add_935 = None
    copy__528: "f32[864]" = torch.ops.aten.copy_.default(primals_1306, add_942);  primals_1306 = add_942 = None
    copy__529: "f32[864]" = torch.ops.aten.copy_.default(primals_1307, add_943);  primals_1307 = add_943 = None
    copy__530: "i64[]" = torch.ops.aten.copy_.default(primals_1308, add_940);  primals_1308 = add_940 = None
    copy__531: "f32[864]" = torch.ops.aten.copy_.default(primals_1309, add_948);  primals_1309 = add_948 = None
    copy__532: "f32[864]" = torch.ops.aten.copy_.default(primals_1310, add_949);  primals_1310 = add_949 = None
    copy__533: "i64[]" = torch.ops.aten.copy_.default(primals_1311, add_946);  primals_1311 = add_946 = None
    copy__534: "f32[864]" = torch.ops.aten.copy_.default(primals_1312, add_953);  primals_1312 = add_953 = None
    copy__535: "f32[864]" = torch.ops.aten.copy_.default(primals_1313, add_954);  primals_1313 = add_954 = None
    copy__536: "i64[]" = torch.ops.aten.copy_.default(primals_1314, add_951);  primals_1314 = add_951 = None
    copy__537: "f32[864]" = torch.ops.aten.copy_.default(primals_1315, add_959);  primals_1315 = add_959 = None
    copy__538: "f32[864]" = torch.ops.aten.copy_.default(primals_1316, add_960);  primals_1316 = add_960 = None
    copy__539: "i64[]" = torch.ops.aten.copy_.default(primals_1317, add_957);  primals_1317 = add_957 = None
    copy__540: "f32[864]" = torch.ops.aten.copy_.default(primals_1318, add_964);  primals_1318 = add_964 = None
    copy__541: "f32[864]" = torch.ops.aten.copy_.default(primals_1319, add_965);  primals_1319 = add_965 = None
    copy__542: "i64[]" = torch.ops.aten.copy_.default(primals_1320, add_962);  primals_1320 = add_962 = None
    copy__543: "f32[864]" = torch.ops.aten.copy_.default(primals_1321, add_969);  primals_1321 = add_969 = None
    copy__544: "f32[864]" = torch.ops.aten.copy_.default(primals_1322, add_970);  primals_1322 = add_970 = None
    copy__545: "i64[]" = torch.ops.aten.copy_.default(primals_1323, add_967);  primals_1323 = add_967 = None
    copy__546: "f32[864]" = torch.ops.aten.copy_.default(primals_1324, add_974);  primals_1324 = add_974 = None
    copy__547: "f32[864]" = torch.ops.aten.copy_.default(primals_1325, add_975);  primals_1325 = add_975 = None
    copy__548: "i64[]" = torch.ops.aten.copy_.default(primals_1326, add_972);  primals_1326 = add_972 = None
    copy__549: "f32[864]" = torch.ops.aten.copy_.default(primals_1327, add_980);  primals_1327 = add_980 = None
    copy__550: "f32[864]" = torch.ops.aten.copy_.default(primals_1328, add_981);  primals_1328 = add_981 = None
    copy__551: "i64[]" = torch.ops.aten.copy_.default(primals_1329, add_978);  primals_1329 = add_978 = None
    copy__552: "f32[864]" = torch.ops.aten.copy_.default(primals_1330, add_985);  primals_1330 = add_985 = None
    copy__553: "f32[864]" = torch.ops.aten.copy_.default(primals_1331, add_986);  primals_1331 = add_986 = None
    copy__554: "i64[]" = torch.ops.aten.copy_.default(primals_1332, add_983);  primals_1332 = add_983 = None
    copy__555: "f32[864]" = torch.ops.aten.copy_.default(primals_1333, add_991);  primals_1333 = add_991 = None
    copy__556: "f32[864]" = torch.ops.aten.copy_.default(primals_1334, add_992);  primals_1334 = add_992 = None
    copy__557: "i64[]" = torch.ops.aten.copy_.default(primals_1335, add_989);  primals_1335 = add_989 = None
    copy__558: "f32[864]" = torch.ops.aten.copy_.default(primals_1336, add_996);  primals_1336 = add_996 = None
    copy__559: "f32[864]" = torch.ops.aten.copy_.default(primals_1337, add_997);  primals_1337 = add_997 = None
    copy__560: "i64[]" = torch.ops.aten.copy_.default(primals_1338, add_994);  primals_1338 = add_994 = None
    copy__561: "f32[864]" = torch.ops.aten.copy_.default(primals_1339, add_1002);  primals_1339 = add_1002 = None
    copy__562: "f32[864]" = torch.ops.aten.copy_.default(primals_1340, add_1003);  primals_1340 = add_1003 = None
    copy__563: "i64[]" = torch.ops.aten.copy_.default(primals_1341, add_1000);  primals_1341 = add_1000 = None
    copy__564: "f32[864]" = torch.ops.aten.copy_.default(primals_1342, add_1007);  primals_1342 = add_1007 = None
    copy__565: "f32[864]" = torch.ops.aten.copy_.default(primals_1343, add_1008);  primals_1343 = add_1008 = None
    copy__566: "i64[]" = torch.ops.aten.copy_.default(primals_1344, add_1005);  primals_1344 = add_1005 = None
    copy__567: "f32[864]" = torch.ops.aten.copy_.default(primals_1345, add_1012);  primals_1345 = add_1012 = None
    copy__568: "f32[864]" = torch.ops.aten.copy_.default(primals_1346, add_1013);  primals_1346 = add_1013 = None
    copy__569: "i64[]" = torch.ops.aten.copy_.default(primals_1347, add_1010);  primals_1347 = add_1010 = None
    copy__570: "f32[864]" = torch.ops.aten.copy_.default(primals_1348, add_1017);  primals_1348 = add_1017 = None
    copy__571: "f32[864]" = torch.ops.aten.copy_.default(primals_1349, add_1018);  primals_1349 = add_1018 = None
    copy__572: "i64[]" = torch.ops.aten.copy_.default(primals_1350, add_1015);  primals_1350 = add_1015 = None
    copy__573: "f32[864]" = torch.ops.aten.copy_.default(primals_1351, add_1023);  primals_1351 = add_1023 = None
    copy__574: "f32[864]" = torch.ops.aten.copy_.default(primals_1352, add_1024);  primals_1352 = add_1024 = None
    copy__575: "i64[]" = torch.ops.aten.copy_.default(primals_1353, add_1021);  primals_1353 = add_1021 = None
    copy__576: "f32[864]" = torch.ops.aten.copy_.default(primals_1354, add_1028);  primals_1354 = add_1028 = None
    copy__577: "f32[864]" = torch.ops.aten.copy_.default(primals_1355, add_1029);  primals_1355 = add_1029 = None
    copy__578: "i64[]" = torch.ops.aten.copy_.default(primals_1356, add_1026);  primals_1356 = add_1026 = None
    copy__579: "f32[864]" = torch.ops.aten.copy_.default(primals_1357, add_1034);  primals_1357 = add_1034 = None
    copy__580: "f32[864]" = torch.ops.aten.copy_.default(primals_1358, add_1035);  primals_1358 = add_1035 = None
    copy__581: "i64[]" = torch.ops.aten.copy_.default(primals_1359, add_1032);  primals_1359 = add_1032 = None
    copy__582: "f32[864]" = torch.ops.aten.copy_.default(primals_1360, add_1039);  primals_1360 = add_1039 = None
    copy__583: "f32[864]" = torch.ops.aten.copy_.default(primals_1361, add_1040);  primals_1361 = add_1040 = None
    copy__584: "i64[]" = torch.ops.aten.copy_.default(primals_1362, add_1037);  primals_1362 = add_1037 = None
    copy__585: "f32[864]" = torch.ops.aten.copy_.default(primals_1363, add_1044);  primals_1363 = add_1044 = None
    copy__586: "f32[864]" = torch.ops.aten.copy_.default(primals_1364, add_1045);  primals_1364 = add_1045 = None
    copy__587: "i64[]" = torch.ops.aten.copy_.default(primals_1365, add_1042);  primals_1365 = add_1042 = None
    copy__588: "f32[864]" = torch.ops.aten.copy_.default(primals_1366, add_1049);  primals_1366 = add_1049 = None
    copy__589: "f32[864]" = torch.ops.aten.copy_.default(primals_1367, add_1050);  primals_1367 = add_1050 = None
    copy__590: "i64[]" = torch.ops.aten.copy_.default(primals_1368, add_1047);  primals_1368 = add_1047 = None
    copy__591: "f32[864]" = torch.ops.aten.copy_.default(primals_1369, add_1055);  primals_1369 = add_1055 = None
    copy__592: "f32[864]" = torch.ops.aten.copy_.default(primals_1370, add_1056);  primals_1370 = add_1056 = None
    copy__593: "i64[]" = torch.ops.aten.copy_.default(primals_1371, add_1053);  primals_1371 = add_1053 = None
    copy__594: "f32[864]" = torch.ops.aten.copy_.default(primals_1372, add_1060);  primals_1372 = add_1060 = None
    copy__595: "f32[864]" = torch.ops.aten.copy_.default(primals_1373, add_1061);  primals_1373 = add_1061 = None
    copy__596: "i64[]" = torch.ops.aten.copy_.default(primals_1374, add_1058);  primals_1374 = add_1058 = None
    copy__597: "f32[864]" = torch.ops.aten.copy_.default(primals_1375, add_1066);  primals_1375 = add_1066 = None
    copy__598: "f32[864]" = torch.ops.aten.copy_.default(primals_1376, add_1067);  primals_1376 = add_1067 = None
    copy__599: "i64[]" = torch.ops.aten.copy_.default(primals_1377, add_1064);  primals_1377 = add_1064 = None
    copy__600: "f32[864]" = torch.ops.aten.copy_.default(primals_1378, add_1071);  primals_1378 = add_1071 = None
    copy__601: "f32[864]" = torch.ops.aten.copy_.default(primals_1379, add_1072);  primals_1379 = add_1072 = None
    copy__602: "i64[]" = torch.ops.aten.copy_.default(primals_1380, add_1069);  primals_1380 = add_1069 = None
    return [addmm, primals_1, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_66, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_75, primals_77, primals_79, primals_80, primals_81, primals_83, primals_84, primals_86, primals_87, primals_89, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_105, primals_107, primals_108, primals_110, primals_111, primals_112, primals_114, primals_115, primals_116, primals_118, primals_119, primals_120, primals_122, primals_123, primals_125, primals_126, primals_127, primals_129, primals_131, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_140, primals_142, primals_143, primals_144, primals_146, primals_147, primals_148, primals_150, primals_151, primals_152, primals_154, primals_155, primals_156, primals_158, primals_159, primals_160, primals_162, primals_163, primals_164, primals_166, primals_167, primals_168, primals_170, primals_171, primals_172, primals_174, primals_175, primals_176, primals_178, primals_179, primals_180, primals_182, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_194, primals_196, primals_197, primals_198, primals_200, primals_201, primals_202, primals_204, primals_205, primals_206, primals_208, primals_209, primals_210, primals_212, primals_213, primals_214, primals_216, primals_217, primals_218, primals_220, primals_221, primals_222, primals_224, primals_225, primals_226, primals_228, primals_229, primals_230, primals_232, primals_233, primals_234, primals_236, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_248, primals_250, primals_251, primals_252, primals_254, primals_255, primals_256, primals_258, primals_259, primals_260, primals_262, primals_263, primals_264, primals_266, primals_267, primals_268, primals_270, primals_271, primals_272, primals_274, primals_275, primals_276, primals_278, primals_279, primals_280, primals_282, primals_283, primals_284, primals_286, primals_287, primals_288, primals_290, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_302, primals_304, primals_305, primals_306, primals_308, primals_309, primals_310, primals_312, primals_313, primals_314, primals_316, primals_317, primals_318, primals_320, primals_321, primals_322, primals_324, primals_325, primals_326, primals_328, primals_329, primals_330, primals_332, primals_333, primals_334, primals_336, primals_337, primals_338, primals_340, primals_341, primals_342, primals_344, primals_345, primals_346, primals_348, primals_349, primals_351, primals_352, primals_354, primals_355, primals_357, primals_358, primals_359, primals_361, primals_362, primals_364, primals_365, primals_366, primals_368, primals_369, primals_371, primals_372, primals_373, primals_375, primals_376, primals_378, primals_379, primals_380, primals_382, primals_383, primals_384, primals_386, primals_387, primals_388, primals_390, primals_391, primals_393, primals_394, primals_395, primals_397, primals_399, primals_400, primals_401, primals_403, primals_404, primals_406, primals_407, primals_408, primals_410, primals_411, primals_412, primals_414, primals_415, primals_416, primals_418, primals_419, primals_420, primals_422, primals_423, primals_424, primals_426, primals_427, primals_428, primals_430, primals_431, primals_432, primals_434, primals_435, primals_436, primals_438, primals_439, primals_440, primals_442, primals_443, primals_444, primals_446, primals_447, primals_448, primals_450, primals_451, primals_452, primals_454, primals_455, primals_457, primals_458, primals_460, primals_461, primals_462, primals_464, primals_465, primals_466, primals_468, primals_469, primals_470, primals_472, primals_473, primals_474, primals_476, primals_477, primals_478, primals_480, primals_481, primals_482, primals_484, primals_485, primals_486, primals_488, primals_489, primals_490, primals_492, primals_493, primals_494, primals_496, primals_497, primals_498, primals_500, primals_501, primals_502, primals_504, primals_505, primals_506, primals_508, primals_509, primals_511, primals_512, primals_514, primals_515, primals_516, primals_518, primals_519, primals_520, primals_522, primals_523, primals_524, primals_526, primals_527, primals_528, primals_530, primals_531, primals_532, primals_534, primals_535, primals_536, primals_538, primals_539, primals_540, primals_542, primals_543, primals_544, primals_546, primals_547, primals_548, primals_550, primals_551, primals_552, primals_554, primals_555, primals_556, primals_558, primals_559, primals_560, primals_562, primals_563, primals_565, primals_566, primals_568, primals_569, primals_571, primals_572, primals_573, primals_575, primals_576, primals_578, primals_579, primals_580, primals_582, primals_583, primals_585, primals_586, primals_587, primals_589, primals_590, primals_592, primals_593, primals_594, primals_596, primals_597, primals_598, primals_600, primals_601, primals_602, primals_604, primals_605, primals_607, primals_608, primals_609, primals_611, primals_613, primals_614, primals_615, primals_617, primals_618, primals_620, primals_621, primals_622, primals_624, primals_625, primals_626, primals_628, primals_629, primals_630, primals_632, primals_633, primals_634, primals_636, primals_637, primals_638, primals_640, primals_641, primals_642, primals_644, primals_645, primals_646, primals_648, primals_649, primals_650, primals_652, primals_653, primals_654, primals_656, primals_657, primals_658, primals_660, primals_661, primals_662, primals_664, primals_665, primals_666, primals_668, primals_669, primals_671, primals_672, primals_674, primals_675, primals_676, primals_678, primals_679, primals_680, primals_682, primals_683, primals_684, primals_686, primals_687, primals_688, primals_690, primals_691, primals_692, primals_694, primals_695, primals_696, primals_698, primals_699, primals_700, primals_702, primals_703, primals_704, primals_706, primals_707, primals_708, primals_710, primals_711, primals_712, primals_714, primals_715, primals_716, primals_718, primals_719, primals_720, primals_722, primals_723, primals_725, primals_726, primals_728, primals_729, primals_730, primals_732, primals_733, primals_734, primals_736, primals_737, primals_738, primals_740, primals_741, primals_742, primals_744, primals_745, primals_746, primals_748, primals_749, primals_750, primals_752, primals_753, primals_754, primals_756, primals_757, primals_758, primals_760, primals_761, primals_762, primals_764, primals_765, primals_766, primals_768, primals_769, primals_770, primals_772, primals_773, primals_774, primals_1381, convolution, squeeze_1, relu, convolution_1, squeeze_4, constant_pad_nd, convolution_2, convolution_3, squeeze_7, relu_2, convolution_4, convolution_5, squeeze_10, constant_pad_nd_1, getitem_8, getitem_9, convolution_6, squeeze_13, constant_pad_nd_2, convolution_7, convolution_8, squeeze_16, relu_4, convolution_9, convolution_10, squeeze_19, constant_pad_nd_3, getitem_17, constant_pad_nd_4, convolution_11, convolution_12, squeeze_22, relu_6, convolution_13, convolution_14, squeeze_25, constant_pad_nd_5, convolution_15, convolution_16, squeeze_28, relu_8, convolution_17, convolution_18, squeeze_31, relu_9, convolution_19, convolution_20, squeeze_34, relu_10, convolution_21, convolution_22, squeeze_37, constant_pad_nd_7, convolution_23, convolution_24, squeeze_40, relu_12, convolution_25, convolution_26, squeeze_43, relu_3, convolution_27, squeeze_46, avg_pool2d, constant_pad_nd_9, avg_pool2d_1, cat_1, squeeze_49, relu_15, convolution_30, squeeze_52, constant_pad_nd_10, convolution_31, convolution_32, squeeze_55, relu_17, convolution_33, convolution_34, squeeze_58, constant_pad_nd_11, getitem_47, constant_pad_nd_12, convolution_35, convolution_36, squeeze_61, relu_19, convolution_37, convolution_38, squeeze_64, constant_pad_nd_13, getitem_53, constant_pad_nd_14, convolution_39, convolution_40, squeeze_67, relu_21, convolution_41, convolution_42, squeeze_70, constant_pad_nd_15, convolution_43, convolution_44, squeeze_73, relu_23, convolution_45, convolution_46, squeeze_76, relu_24, convolution_47, convolution_48, squeeze_79, relu_25, convolution_49, convolution_50, squeeze_82, constant_pad_nd_17, convolution_51, convolution_52, squeeze_85, relu_27, convolution_53, convolution_54, squeeze_88, relu_18, convolution_55, squeeze_91, avg_pool2d_2, constant_pad_nd_19, avg_pool2d_3, cat_3, squeeze_94, add_169, relu_30, convolution_58, squeeze_97, add_174, relu_31, convolution_59, convolution_60, squeeze_100, relu_32, convolution_61, convolution_62, squeeze_103, getitem_83, relu_33, convolution_63, convolution_64, squeeze_106, relu_34, convolution_65, convolution_66, squeeze_109, getitem_89, convolution_67, convolution_68, squeeze_112, relu_36, convolution_69, convolution_70, squeeze_115, convolution_71, convolution_72, squeeze_118, relu_38, convolution_73, convolution_74, squeeze_121, relu_39, convolution_75, convolution_76, squeeze_124, relu_40, convolution_77, convolution_78, squeeze_127, convolution_79, convolution_80, squeeze_130, relu_42, convolution_81, convolution_82, squeeze_133, convolution_83, squeeze_136, add_244, relu_44, convolution_84, squeeze_139, add_249, relu_45, convolution_85, convolution_86, squeeze_142, relu_46, convolution_87, convolution_88, squeeze_145, getitem_117, relu_47, convolution_89, convolution_90, squeeze_148, relu_48, convolution_91, convolution_92, squeeze_151, getitem_123, convolution_93, convolution_94, squeeze_154, relu_50, convolution_95, convolution_96, squeeze_157, convolution_97, convolution_98, squeeze_160, relu_52, convolution_99, convolution_100, squeeze_163, relu_53, convolution_101, convolution_102, squeeze_166, relu_54, convolution_103, convolution_104, squeeze_169, convolution_105, convolution_106, squeeze_172, relu_56, convolution_107, convolution_108, squeeze_175, convolution_109, squeeze_178, add_319, relu_58, convolution_110, squeeze_181, add_324, relu_59, convolution_111, convolution_112, squeeze_184, relu_60, convolution_113, convolution_114, squeeze_187, getitem_151, relu_61, convolution_115, convolution_116, squeeze_190, relu_62, convolution_117, convolution_118, squeeze_193, getitem_157, convolution_119, convolution_120, squeeze_196, relu_64, convolution_121, convolution_122, squeeze_199, convolution_123, convolution_124, squeeze_202, relu_66, convolution_125, convolution_126, squeeze_205, relu_67, convolution_127, convolution_128, squeeze_208, relu_68, convolution_129, convolution_130, squeeze_211, convolution_131, convolution_132, squeeze_214, relu_70, convolution_133, convolution_134, squeeze_217, convolution_135, squeeze_220, add_394, relu_72, convolution_136, squeeze_223, add_399, relu_73, convolution_137, convolution_138, squeeze_226, relu_74, convolution_139, convolution_140, squeeze_229, getitem_185, relu_75, convolution_141, convolution_142, squeeze_232, relu_76, convolution_143, convolution_144, squeeze_235, getitem_191, convolution_145, convolution_146, squeeze_238, relu_78, convolution_147, convolution_148, squeeze_241, convolution_149, convolution_150, squeeze_244, relu_80, convolution_151, convolution_152, squeeze_247, relu_81, convolution_153, convolution_154, squeeze_250, relu_82, convolution_155, convolution_156, squeeze_253, convolution_157, convolution_158, squeeze_256, relu_84, convolution_159, convolution_160, squeeze_259, convolution_161, squeeze_262, relu_86, convolution_162, squeeze_265, constant_pad_nd_20, convolution_163, convolution_164, squeeze_268, relu_88, convolution_165, convolution_166, squeeze_271, constant_pad_nd_21, getitem_219, constant_pad_nd_22, convolution_167, convolution_168, squeeze_274, relu_90, convolution_169, convolution_170, squeeze_277, constant_pad_nd_23, getitem_225, constant_pad_nd_24, convolution_171, convolution_172, squeeze_280, relu_92, convolution_173, convolution_174, squeeze_283, constant_pad_nd_25, convolution_175, convolution_176, squeeze_286, relu_94, convolution_177, convolution_178, squeeze_289, relu_95, convolution_179, convolution_180, squeeze_292, relu_96, convolution_181, convolution_182, squeeze_295, constant_pad_nd_27, convolution_183, convolution_184, squeeze_298, relu_98, convolution_185, convolution_186, squeeze_301, relu_89, convolution_187, squeeze_304, avg_pool2d_4, constant_pad_nd_29, avg_pool2d_5, cat_9, squeeze_307, add_549, relu_101, convolution_190, squeeze_310, add_554, relu_102, convolution_191, convolution_192, squeeze_313, relu_103, convolution_193, convolution_194, squeeze_316, getitem_255, relu_104, convolution_195, convolution_196, squeeze_319, relu_105, convolution_197, convolution_198, squeeze_322, getitem_261, convolution_199, convolution_200, squeeze_325, relu_107, convolution_201, convolution_202, squeeze_328, convolution_203, convolution_204, squeeze_331, relu_109, convolution_205, convolution_206, squeeze_334, relu_110, convolution_207, convolution_208, squeeze_337, relu_111, convolution_209, convolution_210, squeeze_340, convolution_211, convolution_212, squeeze_343, relu_113, convolution_213, convolution_214, squeeze_346, convolution_215, squeeze_349, add_624, relu_115, convolution_216, squeeze_352, add_629, relu_116, convolution_217, convolution_218, squeeze_355, relu_117, convolution_219, convolution_220, squeeze_358, getitem_289, relu_118, convolution_221, convolution_222, squeeze_361, relu_119, convolution_223, convolution_224, squeeze_364, getitem_295, convolution_225, convolution_226, squeeze_367, relu_121, convolution_227, convolution_228, squeeze_370, convolution_229, convolution_230, squeeze_373, relu_123, convolution_231, convolution_232, squeeze_376, relu_124, convolution_233, convolution_234, squeeze_379, relu_125, convolution_235, convolution_236, squeeze_382, convolution_237, convolution_238, squeeze_385, relu_127, convolution_239, convolution_240, squeeze_388, convolution_241, squeeze_391, add_699, relu_129, convolution_242, squeeze_394, add_704, relu_130, convolution_243, convolution_244, squeeze_397, relu_131, convolution_245, convolution_246, squeeze_400, getitem_323, relu_132, convolution_247, convolution_248, squeeze_403, relu_133, convolution_249, convolution_250, squeeze_406, getitem_329, convolution_251, convolution_252, squeeze_409, relu_135, convolution_253, convolution_254, squeeze_412, convolution_255, convolution_256, squeeze_415, relu_137, convolution_257, convolution_258, squeeze_418, relu_138, convolution_259, convolution_260, squeeze_421, relu_139, convolution_261, convolution_262, squeeze_424, convolution_263, convolution_264, squeeze_427, relu_141, convolution_265, convolution_266, squeeze_430, convolution_267, squeeze_433, relu_143, convolution_268, squeeze_436, constant_pad_nd_30, convolution_269, convolution_270, squeeze_439, relu_145, convolution_271, convolution_272, squeeze_442, constant_pad_nd_31, getitem_357, constant_pad_nd_32, convolution_273, convolution_274, squeeze_445, relu_147, convolution_275, convolution_276, squeeze_448, constant_pad_nd_33, getitem_363, constant_pad_nd_34, convolution_277, convolution_278, squeeze_451, relu_149, convolution_279, convolution_280, squeeze_454, constant_pad_nd_35, convolution_281, convolution_282, squeeze_457, relu_151, convolution_283, convolution_284, squeeze_460, relu_152, convolution_285, convolution_286, squeeze_463, relu_153, convolution_287, convolution_288, squeeze_466, constant_pad_nd_37, convolution_289, convolution_290, squeeze_469, relu_155, convolution_291, convolution_292, squeeze_472, relu_146, convolution_293, squeeze_475, avg_pool2d_6, constant_pad_nd_39, avg_pool2d_7, cat_14, squeeze_478, add_854, relu_158, convolution_296, squeeze_481, add_859, relu_159, convolution_297, convolution_298, squeeze_484, relu_160, convolution_299, convolution_300, squeeze_487, getitem_393, relu_161, convolution_301, convolution_302, squeeze_490, relu_162, convolution_303, convolution_304, squeeze_493, getitem_399, convolution_305, convolution_306, squeeze_496, relu_164, convolution_307, convolution_308, squeeze_499, convolution_309, convolution_310, squeeze_502, relu_166, convolution_311, convolution_312, squeeze_505, relu_167, convolution_313, convolution_314, squeeze_508, relu_168, convolution_315, convolution_316, squeeze_511, convolution_317, convolution_318, squeeze_514, relu_170, convolution_319, convolution_320, squeeze_517, convolution_321, squeeze_520, add_929, relu_172, convolution_322, squeeze_523, add_934, relu_173, convolution_323, convolution_324, squeeze_526, relu_174, convolution_325, convolution_326, squeeze_529, getitem_427, relu_175, convolution_327, convolution_328, squeeze_532, relu_176, convolution_329, convolution_330, squeeze_535, getitem_433, convolution_331, convolution_332, squeeze_538, relu_178, convolution_333, convolution_334, squeeze_541, convolution_335, convolution_336, squeeze_544, relu_180, convolution_337, convolution_338, squeeze_547, relu_181, convolution_339, convolution_340, squeeze_550, relu_182, convolution_341, convolution_342, squeeze_553, convolution_343, convolution_344, squeeze_556, relu_184, convolution_345, convolution_346, squeeze_559, convolution_347, squeeze_562, add_1004, relu_186, convolution_348, squeeze_565, add_1009, relu_187, convolution_349, convolution_350, squeeze_568, relu_188, convolution_351, convolution_352, squeeze_571, getitem_461, relu_189, convolution_353, convolution_354, squeeze_574, relu_190, convolution_355, convolution_356, squeeze_577, getitem_467, convolution_357, convolution_358, squeeze_580, relu_192, convolution_359, convolution_360, squeeze_583, convolution_361, convolution_362, squeeze_586, relu_194, convolution_363, convolution_364, squeeze_589, relu_195, convolution_365, convolution_366, squeeze_592, relu_196, convolution_367, convolution_368, squeeze_595, convolution_369, convolution_370, squeeze_598, relu_198, convolution_371, convolution_372, squeeze_601, view, permute_1, le, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, unsqueeze_914, unsqueeze_926, unsqueeze_938, unsqueeze_950, unsqueeze_962, unsqueeze_974, unsqueeze_986, unsqueeze_998, unsqueeze_1010, unsqueeze_1022, unsqueeze_1034, unsqueeze_1046, unsqueeze_1058, unsqueeze_1070, unsqueeze_1082, unsqueeze_1094, unsqueeze_1106, unsqueeze_1118, unsqueeze_1130, unsqueeze_1142, unsqueeze_1154, unsqueeze_1166, unsqueeze_1178, unsqueeze_1190, unsqueeze_1202, unsqueeze_1214, unsqueeze_1226, unsqueeze_1238, unsqueeze_1250, unsqueeze_1262, unsqueeze_1274, unsqueeze_1286, unsqueeze_1298, unsqueeze_1310, le_43, unsqueeze_1322, unsqueeze_1334, le_45, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, unsqueeze_1382, unsqueeze_1394, unsqueeze_1406, unsqueeze_1418, unsqueeze_1430, unsqueeze_1442, unsqueeze_1454, unsqueeze_1466, unsqueeze_1478, unsqueeze_1490, unsqueeze_1502, unsqueeze_1514, unsqueeze_1526, unsqueeze_1538, unsqueeze_1550, unsqueeze_1562, unsqueeze_1574, unsqueeze_1586, unsqueeze_1598, unsqueeze_1610, unsqueeze_1622, unsqueeze_1634, unsqueeze_1646, unsqueeze_1658, unsqueeze_1670, unsqueeze_1682, unsqueeze_1694, unsqueeze_1706, unsqueeze_1718, unsqueeze_1730, unsqueeze_1742, unsqueeze_1754, unsqueeze_1766, unsqueeze_1778, unsqueeze_1790, unsqueeze_1802, unsqueeze_1814, unsqueeze_1826, unsqueeze_1838, unsqueeze_1850, unsqueeze_1862, unsqueeze_1874, unsqueeze_1886, unsqueeze_1898, unsqueeze_1910, unsqueeze_1922, unsqueeze_1934, unsqueeze_1946, unsqueeze_1958, unsqueeze_1970, unsqueeze_1982, unsqueeze_1994, le_100, unsqueeze_2006, unsqueeze_2018, le_102, unsqueeze_2030, unsqueeze_2042, unsqueeze_2054, unsqueeze_2066, unsqueeze_2078, unsqueeze_2090, unsqueeze_2102, unsqueeze_2114, unsqueeze_2126, unsqueeze_2138, unsqueeze_2150, unsqueeze_2162, unsqueeze_2174, unsqueeze_2186, unsqueeze_2198, unsqueeze_2210, unsqueeze_2222, unsqueeze_2234, unsqueeze_2246, unsqueeze_2258, unsqueeze_2270, unsqueeze_2282, unsqueeze_2294, unsqueeze_2306, unsqueeze_2318, unsqueeze_2330, unsqueeze_2342, unsqueeze_2354, unsqueeze_2366, unsqueeze_2378, unsqueeze_2390, unsqueeze_2402, unsqueeze_2414, unsqueeze_2426, unsqueeze_2438, unsqueeze_2450, unsqueeze_2462, unsqueeze_2474, unsqueeze_2486, unsqueeze_2498, unsqueeze_2510, unsqueeze_2522, unsqueeze_2534, unsqueeze_2546, unsqueeze_2558, unsqueeze_2570, unsqueeze_2582, unsqueeze_2594, unsqueeze_2606, unsqueeze_2618, unsqueeze_2630, unsqueeze_2642, unsqueeze_2654, unsqueeze_2666, unsqueeze_2678, unsqueeze_2690, unsqueeze_2702, unsqueeze_2714, unsqueeze_2726, unsqueeze_2738, unsqueeze_2750, unsqueeze_2762, unsqueeze_2774, unsqueeze_2786, unsqueeze_2798, unsqueeze_2810, unsqueeze_2822, unsqueeze_2834, unsqueeze_2846, le_171, unsqueeze_2858, unsqueeze_2870, le_173, unsqueeze_2882, unsqueeze_2894, unsqueeze_2906, unsqueeze_2918, unsqueeze_2930, unsqueeze_2942, unsqueeze_2954, unsqueeze_2966, unsqueeze_2978, unsqueeze_2990, unsqueeze_3002, unsqueeze_3014, unsqueeze_3026, le_186, unsqueeze_3038, unsqueeze_3050, unsqueeze_3062, unsqueeze_3074, unsqueeze_3086, unsqueeze_3098, unsqueeze_3110, unsqueeze_3122, unsqueeze_3134, unsqueeze_3146, unsqueeze_3158, unsqueeze_3170, unsqueeze_3182, unsqueeze_3194, unsqueeze_3206]
    