from __future__ import annotations



def forward(self, primals_2: "f32[768]", primals_3: "f32[768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[768]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[768]", primals_27: "f32[768]", primals_28: "f32[768]", primals_29: "f32[768]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[768]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[768]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[768]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[768]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[768]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[768]", primals_78: "f32[768]", primals_79: "f32[768, 3, 16, 16]", primals_81: "f32[768]", primals_91: "f32[768]", primals_97: "f32[768]", primals_107: "f32[768]", primals_113: "f32[768]", primals_123: "f32[768]", primals_129: "f32[768]", primals_139: "f32[768]", primals_145: "f32[768]", primals_155: "f32[768]", primals_161: "f32[768]", primals_171: "f32[768]", primals_177: "f32[768]", primals_187: "f32[768]", primals_193: "f32[768]", primals_203: "f32[768]", primals_209: "f32[768]", primals_219: "f32[768]", primals_225: "f32[768]", primals_235: "f32[768]", primals_241: "f32[768]", primals_251: "f32[768]", primals_257: "f32[768]", primals_267: "f32[768]", primals_273: "f32[768]", primals_283: "f32[768]", primals_289: "f32[768]", primals_299: "f32[768]", primals_305: "f32[768]", primals_315: "f32[768]", primals_321: "f32[768]", primals_331: "f32[768]", primals_337: "f32[768]", primals_347: "f32[768]", primals_353: "f32[768]", primals_363: "f32[768]", primals_369: "f32[768]", primals_379: "f32[768]", primals_385: "f32[768]", primals_395: "f32[768]", primals_401: "f32[768]", primals_411: "f32[768]", primals_417: "f32[768]", primals_427: "f32[768]", primals_433: "f32[768]", primals_443: "f32[768]", primals_449: "f32[768]", primals_459: "f32[768]", primals_465: "f32[768]", primals_475: "f32[768]", primals_481: "f32[768]", primals_491: "f32[768]", primals_497: "f32[768]", primals_507: "f32[768]", primals_513: "f32[768]", primals_523: "f32[768]", primals_529: "f32[768]", primals_539: "f32[768]", primals_545: "f32[768]", primals_555: "f32[768]", primals_561: "f32[768]", primals_571: "f32[768]", primals_577: "f32[768]", primals_587: "f32[768]", primals_593: "f32[768]", primals_603: "f32[768]", primals_609: "f32[768]", primals_619: "f32[768]", primals_625: "f32[768]", primals_635: "f32[768]", primals_641: "f32[768]", primals_651: "f32[768]", primals_657: "f32[768]", primals_667: "f32[768]", primals_673: "f32[768]", primals_683: "f32[768]", primals_689: "f32[768]", primals_693: "f32[8, 3, 384, 384]", mul: "f32[8, 576, 768]", view_1: "f32[4608, 768]", view_7: "f32[2654208, 16]", view_9: "f32[2654208, 16]", view_15: "f32[4608, 768]", addmm_1: "f32[4608, 768]", mul_4: "f32[8, 576, 768]", view_17: "f32[4608, 768]", addmm_2: "f32[4608, 3072]", view_19: "f32[4608, 3072]", addmm_3: "f32[4608, 768]", mul_10: "f32[8, 576, 768]", view_21: "f32[4608, 768]", view_27: "f32[2654208, 16]", view_29: "f32[2654208, 16]", view_35: "f32[4608, 768]", addmm_5: "f32[4608, 768]", mul_14: "f32[8, 576, 768]", view_37: "f32[4608, 768]", addmm_6: "f32[4608, 3072]", view_39: "f32[4608, 3072]", addmm_7: "f32[4608, 768]", mul_20: "f32[8, 576, 768]", view_41: "f32[4608, 768]", view_47: "f32[2654208, 16]", view_49: "f32[2654208, 16]", view_55: "f32[4608, 768]", addmm_9: "f32[4608, 768]", mul_24: "f32[8, 576, 768]", view_57: "f32[4608, 768]", addmm_10: "f32[4608, 3072]", view_59: "f32[4608, 3072]", addmm_11: "f32[4608, 768]", mul_30: "f32[8, 576, 768]", view_61: "f32[4608, 768]", view_67: "f32[2654208, 16]", view_69: "f32[2654208, 16]", view_75: "f32[4608, 768]", addmm_13: "f32[4608, 768]", mul_34: "f32[8, 576, 768]", view_77: "f32[4608, 768]", addmm_14: "f32[4608, 3072]", view_79: "f32[4608, 3072]", addmm_15: "f32[4608, 768]", mul_40: "f32[8, 576, 768]", view_81: "f32[4608, 768]", view_87: "f32[2654208, 16]", view_89: "f32[2654208, 16]", view_95: "f32[4608, 768]", addmm_17: "f32[4608, 768]", mul_44: "f32[8, 576, 768]", view_97: "f32[4608, 768]", addmm_18: "f32[4608, 3072]", view_99: "f32[4608, 3072]", addmm_19: "f32[4608, 768]", mul_50: "f32[8, 576, 768]", view_101: "f32[4608, 768]", view_107: "f32[2654208, 16]", view_109: "f32[2654208, 16]", view_115: "f32[4608, 768]", addmm_21: "f32[4608, 768]", mul_54: "f32[8, 576, 768]", view_117: "f32[4608, 768]", addmm_22: "f32[4608, 3072]", view_119: "f32[4608, 3072]", addmm_23: "f32[4608, 768]", mul_60: "f32[8, 576, 768]", view_121: "f32[4608, 768]", view_127: "f32[2654208, 16]", view_129: "f32[2654208, 16]", view_135: "f32[4608, 768]", addmm_25: "f32[4608, 768]", mul_64: "f32[8, 576, 768]", view_137: "f32[4608, 768]", addmm_26: "f32[4608, 3072]", view_139: "f32[4608, 3072]", addmm_27: "f32[4608, 768]", mul_70: "f32[8, 576, 768]", view_141: "f32[4608, 768]", view_147: "f32[2654208, 16]", view_149: "f32[2654208, 16]", view_155: "f32[4608, 768]", addmm_29: "f32[4608, 768]", mul_74: "f32[8, 576, 768]", view_157: "f32[4608, 768]", addmm_30: "f32[4608, 3072]", view_159: "f32[4608, 3072]", addmm_31: "f32[4608, 768]", mul_80: "f32[8, 576, 768]", view_161: "f32[4608, 768]", view_167: "f32[2654208, 16]", view_169: "f32[2654208, 16]", view_175: "f32[4608, 768]", addmm_33: "f32[4608, 768]", mul_84: "f32[8, 576, 768]", view_177: "f32[4608, 768]", addmm_34: "f32[4608, 3072]", view_179: "f32[4608, 3072]", addmm_35: "f32[4608, 768]", mul_90: "f32[8, 576, 768]", view_181: "f32[4608, 768]", view_187: "f32[2654208, 16]", view_189: "f32[2654208, 16]", view_195: "f32[4608, 768]", addmm_37: "f32[4608, 768]", mul_94: "f32[8, 576, 768]", view_197: "f32[4608, 768]", addmm_38: "f32[4608, 3072]", view_199: "f32[4608, 3072]", addmm_39: "f32[4608, 768]", mul_100: "f32[8, 576, 768]", view_201: "f32[4608, 768]", view_207: "f32[2654208, 16]", view_209: "f32[2654208, 16]", view_215: "f32[4608, 768]", addmm_41: "f32[4608, 768]", mul_104: "f32[8, 576, 768]", view_217: "f32[4608, 768]", addmm_42: "f32[4608, 3072]", view_219: "f32[4608, 3072]", addmm_43: "f32[4608, 768]", mul_110: "f32[8, 576, 768]", view_221: "f32[4608, 768]", view_227: "f32[2654208, 16]", view_229: "f32[2654208, 16]", view_235: "f32[4608, 768]", addmm_45: "f32[4608, 768]", mul_114: "f32[8, 576, 768]", view_237: "f32[4608, 768]", addmm_46: "f32[4608, 3072]", view_239: "f32[4608, 3072]", addmm_47: "f32[4608, 768]", mul_120: "f32[8, 576, 768]", view_241: "f32[4608, 768]", view_247: "f32[2654208, 16]", view_249: "f32[2654208, 16]", view_255: "f32[4608, 768]", addmm_49: "f32[4608, 768]", mul_124: "f32[8, 576, 768]", view_257: "f32[4608, 768]", addmm_50: "f32[4608, 3072]", view_259: "f32[4608, 3072]", addmm_51: "f32[4608, 768]", mul_130: "f32[8, 576, 768]", view_261: "f32[4608, 768]", view_267: "f32[2654208, 16]", view_269: "f32[2654208, 16]", view_275: "f32[4608, 768]", addmm_53: "f32[4608, 768]", mul_134: "f32[8, 576, 768]", view_277: "f32[4608, 768]", addmm_54: "f32[4608, 3072]", view_279: "f32[4608, 3072]", addmm_55: "f32[4608, 768]", mul_140: "f32[8, 576, 768]", view_281: "f32[4608, 768]", view_287: "f32[2654208, 16]", view_289: "f32[2654208, 16]", view_295: "f32[4608, 768]", addmm_57: "f32[4608, 768]", mul_144: "f32[8, 576, 768]", view_297: "f32[4608, 768]", addmm_58: "f32[4608, 3072]", view_299: "f32[4608, 3072]", addmm_59: "f32[4608, 768]", mul_150: "f32[8, 576, 768]", view_301: "f32[4608, 768]", view_307: "f32[2654208, 16]", view_309: "f32[2654208, 16]", view_315: "f32[4608, 768]", addmm_61: "f32[4608, 768]", mul_154: "f32[8, 576, 768]", view_317: "f32[4608, 768]", addmm_62: "f32[4608, 3072]", view_319: "f32[4608, 3072]", addmm_63: "f32[4608, 768]", mul_160: "f32[8, 576, 768]", view_321: "f32[4608, 768]", view_327: "f32[2654208, 16]", view_329: "f32[2654208, 16]", view_335: "f32[4608, 768]", addmm_65: "f32[4608, 768]", mul_164: "f32[8, 576, 768]", view_337: "f32[4608, 768]", addmm_66: "f32[4608, 3072]", view_339: "f32[4608, 3072]", addmm_67: "f32[4608, 768]", mul_170: "f32[8, 576, 768]", view_341: "f32[4608, 768]", view_347: "f32[2654208, 16]", view_349: "f32[2654208, 16]", view_355: "f32[4608, 768]", addmm_69: "f32[4608, 768]", mul_174: "f32[8, 576, 768]", view_357: "f32[4608, 768]", addmm_70: "f32[4608, 3072]", view_359: "f32[4608, 3072]", addmm_71: "f32[4608, 768]", mul_180: "f32[8, 576, 768]", view_361: "f32[4608, 768]", view_367: "f32[2654208, 16]", view_369: "f32[2654208, 16]", view_375: "f32[4608, 768]", addmm_73: "f32[4608, 768]", mul_184: "f32[8, 576, 768]", view_377: "f32[4608, 768]", addmm_74: "f32[4608, 3072]", view_379: "f32[4608, 3072]", addmm_75: "f32[4608, 768]", mul_190: "f32[8, 576, 768]", view_381: "f32[4608, 768]", view_387: "f32[2654208, 16]", view_389: "f32[2654208, 16]", view_395: "f32[4608, 768]", addmm_77: "f32[4608, 768]", mul_194: "f32[8, 576, 768]", view_397: "f32[4608, 768]", addmm_78: "f32[4608, 3072]", view_399: "f32[4608, 3072]", addmm_79: "f32[4608, 768]", mul_200: "f32[8, 576, 768]", view_401: "f32[4608, 768]", view_407: "f32[2654208, 16]", view_409: "f32[2654208, 16]", view_415: "f32[4608, 768]", addmm_81: "f32[4608, 768]", mul_204: "f32[8, 576, 768]", view_417: "f32[4608, 768]", addmm_82: "f32[4608, 3072]", view_419: "f32[4608, 3072]", addmm_83: "f32[4608, 768]", mul_210: "f32[8, 576, 768]", view_421: "f32[4608, 768]", view_427: "f32[2654208, 16]", view_429: "f32[2654208, 16]", view_435: "f32[4608, 768]", addmm_85: "f32[4608, 768]", mul_214: "f32[8, 576, 768]", view_437: "f32[4608, 768]", addmm_86: "f32[4608, 3072]", view_439: "f32[4608, 3072]", addmm_87: "f32[4608, 768]", mul_220: "f32[8, 576, 768]", view_441: "f32[4608, 768]", view_447: "f32[2654208, 16]", view_449: "f32[2654208, 16]", view_455: "f32[4608, 768]", addmm_89: "f32[4608, 768]", mul_224: "f32[8, 576, 768]", view_457: "f32[4608, 768]", addmm_90: "f32[4608, 3072]", view_459: "f32[4608, 3072]", addmm_91: "f32[4608, 768]", mul_230: "f32[8, 576, 768]", view_461: "f32[4608, 768]", view_467: "f32[2654208, 16]", view_469: "f32[2654208, 16]", view_475: "f32[4608, 768]", addmm_93: "f32[4608, 768]", mul_234: "f32[8, 576, 768]", view_477: "f32[4608, 768]", addmm_94: "f32[4608, 3072]", view_479: "f32[4608, 3072]", addmm_95: "f32[4608, 768]", mul_240: "f32[8, 576, 768]", view_481: "f32[4608, 768]", view_487: "f32[2654208, 16]", view_489: "f32[2654208, 16]", view_495: "f32[4608, 768]", addmm_97: "f32[4608, 768]", mul_244: "f32[8, 576, 768]", view_497: "f32[4608, 768]", addmm_98: "f32[4608, 3072]", view_499: "f32[4608, 3072]", addmm_99: "f32[4608, 768]", mul_250: "f32[8, 576, 768]", view_501: "f32[4608, 768]", view_507: "f32[2654208, 16]", view_509: "f32[2654208, 16]", view_515: "f32[4608, 768]", addmm_101: "f32[4608, 768]", mul_254: "f32[8, 576, 768]", view_517: "f32[4608, 768]", addmm_102: "f32[4608, 3072]", view_519: "f32[4608, 3072]", addmm_103: "f32[4608, 768]", mul_260: "f32[8, 576, 768]", view_521: "f32[4608, 768]", view_527: "f32[2654208, 16]", view_529: "f32[2654208, 16]", view_535: "f32[4608, 768]", addmm_105: "f32[4608, 768]", mul_264: "f32[8, 576, 768]", view_537: "f32[4608, 768]", addmm_106: "f32[4608, 3072]", view_539: "f32[4608, 3072]", addmm_107: "f32[4608, 768]", mul_270: "f32[8, 576, 768]", view_541: "f32[4608, 768]", view_547: "f32[2654208, 16]", view_549: "f32[2654208, 16]", view_555: "f32[4608, 768]", addmm_109: "f32[4608, 768]", mul_274: "f32[8, 576, 768]", view_557: "f32[4608, 768]", addmm_110: "f32[4608, 3072]", view_559: "f32[4608, 3072]", addmm_111: "f32[4608, 768]", mul_280: "f32[8, 576, 768]", view_561: "f32[4608, 768]", view_567: "f32[2654208, 16]", view_569: "f32[2654208, 16]", view_575: "f32[4608, 768]", addmm_113: "f32[4608, 768]", mul_284: "f32[8, 576, 768]", view_577: "f32[4608, 768]", addmm_114: "f32[4608, 3072]", view_579: "f32[4608, 3072]", addmm_115: "f32[4608, 768]", mul_290: "f32[8, 576, 768]", view_581: "f32[4608, 768]", view_587: "f32[2654208, 16]", view_589: "f32[2654208, 16]", view_595: "f32[4608, 768]", addmm_117: "f32[4608, 768]", mul_294: "f32[8, 576, 768]", view_597: "f32[4608, 768]", addmm_118: "f32[4608, 3072]", view_599: "f32[4608, 3072]", addmm_119: "f32[4608, 768]", mul_300: "f32[8, 576, 768]", view_601: "f32[4608, 768]", view_607: "f32[2654208, 16]", view_609: "f32[2654208, 16]", view_615: "f32[4608, 768]", addmm_121: "f32[4608, 768]", mul_304: "f32[8, 576, 768]", view_617: "f32[4608, 768]", addmm_122: "f32[4608, 3072]", view_619: "f32[4608, 3072]", addmm_123: "f32[4608, 768]", mul_310: "f32[8, 576, 768]", view_621: "f32[4608, 768]", view_627: "f32[2654208, 16]", view_629: "f32[2654208, 16]", view_635: "f32[4608, 768]", addmm_125: "f32[4608, 768]", mul_314: "f32[8, 576, 768]", view_637: "f32[4608, 768]", addmm_126: "f32[4608, 3072]", view_639: "f32[4608, 3072]", addmm_127: "f32[4608, 768]", mul_320: "f32[8, 576, 768]", view_641: "f32[4608, 768]", view_647: "f32[2654208, 16]", view_649: "f32[2654208, 16]", view_655: "f32[4608, 768]", addmm_129: "f32[4608, 768]", mul_324: "f32[8, 576, 768]", view_657: "f32[4608, 768]", addmm_130: "f32[4608, 3072]", view_659: "f32[4608, 3072]", addmm_131: "f32[4608, 768]", mul_330: "f32[8, 576, 768]", view_661: "f32[4608, 768]", view_667: "f32[2654208, 16]", view_669: "f32[2654208, 16]", view_675: "f32[4608, 768]", addmm_133: "f32[4608, 768]", mul_334: "f32[8, 576, 768]", view_677: "f32[4608, 768]", addmm_134: "f32[4608, 3072]", view_679: "f32[4608, 3072]", addmm_135: "f32[4608, 768]", mul_340: "f32[8, 576, 768]", view_681: "f32[4608, 768]", view_687: "f32[2654208, 16]", view_689: "f32[2654208, 16]", view_695: "f32[4608, 768]", addmm_137: "f32[4608, 768]", mul_344: "f32[8, 576, 768]", view_697: "f32[4608, 768]", addmm_138: "f32[4608, 3072]", view_699: "f32[4608, 3072]", addmm_139: "f32[4608, 768]", mul_350: "f32[8, 576, 768]", view_701: "f32[4608, 768]", view_707: "f32[2654208, 16]", view_709: "f32[2654208, 16]", view_715: "f32[4608, 768]", addmm_141: "f32[4608, 768]", mul_354: "f32[8, 576, 768]", view_717: "f32[4608, 768]", addmm_142: "f32[4608, 3072]", view_719: "f32[4608, 3072]", addmm_143: "f32[4608, 768]", cat: "f32[8, 577, 768]", getitem_145: "f32[8, 577, 1]", rsqrt_72: "f32[8, 577, 1]", select_108: "f32[8, 768]", permute_470: "f32[8, 16, 1, 48]", view_722: "f32[4616, 768]", permute_472: "f32[8, 16, 577, 48]", permute_474: "f32[8, 16, 577, 48]", getitem_147: "f32[8, 16, 1]", getitem_148: "i32[]", getitem_149: "i32[]", getitem_152: "i64[]", getitem_153: "i64[]", view_729: "f32[8, 768]", addmm_147: "f32[8, 768]", mul_363: "f32[8, 1, 768]", view_731: "f32[8, 768]", addmm_148: "f32[8, 3072]", view_733: "f32[8, 3072]", addmm_149: "f32[8, 768]", cat_1: "f32[8, 577, 768]", getitem_158: "f32[8, 577, 1]", rsqrt_74: "f32[8, 577, 1]", select_109: "f32[8, 768]", permute_480: "f32[8, 16, 1, 48]", view_736: "f32[4616, 768]", permute_482: "f32[8, 16, 577, 48]", permute_484: "f32[8, 16, 577, 48]", getitem_160: "f32[8, 16, 1]", getitem_161: "i32[]", getitem_162: "i32[]", getitem_165: "i64[]", getitem_166: "i64[]", view_743: "f32[8, 768]", addmm_153: "f32[8, 768]", mul_372: "f32[8, 1, 768]", view_745: "f32[8, 768]", addmm_154: "f32[8, 3072]", view_747: "f32[8, 3072]", addmm_155: "f32[8, 768]", cat_2: "f32[8, 577, 768]", getitem_171: "f32[8, 577, 1]", rsqrt_76: "f32[8, 577, 1]", clone_511: "f32[8, 768]", permute_490: "f32[1000, 768]", permute_494: "f32[768, 3072]", permute_498: "f32[3072, 768]", div_37: "f32[8, 1, 1]", permute_502: "f32[768, 768]", alias_38: "f32[8, 16, 1, 48]", permute_508: "f32[768, 768]", permute_513: "f32[768, 768]", permute_518: "f32[768, 768]", permute_522: "f32[768, 3072]", permute_526: "f32[3072, 768]", div_39: "f32[8, 1, 1]", permute_530: "f32[768, 768]", alias_39: "f32[8, 16, 1, 48]", permute_536: "f32[768, 768]", permute_541: "f32[768, 768]", permute_546: "f32[768, 768]", permute_550: "f32[768, 3072]", permute_554: "f32[3072, 768]", div_41: "f32[8, 576, 1]", permute_558: "f32[768, 768]", permute_563: "f32[128, 576, 576]", permute_564: "f32[128, 48, 576]", permute_568: "f32[16, 16]", alias_40: "f32[8, 16, 576, 576]", permute_574: "f32[16, 16]", permute_577: "f32[128, 48, 576]", permute_578: "f32[128, 576, 48]", permute_581: "f32[2304, 768]", div_42: "f32[8, 576, 1]", permute_585: "f32[768, 3072]", permute_589: "f32[3072, 768]", div_43: "f32[8, 576, 1]", permute_593: "f32[768, 768]", permute_598: "f32[128, 576, 576]", permute_599: "f32[128, 48, 576]", permute_603: "f32[16, 16]", alias_41: "f32[8, 16, 576, 576]", permute_609: "f32[16, 16]", permute_612: "f32[128, 48, 576]", permute_613: "f32[128, 576, 48]", permute_616: "f32[2304, 768]", div_44: "f32[8, 576, 1]", permute_620: "f32[768, 3072]", permute_624: "f32[3072, 768]", div_45: "f32[8, 576, 1]", permute_628: "f32[768, 768]", permute_633: "f32[128, 576, 576]", permute_634: "f32[128, 48, 576]", permute_638: "f32[16, 16]", alias_42: "f32[8, 16, 576, 576]", permute_644: "f32[16, 16]", permute_647: "f32[128, 48, 576]", permute_648: "f32[128, 576, 48]", permute_651: "f32[2304, 768]", div_46: "f32[8, 576, 1]", permute_655: "f32[768, 3072]", permute_659: "f32[3072, 768]", div_47: "f32[8, 576, 1]", permute_663: "f32[768, 768]", permute_668: "f32[128, 576, 576]", permute_669: "f32[128, 48, 576]", permute_673: "f32[16, 16]", alias_43: "f32[8, 16, 576, 576]", permute_679: "f32[16, 16]", permute_682: "f32[128, 48, 576]", permute_683: "f32[128, 576, 48]", permute_686: "f32[2304, 768]", div_48: "f32[8, 576, 1]", permute_690: "f32[768, 3072]", permute_694: "f32[3072, 768]", div_49: "f32[8, 576, 1]", permute_698: "f32[768, 768]", permute_703: "f32[128, 576, 576]", permute_704: "f32[128, 48, 576]", permute_708: "f32[16, 16]", alias_44: "f32[8, 16, 576, 576]", permute_714: "f32[16, 16]", permute_717: "f32[128, 48, 576]", permute_718: "f32[128, 576, 48]", permute_721: "f32[2304, 768]", div_50: "f32[8, 576, 1]", permute_725: "f32[768, 3072]", permute_729: "f32[3072, 768]", div_51: "f32[8, 576, 1]", permute_733: "f32[768, 768]", permute_738: "f32[128, 576, 576]", permute_739: "f32[128, 48, 576]", permute_743: "f32[16, 16]", alias_45: "f32[8, 16, 576, 576]", permute_749: "f32[16, 16]", permute_752: "f32[128, 48, 576]", permute_753: "f32[128, 576, 48]", permute_756: "f32[2304, 768]", div_52: "f32[8, 576, 1]", permute_760: "f32[768, 3072]", permute_764: "f32[3072, 768]", div_53: "f32[8, 576, 1]", permute_768: "f32[768, 768]", permute_773: "f32[128, 576, 576]", permute_774: "f32[128, 48, 576]", permute_778: "f32[16, 16]", alias_46: "f32[8, 16, 576, 576]", permute_784: "f32[16, 16]", permute_787: "f32[128, 48, 576]", permute_788: "f32[128, 576, 48]", permute_791: "f32[2304, 768]", div_54: "f32[8, 576, 1]", permute_795: "f32[768, 3072]", permute_799: "f32[3072, 768]", div_55: "f32[8, 576, 1]", permute_803: "f32[768, 768]", permute_808: "f32[128, 576, 576]", permute_809: "f32[128, 48, 576]", permute_813: "f32[16, 16]", alias_47: "f32[8, 16, 576, 576]", permute_819: "f32[16, 16]", permute_822: "f32[128, 48, 576]", permute_823: "f32[128, 576, 48]", permute_826: "f32[2304, 768]", div_56: "f32[8, 576, 1]", permute_830: "f32[768, 3072]", permute_834: "f32[3072, 768]", div_57: "f32[8, 576, 1]", permute_838: "f32[768, 768]", permute_843: "f32[128, 576, 576]", permute_844: "f32[128, 48, 576]", permute_848: "f32[16, 16]", alias_48: "f32[8, 16, 576, 576]", permute_854: "f32[16, 16]", permute_857: "f32[128, 48, 576]", permute_858: "f32[128, 576, 48]", permute_861: "f32[2304, 768]", div_58: "f32[8, 576, 1]", permute_865: "f32[768, 3072]", permute_869: "f32[3072, 768]", div_59: "f32[8, 576, 1]", permute_873: "f32[768, 768]", permute_878: "f32[128, 576, 576]", permute_879: "f32[128, 48, 576]", permute_883: "f32[16, 16]", alias_49: "f32[8, 16, 576, 576]", permute_889: "f32[16, 16]", permute_892: "f32[128, 48, 576]", permute_893: "f32[128, 576, 48]", permute_896: "f32[2304, 768]", div_60: "f32[8, 576, 1]", permute_900: "f32[768, 3072]", permute_904: "f32[3072, 768]", div_61: "f32[8, 576, 1]", permute_908: "f32[768, 768]", permute_913: "f32[128, 576, 576]", permute_914: "f32[128, 48, 576]", permute_918: "f32[16, 16]", alias_50: "f32[8, 16, 576, 576]", permute_924: "f32[16, 16]", permute_927: "f32[128, 48, 576]", permute_928: "f32[128, 576, 48]", permute_931: "f32[2304, 768]", div_62: "f32[8, 576, 1]", permute_935: "f32[768, 3072]", permute_939: "f32[3072, 768]", div_63: "f32[8, 576, 1]", permute_943: "f32[768, 768]", permute_948: "f32[128, 576, 576]", permute_949: "f32[128, 48, 576]", permute_953: "f32[16, 16]", alias_51: "f32[8, 16, 576, 576]", permute_959: "f32[16, 16]", permute_962: "f32[128, 48, 576]", permute_963: "f32[128, 576, 48]", permute_966: "f32[2304, 768]", div_64: "f32[8, 576, 1]", permute_970: "f32[768, 3072]", permute_974: "f32[3072, 768]", div_65: "f32[8, 576, 1]", permute_978: "f32[768, 768]", permute_983: "f32[128, 576, 576]", permute_984: "f32[128, 48, 576]", permute_988: "f32[16, 16]", alias_52: "f32[8, 16, 576, 576]", permute_994: "f32[16, 16]", permute_997: "f32[128, 48, 576]", permute_998: "f32[128, 576, 48]", permute_1001: "f32[2304, 768]", div_66: "f32[8, 576, 1]", permute_1005: "f32[768, 3072]", permute_1009: "f32[3072, 768]", div_67: "f32[8, 576, 1]", permute_1013: "f32[768, 768]", permute_1018: "f32[128, 576, 576]", permute_1019: "f32[128, 48, 576]", permute_1023: "f32[16, 16]", alias_53: "f32[8, 16, 576, 576]", permute_1029: "f32[16, 16]", permute_1032: "f32[128, 48, 576]", permute_1033: "f32[128, 576, 48]", permute_1036: "f32[2304, 768]", div_68: "f32[8, 576, 1]", permute_1040: "f32[768, 3072]", permute_1044: "f32[3072, 768]", div_69: "f32[8, 576, 1]", permute_1048: "f32[768, 768]", permute_1053: "f32[128, 576, 576]", permute_1054: "f32[128, 48, 576]", permute_1058: "f32[16, 16]", alias_54: "f32[8, 16, 576, 576]", permute_1064: "f32[16, 16]", permute_1067: "f32[128, 48, 576]", permute_1068: "f32[128, 576, 48]", permute_1071: "f32[2304, 768]", div_70: "f32[8, 576, 1]", permute_1075: "f32[768, 3072]", permute_1079: "f32[3072, 768]", div_71: "f32[8, 576, 1]", permute_1083: "f32[768, 768]", permute_1088: "f32[128, 576, 576]", permute_1089: "f32[128, 48, 576]", permute_1093: "f32[16, 16]", alias_55: "f32[8, 16, 576, 576]", permute_1099: "f32[16, 16]", permute_1102: "f32[128, 48, 576]", permute_1103: "f32[128, 576, 48]", permute_1106: "f32[2304, 768]", div_72: "f32[8, 576, 1]", permute_1110: "f32[768, 3072]", permute_1114: "f32[3072, 768]", div_73: "f32[8, 576, 1]", permute_1118: "f32[768, 768]", permute_1123: "f32[128, 576, 576]", permute_1124: "f32[128, 48, 576]", permute_1128: "f32[16, 16]", alias_56: "f32[8, 16, 576, 576]", permute_1134: "f32[16, 16]", permute_1137: "f32[128, 48, 576]", permute_1138: "f32[128, 576, 48]", permute_1141: "f32[2304, 768]", div_74: "f32[8, 576, 1]", permute_1145: "f32[768, 3072]", permute_1149: "f32[3072, 768]", div_75: "f32[8, 576, 1]", permute_1153: "f32[768, 768]", permute_1158: "f32[128, 576, 576]", permute_1159: "f32[128, 48, 576]", permute_1163: "f32[16, 16]", alias_57: "f32[8, 16, 576, 576]", permute_1169: "f32[16, 16]", permute_1172: "f32[128, 48, 576]", permute_1173: "f32[128, 576, 48]", permute_1176: "f32[2304, 768]", div_76: "f32[8, 576, 1]", permute_1180: "f32[768, 3072]", permute_1184: "f32[3072, 768]", div_77: "f32[8, 576, 1]", permute_1188: "f32[768, 768]", permute_1193: "f32[128, 576, 576]", permute_1194: "f32[128, 48, 576]", permute_1198: "f32[16, 16]", alias_58: "f32[8, 16, 576, 576]", permute_1204: "f32[16, 16]", permute_1207: "f32[128, 48, 576]", permute_1208: "f32[128, 576, 48]", permute_1211: "f32[2304, 768]", div_78: "f32[8, 576, 1]", permute_1215: "f32[768, 3072]", permute_1219: "f32[3072, 768]", div_79: "f32[8, 576, 1]", permute_1223: "f32[768, 768]", permute_1228: "f32[128, 576, 576]", permute_1229: "f32[128, 48, 576]", permute_1233: "f32[16, 16]", alias_59: "f32[8, 16, 576, 576]", permute_1239: "f32[16, 16]", permute_1242: "f32[128, 48, 576]", permute_1243: "f32[128, 576, 48]", permute_1246: "f32[2304, 768]", div_80: "f32[8, 576, 1]", permute_1250: "f32[768, 3072]", permute_1254: "f32[3072, 768]", div_81: "f32[8, 576, 1]", permute_1258: "f32[768, 768]", permute_1263: "f32[128, 576, 576]", permute_1264: "f32[128, 48, 576]", permute_1268: "f32[16, 16]", alias_60: "f32[8, 16, 576, 576]", permute_1274: "f32[16, 16]", permute_1277: "f32[128, 48, 576]", permute_1278: "f32[128, 576, 48]", permute_1281: "f32[2304, 768]", div_82: "f32[8, 576, 1]", permute_1285: "f32[768, 3072]", permute_1289: "f32[3072, 768]", div_83: "f32[8, 576, 1]", permute_1293: "f32[768, 768]", permute_1298: "f32[128, 576, 576]", permute_1299: "f32[128, 48, 576]", permute_1303: "f32[16, 16]", alias_61: "f32[8, 16, 576, 576]", permute_1309: "f32[16, 16]", permute_1312: "f32[128, 48, 576]", permute_1313: "f32[128, 576, 48]", permute_1316: "f32[2304, 768]", div_84: "f32[8, 576, 1]", permute_1320: "f32[768, 3072]", permute_1324: "f32[3072, 768]", div_85: "f32[8, 576, 1]", permute_1328: "f32[768, 768]", permute_1333: "f32[128, 576, 576]", permute_1334: "f32[128, 48, 576]", permute_1338: "f32[16, 16]", alias_62: "f32[8, 16, 576, 576]", permute_1344: "f32[16, 16]", permute_1347: "f32[128, 48, 576]", permute_1348: "f32[128, 576, 48]", permute_1351: "f32[2304, 768]", div_86: "f32[8, 576, 1]", permute_1355: "f32[768, 3072]", permute_1359: "f32[3072, 768]", div_87: "f32[8, 576, 1]", permute_1363: "f32[768, 768]", permute_1368: "f32[128, 576, 576]", permute_1369: "f32[128, 48, 576]", permute_1373: "f32[16, 16]", alias_63: "f32[8, 16, 576, 576]", permute_1379: "f32[16, 16]", permute_1382: "f32[128, 48, 576]", permute_1383: "f32[128, 576, 48]", permute_1386: "f32[2304, 768]", div_88: "f32[8, 576, 1]", permute_1390: "f32[768, 3072]", permute_1394: "f32[3072, 768]", div_89: "f32[8, 576, 1]", permute_1398: "f32[768, 768]", permute_1403: "f32[128, 576, 576]", permute_1404: "f32[128, 48, 576]", permute_1408: "f32[16, 16]", alias_64: "f32[8, 16, 576, 576]", permute_1414: "f32[16, 16]", permute_1417: "f32[128, 48, 576]", permute_1418: "f32[128, 576, 48]", permute_1421: "f32[2304, 768]", div_90: "f32[8, 576, 1]", permute_1425: "f32[768, 3072]", permute_1429: "f32[3072, 768]", div_91: "f32[8, 576, 1]", permute_1433: "f32[768, 768]", permute_1438: "f32[128, 576, 576]", permute_1439: "f32[128, 48, 576]", permute_1443: "f32[16, 16]", alias_65: "f32[8, 16, 576, 576]", permute_1449: "f32[16, 16]", permute_1452: "f32[128, 48, 576]", permute_1453: "f32[128, 576, 48]", permute_1456: "f32[2304, 768]", div_92: "f32[8, 576, 1]", permute_1460: "f32[768, 3072]", permute_1464: "f32[3072, 768]", div_93: "f32[8, 576, 1]", permute_1468: "f32[768, 768]", permute_1473: "f32[128, 576, 576]", permute_1474: "f32[128, 48, 576]", permute_1478: "f32[16, 16]", alias_66: "f32[8, 16, 576, 576]", permute_1484: "f32[16, 16]", permute_1487: "f32[128, 48, 576]", permute_1488: "f32[128, 576, 48]", permute_1491: "f32[2304, 768]", div_94: "f32[8, 576, 1]", permute_1495: "f32[768, 3072]", permute_1499: "f32[3072, 768]", div_95: "f32[8, 576, 1]", permute_1503: "f32[768, 768]", permute_1508: "f32[128, 576, 576]", permute_1509: "f32[128, 48, 576]", permute_1513: "f32[16, 16]", alias_67: "f32[8, 16, 576, 576]", permute_1519: "f32[16, 16]", permute_1522: "f32[128, 48, 576]", permute_1523: "f32[128, 576, 48]", permute_1526: "f32[2304, 768]", div_96: "f32[8, 576, 1]", permute_1530: "f32[768, 3072]", permute_1534: "f32[3072, 768]", div_97: "f32[8, 576, 1]", permute_1538: "f32[768, 768]", permute_1543: "f32[128, 576, 576]", permute_1544: "f32[128, 48, 576]", permute_1548: "f32[16, 16]", alias_68: "f32[8, 16, 576, 576]", permute_1554: "f32[16, 16]", permute_1557: "f32[128, 48, 576]", permute_1558: "f32[128, 576, 48]", permute_1561: "f32[2304, 768]", div_98: "f32[8, 576, 1]", permute_1565: "f32[768, 3072]", permute_1569: "f32[3072, 768]", div_99: "f32[8, 576, 1]", permute_1573: "f32[768, 768]", permute_1578: "f32[128, 576, 576]", permute_1579: "f32[128, 48, 576]", permute_1583: "f32[16, 16]", alias_69: "f32[8, 16, 576, 576]", permute_1589: "f32[16, 16]", permute_1592: "f32[128, 48, 576]", permute_1593: "f32[128, 576, 48]", permute_1596: "f32[2304, 768]", div_100: "f32[8, 576, 1]", permute_1600: "f32[768, 3072]", permute_1604: "f32[3072, 768]", div_101: "f32[8, 576, 1]", permute_1608: "f32[768, 768]", permute_1613: "f32[128, 576, 576]", permute_1614: "f32[128, 48, 576]", permute_1618: "f32[16, 16]", alias_70: "f32[8, 16, 576, 576]", permute_1624: "f32[16, 16]", permute_1627: "f32[128, 48, 576]", permute_1628: "f32[128, 576, 48]", permute_1631: "f32[2304, 768]", div_102: "f32[8, 576, 1]", permute_1635: "f32[768, 3072]", permute_1639: "f32[3072, 768]", div_103: "f32[8, 576, 1]", permute_1643: "f32[768, 768]", permute_1648: "f32[128, 576, 576]", permute_1649: "f32[128, 48, 576]", permute_1653: "f32[16, 16]", alias_71: "f32[8, 16, 576, 576]", permute_1659: "f32[16, 16]", permute_1662: "f32[128, 48, 576]", permute_1663: "f32[128, 576, 48]", permute_1666: "f32[2304, 768]", div_104: "f32[8, 576, 1]", permute_1670: "f32[768, 3072]", permute_1674: "f32[3072, 768]", div_105: "f32[8, 576, 1]", permute_1678: "f32[768, 768]", permute_1683: "f32[128, 576, 576]", permute_1684: "f32[128, 48, 576]", permute_1688: "f32[16, 16]", alias_72: "f32[8, 16, 576, 576]", permute_1694: "f32[16, 16]", permute_1697: "f32[128, 48, 576]", permute_1698: "f32[128, 576, 48]", permute_1701: "f32[2304, 768]", div_106: "f32[8, 576, 1]", permute_1705: "f32[768, 3072]", permute_1709: "f32[3072, 768]", div_107: "f32[8, 576, 1]", permute_1713: "f32[768, 768]", permute_1718: "f32[128, 576, 576]", permute_1719: "f32[128, 48, 576]", permute_1723: "f32[16, 16]", alias_73: "f32[8, 16, 576, 576]", permute_1729: "f32[16, 16]", permute_1732: "f32[128, 48, 576]", permute_1733: "f32[128, 576, 48]", permute_1736: "f32[2304, 768]", div_108: "f32[8, 576, 1]", permute_1740: "f32[768, 3072]", permute_1744: "f32[3072, 768]", div_109: "f32[8, 576, 1]", permute_1748: "f32[768, 768]", permute_1753: "f32[128, 576, 576]", permute_1754: "f32[128, 48, 576]", permute_1758: "f32[16, 16]", alias_74: "f32[8, 16, 576, 576]", permute_1764: "f32[16, 16]", permute_1767: "f32[128, 48, 576]", permute_1768: "f32[128, 576, 48]", permute_1771: "f32[2304, 768]", div_110: "f32[8, 576, 1]", permute_1775: "f32[768, 3072]", permute_1779: "f32[3072, 768]", div_111: "f32[8, 576, 1]", permute_1783: "f32[768, 768]", permute_1788: "f32[128, 576, 576]", permute_1789: "f32[128, 48, 576]", permute_1793: "f32[16, 16]", alias_75: "f32[8, 16, 576, 576]", permute_1799: "f32[16, 16]", permute_1802: "f32[128, 48, 576]", permute_1803: "f32[128, 576, 48]", permute_1806: "f32[2304, 768]", div_112: "f32[8, 576, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_16: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_1, [8, 576, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_18: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_2, [8, 576, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_7: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_8: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_20: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_3, [8, 576, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_36: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_5, [8, 576, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_38: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_6, [8, 576, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_1: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
    add_17: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_40: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_7, [8, 576, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_56: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_9, [8, 576, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_58: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 576, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_27: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
    erf_2: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_26: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_60: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_11, [8, 576, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_76: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_13, [8, 576, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_78: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_14, [8, 576, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476)
    erf_3: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_37);  mul_37 = None
    add_35: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_80: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_15, [8, 576, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_96: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_17, [8, 576, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_98: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_18, [8, 576, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_47: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
    erf_4: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_44: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_100: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_19, [8, 576, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_116: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_21, [8, 576, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_118: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 576, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_57: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
    erf_5: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
    add_53: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_120: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_23, [8, 576, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_136: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_25, [8, 576, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_138: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_26, [8, 576, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_138, 0.7071067811865476)
    erf_6: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_67);  mul_67 = None
    add_62: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_140: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_27, [8, 576, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_156: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_29, [8, 576, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_158: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_30, [8, 576, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476)
    erf_7: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
    add_71: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_160: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_31, [8, 576, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_176: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_33, [8, 576, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_178: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 576, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_87: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.7071067811865476)
    erf_8: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_80: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_180: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_35, [8, 576, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_196: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_37, [8, 576, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_198: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_38, [8, 576, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_97: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_198, 0.7071067811865476)
    erf_9: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
    add_89: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_200: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_39, [8, 576, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_216: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_41, [8, 576, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_218: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_42, [8, 576, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_107: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_218, 0.7071067811865476)
    erf_10: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
    add_98: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_220: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_43, [8, 576, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_236: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_45, [8, 576, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_238: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_46, [8, 576, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_117: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_238, 0.7071067811865476)
    erf_11: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_117);  mul_117 = None
    add_107: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_240: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_47, [8, 576, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_256: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_49, [8, 576, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_258: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_50, [8, 576, 3072]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_127: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.7071067811865476)
    erf_12: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_127);  mul_127 = None
    add_116: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_260: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_51, [8, 576, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_276: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_53, [8, 576, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_278: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_54, [8, 576, 3072]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_137: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_278, 0.7071067811865476)
    erf_13: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_137);  mul_137 = None
    add_125: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_280: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_55, [8, 576, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_296: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_57, [8, 576, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_298: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_58, [8, 576, 3072]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_147: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_298, 0.7071067811865476)
    erf_14: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_134: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_300: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_59, [8, 576, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_316: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_61, [8, 576, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_318: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_62, [8, 576, 3072]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_157: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_318, 0.7071067811865476)
    erf_15: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_157);  mul_157 = None
    add_143: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_320: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_63, [8, 576, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_336: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_65, [8, 576, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_338: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_66, [8, 576, 3072]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_167: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_338, 0.7071067811865476)
    erf_16: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
    add_152: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_340: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_67, [8, 576, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_356: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_69, [8, 576, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_358: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_70, [8, 576, 3072]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_177: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_358, 0.7071067811865476)
    erf_17: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_177);  mul_177 = None
    add_161: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_360: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_71, [8, 576, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_376: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_73, [8, 576, 768]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_378: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_74, [8, 576, 3072]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_187: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_378, 0.7071067811865476)
    erf_18: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_187);  mul_187 = None
    add_170: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_380: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_75, [8, 576, 768]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_396: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_77, [8, 576, 768]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_398: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_78, [8, 576, 3072]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_197: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_398, 0.7071067811865476)
    erf_19: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_197);  mul_197 = None
    add_179: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_400: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_79, [8, 576, 768]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_416: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_81, [8, 576, 768]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_418: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_82, [8, 576, 3072]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_207: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_418, 0.7071067811865476)
    erf_20: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
    add_188: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_420: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_83, [8, 576, 768]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_436: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_85, [8, 576, 768]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_438: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_86, [8, 576, 3072]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_217: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_438, 0.7071067811865476)
    erf_21: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_217);  mul_217 = None
    add_197: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_440: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_87, [8, 576, 768]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_456: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_89, [8, 576, 768]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_458: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_90, [8, 576, 3072]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_227: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_458, 0.7071067811865476)
    erf_22: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_227);  mul_227 = None
    add_206: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_460: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_91, [8, 576, 768]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_476: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_93, [8, 576, 768]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_478: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_94, [8, 576, 3072]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_237: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_478, 0.7071067811865476)
    erf_23: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_237);  mul_237 = None
    add_215: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_480: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_95, [8, 576, 768]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_496: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_97, [8, 576, 768]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_498: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_98, [8, 576, 3072]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_247: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_498, 0.7071067811865476)
    erf_24: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_247);  mul_247 = None
    add_224: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_500: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_99, [8, 576, 768]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_516: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_101, [8, 576, 768]);  addmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_518: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_102, [8, 576, 3072]);  addmm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_257: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_518, 0.7071067811865476)
    erf_25: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_257);  mul_257 = None
    add_233: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_520: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_103, [8, 576, 768]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_536: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_105, [8, 576, 768]);  addmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_538: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_106, [8, 576, 3072]);  addmm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_267: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_538, 0.7071067811865476)
    erf_26: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_267);  mul_267 = None
    add_242: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_540: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_107, [8, 576, 768]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_556: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_109, [8, 576, 768]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_558: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_110, [8, 576, 3072]);  addmm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_277: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_558, 0.7071067811865476)
    erf_27: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_277);  mul_277 = None
    add_251: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_560: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_111, [8, 576, 768]);  addmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_576: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_113, [8, 576, 768]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_578: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_114, [8, 576, 3072]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_287: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_578, 0.7071067811865476)
    erf_28: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_287);  mul_287 = None
    add_260: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_580: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_115, [8, 576, 768]);  addmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_596: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_117, [8, 576, 768]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_598: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_118, [8, 576, 3072]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_297: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_598, 0.7071067811865476)
    erf_29: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_297);  mul_297 = None
    add_269: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_600: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_119, [8, 576, 768]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_616: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_121, [8, 576, 768]);  addmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_618: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_122, [8, 576, 3072]);  addmm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_307: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_618, 0.7071067811865476)
    erf_30: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_307);  mul_307 = None
    add_278: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_620: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_123, [8, 576, 768]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_636: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_125, [8, 576, 768]);  addmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_638: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_126, [8, 576, 3072]);  addmm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_317: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_638, 0.7071067811865476)
    erf_31: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_317);  mul_317 = None
    add_287: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_640: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_127, [8, 576, 768]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_656: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_129, [8, 576, 768]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_658: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_130, [8, 576, 3072]);  addmm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_327: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_658, 0.7071067811865476)
    erf_32: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_327);  mul_327 = None
    add_296: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_660: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_131, [8, 576, 768]);  addmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_676: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_133, [8, 576, 768]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_678: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_134, [8, 576, 3072]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_337: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_678, 0.7071067811865476)
    erf_33: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_337);  mul_337 = None
    add_305: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_680: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_135, [8, 576, 768]);  addmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_696: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_137, [8, 576, 768]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_698: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_138, [8, 576, 3072]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_347: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_698, 0.7071067811865476)
    erf_34: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_347);  mul_347 = None
    add_314: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_700: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_139, [8, 576, 768]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_716: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_141, [8, 576, 768]);  addmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_718: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(addmm_142, [8, 576, 3072]);  addmm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_357: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_718, 0.7071067811865476)
    erf_35: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_357);  mul_357 = None
    add_323: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_720: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(addmm_143, [8, 576, 768]);  addmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    sub_108: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_145);  cat = getitem_145 = None
    mul_360: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_72);  sub_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_730: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(addmm_147, [8, 1, 768]);  addmm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_732: "f32[8, 1, 3072]" = torch.ops.aten.reshape.default(addmm_148, [8, 1, 3072]);  addmm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_366: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_732, 0.7071067811865476)
    erf_36: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_366);  mul_366 = None
    add_330: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_734: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(addmm_149, [8, 1, 768]);  addmm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    sub_110: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(cat_1, getitem_158);  cat_1 = getitem_158 = None
    mul_369: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_74);  sub_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_744: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(addmm_153, [8, 1, 768]);  addmm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_746: "f32[8, 1, 3072]" = torch.ops.aten.reshape.default(addmm_154, [8, 1, 3072]);  addmm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_375: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_746, 0.7071067811865476)
    erf_37: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_375);  mul_375 = None
    add_337: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_748: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(addmm_155, [8, 1, 768]);  addmm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:351, code: x = self.norm(x)
    sub_112: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(cat_2, getitem_171);  cat_2 = getitem_171 = None
    mul_378: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_76);  sub_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:358, code: return x if pre_logits else self.head(x)
    mm_72: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_490);  permute_490 = None
    permute_491: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_73: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_491, clone_511);  permute_491 = clone_511 = None
    permute_492: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_37: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_749: "f32[1000]" = torch.ops.aten.reshape.default(sum_37, [1000]);  sum_37 = None
    permute_493: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_492, [1, 0]);  permute_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:356, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    full_default: "f32[8, 577, 768]" = torch.ops.aten.full.default([8, 577, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter: "f32[8, 577, 768]" = torch.ops.aten.select_scatter.default(full_default, mm_72, 1, 0);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:351, code: x = self.norm(x)
    mul_381: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(select_scatter, primals_689);  primals_689 = None
    mul_382: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_381, 768)
    sum_38: "f32[8, 577, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True)
    mul_383: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_381, mul_378);  mul_381 = None
    sum_39: "f32[8, 577, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
    mul_384: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_378, sum_39);  sum_39 = None
    sub_114: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(mul_382, sum_38);  mul_382 = sum_38 = None
    sub_115: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(sub_114, mul_384);  sub_114 = mul_384 = None
    div_36: "f32[8, 577, 1]" = torch.ops.aten.div.Tensor(rsqrt_76, 768);  rsqrt_76 = None
    mul_385: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_115);  div_36 = sub_115 = None
    mul_386: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(select_scatter, mul_378);  mul_378 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1]);  mul_386 = None
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:350, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_4: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(mul_385, 1, 0, 1)
    slice_5: "f32[8, 576, 768]" = torch.ops.aten.slice.Tensor(mul_385, 1, 1, 577);  mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    mul_387: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(slice_4, primals_78);  primals_78 = None
    mul_388: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(slice_4, view_748);  view_748 = None
    sum_42: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1], True);  mul_388 = None
    view_750: "f32[768]" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_751: "f32[8, 768]" = torch.ops.aten.reshape.default(mul_387, [8, 768]);  mul_387 = None
    mm_74: "f32[8, 3072]" = torch.ops.aten.mm.default(view_751, permute_494);  permute_494 = None
    permute_495: "f32[768, 8]" = torch.ops.aten.permute.default(view_751, [1, 0])
    mm_75: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_495, view_747);  permute_495 = view_747 = None
    permute_496: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_751, [0], True);  view_751 = None
    view_752: "f32[768]" = torch.ops.aten.reshape.default(sum_43, [768]);  sum_43 = None
    permute_497: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_496, [1, 0]);  permute_496 = None
    view_753: "f32[8, 1, 3072]" = torch.ops.aten.reshape.default(mm_74, [8, 1, 3072]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_390: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_337, 0.5);  add_337 = None
    mul_391: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_746, view_746)
    mul_392: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_391, -0.5);  mul_391 = None
    exp_36: "f32[8, 1, 3072]" = torch.ops.aten.exp.default(mul_392);  mul_392 = None
    mul_393: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_394: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_746, mul_393);  view_746 = mul_393 = None
    add_342: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(mul_390, mul_394);  mul_390 = mul_394 = None
    mul_395: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_753, add_342);  view_753 = add_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_754: "f32[8, 3072]" = torch.ops.aten.reshape.default(mul_395, [8, 3072]);  mul_395 = None
    mm_76: "f32[8, 768]" = torch.ops.aten.mm.default(view_754, permute_498);  permute_498 = None
    permute_499: "f32[3072, 8]" = torch.ops.aten.permute.default(view_754, [1, 0])
    mm_77: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_499, view_745);  permute_499 = view_745 = None
    permute_500: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_44: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_754, [0], True);  view_754 = None
    view_755: "f32[3072]" = torch.ops.aten.reshape.default(sum_44, [3072]);  sum_44 = None
    permute_501: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    view_756: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(mm_76, [8, 1, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    mul_397: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_756, primals_683);  primals_683 = None
    mul_398: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(mul_397, 768)
    sum_45: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True)
    mul_399: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(mul_397, mul_372);  mul_397 = None
    sum_46: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True);  mul_399 = None
    mul_400: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(mul_372, sum_46);  sum_46 = None
    sub_117: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(mul_398, sum_45);  mul_398 = sum_45 = None
    sub_118: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(sub_117, mul_400);  sub_117 = mul_400 = None
    mul_401: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_118);  div_37 = sub_118 = None
    mul_402: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_756, mul_372);  mul_372 = None
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_402, [0, 1]);  mul_402 = None
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_756, [0, 1]);  view_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    add_343: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(slice_4, mul_401);  slice_4 = mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    mul_403: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(add_343, primals_77);  primals_77 = None
    mul_404: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(add_343, view_744);  view_744 = None
    sum_49: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 1], True);  mul_404 = None
    view_757: "f32[768]" = torch.ops.aten.reshape.default(sum_49, [768]);  sum_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_758: "f32[8, 768]" = torch.ops.aten.reshape.default(mul_403, [8, 768]);  mul_403 = None
    mm_78: "f32[8, 768]" = torch.ops.aten.mm.default(view_758, permute_502);  permute_502 = None
    permute_503: "f32[768, 8]" = torch.ops.aten.permute.default(view_758, [1, 0])
    mm_79: "f32[768, 768]" = torch.ops.aten.mm.default(permute_503, view_743);  permute_503 = view_743 = None
    permute_504: "f32[768, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_758, [0], True);  view_758 = None
    view_759: "f32[768]" = torch.ops.aten.reshape.default(sum_50, [768]);  sum_50 = None
    permute_505: "f32[768, 768]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    view_760: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(mm_78, [8, 1, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    view_761: "f32[8, 1, 16, 48]" = torch.ops.aten.reshape.default(view_760, [8, 1, 16, 48]);  view_760 = None
    permute_506: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_761, [0, 2, 1, 3]);  view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_506, permute_480, permute_482, permute_484, alias_38, getitem_160, getitem_161, getitem_162, 0, 0, 0.0, False, getitem_165, getitem_166);  permute_506 = permute_480 = permute_482 = permute_484 = alias_38 = getitem_160 = getitem_161 = getitem_162 = getitem_165 = getitem_166 = None
    getitem_172: "f32[8, 16, 1, 48]" = _scaled_dot_product_flash_attention_backward[0]
    getitem_173: "f32[8, 16, 577, 48]" = _scaled_dot_product_flash_attention_backward[1]
    getitem_174: "f32[8, 16, 577, 48]" = _scaled_dot_product_flash_attention_backward[2];  _scaled_dot_product_flash_attention_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_507: "f32[8, 577, 16, 48]" = torch.ops.aten.permute.default(getitem_174, [0, 2, 1, 3]);  getitem_174 = None
    view_762: "f32[8, 577, 768]" = torch.ops.aten.reshape.default(permute_507, [8, 577, 768]);  permute_507 = None
    view_763: "f32[4616, 768]" = torch.ops.aten.reshape.default(view_762, [4616, 768]);  view_762 = None
    mm_80: "f32[4616, 768]" = torch.ops.aten.mm.default(view_763, permute_508);  permute_508 = None
    permute_509: "f32[768, 4616]" = torch.ops.aten.permute.default(view_763, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_509, view_736);  permute_509 = None
    permute_510: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_763, [0], True);  view_763 = None
    view_764: "f32[768]" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
    permute_511: "f32[768, 768]" = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
    view_765: "f32[8, 577, 768]" = torch.ops.aten.reshape.default(mm_80, [8, 577, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_512: "f32[8, 577, 16, 48]" = torch.ops.aten.permute.default(getitem_173, [0, 2, 1, 3]);  getitem_173 = None
    view_766: "f32[8, 577, 768]" = torch.ops.aten.reshape.default(permute_512, [8, 577, 768]);  permute_512 = None
    view_767: "f32[4616, 768]" = torch.ops.aten.reshape.default(view_766, [4616, 768]);  view_766 = None
    mm_82: "f32[4616, 768]" = torch.ops.aten.mm.default(view_767, permute_513);  permute_513 = None
    permute_514: "f32[768, 4616]" = torch.ops.aten.permute.default(view_767, [1, 0])
    mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_514, view_736);  permute_514 = view_736 = None
    permute_515: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_767, [0], True);  view_767 = None
    view_768: "f32[768]" = torch.ops.aten.reshape.default(sum_52, [768]);  sum_52 = None
    permute_516: "f32[768, 768]" = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
    view_769: "f32[8, 577, 768]" = torch.ops.aten.reshape.default(mm_82, [8, 577, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_344: "f32[8, 577, 768]" = torch.ops.aten.add.Tensor(view_765, view_769);  view_765 = view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_517: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
    view_770: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(permute_517, [8, 1, 768]);  permute_517 = None
    squeeze: "f32[8, 768]" = torch.ops.aten.squeeze.dim(view_770, 1);  view_770 = None
    mm_84: "f32[8, 768]" = torch.ops.aten.mm.default(squeeze, permute_518);  permute_518 = None
    permute_519: "f32[768, 8]" = torch.ops.aten.permute.default(squeeze, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_519, select_109);  permute_519 = select_109 = None
    permute_520: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(squeeze, [0], True);  squeeze = None
    view_771: "f32[768]" = torch.ops.aten.reshape.default(sum_53, [768]);  sum_53 = None
    permute_521: "f32[768, 768]" = torch.ops.aten.permute.default(permute_520, [1, 0]);  permute_520 = None
    select_scatter_1: "f32[8, 577, 768]" = torch.ops.aten.select_scatter.default(full_default, mm_84, 1, 0);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_345: "f32[8, 577, 768]" = torch.ops.aten.add.Tensor(add_344, select_scatter_1);  add_344 = select_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    mul_406: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(add_345, primals_673);  primals_673 = None
    mul_407: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_406, 768)
    sum_54: "f32[8, 577, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [2], True)
    mul_408: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_406, mul_369);  mul_406 = None
    sum_55: "f32[8, 577, 1]" = torch.ops.aten.sum.dim_IntList(mul_408, [2], True);  mul_408 = None
    mul_409: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_369, sum_55);  sum_55 = None
    sub_120: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(mul_407, sum_54);  mul_407 = sum_54 = None
    sub_121: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(sub_120, mul_409);  sub_120 = mul_409 = None
    div_38: "f32[8, 577, 1]" = torch.ops.aten.div.Tensor(rsqrt_74, 768);  rsqrt_74 = None
    mul_410: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_121);  div_38 = sub_121 = None
    mul_411: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(add_345, mul_369);  mul_369 = None
    sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_411, [0, 1]);  mul_411 = None
    sum_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_345, [0, 1]);  add_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:109, code: u = torch.cat((x_cls, x), dim=1)
    slice_6: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(mul_410, 1, 0, 1)
    slice_7: "f32[8, 576, 768]" = torch.ops.aten.slice.Tensor(mul_410, 1, 1, 577);  mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:109, code: u = torch.cat((x_cls, x), dim=1)
    add_346: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(add_343, slice_6);  add_343 = slice_6 = None
    add_347: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(slice_5, slice_7);  slice_5 = slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    mul_412: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(add_346, primals_76);  primals_76 = None
    mul_413: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(add_346, view_734);  view_734 = None
    sum_58: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_413, [0, 1], True);  mul_413 = None
    view_772: "f32[768]" = torch.ops.aten.reshape.default(sum_58, [768]);  sum_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_773: "f32[8, 768]" = torch.ops.aten.reshape.default(mul_412, [8, 768]);  mul_412 = None
    mm_86: "f32[8, 3072]" = torch.ops.aten.mm.default(view_773, permute_522);  permute_522 = None
    permute_523: "f32[768, 8]" = torch.ops.aten.permute.default(view_773, [1, 0])
    mm_87: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_523, view_733);  permute_523 = view_733 = None
    permute_524: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_773, [0], True);  view_773 = None
    view_774: "f32[768]" = torch.ops.aten.reshape.default(sum_59, [768]);  sum_59 = None
    permute_525: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_524, [1, 0]);  permute_524 = None
    view_775: "f32[8, 1, 3072]" = torch.ops.aten.reshape.default(mm_86, [8, 1, 3072]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_415: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_330, 0.5);  add_330 = None
    mul_416: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_732, view_732)
    mul_417: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_416, -0.5);  mul_416 = None
    exp_37: "f32[8, 1, 3072]" = torch.ops.aten.exp.default(mul_417);  mul_417 = None
    mul_418: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_419: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_732, mul_418);  view_732 = mul_418 = None
    add_349: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(mul_415, mul_419);  mul_415 = mul_419 = None
    mul_420: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_775, add_349);  view_775 = add_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_776: "f32[8, 3072]" = torch.ops.aten.reshape.default(mul_420, [8, 3072]);  mul_420 = None
    mm_88: "f32[8, 768]" = torch.ops.aten.mm.default(view_776, permute_526);  permute_526 = None
    permute_527: "f32[3072, 8]" = torch.ops.aten.permute.default(view_776, [1, 0])
    mm_89: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_527, view_731);  permute_527 = view_731 = None
    permute_528: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_60: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_776, [0], True);  view_776 = None
    view_777: "f32[3072]" = torch.ops.aten.reshape.default(sum_60, [3072]);  sum_60 = None
    permute_529: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
    view_778: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(mm_88, [8, 1, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    mul_422: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_778, primals_667);  primals_667 = None
    mul_423: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(mul_422, 768)
    sum_61: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [2], True)
    mul_424: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(mul_422, mul_363);  mul_422 = None
    sum_62: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_424, [2], True);  mul_424 = None
    mul_425: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(mul_363, sum_62);  sum_62 = None
    sub_123: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(mul_423, sum_61);  mul_423 = sum_61 = None
    sub_124: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(sub_123, mul_425);  sub_123 = mul_425 = None
    mul_426: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_124);  div_39 = sub_124 = None
    mul_427: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_778, mul_363);  mul_363 = None
    sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 1]);  mul_427 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_778, [0, 1]);  view_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    add_350: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(add_346, mul_426);  add_346 = mul_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    mul_428: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(add_350, primals_75);  primals_75 = None
    mul_429: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(add_350, view_730);  view_730 = None
    sum_65: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_429, [0, 1], True);  mul_429 = None
    view_779: "f32[768]" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_780: "f32[8, 768]" = torch.ops.aten.reshape.default(mul_428, [8, 768]);  mul_428 = None
    mm_90: "f32[8, 768]" = torch.ops.aten.mm.default(view_780, permute_530);  permute_530 = None
    permute_531: "f32[768, 8]" = torch.ops.aten.permute.default(view_780, [1, 0])
    mm_91: "f32[768, 768]" = torch.ops.aten.mm.default(permute_531, view_729);  permute_531 = view_729 = None
    permute_532: "f32[768, 768]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_780, [0], True);  view_780 = None
    view_781: "f32[768]" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
    permute_533: "f32[768, 768]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    view_782: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(mm_90, [8, 1, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    view_783: "f32[8, 1, 16, 48]" = torch.ops.aten.reshape.default(view_782, [8, 1, 16, 48]);  view_782 = None
    permute_534: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_783, [0, 2, 1, 3]);  view_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_534, permute_470, permute_472, permute_474, alias_39, getitem_147, getitem_148, getitem_149, 0, 0, 0.0, False, getitem_152, getitem_153);  permute_534 = permute_470 = permute_472 = permute_474 = alias_39 = getitem_147 = getitem_148 = getitem_149 = getitem_152 = getitem_153 = None
    getitem_175: "f32[8, 16, 1, 48]" = _scaled_dot_product_flash_attention_backward_1[0]
    getitem_176: "f32[8, 16, 577, 48]" = _scaled_dot_product_flash_attention_backward_1[1]
    getitem_177: "f32[8, 16, 577, 48]" = _scaled_dot_product_flash_attention_backward_1[2];  _scaled_dot_product_flash_attention_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_535: "f32[8, 577, 16, 48]" = torch.ops.aten.permute.default(getitem_177, [0, 2, 1, 3]);  getitem_177 = None
    view_784: "f32[8, 577, 768]" = torch.ops.aten.reshape.default(permute_535, [8, 577, 768]);  permute_535 = None
    view_785: "f32[4616, 768]" = torch.ops.aten.reshape.default(view_784, [4616, 768]);  view_784 = None
    mm_92: "f32[4616, 768]" = torch.ops.aten.mm.default(view_785, permute_536);  permute_536 = None
    permute_537: "f32[768, 4616]" = torch.ops.aten.permute.default(view_785, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_537, view_722);  permute_537 = None
    permute_538: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_785, [0], True);  view_785 = None
    view_786: "f32[768]" = torch.ops.aten.reshape.default(sum_67, [768]);  sum_67 = None
    permute_539: "f32[768, 768]" = torch.ops.aten.permute.default(permute_538, [1, 0]);  permute_538 = None
    view_787: "f32[8, 577, 768]" = torch.ops.aten.reshape.default(mm_92, [8, 577, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_540: "f32[8, 577, 16, 48]" = torch.ops.aten.permute.default(getitem_176, [0, 2, 1, 3]);  getitem_176 = None
    view_788: "f32[8, 577, 768]" = torch.ops.aten.reshape.default(permute_540, [8, 577, 768]);  permute_540 = None
    view_789: "f32[4616, 768]" = torch.ops.aten.reshape.default(view_788, [4616, 768]);  view_788 = None
    mm_94: "f32[4616, 768]" = torch.ops.aten.mm.default(view_789, permute_541);  permute_541 = None
    permute_542: "f32[768, 4616]" = torch.ops.aten.permute.default(view_789, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_542, view_722);  permute_542 = view_722 = None
    permute_543: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_789, [0], True);  view_789 = None
    view_790: "f32[768]" = torch.ops.aten.reshape.default(sum_68, [768]);  sum_68 = None
    permute_544: "f32[768, 768]" = torch.ops.aten.permute.default(permute_543, [1, 0]);  permute_543 = None
    view_791: "f32[8, 577, 768]" = torch.ops.aten.reshape.default(mm_94, [8, 577, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_351: "f32[8, 577, 768]" = torch.ops.aten.add.Tensor(view_787, view_791);  view_787 = view_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_545: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_175, [0, 2, 1, 3]);  getitem_175 = None
    view_792: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(permute_545, [8, 1, 768]);  permute_545 = None
    squeeze_1: "f32[8, 768]" = torch.ops.aten.squeeze.dim(view_792, 1);  view_792 = None
    mm_96: "f32[8, 768]" = torch.ops.aten.mm.default(squeeze_1, permute_546);  permute_546 = None
    permute_547: "f32[768, 8]" = torch.ops.aten.permute.default(squeeze_1, [1, 0])
    mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_547, select_108);  permute_547 = select_108 = None
    permute_548: "f32[768, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_69: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(squeeze_1, [0], True);  squeeze_1 = None
    view_793: "f32[768]" = torch.ops.aten.reshape.default(sum_69, [768]);  sum_69 = None
    permute_549: "f32[768, 768]" = torch.ops.aten.permute.default(permute_548, [1, 0]);  permute_548 = None
    select_scatter_2: "f32[8, 577, 768]" = torch.ops.aten.select_scatter.default(full_default, mm_96, 1, 0);  full_default = mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_352: "f32[8, 577, 768]" = torch.ops.aten.add.Tensor(add_351, select_scatter_2);  add_351 = select_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    mul_431: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(add_352, primals_657);  primals_657 = None
    mul_432: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_431, 768)
    sum_70: "f32[8, 577, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [2], True)
    mul_433: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_431, mul_360);  mul_431 = None
    sum_71: "f32[8, 577, 1]" = torch.ops.aten.sum.dim_IntList(mul_433, [2], True);  mul_433 = None
    mul_434: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_360, sum_71);  sum_71 = None
    sub_126: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(mul_432, sum_70);  mul_432 = sum_70 = None
    sub_127: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(sub_126, mul_434);  sub_126 = mul_434 = None
    div_40: "f32[8, 577, 1]" = torch.ops.aten.div.Tensor(rsqrt_72, 768);  rsqrt_72 = None
    mul_435: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_127);  div_40 = sub_127 = None
    mul_436: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(add_352, mul_360);  mul_360 = None
    sum_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 1]);  mul_436 = None
    sum_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_352, [0, 1]);  add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:109, code: u = torch.cat((x_cls, x), dim=1)
    slice_8: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(mul_435, 1, 0, 1)
    slice_9: "f32[8, 576, 768]" = torch.ops.aten.slice.Tensor(mul_435, 1, 1, 577);  mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:109, code: u = torch.cat((x_cls, x), dim=1)
    add_353: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(add_350, slice_8);  add_350 = slice_8 = None
    add_354: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_347, slice_9);  add_347 = slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:347, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    sum_74: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_353, [0], True);  add_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_437: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_354, primals_73);  primals_73 = None
    mul_438: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_354, view_720);  view_720 = None
    sum_75: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 1], True);  mul_438 = None
    view_794: "f32[768]" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_795: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_437, [4608, 768]);  mul_437 = None
    mm_98: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_795, permute_550);  permute_550 = None
    permute_551: "f32[768, 4608]" = torch.ops.aten.permute.default(view_795, [1, 0])
    mm_99: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_551, view_719);  permute_551 = view_719 = None
    permute_552: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_795, [0], True);  view_795 = None
    view_796: "f32[768]" = torch.ops.aten.reshape.default(sum_76, [768]);  sum_76 = None
    permute_553: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_552, [1, 0]);  permute_552 = None
    view_797: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_98, [8, 576, 3072]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_440: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_323, 0.5);  add_323 = None
    mul_441: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_718, view_718)
    mul_442: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_441, -0.5);  mul_441 = None
    exp_38: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_442);  mul_442 = None
    mul_443: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_444: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_718, mul_443);  view_718 = mul_443 = None
    add_356: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_440, mul_444);  mul_440 = mul_444 = None
    mul_445: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_797, add_356);  view_797 = add_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_798: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_445, [4608, 3072]);  mul_445 = None
    mm_100: "f32[4608, 768]" = torch.ops.aten.mm.default(view_798, permute_554);  permute_554 = None
    permute_555: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_798, [1, 0])
    mm_101: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_555, view_717);  permute_555 = view_717 = None
    permute_556: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_77: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_798, [0], True);  view_798 = None
    view_799: "f32[3072]" = torch.ops.aten.reshape.default(sum_77, [3072]);  sum_77 = None
    permute_557: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_556, [1, 0]);  permute_556 = None
    view_800: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_100, [8, 576, 768]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_447: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_800, primals_651);  primals_651 = None
    mul_448: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_447, 768)
    sum_78: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True)
    mul_449: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_447, mul_354);  mul_447 = None
    sum_79: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_449, [2], True);  mul_449 = None
    mul_450: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_354, sum_79);  sum_79 = None
    sub_129: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_448, sum_78);  mul_448 = sum_78 = None
    sub_130: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_129, mul_450);  sub_129 = mul_450 = None
    mul_451: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_41, sub_130);  div_41 = sub_130 = None
    mul_452: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_800, mul_354);  mul_354 = None
    sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 1]);  mul_452 = None
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_800, [0, 1]);  view_800 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_357: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_354, mul_451);  add_354 = mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_453: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_357, primals_72);  primals_72 = None
    mul_454: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_357, view_716);  view_716 = None
    sum_82: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 1], True);  mul_454 = None
    view_801: "f32[768]" = torch.ops.aten.reshape.default(sum_82, [768]);  sum_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_802: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_453, [4608, 768]);  mul_453 = None
    mm_102: "f32[4608, 768]" = torch.ops.aten.mm.default(view_802, permute_558);  permute_558 = None
    permute_559: "f32[768, 4608]" = torch.ops.aten.permute.default(view_802, [1, 0])
    mm_103: "f32[768, 768]" = torch.ops.aten.mm.default(permute_559, view_715);  permute_559 = view_715 = None
    permute_560: "f32[768, 768]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_802, [0], True);  view_802 = None
    view_803: "f32[768]" = torch.ops.aten.reshape.default(sum_83, [768]);  sum_83 = None
    permute_561: "f32[768, 768]" = torch.ops.aten.permute.default(permute_560, [1, 0]);  permute_560 = None
    view_804: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_102, [8, 576, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_805: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_804, [8, 576, 16, 48]);  view_804 = None
    permute_562: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_805, [0, 2, 1, 3]);  view_805 = None
    clone_513: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_562, memory_format = torch.contiguous_format);  permute_562 = None
    view_806: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_513, [128, 576, 48]);  clone_513 = None
    bmm_72: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_563, view_806);  permute_563 = None
    bmm_73: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_806, permute_564);  view_806 = permute_564 = None
    view_807: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_72, [8, 16, 576, 48]);  bmm_72 = None
    view_808: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_73, [8, 16, 576, 576]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_565: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_808, [0, 2, 3, 1]);  view_808 = None
    sum_84: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_565, [0, 1, 2], True)
    view_809: "f32[16]" = torch.ops.aten.reshape.default(sum_84, [16]);  sum_84 = None
    clone_514: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_565, memory_format = torch.contiguous_format);  permute_565 = None
    view_810: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_514, [2654208, 16]);  clone_514 = None
    permute_566: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_810, [1, 0])
    mm_104: "f32[16, 16]" = torch.ops.aten.mm.default(permute_566, view_709);  permute_566 = view_709 = None
    permute_567: "f32[16, 16]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    mm_105: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_810, permute_568);  view_810 = permute_568 = None
    view_811: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_105, [8, 576, 576, 16]);  mm_105 = None
    permute_569: "f32[16, 16]" = torch.ops.aten.permute.default(permute_567, [1, 0]);  permute_567 = None
    permute_570: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_811, [0, 3, 1, 2]);  view_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_455: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_570, alias_40);  permute_570 = None
    sum_85: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [-1], True)
    mul_456: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_40, sum_85);  alias_40 = sum_85 = None
    sub_131: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_455, mul_456);  mul_455 = mul_456 = None
    clone_515: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_131, memory_format = torch.contiguous_format);  sub_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_571: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_515, [0, 2, 3, 1]);  clone_515 = None
    sum_86: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_571, [0, 1, 2], True)
    view_812: "f32[16]" = torch.ops.aten.reshape.default(sum_86, [16]);  sum_86 = None
    clone_516: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_571, memory_format = torch.contiguous_format);  permute_571 = None
    view_813: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_516, [2654208, 16]);  clone_516 = None
    permute_572: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_813, [1, 0])
    mm_106: "f32[16, 16]" = torch.ops.aten.mm.default(permute_572, view_707);  permute_572 = view_707 = None
    permute_573: "f32[16, 16]" = torch.ops.aten.permute.default(mm_106, [1, 0]);  mm_106 = None
    mm_107: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_813, permute_574);  view_813 = permute_574 = None
    view_814: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_107, [8, 576, 576, 16]);  mm_107 = None
    permute_575: "f32[16, 16]" = torch.ops.aten.permute.default(permute_573, [1, 0]);  permute_573 = None
    permute_576: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_814, [0, 3, 1, 2]);  view_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_517: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_576, memory_format = torch.contiguous_format);  permute_576 = None
    view_815: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_517, [128, 576, 576]);  clone_517 = None
    bmm_74: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_577, view_815);  permute_577 = None
    bmm_75: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_815, permute_578);  view_815 = permute_578 = None
    view_816: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_74, [8, 16, 48, 576]);  bmm_74 = None
    view_817: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_75, [8, 16, 576, 48]);  bmm_75 = None
    permute_579: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_816, [0, 1, 3, 2]);  view_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    full_default_6: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.full.default([3, 8, 16, 576, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_3: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_807, 0, 2);  view_807 = None
    select_scatter_4: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_579, 0, 1);  permute_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_358: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_3, select_scatter_4);  select_scatter_3 = select_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_457: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_817, 0.14433756729740643);  view_817 = None
    select_scatter_5: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_457, 0, 0);  mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_359: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_358, select_scatter_5);  add_358 = select_scatter_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_580: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_359, [1, 3, 0, 2, 4]);  add_359 = None
    clone_518: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_580, memory_format = torch.contiguous_format);  permute_580 = None
    view_818: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_518, [8, 576, 2304]);  clone_518 = None
    view_819: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_818, [4608, 2304]);  view_818 = None
    mm_108: "f32[4608, 768]" = torch.ops.aten.mm.default(view_819, permute_581);  permute_581 = None
    permute_582: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_819, [1, 0])
    mm_109: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_582, view_701);  permute_582 = view_701 = None
    permute_583: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_87: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_819, [0], True);  view_819 = None
    view_820: "f32[2304]" = torch.ops.aten.reshape.default(sum_87, [2304]);  sum_87 = None
    permute_584: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_583, [1, 0]);  permute_583 = None
    view_821: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_108, [8, 576, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_459: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_821, primals_641);  primals_641 = None
    mul_460: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_459, 768)
    sum_88: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True)
    mul_461: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_459, mul_350);  mul_459 = None
    sum_89: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True);  mul_461 = None
    mul_462: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_350, sum_89);  sum_89 = None
    sub_133: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_460, sum_88);  mul_460 = sum_88 = None
    sub_134: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_133, mul_462);  sub_133 = mul_462 = None
    mul_463: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_134);  div_42 = sub_134 = None
    mul_464: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_821, mul_350);  mul_350 = None
    sum_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 1]);  mul_464 = None
    sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_821, [0, 1]);  view_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_360: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_357, mul_463);  add_357 = mul_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_465: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_360, primals_71);  primals_71 = None
    mul_466: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_360, view_700);  view_700 = None
    sum_92: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_466, [0, 1], True);  mul_466 = None
    view_822: "f32[768]" = torch.ops.aten.reshape.default(sum_92, [768]);  sum_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_823: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_465, [4608, 768]);  mul_465 = None
    mm_110: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_823, permute_585);  permute_585 = None
    permute_586: "f32[768, 4608]" = torch.ops.aten.permute.default(view_823, [1, 0])
    mm_111: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_586, view_699);  permute_586 = view_699 = None
    permute_587: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_823, [0], True);  view_823 = None
    view_824: "f32[768]" = torch.ops.aten.reshape.default(sum_93, [768]);  sum_93 = None
    permute_588: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_587, [1, 0]);  permute_587 = None
    view_825: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_110, [8, 576, 3072]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_468: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_314, 0.5);  add_314 = None
    mul_469: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_698, view_698)
    mul_470: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_469, -0.5);  mul_469 = None
    exp_39: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_470);  mul_470 = None
    mul_471: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_472: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_698, mul_471);  view_698 = mul_471 = None
    add_362: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_468, mul_472);  mul_468 = mul_472 = None
    mul_473: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_825, add_362);  view_825 = add_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_826: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_473, [4608, 3072]);  mul_473 = None
    mm_112: "f32[4608, 768]" = torch.ops.aten.mm.default(view_826, permute_589);  permute_589 = None
    permute_590: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_826, [1, 0])
    mm_113: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_590, view_697);  permute_590 = view_697 = None
    permute_591: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_94: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_826, [0], True);  view_826 = None
    view_827: "f32[3072]" = torch.ops.aten.reshape.default(sum_94, [3072]);  sum_94 = None
    permute_592: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_591, [1, 0]);  permute_591 = None
    view_828: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_112, [8, 576, 768]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_475: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_828, primals_635);  primals_635 = None
    mul_476: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_475, 768)
    sum_95: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_475, [2], True)
    mul_477: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_475, mul_344);  mul_475 = None
    sum_96: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_477, [2], True);  mul_477 = None
    mul_478: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_344, sum_96);  sum_96 = None
    sub_136: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_476, sum_95);  mul_476 = sum_95 = None
    sub_137: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_136, mul_478);  sub_136 = mul_478 = None
    mul_479: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_137);  div_43 = sub_137 = None
    mul_480: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_828, mul_344);  mul_344 = None
    sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_480, [0, 1]);  mul_480 = None
    sum_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_828, [0, 1]);  view_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_363: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_360, mul_479);  add_360 = mul_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_481: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_363, primals_70);  primals_70 = None
    mul_482: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_363, view_696);  view_696 = None
    sum_99: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 1], True);  mul_482 = None
    view_829: "f32[768]" = torch.ops.aten.reshape.default(sum_99, [768]);  sum_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_830: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_481, [4608, 768]);  mul_481 = None
    mm_114: "f32[4608, 768]" = torch.ops.aten.mm.default(view_830, permute_593);  permute_593 = None
    permute_594: "f32[768, 4608]" = torch.ops.aten.permute.default(view_830, [1, 0])
    mm_115: "f32[768, 768]" = torch.ops.aten.mm.default(permute_594, view_695);  permute_594 = view_695 = None
    permute_595: "f32[768, 768]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_100: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_830, [0], True);  view_830 = None
    view_831: "f32[768]" = torch.ops.aten.reshape.default(sum_100, [768]);  sum_100 = None
    permute_596: "f32[768, 768]" = torch.ops.aten.permute.default(permute_595, [1, 0]);  permute_595 = None
    view_832: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_114, [8, 576, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_833: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_832, [8, 576, 16, 48]);  view_832 = None
    permute_597: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_833, [0, 2, 1, 3]);  view_833 = None
    clone_521: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_597, memory_format = torch.contiguous_format);  permute_597 = None
    view_834: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_521, [128, 576, 48]);  clone_521 = None
    bmm_76: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_598, view_834);  permute_598 = None
    bmm_77: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_834, permute_599);  view_834 = permute_599 = None
    view_835: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_76, [8, 16, 576, 48]);  bmm_76 = None
    view_836: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_77, [8, 16, 576, 576]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_600: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_836, [0, 2, 3, 1]);  view_836 = None
    sum_101: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_600, [0, 1, 2], True)
    view_837: "f32[16]" = torch.ops.aten.reshape.default(sum_101, [16]);  sum_101 = None
    clone_522: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_600, memory_format = torch.contiguous_format);  permute_600 = None
    view_838: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_522, [2654208, 16]);  clone_522 = None
    permute_601: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_838, [1, 0])
    mm_116: "f32[16, 16]" = torch.ops.aten.mm.default(permute_601, view_689);  permute_601 = view_689 = None
    permute_602: "f32[16, 16]" = torch.ops.aten.permute.default(mm_116, [1, 0]);  mm_116 = None
    mm_117: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_838, permute_603);  view_838 = permute_603 = None
    view_839: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_117, [8, 576, 576, 16]);  mm_117 = None
    permute_604: "f32[16, 16]" = torch.ops.aten.permute.default(permute_602, [1, 0]);  permute_602 = None
    permute_605: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_839, [0, 3, 1, 2]);  view_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_483: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_605, alias_41);  permute_605 = None
    sum_102: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [-1], True)
    mul_484: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_41, sum_102);  alias_41 = sum_102 = None
    sub_138: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
    clone_523: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_138, memory_format = torch.contiguous_format);  sub_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_606: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_523, [0, 2, 3, 1]);  clone_523 = None
    sum_103: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_606, [0, 1, 2], True)
    view_840: "f32[16]" = torch.ops.aten.reshape.default(sum_103, [16]);  sum_103 = None
    clone_524: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_606, memory_format = torch.contiguous_format);  permute_606 = None
    view_841: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_524, [2654208, 16]);  clone_524 = None
    permute_607: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_841, [1, 0])
    mm_118: "f32[16, 16]" = torch.ops.aten.mm.default(permute_607, view_687);  permute_607 = view_687 = None
    permute_608: "f32[16, 16]" = torch.ops.aten.permute.default(mm_118, [1, 0]);  mm_118 = None
    mm_119: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_841, permute_609);  view_841 = permute_609 = None
    view_842: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_119, [8, 576, 576, 16]);  mm_119 = None
    permute_610: "f32[16, 16]" = torch.ops.aten.permute.default(permute_608, [1, 0]);  permute_608 = None
    permute_611: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_842, [0, 3, 1, 2]);  view_842 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_525: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_611, memory_format = torch.contiguous_format);  permute_611 = None
    view_843: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_525, [128, 576, 576]);  clone_525 = None
    bmm_78: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_612, view_843);  permute_612 = None
    bmm_79: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_843, permute_613);  view_843 = permute_613 = None
    view_844: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_78, [8, 16, 48, 576]);  bmm_78 = None
    view_845: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_79, [8, 16, 576, 48]);  bmm_79 = None
    permute_614: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_844, [0, 1, 3, 2]);  view_844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_6: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_835, 0, 2);  view_835 = None
    select_scatter_7: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_614, 0, 1);  permute_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_364: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_6, select_scatter_7);  select_scatter_6 = select_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_485: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_845, 0.14433756729740643);  view_845 = None
    select_scatter_8: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_485, 0, 0);  mul_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_365: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_364, select_scatter_8);  add_364 = select_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_615: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_365, [1, 3, 0, 2, 4]);  add_365 = None
    clone_526: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_615, memory_format = torch.contiguous_format);  permute_615 = None
    view_846: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_526, [8, 576, 2304]);  clone_526 = None
    view_847: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_846, [4608, 2304]);  view_846 = None
    mm_120: "f32[4608, 768]" = torch.ops.aten.mm.default(view_847, permute_616);  permute_616 = None
    permute_617: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_847, [1, 0])
    mm_121: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_617, view_681);  permute_617 = view_681 = None
    permute_618: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_104: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_847, [0], True);  view_847 = None
    view_848: "f32[2304]" = torch.ops.aten.reshape.default(sum_104, [2304]);  sum_104 = None
    permute_619: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_618, [1, 0]);  permute_618 = None
    view_849: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_120, [8, 576, 768]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_487: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_849, primals_625);  primals_625 = None
    mul_488: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_487, 768)
    sum_105: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_487, [2], True)
    mul_489: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_487, mul_340);  mul_487 = None
    sum_106: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_489, [2], True);  mul_489 = None
    mul_490: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_340, sum_106);  sum_106 = None
    sub_140: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_488, sum_105);  mul_488 = sum_105 = None
    sub_141: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_140, mul_490);  sub_140 = mul_490 = None
    mul_491: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_44, sub_141);  div_44 = sub_141 = None
    mul_492: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_849, mul_340);  mul_340 = None
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_492, [0, 1]);  mul_492 = None
    sum_108: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_849, [0, 1]);  view_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_366: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_363, mul_491);  add_363 = mul_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_493: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_366, primals_69);  primals_69 = None
    mul_494: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_366, view_680);  view_680 = None
    sum_109: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_494, [0, 1], True);  mul_494 = None
    view_850: "f32[768]" = torch.ops.aten.reshape.default(sum_109, [768]);  sum_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_851: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_493, [4608, 768]);  mul_493 = None
    mm_122: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_851, permute_620);  permute_620 = None
    permute_621: "f32[768, 4608]" = torch.ops.aten.permute.default(view_851, [1, 0])
    mm_123: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_621, view_679);  permute_621 = view_679 = None
    permute_622: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_851, [0], True);  view_851 = None
    view_852: "f32[768]" = torch.ops.aten.reshape.default(sum_110, [768]);  sum_110 = None
    permute_623: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_622, [1, 0]);  permute_622 = None
    view_853: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_122, [8, 576, 3072]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_496: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_305, 0.5);  add_305 = None
    mul_497: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_678, view_678)
    mul_498: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_497, -0.5);  mul_497 = None
    exp_40: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_498);  mul_498 = None
    mul_499: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
    mul_500: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_678, mul_499);  view_678 = mul_499 = None
    add_368: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_496, mul_500);  mul_496 = mul_500 = None
    mul_501: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_853, add_368);  view_853 = add_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_854: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_501, [4608, 3072]);  mul_501 = None
    mm_124: "f32[4608, 768]" = torch.ops.aten.mm.default(view_854, permute_624);  permute_624 = None
    permute_625: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_854, [1, 0])
    mm_125: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_625, view_677);  permute_625 = view_677 = None
    permute_626: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_111: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_854, [0], True);  view_854 = None
    view_855: "f32[3072]" = torch.ops.aten.reshape.default(sum_111, [3072]);  sum_111 = None
    permute_627: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_626, [1, 0]);  permute_626 = None
    view_856: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_124, [8, 576, 768]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_503: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_856, primals_619);  primals_619 = None
    mul_504: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_503, 768)
    sum_112: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_503, [2], True)
    mul_505: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_503, mul_334);  mul_503 = None
    sum_113: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_505, [2], True);  mul_505 = None
    mul_506: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_334, sum_113);  sum_113 = None
    sub_143: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_504, sum_112);  mul_504 = sum_112 = None
    sub_144: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_143, mul_506);  sub_143 = mul_506 = None
    mul_507: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_144);  div_45 = sub_144 = None
    mul_508: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_856, mul_334);  mul_334 = None
    sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_508, [0, 1]);  mul_508 = None
    sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_856, [0, 1]);  view_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_369: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_366, mul_507);  add_366 = mul_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_509: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_369, primals_68);  primals_68 = None
    mul_510: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_369, view_676);  view_676 = None
    sum_116: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 1], True);  mul_510 = None
    view_857: "f32[768]" = torch.ops.aten.reshape.default(sum_116, [768]);  sum_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_858: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_509, [4608, 768]);  mul_509 = None
    mm_126: "f32[4608, 768]" = torch.ops.aten.mm.default(view_858, permute_628);  permute_628 = None
    permute_629: "f32[768, 4608]" = torch.ops.aten.permute.default(view_858, [1, 0])
    mm_127: "f32[768, 768]" = torch.ops.aten.mm.default(permute_629, view_675);  permute_629 = view_675 = None
    permute_630: "f32[768, 768]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_858, [0], True);  view_858 = None
    view_859: "f32[768]" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
    permute_631: "f32[768, 768]" = torch.ops.aten.permute.default(permute_630, [1, 0]);  permute_630 = None
    view_860: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_126, [8, 576, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_861: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_860, [8, 576, 16, 48]);  view_860 = None
    permute_632: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_861, [0, 2, 1, 3]);  view_861 = None
    clone_529: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_632, memory_format = torch.contiguous_format);  permute_632 = None
    view_862: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_529, [128, 576, 48]);  clone_529 = None
    bmm_80: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_633, view_862);  permute_633 = None
    bmm_81: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_862, permute_634);  view_862 = permute_634 = None
    view_863: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_80, [8, 16, 576, 48]);  bmm_80 = None
    view_864: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_81, [8, 16, 576, 576]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_635: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_864, [0, 2, 3, 1]);  view_864 = None
    sum_118: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_635, [0, 1, 2], True)
    view_865: "f32[16]" = torch.ops.aten.reshape.default(sum_118, [16]);  sum_118 = None
    clone_530: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_635, memory_format = torch.contiguous_format);  permute_635 = None
    view_866: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_530, [2654208, 16]);  clone_530 = None
    permute_636: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_866, [1, 0])
    mm_128: "f32[16, 16]" = torch.ops.aten.mm.default(permute_636, view_669);  permute_636 = view_669 = None
    permute_637: "f32[16, 16]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    mm_129: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_866, permute_638);  view_866 = permute_638 = None
    view_867: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_129, [8, 576, 576, 16]);  mm_129 = None
    permute_639: "f32[16, 16]" = torch.ops.aten.permute.default(permute_637, [1, 0]);  permute_637 = None
    permute_640: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_867, [0, 3, 1, 2]);  view_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_511: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_640, alias_42);  permute_640 = None
    sum_119: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_511, [-1], True)
    mul_512: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_42, sum_119);  alias_42 = sum_119 = None
    sub_145: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_511, mul_512);  mul_511 = mul_512 = None
    clone_531: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_145, memory_format = torch.contiguous_format);  sub_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_641: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_531, [0, 2, 3, 1]);  clone_531 = None
    sum_120: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_641, [0, 1, 2], True)
    view_868: "f32[16]" = torch.ops.aten.reshape.default(sum_120, [16]);  sum_120 = None
    clone_532: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_641, memory_format = torch.contiguous_format);  permute_641 = None
    view_869: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_532, [2654208, 16]);  clone_532 = None
    permute_642: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_869, [1, 0])
    mm_130: "f32[16, 16]" = torch.ops.aten.mm.default(permute_642, view_667);  permute_642 = view_667 = None
    permute_643: "f32[16, 16]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    mm_131: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_869, permute_644);  view_869 = permute_644 = None
    view_870: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_131, [8, 576, 576, 16]);  mm_131 = None
    permute_645: "f32[16, 16]" = torch.ops.aten.permute.default(permute_643, [1, 0]);  permute_643 = None
    permute_646: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_870, [0, 3, 1, 2]);  view_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_533: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_646, memory_format = torch.contiguous_format);  permute_646 = None
    view_871: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_533, [128, 576, 576]);  clone_533 = None
    bmm_82: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_647, view_871);  permute_647 = None
    bmm_83: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_871, permute_648);  view_871 = permute_648 = None
    view_872: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_82, [8, 16, 48, 576]);  bmm_82 = None
    view_873: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_83, [8, 16, 576, 48]);  bmm_83 = None
    permute_649: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_872, [0, 1, 3, 2]);  view_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_9: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_863, 0, 2);  view_863 = None
    select_scatter_10: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_649, 0, 1);  permute_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_370: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_9, select_scatter_10);  select_scatter_9 = select_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_513: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_873, 0.14433756729740643);  view_873 = None
    select_scatter_11: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_513, 0, 0);  mul_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_371: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_370, select_scatter_11);  add_370 = select_scatter_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_650: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_371, [1, 3, 0, 2, 4]);  add_371 = None
    clone_534: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
    view_874: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_534, [8, 576, 2304]);  clone_534 = None
    view_875: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_874, [4608, 2304]);  view_874 = None
    mm_132: "f32[4608, 768]" = torch.ops.aten.mm.default(view_875, permute_651);  permute_651 = None
    permute_652: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_875, [1, 0])
    mm_133: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_652, view_661);  permute_652 = view_661 = None
    permute_653: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_121: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_875, [0], True);  view_875 = None
    view_876: "f32[2304]" = torch.ops.aten.reshape.default(sum_121, [2304]);  sum_121 = None
    permute_654: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_653, [1, 0]);  permute_653 = None
    view_877: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_132, [8, 576, 768]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_515: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_877, primals_609);  primals_609 = None
    mul_516: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_515, 768)
    sum_122: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_515, [2], True)
    mul_517: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_515, mul_330);  mul_515 = None
    sum_123: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_517, [2], True);  mul_517 = None
    mul_518: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_330, sum_123);  sum_123 = None
    sub_147: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_516, sum_122);  mul_516 = sum_122 = None
    sub_148: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_147, mul_518);  sub_147 = mul_518 = None
    mul_519: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_148);  div_46 = sub_148 = None
    mul_520: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_877, mul_330);  mul_330 = None
    sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_520, [0, 1]);  mul_520 = None
    sum_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_877, [0, 1]);  view_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_372: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_369, mul_519);  add_369 = mul_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_521: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_372, primals_67);  primals_67 = None
    mul_522: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_372, view_660);  view_660 = None
    sum_126: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_522, [0, 1], True);  mul_522 = None
    view_878: "f32[768]" = torch.ops.aten.reshape.default(sum_126, [768]);  sum_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_879: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_521, [4608, 768]);  mul_521 = None
    mm_134: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_879, permute_655);  permute_655 = None
    permute_656: "f32[768, 4608]" = torch.ops.aten.permute.default(view_879, [1, 0])
    mm_135: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_656, view_659);  permute_656 = view_659 = None
    permute_657: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_879, [0], True);  view_879 = None
    view_880: "f32[768]" = torch.ops.aten.reshape.default(sum_127, [768]);  sum_127 = None
    permute_658: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_657, [1, 0]);  permute_657 = None
    view_881: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_134, [8, 576, 3072]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_524: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_296, 0.5);  add_296 = None
    mul_525: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_658, view_658)
    mul_526: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_525, -0.5);  mul_525 = None
    exp_41: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_526);  mul_526 = None
    mul_527: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
    mul_528: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_658, mul_527);  view_658 = mul_527 = None
    add_374: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_524, mul_528);  mul_524 = mul_528 = None
    mul_529: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_881, add_374);  view_881 = add_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_882: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_529, [4608, 3072]);  mul_529 = None
    mm_136: "f32[4608, 768]" = torch.ops.aten.mm.default(view_882, permute_659);  permute_659 = None
    permute_660: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_882, [1, 0])
    mm_137: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_660, view_657);  permute_660 = view_657 = None
    permute_661: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_128: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_882, [0], True);  view_882 = None
    view_883: "f32[3072]" = torch.ops.aten.reshape.default(sum_128, [3072]);  sum_128 = None
    permute_662: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_661, [1, 0]);  permute_661 = None
    view_884: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_136, [8, 576, 768]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_531: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_884, primals_603);  primals_603 = None
    mul_532: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_531, 768)
    sum_129: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_531, [2], True)
    mul_533: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_531, mul_324);  mul_531 = None
    sum_130: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_533, [2], True);  mul_533 = None
    mul_534: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_324, sum_130);  sum_130 = None
    sub_150: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_532, sum_129);  mul_532 = sum_129 = None
    sub_151: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_150, mul_534);  sub_150 = mul_534 = None
    mul_535: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_47, sub_151);  div_47 = sub_151 = None
    mul_536: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_884, mul_324);  mul_324 = None
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 1]);  mul_536 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_884, [0, 1]);  view_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_375: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_372, mul_535);  add_372 = mul_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_537: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_375, primals_66);  primals_66 = None
    mul_538: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_375, view_656);  view_656 = None
    sum_133: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_538, [0, 1], True);  mul_538 = None
    view_885: "f32[768]" = torch.ops.aten.reshape.default(sum_133, [768]);  sum_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_886: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_537, [4608, 768]);  mul_537 = None
    mm_138: "f32[4608, 768]" = torch.ops.aten.mm.default(view_886, permute_663);  permute_663 = None
    permute_664: "f32[768, 4608]" = torch.ops.aten.permute.default(view_886, [1, 0])
    mm_139: "f32[768, 768]" = torch.ops.aten.mm.default(permute_664, view_655);  permute_664 = view_655 = None
    permute_665: "f32[768, 768]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_886, [0], True);  view_886 = None
    view_887: "f32[768]" = torch.ops.aten.reshape.default(sum_134, [768]);  sum_134 = None
    permute_666: "f32[768, 768]" = torch.ops.aten.permute.default(permute_665, [1, 0]);  permute_665 = None
    view_888: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_138, [8, 576, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_889: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_888, [8, 576, 16, 48]);  view_888 = None
    permute_667: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_889, [0, 2, 1, 3]);  view_889 = None
    clone_537: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_667, memory_format = torch.contiguous_format);  permute_667 = None
    view_890: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_537, [128, 576, 48]);  clone_537 = None
    bmm_84: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_668, view_890);  permute_668 = None
    bmm_85: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_890, permute_669);  view_890 = permute_669 = None
    view_891: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_84, [8, 16, 576, 48]);  bmm_84 = None
    view_892: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_85, [8, 16, 576, 576]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_670: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_892, [0, 2, 3, 1]);  view_892 = None
    sum_135: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_670, [0, 1, 2], True)
    view_893: "f32[16]" = torch.ops.aten.reshape.default(sum_135, [16]);  sum_135 = None
    clone_538: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_670, memory_format = torch.contiguous_format);  permute_670 = None
    view_894: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_538, [2654208, 16]);  clone_538 = None
    permute_671: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_894, [1, 0])
    mm_140: "f32[16, 16]" = torch.ops.aten.mm.default(permute_671, view_649);  permute_671 = view_649 = None
    permute_672: "f32[16, 16]" = torch.ops.aten.permute.default(mm_140, [1, 0]);  mm_140 = None
    mm_141: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_894, permute_673);  view_894 = permute_673 = None
    view_895: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_141, [8, 576, 576, 16]);  mm_141 = None
    permute_674: "f32[16, 16]" = torch.ops.aten.permute.default(permute_672, [1, 0]);  permute_672 = None
    permute_675: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_895, [0, 3, 1, 2]);  view_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_539: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_675, alias_43);  permute_675 = None
    sum_136: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_539, [-1], True)
    mul_540: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_43, sum_136);  alias_43 = sum_136 = None
    sub_152: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_539, mul_540);  mul_539 = mul_540 = None
    clone_539: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_152, memory_format = torch.contiguous_format);  sub_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_676: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_539, [0, 2, 3, 1]);  clone_539 = None
    sum_137: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_676, [0, 1, 2], True)
    view_896: "f32[16]" = torch.ops.aten.reshape.default(sum_137, [16]);  sum_137 = None
    clone_540: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_676, memory_format = torch.contiguous_format);  permute_676 = None
    view_897: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_540, [2654208, 16]);  clone_540 = None
    permute_677: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_897, [1, 0])
    mm_142: "f32[16, 16]" = torch.ops.aten.mm.default(permute_677, view_647);  permute_677 = view_647 = None
    permute_678: "f32[16, 16]" = torch.ops.aten.permute.default(mm_142, [1, 0]);  mm_142 = None
    mm_143: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_897, permute_679);  view_897 = permute_679 = None
    view_898: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_143, [8, 576, 576, 16]);  mm_143 = None
    permute_680: "f32[16, 16]" = torch.ops.aten.permute.default(permute_678, [1, 0]);  permute_678 = None
    permute_681: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_898, [0, 3, 1, 2]);  view_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_541: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_681, memory_format = torch.contiguous_format);  permute_681 = None
    view_899: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_541, [128, 576, 576]);  clone_541 = None
    bmm_86: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_682, view_899);  permute_682 = None
    bmm_87: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_899, permute_683);  view_899 = permute_683 = None
    view_900: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_86, [8, 16, 48, 576]);  bmm_86 = None
    view_901: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_87, [8, 16, 576, 48]);  bmm_87 = None
    permute_684: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_900, [0, 1, 3, 2]);  view_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_12: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_891, 0, 2);  view_891 = None
    select_scatter_13: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_684, 0, 1);  permute_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_376: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_12, select_scatter_13);  select_scatter_12 = select_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_541: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_901, 0.14433756729740643);  view_901 = None
    select_scatter_14: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_541, 0, 0);  mul_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_377: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_376, select_scatter_14);  add_376 = select_scatter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_685: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_377, [1, 3, 0, 2, 4]);  add_377 = None
    clone_542: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_685, memory_format = torch.contiguous_format);  permute_685 = None
    view_902: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_542, [8, 576, 2304]);  clone_542 = None
    view_903: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_902, [4608, 2304]);  view_902 = None
    mm_144: "f32[4608, 768]" = torch.ops.aten.mm.default(view_903, permute_686);  permute_686 = None
    permute_687: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_903, [1, 0])
    mm_145: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_687, view_641);  permute_687 = view_641 = None
    permute_688: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_138: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_903, [0], True);  view_903 = None
    view_904: "f32[2304]" = torch.ops.aten.reshape.default(sum_138, [2304]);  sum_138 = None
    permute_689: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_688, [1, 0]);  permute_688 = None
    view_905: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_144, [8, 576, 768]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_543: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_905, primals_593);  primals_593 = None
    mul_544: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_543, 768)
    sum_139: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_543, [2], True)
    mul_545: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_543, mul_320);  mul_543 = None
    sum_140: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_545, [2], True);  mul_545 = None
    mul_546: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_320, sum_140);  sum_140 = None
    sub_154: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_544, sum_139);  mul_544 = sum_139 = None
    sub_155: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_154, mul_546);  sub_154 = mul_546 = None
    mul_547: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_155);  div_48 = sub_155 = None
    mul_548: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_905, mul_320);  mul_320 = None
    sum_141: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_548, [0, 1]);  mul_548 = None
    sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_905, [0, 1]);  view_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_378: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_375, mul_547);  add_375 = mul_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_549: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_378, primals_65);  primals_65 = None
    mul_550: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_378, view_640);  view_640 = None
    sum_143: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_550, [0, 1], True);  mul_550 = None
    view_906: "f32[768]" = torch.ops.aten.reshape.default(sum_143, [768]);  sum_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_907: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_549, [4608, 768]);  mul_549 = None
    mm_146: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_907, permute_690);  permute_690 = None
    permute_691: "f32[768, 4608]" = torch.ops.aten.permute.default(view_907, [1, 0])
    mm_147: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_691, view_639);  permute_691 = view_639 = None
    permute_692: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_144: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_907, [0], True);  view_907 = None
    view_908: "f32[768]" = torch.ops.aten.reshape.default(sum_144, [768]);  sum_144 = None
    permute_693: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_692, [1, 0]);  permute_692 = None
    view_909: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_146, [8, 576, 3072]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_552: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_287, 0.5);  add_287 = None
    mul_553: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_638, view_638)
    mul_554: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_553, -0.5);  mul_553 = None
    exp_42: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_554);  mul_554 = None
    mul_555: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
    mul_556: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_638, mul_555);  view_638 = mul_555 = None
    add_380: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_552, mul_556);  mul_552 = mul_556 = None
    mul_557: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_909, add_380);  view_909 = add_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_910: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_557, [4608, 3072]);  mul_557 = None
    mm_148: "f32[4608, 768]" = torch.ops.aten.mm.default(view_910, permute_694);  permute_694 = None
    permute_695: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_910, [1, 0])
    mm_149: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_695, view_637);  permute_695 = view_637 = None
    permute_696: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_145: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_910, [0], True);  view_910 = None
    view_911: "f32[3072]" = torch.ops.aten.reshape.default(sum_145, [3072]);  sum_145 = None
    permute_697: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_696, [1, 0]);  permute_696 = None
    view_912: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_148, [8, 576, 768]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_559: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_912, primals_587);  primals_587 = None
    mul_560: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_559, 768)
    sum_146: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_559, [2], True)
    mul_561: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_559, mul_314);  mul_559 = None
    sum_147: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_561, [2], True);  mul_561 = None
    mul_562: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_314, sum_147);  sum_147 = None
    sub_157: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_560, sum_146);  mul_560 = sum_146 = None
    sub_158: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_157, mul_562);  sub_157 = mul_562 = None
    mul_563: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_158);  div_49 = sub_158 = None
    mul_564: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_912, mul_314);  mul_314 = None
    sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_564, [0, 1]);  mul_564 = None
    sum_149: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_912, [0, 1]);  view_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_381: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_378, mul_563);  add_378 = mul_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_565: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_381, primals_64);  primals_64 = None
    mul_566: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_381, view_636);  view_636 = None
    sum_150: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_566, [0, 1], True);  mul_566 = None
    view_913: "f32[768]" = torch.ops.aten.reshape.default(sum_150, [768]);  sum_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_914: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_565, [4608, 768]);  mul_565 = None
    mm_150: "f32[4608, 768]" = torch.ops.aten.mm.default(view_914, permute_698);  permute_698 = None
    permute_699: "f32[768, 4608]" = torch.ops.aten.permute.default(view_914, [1, 0])
    mm_151: "f32[768, 768]" = torch.ops.aten.mm.default(permute_699, view_635);  permute_699 = view_635 = None
    permute_700: "f32[768, 768]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_151: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_914, [0], True);  view_914 = None
    view_915: "f32[768]" = torch.ops.aten.reshape.default(sum_151, [768]);  sum_151 = None
    permute_701: "f32[768, 768]" = torch.ops.aten.permute.default(permute_700, [1, 0]);  permute_700 = None
    view_916: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_150, [8, 576, 768]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_917: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_916, [8, 576, 16, 48]);  view_916 = None
    permute_702: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_917, [0, 2, 1, 3]);  view_917 = None
    clone_545: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_702, memory_format = torch.contiguous_format);  permute_702 = None
    view_918: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_545, [128, 576, 48]);  clone_545 = None
    bmm_88: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_703, view_918);  permute_703 = None
    bmm_89: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_918, permute_704);  view_918 = permute_704 = None
    view_919: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_88, [8, 16, 576, 48]);  bmm_88 = None
    view_920: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_89, [8, 16, 576, 576]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_705: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_920, [0, 2, 3, 1]);  view_920 = None
    sum_152: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_705, [0, 1, 2], True)
    view_921: "f32[16]" = torch.ops.aten.reshape.default(sum_152, [16]);  sum_152 = None
    clone_546: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_705, memory_format = torch.contiguous_format);  permute_705 = None
    view_922: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_546, [2654208, 16]);  clone_546 = None
    permute_706: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_922, [1, 0])
    mm_152: "f32[16, 16]" = torch.ops.aten.mm.default(permute_706, view_629);  permute_706 = view_629 = None
    permute_707: "f32[16, 16]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    mm_153: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_922, permute_708);  view_922 = permute_708 = None
    view_923: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_153, [8, 576, 576, 16]);  mm_153 = None
    permute_709: "f32[16, 16]" = torch.ops.aten.permute.default(permute_707, [1, 0]);  permute_707 = None
    permute_710: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_923, [0, 3, 1, 2]);  view_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_567: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_710, alias_44);  permute_710 = None
    sum_153: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_567, [-1], True)
    mul_568: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_44, sum_153);  alias_44 = sum_153 = None
    sub_159: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_567, mul_568);  mul_567 = mul_568 = None
    clone_547: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_159, memory_format = torch.contiguous_format);  sub_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_711: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_547, [0, 2, 3, 1]);  clone_547 = None
    sum_154: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_711, [0, 1, 2], True)
    view_924: "f32[16]" = torch.ops.aten.reshape.default(sum_154, [16]);  sum_154 = None
    clone_548: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_711, memory_format = torch.contiguous_format);  permute_711 = None
    view_925: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_548, [2654208, 16]);  clone_548 = None
    permute_712: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_925, [1, 0])
    mm_154: "f32[16, 16]" = torch.ops.aten.mm.default(permute_712, view_627);  permute_712 = view_627 = None
    permute_713: "f32[16, 16]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
    mm_155: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_925, permute_714);  view_925 = permute_714 = None
    view_926: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_155, [8, 576, 576, 16]);  mm_155 = None
    permute_715: "f32[16, 16]" = torch.ops.aten.permute.default(permute_713, [1, 0]);  permute_713 = None
    permute_716: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_926, [0, 3, 1, 2]);  view_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_549: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_716, memory_format = torch.contiguous_format);  permute_716 = None
    view_927: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_549, [128, 576, 576]);  clone_549 = None
    bmm_90: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_717, view_927);  permute_717 = None
    bmm_91: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_927, permute_718);  view_927 = permute_718 = None
    view_928: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_90, [8, 16, 48, 576]);  bmm_90 = None
    view_929: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_91, [8, 16, 576, 48]);  bmm_91 = None
    permute_719: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_928, [0, 1, 3, 2]);  view_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_15: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_919, 0, 2);  view_919 = None
    select_scatter_16: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_719, 0, 1);  permute_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_382: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_15, select_scatter_16);  select_scatter_15 = select_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_569: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_929, 0.14433756729740643);  view_929 = None
    select_scatter_17: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_569, 0, 0);  mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_383: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_382, select_scatter_17);  add_382 = select_scatter_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_720: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_383, [1, 3, 0, 2, 4]);  add_383 = None
    clone_550: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_720, memory_format = torch.contiguous_format);  permute_720 = None
    view_930: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_550, [8, 576, 2304]);  clone_550 = None
    view_931: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_930, [4608, 2304]);  view_930 = None
    mm_156: "f32[4608, 768]" = torch.ops.aten.mm.default(view_931, permute_721);  permute_721 = None
    permute_722: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_931, [1, 0])
    mm_157: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_722, view_621);  permute_722 = view_621 = None
    permute_723: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_155: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_931, [0], True);  view_931 = None
    view_932: "f32[2304]" = torch.ops.aten.reshape.default(sum_155, [2304]);  sum_155 = None
    permute_724: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_723, [1, 0]);  permute_723 = None
    view_933: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_156, [8, 576, 768]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_571: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_933, primals_577);  primals_577 = None
    mul_572: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_571, 768)
    sum_156: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_571, [2], True)
    mul_573: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_571, mul_310);  mul_571 = None
    sum_157: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_573, [2], True);  mul_573 = None
    mul_574: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_310, sum_157);  sum_157 = None
    sub_161: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_572, sum_156);  mul_572 = sum_156 = None
    sub_162: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_161, mul_574);  sub_161 = mul_574 = None
    mul_575: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_50, sub_162);  div_50 = sub_162 = None
    mul_576: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_933, mul_310);  mul_310 = None
    sum_158: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_576, [0, 1]);  mul_576 = None
    sum_159: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_933, [0, 1]);  view_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_384: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_381, mul_575);  add_381 = mul_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_577: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_384, primals_63);  primals_63 = None
    mul_578: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_384, view_620);  view_620 = None
    sum_160: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_578, [0, 1], True);  mul_578 = None
    view_934: "f32[768]" = torch.ops.aten.reshape.default(sum_160, [768]);  sum_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_935: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_577, [4608, 768]);  mul_577 = None
    mm_158: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_935, permute_725);  permute_725 = None
    permute_726: "f32[768, 4608]" = torch.ops.aten.permute.default(view_935, [1, 0])
    mm_159: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_726, view_619);  permute_726 = view_619 = None
    permute_727: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_161: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_935, [0], True);  view_935 = None
    view_936: "f32[768]" = torch.ops.aten.reshape.default(sum_161, [768]);  sum_161 = None
    permute_728: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_727, [1, 0]);  permute_727 = None
    view_937: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_158, [8, 576, 3072]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_580: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_278, 0.5);  add_278 = None
    mul_581: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_618, view_618)
    mul_582: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_581, -0.5);  mul_581 = None
    exp_43: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_582);  mul_582 = None
    mul_583: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
    mul_584: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_618, mul_583);  view_618 = mul_583 = None
    add_386: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_580, mul_584);  mul_580 = mul_584 = None
    mul_585: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_937, add_386);  view_937 = add_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_938: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_585, [4608, 3072]);  mul_585 = None
    mm_160: "f32[4608, 768]" = torch.ops.aten.mm.default(view_938, permute_729);  permute_729 = None
    permute_730: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_938, [1, 0])
    mm_161: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_730, view_617);  permute_730 = view_617 = None
    permute_731: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    sum_162: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_938, [0], True);  view_938 = None
    view_939: "f32[3072]" = torch.ops.aten.reshape.default(sum_162, [3072]);  sum_162 = None
    permute_732: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_731, [1, 0]);  permute_731 = None
    view_940: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_160, [8, 576, 768]);  mm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_587: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_940, primals_571);  primals_571 = None
    mul_588: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_587, 768)
    sum_163: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_587, [2], True)
    mul_589: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_587, mul_304);  mul_587 = None
    sum_164: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_589, [2], True);  mul_589 = None
    mul_590: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_304, sum_164);  sum_164 = None
    sub_164: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_588, sum_163);  mul_588 = sum_163 = None
    sub_165: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_164, mul_590);  sub_164 = mul_590 = None
    mul_591: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_165);  div_51 = sub_165 = None
    mul_592: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_940, mul_304);  mul_304 = None
    sum_165: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_592, [0, 1]);  mul_592 = None
    sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_940, [0, 1]);  view_940 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_387: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_384, mul_591);  add_384 = mul_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_593: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_387, primals_62);  primals_62 = None
    mul_594: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_387, view_616);  view_616 = None
    sum_167: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_594, [0, 1], True);  mul_594 = None
    view_941: "f32[768]" = torch.ops.aten.reshape.default(sum_167, [768]);  sum_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_942: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_593, [4608, 768]);  mul_593 = None
    mm_162: "f32[4608, 768]" = torch.ops.aten.mm.default(view_942, permute_733);  permute_733 = None
    permute_734: "f32[768, 4608]" = torch.ops.aten.permute.default(view_942, [1, 0])
    mm_163: "f32[768, 768]" = torch.ops.aten.mm.default(permute_734, view_615);  permute_734 = view_615 = None
    permute_735: "f32[768, 768]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_168: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_942, [0], True);  view_942 = None
    view_943: "f32[768]" = torch.ops.aten.reshape.default(sum_168, [768]);  sum_168 = None
    permute_736: "f32[768, 768]" = torch.ops.aten.permute.default(permute_735, [1, 0]);  permute_735 = None
    view_944: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_162, [8, 576, 768]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_945: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_944, [8, 576, 16, 48]);  view_944 = None
    permute_737: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_945, [0, 2, 1, 3]);  view_945 = None
    clone_553: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_737, memory_format = torch.contiguous_format);  permute_737 = None
    view_946: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_553, [128, 576, 48]);  clone_553 = None
    bmm_92: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_738, view_946);  permute_738 = None
    bmm_93: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_946, permute_739);  view_946 = permute_739 = None
    view_947: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_92, [8, 16, 576, 48]);  bmm_92 = None
    view_948: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_93, [8, 16, 576, 576]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_740: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_948, [0, 2, 3, 1]);  view_948 = None
    sum_169: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_740, [0, 1, 2], True)
    view_949: "f32[16]" = torch.ops.aten.reshape.default(sum_169, [16]);  sum_169 = None
    clone_554: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_740, memory_format = torch.contiguous_format);  permute_740 = None
    view_950: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_554, [2654208, 16]);  clone_554 = None
    permute_741: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_950, [1, 0])
    mm_164: "f32[16, 16]" = torch.ops.aten.mm.default(permute_741, view_609);  permute_741 = view_609 = None
    permute_742: "f32[16, 16]" = torch.ops.aten.permute.default(mm_164, [1, 0]);  mm_164 = None
    mm_165: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_950, permute_743);  view_950 = permute_743 = None
    view_951: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_165, [8, 576, 576, 16]);  mm_165 = None
    permute_744: "f32[16, 16]" = torch.ops.aten.permute.default(permute_742, [1, 0]);  permute_742 = None
    permute_745: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_951, [0, 3, 1, 2]);  view_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_595: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_745, alias_45);  permute_745 = None
    sum_170: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_595, [-1], True)
    mul_596: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_45, sum_170);  alias_45 = sum_170 = None
    sub_166: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_595, mul_596);  mul_595 = mul_596 = None
    clone_555: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_166, memory_format = torch.contiguous_format);  sub_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_746: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_555, [0, 2, 3, 1]);  clone_555 = None
    sum_171: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_746, [0, 1, 2], True)
    view_952: "f32[16]" = torch.ops.aten.reshape.default(sum_171, [16]);  sum_171 = None
    clone_556: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_746, memory_format = torch.contiguous_format);  permute_746 = None
    view_953: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_556, [2654208, 16]);  clone_556 = None
    permute_747: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_953, [1, 0])
    mm_166: "f32[16, 16]" = torch.ops.aten.mm.default(permute_747, view_607);  permute_747 = view_607 = None
    permute_748: "f32[16, 16]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    mm_167: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_953, permute_749);  view_953 = permute_749 = None
    view_954: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_167, [8, 576, 576, 16]);  mm_167 = None
    permute_750: "f32[16, 16]" = torch.ops.aten.permute.default(permute_748, [1, 0]);  permute_748 = None
    permute_751: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_954, [0, 3, 1, 2]);  view_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_557: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_751, memory_format = torch.contiguous_format);  permute_751 = None
    view_955: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_557, [128, 576, 576]);  clone_557 = None
    bmm_94: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_752, view_955);  permute_752 = None
    bmm_95: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_955, permute_753);  view_955 = permute_753 = None
    view_956: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_94, [8, 16, 48, 576]);  bmm_94 = None
    view_957: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_95, [8, 16, 576, 48]);  bmm_95 = None
    permute_754: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_956, [0, 1, 3, 2]);  view_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_18: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_947, 0, 2);  view_947 = None
    select_scatter_19: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_754, 0, 1);  permute_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_388: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_18, select_scatter_19);  select_scatter_18 = select_scatter_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_597: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_957, 0.14433756729740643);  view_957 = None
    select_scatter_20: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_597, 0, 0);  mul_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_389: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_388, select_scatter_20);  add_388 = select_scatter_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_755: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_389, [1, 3, 0, 2, 4]);  add_389 = None
    clone_558: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_755, memory_format = torch.contiguous_format);  permute_755 = None
    view_958: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_558, [8, 576, 2304]);  clone_558 = None
    view_959: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_958, [4608, 2304]);  view_958 = None
    mm_168: "f32[4608, 768]" = torch.ops.aten.mm.default(view_959, permute_756);  permute_756 = None
    permute_757: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_959, [1, 0])
    mm_169: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_757, view_601);  permute_757 = view_601 = None
    permute_758: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    sum_172: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_959, [0], True);  view_959 = None
    view_960: "f32[2304]" = torch.ops.aten.reshape.default(sum_172, [2304]);  sum_172 = None
    permute_759: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_758, [1, 0]);  permute_758 = None
    view_961: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_168, [8, 576, 768]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_599: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_961, primals_561);  primals_561 = None
    mul_600: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_599, 768)
    sum_173: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_599, [2], True)
    mul_601: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_599, mul_300);  mul_599 = None
    sum_174: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_601, [2], True);  mul_601 = None
    mul_602: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_300, sum_174);  sum_174 = None
    sub_168: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_600, sum_173);  mul_600 = sum_173 = None
    sub_169: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_168, mul_602);  sub_168 = mul_602 = None
    mul_603: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_169);  div_52 = sub_169 = None
    mul_604: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_961, mul_300);  mul_300 = None
    sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_604, [0, 1]);  mul_604 = None
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_961, [0, 1]);  view_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_390: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_387, mul_603);  add_387 = mul_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_605: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_390, primals_61);  primals_61 = None
    mul_606: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_390, view_600);  view_600 = None
    sum_177: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_606, [0, 1], True);  mul_606 = None
    view_962: "f32[768]" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_963: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_605, [4608, 768]);  mul_605 = None
    mm_170: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_963, permute_760);  permute_760 = None
    permute_761: "f32[768, 4608]" = torch.ops.aten.permute.default(view_963, [1, 0])
    mm_171: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_761, view_599);  permute_761 = view_599 = None
    permute_762: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_178: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_963, [0], True);  view_963 = None
    view_964: "f32[768]" = torch.ops.aten.reshape.default(sum_178, [768]);  sum_178 = None
    permute_763: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_762, [1, 0]);  permute_762 = None
    view_965: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_170, [8, 576, 3072]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_608: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_269, 0.5);  add_269 = None
    mul_609: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_598, view_598)
    mul_610: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_609, -0.5);  mul_609 = None
    exp_44: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_610);  mul_610 = None
    mul_611: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
    mul_612: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_598, mul_611);  view_598 = mul_611 = None
    add_392: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_608, mul_612);  mul_608 = mul_612 = None
    mul_613: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_965, add_392);  view_965 = add_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_966: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_613, [4608, 3072]);  mul_613 = None
    mm_172: "f32[4608, 768]" = torch.ops.aten.mm.default(view_966, permute_764);  permute_764 = None
    permute_765: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_966, [1, 0])
    mm_173: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_765, view_597);  permute_765 = view_597 = None
    permute_766: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_179: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_966, [0], True);  view_966 = None
    view_967: "f32[3072]" = torch.ops.aten.reshape.default(sum_179, [3072]);  sum_179 = None
    permute_767: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_766, [1, 0]);  permute_766 = None
    view_968: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_172, [8, 576, 768]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_615: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_968, primals_555);  primals_555 = None
    mul_616: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_615, 768)
    sum_180: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_615, [2], True)
    mul_617: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_615, mul_294);  mul_615 = None
    sum_181: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_617, [2], True);  mul_617 = None
    mul_618: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_294, sum_181);  sum_181 = None
    sub_171: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_616, sum_180);  mul_616 = sum_180 = None
    sub_172: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_171, mul_618);  sub_171 = mul_618 = None
    mul_619: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_53, sub_172);  div_53 = sub_172 = None
    mul_620: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_968, mul_294);  mul_294 = None
    sum_182: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 1]);  mul_620 = None
    sum_183: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_968, [0, 1]);  view_968 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_393: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_390, mul_619);  add_390 = mul_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_621: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_393, primals_60);  primals_60 = None
    mul_622: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_393, view_596);  view_596 = None
    sum_184: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_622, [0, 1], True);  mul_622 = None
    view_969: "f32[768]" = torch.ops.aten.reshape.default(sum_184, [768]);  sum_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_970: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_621, [4608, 768]);  mul_621 = None
    mm_174: "f32[4608, 768]" = torch.ops.aten.mm.default(view_970, permute_768);  permute_768 = None
    permute_769: "f32[768, 4608]" = torch.ops.aten.permute.default(view_970, [1, 0])
    mm_175: "f32[768, 768]" = torch.ops.aten.mm.default(permute_769, view_595);  permute_769 = view_595 = None
    permute_770: "f32[768, 768]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_185: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_970, [0], True);  view_970 = None
    view_971: "f32[768]" = torch.ops.aten.reshape.default(sum_185, [768]);  sum_185 = None
    permute_771: "f32[768, 768]" = torch.ops.aten.permute.default(permute_770, [1, 0]);  permute_770 = None
    view_972: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_174, [8, 576, 768]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_973: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_972, [8, 576, 16, 48]);  view_972 = None
    permute_772: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_973, [0, 2, 1, 3]);  view_973 = None
    clone_561: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_772, memory_format = torch.contiguous_format);  permute_772 = None
    view_974: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_561, [128, 576, 48]);  clone_561 = None
    bmm_96: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_773, view_974);  permute_773 = None
    bmm_97: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_974, permute_774);  view_974 = permute_774 = None
    view_975: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_96, [8, 16, 576, 48]);  bmm_96 = None
    view_976: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_97, [8, 16, 576, 576]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_775: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_976, [0, 2, 3, 1]);  view_976 = None
    sum_186: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_775, [0, 1, 2], True)
    view_977: "f32[16]" = torch.ops.aten.reshape.default(sum_186, [16]);  sum_186 = None
    clone_562: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_775, memory_format = torch.contiguous_format);  permute_775 = None
    view_978: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_562, [2654208, 16]);  clone_562 = None
    permute_776: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_978, [1, 0])
    mm_176: "f32[16, 16]" = torch.ops.aten.mm.default(permute_776, view_589);  permute_776 = view_589 = None
    permute_777: "f32[16, 16]" = torch.ops.aten.permute.default(mm_176, [1, 0]);  mm_176 = None
    mm_177: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_978, permute_778);  view_978 = permute_778 = None
    view_979: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_177, [8, 576, 576, 16]);  mm_177 = None
    permute_779: "f32[16, 16]" = torch.ops.aten.permute.default(permute_777, [1, 0]);  permute_777 = None
    permute_780: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_979, [0, 3, 1, 2]);  view_979 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_623: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_780, alias_46);  permute_780 = None
    sum_187: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_623, [-1], True)
    mul_624: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_46, sum_187);  alias_46 = sum_187 = None
    sub_173: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_623, mul_624);  mul_623 = mul_624 = None
    clone_563: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_173, memory_format = torch.contiguous_format);  sub_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_781: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_563, [0, 2, 3, 1]);  clone_563 = None
    sum_188: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_781, [0, 1, 2], True)
    view_980: "f32[16]" = torch.ops.aten.reshape.default(sum_188, [16]);  sum_188 = None
    clone_564: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_781, memory_format = torch.contiguous_format);  permute_781 = None
    view_981: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_564, [2654208, 16]);  clone_564 = None
    permute_782: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_981, [1, 0])
    mm_178: "f32[16, 16]" = torch.ops.aten.mm.default(permute_782, view_587);  permute_782 = view_587 = None
    permute_783: "f32[16, 16]" = torch.ops.aten.permute.default(mm_178, [1, 0]);  mm_178 = None
    mm_179: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_981, permute_784);  view_981 = permute_784 = None
    view_982: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_179, [8, 576, 576, 16]);  mm_179 = None
    permute_785: "f32[16, 16]" = torch.ops.aten.permute.default(permute_783, [1, 0]);  permute_783 = None
    permute_786: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_982, [0, 3, 1, 2]);  view_982 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_565: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_786, memory_format = torch.contiguous_format);  permute_786 = None
    view_983: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_565, [128, 576, 576]);  clone_565 = None
    bmm_98: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_787, view_983);  permute_787 = None
    bmm_99: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_983, permute_788);  view_983 = permute_788 = None
    view_984: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_98, [8, 16, 48, 576]);  bmm_98 = None
    view_985: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_99, [8, 16, 576, 48]);  bmm_99 = None
    permute_789: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_984, [0, 1, 3, 2]);  view_984 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_21: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_975, 0, 2);  view_975 = None
    select_scatter_22: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_789, 0, 1);  permute_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_394: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_21, select_scatter_22);  select_scatter_21 = select_scatter_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_625: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_985, 0.14433756729740643);  view_985 = None
    select_scatter_23: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_625, 0, 0);  mul_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_395: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_394, select_scatter_23);  add_394 = select_scatter_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_790: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_395, [1, 3, 0, 2, 4]);  add_395 = None
    clone_566: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_790, memory_format = torch.contiguous_format);  permute_790 = None
    view_986: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_566, [8, 576, 2304]);  clone_566 = None
    view_987: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_986, [4608, 2304]);  view_986 = None
    mm_180: "f32[4608, 768]" = torch.ops.aten.mm.default(view_987, permute_791);  permute_791 = None
    permute_792: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_987, [1, 0])
    mm_181: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_792, view_581);  permute_792 = view_581 = None
    permute_793: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_189: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_987, [0], True);  view_987 = None
    view_988: "f32[2304]" = torch.ops.aten.reshape.default(sum_189, [2304]);  sum_189 = None
    permute_794: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_793, [1, 0]);  permute_793 = None
    view_989: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_180, [8, 576, 768]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_627: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_989, primals_545);  primals_545 = None
    mul_628: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_627, 768)
    sum_190: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_627, [2], True)
    mul_629: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_627, mul_290);  mul_627 = None
    sum_191: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_629, [2], True);  mul_629 = None
    mul_630: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_290, sum_191);  sum_191 = None
    sub_175: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_628, sum_190);  mul_628 = sum_190 = None
    sub_176: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_175, mul_630);  sub_175 = mul_630 = None
    mul_631: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_176);  div_54 = sub_176 = None
    mul_632: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_989, mul_290);  mul_290 = None
    sum_192: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_632, [0, 1]);  mul_632 = None
    sum_193: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_989, [0, 1]);  view_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_396: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_393, mul_631);  add_393 = mul_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_633: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_396, primals_59);  primals_59 = None
    mul_634: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_396, view_580);  view_580 = None
    sum_194: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_634, [0, 1], True);  mul_634 = None
    view_990: "f32[768]" = torch.ops.aten.reshape.default(sum_194, [768]);  sum_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_991: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_633, [4608, 768]);  mul_633 = None
    mm_182: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_991, permute_795);  permute_795 = None
    permute_796: "f32[768, 4608]" = torch.ops.aten.permute.default(view_991, [1, 0])
    mm_183: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_796, view_579);  permute_796 = view_579 = None
    permute_797: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_195: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_991, [0], True);  view_991 = None
    view_992: "f32[768]" = torch.ops.aten.reshape.default(sum_195, [768]);  sum_195 = None
    permute_798: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_797, [1, 0]);  permute_797 = None
    view_993: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_182, [8, 576, 3072]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_636: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_260, 0.5);  add_260 = None
    mul_637: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_578, view_578)
    mul_638: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_637, -0.5);  mul_637 = None
    exp_45: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_638);  mul_638 = None
    mul_639: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
    mul_640: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_578, mul_639);  view_578 = mul_639 = None
    add_398: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_636, mul_640);  mul_636 = mul_640 = None
    mul_641: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_993, add_398);  view_993 = add_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_994: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_641, [4608, 3072]);  mul_641 = None
    mm_184: "f32[4608, 768]" = torch.ops.aten.mm.default(view_994, permute_799);  permute_799 = None
    permute_800: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_994, [1, 0])
    mm_185: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_800, view_577);  permute_800 = view_577 = None
    permute_801: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    sum_196: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_994, [0], True);  view_994 = None
    view_995: "f32[3072]" = torch.ops.aten.reshape.default(sum_196, [3072]);  sum_196 = None
    permute_802: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_801, [1, 0]);  permute_801 = None
    view_996: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_184, [8, 576, 768]);  mm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_643: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_996, primals_539);  primals_539 = None
    mul_644: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_643, 768)
    sum_197: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_643, [2], True)
    mul_645: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_643, mul_284);  mul_643 = None
    sum_198: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_645, [2], True);  mul_645 = None
    mul_646: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_284, sum_198);  sum_198 = None
    sub_178: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_644, sum_197);  mul_644 = sum_197 = None
    sub_179: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_178, mul_646);  sub_178 = mul_646 = None
    mul_647: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_179);  div_55 = sub_179 = None
    mul_648: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_996, mul_284);  mul_284 = None
    sum_199: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_648, [0, 1]);  mul_648 = None
    sum_200: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_996, [0, 1]);  view_996 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_399: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_396, mul_647);  add_396 = mul_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_649: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_399, primals_58);  primals_58 = None
    mul_650: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_399, view_576);  view_576 = None
    sum_201: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 1], True);  mul_650 = None
    view_997: "f32[768]" = torch.ops.aten.reshape.default(sum_201, [768]);  sum_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_998: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_649, [4608, 768]);  mul_649 = None
    mm_186: "f32[4608, 768]" = torch.ops.aten.mm.default(view_998, permute_803);  permute_803 = None
    permute_804: "f32[768, 4608]" = torch.ops.aten.permute.default(view_998, [1, 0])
    mm_187: "f32[768, 768]" = torch.ops.aten.mm.default(permute_804, view_575);  permute_804 = view_575 = None
    permute_805: "f32[768, 768]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_202: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_998, [0], True);  view_998 = None
    view_999: "f32[768]" = torch.ops.aten.reshape.default(sum_202, [768]);  sum_202 = None
    permute_806: "f32[768, 768]" = torch.ops.aten.permute.default(permute_805, [1, 0]);  permute_805 = None
    view_1000: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_186, [8, 576, 768]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1001: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1000, [8, 576, 16, 48]);  view_1000 = None
    permute_807: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1001, [0, 2, 1, 3]);  view_1001 = None
    clone_569: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_807, memory_format = torch.contiguous_format);  permute_807 = None
    view_1002: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_569, [128, 576, 48]);  clone_569 = None
    bmm_100: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_808, view_1002);  permute_808 = None
    bmm_101: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1002, permute_809);  view_1002 = permute_809 = None
    view_1003: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_100, [8, 16, 576, 48]);  bmm_100 = None
    view_1004: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_101, [8, 16, 576, 576]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_810: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1004, [0, 2, 3, 1]);  view_1004 = None
    sum_203: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_810, [0, 1, 2], True)
    view_1005: "f32[16]" = torch.ops.aten.reshape.default(sum_203, [16]);  sum_203 = None
    clone_570: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_810, memory_format = torch.contiguous_format);  permute_810 = None
    view_1006: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_570, [2654208, 16]);  clone_570 = None
    permute_811: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1006, [1, 0])
    mm_188: "f32[16, 16]" = torch.ops.aten.mm.default(permute_811, view_569);  permute_811 = view_569 = None
    permute_812: "f32[16, 16]" = torch.ops.aten.permute.default(mm_188, [1, 0]);  mm_188 = None
    mm_189: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1006, permute_813);  view_1006 = permute_813 = None
    view_1007: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_189, [8, 576, 576, 16]);  mm_189 = None
    permute_814: "f32[16, 16]" = torch.ops.aten.permute.default(permute_812, [1, 0]);  permute_812 = None
    permute_815: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1007, [0, 3, 1, 2]);  view_1007 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_651: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_815, alias_47);  permute_815 = None
    sum_204: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_651, [-1], True)
    mul_652: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_47, sum_204);  alias_47 = sum_204 = None
    sub_180: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_651, mul_652);  mul_651 = mul_652 = None
    clone_571: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_180, memory_format = torch.contiguous_format);  sub_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_816: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_571, [0, 2, 3, 1]);  clone_571 = None
    sum_205: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_816, [0, 1, 2], True)
    view_1008: "f32[16]" = torch.ops.aten.reshape.default(sum_205, [16]);  sum_205 = None
    clone_572: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_816, memory_format = torch.contiguous_format);  permute_816 = None
    view_1009: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_572, [2654208, 16]);  clone_572 = None
    permute_817: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1009, [1, 0])
    mm_190: "f32[16, 16]" = torch.ops.aten.mm.default(permute_817, view_567);  permute_817 = view_567 = None
    permute_818: "f32[16, 16]" = torch.ops.aten.permute.default(mm_190, [1, 0]);  mm_190 = None
    mm_191: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1009, permute_819);  view_1009 = permute_819 = None
    view_1010: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_191, [8, 576, 576, 16]);  mm_191 = None
    permute_820: "f32[16, 16]" = torch.ops.aten.permute.default(permute_818, [1, 0]);  permute_818 = None
    permute_821: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1010, [0, 3, 1, 2]);  view_1010 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_573: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_821, memory_format = torch.contiguous_format);  permute_821 = None
    view_1011: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_573, [128, 576, 576]);  clone_573 = None
    bmm_102: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_822, view_1011);  permute_822 = None
    bmm_103: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1011, permute_823);  view_1011 = permute_823 = None
    view_1012: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_102, [8, 16, 48, 576]);  bmm_102 = None
    view_1013: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_103, [8, 16, 576, 48]);  bmm_103 = None
    permute_824: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1012, [0, 1, 3, 2]);  view_1012 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_24: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1003, 0, 2);  view_1003 = None
    select_scatter_25: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_824, 0, 1);  permute_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_400: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_24, select_scatter_25);  select_scatter_24 = select_scatter_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_653: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1013, 0.14433756729740643);  view_1013 = None
    select_scatter_26: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_653, 0, 0);  mul_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_401: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_400, select_scatter_26);  add_400 = select_scatter_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_825: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_401, [1, 3, 0, 2, 4]);  add_401 = None
    clone_574: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_825, memory_format = torch.contiguous_format);  permute_825 = None
    view_1014: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_574, [8, 576, 2304]);  clone_574 = None
    view_1015: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1014, [4608, 2304]);  view_1014 = None
    mm_192: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1015, permute_826);  permute_826 = None
    permute_827: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1015, [1, 0])
    mm_193: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_827, view_561);  permute_827 = view_561 = None
    permute_828: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_206: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1015, [0], True);  view_1015 = None
    view_1016: "f32[2304]" = torch.ops.aten.reshape.default(sum_206, [2304]);  sum_206 = None
    permute_829: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_828, [1, 0]);  permute_828 = None
    view_1017: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_192, [8, 576, 768]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_655: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1017, primals_529);  primals_529 = None
    mul_656: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_655, 768)
    sum_207: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_655, [2], True)
    mul_657: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_655, mul_280);  mul_655 = None
    sum_208: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_657, [2], True);  mul_657 = None
    mul_658: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_280, sum_208);  sum_208 = None
    sub_182: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_656, sum_207);  mul_656 = sum_207 = None
    sub_183: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_182, mul_658);  sub_182 = mul_658 = None
    mul_659: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_56, sub_183);  div_56 = sub_183 = None
    mul_660: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1017, mul_280);  mul_280 = None
    sum_209: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_660, [0, 1]);  mul_660 = None
    sum_210: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1017, [0, 1]);  view_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_402: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_399, mul_659);  add_399 = mul_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_661: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_402, primals_57);  primals_57 = None
    mul_662: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_402, view_560);  view_560 = None
    sum_211: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 1], True);  mul_662 = None
    view_1018: "f32[768]" = torch.ops.aten.reshape.default(sum_211, [768]);  sum_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1019: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_661, [4608, 768]);  mul_661 = None
    mm_194: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1019, permute_830);  permute_830 = None
    permute_831: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1019, [1, 0])
    mm_195: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_831, view_559);  permute_831 = view_559 = None
    permute_832: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_212: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1019, [0], True);  view_1019 = None
    view_1020: "f32[768]" = torch.ops.aten.reshape.default(sum_212, [768]);  sum_212 = None
    permute_833: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_832, [1, 0]);  permute_832 = None
    view_1021: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_194, [8, 576, 3072]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_664: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_251, 0.5);  add_251 = None
    mul_665: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_558, view_558)
    mul_666: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_665, -0.5);  mul_665 = None
    exp_46: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_666);  mul_666 = None
    mul_667: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
    mul_668: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_558, mul_667);  view_558 = mul_667 = None
    add_404: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_664, mul_668);  mul_664 = mul_668 = None
    mul_669: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1021, add_404);  view_1021 = add_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1022: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_669, [4608, 3072]);  mul_669 = None
    mm_196: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1022, permute_834);  permute_834 = None
    permute_835: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1022, [1, 0])
    mm_197: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_835, view_557);  permute_835 = view_557 = None
    permute_836: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    sum_213: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1022, [0], True);  view_1022 = None
    view_1023: "f32[3072]" = torch.ops.aten.reshape.default(sum_213, [3072]);  sum_213 = None
    permute_837: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_836, [1, 0]);  permute_836 = None
    view_1024: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_196, [8, 576, 768]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_671: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1024, primals_523);  primals_523 = None
    mul_672: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_671, 768)
    sum_214: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_671, [2], True)
    mul_673: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_671, mul_274);  mul_671 = None
    sum_215: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_673, [2], True);  mul_673 = None
    mul_674: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_274, sum_215);  sum_215 = None
    sub_185: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_672, sum_214);  mul_672 = sum_214 = None
    sub_186: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_185, mul_674);  sub_185 = mul_674 = None
    mul_675: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_186);  div_57 = sub_186 = None
    mul_676: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1024, mul_274);  mul_274 = None
    sum_216: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_676, [0, 1]);  mul_676 = None
    sum_217: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1024, [0, 1]);  view_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_405: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_402, mul_675);  add_402 = mul_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_677: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_405, primals_56);  primals_56 = None
    mul_678: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_405, view_556);  view_556 = None
    sum_218: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_678, [0, 1], True);  mul_678 = None
    view_1025: "f32[768]" = torch.ops.aten.reshape.default(sum_218, [768]);  sum_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1026: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_677, [4608, 768]);  mul_677 = None
    mm_198: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1026, permute_838);  permute_838 = None
    permute_839: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1026, [1, 0])
    mm_199: "f32[768, 768]" = torch.ops.aten.mm.default(permute_839, view_555);  permute_839 = view_555 = None
    permute_840: "f32[768, 768]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_219: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1026, [0], True);  view_1026 = None
    view_1027: "f32[768]" = torch.ops.aten.reshape.default(sum_219, [768]);  sum_219 = None
    permute_841: "f32[768, 768]" = torch.ops.aten.permute.default(permute_840, [1, 0]);  permute_840 = None
    view_1028: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_198, [8, 576, 768]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1029: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1028, [8, 576, 16, 48]);  view_1028 = None
    permute_842: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1029, [0, 2, 1, 3]);  view_1029 = None
    clone_577: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_842, memory_format = torch.contiguous_format);  permute_842 = None
    view_1030: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_577, [128, 576, 48]);  clone_577 = None
    bmm_104: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_843, view_1030);  permute_843 = None
    bmm_105: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1030, permute_844);  view_1030 = permute_844 = None
    view_1031: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_104, [8, 16, 576, 48]);  bmm_104 = None
    view_1032: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_105, [8, 16, 576, 576]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_845: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1032, [0, 2, 3, 1]);  view_1032 = None
    sum_220: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_845, [0, 1, 2], True)
    view_1033: "f32[16]" = torch.ops.aten.reshape.default(sum_220, [16]);  sum_220 = None
    clone_578: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_845, memory_format = torch.contiguous_format);  permute_845 = None
    view_1034: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_578, [2654208, 16]);  clone_578 = None
    permute_846: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1034, [1, 0])
    mm_200: "f32[16, 16]" = torch.ops.aten.mm.default(permute_846, view_549);  permute_846 = view_549 = None
    permute_847: "f32[16, 16]" = torch.ops.aten.permute.default(mm_200, [1, 0]);  mm_200 = None
    mm_201: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1034, permute_848);  view_1034 = permute_848 = None
    view_1035: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_201, [8, 576, 576, 16]);  mm_201 = None
    permute_849: "f32[16, 16]" = torch.ops.aten.permute.default(permute_847, [1, 0]);  permute_847 = None
    permute_850: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1035, [0, 3, 1, 2]);  view_1035 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_679: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_850, alias_48);  permute_850 = None
    sum_221: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_679, [-1], True)
    mul_680: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_48, sum_221);  alias_48 = sum_221 = None
    sub_187: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_679, mul_680);  mul_679 = mul_680 = None
    clone_579: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_187, memory_format = torch.contiguous_format);  sub_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_851: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_579, [0, 2, 3, 1]);  clone_579 = None
    sum_222: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_851, [0, 1, 2], True)
    view_1036: "f32[16]" = torch.ops.aten.reshape.default(sum_222, [16]);  sum_222 = None
    clone_580: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_851, memory_format = torch.contiguous_format);  permute_851 = None
    view_1037: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_580, [2654208, 16]);  clone_580 = None
    permute_852: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1037, [1, 0])
    mm_202: "f32[16, 16]" = torch.ops.aten.mm.default(permute_852, view_547);  permute_852 = view_547 = None
    permute_853: "f32[16, 16]" = torch.ops.aten.permute.default(mm_202, [1, 0]);  mm_202 = None
    mm_203: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1037, permute_854);  view_1037 = permute_854 = None
    view_1038: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_203, [8, 576, 576, 16]);  mm_203 = None
    permute_855: "f32[16, 16]" = torch.ops.aten.permute.default(permute_853, [1, 0]);  permute_853 = None
    permute_856: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1038, [0, 3, 1, 2]);  view_1038 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_581: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_856, memory_format = torch.contiguous_format);  permute_856 = None
    view_1039: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_581, [128, 576, 576]);  clone_581 = None
    bmm_106: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_857, view_1039);  permute_857 = None
    bmm_107: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1039, permute_858);  view_1039 = permute_858 = None
    view_1040: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_106, [8, 16, 48, 576]);  bmm_106 = None
    view_1041: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_107, [8, 16, 576, 48]);  bmm_107 = None
    permute_859: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1040, [0, 1, 3, 2]);  view_1040 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_27: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1031, 0, 2);  view_1031 = None
    select_scatter_28: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_859, 0, 1);  permute_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_406: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_27, select_scatter_28);  select_scatter_27 = select_scatter_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_681: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1041, 0.14433756729740643);  view_1041 = None
    select_scatter_29: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_681, 0, 0);  mul_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_407: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_406, select_scatter_29);  add_406 = select_scatter_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_860: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_407, [1, 3, 0, 2, 4]);  add_407 = None
    clone_582: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_860, memory_format = torch.contiguous_format);  permute_860 = None
    view_1042: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_582, [8, 576, 2304]);  clone_582 = None
    view_1043: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1042, [4608, 2304]);  view_1042 = None
    mm_204: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1043, permute_861);  permute_861 = None
    permute_862: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1043, [1, 0])
    mm_205: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_862, view_541);  permute_862 = view_541 = None
    permute_863: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_223: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1043, [0], True);  view_1043 = None
    view_1044: "f32[2304]" = torch.ops.aten.reshape.default(sum_223, [2304]);  sum_223 = None
    permute_864: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_863, [1, 0]);  permute_863 = None
    view_1045: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_204, [8, 576, 768]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_683: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1045, primals_513);  primals_513 = None
    mul_684: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_683, 768)
    sum_224: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_683, [2], True)
    mul_685: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_683, mul_270);  mul_683 = None
    sum_225: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_685, [2], True);  mul_685 = None
    mul_686: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_270, sum_225);  sum_225 = None
    sub_189: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_684, sum_224);  mul_684 = sum_224 = None
    sub_190: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_189, mul_686);  sub_189 = mul_686 = None
    mul_687: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_190);  div_58 = sub_190 = None
    mul_688: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1045, mul_270);  mul_270 = None
    sum_226: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_688, [0, 1]);  mul_688 = None
    sum_227: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1045, [0, 1]);  view_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_408: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_405, mul_687);  add_405 = mul_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_689: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_408, primals_55);  primals_55 = None
    mul_690: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_408, view_540);  view_540 = None
    sum_228: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_690, [0, 1], True);  mul_690 = None
    view_1046: "f32[768]" = torch.ops.aten.reshape.default(sum_228, [768]);  sum_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1047: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_689, [4608, 768]);  mul_689 = None
    mm_206: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1047, permute_865);  permute_865 = None
    permute_866: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1047, [1, 0])
    mm_207: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_866, view_539);  permute_866 = view_539 = None
    permute_867: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    sum_229: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1047, [0], True);  view_1047 = None
    view_1048: "f32[768]" = torch.ops.aten.reshape.default(sum_229, [768]);  sum_229 = None
    permute_868: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_867, [1, 0]);  permute_867 = None
    view_1049: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_206, [8, 576, 3072]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_692: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_242, 0.5);  add_242 = None
    mul_693: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_538, view_538)
    mul_694: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_693, -0.5);  mul_693 = None
    exp_47: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_694);  mul_694 = None
    mul_695: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
    mul_696: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_538, mul_695);  view_538 = mul_695 = None
    add_410: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_692, mul_696);  mul_692 = mul_696 = None
    mul_697: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1049, add_410);  view_1049 = add_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1050: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_697, [4608, 3072]);  mul_697 = None
    mm_208: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1050, permute_869);  permute_869 = None
    permute_870: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1050, [1, 0])
    mm_209: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_870, view_537);  permute_870 = view_537 = None
    permute_871: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
    sum_230: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1050, [0], True);  view_1050 = None
    view_1051: "f32[3072]" = torch.ops.aten.reshape.default(sum_230, [3072]);  sum_230 = None
    permute_872: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_871, [1, 0]);  permute_871 = None
    view_1052: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_208, [8, 576, 768]);  mm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_699: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1052, primals_507);  primals_507 = None
    mul_700: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_699, 768)
    sum_231: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_699, [2], True)
    mul_701: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_699, mul_264);  mul_699 = None
    sum_232: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_701, [2], True);  mul_701 = None
    mul_702: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_264, sum_232);  sum_232 = None
    sub_192: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_700, sum_231);  mul_700 = sum_231 = None
    sub_193: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_192, mul_702);  sub_192 = mul_702 = None
    mul_703: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_59, sub_193);  div_59 = sub_193 = None
    mul_704: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1052, mul_264);  mul_264 = None
    sum_233: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 1]);  mul_704 = None
    sum_234: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1052, [0, 1]);  view_1052 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_411: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_408, mul_703);  add_408 = mul_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_705: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_411, primals_54);  primals_54 = None
    mul_706: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_411, view_536);  view_536 = None
    sum_235: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 1], True);  mul_706 = None
    view_1053: "f32[768]" = torch.ops.aten.reshape.default(sum_235, [768]);  sum_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1054: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_705, [4608, 768]);  mul_705 = None
    mm_210: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1054, permute_873);  permute_873 = None
    permute_874: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1054, [1, 0])
    mm_211: "f32[768, 768]" = torch.ops.aten.mm.default(permute_874, view_535);  permute_874 = view_535 = None
    permute_875: "f32[768, 768]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    sum_236: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1054, [0], True);  view_1054 = None
    view_1055: "f32[768]" = torch.ops.aten.reshape.default(sum_236, [768]);  sum_236 = None
    permute_876: "f32[768, 768]" = torch.ops.aten.permute.default(permute_875, [1, 0]);  permute_875 = None
    view_1056: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_210, [8, 576, 768]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1057: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1056, [8, 576, 16, 48]);  view_1056 = None
    permute_877: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1057, [0, 2, 1, 3]);  view_1057 = None
    clone_585: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_877, memory_format = torch.contiguous_format);  permute_877 = None
    view_1058: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_585, [128, 576, 48]);  clone_585 = None
    bmm_108: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_878, view_1058);  permute_878 = None
    bmm_109: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1058, permute_879);  view_1058 = permute_879 = None
    view_1059: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_108, [8, 16, 576, 48]);  bmm_108 = None
    view_1060: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_109, [8, 16, 576, 576]);  bmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_880: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1060, [0, 2, 3, 1]);  view_1060 = None
    sum_237: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_880, [0, 1, 2], True)
    view_1061: "f32[16]" = torch.ops.aten.reshape.default(sum_237, [16]);  sum_237 = None
    clone_586: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_880, memory_format = torch.contiguous_format);  permute_880 = None
    view_1062: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_586, [2654208, 16]);  clone_586 = None
    permute_881: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1062, [1, 0])
    mm_212: "f32[16, 16]" = torch.ops.aten.mm.default(permute_881, view_529);  permute_881 = view_529 = None
    permute_882: "f32[16, 16]" = torch.ops.aten.permute.default(mm_212, [1, 0]);  mm_212 = None
    mm_213: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1062, permute_883);  view_1062 = permute_883 = None
    view_1063: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_213, [8, 576, 576, 16]);  mm_213 = None
    permute_884: "f32[16, 16]" = torch.ops.aten.permute.default(permute_882, [1, 0]);  permute_882 = None
    permute_885: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1063, [0, 3, 1, 2]);  view_1063 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_707: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_885, alias_49);  permute_885 = None
    sum_238: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_707, [-1], True)
    mul_708: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_49, sum_238);  alias_49 = sum_238 = None
    sub_194: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_707, mul_708);  mul_707 = mul_708 = None
    clone_587: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_194, memory_format = torch.contiguous_format);  sub_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_886: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_587, [0, 2, 3, 1]);  clone_587 = None
    sum_239: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_886, [0, 1, 2], True)
    view_1064: "f32[16]" = torch.ops.aten.reshape.default(sum_239, [16]);  sum_239 = None
    clone_588: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_886, memory_format = torch.contiguous_format);  permute_886 = None
    view_1065: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_588, [2654208, 16]);  clone_588 = None
    permute_887: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1065, [1, 0])
    mm_214: "f32[16, 16]" = torch.ops.aten.mm.default(permute_887, view_527);  permute_887 = view_527 = None
    permute_888: "f32[16, 16]" = torch.ops.aten.permute.default(mm_214, [1, 0]);  mm_214 = None
    mm_215: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1065, permute_889);  view_1065 = permute_889 = None
    view_1066: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_215, [8, 576, 576, 16]);  mm_215 = None
    permute_890: "f32[16, 16]" = torch.ops.aten.permute.default(permute_888, [1, 0]);  permute_888 = None
    permute_891: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1066, [0, 3, 1, 2]);  view_1066 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_589: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_891, memory_format = torch.contiguous_format);  permute_891 = None
    view_1067: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_589, [128, 576, 576]);  clone_589 = None
    bmm_110: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_892, view_1067);  permute_892 = None
    bmm_111: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1067, permute_893);  view_1067 = permute_893 = None
    view_1068: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_110, [8, 16, 48, 576]);  bmm_110 = None
    view_1069: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_111, [8, 16, 576, 48]);  bmm_111 = None
    permute_894: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1068, [0, 1, 3, 2]);  view_1068 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_30: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1059, 0, 2);  view_1059 = None
    select_scatter_31: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_894, 0, 1);  permute_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_412: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_30, select_scatter_31);  select_scatter_30 = select_scatter_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_709: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1069, 0.14433756729740643);  view_1069 = None
    select_scatter_32: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_709, 0, 0);  mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_413: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_412, select_scatter_32);  add_412 = select_scatter_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_895: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_413, [1, 3, 0, 2, 4]);  add_413 = None
    clone_590: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_895, memory_format = torch.contiguous_format);  permute_895 = None
    view_1070: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_590, [8, 576, 2304]);  clone_590 = None
    view_1071: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1070, [4608, 2304]);  view_1070 = None
    mm_216: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1071, permute_896);  permute_896 = None
    permute_897: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1071, [1, 0])
    mm_217: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_897, view_521);  permute_897 = view_521 = None
    permute_898: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
    sum_240: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1071, [0], True);  view_1071 = None
    view_1072: "f32[2304]" = torch.ops.aten.reshape.default(sum_240, [2304]);  sum_240 = None
    permute_899: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_898, [1, 0]);  permute_898 = None
    view_1073: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_216, [8, 576, 768]);  mm_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_711: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1073, primals_497);  primals_497 = None
    mul_712: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_711, 768)
    sum_241: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_711, [2], True)
    mul_713: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_711, mul_260);  mul_711 = None
    sum_242: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_713, [2], True);  mul_713 = None
    mul_714: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_260, sum_242);  sum_242 = None
    sub_196: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_712, sum_241);  mul_712 = sum_241 = None
    sub_197: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_196, mul_714);  sub_196 = mul_714 = None
    mul_715: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_197);  div_60 = sub_197 = None
    mul_716: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1073, mul_260);  mul_260 = None
    sum_243: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_716, [0, 1]);  mul_716 = None
    sum_244: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1073, [0, 1]);  view_1073 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_414: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_411, mul_715);  add_411 = mul_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_717: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_414, primals_53);  primals_53 = None
    mul_718: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_414, view_520);  view_520 = None
    sum_245: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_718, [0, 1], True);  mul_718 = None
    view_1074: "f32[768]" = torch.ops.aten.reshape.default(sum_245, [768]);  sum_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1075: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_717, [4608, 768]);  mul_717 = None
    mm_218: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1075, permute_900);  permute_900 = None
    permute_901: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1075, [1, 0])
    mm_219: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_901, view_519);  permute_901 = view_519 = None
    permute_902: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
    sum_246: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1075, [0], True);  view_1075 = None
    view_1076: "f32[768]" = torch.ops.aten.reshape.default(sum_246, [768]);  sum_246 = None
    permute_903: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_902, [1, 0]);  permute_902 = None
    view_1077: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_218, [8, 576, 3072]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_720: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_233, 0.5);  add_233 = None
    mul_721: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_518, view_518)
    mul_722: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_721, -0.5);  mul_721 = None
    exp_48: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_722);  mul_722 = None
    mul_723: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
    mul_724: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_518, mul_723);  view_518 = mul_723 = None
    add_416: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_720, mul_724);  mul_720 = mul_724 = None
    mul_725: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1077, add_416);  view_1077 = add_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1078: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_725, [4608, 3072]);  mul_725 = None
    mm_220: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1078, permute_904);  permute_904 = None
    permute_905: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1078, [1, 0])
    mm_221: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_905, view_517);  permute_905 = view_517 = None
    permute_906: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
    sum_247: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1078, [0], True);  view_1078 = None
    view_1079: "f32[3072]" = torch.ops.aten.reshape.default(sum_247, [3072]);  sum_247 = None
    permute_907: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_906, [1, 0]);  permute_906 = None
    view_1080: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_220, [8, 576, 768]);  mm_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_727: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1080, primals_491);  primals_491 = None
    mul_728: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_727, 768)
    sum_248: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_727, [2], True)
    mul_729: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_727, mul_254);  mul_727 = None
    sum_249: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_729, [2], True);  mul_729 = None
    mul_730: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_254, sum_249);  sum_249 = None
    sub_199: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_728, sum_248);  mul_728 = sum_248 = None
    sub_200: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_199, mul_730);  sub_199 = mul_730 = None
    mul_731: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_200);  div_61 = sub_200 = None
    mul_732: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1080, mul_254);  mul_254 = None
    sum_250: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 1]);  mul_732 = None
    sum_251: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1080, [0, 1]);  view_1080 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_417: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_414, mul_731);  add_414 = mul_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_733: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_417, primals_52);  primals_52 = None
    mul_734: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_417, view_516);  view_516 = None
    sum_252: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_734, [0, 1], True);  mul_734 = None
    view_1081: "f32[768]" = torch.ops.aten.reshape.default(sum_252, [768]);  sum_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1082: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_733, [4608, 768]);  mul_733 = None
    mm_222: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1082, permute_908);  permute_908 = None
    permute_909: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1082, [1, 0])
    mm_223: "f32[768, 768]" = torch.ops.aten.mm.default(permute_909, view_515);  permute_909 = view_515 = None
    permute_910: "f32[768, 768]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    sum_253: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1082, [0], True);  view_1082 = None
    view_1083: "f32[768]" = torch.ops.aten.reshape.default(sum_253, [768]);  sum_253 = None
    permute_911: "f32[768, 768]" = torch.ops.aten.permute.default(permute_910, [1, 0]);  permute_910 = None
    view_1084: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_222, [8, 576, 768]);  mm_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1085: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1084, [8, 576, 16, 48]);  view_1084 = None
    permute_912: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1085, [0, 2, 1, 3]);  view_1085 = None
    clone_593: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_912, memory_format = torch.contiguous_format);  permute_912 = None
    view_1086: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_593, [128, 576, 48]);  clone_593 = None
    bmm_112: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_913, view_1086);  permute_913 = None
    bmm_113: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1086, permute_914);  view_1086 = permute_914 = None
    view_1087: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_112, [8, 16, 576, 48]);  bmm_112 = None
    view_1088: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_113, [8, 16, 576, 576]);  bmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_915: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1088, [0, 2, 3, 1]);  view_1088 = None
    sum_254: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_915, [0, 1, 2], True)
    view_1089: "f32[16]" = torch.ops.aten.reshape.default(sum_254, [16]);  sum_254 = None
    clone_594: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_915, memory_format = torch.contiguous_format);  permute_915 = None
    view_1090: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_594, [2654208, 16]);  clone_594 = None
    permute_916: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1090, [1, 0])
    mm_224: "f32[16, 16]" = torch.ops.aten.mm.default(permute_916, view_509);  permute_916 = view_509 = None
    permute_917: "f32[16, 16]" = torch.ops.aten.permute.default(mm_224, [1, 0]);  mm_224 = None
    mm_225: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1090, permute_918);  view_1090 = permute_918 = None
    view_1091: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_225, [8, 576, 576, 16]);  mm_225 = None
    permute_919: "f32[16, 16]" = torch.ops.aten.permute.default(permute_917, [1, 0]);  permute_917 = None
    permute_920: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1091, [0, 3, 1, 2]);  view_1091 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_735: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_920, alias_50);  permute_920 = None
    sum_255: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_735, [-1], True)
    mul_736: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_50, sum_255);  alias_50 = sum_255 = None
    sub_201: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_735, mul_736);  mul_735 = mul_736 = None
    clone_595: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_201, memory_format = torch.contiguous_format);  sub_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_921: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_595, [0, 2, 3, 1]);  clone_595 = None
    sum_256: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_921, [0, 1, 2], True)
    view_1092: "f32[16]" = torch.ops.aten.reshape.default(sum_256, [16]);  sum_256 = None
    clone_596: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_921, memory_format = torch.contiguous_format);  permute_921 = None
    view_1093: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_596, [2654208, 16]);  clone_596 = None
    permute_922: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1093, [1, 0])
    mm_226: "f32[16, 16]" = torch.ops.aten.mm.default(permute_922, view_507);  permute_922 = view_507 = None
    permute_923: "f32[16, 16]" = torch.ops.aten.permute.default(mm_226, [1, 0]);  mm_226 = None
    mm_227: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1093, permute_924);  view_1093 = permute_924 = None
    view_1094: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_227, [8, 576, 576, 16]);  mm_227 = None
    permute_925: "f32[16, 16]" = torch.ops.aten.permute.default(permute_923, [1, 0]);  permute_923 = None
    permute_926: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1094, [0, 3, 1, 2]);  view_1094 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_597: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_926, memory_format = torch.contiguous_format);  permute_926 = None
    view_1095: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_597, [128, 576, 576]);  clone_597 = None
    bmm_114: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_927, view_1095);  permute_927 = None
    bmm_115: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1095, permute_928);  view_1095 = permute_928 = None
    view_1096: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_114, [8, 16, 48, 576]);  bmm_114 = None
    view_1097: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_115, [8, 16, 576, 48]);  bmm_115 = None
    permute_929: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1096, [0, 1, 3, 2]);  view_1096 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_33: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1087, 0, 2);  view_1087 = None
    select_scatter_34: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_929, 0, 1);  permute_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_418: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_33, select_scatter_34);  select_scatter_33 = select_scatter_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_737: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1097, 0.14433756729740643);  view_1097 = None
    select_scatter_35: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_737, 0, 0);  mul_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_419: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_418, select_scatter_35);  add_418 = select_scatter_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_930: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_419, [1, 3, 0, 2, 4]);  add_419 = None
    clone_598: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_930, memory_format = torch.contiguous_format);  permute_930 = None
    view_1098: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_598, [8, 576, 2304]);  clone_598 = None
    view_1099: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1098, [4608, 2304]);  view_1098 = None
    mm_228: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1099, permute_931);  permute_931 = None
    permute_932: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1099, [1, 0])
    mm_229: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_932, view_501);  permute_932 = view_501 = None
    permute_933: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    sum_257: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1099, [0], True);  view_1099 = None
    view_1100: "f32[2304]" = torch.ops.aten.reshape.default(sum_257, [2304]);  sum_257 = None
    permute_934: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_933, [1, 0]);  permute_933 = None
    view_1101: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_228, [8, 576, 768]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_739: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1101, primals_481);  primals_481 = None
    mul_740: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_739, 768)
    sum_258: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_739, [2], True)
    mul_741: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_739, mul_250);  mul_739 = None
    sum_259: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_741, [2], True);  mul_741 = None
    mul_742: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_250, sum_259);  sum_259 = None
    sub_203: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_740, sum_258);  mul_740 = sum_258 = None
    sub_204: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_203, mul_742);  sub_203 = mul_742 = None
    mul_743: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_62, sub_204);  div_62 = sub_204 = None
    mul_744: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1101, mul_250);  mul_250 = None
    sum_260: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_744, [0, 1]);  mul_744 = None
    sum_261: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1101, [0, 1]);  view_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_420: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_417, mul_743);  add_417 = mul_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_745: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_420, primals_51);  primals_51 = None
    mul_746: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_420, view_500);  view_500 = None
    sum_262: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_746, [0, 1], True);  mul_746 = None
    view_1102: "f32[768]" = torch.ops.aten.reshape.default(sum_262, [768]);  sum_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1103: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_745, [4608, 768]);  mul_745 = None
    mm_230: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1103, permute_935);  permute_935 = None
    permute_936: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1103, [1, 0])
    mm_231: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_936, view_499);  permute_936 = view_499 = None
    permute_937: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
    sum_263: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1103, [0], True);  view_1103 = None
    view_1104: "f32[768]" = torch.ops.aten.reshape.default(sum_263, [768]);  sum_263 = None
    permute_938: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_937, [1, 0]);  permute_937 = None
    view_1105: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_230, [8, 576, 3072]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_748: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_224, 0.5);  add_224 = None
    mul_749: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_498, view_498)
    mul_750: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_749, -0.5);  mul_749 = None
    exp_49: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_750);  mul_750 = None
    mul_751: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
    mul_752: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_498, mul_751);  view_498 = mul_751 = None
    add_422: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_748, mul_752);  mul_748 = mul_752 = None
    mul_753: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1105, add_422);  view_1105 = add_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1106: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_753, [4608, 3072]);  mul_753 = None
    mm_232: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1106, permute_939);  permute_939 = None
    permute_940: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1106, [1, 0])
    mm_233: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_940, view_497);  permute_940 = view_497 = None
    permute_941: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
    sum_264: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1106, [0], True);  view_1106 = None
    view_1107: "f32[3072]" = torch.ops.aten.reshape.default(sum_264, [3072]);  sum_264 = None
    permute_942: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_941, [1, 0]);  permute_941 = None
    view_1108: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_232, [8, 576, 768]);  mm_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_755: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1108, primals_475);  primals_475 = None
    mul_756: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_755, 768)
    sum_265: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_755, [2], True)
    mul_757: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_755, mul_244);  mul_755 = None
    sum_266: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_757, [2], True);  mul_757 = None
    mul_758: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_244, sum_266);  sum_266 = None
    sub_206: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_756, sum_265);  mul_756 = sum_265 = None
    sub_207: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_206, mul_758);  sub_206 = mul_758 = None
    mul_759: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_63, sub_207);  div_63 = sub_207 = None
    mul_760: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1108, mul_244);  mul_244 = None
    sum_267: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_760, [0, 1]);  mul_760 = None
    sum_268: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1108, [0, 1]);  view_1108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_423: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_420, mul_759);  add_420 = mul_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_761: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_423, primals_50);  primals_50 = None
    mul_762: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_423, view_496);  view_496 = None
    sum_269: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_762, [0, 1], True);  mul_762 = None
    view_1109: "f32[768]" = torch.ops.aten.reshape.default(sum_269, [768]);  sum_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1110: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_761, [4608, 768]);  mul_761 = None
    mm_234: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1110, permute_943);  permute_943 = None
    permute_944: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1110, [1, 0])
    mm_235: "f32[768, 768]" = torch.ops.aten.mm.default(permute_944, view_495);  permute_944 = view_495 = None
    permute_945: "f32[768, 768]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    sum_270: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1110, [0], True);  view_1110 = None
    view_1111: "f32[768]" = torch.ops.aten.reshape.default(sum_270, [768]);  sum_270 = None
    permute_946: "f32[768, 768]" = torch.ops.aten.permute.default(permute_945, [1, 0]);  permute_945 = None
    view_1112: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_234, [8, 576, 768]);  mm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1113: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1112, [8, 576, 16, 48]);  view_1112 = None
    permute_947: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1113, [0, 2, 1, 3]);  view_1113 = None
    clone_601: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_947, memory_format = torch.contiguous_format);  permute_947 = None
    view_1114: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_601, [128, 576, 48]);  clone_601 = None
    bmm_116: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_948, view_1114);  permute_948 = None
    bmm_117: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1114, permute_949);  view_1114 = permute_949 = None
    view_1115: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_116, [8, 16, 576, 48]);  bmm_116 = None
    view_1116: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_117, [8, 16, 576, 576]);  bmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_950: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1116, [0, 2, 3, 1]);  view_1116 = None
    sum_271: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_950, [0, 1, 2], True)
    view_1117: "f32[16]" = torch.ops.aten.reshape.default(sum_271, [16]);  sum_271 = None
    clone_602: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_950, memory_format = torch.contiguous_format);  permute_950 = None
    view_1118: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_602, [2654208, 16]);  clone_602 = None
    permute_951: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1118, [1, 0])
    mm_236: "f32[16, 16]" = torch.ops.aten.mm.default(permute_951, view_489);  permute_951 = view_489 = None
    permute_952: "f32[16, 16]" = torch.ops.aten.permute.default(mm_236, [1, 0]);  mm_236 = None
    mm_237: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1118, permute_953);  view_1118 = permute_953 = None
    view_1119: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_237, [8, 576, 576, 16]);  mm_237 = None
    permute_954: "f32[16, 16]" = torch.ops.aten.permute.default(permute_952, [1, 0]);  permute_952 = None
    permute_955: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1119, [0, 3, 1, 2]);  view_1119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_763: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_955, alias_51);  permute_955 = None
    sum_272: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_763, [-1], True)
    mul_764: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_51, sum_272);  alias_51 = sum_272 = None
    sub_208: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_763, mul_764);  mul_763 = mul_764 = None
    clone_603: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_208, memory_format = torch.contiguous_format);  sub_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_956: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_603, [0, 2, 3, 1]);  clone_603 = None
    sum_273: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_956, [0, 1, 2], True)
    view_1120: "f32[16]" = torch.ops.aten.reshape.default(sum_273, [16]);  sum_273 = None
    clone_604: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_956, memory_format = torch.contiguous_format);  permute_956 = None
    view_1121: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_604, [2654208, 16]);  clone_604 = None
    permute_957: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1121, [1, 0])
    mm_238: "f32[16, 16]" = torch.ops.aten.mm.default(permute_957, view_487);  permute_957 = view_487 = None
    permute_958: "f32[16, 16]" = torch.ops.aten.permute.default(mm_238, [1, 0]);  mm_238 = None
    mm_239: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1121, permute_959);  view_1121 = permute_959 = None
    view_1122: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_239, [8, 576, 576, 16]);  mm_239 = None
    permute_960: "f32[16, 16]" = torch.ops.aten.permute.default(permute_958, [1, 0]);  permute_958 = None
    permute_961: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1122, [0, 3, 1, 2]);  view_1122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_605: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_961, memory_format = torch.contiguous_format);  permute_961 = None
    view_1123: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_605, [128, 576, 576]);  clone_605 = None
    bmm_118: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_962, view_1123);  permute_962 = None
    bmm_119: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1123, permute_963);  view_1123 = permute_963 = None
    view_1124: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_118, [8, 16, 48, 576]);  bmm_118 = None
    view_1125: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_119, [8, 16, 576, 48]);  bmm_119 = None
    permute_964: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1124, [0, 1, 3, 2]);  view_1124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_36: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1115, 0, 2);  view_1115 = None
    select_scatter_37: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_964, 0, 1);  permute_964 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_424: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_36, select_scatter_37);  select_scatter_36 = select_scatter_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_765: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1125, 0.14433756729740643);  view_1125 = None
    select_scatter_38: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_765, 0, 0);  mul_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_425: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_424, select_scatter_38);  add_424 = select_scatter_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_965: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_425, [1, 3, 0, 2, 4]);  add_425 = None
    clone_606: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_965, memory_format = torch.contiguous_format);  permute_965 = None
    view_1126: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_606, [8, 576, 2304]);  clone_606 = None
    view_1127: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1126, [4608, 2304]);  view_1126 = None
    mm_240: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1127, permute_966);  permute_966 = None
    permute_967: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1127, [1, 0])
    mm_241: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_967, view_481);  permute_967 = view_481 = None
    permute_968: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
    sum_274: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1127, [0], True);  view_1127 = None
    view_1128: "f32[2304]" = torch.ops.aten.reshape.default(sum_274, [2304]);  sum_274 = None
    permute_969: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_968, [1, 0]);  permute_968 = None
    view_1129: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_240, [8, 576, 768]);  mm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_767: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1129, primals_465);  primals_465 = None
    mul_768: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_767, 768)
    sum_275: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_767, [2], True)
    mul_769: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_767, mul_240);  mul_767 = None
    sum_276: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_769, [2], True);  mul_769 = None
    mul_770: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_240, sum_276);  sum_276 = None
    sub_210: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_768, sum_275);  mul_768 = sum_275 = None
    sub_211: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_210, mul_770);  sub_210 = mul_770 = None
    mul_771: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_64, sub_211);  div_64 = sub_211 = None
    mul_772: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1129, mul_240);  mul_240 = None
    sum_277: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_772, [0, 1]);  mul_772 = None
    sum_278: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1129, [0, 1]);  view_1129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_426: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_423, mul_771);  add_423 = mul_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_773: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_426, primals_49);  primals_49 = None
    mul_774: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_426, view_480);  view_480 = None
    sum_279: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_774, [0, 1], True);  mul_774 = None
    view_1130: "f32[768]" = torch.ops.aten.reshape.default(sum_279, [768]);  sum_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1131: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_773, [4608, 768]);  mul_773 = None
    mm_242: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1131, permute_970);  permute_970 = None
    permute_971: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1131, [1, 0])
    mm_243: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_971, view_479);  permute_971 = view_479 = None
    permute_972: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
    sum_280: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1131, [0], True);  view_1131 = None
    view_1132: "f32[768]" = torch.ops.aten.reshape.default(sum_280, [768]);  sum_280 = None
    permute_973: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_972, [1, 0]);  permute_972 = None
    view_1133: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_242, [8, 576, 3072]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_776: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_215, 0.5);  add_215 = None
    mul_777: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_478, view_478)
    mul_778: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_777, -0.5);  mul_777 = None
    exp_50: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_778);  mul_778 = None
    mul_779: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
    mul_780: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_478, mul_779);  view_478 = mul_779 = None
    add_428: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_776, mul_780);  mul_776 = mul_780 = None
    mul_781: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1133, add_428);  view_1133 = add_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1134: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_781, [4608, 3072]);  mul_781 = None
    mm_244: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1134, permute_974);  permute_974 = None
    permute_975: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1134, [1, 0])
    mm_245: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_975, view_477);  permute_975 = view_477 = None
    permute_976: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_245, [1, 0]);  mm_245 = None
    sum_281: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1134, [0], True);  view_1134 = None
    view_1135: "f32[3072]" = torch.ops.aten.reshape.default(sum_281, [3072]);  sum_281 = None
    permute_977: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_976, [1, 0]);  permute_976 = None
    view_1136: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_244, [8, 576, 768]);  mm_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_783: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1136, primals_459);  primals_459 = None
    mul_784: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_783, 768)
    sum_282: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_783, [2], True)
    mul_785: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_783, mul_234);  mul_783 = None
    sum_283: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_785, [2], True);  mul_785 = None
    mul_786: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_234, sum_283);  sum_283 = None
    sub_213: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_784, sum_282);  mul_784 = sum_282 = None
    sub_214: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_213, mul_786);  sub_213 = mul_786 = None
    mul_787: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_65, sub_214);  div_65 = sub_214 = None
    mul_788: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1136, mul_234);  mul_234 = None
    sum_284: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_788, [0, 1]);  mul_788 = None
    sum_285: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1136, [0, 1]);  view_1136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_429: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_426, mul_787);  add_426 = mul_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_789: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_429, primals_48);  primals_48 = None
    mul_790: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_429, view_476);  view_476 = None
    sum_286: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_790, [0, 1], True);  mul_790 = None
    view_1137: "f32[768]" = torch.ops.aten.reshape.default(sum_286, [768]);  sum_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1138: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_789, [4608, 768]);  mul_789 = None
    mm_246: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1138, permute_978);  permute_978 = None
    permute_979: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1138, [1, 0])
    mm_247: "f32[768, 768]" = torch.ops.aten.mm.default(permute_979, view_475);  permute_979 = view_475 = None
    permute_980: "f32[768, 768]" = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
    sum_287: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1138, [0], True);  view_1138 = None
    view_1139: "f32[768]" = torch.ops.aten.reshape.default(sum_287, [768]);  sum_287 = None
    permute_981: "f32[768, 768]" = torch.ops.aten.permute.default(permute_980, [1, 0]);  permute_980 = None
    view_1140: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_246, [8, 576, 768]);  mm_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1141: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1140, [8, 576, 16, 48]);  view_1140 = None
    permute_982: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1141, [0, 2, 1, 3]);  view_1141 = None
    clone_609: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_982, memory_format = torch.contiguous_format);  permute_982 = None
    view_1142: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_609, [128, 576, 48]);  clone_609 = None
    bmm_120: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_983, view_1142);  permute_983 = None
    bmm_121: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1142, permute_984);  view_1142 = permute_984 = None
    view_1143: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_120, [8, 16, 576, 48]);  bmm_120 = None
    view_1144: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_121, [8, 16, 576, 576]);  bmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_985: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1144, [0, 2, 3, 1]);  view_1144 = None
    sum_288: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_985, [0, 1, 2], True)
    view_1145: "f32[16]" = torch.ops.aten.reshape.default(sum_288, [16]);  sum_288 = None
    clone_610: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_985, memory_format = torch.contiguous_format);  permute_985 = None
    view_1146: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_610, [2654208, 16]);  clone_610 = None
    permute_986: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1146, [1, 0])
    mm_248: "f32[16, 16]" = torch.ops.aten.mm.default(permute_986, view_469);  permute_986 = view_469 = None
    permute_987: "f32[16, 16]" = torch.ops.aten.permute.default(mm_248, [1, 0]);  mm_248 = None
    mm_249: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1146, permute_988);  view_1146 = permute_988 = None
    view_1147: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_249, [8, 576, 576, 16]);  mm_249 = None
    permute_989: "f32[16, 16]" = torch.ops.aten.permute.default(permute_987, [1, 0]);  permute_987 = None
    permute_990: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1147, [0, 3, 1, 2]);  view_1147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_791: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_990, alias_52);  permute_990 = None
    sum_289: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_791, [-1], True)
    mul_792: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_52, sum_289);  alias_52 = sum_289 = None
    sub_215: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_791, mul_792);  mul_791 = mul_792 = None
    clone_611: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_215, memory_format = torch.contiguous_format);  sub_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_991: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_611, [0, 2, 3, 1]);  clone_611 = None
    sum_290: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_991, [0, 1, 2], True)
    view_1148: "f32[16]" = torch.ops.aten.reshape.default(sum_290, [16]);  sum_290 = None
    clone_612: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_991, memory_format = torch.contiguous_format);  permute_991 = None
    view_1149: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_612, [2654208, 16]);  clone_612 = None
    permute_992: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1149, [1, 0])
    mm_250: "f32[16, 16]" = torch.ops.aten.mm.default(permute_992, view_467);  permute_992 = view_467 = None
    permute_993: "f32[16, 16]" = torch.ops.aten.permute.default(mm_250, [1, 0]);  mm_250 = None
    mm_251: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1149, permute_994);  view_1149 = permute_994 = None
    view_1150: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_251, [8, 576, 576, 16]);  mm_251 = None
    permute_995: "f32[16, 16]" = torch.ops.aten.permute.default(permute_993, [1, 0]);  permute_993 = None
    permute_996: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1150, [0, 3, 1, 2]);  view_1150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_613: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_996, memory_format = torch.contiguous_format);  permute_996 = None
    view_1151: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_613, [128, 576, 576]);  clone_613 = None
    bmm_122: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_997, view_1151);  permute_997 = None
    bmm_123: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1151, permute_998);  view_1151 = permute_998 = None
    view_1152: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_122, [8, 16, 48, 576]);  bmm_122 = None
    view_1153: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_123, [8, 16, 576, 48]);  bmm_123 = None
    permute_999: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1152, [0, 1, 3, 2]);  view_1152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_39: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1143, 0, 2);  view_1143 = None
    select_scatter_40: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_999, 0, 1);  permute_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_430: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_39, select_scatter_40);  select_scatter_39 = select_scatter_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_793: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1153, 0.14433756729740643);  view_1153 = None
    select_scatter_41: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_793, 0, 0);  mul_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_431: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_430, select_scatter_41);  add_430 = select_scatter_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1000: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_431, [1, 3, 0, 2, 4]);  add_431 = None
    clone_614: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1000, memory_format = torch.contiguous_format);  permute_1000 = None
    view_1154: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_614, [8, 576, 2304]);  clone_614 = None
    view_1155: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1154, [4608, 2304]);  view_1154 = None
    mm_252: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1155, permute_1001);  permute_1001 = None
    permute_1002: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1155, [1, 0])
    mm_253: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1002, view_461);  permute_1002 = view_461 = None
    permute_1003: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    sum_291: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1155, [0], True);  view_1155 = None
    view_1156: "f32[2304]" = torch.ops.aten.reshape.default(sum_291, [2304]);  sum_291 = None
    permute_1004: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1003, [1, 0]);  permute_1003 = None
    view_1157: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_252, [8, 576, 768]);  mm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_795: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1157, primals_449);  primals_449 = None
    mul_796: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_795, 768)
    sum_292: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_795, [2], True)
    mul_797: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_795, mul_230);  mul_795 = None
    sum_293: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_797, [2], True);  mul_797 = None
    mul_798: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_230, sum_293);  sum_293 = None
    sub_217: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_796, sum_292);  mul_796 = sum_292 = None
    sub_218: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_217, mul_798);  sub_217 = mul_798 = None
    mul_799: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_66, sub_218);  div_66 = sub_218 = None
    mul_800: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1157, mul_230);  mul_230 = None
    sum_294: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_800, [0, 1]);  mul_800 = None
    sum_295: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1157, [0, 1]);  view_1157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_432: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_429, mul_799);  add_429 = mul_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_801: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_432, primals_47);  primals_47 = None
    mul_802: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_432, view_460);  view_460 = None
    sum_296: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_802, [0, 1], True);  mul_802 = None
    view_1158: "f32[768]" = torch.ops.aten.reshape.default(sum_296, [768]);  sum_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1159: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_801, [4608, 768]);  mul_801 = None
    mm_254: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1159, permute_1005);  permute_1005 = None
    permute_1006: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1159, [1, 0])
    mm_255: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1006, view_459);  permute_1006 = view_459 = None
    permute_1007: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
    sum_297: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1159, [0], True);  view_1159 = None
    view_1160: "f32[768]" = torch.ops.aten.reshape.default(sum_297, [768]);  sum_297 = None
    permute_1008: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1007, [1, 0]);  permute_1007 = None
    view_1161: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_254, [8, 576, 3072]);  mm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_804: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_206, 0.5);  add_206 = None
    mul_805: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_458, view_458)
    mul_806: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_805, -0.5);  mul_805 = None
    exp_51: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_806);  mul_806 = None
    mul_807: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_51, 0.3989422804014327);  exp_51 = None
    mul_808: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_458, mul_807);  view_458 = mul_807 = None
    add_434: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_804, mul_808);  mul_804 = mul_808 = None
    mul_809: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1161, add_434);  view_1161 = add_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1162: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_809, [4608, 3072]);  mul_809 = None
    mm_256: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1162, permute_1009);  permute_1009 = None
    permute_1010: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1162, [1, 0])
    mm_257: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1010, view_457);  permute_1010 = view_457 = None
    permute_1011: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_257, [1, 0]);  mm_257 = None
    sum_298: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1162, [0], True);  view_1162 = None
    view_1163: "f32[3072]" = torch.ops.aten.reshape.default(sum_298, [3072]);  sum_298 = None
    permute_1012: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1011, [1, 0]);  permute_1011 = None
    view_1164: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_256, [8, 576, 768]);  mm_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_811: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1164, primals_443);  primals_443 = None
    mul_812: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_811, 768)
    sum_299: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_811, [2], True)
    mul_813: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_811, mul_224);  mul_811 = None
    sum_300: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_813, [2], True);  mul_813 = None
    mul_814: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_224, sum_300);  sum_300 = None
    sub_220: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_812, sum_299);  mul_812 = sum_299 = None
    sub_221: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_220, mul_814);  sub_220 = mul_814 = None
    mul_815: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_67, sub_221);  div_67 = sub_221 = None
    mul_816: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1164, mul_224);  mul_224 = None
    sum_301: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_816, [0, 1]);  mul_816 = None
    sum_302: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1164, [0, 1]);  view_1164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_435: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_432, mul_815);  add_432 = mul_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_817: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_435, primals_46);  primals_46 = None
    mul_818: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_435, view_456);  view_456 = None
    sum_303: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_818, [0, 1], True);  mul_818 = None
    view_1165: "f32[768]" = torch.ops.aten.reshape.default(sum_303, [768]);  sum_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1166: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_817, [4608, 768]);  mul_817 = None
    mm_258: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1166, permute_1013);  permute_1013 = None
    permute_1014: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1166, [1, 0])
    mm_259: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1014, view_455);  permute_1014 = view_455 = None
    permute_1015: "f32[768, 768]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    sum_304: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1166, [0], True);  view_1166 = None
    view_1167: "f32[768]" = torch.ops.aten.reshape.default(sum_304, [768]);  sum_304 = None
    permute_1016: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1015, [1, 0]);  permute_1015 = None
    view_1168: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_258, [8, 576, 768]);  mm_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1169: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1168, [8, 576, 16, 48]);  view_1168 = None
    permute_1017: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1169, [0, 2, 1, 3]);  view_1169 = None
    clone_617: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1017, memory_format = torch.contiguous_format);  permute_1017 = None
    view_1170: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_617, [128, 576, 48]);  clone_617 = None
    bmm_124: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1018, view_1170);  permute_1018 = None
    bmm_125: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1170, permute_1019);  view_1170 = permute_1019 = None
    view_1171: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_124, [8, 16, 576, 48]);  bmm_124 = None
    view_1172: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_125, [8, 16, 576, 576]);  bmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1020: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1172, [0, 2, 3, 1]);  view_1172 = None
    sum_305: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1020, [0, 1, 2], True)
    view_1173: "f32[16]" = torch.ops.aten.reshape.default(sum_305, [16]);  sum_305 = None
    clone_618: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1020, memory_format = torch.contiguous_format);  permute_1020 = None
    view_1174: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_618, [2654208, 16]);  clone_618 = None
    permute_1021: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1174, [1, 0])
    mm_260: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1021, view_449);  permute_1021 = view_449 = None
    permute_1022: "f32[16, 16]" = torch.ops.aten.permute.default(mm_260, [1, 0]);  mm_260 = None
    mm_261: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1174, permute_1023);  view_1174 = permute_1023 = None
    view_1175: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_261, [8, 576, 576, 16]);  mm_261 = None
    permute_1024: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1022, [1, 0]);  permute_1022 = None
    permute_1025: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1175, [0, 3, 1, 2]);  view_1175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_819: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1025, alias_53);  permute_1025 = None
    sum_306: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_819, [-1], True)
    mul_820: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_53, sum_306);  alias_53 = sum_306 = None
    sub_222: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_819, mul_820);  mul_819 = mul_820 = None
    clone_619: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_222, memory_format = torch.contiguous_format);  sub_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1026: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_619, [0, 2, 3, 1]);  clone_619 = None
    sum_307: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1026, [0, 1, 2], True)
    view_1176: "f32[16]" = torch.ops.aten.reshape.default(sum_307, [16]);  sum_307 = None
    clone_620: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1026, memory_format = torch.contiguous_format);  permute_1026 = None
    view_1177: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_620, [2654208, 16]);  clone_620 = None
    permute_1027: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1177, [1, 0])
    mm_262: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1027, view_447);  permute_1027 = view_447 = None
    permute_1028: "f32[16, 16]" = torch.ops.aten.permute.default(mm_262, [1, 0]);  mm_262 = None
    mm_263: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1177, permute_1029);  view_1177 = permute_1029 = None
    view_1178: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_263, [8, 576, 576, 16]);  mm_263 = None
    permute_1030: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1028, [1, 0]);  permute_1028 = None
    permute_1031: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1178, [0, 3, 1, 2]);  view_1178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_621: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1031, memory_format = torch.contiguous_format);  permute_1031 = None
    view_1179: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_621, [128, 576, 576]);  clone_621 = None
    bmm_126: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1032, view_1179);  permute_1032 = None
    bmm_127: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1179, permute_1033);  view_1179 = permute_1033 = None
    view_1180: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_126, [8, 16, 48, 576]);  bmm_126 = None
    view_1181: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_127, [8, 16, 576, 48]);  bmm_127 = None
    permute_1034: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1180, [0, 1, 3, 2]);  view_1180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_42: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1171, 0, 2);  view_1171 = None
    select_scatter_43: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1034, 0, 1);  permute_1034 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_436: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_42, select_scatter_43);  select_scatter_42 = select_scatter_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_821: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1181, 0.14433756729740643);  view_1181 = None
    select_scatter_44: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_821, 0, 0);  mul_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_437: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_436, select_scatter_44);  add_436 = select_scatter_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1035: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_437, [1, 3, 0, 2, 4]);  add_437 = None
    clone_622: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1035, memory_format = torch.contiguous_format);  permute_1035 = None
    view_1182: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_622, [8, 576, 2304]);  clone_622 = None
    view_1183: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1182, [4608, 2304]);  view_1182 = None
    mm_264: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1183, permute_1036);  permute_1036 = None
    permute_1037: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1183, [1, 0])
    mm_265: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1037, view_441);  permute_1037 = view_441 = None
    permute_1038: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    sum_308: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1183, [0], True);  view_1183 = None
    view_1184: "f32[2304]" = torch.ops.aten.reshape.default(sum_308, [2304]);  sum_308 = None
    permute_1039: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1038, [1, 0]);  permute_1038 = None
    view_1185: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_264, [8, 576, 768]);  mm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_823: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1185, primals_433);  primals_433 = None
    mul_824: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_823, 768)
    sum_309: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_823, [2], True)
    mul_825: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_823, mul_220);  mul_823 = None
    sum_310: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_825, [2], True);  mul_825 = None
    mul_826: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_220, sum_310);  sum_310 = None
    sub_224: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_824, sum_309);  mul_824 = sum_309 = None
    sub_225: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_224, mul_826);  sub_224 = mul_826 = None
    mul_827: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_68, sub_225);  div_68 = sub_225 = None
    mul_828: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1185, mul_220);  mul_220 = None
    sum_311: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_828, [0, 1]);  mul_828 = None
    sum_312: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1185, [0, 1]);  view_1185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_438: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_435, mul_827);  add_435 = mul_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_829: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_438, primals_45);  primals_45 = None
    mul_830: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_438, view_440);  view_440 = None
    sum_313: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_830, [0, 1], True);  mul_830 = None
    view_1186: "f32[768]" = torch.ops.aten.reshape.default(sum_313, [768]);  sum_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1187: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_829, [4608, 768]);  mul_829 = None
    mm_266: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1187, permute_1040);  permute_1040 = None
    permute_1041: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1187, [1, 0])
    mm_267: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1041, view_439);  permute_1041 = view_439 = None
    permute_1042: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_267, [1, 0]);  mm_267 = None
    sum_314: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1187, [0], True);  view_1187 = None
    view_1188: "f32[768]" = torch.ops.aten.reshape.default(sum_314, [768]);  sum_314 = None
    permute_1043: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1042, [1, 0]);  permute_1042 = None
    view_1189: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_266, [8, 576, 3072]);  mm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_832: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_197, 0.5);  add_197 = None
    mul_833: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_438, view_438)
    mul_834: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_833, -0.5);  mul_833 = None
    exp_52: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_834);  mul_834 = None
    mul_835: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_52, 0.3989422804014327);  exp_52 = None
    mul_836: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_438, mul_835);  view_438 = mul_835 = None
    add_440: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_832, mul_836);  mul_832 = mul_836 = None
    mul_837: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1189, add_440);  view_1189 = add_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1190: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_837, [4608, 3072]);  mul_837 = None
    mm_268: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1190, permute_1044);  permute_1044 = None
    permute_1045: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1190, [1, 0])
    mm_269: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1045, view_437);  permute_1045 = view_437 = None
    permute_1046: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_269, [1, 0]);  mm_269 = None
    sum_315: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1190, [0], True);  view_1190 = None
    view_1191: "f32[3072]" = torch.ops.aten.reshape.default(sum_315, [3072]);  sum_315 = None
    permute_1047: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1046, [1, 0]);  permute_1046 = None
    view_1192: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_268, [8, 576, 768]);  mm_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_839: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1192, primals_427);  primals_427 = None
    mul_840: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_839, 768)
    sum_316: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_839, [2], True)
    mul_841: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_839, mul_214);  mul_839 = None
    sum_317: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_841, [2], True);  mul_841 = None
    mul_842: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_214, sum_317);  sum_317 = None
    sub_227: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_840, sum_316);  mul_840 = sum_316 = None
    sub_228: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_227, mul_842);  sub_227 = mul_842 = None
    mul_843: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_69, sub_228);  div_69 = sub_228 = None
    mul_844: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1192, mul_214);  mul_214 = None
    sum_318: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_844, [0, 1]);  mul_844 = None
    sum_319: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1192, [0, 1]);  view_1192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_441: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_438, mul_843);  add_438 = mul_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_845: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_441, primals_44);  primals_44 = None
    mul_846: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_441, view_436);  view_436 = None
    sum_320: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_846, [0, 1], True);  mul_846 = None
    view_1193: "f32[768]" = torch.ops.aten.reshape.default(sum_320, [768]);  sum_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1194: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_845, [4608, 768]);  mul_845 = None
    mm_270: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1194, permute_1048);  permute_1048 = None
    permute_1049: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1194, [1, 0])
    mm_271: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1049, view_435);  permute_1049 = view_435 = None
    permute_1050: "f32[768, 768]" = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
    sum_321: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1194, [0], True);  view_1194 = None
    view_1195: "f32[768]" = torch.ops.aten.reshape.default(sum_321, [768]);  sum_321 = None
    permute_1051: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1050, [1, 0]);  permute_1050 = None
    view_1196: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_270, [8, 576, 768]);  mm_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1197: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1196, [8, 576, 16, 48]);  view_1196 = None
    permute_1052: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1197, [0, 2, 1, 3]);  view_1197 = None
    clone_625: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1052, memory_format = torch.contiguous_format);  permute_1052 = None
    view_1198: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_625, [128, 576, 48]);  clone_625 = None
    bmm_128: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1053, view_1198);  permute_1053 = None
    bmm_129: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1198, permute_1054);  view_1198 = permute_1054 = None
    view_1199: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_128, [8, 16, 576, 48]);  bmm_128 = None
    view_1200: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_129, [8, 16, 576, 576]);  bmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1055: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1200, [0, 2, 3, 1]);  view_1200 = None
    sum_322: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1055, [0, 1, 2], True)
    view_1201: "f32[16]" = torch.ops.aten.reshape.default(sum_322, [16]);  sum_322 = None
    clone_626: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1055, memory_format = torch.contiguous_format);  permute_1055 = None
    view_1202: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_626, [2654208, 16]);  clone_626 = None
    permute_1056: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1202, [1, 0])
    mm_272: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1056, view_429);  permute_1056 = view_429 = None
    permute_1057: "f32[16, 16]" = torch.ops.aten.permute.default(mm_272, [1, 0]);  mm_272 = None
    mm_273: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1202, permute_1058);  view_1202 = permute_1058 = None
    view_1203: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_273, [8, 576, 576, 16]);  mm_273 = None
    permute_1059: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1057, [1, 0]);  permute_1057 = None
    permute_1060: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1203, [0, 3, 1, 2]);  view_1203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_847: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1060, alias_54);  permute_1060 = None
    sum_323: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_847, [-1], True)
    mul_848: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_54, sum_323);  alias_54 = sum_323 = None
    sub_229: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_847, mul_848);  mul_847 = mul_848 = None
    clone_627: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_229, memory_format = torch.contiguous_format);  sub_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1061: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_627, [0, 2, 3, 1]);  clone_627 = None
    sum_324: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1061, [0, 1, 2], True)
    view_1204: "f32[16]" = torch.ops.aten.reshape.default(sum_324, [16]);  sum_324 = None
    clone_628: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1061, memory_format = torch.contiguous_format);  permute_1061 = None
    view_1205: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_628, [2654208, 16]);  clone_628 = None
    permute_1062: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1205, [1, 0])
    mm_274: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1062, view_427);  permute_1062 = view_427 = None
    permute_1063: "f32[16, 16]" = torch.ops.aten.permute.default(mm_274, [1, 0]);  mm_274 = None
    mm_275: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1205, permute_1064);  view_1205 = permute_1064 = None
    view_1206: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_275, [8, 576, 576, 16]);  mm_275 = None
    permute_1065: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1063, [1, 0]);  permute_1063 = None
    permute_1066: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1206, [0, 3, 1, 2]);  view_1206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_629: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1066, memory_format = torch.contiguous_format);  permute_1066 = None
    view_1207: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_629, [128, 576, 576]);  clone_629 = None
    bmm_130: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1067, view_1207);  permute_1067 = None
    bmm_131: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1207, permute_1068);  view_1207 = permute_1068 = None
    view_1208: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_130, [8, 16, 48, 576]);  bmm_130 = None
    view_1209: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_131, [8, 16, 576, 48]);  bmm_131 = None
    permute_1069: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1208, [0, 1, 3, 2]);  view_1208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_45: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1199, 0, 2);  view_1199 = None
    select_scatter_46: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1069, 0, 1);  permute_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_442: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_45, select_scatter_46);  select_scatter_45 = select_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_849: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1209, 0.14433756729740643);  view_1209 = None
    select_scatter_47: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_849, 0, 0);  mul_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_443: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_442, select_scatter_47);  add_442 = select_scatter_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1070: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_443, [1, 3, 0, 2, 4]);  add_443 = None
    clone_630: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1070, memory_format = torch.contiguous_format);  permute_1070 = None
    view_1210: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_630, [8, 576, 2304]);  clone_630 = None
    view_1211: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1210, [4608, 2304]);  view_1210 = None
    mm_276: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1211, permute_1071);  permute_1071 = None
    permute_1072: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1211, [1, 0])
    mm_277: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1072, view_421);  permute_1072 = view_421 = None
    permute_1073: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_277, [1, 0]);  mm_277 = None
    sum_325: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1211, [0], True);  view_1211 = None
    view_1212: "f32[2304]" = torch.ops.aten.reshape.default(sum_325, [2304]);  sum_325 = None
    permute_1074: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1073, [1, 0]);  permute_1073 = None
    view_1213: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_276, [8, 576, 768]);  mm_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_851: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1213, primals_417);  primals_417 = None
    mul_852: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_851, 768)
    sum_326: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_851, [2], True)
    mul_853: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_851, mul_210);  mul_851 = None
    sum_327: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_853, [2], True);  mul_853 = None
    mul_854: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_210, sum_327);  sum_327 = None
    sub_231: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_852, sum_326);  mul_852 = sum_326 = None
    sub_232: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_231, mul_854);  sub_231 = mul_854 = None
    mul_855: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_70, sub_232);  div_70 = sub_232 = None
    mul_856: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1213, mul_210);  mul_210 = None
    sum_328: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_856, [0, 1]);  mul_856 = None
    sum_329: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1213, [0, 1]);  view_1213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_444: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_441, mul_855);  add_441 = mul_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_857: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_444, primals_43);  primals_43 = None
    mul_858: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_444, view_420);  view_420 = None
    sum_330: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_858, [0, 1], True);  mul_858 = None
    view_1214: "f32[768]" = torch.ops.aten.reshape.default(sum_330, [768]);  sum_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1215: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_857, [4608, 768]);  mul_857 = None
    mm_278: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1215, permute_1075);  permute_1075 = None
    permute_1076: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1215, [1, 0])
    mm_279: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1076, view_419);  permute_1076 = view_419 = None
    permute_1077: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_279, [1, 0]);  mm_279 = None
    sum_331: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1215, [0], True);  view_1215 = None
    view_1216: "f32[768]" = torch.ops.aten.reshape.default(sum_331, [768]);  sum_331 = None
    permute_1078: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1077, [1, 0]);  permute_1077 = None
    view_1217: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_278, [8, 576, 3072]);  mm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_860: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_188, 0.5);  add_188 = None
    mul_861: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_418, view_418)
    mul_862: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_861, -0.5);  mul_861 = None
    exp_53: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_862);  mul_862 = None
    mul_863: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_53, 0.3989422804014327);  exp_53 = None
    mul_864: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_418, mul_863);  view_418 = mul_863 = None
    add_446: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_860, mul_864);  mul_860 = mul_864 = None
    mul_865: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1217, add_446);  view_1217 = add_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1218: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_865, [4608, 3072]);  mul_865 = None
    mm_280: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1218, permute_1079);  permute_1079 = None
    permute_1080: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1218, [1, 0])
    mm_281: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1080, view_417);  permute_1080 = view_417 = None
    permute_1081: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_281, [1, 0]);  mm_281 = None
    sum_332: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1218, [0], True);  view_1218 = None
    view_1219: "f32[3072]" = torch.ops.aten.reshape.default(sum_332, [3072]);  sum_332 = None
    permute_1082: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1081, [1, 0]);  permute_1081 = None
    view_1220: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_280, [8, 576, 768]);  mm_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_867: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1220, primals_411);  primals_411 = None
    mul_868: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_867, 768)
    sum_333: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_867, [2], True)
    mul_869: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_867, mul_204);  mul_867 = None
    sum_334: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_869, [2], True);  mul_869 = None
    mul_870: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_204, sum_334);  sum_334 = None
    sub_234: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_868, sum_333);  mul_868 = sum_333 = None
    sub_235: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_234, mul_870);  sub_234 = mul_870 = None
    mul_871: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_71, sub_235);  div_71 = sub_235 = None
    mul_872: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1220, mul_204);  mul_204 = None
    sum_335: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_872, [0, 1]);  mul_872 = None
    sum_336: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1220, [0, 1]);  view_1220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_447: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_444, mul_871);  add_444 = mul_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_873: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_447, primals_42);  primals_42 = None
    mul_874: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_447, view_416);  view_416 = None
    sum_337: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_874, [0, 1], True);  mul_874 = None
    view_1221: "f32[768]" = torch.ops.aten.reshape.default(sum_337, [768]);  sum_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1222: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_873, [4608, 768]);  mul_873 = None
    mm_282: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1222, permute_1083);  permute_1083 = None
    permute_1084: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1222, [1, 0])
    mm_283: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1084, view_415);  permute_1084 = view_415 = None
    permute_1085: "f32[768, 768]" = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
    sum_338: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1222, [0], True);  view_1222 = None
    view_1223: "f32[768]" = torch.ops.aten.reshape.default(sum_338, [768]);  sum_338 = None
    permute_1086: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1085, [1, 0]);  permute_1085 = None
    view_1224: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_282, [8, 576, 768]);  mm_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1225: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1224, [8, 576, 16, 48]);  view_1224 = None
    permute_1087: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1225, [0, 2, 1, 3]);  view_1225 = None
    clone_633: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1087, memory_format = torch.contiguous_format);  permute_1087 = None
    view_1226: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_633, [128, 576, 48]);  clone_633 = None
    bmm_132: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1088, view_1226);  permute_1088 = None
    bmm_133: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1226, permute_1089);  view_1226 = permute_1089 = None
    view_1227: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_132, [8, 16, 576, 48]);  bmm_132 = None
    view_1228: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_133, [8, 16, 576, 576]);  bmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1090: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1228, [0, 2, 3, 1]);  view_1228 = None
    sum_339: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1090, [0, 1, 2], True)
    view_1229: "f32[16]" = torch.ops.aten.reshape.default(sum_339, [16]);  sum_339 = None
    clone_634: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1090, memory_format = torch.contiguous_format);  permute_1090 = None
    view_1230: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_634, [2654208, 16]);  clone_634 = None
    permute_1091: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1230, [1, 0])
    mm_284: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1091, view_409);  permute_1091 = view_409 = None
    permute_1092: "f32[16, 16]" = torch.ops.aten.permute.default(mm_284, [1, 0]);  mm_284 = None
    mm_285: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1230, permute_1093);  view_1230 = permute_1093 = None
    view_1231: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_285, [8, 576, 576, 16]);  mm_285 = None
    permute_1094: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1092, [1, 0]);  permute_1092 = None
    permute_1095: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1231, [0, 3, 1, 2]);  view_1231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_875: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1095, alias_55);  permute_1095 = None
    sum_340: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_875, [-1], True)
    mul_876: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_55, sum_340);  alias_55 = sum_340 = None
    sub_236: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_875, mul_876);  mul_875 = mul_876 = None
    clone_635: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_236, memory_format = torch.contiguous_format);  sub_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1096: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_635, [0, 2, 3, 1]);  clone_635 = None
    sum_341: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1096, [0, 1, 2], True)
    view_1232: "f32[16]" = torch.ops.aten.reshape.default(sum_341, [16]);  sum_341 = None
    clone_636: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1096, memory_format = torch.contiguous_format);  permute_1096 = None
    view_1233: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_636, [2654208, 16]);  clone_636 = None
    permute_1097: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1233, [1, 0])
    mm_286: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1097, view_407);  permute_1097 = view_407 = None
    permute_1098: "f32[16, 16]" = torch.ops.aten.permute.default(mm_286, [1, 0]);  mm_286 = None
    mm_287: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1233, permute_1099);  view_1233 = permute_1099 = None
    view_1234: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_287, [8, 576, 576, 16]);  mm_287 = None
    permute_1100: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1098, [1, 0]);  permute_1098 = None
    permute_1101: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1234, [0, 3, 1, 2]);  view_1234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_637: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1101, memory_format = torch.contiguous_format);  permute_1101 = None
    view_1235: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_637, [128, 576, 576]);  clone_637 = None
    bmm_134: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1102, view_1235);  permute_1102 = None
    bmm_135: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1235, permute_1103);  view_1235 = permute_1103 = None
    view_1236: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_134, [8, 16, 48, 576]);  bmm_134 = None
    view_1237: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_135, [8, 16, 576, 48]);  bmm_135 = None
    permute_1104: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1236, [0, 1, 3, 2]);  view_1236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_48: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1227, 0, 2);  view_1227 = None
    select_scatter_49: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1104, 0, 1);  permute_1104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_448: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_48, select_scatter_49);  select_scatter_48 = select_scatter_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_877: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1237, 0.14433756729740643);  view_1237 = None
    select_scatter_50: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_877, 0, 0);  mul_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_449: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_448, select_scatter_50);  add_448 = select_scatter_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1105: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_449, [1, 3, 0, 2, 4]);  add_449 = None
    clone_638: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1105, memory_format = torch.contiguous_format);  permute_1105 = None
    view_1238: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_638, [8, 576, 2304]);  clone_638 = None
    view_1239: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1238, [4608, 2304]);  view_1238 = None
    mm_288: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1239, permute_1106);  permute_1106 = None
    permute_1107: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1239, [1, 0])
    mm_289: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1107, view_401);  permute_1107 = view_401 = None
    permute_1108: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_289, [1, 0]);  mm_289 = None
    sum_342: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1239, [0], True);  view_1239 = None
    view_1240: "f32[2304]" = torch.ops.aten.reshape.default(sum_342, [2304]);  sum_342 = None
    permute_1109: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1108, [1, 0]);  permute_1108 = None
    view_1241: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_288, [8, 576, 768]);  mm_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_879: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1241, primals_401);  primals_401 = None
    mul_880: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_879, 768)
    sum_343: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_879, [2], True)
    mul_881: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_879, mul_200);  mul_879 = None
    sum_344: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_881, [2], True);  mul_881 = None
    mul_882: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_200, sum_344);  sum_344 = None
    sub_238: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_880, sum_343);  mul_880 = sum_343 = None
    sub_239: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_238, mul_882);  sub_238 = mul_882 = None
    mul_883: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_72, sub_239);  div_72 = sub_239 = None
    mul_884: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1241, mul_200);  mul_200 = None
    sum_345: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_884, [0, 1]);  mul_884 = None
    sum_346: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1241, [0, 1]);  view_1241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_450: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_447, mul_883);  add_447 = mul_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_885: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_450, primals_41);  primals_41 = None
    mul_886: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_450, view_400);  view_400 = None
    sum_347: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_886, [0, 1], True);  mul_886 = None
    view_1242: "f32[768]" = torch.ops.aten.reshape.default(sum_347, [768]);  sum_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1243: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_885, [4608, 768]);  mul_885 = None
    mm_290: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1243, permute_1110);  permute_1110 = None
    permute_1111: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1243, [1, 0])
    mm_291: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1111, view_399);  permute_1111 = view_399 = None
    permute_1112: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_291, [1, 0]);  mm_291 = None
    sum_348: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1243, [0], True);  view_1243 = None
    view_1244: "f32[768]" = torch.ops.aten.reshape.default(sum_348, [768]);  sum_348 = None
    permute_1113: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1112, [1, 0]);  permute_1112 = None
    view_1245: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_290, [8, 576, 3072]);  mm_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_888: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_179, 0.5);  add_179 = None
    mul_889: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_398, view_398)
    mul_890: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_889, -0.5);  mul_889 = None
    exp_54: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_890);  mul_890 = None
    mul_891: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_54, 0.3989422804014327);  exp_54 = None
    mul_892: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_398, mul_891);  view_398 = mul_891 = None
    add_452: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_888, mul_892);  mul_888 = mul_892 = None
    mul_893: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1245, add_452);  view_1245 = add_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1246: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_893, [4608, 3072]);  mul_893 = None
    mm_292: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1246, permute_1114);  permute_1114 = None
    permute_1115: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1246, [1, 0])
    mm_293: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1115, view_397);  permute_1115 = view_397 = None
    permute_1116: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_293, [1, 0]);  mm_293 = None
    sum_349: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1246, [0], True);  view_1246 = None
    view_1247: "f32[3072]" = torch.ops.aten.reshape.default(sum_349, [3072]);  sum_349 = None
    permute_1117: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1116, [1, 0]);  permute_1116 = None
    view_1248: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_292, [8, 576, 768]);  mm_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_895: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1248, primals_395);  primals_395 = None
    mul_896: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_895, 768)
    sum_350: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_895, [2], True)
    mul_897: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_895, mul_194);  mul_895 = None
    sum_351: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_897, [2], True);  mul_897 = None
    mul_898: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_194, sum_351);  sum_351 = None
    sub_241: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_896, sum_350);  mul_896 = sum_350 = None
    sub_242: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_241, mul_898);  sub_241 = mul_898 = None
    mul_899: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_73, sub_242);  div_73 = sub_242 = None
    mul_900: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1248, mul_194);  mul_194 = None
    sum_352: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_900, [0, 1]);  mul_900 = None
    sum_353: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1248, [0, 1]);  view_1248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_453: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_450, mul_899);  add_450 = mul_899 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_901: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_453, primals_40);  primals_40 = None
    mul_902: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_453, view_396);  view_396 = None
    sum_354: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_902, [0, 1], True);  mul_902 = None
    view_1249: "f32[768]" = torch.ops.aten.reshape.default(sum_354, [768]);  sum_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1250: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_901, [4608, 768]);  mul_901 = None
    mm_294: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1250, permute_1118);  permute_1118 = None
    permute_1119: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1250, [1, 0])
    mm_295: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1119, view_395);  permute_1119 = view_395 = None
    permute_1120: "f32[768, 768]" = torch.ops.aten.permute.default(mm_295, [1, 0]);  mm_295 = None
    sum_355: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1250, [0], True);  view_1250 = None
    view_1251: "f32[768]" = torch.ops.aten.reshape.default(sum_355, [768]);  sum_355 = None
    permute_1121: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1120, [1, 0]);  permute_1120 = None
    view_1252: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_294, [8, 576, 768]);  mm_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1253: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1252, [8, 576, 16, 48]);  view_1252 = None
    permute_1122: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1253, [0, 2, 1, 3]);  view_1253 = None
    clone_641: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1122, memory_format = torch.contiguous_format);  permute_1122 = None
    view_1254: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_641, [128, 576, 48]);  clone_641 = None
    bmm_136: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1123, view_1254);  permute_1123 = None
    bmm_137: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1254, permute_1124);  view_1254 = permute_1124 = None
    view_1255: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_136, [8, 16, 576, 48]);  bmm_136 = None
    view_1256: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_137, [8, 16, 576, 576]);  bmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1125: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1256, [0, 2, 3, 1]);  view_1256 = None
    sum_356: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1125, [0, 1, 2], True)
    view_1257: "f32[16]" = torch.ops.aten.reshape.default(sum_356, [16]);  sum_356 = None
    clone_642: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1125, memory_format = torch.contiguous_format);  permute_1125 = None
    view_1258: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_642, [2654208, 16]);  clone_642 = None
    permute_1126: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1258, [1, 0])
    mm_296: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1126, view_389);  permute_1126 = view_389 = None
    permute_1127: "f32[16, 16]" = torch.ops.aten.permute.default(mm_296, [1, 0]);  mm_296 = None
    mm_297: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1258, permute_1128);  view_1258 = permute_1128 = None
    view_1259: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_297, [8, 576, 576, 16]);  mm_297 = None
    permute_1129: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1127, [1, 0]);  permute_1127 = None
    permute_1130: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1259, [0, 3, 1, 2]);  view_1259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_903: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1130, alias_56);  permute_1130 = None
    sum_357: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_903, [-1], True)
    mul_904: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_56, sum_357);  alias_56 = sum_357 = None
    sub_243: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_903, mul_904);  mul_903 = mul_904 = None
    clone_643: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_243, memory_format = torch.contiguous_format);  sub_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1131: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_643, [0, 2, 3, 1]);  clone_643 = None
    sum_358: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1131, [0, 1, 2], True)
    view_1260: "f32[16]" = torch.ops.aten.reshape.default(sum_358, [16]);  sum_358 = None
    clone_644: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1131, memory_format = torch.contiguous_format);  permute_1131 = None
    view_1261: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_644, [2654208, 16]);  clone_644 = None
    permute_1132: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1261, [1, 0])
    mm_298: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1132, view_387);  permute_1132 = view_387 = None
    permute_1133: "f32[16, 16]" = torch.ops.aten.permute.default(mm_298, [1, 0]);  mm_298 = None
    mm_299: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1261, permute_1134);  view_1261 = permute_1134 = None
    view_1262: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_299, [8, 576, 576, 16]);  mm_299 = None
    permute_1135: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1133, [1, 0]);  permute_1133 = None
    permute_1136: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1262, [0, 3, 1, 2]);  view_1262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_645: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1136, memory_format = torch.contiguous_format);  permute_1136 = None
    view_1263: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_645, [128, 576, 576]);  clone_645 = None
    bmm_138: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1137, view_1263);  permute_1137 = None
    bmm_139: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1263, permute_1138);  view_1263 = permute_1138 = None
    view_1264: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_138, [8, 16, 48, 576]);  bmm_138 = None
    view_1265: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_139, [8, 16, 576, 48]);  bmm_139 = None
    permute_1139: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1264, [0, 1, 3, 2]);  view_1264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_51: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1255, 0, 2);  view_1255 = None
    select_scatter_52: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1139, 0, 1);  permute_1139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_454: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_51, select_scatter_52);  select_scatter_51 = select_scatter_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_905: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1265, 0.14433756729740643);  view_1265 = None
    select_scatter_53: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_905, 0, 0);  mul_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_455: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_454, select_scatter_53);  add_454 = select_scatter_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1140: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_455, [1, 3, 0, 2, 4]);  add_455 = None
    clone_646: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1140, memory_format = torch.contiguous_format);  permute_1140 = None
    view_1266: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_646, [8, 576, 2304]);  clone_646 = None
    view_1267: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1266, [4608, 2304]);  view_1266 = None
    mm_300: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1267, permute_1141);  permute_1141 = None
    permute_1142: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1267, [1, 0])
    mm_301: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1142, view_381);  permute_1142 = view_381 = None
    permute_1143: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_301, [1, 0]);  mm_301 = None
    sum_359: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1267, [0], True);  view_1267 = None
    view_1268: "f32[2304]" = torch.ops.aten.reshape.default(sum_359, [2304]);  sum_359 = None
    permute_1144: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1143, [1, 0]);  permute_1143 = None
    view_1269: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_300, [8, 576, 768]);  mm_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_907: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1269, primals_385);  primals_385 = None
    mul_908: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_907, 768)
    sum_360: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_907, [2], True)
    mul_909: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_907, mul_190);  mul_907 = None
    sum_361: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_909, [2], True);  mul_909 = None
    mul_910: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_190, sum_361);  sum_361 = None
    sub_245: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_908, sum_360);  mul_908 = sum_360 = None
    sub_246: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_245, mul_910);  sub_245 = mul_910 = None
    mul_911: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_74, sub_246);  div_74 = sub_246 = None
    mul_912: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1269, mul_190);  mul_190 = None
    sum_362: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_912, [0, 1]);  mul_912 = None
    sum_363: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1269, [0, 1]);  view_1269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_456: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_453, mul_911);  add_453 = mul_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_913: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_456, primals_39);  primals_39 = None
    mul_914: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_456, view_380);  view_380 = None
    sum_364: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_914, [0, 1], True);  mul_914 = None
    view_1270: "f32[768]" = torch.ops.aten.reshape.default(sum_364, [768]);  sum_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1271: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_913, [4608, 768]);  mul_913 = None
    mm_302: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1271, permute_1145);  permute_1145 = None
    permute_1146: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1271, [1, 0])
    mm_303: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1146, view_379);  permute_1146 = view_379 = None
    permute_1147: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_303, [1, 0]);  mm_303 = None
    sum_365: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1271, [0], True);  view_1271 = None
    view_1272: "f32[768]" = torch.ops.aten.reshape.default(sum_365, [768]);  sum_365 = None
    permute_1148: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1147, [1, 0]);  permute_1147 = None
    view_1273: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_302, [8, 576, 3072]);  mm_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_916: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_170, 0.5);  add_170 = None
    mul_917: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_378, view_378)
    mul_918: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_917, -0.5);  mul_917 = None
    exp_55: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_918);  mul_918 = None
    mul_919: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_55, 0.3989422804014327);  exp_55 = None
    mul_920: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_378, mul_919);  view_378 = mul_919 = None
    add_458: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_916, mul_920);  mul_916 = mul_920 = None
    mul_921: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1273, add_458);  view_1273 = add_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1274: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_921, [4608, 3072]);  mul_921 = None
    mm_304: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1274, permute_1149);  permute_1149 = None
    permute_1150: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1274, [1, 0])
    mm_305: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1150, view_377);  permute_1150 = view_377 = None
    permute_1151: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_305, [1, 0]);  mm_305 = None
    sum_366: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1274, [0], True);  view_1274 = None
    view_1275: "f32[3072]" = torch.ops.aten.reshape.default(sum_366, [3072]);  sum_366 = None
    permute_1152: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1151, [1, 0]);  permute_1151 = None
    view_1276: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_304, [8, 576, 768]);  mm_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_923: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1276, primals_379);  primals_379 = None
    mul_924: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_923, 768)
    sum_367: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_923, [2], True)
    mul_925: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_923, mul_184);  mul_923 = None
    sum_368: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_925, [2], True);  mul_925 = None
    mul_926: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_184, sum_368);  sum_368 = None
    sub_248: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_924, sum_367);  mul_924 = sum_367 = None
    sub_249: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_248, mul_926);  sub_248 = mul_926 = None
    mul_927: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_75, sub_249);  div_75 = sub_249 = None
    mul_928: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1276, mul_184);  mul_184 = None
    sum_369: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_928, [0, 1]);  mul_928 = None
    sum_370: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1276, [0, 1]);  view_1276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_459: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_456, mul_927);  add_456 = mul_927 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_929: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_459, primals_38);  primals_38 = None
    mul_930: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_459, view_376);  view_376 = None
    sum_371: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_930, [0, 1], True);  mul_930 = None
    view_1277: "f32[768]" = torch.ops.aten.reshape.default(sum_371, [768]);  sum_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1278: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_929, [4608, 768]);  mul_929 = None
    mm_306: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1278, permute_1153);  permute_1153 = None
    permute_1154: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1278, [1, 0])
    mm_307: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1154, view_375);  permute_1154 = view_375 = None
    permute_1155: "f32[768, 768]" = torch.ops.aten.permute.default(mm_307, [1, 0]);  mm_307 = None
    sum_372: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1278, [0], True);  view_1278 = None
    view_1279: "f32[768]" = torch.ops.aten.reshape.default(sum_372, [768]);  sum_372 = None
    permute_1156: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1155, [1, 0]);  permute_1155 = None
    view_1280: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_306, [8, 576, 768]);  mm_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1281: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1280, [8, 576, 16, 48]);  view_1280 = None
    permute_1157: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1281, [0, 2, 1, 3]);  view_1281 = None
    clone_649: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1157, memory_format = torch.contiguous_format);  permute_1157 = None
    view_1282: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_649, [128, 576, 48]);  clone_649 = None
    bmm_140: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1158, view_1282);  permute_1158 = None
    bmm_141: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1282, permute_1159);  view_1282 = permute_1159 = None
    view_1283: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_140, [8, 16, 576, 48]);  bmm_140 = None
    view_1284: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_141, [8, 16, 576, 576]);  bmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1160: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1284, [0, 2, 3, 1]);  view_1284 = None
    sum_373: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1160, [0, 1, 2], True)
    view_1285: "f32[16]" = torch.ops.aten.reshape.default(sum_373, [16]);  sum_373 = None
    clone_650: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1160, memory_format = torch.contiguous_format);  permute_1160 = None
    view_1286: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_650, [2654208, 16]);  clone_650 = None
    permute_1161: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1286, [1, 0])
    mm_308: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1161, view_369);  permute_1161 = view_369 = None
    permute_1162: "f32[16, 16]" = torch.ops.aten.permute.default(mm_308, [1, 0]);  mm_308 = None
    mm_309: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1286, permute_1163);  view_1286 = permute_1163 = None
    view_1287: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_309, [8, 576, 576, 16]);  mm_309 = None
    permute_1164: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1162, [1, 0]);  permute_1162 = None
    permute_1165: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1287, [0, 3, 1, 2]);  view_1287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_931: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1165, alias_57);  permute_1165 = None
    sum_374: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_931, [-1], True)
    mul_932: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_57, sum_374);  alias_57 = sum_374 = None
    sub_250: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_931, mul_932);  mul_931 = mul_932 = None
    clone_651: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_250, memory_format = torch.contiguous_format);  sub_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1166: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_651, [0, 2, 3, 1]);  clone_651 = None
    sum_375: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1166, [0, 1, 2], True)
    view_1288: "f32[16]" = torch.ops.aten.reshape.default(sum_375, [16]);  sum_375 = None
    clone_652: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1166, memory_format = torch.contiguous_format);  permute_1166 = None
    view_1289: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_652, [2654208, 16]);  clone_652 = None
    permute_1167: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1289, [1, 0])
    mm_310: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1167, view_367);  permute_1167 = view_367 = None
    permute_1168: "f32[16, 16]" = torch.ops.aten.permute.default(mm_310, [1, 0]);  mm_310 = None
    mm_311: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1289, permute_1169);  view_1289 = permute_1169 = None
    view_1290: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_311, [8, 576, 576, 16]);  mm_311 = None
    permute_1170: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1168, [1, 0]);  permute_1168 = None
    permute_1171: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1290, [0, 3, 1, 2]);  view_1290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_653: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1171, memory_format = torch.contiguous_format);  permute_1171 = None
    view_1291: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_653, [128, 576, 576]);  clone_653 = None
    bmm_142: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1172, view_1291);  permute_1172 = None
    bmm_143: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1291, permute_1173);  view_1291 = permute_1173 = None
    view_1292: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_142, [8, 16, 48, 576]);  bmm_142 = None
    view_1293: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_143, [8, 16, 576, 48]);  bmm_143 = None
    permute_1174: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1292, [0, 1, 3, 2]);  view_1292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_54: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1283, 0, 2);  view_1283 = None
    select_scatter_55: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1174, 0, 1);  permute_1174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_460: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_54, select_scatter_55);  select_scatter_54 = select_scatter_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_933: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1293, 0.14433756729740643);  view_1293 = None
    select_scatter_56: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_933, 0, 0);  mul_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_461: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_460, select_scatter_56);  add_460 = select_scatter_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1175: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_461, [1, 3, 0, 2, 4]);  add_461 = None
    clone_654: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1175, memory_format = torch.contiguous_format);  permute_1175 = None
    view_1294: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_654, [8, 576, 2304]);  clone_654 = None
    view_1295: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1294, [4608, 2304]);  view_1294 = None
    mm_312: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1295, permute_1176);  permute_1176 = None
    permute_1177: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1295, [1, 0])
    mm_313: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1177, view_361);  permute_1177 = view_361 = None
    permute_1178: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_313, [1, 0]);  mm_313 = None
    sum_376: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1295, [0], True);  view_1295 = None
    view_1296: "f32[2304]" = torch.ops.aten.reshape.default(sum_376, [2304]);  sum_376 = None
    permute_1179: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1178, [1, 0]);  permute_1178 = None
    view_1297: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_312, [8, 576, 768]);  mm_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_935: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1297, primals_369);  primals_369 = None
    mul_936: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_935, 768)
    sum_377: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_935, [2], True)
    mul_937: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_935, mul_180);  mul_935 = None
    sum_378: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_937, [2], True);  mul_937 = None
    mul_938: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_180, sum_378);  sum_378 = None
    sub_252: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_936, sum_377);  mul_936 = sum_377 = None
    sub_253: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_252, mul_938);  sub_252 = mul_938 = None
    mul_939: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_76, sub_253);  div_76 = sub_253 = None
    mul_940: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1297, mul_180);  mul_180 = None
    sum_379: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_940, [0, 1]);  mul_940 = None
    sum_380: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1297, [0, 1]);  view_1297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_462: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_459, mul_939);  add_459 = mul_939 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_941: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_462, primals_37);  primals_37 = None
    mul_942: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_462, view_360);  view_360 = None
    sum_381: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_942, [0, 1], True);  mul_942 = None
    view_1298: "f32[768]" = torch.ops.aten.reshape.default(sum_381, [768]);  sum_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1299: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_941, [4608, 768]);  mul_941 = None
    mm_314: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1299, permute_1180);  permute_1180 = None
    permute_1181: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1299, [1, 0])
    mm_315: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1181, view_359);  permute_1181 = view_359 = None
    permute_1182: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_315, [1, 0]);  mm_315 = None
    sum_382: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1299, [0], True);  view_1299 = None
    view_1300: "f32[768]" = torch.ops.aten.reshape.default(sum_382, [768]);  sum_382 = None
    permute_1183: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1182, [1, 0]);  permute_1182 = None
    view_1301: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_314, [8, 576, 3072]);  mm_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_944: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_161, 0.5);  add_161 = None
    mul_945: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_358, view_358)
    mul_946: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_945, -0.5);  mul_945 = None
    exp_56: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_946);  mul_946 = None
    mul_947: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_56, 0.3989422804014327);  exp_56 = None
    mul_948: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_358, mul_947);  view_358 = mul_947 = None
    add_464: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_944, mul_948);  mul_944 = mul_948 = None
    mul_949: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1301, add_464);  view_1301 = add_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1302: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_949, [4608, 3072]);  mul_949 = None
    mm_316: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1302, permute_1184);  permute_1184 = None
    permute_1185: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1302, [1, 0])
    mm_317: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1185, view_357);  permute_1185 = view_357 = None
    permute_1186: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_317, [1, 0]);  mm_317 = None
    sum_383: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1302, [0], True);  view_1302 = None
    view_1303: "f32[3072]" = torch.ops.aten.reshape.default(sum_383, [3072]);  sum_383 = None
    permute_1187: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1186, [1, 0]);  permute_1186 = None
    view_1304: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_316, [8, 576, 768]);  mm_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_951: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1304, primals_363);  primals_363 = None
    mul_952: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_951, 768)
    sum_384: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_951, [2], True)
    mul_953: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_951, mul_174);  mul_951 = None
    sum_385: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_953, [2], True);  mul_953 = None
    mul_954: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_174, sum_385);  sum_385 = None
    sub_255: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_952, sum_384);  mul_952 = sum_384 = None
    sub_256: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_255, mul_954);  sub_255 = mul_954 = None
    mul_955: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_77, sub_256);  div_77 = sub_256 = None
    mul_956: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1304, mul_174);  mul_174 = None
    sum_386: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_956, [0, 1]);  mul_956 = None
    sum_387: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1304, [0, 1]);  view_1304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_465: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_462, mul_955);  add_462 = mul_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_957: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_465, primals_36);  primals_36 = None
    mul_958: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_465, view_356);  view_356 = None
    sum_388: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_958, [0, 1], True);  mul_958 = None
    view_1305: "f32[768]" = torch.ops.aten.reshape.default(sum_388, [768]);  sum_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1306: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_957, [4608, 768]);  mul_957 = None
    mm_318: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1306, permute_1188);  permute_1188 = None
    permute_1189: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1306, [1, 0])
    mm_319: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1189, view_355);  permute_1189 = view_355 = None
    permute_1190: "f32[768, 768]" = torch.ops.aten.permute.default(mm_319, [1, 0]);  mm_319 = None
    sum_389: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1306, [0], True);  view_1306 = None
    view_1307: "f32[768]" = torch.ops.aten.reshape.default(sum_389, [768]);  sum_389 = None
    permute_1191: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1190, [1, 0]);  permute_1190 = None
    view_1308: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_318, [8, 576, 768]);  mm_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1309: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1308, [8, 576, 16, 48]);  view_1308 = None
    permute_1192: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1309, [0, 2, 1, 3]);  view_1309 = None
    clone_657: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1192, memory_format = torch.contiguous_format);  permute_1192 = None
    view_1310: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_657, [128, 576, 48]);  clone_657 = None
    bmm_144: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1193, view_1310);  permute_1193 = None
    bmm_145: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1310, permute_1194);  view_1310 = permute_1194 = None
    view_1311: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_144, [8, 16, 576, 48]);  bmm_144 = None
    view_1312: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_145, [8, 16, 576, 576]);  bmm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1195: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1312, [0, 2, 3, 1]);  view_1312 = None
    sum_390: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1195, [0, 1, 2], True)
    view_1313: "f32[16]" = torch.ops.aten.reshape.default(sum_390, [16]);  sum_390 = None
    clone_658: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1195, memory_format = torch.contiguous_format);  permute_1195 = None
    view_1314: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_658, [2654208, 16]);  clone_658 = None
    permute_1196: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1314, [1, 0])
    mm_320: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1196, view_349);  permute_1196 = view_349 = None
    permute_1197: "f32[16, 16]" = torch.ops.aten.permute.default(mm_320, [1, 0]);  mm_320 = None
    mm_321: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1314, permute_1198);  view_1314 = permute_1198 = None
    view_1315: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_321, [8, 576, 576, 16]);  mm_321 = None
    permute_1199: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1197, [1, 0]);  permute_1197 = None
    permute_1200: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1315, [0, 3, 1, 2]);  view_1315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_959: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1200, alias_58);  permute_1200 = None
    sum_391: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_959, [-1], True)
    mul_960: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_58, sum_391);  alias_58 = sum_391 = None
    sub_257: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_959, mul_960);  mul_959 = mul_960 = None
    clone_659: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_257, memory_format = torch.contiguous_format);  sub_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1201: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_659, [0, 2, 3, 1]);  clone_659 = None
    sum_392: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1201, [0, 1, 2], True)
    view_1316: "f32[16]" = torch.ops.aten.reshape.default(sum_392, [16]);  sum_392 = None
    clone_660: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1201, memory_format = torch.contiguous_format);  permute_1201 = None
    view_1317: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_660, [2654208, 16]);  clone_660 = None
    permute_1202: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1317, [1, 0])
    mm_322: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1202, view_347);  permute_1202 = view_347 = None
    permute_1203: "f32[16, 16]" = torch.ops.aten.permute.default(mm_322, [1, 0]);  mm_322 = None
    mm_323: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1317, permute_1204);  view_1317 = permute_1204 = None
    view_1318: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_323, [8, 576, 576, 16]);  mm_323 = None
    permute_1205: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1203, [1, 0]);  permute_1203 = None
    permute_1206: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1318, [0, 3, 1, 2]);  view_1318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_661: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1206, memory_format = torch.contiguous_format);  permute_1206 = None
    view_1319: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_661, [128, 576, 576]);  clone_661 = None
    bmm_146: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1207, view_1319);  permute_1207 = None
    bmm_147: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1319, permute_1208);  view_1319 = permute_1208 = None
    view_1320: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_146, [8, 16, 48, 576]);  bmm_146 = None
    view_1321: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_147, [8, 16, 576, 48]);  bmm_147 = None
    permute_1209: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1320, [0, 1, 3, 2]);  view_1320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_57: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1311, 0, 2);  view_1311 = None
    select_scatter_58: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1209, 0, 1);  permute_1209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_466: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_57, select_scatter_58);  select_scatter_57 = select_scatter_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_961: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1321, 0.14433756729740643);  view_1321 = None
    select_scatter_59: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_961, 0, 0);  mul_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_467: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_466, select_scatter_59);  add_466 = select_scatter_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1210: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_467, [1, 3, 0, 2, 4]);  add_467 = None
    clone_662: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1210, memory_format = torch.contiguous_format);  permute_1210 = None
    view_1322: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_662, [8, 576, 2304]);  clone_662 = None
    view_1323: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1322, [4608, 2304]);  view_1322 = None
    mm_324: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1323, permute_1211);  permute_1211 = None
    permute_1212: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1323, [1, 0])
    mm_325: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1212, view_341);  permute_1212 = view_341 = None
    permute_1213: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_325, [1, 0]);  mm_325 = None
    sum_393: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1323, [0], True);  view_1323 = None
    view_1324: "f32[2304]" = torch.ops.aten.reshape.default(sum_393, [2304]);  sum_393 = None
    permute_1214: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1213, [1, 0]);  permute_1213 = None
    view_1325: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_324, [8, 576, 768]);  mm_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_963: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1325, primals_353);  primals_353 = None
    mul_964: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_963, 768)
    sum_394: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_963, [2], True)
    mul_965: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_963, mul_170);  mul_963 = None
    sum_395: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_965, [2], True);  mul_965 = None
    mul_966: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_170, sum_395);  sum_395 = None
    sub_259: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_964, sum_394);  mul_964 = sum_394 = None
    sub_260: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_259, mul_966);  sub_259 = mul_966 = None
    mul_967: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_78, sub_260);  div_78 = sub_260 = None
    mul_968: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1325, mul_170);  mul_170 = None
    sum_396: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_968, [0, 1]);  mul_968 = None
    sum_397: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1325, [0, 1]);  view_1325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_468: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_465, mul_967);  add_465 = mul_967 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_969: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_468, primals_35);  primals_35 = None
    mul_970: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_468, view_340);  view_340 = None
    sum_398: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_970, [0, 1], True);  mul_970 = None
    view_1326: "f32[768]" = torch.ops.aten.reshape.default(sum_398, [768]);  sum_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1327: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_969, [4608, 768]);  mul_969 = None
    mm_326: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1327, permute_1215);  permute_1215 = None
    permute_1216: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1327, [1, 0])
    mm_327: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1216, view_339);  permute_1216 = view_339 = None
    permute_1217: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_327, [1, 0]);  mm_327 = None
    sum_399: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1327, [0], True);  view_1327 = None
    view_1328: "f32[768]" = torch.ops.aten.reshape.default(sum_399, [768]);  sum_399 = None
    permute_1218: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1217, [1, 0]);  permute_1217 = None
    view_1329: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_326, [8, 576, 3072]);  mm_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_972: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_152, 0.5);  add_152 = None
    mul_973: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_338, view_338)
    mul_974: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_973, -0.5);  mul_973 = None
    exp_57: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_974);  mul_974 = None
    mul_975: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_57, 0.3989422804014327);  exp_57 = None
    mul_976: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_338, mul_975);  view_338 = mul_975 = None
    add_470: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_972, mul_976);  mul_972 = mul_976 = None
    mul_977: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1329, add_470);  view_1329 = add_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1330: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_977, [4608, 3072]);  mul_977 = None
    mm_328: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1330, permute_1219);  permute_1219 = None
    permute_1220: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1330, [1, 0])
    mm_329: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1220, view_337);  permute_1220 = view_337 = None
    permute_1221: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_329, [1, 0]);  mm_329 = None
    sum_400: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1330, [0], True);  view_1330 = None
    view_1331: "f32[3072]" = torch.ops.aten.reshape.default(sum_400, [3072]);  sum_400 = None
    permute_1222: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1221, [1, 0]);  permute_1221 = None
    view_1332: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_328, [8, 576, 768]);  mm_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_979: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1332, primals_347);  primals_347 = None
    mul_980: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_979, 768)
    sum_401: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_979, [2], True)
    mul_981: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_979, mul_164);  mul_979 = None
    sum_402: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_981, [2], True);  mul_981 = None
    mul_982: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_164, sum_402);  sum_402 = None
    sub_262: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_980, sum_401);  mul_980 = sum_401 = None
    sub_263: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_262, mul_982);  sub_262 = mul_982 = None
    mul_983: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_79, sub_263);  div_79 = sub_263 = None
    mul_984: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1332, mul_164);  mul_164 = None
    sum_403: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_984, [0, 1]);  mul_984 = None
    sum_404: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1332, [0, 1]);  view_1332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_471: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_468, mul_983);  add_468 = mul_983 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_985: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_471, primals_34);  primals_34 = None
    mul_986: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_471, view_336);  view_336 = None
    sum_405: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_986, [0, 1], True);  mul_986 = None
    view_1333: "f32[768]" = torch.ops.aten.reshape.default(sum_405, [768]);  sum_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1334: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_985, [4608, 768]);  mul_985 = None
    mm_330: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1334, permute_1223);  permute_1223 = None
    permute_1224: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1334, [1, 0])
    mm_331: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1224, view_335);  permute_1224 = view_335 = None
    permute_1225: "f32[768, 768]" = torch.ops.aten.permute.default(mm_331, [1, 0]);  mm_331 = None
    sum_406: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1334, [0], True);  view_1334 = None
    view_1335: "f32[768]" = torch.ops.aten.reshape.default(sum_406, [768]);  sum_406 = None
    permute_1226: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1225, [1, 0]);  permute_1225 = None
    view_1336: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_330, [8, 576, 768]);  mm_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1337: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1336, [8, 576, 16, 48]);  view_1336 = None
    permute_1227: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1337, [0, 2, 1, 3]);  view_1337 = None
    clone_665: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1227, memory_format = torch.contiguous_format);  permute_1227 = None
    view_1338: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_665, [128, 576, 48]);  clone_665 = None
    bmm_148: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1228, view_1338);  permute_1228 = None
    bmm_149: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1338, permute_1229);  view_1338 = permute_1229 = None
    view_1339: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_148, [8, 16, 576, 48]);  bmm_148 = None
    view_1340: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_149, [8, 16, 576, 576]);  bmm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1230: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1340, [0, 2, 3, 1]);  view_1340 = None
    sum_407: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1230, [0, 1, 2], True)
    view_1341: "f32[16]" = torch.ops.aten.reshape.default(sum_407, [16]);  sum_407 = None
    clone_666: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1230, memory_format = torch.contiguous_format);  permute_1230 = None
    view_1342: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_666, [2654208, 16]);  clone_666 = None
    permute_1231: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1342, [1, 0])
    mm_332: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1231, view_329);  permute_1231 = view_329 = None
    permute_1232: "f32[16, 16]" = torch.ops.aten.permute.default(mm_332, [1, 0]);  mm_332 = None
    mm_333: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1342, permute_1233);  view_1342 = permute_1233 = None
    view_1343: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_333, [8, 576, 576, 16]);  mm_333 = None
    permute_1234: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1232, [1, 0]);  permute_1232 = None
    permute_1235: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1343, [0, 3, 1, 2]);  view_1343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_987: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1235, alias_59);  permute_1235 = None
    sum_408: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_987, [-1], True)
    mul_988: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_59, sum_408);  alias_59 = sum_408 = None
    sub_264: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_987, mul_988);  mul_987 = mul_988 = None
    clone_667: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_264, memory_format = torch.contiguous_format);  sub_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1236: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_667, [0, 2, 3, 1]);  clone_667 = None
    sum_409: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1236, [0, 1, 2], True)
    view_1344: "f32[16]" = torch.ops.aten.reshape.default(sum_409, [16]);  sum_409 = None
    clone_668: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1236, memory_format = torch.contiguous_format);  permute_1236 = None
    view_1345: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_668, [2654208, 16]);  clone_668 = None
    permute_1237: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1345, [1, 0])
    mm_334: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1237, view_327);  permute_1237 = view_327 = None
    permute_1238: "f32[16, 16]" = torch.ops.aten.permute.default(mm_334, [1, 0]);  mm_334 = None
    mm_335: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1345, permute_1239);  view_1345 = permute_1239 = None
    view_1346: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_335, [8, 576, 576, 16]);  mm_335 = None
    permute_1240: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1238, [1, 0]);  permute_1238 = None
    permute_1241: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1346, [0, 3, 1, 2]);  view_1346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_669: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1241, memory_format = torch.contiguous_format);  permute_1241 = None
    view_1347: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_669, [128, 576, 576]);  clone_669 = None
    bmm_150: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1242, view_1347);  permute_1242 = None
    bmm_151: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1347, permute_1243);  view_1347 = permute_1243 = None
    view_1348: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_150, [8, 16, 48, 576]);  bmm_150 = None
    view_1349: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_151, [8, 16, 576, 48]);  bmm_151 = None
    permute_1244: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1348, [0, 1, 3, 2]);  view_1348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_60: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1339, 0, 2);  view_1339 = None
    select_scatter_61: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1244, 0, 1);  permute_1244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_472: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_60, select_scatter_61);  select_scatter_60 = select_scatter_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_989: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1349, 0.14433756729740643);  view_1349 = None
    select_scatter_62: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_989, 0, 0);  mul_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_473: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_472, select_scatter_62);  add_472 = select_scatter_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1245: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_473, [1, 3, 0, 2, 4]);  add_473 = None
    clone_670: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1245, memory_format = torch.contiguous_format);  permute_1245 = None
    view_1350: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_670, [8, 576, 2304]);  clone_670 = None
    view_1351: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1350, [4608, 2304]);  view_1350 = None
    mm_336: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1351, permute_1246);  permute_1246 = None
    permute_1247: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1351, [1, 0])
    mm_337: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1247, view_321);  permute_1247 = view_321 = None
    permute_1248: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_337, [1, 0]);  mm_337 = None
    sum_410: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1351, [0], True);  view_1351 = None
    view_1352: "f32[2304]" = torch.ops.aten.reshape.default(sum_410, [2304]);  sum_410 = None
    permute_1249: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1248, [1, 0]);  permute_1248 = None
    view_1353: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_336, [8, 576, 768]);  mm_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_991: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1353, primals_337);  primals_337 = None
    mul_992: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_991, 768)
    sum_411: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_991, [2], True)
    mul_993: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_991, mul_160);  mul_991 = None
    sum_412: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_993, [2], True);  mul_993 = None
    mul_994: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_160, sum_412);  sum_412 = None
    sub_266: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_992, sum_411);  mul_992 = sum_411 = None
    sub_267: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_266, mul_994);  sub_266 = mul_994 = None
    mul_995: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_80, sub_267);  div_80 = sub_267 = None
    mul_996: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1353, mul_160);  mul_160 = None
    sum_413: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_996, [0, 1]);  mul_996 = None
    sum_414: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1353, [0, 1]);  view_1353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_474: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_471, mul_995);  add_471 = mul_995 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_997: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_474, primals_33);  primals_33 = None
    mul_998: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_474, view_320);  view_320 = None
    sum_415: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_998, [0, 1], True);  mul_998 = None
    view_1354: "f32[768]" = torch.ops.aten.reshape.default(sum_415, [768]);  sum_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1355: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_997, [4608, 768]);  mul_997 = None
    mm_338: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1355, permute_1250);  permute_1250 = None
    permute_1251: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1355, [1, 0])
    mm_339: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1251, view_319);  permute_1251 = view_319 = None
    permute_1252: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_339, [1, 0]);  mm_339 = None
    sum_416: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1355, [0], True);  view_1355 = None
    view_1356: "f32[768]" = torch.ops.aten.reshape.default(sum_416, [768]);  sum_416 = None
    permute_1253: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1252, [1, 0]);  permute_1252 = None
    view_1357: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_338, [8, 576, 3072]);  mm_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1000: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_143, 0.5);  add_143 = None
    mul_1001: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_318, view_318)
    mul_1002: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1001, -0.5);  mul_1001 = None
    exp_58: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1002);  mul_1002 = None
    mul_1003: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_58, 0.3989422804014327);  exp_58 = None
    mul_1004: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_318, mul_1003);  view_318 = mul_1003 = None
    add_476: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1000, mul_1004);  mul_1000 = mul_1004 = None
    mul_1005: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1357, add_476);  view_1357 = add_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1358: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1005, [4608, 3072]);  mul_1005 = None
    mm_340: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1358, permute_1254);  permute_1254 = None
    permute_1255: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1358, [1, 0])
    mm_341: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1255, view_317);  permute_1255 = view_317 = None
    permute_1256: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_341, [1, 0]);  mm_341 = None
    sum_417: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1358, [0], True);  view_1358 = None
    view_1359: "f32[3072]" = torch.ops.aten.reshape.default(sum_417, [3072]);  sum_417 = None
    permute_1257: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1256, [1, 0]);  permute_1256 = None
    view_1360: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_340, [8, 576, 768]);  mm_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1007: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1360, primals_331);  primals_331 = None
    mul_1008: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1007, 768)
    sum_418: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1007, [2], True)
    mul_1009: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1007, mul_154);  mul_1007 = None
    sum_419: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1009, [2], True);  mul_1009 = None
    mul_1010: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_154, sum_419);  sum_419 = None
    sub_269: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1008, sum_418);  mul_1008 = sum_418 = None
    sub_270: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_269, mul_1010);  sub_269 = mul_1010 = None
    mul_1011: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_81, sub_270);  div_81 = sub_270 = None
    mul_1012: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1360, mul_154);  mul_154 = None
    sum_420: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1012, [0, 1]);  mul_1012 = None
    sum_421: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1360, [0, 1]);  view_1360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_477: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_474, mul_1011);  add_474 = mul_1011 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1013: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_477, primals_32);  primals_32 = None
    mul_1014: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_477, view_316);  view_316 = None
    sum_422: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1014, [0, 1], True);  mul_1014 = None
    view_1361: "f32[768]" = torch.ops.aten.reshape.default(sum_422, [768]);  sum_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1362: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1013, [4608, 768]);  mul_1013 = None
    mm_342: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1362, permute_1258);  permute_1258 = None
    permute_1259: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1362, [1, 0])
    mm_343: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1259, view_315);  permute_1259 = view_315 = None
    permute_1260: "f32[768, 768]" = torch.ops.aten.permute.default(mm_343, [1, 0]);  mm_343 = None
    sum_423: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1362, [0], True);  view_1362 = None
    view_1363: "f32[768]" = torch.ops.aten.reshape.default(sum_423, [768]);  sum_423 = None
    permute_1261: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1260, [1, 0]);  permute_1260 = None
    view_1364: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_342, [8, 576, 768]);  mm_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1365: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1364, [8, 576, 16, 48]);  view_1364 = None
    permute_1262: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1365, [0, 2, 1, 3]);  view_1365 = None
    clone_673: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1262, memory_format = torch.contiguous_format);  permute_1262 = None
    view_1366: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_673, [128, 576, 48]);  clone_673 = None
    bmm_152: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1263, view_1366);  permute_1263 = None
    bmm_153: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1366, permute_1264);  view_1366 = permute_1264 = None
    view_1367: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_152, [8, 16, 576, 48]);  bmm_152 = None
    view_1368: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_153, [8, 16, 576, 576]);  bmm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1265: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1368, [0, 2, 3, 1]);  view_1368 = None
    sum_424: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1265, [0, 1, 2], True)
    view_1369: "f32[16]" = torch.ops.aten.reshape.default(sum_424, [16]);  sum_424 = None
    clone_674: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1265, memory_format = torch.contiguous_format);  permute_1265 = None
    view_1370: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_674, [2654208, 16]);  clone_674 = None
    permute_1266: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1370, [1, 0])
    mm_344: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1266, view_309);  permute_1266 = view_309 = None
    permute_1267: "f32[16, 16]" = torch.ops.aten.permute.default(mm_344, [1, 0]);  mm_344 = None
    mm_345: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1370, permute_1268);  view_1370 = permute_1268 = None
    view_1371: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_345, [8, 576, 576, 16]);  mm_345 = None
    permute_1269: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1267, [1, 0]);  permute_1267 = None
    permute_1270: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1371, [0, 3, 1, 2]);  view_1371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1015: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1270, alias_60);  permute_1270 = None
    sum_425: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1015, [-1], True)
    mul_1016: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_60, sum_425);  alias_60 = sum_425 = None
    sub_271: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1015, mul_1016);  mul_1015 = mul_1016 = None
    clone_675: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_271, memory_format = torch.contiguous_format);  sub_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1271: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_675, [0, 2, 3, 1]);  clone_675 = None
    sum_426: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1271, [0, 1, 2], True)
    view_1372: "f32[16]" = torch.ops.aten.reshape.default(sum_426, [16]);  sum_426 = None
    clone_676: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1271, memory_format = torch.contiguous_format);  permute_1271 = None
    view_1373: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_676, [2654208, 16]);  clone_676 = None
    permute_1272: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1373, [1, 0])
    mm_346: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1272, view_307);  permute_1272 = view_307 = None
    permute_1273: "f32[16, 16]" = torch.ops.aten.permute.default(mm_346, [1, 0]);  mm_346 = None
    mm_347: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1373, permute_1274);  view_1373 = permute_1274 = None
    view_1374: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_347, [8, 576, 576, 16]);  mm_347 = None
    permute_1275: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1273, [1, 0]);  permute_1273 = None
    permute_1276: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1374, [0, 3, 1, 2]);  view_1374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_677: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1276, memory_format = torch.contiguous_format);  permute_1276 = None
    view_1375: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_677, [128, 576, 576]);  clone_677 = None
    bmm_154: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1277, view_1375);  permute_1277 = None
    bmm_155: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1375, permute_1278);  view_1375 = permute_1278 = None
    view_1376: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_154, [8, 16, 48, 576]);  bmm_154 = None
    view_1377: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_155, [8, 16, 576, 48]);  bmm_155 = None
    permute_1279: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1376, [0, 1, 3, 2]);  view_1376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_63: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1367, 0, 2);  view_1367 = None
    select_scatter_64: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1279, 0, 1);  permute_1279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_478: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_63, select_scatter_64);  select_scatter_63 = select_scatter_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1017: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1377, 0.14433756729740643);  view_1377 = None
    select_scatter_65: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1017, 0, 0);  mul_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_479: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_478, select_scatter_65);  add_478 = select_scatter_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1280: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_479, [1, 3, 0, 2, 4]);  add_479 = None
    clone_678: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1280, memory_format = torch.contiguous_format);  permute_1280 = None
    view_1378: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_678, [8, 576, 2304]);  clone_678 = None
    view_1379: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1378, [4608, 2304]);  view_1378 = None
    mm_348: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1379, permute_1281);  permute_1281 = None
    permute_1282: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1379, [1, 0])
    mm_349: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1282, view_301);  permute_1282 = view_301 = None
    permute_1283: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_349, [1, 0]);  mm_349 = None
    sum_427: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1379, [0], True);  view_1379 = None
    view_1380: "f32[2304]" = torch.ops.aten.reshape.default(sum_427, [2304]);  sum_427 = None
    permute_1284: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1283, [1, 0]);  permute_1283 = None
    view_1381: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_348, [8, 576, 768]);  mm_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1019: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1381, primals_321);  primals_321 = None
    mul_1020: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1019, 768)
    sum_428: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1019, [2], True)
    mul_1021: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1019, mul_150);  mul_1019 = None
    sum_429: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1021, [2], True);  mul_1021 = None
    mul_1022: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_150, sum_429);  sum_429 = None
    sub_273: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1020, sum_428);  mul_1020 = sum_428 = None
    sub_274: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_273, mul_1022);  sub_273 = mul_1022 = None
    mul_1023: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_82, sub_274);  div_82 = sub_274 = None
    mul_1024: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1381, mul_150);  mul_150 = None
    sum_430: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1024, [0, 1]);  mul_1024 = None
    sum_431: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1381, [0, 1]);  view_1381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_480: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_477, mul_1023);  add_477 = mul_1023 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1025: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_480, primals_31);  primals_31 = None
    mul_1026: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_480, view_300);  view_300 = None
    sum_432: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1026, [0, 1], True);  mul_1026 = None
    view_1382: "f32[768]" = torch.ops.aten.reshape.default(sum_432, [768]);  sum_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1383: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1025, [4608, 768]);  mul_1025 = None
    mm_350: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1383, permute_1285);  permute_1285 = None
    permute_1286: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1383, [1, 0])
    mm_351: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1286, view_299);  permute_1286 = view_299 = None
    permute_1287: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_351, [1, 0]);  mm_351 = None
    sum_433: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1383, [0], True);  view_1383 = None
    view_1384: "f32[768]" = torch.ops.aten.reshape.default(sum_433, [768]);  sum_433 = None
    permute_1288: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1287, [1, 0]);  permute_1287 = None
    view_1385: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_350, [8, 576, 3072]);  mm_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1028: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_134, 0.5);  add_134 = None
    mul_1029: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_298, view_298)
    mul_1030: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1029, -0.5);  mul_1029 = None
    exp_59: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1030);  mul_1030 = None
    mul_1031: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_59, 0.3989422804014327);  exp_59 = None
    mul_1032: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_298, mul_1031);  view_298 = mul_1031 = None
    add_482: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1028, mul_1032);  mul_1028 = mul_1032 = None
    mul_1033: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1385, add_482);  view_1385 = add_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1386: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1033, [4608, 3072]);  mul_1033 = None
    mm_352: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1386, permute_1289);  permute_1289 = None
    permute_1290: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1386, [1, 0])
    mm_353: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1290, view_297);  permute_1290 = view_297 = None
    permute_1291: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_353, [1, 0]);  mm_353 = None
    sum_434: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1386, [0], True);  view_1386 = None
    view_1387: "f32[3072]" = torch.ops.aten.reshape.default(sum_434, [3072]);  sum_434 = None
    permute_1292: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1291, [1, 0]);  permute_1291 = None
    view_1388: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_352, [8, 576, 768]);  mm_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1035: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1388, primals_315);  primals_315 = None
    mul_1036: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1035, 768)
    sum_435: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1035, [2], True)
    mul_1037: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1035, mul_144);  mul_1035 = None
    sum_436: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1037, [2], True);  mul_1037 = None
    mul_1038: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_144, sum_436);  sum_436 = None
    sub_276: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1036, sum_435);  mul_1036 = sum_435 = None
    sub_277: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_276, mul_1038);  sub_276 = mul_1038 = None
    mul_1039: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_83, sub_277);  div_83 = sub_277 = None
    mul_1040: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1388, mul_144);  mul_144 = None
    sum_437: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1040, [0, 1]);  mul_1040 = None
    sum_438: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1388, [0, 1]);  view_1388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_483: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_480, mul_1039);  add_480 = mul_1039 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1041: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_483, primals_30);  primals_30 = None
    mul_1042: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_483, view_296);  view_296 = None
    sum_439: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1042, [0, 1], True);  mul_1042 = None
    view_1389: "f32[768]" = torch.ops.aten.reshape.default(sum_439, [768]);  sum_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1390: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1041, [4608, 768]);  mul_1041 = None
    mm_354: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1390, permute_1293);  permute_1293 = None
    permute_1294: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1390, [1, 0])
    mm_355: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1294, view_295);  permute_1294 = view_295 = None
    permute_1295: "f32[768, 768]" = torch.ops.aten.permute.default(mm_355, [1, 0]);  mm_355 = None
    sum_440: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1390, [0], True);  view_1390 = None
    view_1391: "f32[768]" = torch.ops.aten.reshape.default(sum_440, [768]);  sum_440 = None
    permute_1296: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1295, [1, 0]);  permute_1295 = None
    view_1392: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_354, [8, 576, 768]);  mm_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1393: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1392, [8, 576, 16, 48]);  view_1392 = None
    permute_1297: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1393, [0, 2, 1, 3]);  view_1393 = None
    clone_681: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1297, memory_format = torch.contiguous_format);  permute_1297 = None
    view_1394: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_681, [128, 576, 48]);  clone_681 = None
    bmm_156: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1298, view_1394);  permute_1298 = None
    bmm_157: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1394, permute_1299);  view_1394 = permute_1299 = None
    view_1395: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_156, [8, 16, 576, 48]);  bmm_156 = None
    view_1396: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_157, [8, 16, 576, 576]);  bmm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1300: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1396, [0, 2, 3, 1]);  view_1396 = None
    sum_441: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1300, [0, 1, 2], True)
    view_1397: "f32[16]" = torch.ops.aten.reshape.default(sum_441, [16]);  sum_441 = None
    clone_682: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1300, memory_format = torch.contiguous_format);  permute_1300 = None
    view_1398: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_682, [2654208, 16]);  clone_682 = None
    permute_1301: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1398, [1, 0])
    mm_356: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1301, view_289);  permute_1301 = view_289 = None
    permute_1302: "f32[16, 16]" = torch.ops.aten.permute.default(mm_356, [1, 0]);  mm_356 = None
    mm_357: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1398, permute_1303);  view_1398 = permute_1303 = None
    view_1399: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_357, [8, 576, 576, 16]);  mm_357 = None
    permute_1304: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1302, [1, 0]);  permute_1302 = None
    permute_1305: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1399, [0, 3, 1, 2]);  view_1399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1043: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1305, alias_61);  permute_1305 = None
    sum_442: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1043, [-1], True)
    mul_1044: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_61, sum_442);  alias_61 = sum_442 = None
    sub_278: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1043, mul_1044);  mul_1043 = mul_1044 = None
    clone_683: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_278, memory_format = torch.contiguous_format);  sub_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1306: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_683, [0, 2, 3, 1]);  clone_683 = None
    sum_443: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1306, [0, 1, 2], True)
    view_1400: "f32[16]" = torch.ops.aten.reshape.default(sum_443, [16]);  sum_443 = None
    clone_684: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1306, memory_format = torch.contiguous_format);  permute_1306 = None
    view_1401: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_684, [2654208, 16]);  clone_684 = None
    permute_1307: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1401, [1, 0])
    mm_358: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1307, view_287);  permute_1307 = view_287 = None
    permute_1308: "f32[16, 16]" = torch.ops.aten.permute.default(mm_358, [1, 0]);  mm_358 = None
    mm_359: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1401, permute_1309);  view_1401 = permute_1309 = None
    view_1402: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_359, [8, 576, 576, 16]);  mm_359 = None
    permute_1310: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1308, [1, 0]);  permute_1308 = None
    permute_1311: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1402, [0, 3, 1, 2]);  view_1402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_685: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1311, memory_format = torch.contiguous_format);  permute_1311 = None
    view_1403: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_685, [128, 576, 576]);  clone_685 = None
    bmm_158: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1312, view_1403);  permute_1312 = None
    bmm_159: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1403, permute_1313);  view_1403 = permute_1313 = None
    view_1404: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_158, [8, 16, 48, 576]);  bmm_158 = None
    view_1405: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_159, [8, 16, 576, 48]);  bmm_159 = None
    permute_1314: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1404, [0, 1, 3, 2]);  view_1404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_66: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1395, 0, 2);  view_1395 = None
    select_scatter_67: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1314, 0, 1);  permute_1314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_484: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_66, select_scatter_67);  select_scatter_66 = select_scatter_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1045: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1405, 0.14433756729740643);  view_1405 = None
    select_scatter_68: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1045, 0, 0);  mul_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_485: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_484, select_scatter_68);  add_484 = select_scatter_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1315: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_485, [1, 3, 0, 2, 4]);  add_485 = None
    clone_686: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1315, memory_format = torch.contiguous_format);  permute_1315 = None
    view_1406: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_686, [8, 576, 2304]);  clone_686 = None
    view_1407: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1406, [4608, 2304]);  view_1406 = None
    mm_360: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1407, permute_1316);  permute_1316 = None
    permute_1317: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1407, [1, 0])
    mm_361: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1317, view_281);  permute_1317 = view_281 = None
    permute_1318: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_361, [1, 0]);  mm_361 = None
    sum_444: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1407, [0], True);  view_1407 = None
    view_1408: "f32[2304]" = torch.ops.aten.reshape.default(sum_444, [2304]);  sum_444 = None
    permute_1319: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1318, [1, 0]);  permute_1318 = None
    view_1409: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_360, [8, 576, 768]);  mm_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1047: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1409, primals_305);  primals_305 = None
    mul_1048: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1047, 768)
    sum_445: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1047, [2], True)
    mul_1049: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1047, mul_140);  mul_1047 = None
    sum_446: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1049, [2], True);  mul_1049 = None
    mul_1050: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_140, sum_446);  sum_446 = None
    sub_280: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1048, sum_445);  mul_1048 = sum_445 = None
    sub_281: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_280, mul_1050);  sub_280 = mul_1050 = None
    mul_1051: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_84, sub_281);  div_84 = sub_281 = None
    mul_1052: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1409, mul_140);  mul_140 = None
    sum_447: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1052, [0, 1]);  mul_1052 = None
    sum_448: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1409, [0, 1]);  view_1409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_486: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_483, mul_1051);  add_483 = mul_1051 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1053: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_486, primals_29);  primals_29 = None
    mul_1054: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_486, view_280);  view_280 = None
    sum_449: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1054, [0, 1], True);  mul_1054 = None
    view_1410: "f32[768]" = torch.ops.aten.reshape.default(sum_449, [768]);  sum_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1411: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1053, [4608, 768]);  mul_1053 = None
    mm_362: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1411, permute_1320);  permute_1320 = None
    permute_1321: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1411, [1, 0])
    mm_363: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1321, view_279);  permute_1321 = view_279 = None
    permute_1322: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_363, [1, 0]);  mm_363 = None
    sum_450: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1411, [0], True);  view_1411 = None
    view_1412: "f32[768]" = torch.ops.aten.reshape.default(sum_450, [768]);  sum_450 = None
    permute_1323: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1322, [1, 0]);  permute_1322 = None
    view_1413: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_362, [8, 576, 3072]);  mm_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1056: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_125, 0.5);  add_125 = None
    mul_1057: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_278, view_278)
    mul_1058: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1057, -0.5);  mul_1057 = None
    exp_60: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1058);  mul_1058 = None
    mul_1059: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_60, 0.3989422804014327);  exp_60 = None
    mul_1060: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_278, mul_1059);  view_278 = mul_1059 = None
    add_488: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1056, mul_1060);  mul_1056 = mul_1060 = None
    mul_1061: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1413, add_488);  view_1413 = add_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1414: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1061, [4608, 3072]);  mul_1061 = None
    mm_364: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1414, permute_1324);  permute_1324 = None
    permute_1325: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1414, [1, 0])
    mm_365: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1325, view_277);  permute_1325 = view_277 = None
    permute_1326: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_365, [1, 0]);  mm_365 = None
    sum_451: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1414, [0], True);  view_1414 = None
    view_1415: "f32[3072]" = torch.ops.aten.reshape.default(sum_451, [3072]);  sum_451 = None
    permute_1327: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1326, [1, 0]);  permute_1326 = None
    view_1416: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_364, [8, 576, 768]);  mm_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1063: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1416, primals_299);  primals_299 = None
    mul_1064: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1063, 768)
    sum_452: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1063, [2], True)
    mul_1065: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1063, mul_134);  mul_1063 = None
    sum_453: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1065, [2], True);  mul_1065 = None
    mul_1066: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_134, sum_453);  sum_453 = None
    sub_283: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1064, sum_452);  mul_1064 = sum_452 = None
    sub_284: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_283, mul_1066);  sub_283 = mul_1066 = None
    mul_1067: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_85, sub_284);  div_85 = sub_284 = None
    mul_1068: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1416, mul_134);  mul_134 = None
    sum_454: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1068, [0, 1]);  mul_1068 = None
    sum_455: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1416, [0, 1]);  view_1416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_489: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_486, mul_1067);  add_486 = mul_1067 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1069: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_489, primals_28);  primals_28 = None
    mul_1070: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_489, view_276);  view_276 = None
    sum_456: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1070, [0, 1], True);  mul_1070 = None
    view_1417: "f32[768]" = torch.ops.aten.reshape.default(sum_456, [768]);  sum_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1418: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1069, [4608, 768]);  mul_1069 = None
    mm_366: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1418, permute_1328);  permute_1328 = None
    permute_1329: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1418, [1, 0])
    mm_367: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1329, view_275);  permute_1329 = view_275 = None
    permute_1330: "f32[768, 768]" = torch.ops.aten.permute.default(mm_367, [1, 0]);  mm_367 = None
    sum_457: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1418, [0], True);  view_1418 = None
    view_1419: "f32[768]" = torch.ops.aten.reshape.default(sum_457, [768]);  sum_457 = None
    permute_1331: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1330, [1, 0]);  permute_1330 = None
    view_1420: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_366, [8, 576, 768]);  mm_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1421: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1420, [8, 576, 16, 48]);  view_1420 = None
    permute_1332: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1421, [0, 2, 1, 3]);  view_1421 = None
    clone_689: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1332, memory_format = torch.contiguous_format);  permute_1332 = None
    view_1422: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_689, [128, 576, 48]);  clone_689 = None
    bmm_160: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1333, view_1422);  permute_1333 = None
    bmm_161: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1422, permute_1334);  view_1422 = permute_1334 = None
    view_1423: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_160, [8, 16, 576, 48]);  bmm_160 = None
    view_1424: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_161, [8, 16, 576, 576]);  bmm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1335: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1424, [0, 2, 3, 1]);  view_1424 = None
    sum_458: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1335, [0, 1, 2], True)
    view_1425: "f32[16]" = torch.ops.aten.reshape.default(sum_458, [16]);  sum_458 = None
    clone_690: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1335, memory_format = torch.contiguous_format);  permute_1335 = None
    view_1426: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_690, [2654208, 16]);  clone_690 = None
    permute_1336: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1426, [1, 0])
    mm_368: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1336, view_269);  permute_1336 = view_269 = None
    permute_1337: "f32[16, 16]" = torch.ops.aten.permute.default(mm_368, [1, 0]);  mm_368 = None
    mm_369: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1426, permute_1338);  view_1426 = permute_1338 = None
    view_1427: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_369, [8, 576, 576, 16]);  mm_369 = None
    permute_1339: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1337, [1, 0]);  permute_1337 = None
    permute_1340: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1427, [0, 3, 1, 2]);  view_1427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1071: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1340, alias_62);  permute_1340 = None
    sum_459: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1071, [-1], True)
    mul_1072: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_62, sum_459);  alias_62 = sum_459 = None
    sub_285: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1071, mul_1072);  mul_1071 = mul_1072 = None
    clone_691: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_285, memory_format = torch.contiguous_format);  sub_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1341: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_691, [0, 2, 3, 1]);  clone_691 = None
    sum_460: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1341, [0, 1, 2], True)
    view_1428: "f32[16]" = torch.ops.aten.reshape.default(sum_460, [16]);  sum_460 = None
    clone_692: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1341, memory_format = torch.contiguous_format);  permute_1341 = None
    view_1429: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_692, [2654208, 16]);  clone_692 = None
    permute_1342: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1429, [1, 0])
    mm_370: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1342, view_267);  permute_1342 = view_267 = None
    permute_1343: "f32[16, 16]" = torch.ops.aten.permute.default(mm_370, [1, 0]);  mm_370 = None
    mm_371: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1429, permute_1344);  view_1429 = permute_1344 = None
    view_1430: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_371, [8, 576, 576, 16]);  mm_371 = None
    permute_1345: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1343, [1, 0]);  permute_1343 = None
    permute_1346: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1430, [0, 3, 1, 2]);  view_1430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_693: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1346, memory_format = torch.contiguous_format);  permute_1346 = None
    view_1431: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_693, [128, 576, 576]);  clone_693 = None
    bmm_162: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1347, view_1431);  permute_1347 = None
    bmm_163: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1431, permute_1348);  view_1431 = permute_1348 = None
    view_1432: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_162, [8, 16, 48, 576]);  bmm_162 = None
    view_1433: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_163, [8, 16, 576, 48]);  bmm_163 = None
    permute_1349: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1432, [0, 1, 3, 2]);  view_1432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_69: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1423, 0, 2);  view_1423 = None
    select_scatter_70: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1349, 0, 1);  permute_1349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_490: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_69, select_scatter_70);  select_scatter_69 = select_scatter_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1073: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1433, 0.14433756729740643);  view_1433 = None
    select_scatter_71: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1073, 0, 0);  mul_1073 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_491: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_490, select_scatter_71);  add_490 = select_scatter_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1350: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_491, [1, 3, 0, 2, 4]);  add_491 = None
    clone_694: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1350, memory_format = torch.contiguous_format);  permute_1350 = None
    view_1434: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_694, [8, 576, 2304]);  clone_694 = None
    view_1435: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1434, [4608, 2304]);  view_1434 = None
    mm_372: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1435, permute_1351);  permute_1351 = None
    permute_1352: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1435, [1, 0])
    mm_373: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1352, view_261);  permute_1352 = view_261 = None
    permute_1353: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_373, [1, 0]);  mm_373 = None
    sum_461: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1435, [0], True);  view_1435 = None
    view_1436: "f32[2304]" = torch.ops.aten.reshape.default(sum_461, [2304]);  sum_461 = None
    permute_1354: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1353, [1, 0]);  permute_1353 = None
    view_1437: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_372, [8, 576, 768]);  mm_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1075: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1437, primals_289);  primals_289 = None
    mul_1076: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1075, 768)
    sum_462: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1075, [2], True)
    mul_1077: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1075, mul_130);  mul_1075 = None
    sum_463: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1077, [2], True);  mul_1077 = None
    mul_1078: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_130, sum_463);  sum_463 = None
    sub_287: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1076, sum_462);  mul_1076 = sum_462 = None
    sub_288: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_287, mul_1078);  sub_287 = mul_1078 = None
    mul_1079: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_86, sub_288);  div_86 = sub_288 = None
    mul_1080: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1437, mul_130);  mul_130 = None
    sum_464: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1080, [0, 1]);  mul_1080 = None
    sum_465: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1437, [0, 1]);  view_1437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_492: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_489, mul_1079);  add_489 = mul_1079 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1081: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_492, primals_27);  primals_27 = None
    mul_1082: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_492, view_260);  view_260 = None
    sum_466: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1082, [0, 1], True);  mul_1082 = None
    view_1438: "f32[768]" = torch.ops.aten.reshape.default(sum_466, [768]);  sum_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1439: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1081, [4608, 768]);  mul_1081 = None
    mm_374: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1439, permute_1355);  permute_1355 = None
    permute_1356: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1439, [1, 0])
    mm_375: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1356, view_259);  permute_1356 = view_259 = None
    permute_1357: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_375, [1, 0]);  mm_375 = None
    sum_467: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1439, [0], True);  view_1439 = None
    view_1440: "f32[768]" = torch.ops.aten.reshape.default(sum_467, [768]);  sum_467 = None
    permute_1358: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1357, [1, 0]);  permute_1357 = None
    view_1441: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_374, [8, 576, 3072]);  mm_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1084: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_116, 0.5);  add_116 = None
    mul_1085: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_258, view_258)
    mul_1086: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1085, -0.5);  mul_1085 = None
    exp_61: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1086);  mul_1086 = None
    mul_1087: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_61, 0.3989422804014327);  exp_61 = None
    mul_1088: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_258, mul_1087);  view_258 = mul_1087 = None
    add_494: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1084, mul_1088);  mul_1084 = mul_1088 = None
    mul_1089: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1441, add_494);  view_1441 = add_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1442: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1089, [4608, 3072]);  mul_1089 = None
    mm_376: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1442, permute_1359);  permute_1359 = None
    permute_1360: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1442, [1, 0])
    mm_377: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1360, view_257);  permute_1360 = view_257 = None
    permute_1361: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_377, [1, 0]);  mm_377 = None
    sum_468: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1442, [0], True);  view_1442 = None
    view_1443: "f32[3072]" = torch.ops.aten.reshape.default(sum_468, [3072]);  sum_468 = None
    permute_1362: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1361, [1, 0]);  permute_1361 = None
    view_1444: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_376, [8, 576, 768]);  mm_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1091: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1444, primals_283);  primals_283 = None
    mul_1092: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1091, 768)
    sum_469: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1091, [2], True)
    mul_1093: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1091, mul_124);  mul_1091 = None
    sum_470: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1093, [2], True);  mul_1093 = None
    mul_1094: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_124, sum_470);  sum_470 = None
    sub_290: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1092, sum_469);  mul_1092 = sum_469 = None
    sub_291: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_290, mul_1094);  sub_290 = mul_1094 = None
    mul_1095: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_87, sub_291);  div_87 = sub_291 = None
    mul_1096: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1444, mul_124);  mul_124 = None
    sum_471: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1096, [0, 1]);  mul_1096 = None
    sum_472: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1444, [0, 1]);  view_1444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_495: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_492, mul_1095);  add_492 = mul_1095 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1097: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_495, primals_26);  primals_26 = None
    mul_1098: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_495, view_256);  view_256 = None
    sum_473: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1098, [0, 1], True);  mul_1098 = None
    view_1445: "f32[768]" = torch.ops.aten.reshape.default(sum_473, [768]);  sum_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1446: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1097, [4608, 768]);  mul_1097 = None
    mm_378: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1446, permute_1363);  permute_1363 = None
    permute_1364: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1446, [1, 0])
    mm_379: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1364, view_255);  permute_1364 = view_255 = None
    permute_1365: "f32[768, 768]" = torch.ops.aten.permute.default(mm_379, [1, 0]);  mm_379 = None
    sum_474: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1446, [0], True);  view_1446 = None
    view_1447: "f32[768]" = torch.ops.aten.reshape.default(sum_474, [768]);  sum_474 = None
    permute_1366: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1365, [1, 0]);  permute_1365 = None
    view_1448: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_378, [8, 576, 768]);  mm_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1449: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1448, [8, 576, 16, 48]);  view_1448 = None
    permute_1367: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1449, [0, 2, 1, 3]);  view_1449 = None
    clone_697: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1367, memory_format = torch.contiguous_format);  permute_1367 = None
    view_1450: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_697, [128, 576, 48]);  clone_697 = None
    bmm_164: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1368, view_1450);  permute_1368 = None
    bmm_165: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1450, permute_1369);  view_1450 = permute_1369 = None
    view_1451: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_164, [8, 16, 576, 48]);  bmm_164 = None
    view_1452: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_165, [8, 16, 576, 576]);  bmm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1370: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1452, [0, 2, 3, 1]);  view_1452 = None
    sum_475: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1370, [0, 1, 2], True)
    view_1453: "f32[16]" = torch.ops.aten.reshape.default(sum_475, [16]);  sum_475 = None
    clone_698: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1370, memory_format = torch.contiguous_format);  permute_1370 = None
    view_1454: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_698, [2654208, 16]);  clone_698 = None
    permute_1371: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1454, [1, 0])
    mm_380: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1371, view_249);  permute_1371 = view_249 = None
    permute_1372: "f32[16, 16]" = torch.ops.aten.permute.default(mm_380, [1, 0]);  mm_380 = None
    mm_381: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1454, permute_1373);  view_1454 = permute_1373 = None
    view_1455: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_381, [8, 576, 576, 16]);  mm_381 = None
    permute_1374: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1372, [1, 0]);  permute_1372 = None
    permute_1375: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1455, [0, 3, 1, 2]);  view_1455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1099: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1375, alias_63);  permute_1375 = None
    sum_476: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1099, [-1], True)
    mul_1100: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_63, sum_476);  alias_63 = sum_476 = None
    sub_292: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1099, mul_1100);  mul_1099 = mul_1100 = None
    clone_699: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_292, memory_format = torch.contiguous_format);  sub_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1376: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_699, [0, 2, 3, 1]);  clone_699 = None
    sum_477: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1376, [0, 1, 2], True)
    view_1456: "f32[16]" = torch.ops.aten.reshape.default(sum_477, [16]);  sum_477 = None
    clone_700: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1376, memory_format = torch.contiguous_format);  permute_1376 = None
    view_1457: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_700, [2654208, 16]);  clone_700 = None
    permute_1377: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1457, [1, 0])
    mm_382: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1377, view_247);  permute_1377 = view_247 = None
    permute_1378: "f32[16, 16]" = torch.ops.aten.permute.default(mm_382, [1, 0]);  mm_382 = None
    mm_383: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1457, permute_1379);  view_1457 = permute_1379 = None
    view_1458: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_383, [8, 576, 576, 16]);  mm_383 = None
    permute_1380: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1378, [1, 0]);  permute_1378 = None
    permute_1381: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1458, [0, 3, 1, 2]);  view_1458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_701: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1381, memory_format = torch.contiguous_format);  permute_1381 = None
    view_1459: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_701, [128, 576, 576]);  clone_701 = None
    bmm_166: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1382, view_1459);  permute_1382 = None
    bmm_167: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1459, permute_1383);  view_1459 = permute_1383 = None
    view_1460: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_166, [8, 16, 48, 576]);  bmm_166 = None
    view_1461: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_167, [8, 16, 576, 48]);  bmm_167 = None
    permute_1384: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1460, [0, 1, 3, 2]);  view_1460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_72: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1451, 0, 2);  view_1451 = None
    select_scatter_73: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1384, 0, 1);  permute_1384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_496: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_72, select_scatter_73);  select_scatter_72 = select_scatter_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1101: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1461, 0.14433756729740643);  view_1461 = None
    select_scatter_74: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1101, 0, 0);  mul_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_497: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_496, select_scatter_74);  add_496 = select_scatter_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1385: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_497, [1, 3, 0, 2, 4]);  add_497 = None
    clone_702: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1385, memory_format = torch.contiguous_format);  permute_1385 = None
    view_1462: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_702, [8, 576, 2304]);  clone_702 = None
    view_1463: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1462, [4608, 2304]);  view_1462 = None
    mm_384: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1463, permute_1386);  permute_1386 = None
    permute_1387: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1463, [1, 0])
    mm_385: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1387, view_241);  permute_1387 = view_241 = None
    permute_1388: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_385, [1, 0]);  mm_385 = None
    sum_478: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1463, [0], True);  view_1463 = None
    view_1464: "f32[2304]" = torch.ops.aten.reshape.default(sum_478, [2304]);  sum_478 = None
    permute_1389: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1388, [1, 0]);  permute_1388 = None
    view_1465: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_384, [8, 576, 768]);  mm_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1103: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1465, primals_273);  primals_273 = None
    mul_1104: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1103, 768)
    sum_479: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1103, [2], True)
    mul_1105: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1103, mul_120);  mul_1103 = None
    sum_480: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1105, [2], True);  mul_1105 = None
    mul_1106: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_120, sum_480);  sum_480 = None
    sub_294: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1104, sum_479);  mul_1104 = sum_479 = None
    sub_295: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_294, mul_1106);  sub_294 = mul_1106 = None
    mul_1107: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_88, sub_295);  div_88 = sub_295 = None
    mul_1108: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1465, mul_120);  mul_120 = None
    sum_481: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1108, [0, 1]);  mul_1108 = None
    sum_482: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1465, [0, 1]);  view_1465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_498: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_495, mul_1107);  add_495 = mul_1107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1109: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_498, primals_25);  primals_25 = None
    mul_1110: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_498, view_240);  view_240 = None
    sum_483: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1110, [0, 1], True);  mul_1110 = None
    view_1466: "f32[768]" = torch.ops.aten.reshape.default(sum_483, [768]);  sum_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1467: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1109, [4608, 768]);  mul_1109 = None
    mm_386: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1467, permute_1390);  permute_1390 = None
    permute_1391: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1467, [1, 0])
    mm_387: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1391, view_239);  permute_1391 = view_239 = None
    permute_1392: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_387, [1, 0]);  mm_387 = None
    sum_484: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1467, [0], True);  view_1467 = None
    view_1468: "f32[768]" = torch.ops.aten.reshape.default(sum_484, [768]);  sum_484 = None
    permute_1393: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1392, [1, 0]);  permute_1392 = None
    view_1469: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_386, [8, 576, 3072]);  mm_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1112: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_107, 0.5);  add_107 = None
    mul_1113: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_238, view_238)
    mul_1114: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1113, -0.5);  mul_1113 = None
    exp_62: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1114);  mul_1114 = None
    mul_1115: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_62, 0.3989422804014327);  exp_62 = None
    mul_1116: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_238, mul_1115);  view_238 = mul_1115 = None
    add_500: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1112, mul_1116);  mul_1112 = mul_1116 = None
    mul_1117: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1469, add_500);  view_1469 = add_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1470: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1117, [4608, 3072]);  mul_1117 = None
    mm_388: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1470, permute_1394);  permute_1394 = None
    permute_1395: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1470, [1, 0])
    mm_389: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1395, view_237);  permute_1395 = view_237 = None
    permute_1396: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_389, [1, 0]);  mm_389 = None
    sum_485: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1470, [0], True);  view_1470 = None
    view_1471: "f32[3072]" = torch.ops.aten.reshape.default(sum_485, [3072]);  sum_485 = None
    permute_1397: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1396, [1, 0]);  permute_1396 = None
    view_1472: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_388, [8, 576, 768]);  mm_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1119: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1472, primals_267);  primals_267 = None
    mul_1120: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1119, 768)
    sum_486: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1119, [2], True)
    mul_1121: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1119, mul_114);  mul_1119 = None
    sum_487: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1121, [2], True);  mul_1121 = None
    mul_1122: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_114, sum_487);  sum_487 = None
    sub_297: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1120, sum_486);  mul_1120 = sum_486 = None
    sub_298: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_297, mul_1122);  sub_297 = mul_1122 = None
    mul_1123: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_89, sub_298);  div_89 = sub_298 = None
    mul_1124: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1472, mul_114);  mul_114 = None
    sum_488: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1124, [0, 1]);  mul_1124 = None
    sum_489: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1472, [0, 1]);  view_1472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_501: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_498, mul_1123);  add_498 = mul_1123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1125: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_501, primals_24);  primals_24 = None
    mul_1126: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_501, view_236);  view_236 = None
    sum_490: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1126, [0, 1], True);  mul_1126 = None
    view_1473: "f32[768]" = torch.ops.aten.reshape.default(sum_490, [768]);  sum_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1474: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1125, [4608, 768]);  mul_1125 = None
    mm_390: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1474, permute_1398);  permute_1398 = None
    permute_1399: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1474, [1, 0])
    mm_391: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1399, view_235);  permute_1399 = view_235 = None
    permute_1400: "f32[768, 768]" = torch.ops.aten.permute.default(mm_391, [1, 0]);  mm_391 = None
    sum_491: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1474, [0], True);  view_1474 = None
    view_1475: "f32[768]" = torch.ops.aten.reshape.default(sum_491, [768]);  sum_491 = None
    permute_1401: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1400, [1, 0]);  permute_1400 = None
    view_1476: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_390, [8, 576, 768]);  mm_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1477: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1476, [8, 576, 16, 48]);  view_1476 = None
    permute_1402: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1477, [0, 2, 1, 3]);  view_1477 = None
    clone_705: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1402, memory_format = torch.contiguous_format);  permute_1402 = None
    view_1478: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_705, [128, 576, 48]);  clone_705 = None
    bmm_168: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1403, view_1478);  permute_1403 = None
    bmm_169: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1478, permute_1404);  view_1478 = permute_1404 = None
    view_1479: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_168, [8, 16, 576, 48]);  bmm_168 = None
    view_1480: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_169, [8, 16, 576, 576]);  bmm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1405: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1480, [0, 2, 3, 1]);  view_1480 = None
    sum_492: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1405, [0, 1, 2], True)
    view_1481: "f32[16]" = torch.ops.aten.reshape.default(sum_492, [16]);  sum_492 = None
    clone_706: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1405, memory_format = torch.contiguous_format);  permute_1405 = None
    view_1482: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_706, [2654208, 16]);  clone_706 = None
    permute_1406: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1482, [1, 0])
    mm_392: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1406, view_229);  permute_1406 = view_229 = None
    permute_1407: "f32[16, 16]" = torch.ops.aten.permute.default(mm_392, [1, 0]);  mm_392 = None
    mm_393: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1482, permute_1408);  view_1482 = permute_1408 = None
    view_1483: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_393, [8, 576, 576, 16]);  mm_393 = None
    permute_1409: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1407, [1, 0]);  permute_1407 = None
    permute_1410: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1483, [0, 3, 1, 2]);  view_1483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1127: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1410, alias_64);  permute_1410 = None
    sum_493: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1127, [-1], True)
    mul_1128: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_64, sum_493);  alias_64 = sum_493 = None
    sub_299: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1127, mul_1128);  mul_1127 = mul_1128 = None
    clone_707: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_299, memory_format = torch.contiguous_format);  sub_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1411: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_707, [0, 2, 3, 1]);  clone_707 = None
    sum_494: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1411, [0, 1, 2], True)
    view_1484: "f32[16]" = torch.ops.aten.reshape.default(sum_494, [16]);  sum_494 = None
    clone_708: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1411, memory_format = torch.contiguous_format);  permute_1411 = None
    view_1485: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_708, [2654208, 16]);  clone_708 = None
    permute_1412: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1485, [1, 0])
    mm_394: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1412, view_227);  permute_1412 = view_227 = None
    permute_1413: "f32[16, 16]" = torch.ops.aten.permute.default(mm_394, [1, 0]);  mm_394 = None
    mm_395: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1485, permute_1414);  view_1485 = permute_1414 = None
    view_1486: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_395, [8, 576, 576, 16]);  mm_395 = None
    permute_1415: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1413, [1, 0]);  permute_1413 = None
    permute_1416: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1486, [0, 3, 1, 2]);  view_1486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_709: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1416, memory_format = torch.contiguous_format);  permute_1416 = None
    view_1487: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_709, [128, 576, 576]);  clone_709 = None
    bmm_170: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1417, view_1487);  permute_1417 = None
    bmm_171: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1487, permute_1418);  view_1487 = permute_1418 = None
    view_1488: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_170, [8, 16, 48, 576]);  bmm_170 = None
    view_1489: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_171, [8, 16, 576, 48]);  bmm_171 = None
    permute_1419: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1488, [0, 1, 3, 2]);  view_1488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_75: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1479, 0, 2);  view_1479 = None
    select_scatter_76: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1419, 0, 1);  permute_1419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_502: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_75, select_scatter_76);  select_scatter_75 = select_scatter_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1129: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1489, 0.14433756729740643);  view_1489 = None
    select_scatter_77: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1129, 0, 0);  mul_1129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_503: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_502, select_scatter_77);  add_502 = select_scatter_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1420: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_503, [1, 3, 0, 2, 4]);  add_503 = None
    clone_710: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1420, memory_format = torch.contiguous_format);  permute_1420 = None
    view_1490: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_710, [8, 576, 2304]);  clone_710 = None
    view_1491: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1490, [4608, 2304]);  view_1490 = None
    mm_396: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1491, permute_1421);  permute_1421 = None
    permute_1422: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1491, [1, 0])
    mm_397: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1422, view_221);  permute_1422 = view_221 = None
    permute_1423: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_397, [1, 0]);  mm_397 = None
    sum_495: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1491, [0], True);  view_1491 = None
    view_1492: "f32[2304]" = torch.ops.aten.reshape.default(sum_495, [2304]);  sum_495 = None
    permute_1424: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1423, [1, 0]);  permute_1423 = None
    view_1493: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_396, [8, 576, 768]);  mm_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1131: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1493, primals_257);  primals_257 = None
    mul_1132: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1131, 768)
    sum_496: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1131, [2], True)
    mul_1133: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1131, mul_110);  mul_1131 = None
    sum_497: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1133, [2], True);  mul_1133 = None
    mul_1134: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_110, sum_497);  sum_497 = None
    sub_301: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1132, sum_496);  mul_1132 = sum_496 = None
    sub_302: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_301, mul_1134);  sub_301 = mul_1134 = None
    mul_1135: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_90, sub_302);  div_90 = sub_302 = None
    mul_1136: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1493, mul_110);  mul_110 = None
    sum_498: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1136, [0, 1]);  mul_1136 = None
    sum_499: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1493, [0, 1]);  view_1493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_504: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_501, mul_1135);  add_501 = mul_1135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1137: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_504, primals_23);  primals_23 = None
    mul_1138: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_504, view_220);  view_220 = None
    sum_500: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1138, [0, 1], True);  mul_1138 = None
    view_1494: "f32[768]" = torch.ops.aten.reshape.default(sum_500, [768]);  sum_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1495: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1137, [4608, 768]);  mul_1137 = None
    mm_398: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1495, permute_1425);  permute_1425 = None
    permute_1426: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1495, [1, 0])
    mm_399: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1426, view_219);  permute_1426 = view_219 = None
    permute_1427: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_399, [1, 0]);  mm_399 = None
    sum_501: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1495, [0], True);  view_1495 = None
    view_1496: "f32[768]" = torch.ops.aten.reshape.default(sum_501, [768]);  sum_501 = None
    permute_1428: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1427, [1, 0]);  permute_1427 = None
    view_1497: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_398, [8, 576, 3072]);  mm_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1140: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_98, 0.5);  add_98 = None
    mul_1141: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_218, view_218)
    mul_1142: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1141, -0.5);  mul_1141 = None
    exp_63: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1142);  mul_1142 = None
    mul_1143: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_63, 0.3989422804014327);  exp_63 = None
    mul_1144: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_218, mul_1143);  view_218 = mul_1143 = None
    add_506: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1140, mul_1144);  mul_1140 = mul_1144 = None
    mul_1145: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1497, add_506);  view_1497 = add_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1498: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1145, [4608, 3072]);  mul_1145 = None
    mm_400: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1498, permute_1429);  permute_1429 = None
    permute_1430: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1498, [1, 0])
    mm_401: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1430, view_217);  permute_1430 = view_217 = None
    permute_1431: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_401, [1, 0]);  mm_401 = None
    sum_502: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1498, [0], True);  view_1498 = None
    view_1499: "f32[3072]" = torch.ops.aten.reshape.default(sum_502, [3072]);  sum_502 = None
    permute_1432: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1431, [1, 0]);  permute_1431 = None
    view_1500: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_400, [8, 576, 768]);  mm_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1147: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1500, primals_251);  primals_251 = None
    mul_1148: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1147, 768)
    sum_503: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1147, [2], True)
    mul_1149: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1147, mul_104);  mul_1147 = None
    sum_504: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1149, [2], True);  mul_1149 = None
    mul_1150: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_104, sum_504);  sum_504 = None
    sub_304: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1148, sum_503);  mul_1148 = sum_503 = None
    sub_305: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_304, mul_1150);  sub_304 = mul_1150 = None
    mul_1151: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_91, sub_305);  div_91 = sub_305 = None
    mul_1152: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1500, mul_104);  mul_104 = None
    sum_505: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1152, [0, 1]);  mul_1152 = None
    sum_506: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1500, [0, 1]);  view_1500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_507: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_504, mul_1151);  add_504 = mul_1151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1153: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_507, primals_22);  primals_22 = None
    mul_1154: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_507, view_216);  view_216 = None
    sum_507: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1154, [0, 1], True);  mul_1154 = None
    view_1501: "f32[768]" = torch.ops.aten.reshape.default(sum_507, [768]);  sum_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1502: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1153, [4608, 768]);  mul_1153 = None
    mm_402: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1502, permute_1433);  permute_1433 = None
    permute_1434: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1502, [1, 0])
    mm_403: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1434, view_215);  permute_1434 = view_215 = None
    permute_1435: "f32[768, 768]" = torch.ops.aten.permute.default(mm_403, [1, 0]);  mm_403 = None
    sum_508: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1502, [0], True);  view_1502 = None
    view_1503: "f32[768]" = torch.ops.aten.reshape.default(sum_508, [768]);  sum_508 = None
    permute_1436: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1435, [1, 0]);  permute_1435 = None
    view_1504: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_402, [8, 576, 768]);  mm_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1505: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1504, [8, 576, 16, 48]);  view_1504 = None
    permute_1437: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1505, [0, 2, 1, 3]);  view_1505 = None
    clone_713: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1437, memory_format = torch.contiguous_format);  permute_1437 = None
    view_1506: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_713, [128, 576, 48]);  clone_713 = None
    bmm_172: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1438, view_1506);  permute_1438 = None
    bmm_173: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1506, permute_1439);  view_1506 = permute_1439 = None
    view_1507: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_172, [8, 16, 576, 48]);  bmm_172 = None
    view_1508: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_173, [8, 16, 576, 576]);  bmm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1440: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1508, [0, 2, 3, 1]);  view_1508 = None
    sum_509: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1440, [0, 1, 2], True)
    view_1509: "f32[16]" = torch.ops.aten.reshape.default(sum_509, [16]);  sum_509 = None
    clone_714: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1440, memory_format = torch.contiguous_format);  permute_1440 = None
    view_1510: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_714, [2654208, 16]);  clone_714 = None
    permute_1441: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1510, [1, 0])
    mm_404: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1441, view_209);  permute_1441 = view_209 = None
    permute_1442: "f32[16, 16]" = torch.ops.aten.permute.default(mm_404, [1, 0]);  mm_404 = None
    mm_405: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1510, permute_1443);  view_1510 = permute_1443 = None
    view_1511: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_405, [8, 576, 576, 16]);  mm_405 = None
    permute_1444: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1442, [1, 0]);  permute_1442 = None
    permute_1445: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1511, [0, 3, 1, 2]);  view_1511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1155: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1445, alias_65);  permute_1445 = None
    sum_510: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1155, [-1], True)
    mul_1156: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_65, sum_510);  alias_65 = sum_510 = None
    sub_306: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1155, mul_1156);  mul_1155 = mul_1156 = None
    clone_715: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_306, memory_format = torch.contiguous_format);  sub_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1446: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_715, [0, 2, 3, 1]);  clone_715 = None
    sum_511: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1446, [0, 1, 2], True)
    view_1512: "f32[16]" = torch.ops.aten.reshape.default(sum_511, [16]);  sum_511 = None
    clone_716: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1446, memory_format = torch.contiguous_format);  permute_1446 = None
    view_1513: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_716, [2654208, 16]);  clone_716 = None
    permute_1447: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1513, [1, 0])
    mm_406: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1447, view_207);  permute_1447 = view_207 = None
    permute_1448: "f32[16, 16]" = torch.ops.aten.permute.default(mm_406, [1, 0]);  mm_406 = None
    mm_407: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1513, permute_1449);  view_1513 = permute_1449 = None
    view_1514: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_407, [8, 576, 576, 16]);  mm_407 = None
    permute_1450: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1448, [1, 0]);  permute_1448 = None
    permute_1451: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1514, [0, 3, 1, 2]);  view_1514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_717: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1451, memory_format = torch.contiguous_format);  permute_1451 = None
    view_1515: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_717, [128, 576, 576]);  clone_717 = None
    bmm_174: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1452, view_1515);  permute_1452 = None
    bmm_175: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1515, permute_1453);  view_1515 = permute_1453 = None
    view_1516: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_174, [8, 16, 48, 576]);  bmm_174 = None
    view_1517: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_175, [8, 16, 576, 48]);  bmm_175 = None
    permute_1454: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1516, [0, 1, 3, 2]);  view_1516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_78: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1507, 0, 2);  view_1507 = None
    select_scatter_79: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1454, 0, 1);  permute_1454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_508: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_78, select_scatter_79);  select_scatter_78 = select_scatter_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1157: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1517, 0.14433756729740643);  view_1517 = None
    select_scatter_80: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1157, 0, 0);  mul_1157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_509: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_508, select_scatter_80);  add_508 = select_scatter_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1455: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_509, [1, 3, 0, 2, 4]);  add_509 = None
    clone_718: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1455, memory_format = torch.contiguous_format);  permute_1455 = None
    view_1518: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_718, [8, 576, 2304]);  clone_718 = None
    view_1519: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1518, [4608, 2304]);  view_1518 = None
    mm_408: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1519, permute_1456);  permute_1456 = None
    permute_1457: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1519, [1, 0])
    mm_409: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1457, view_201);  permute_1457 = view_201 = None
    permute_1458: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_409, [1, 0]);  mm_409 = None
    sum_512: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1519, [0], True);  view_1519 = None
    view_1520: "f32[2304]" = torch.ops.aten.reshape.default(sum_512, [2304]);  sum_512 = None
    permute_1459: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1458, [1, 0]);  permute_1458 = None
    view_1521: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_408, [8, 576, 768]);  mm_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1159: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1521, primals_241);  primals_241 = None
    mul_1160: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1159, 768)
    sum_513: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1159, [2], True)
    mul_1161: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1159, mul_100);  mul_1159 = None
    sum_514: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1161, [2], True);  mul_1161 = None
    mul_1162: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_100, sum_514);  sum_514 = None
    sub_308: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1160, sum_513);  mul_1160 = sum_513 = None
    sub_309: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_308, mul_1162);  sub_308 = mul_1162 = None
    mul_1163: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_92, sub_309);  div_92 = sub_309 = None
    mul_1164: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1521, mul_100);  mul_100 = None
    sum_515: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1164, [0, 1]);  mul_1164 = None
    sum_516: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1521, [0, 1]);  view_1521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_510: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_507, mul_1163);  add_507 = mul_1163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1165: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_510, primals_21);  primals_21 = None
    mul_1166: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_510, view_200);  view_200 = None
    sum_517: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1166, [0, 1], True);  mul_1166 = None
    view_1522: "f32[768]" = torch.ops.aten.reshape.default(sum_517, [768]);  sum_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1523: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1165, [4608, 768]);  mul_1165 = None
    mm_410: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1523, permute_1460);  permute_1460 = None
    permute_1461: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1523, [1, 0])
    mm_411: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1461, view_199);  permute_1461 = view_199 = None
    permute_1462: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_411, [1, 0]);  mm_411 = None
    sum_518: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1523, [0], True);  view_1523 = None
    view_1524: "f32[768]" = torch.ops.aten.reshape.default(sum_518, [768]);  sum_518 = None
    permute_1463: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1462, [1, 0]);  permute_1462 = None
    view_1525: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_410, [8, 576, 3072]);  mm_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1168: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_89, 0.5);  add_89 = None
    mul_1169: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_198, view_198)
    mul_1170: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1169, -0.5);  mul_1169 = None
    exp_64: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1170);  mul_1170 = None
    mul_1171: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_64, 0.3989422804014327);  exp_64 = None
    mul_1172: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_198, mul_1171);  view_198 = mul_1171 = None
    add_512: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1168, mul_1172);  mul_1168 = mul_1172 = None
    mul_1173: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1525, add_512);  view_1525 = add_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1526: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1173, [4608, 3072]);  mul_1173 = None
    mm_412: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1526, permute_1464);  permute_1464 = None
    permute_1465: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1526, [1, 0])
    mm_413: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1465, view_197);  permute_1465 = view_197 = None
    permute_1466: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_413, [1, 0]);  mm_413 = None
    sum_519: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1526, [0], True);  view_1526 = None
    view_1527: "f32[3072]" = torch.ops.aten.reshape.default(sum_519, [3072]);  sum_519 = None
    permute_1467: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1466, [1, 0]);  permute_1466 = None
    view_1528: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_412, [8, 576, 768]);  mm_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1175: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1528, primals_235);  primals_235 = None
    mul_1176: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1175, 768)
    sum_520: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1175, [2], True)
    mul_1177: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1175, mul_94);  mul_1175 = None
    sum_521: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1177, [2], True);  mul_1177 = None
    mul_1178: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_94, sum_521);  sum_521 = None
    sub_311: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1176, sum_520);  mul_1176 = sum_520 = None
    sub_312: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_311, mul_1178);  sub_311 = mul_1178 = None
    mul_1179: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_93, sub_312);  div_93 = sub_312 = None
    mul_1180: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1528, mul_94);  mul_94 = None
    sum_522: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1180, [0, 1]);  mul_1180 = None
    sum_523: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1528, [0, 1]);  view_1528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_513: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_510, mul_1179);  add_510 = mul_1179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1181: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_513, primals_20);  primals_20 = None
    mul_1182: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_513, view_196);  view_196 = None
    sum_524: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1182, [0, 1], True);  mul_1182 = None
    view_1529: "f32[768]" = torch.ops.aten.reshape.default(sum_524, [768]);  sum_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1530: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1181, [4608, 768]);  mul_1181 = None
    mm_414: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1530, permute_1468);  permute_1468 = None
    permute_1469: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1530, [1, 0])
    mm_415: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1469, view_195);  permute_1469 = view_195 = None
    permute_1470: "f32[768, 768]" = torch.ops.aten.permute.default(mm_415, [1, 0]);  mm_415 = None
    sum_525: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1530, [0], True);  view_1530 = None
    view_1531: "f32[768]" = torch.ops.aten.reshape.default(sum_525, [768]);  sum_525 = None
    permute_1471: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1470, [1, 0]);  permute_1470 = None
    view_1532: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_414, [8, 576, 768]);  mm_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1533: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1532, [8, 576, 16, 48]);  view_1532 = None
    permute_1472: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1533, [0, 2, 1, 3]);  view_1533 = None
    clone_721: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1472, memory_format = torch.contiguous_format);  permute_1472 = None
    view_1534: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_721, [128, 576, 48]);  clone_721 = None
    bmm_176: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1473, view_1534);  permute_1473 = None
    bmm_177: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1534, permute_1474);  view_1534 = permute_1474 = None
    view_1535: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_176, [8, 16, 576, 48]);  bmm_176 = None
    view_1536: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_177, [8, 16, 576, 576]);  bmm_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1475: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1536, [0, 2, 3, 1]);  view_1536 = None
    sum_526: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1475, [0, 1, 2], True)
    view_1537: "f32[16]" = torch.ops.aten.reshape.default(sum_526, [16]);  sum_526 = None
    clone_722: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1475, memory_format = torch.contiguous_format);  permute_1475 = None
    view_1538: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_722, [2654208, 16]);  clone_722 = None
    permute_1476: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1538, [1, 0])
    mm_416: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1476, view_189);  permute_1476 = view_189 = None
    permute_1477: "f32[16, 16]" = torch.ops.aten.permute.default(mm_416, [1, 0]);  mm_416 = None
    mm_417: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1538, permute_1478);  view_1538 = permute_1478 = None
    view_1539: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_417, [8, 576, 576, 16]);  mm_417 = None
    permute_1479: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1477, [1, 0]);  permute_1477 = None
    permute_1480: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1539, [0, 3, 1, 2]);  view_1539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1183: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1480, alias_66);  permute_1480 = None
    sum_527: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1183, [-1], True)
    mul_1184: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_66, sum_527);  alias_66 = sum_527 = None
    sub_313: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1183, mul_1184);  mul_1183 = mul_1184 = None
    clone_723: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_313, memory_format = torch.contiguous_format);  sub_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1481: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_723, [0, 2, 3, 1]);  clone_723 = None
    sum_528: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1481, [0, 1, 2], True)
    view_1540: "f32[16]" = torch.ops.aten.reshape.default(sum_528, [16]);  sum_528 = None
    clone_724: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1481, memory_format = torch.contiguous_format);  permute_1481 = None
    view_1541: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_724, [2654208, 16]);  clone_724 = None
    permute_1482: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1541, [1, 0])
    mm_418: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1482, view_187);  permute_1482 = view_187 = None
    permute_1483: "f32[16, 16]" = torch.ops.aten.permute.default(mm_418, [1, 0]);  mm_418 = None
    mm_419: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1541, permute_1484);  view_1541 = permute_1484 = None
    view_1542: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_419, [8, 576, 576, 16]);  mm_419 = None
    permute_1485: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1483, [1, 0]);  permute_1483 = None
    permute_1486: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1542, [0, 3, 1, 2]);  view_1542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_725: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1486, memory_format = torch.contiguous_format);  permute_1486 = None
    view_1543: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_725, [128, 576, 576]);  clone_725 = None
    bmm_178: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1487, view_1543);  permute_1487 = None
    bmm_179: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1543, permute_1488);  view_1543 = permute_1488 = None
    view_1544: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_178, [8, 16, 48, 576]);  bmm_178 = None
    view_1545: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_179, [8, 16, 576, 48]);  bmm_179 = None
    permute_1489: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1544, [0, 1, 3, 2]);  view_1544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_81: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1535, 0, 2);  view_1535 = None
    select_scatter_82: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1489, 0, 1);  permute_1489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_514: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_81, select_scatter_82);  select_scatter_81 = select_scatter_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1185: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1545, 0.14433756729740643);  view_1545 = None
    select_scatter_83: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1185, 0, 0);  mul_1185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_515: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_514, select_scatter_83);  add_514 = select_scatter_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1490: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_515, [1, 3, 0, 2, 4]);  add_515 = None
    clone_726: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1490, memory_format = torch.contiguous_format);  permute_1490 = None
    view_1546: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_726, [8, 576, 2304]);  clone_726 = None
    view_1547: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1546, [4608, 2304]);  view_1546 = None
    mm_420: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1547, permute_1491);  permute_1491 = None
    permute_1492: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1547, [1, 0])
    mm_421: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1492, view_181);  permute_1492 = view_181 = None
    permute_1493: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_421, [1, 0]);  mm_421 = None
    sum_529: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1547, [0], True);  view_1547 = None
    view_1548: "f32[2304]" = torch.ops.aten.reshape.default(sum_529, [2304]);  sum_529 = None
    permute_1494: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1493, [1, 0]);  permute_1493 = None
    view_1549: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_420, [8, 576, 768]);  mm_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1187: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1549, primals_225);  primals_225 = None
    mul_1188: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1187, 768)
    sum_530: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1187, [2], True)
    mul_1189: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1187, mul_90);  mul_1187 = None
    sum_531: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1189, [2], True);  mul_1189 = None
    mul_1190: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_90, sum_531);  sum_531 = None
    sub_315: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1188, sum_530);  mul_1188 = sum_530 = None
    sub_316: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_315, mul_1190);  sub_315 = mul_1190 = None
    mul_1191: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_94, sub_316);  div_94 = sub_316 = None
    mul_1192: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1549, mul_90);  mul_90 = None
    sum_532: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1192, [0, 1]);  mul_1192 = None
    sum_533: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1549, [0, 1]);  view_1549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_516: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_513, mul_1191);  add_513 = mul_1191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1193: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_516, primals_19);  primals_19 = None
    mul_1194: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_516, view_180);  view_180 = None
    sum_534: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1194, [0, 1], True);  mul_1194 = None
    view_1550: "f32[768]" = torch.ops.aten.reshape.default(sum_534, [768]);  sum_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1551: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1193, [4608, 768]);  mul_1193 = None
    mm_422: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1551, permute_1495);  permute_1495 = None
    permute_1496: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1551, [1, 0])
    mm_423: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1496, view_179);  permute_1496 = view_179 = None
    permute_1497: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_423, [1, 0]);  mm_423 = None
    sum_535: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1551, [0], True);  view_1551 = None
    view_1552: "f32[768]" = torch.ops.aten.reshape.default(sum_535, [768]);  sum_535 = None
    permute_1498: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1497, [1, 0]);  permute_1497 = None
    view_1553: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_422, [8, 576, 3072]);  mm_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1196: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
    mul_1197: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_178, view_178)
    mul_1198: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1197, -0.5);  mul_1197 = None
    exp_65: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1198);  mul_1198 = None
    mul_1199: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_65, 0.3989422804014327);  exp_65 = None
    mul_1200: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_178, mul_1199);  view_178 = mul_1199 = None
    add_518: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1196, mul_1200);  mul_1196 = mul_1200 = None
    mul_1201: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1553, add_518);  view_1553 = add_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1554: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1201, [4608, 3072]);  mul_1201 = None
    mm_424: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1554, permute_1499);  permute_1499 = None
    permute_1500: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1554, [1, 0])
    mm_425: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1500, view_177);  permute_1500 = view_177 = None
    permute_1501: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_425, [1, 0]);  mm_425 = None
    sum_536: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1554, [0], True);  view_1554 = None
    view_1555: "f32[3072]" = torch.ops.aten.reshape.default(sum_536, [3072]);  sum_536 = None
    permute_1502: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1501, [1, 0]);  permute_1501 = None
    view_1556: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_424, [8, 576, 768]);  mm_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1203: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1556, primals_219);  primals_219 = None
    mul_1204: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1203, 768)
    sum_537: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1203, [2], True)
    mul_1205: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1203, mul_84);  mul_1203 = None
    sum_538: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1205, [2], True);  mul_1205 = None
    mul_1206: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_84, sum_538);  sum_538 = None
    sub_318: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1204, sum_537);  mul_1204 = sum_537 = None
    sub_319: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_318, mul_1206);  sub_318 = mul_1206 = None
    mul_1207: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_95, sub_319);  div_95 = sub_319 = None
    mul_1208: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1556, mul_84);  mul_84 = None
    sum_539: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1208, [0, 1]);  mul_1208 = None
    sum_540: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1556, [0, 1]);  view_1556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_519: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_516, mul_1207);  add_516 = mul_1207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1209: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_519, primals_18);  primals_18 = None
    mul_1210: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_519, view_176);  view_176 = None
    sum_541: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1210, [0, 1], True);  mul_1210 = None
    view_1557: "f32[768]" = torch.ops.aten.reshape.default(sum_541, [768]);  sum_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1558: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1209, [4608, 768]);  mul_1209 = None
    mm_426: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1558, permute_1503);  permute_1503 = None
    permute_1504: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1558, [1, 0])
    mm_427: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1504, view_175);  permute_1504 = view_175 = None
    permute_1505: "f32[768, 768]" = torch.ops.aten.permute.default(mm_427, [1, 0]);  mm_427 = None
    sum_542: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1558, [0], True);  view_1558 = None
    view_1559: "f32[768]" = torch.ops.aten.reshape.default(sum_542, [768]);  sum_542 = None
    permute_1506: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1505, [1, 0]);  permute_1505 = None
    view_1560: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_426, [8, 576, 768]);  mm_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1561: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1560, [8, 576, 16, 48]);  view_1560 = None
    permute_1507: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1561, [0, 2, 1, 3]);  view_1561 = None
    clone_729: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1507, memory_format = torch.contiguous_format);  permute_1507 = None
    view_1562: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_729, [128, 576, 48]);  clone_729 = None
    bmm_180: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1508, view_1562);  permute_1508 = None
    bmm_181: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1562, permute_1509);  view_1562 = permute_1509 = None
    view_1563: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_180, [8, 16, 576, 48]);  bmm_180 = None
    view_1564: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_181, [8, 16, 576, 576]);  bmm_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1510: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1564, [0, 2, 3, 1]);  view_1564 = None
    sum_543: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1510, [0, 1, 2], True)
    view_1565: "f32[16]" = torch.ops.aten.reshape.default(sum_543, [16]);  sum_543 = None
    clone_730: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1510, memory_format = torch.contiguous_format);  permute_1510 = None
    view_1566: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_730, [2654208, 16]);  clone_730 = None
    permute_1511: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1566, [1, 0])
    mm_428: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1511, view_169);  permute_1511 = view_169 = None
    permute_1512: "f32[16, 16]" = torch.ops.aten.permute.default(mm_428, [1, 0]);  mm_428 = None
    mm_429: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1566, permute_1513);  view_1566 = permute_1513 = None
    view_1567: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_429, [8, 576, 576, 16]);  mm_429 = None
    permute_1514: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1512, [1, 0]);  permute_1512 = None
    permute_1515: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1567, [0, 3, 1, 2]);  view_1567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1211: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1515, alias_67);  permute_1515 = None
    sum_544: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1211, [-1], True)
    mul_1212: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_67, sum_544);  alias_67 = sum_544 = None
    sub_320: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1211, mul_1212);  mul_1211 = mul_1212 = None
    clone_731: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_320, memory_format = torch.contiguous_format);  sub_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1516: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_731, [0, 2, 3, 1]);  clone_731 = None
    sum_545: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1516, [0, 1, 2], True)
    view_1568: "f32[16]" = torch.ops.aten.reshape.default(sum_545, [16]);  sum_545 = None
    clone_732: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1516, memory_format = torch.contiguous_format);  permute_1516 = None
    view_1569: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_732, [2654208, 16]);  clone_732 = None
    permute_1517: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1569, [1, 0])
    mm_430: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1517, view_167);  permute_1517 = view_167 = None
    permute_1518: "f32[16, 16]" = torch.ops.aten.permute.default(mm_430, [1, 0]);  mm_430 = None
    mm_431: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1569, permute_1519);  view_1569 = permute_1519 = None
    view_1570: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_431, [8, 576, 576, 16]);  mm_431 = None
    permute_1520: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1518, [1, 0]);  permute_1518 = None
    permute_1521: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1570, [0, 3, 1, 2]);  view_1570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_733: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1521, memory_format = torch.contiguous_format);  permute_1521 = None
    view_1571: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_733, [128, 576, 576]);  clone_733 = None
    bmm_182: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1522, view_1571);  permute_1522 = None
    bmm_183: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1571, permute_1523);  view_1571 = permute_1523 = None
    view_1572: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_182, [8, 16, 48, 576]);  bmm_182 = None
    view_1573: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_183, [8, 16, 576, 48]);  bmm_183 = None
    permute_1524: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1572, [0, 1, 3, 2]);  view_1572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_84: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1563, 0, 2);  view_1563 = None
    select_scatter_85: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1524, 0, 1);  permute_1524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_520: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_84, select_scatter_85);  select_scatter_84 = select_scatter_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1213: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1573, 0.14433756729740643);  view_1573 = None
    select_scatter_86: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1213, 0, 0);  mul_1213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_521: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_520, select_scatter_86);  add_520 = select_scatter_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1525: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_521, [1, 3, 0, 2, 4]);  add_521 = None
    clone_734: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1525, memory_format = torch.contiguous_format);  permute_1525 = None
    view_1574: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_734, [8, 576, 2304]);  clone_734 = None
    view_1575: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1574, [4608, 2304]);  view_1574 = None
    mm_432: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1575, permute_1526);  permute_1526 = None
    permute_1527: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1575, [1, 0])
    mm_433: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1527, view_161);  permute_1527 = view_161 = None
    permute_1528: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_433, [1, 0]);  mm_433 = None
    sum_546: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1575, [0], True);  view_1575 = None
    view_1576: "f32[2304]" = torch.ops.aten.reshape.default(sum_546, [2304]);  sum_546 = None
    permute_1529: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1528, [1, 0]);  permute_1528 = None
    view_1577: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_432, [8, 576, 768]);  mm_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1215: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1577, primals_209);  primals_209 = None
    mul_1216: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1215, 768)
    sum_547: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1215, [2], True)
    mul_1217: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1215, mul_80);  mul_1215 = None
    sum_548: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1217, [2], True);  mul_1217 = None
    mul_1218: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_548);  sum_548 = None
    sub_322: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1216, sum_547);  mul_1216 = sum_547 = None
    sub_323: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_322, mul_1218);  sub_322 = mul_1218 = None
    mul_1219: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_96, sub_323);  div_96 = sub_323 = None
    mul_1220: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1577, mul_80);  mul_80 = None
    sum_549: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1220, [0, 1]);  mul_1220 = None
    sum_550: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1577, [0, 1]);  view_1577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_522: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_519, mul_1219);  add_519 = mul_1219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1221: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_522, primals_17);  primals_17 = None
    mul_1222: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_522, view_160);  view_160 = None
    sum_551: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1222, [0, 1], True);  mul_1222 = None
    view_1578: "f32[768]" = torch.ops.aten.reshape.default(sum_551, [768]);  sum_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1579: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1221, [4608, 768]);  mul_1221 = None
    mm_434: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1579, permute_1530);  permute_1530 = None
    permute_1531: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1579, [1, 0])
    mm_435: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1531, view_159);  permute_1531 = view_159 = None
    permute_1532: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_435, [1, 0]);  mm_435 = None
    sum_552: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1579, [0], True);  view_1579 = None
    view_1580: "f32[768]" = torch.ops.aten.reshape.default(sum_552, [768]);  sum_552 = None
    permute_1533: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1532, [1, 0]);  permute_1532 = None
    view_1581: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_434, [8, 576, 3072]);  mm_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1224: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.5);  add_71 = None
    mul_1225: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_158, view_158)
    mul_1226: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1225, -0.5);  mul_1225 = None
    exp_66: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1226);  mul_1226 = None
    mul_1227: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_66, 0.3989422804014327);  exp_66 = None
    mul_1228: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_158, mul_1227);  view_158 = mul_1227 = None
    add_524: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1224, mul_1228);  mul_1224 = mul_1228 = None
    mul_1229: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1581, add_524);  view_1581 = add_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1582: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1229, [4608, 3072]);  mul_1229 = None
    mm_436: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1582, permute_1534);  permute_1534 = None
    permute_1535: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1582, [1, 0])
    mm_437: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1535, view_157);  permute_1535 = view_157 = None
    permute_1536: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_437, [1, 0]);  mm_437 = None
    sum_553: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1582, [0], True);  view_1582 = None
    view_1583: "f32[3072]" = torch.ops.aten.reshape.default(sum_553, [3072]);  sum_553 = None
    permute_1537: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1536, [1, 0]);  permute_1536 = None
    view_1584: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_436, [8, 576, 768]);  mm_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1231: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1584, primals_203);  primals_203 = None
    mul_1232: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1231, 768)
    sum_554: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1231, [2], True)
    mul_1233: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1231, mul_74);  mul_1231 = None
    sum_555: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1233, [2], True);  mul_1233 = None
    mul_1234: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_74, sum_555);  sum_555 = None
    sub_325: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1232, sum_554);  mul_1232 = sum_554 = None
    sub_326: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_325, mul_1234);  sub_325 = mul_1234 = None
    mul_1235: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_97, sub_326);  div_97 = sub_326 = None
    mul_1236: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1584, mul_74);  mul_74 = None
    sum_556: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1236, [0, 1]);  mul_1236 = None
    sum_557: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1584, [0, 1]);  view_1584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_525: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_522, mul_1235);  add_522 = mul_1235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1237: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_525, primals_16);  primals_16 = None
    mul_1238: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_525, view_156);  view_156 = None
    sum_558: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1238, [0, 1], True);  mul_1238 = None
    view_1585: "f32[768]" = torch.ops.aten.reshape.default(sum_558, [768]);  sum_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1586: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1237, [4608, 768]);  mul_1237 = None
    mm_438: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1586, permute_1538);  permute_1538 = None
    permute_1539: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1586, [1, 0])
    mm_439: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1539, view_155);  permute_1539 = view_155 = None
    permute_1540: "f32[768, 768]" = torch.ops.aten.permute.default(mm_439, [1, 0]);  mm_439 = None
    sum_559: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1586, [0], True);  view_1586 = None
    view_1587: "f32[768]" = torch.ops.aten.reshape.default(sum_559, [768]);  sum_559 = None
    permute_1541: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1540, [1, 0]);  permute_1540 = None
    view_1588: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_438, [8, 576, 768]);  mm_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1589: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1588, [8, 576, 16, 48]);  view_1588 = None
    permute_1542: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1589, [0, 2, 1, 3]);  view_1589 = None
    clone_737: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1542, memory_format = torch.contiguous_format);  permute_1542 = None
    view_1590: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_737, [128, 576, 48]);  clone_737 = None
    bmm_184: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1543, view_1590);  permute_1543 = None
    bmm_185: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1590, permute_1544);  view_1590 = permute_1544 = None
    view_1591: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_184, [8, 16, 576, 48]);  bmm_184 = None
    view_1592: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_185, [8, 16, 576, 576]);  bmm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1545: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1592, [0, 2, 3, 1]);  view_1592 = None
    sum_560: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1545, [0, 1, 2], True)
    view_1593: "f32[16]" = torch.ops.aten.reshape.default(sum_560, [16]);  sum_560 = None
    clone_738: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1545, memory_format = torch.contiguous_format);  permute_1545 = None
    view_1594: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_738, [2654208, 16]);  clone_738 = None
    permute_1546: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1594, [1, 0])
    mm_440: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1546, view_149);  permute_1546 = view_149 = None
    permute_1547: "f32[16, 16]" = torch.ops.aten.permute.default(mm_440, [1, 0]);  mm_440 = None
    mm_441: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1594, permute_1548);  view_1594 = permute_1548 = None
    view_1595: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_441, [8, 576, 576, 16]);  mm_441 = None
    permute_1549: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1547, [1, 0]);  permute_1547 = None
    permute_1550: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1595, [0, 3, 1, 2]);  view_1595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1239: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1550, alias_68);  permute_1550 = None
    sum_561: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1239, [-1], True)
    mul_1240: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_68, sum_561);  alias_68 = sum_561 = None
    sub_327: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1239, mul_1240);  mul_1239 = mul_1240 = None
    clone_739: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_327, memory_format = torch.contiguous_format);  sub_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1551: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_739, [0, 2, 3, 1]);  clone_739 = None
    sum_562: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1551, [0, 1, 2], True)
    view_1596: "f32[16]" = torch.ops.aten.reshape.default(sum_562, [16]);  sum_562 = None
    clone_740: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1551, memory_format = torch.contiguous_format);  permute_1551 = None
    view_1597: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_740, [2654208, 16]);  clone_740 = None
    permute_1552: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1597, [1, 0])
    mm_442: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1552, view_147);  permute_1552 = view_147 = None
    permute_1553: "f32[16, 16]" = torch.ops.aten.permute.default(mm_442, [1, 0]);  mm_442 = None
    mm_443: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1597, permute_1554);  view_1597 = permute_1554 = None
    view_1598: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_443, [8, 576, 576, 16]);  mm_443 = None
    permute_1555: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1553, [1, 0]);  permute_1553 = None
    permute_1556: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1598, [0, 3, 1, 2]);  view_1598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_741: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1556, memory_format = torch.contiguous_format);  permute_1556 = None
    view_1599: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_741, [128, 576, 576]);  clone_741 = None
    bmm_186: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1557, view_1599);  permute_1557 = None
    bmm_187: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1599, permute_1558);  view_1599 = permute_1558 = None
    view_1600: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_186, [8, 16, 48, 576]);  bmm_186 = None
    view_1601: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_187, [8, 16, 576, 48]);  bmm_187 = None
    permute_1559: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1600, [0, 1, 3, 2]);  view_1600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_87: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1591, 0, 2);  view_1591 = None
    select_scatter_88: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1559, 0, 1);  permute_1559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_526: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_87, select_scatter_88);  select_scatter_87 = select_scatter_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1241: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1601, 0.14433756729740643);  view_1601 = None
    select_scatter_89: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1241, 0, 0);  mul_1241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_527: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_526, select_scatter_89);  add_526 = select_scatter_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1560: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_527, [1, 3, 0, 2, 4]);  add_527 = None
    clone_742: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1560, memory_format = torch.contiguous_format);  permute_1560 = None
    view_1602: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_742, [8, 576, 2304]);  clone_742 = None
    view_1603: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1602, [4608, 2304]);  view_1602 = None
    mm_444: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1603, permute_1561);  permute_1561 = None
    permute_1562: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1603, [1, 0])
    mm_445: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1562, view_141);  permute_1562 = view_141 = None
    permute_1563: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_445, [1, 0]);  mm_445 = None
    sum_563: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1603, [0], True);  view_1603 = None
    view_1604: "f32[2304]" = torch.ops.aten.reshape.default(sum_563, [2304]);  sum_563 = None
    permute_1564: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1563, [1, 0]);  permute_1563 = None
    view_1605: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_444, [8, 576, 768]);  mm_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1243: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1605, primals_193);  primals_193 = None
    mul_1244: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1243, 768)
    sum_564: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1243, [2], True)
    mul_1245: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1243, mul_70);  mul_1243 = None
    sum_565: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1245, [2], True);  mul_1245 = None
    mul_1246: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_70, sum_565);  sum_565 = None
    sub_329: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1244, sum_564);  mul_1244 = sum_564 = None
    sub_330: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_329, mul_1246);  sub_329 = mul_1246 = None
    mul_1247: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_98, sub_330);  div_98 = sub_330 = None
    mul_1248: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1605, mul_70);  mul_70 = None
    sum_566: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1248, [0, 1]);  mul_1248 = None
    sum_567: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1605, [0, 1]);  view_1605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_528: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_525, mul_1247);  add_525 = mul_1247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1249: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_528, primals_15);  primals_15 = None
    mul_1250: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_528, view_140);  view_140 = None
    sum_568: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1250, [0, 1], True);  mul_1250 = None
    view_1606: "f32[768]" = torch.ops.aten.reshape.default(sum_568, [768]);  sum_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1607: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1249, [4608, 768]);  mul_1249 = None
    mm_446: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1607, permute_1565);  permute_1565 = None
    permute_1566: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1607, [1, 0])
    mm_447: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1566, view_139);  permute_1566 = view_139 = None
    permute_1567: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_447, [1, 0]);  mm_447 = None
    sum_569: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1607, [0], True);  view_1607 = None
    view_1608: "f32[768]" = torch.ops.aten.reshape.default(sum_569, [768]);  sum_569 = None
    permute_1568: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1567, [1, 0]);  permute_1567 = None
    view_1609: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_446, [8, 576, 3072]);  mm_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1252: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
    mul_1253: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_138, view_138)
    mul_1254: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1253, -0.5);  mul_1253 = None
    exp_67: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1254);  mul_1254 = None
    mul_1255: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_67, 0.3989422804014327);  exp_67 = None
    mul_1256: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_138, mul_1255);  view_138 = mul_1255 = None
    add_530: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1252, mul_1256);  mul_1252 = mul_1256 = None
    mul_1257: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1609, add_530);  view_1609 = add_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1610: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1257, [4608, 3072]);  mul_1257 = None
    mm_448: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1610, permute_1569);  permute_1569 = None
    permute_1570: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1610, [1, 0])
    mm_449: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1570, view_137);  permute_1570 = view_137 = None
    permute_1571: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_449, [1, 0]);  mm_449 = None
    sum_570: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1610, [0], True);  view_1610 = None
    view_1611: "f32[3072]" = torch.ops.aten.reshape.default(sum_570, [3072]);  sum_570 = None
    permute_1572: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1571, [1, 0]);  permute_1571 = None
    view_1612: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_448, [8, 576, 768]);  mm_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1259: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1612, primals_187);  primals_187 = None
    mul_1260: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1259, 768)
    sum_571: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1259, [2], True)
    mul_1261: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1259, mul_64);  mul_1259 = None
    sum_572: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1261, [2], True);  mul_1261 = None
    mul_1262: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_64, sum_572);  sum_572 = None
    sub_332: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1260, sum_571);  mul_1260 = sum_571 = None
    sub_333: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_332, mul_1262);  sub_332 = mul_1262 = None
    mul_1263: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_99, sub_333);  div_99 = sub_333 = None
    mul_1264: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1612, mul_64);  mul_64 = None
    sum_573: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1264, [0, 1]);  mul_1264 = None
    sum_574: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1612, [0, 1]);  view_1612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_531: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_528, mul_1263);  add_528 = mul_1263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1265: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_531, primals_14);  primals_14 = None
    mul_1266: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_531, view_136);  view_136 = None
    sum_575: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1266, [0, 1], True);  mul_1266 = None
    view_1613: "f32[768]" = torch.ops.aten.reshape.default(sum_575, [768]);  sum_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1614: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1265, [4608, 768]);  mul_1265 = None
    mm_450: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1614, permute_1573);  permute_1573 = None
    permute_1574: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1614, [1, 0])
    mm_451: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1574, view_135);  permute_1574 = view_135 = None
    permute_1575: "f32[768, 768]" = torch.ops.aten.permute.default(mm_451, [1, 0]);  mm_451 = None
    sum_576: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1614, [0], True);  view_1614 = None
    view_1615: "f32[768]" = torch.ops.aten.reshape.default(sum_576, [768]);  sum_576 = None
    permute_1576: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1575, [1, 0]);  permute_1575 = None
    view_1616: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_450, [8, 576, 768]);  mm_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1617: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1616, [8, 576, 16, 48]);  view_1616 = None
    permute_1577: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1617, [0, 2, 1, 3]);  view_1617 = None
    clone_745: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1577, memory_format = torch.contiguous_format);  permute_1577 = None
    view_1618: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_745, [128, 576, 48]);  clone_745 = None
    bmm_188: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1578, view_1618);  permute_1578 = None
    bmm_189: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1618, permute_1579);  view_1618 = permute_1579 = None
    view_1619: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_188, [8, 16, 576, 48]);  bmm_188 = None
    view_1620: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_189, [8, 16, 576, 576]);  bmm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1580: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1620, [0, 2, 3, 1]);  view_1620 = None
    sum_577: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1580, [0, 1, 2], True)
    view_1621: "f32[16]" = torch.ops.aten.reshape.default(sum_577, [16]);  sum_577 = None
    clone_746: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1580, memory_format = torch.contiguous_format);  permute_1580 = None
    view_1622: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_746, [2654208, 16]);  clone_746 = None
    permute_1581: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1622, [1, 0])
    mm_452: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1581, view_129);  permute_1581 = view_129 = None
    permute_1582: "f32[16, 16]" = torch.ops.aten.permute.default(mm_452, [1, 0]);  mm_452 = None
    mm_453: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1622, permute_1583);  view_1622 = permute_1583 = None
    view_1623: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_453, [8, 576, 576, 16]);  mm_453 = None
    permute_1584: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1582, [1, 0]);  permute_1582 = None
    permute_1585: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1623, [0, 3, 1, 2]);  view_1623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1267: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1585, alias_69);  permute_1585 = None
    sum_578: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1267, [-1], True)
    mul_1268: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_69, sum_578);  alias_69 = sum_578 = None
    sub_334: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1267, mul_1268);  mul_1267 = mul_1268 = None
    clone_747: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_334, memory_format = torch.contiguous_format);  sub_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1586: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_747, [0, 2, 3, 1]);  clone_747 = None
    sum_579: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1586, [0, 1, 2], True)
    view_1624: "f32[16]" = torch.ops.aten.reshape.default(sum_579, [16]);  sum_579 = None
    clone_748: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1586, memory_format = torch.contiguous_format);  permute_1586 = None
    view_1625: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_748, [2654208, 16]);  clone_748 = None
    permute_1587: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1625, [1, 0])
    mm_454: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1587, view_127);  permute_1587 = view_127 = None
    permute_1588: "f32[16, 16]" = torch.ops.aten.permute.default(mm_454, [1, 0]);  mm_454 = None
    mm_455: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1625, permute_1589);  view_1625 = permute_1589 = None
    view_1626: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_455, [8, 576, 576, 16]);  mm_455 = None
    permute_1590: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1588, [1, 0]);  permute_1588 = None
    permute_1591: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1626, [0, 3, 1, 2]);  view_1626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_749: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1591, memory_format = torch.contiguous_format);  permute_1591 = None
    view_1627: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_749, [128, 576, 576]);  clone_749 = None
    bmm_190: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1592, view_1627);  permute_1592 = None
    bmm_191: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1627, permute_1593);  view_1627 = permute_1593 = None
    view_1628: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_190, [8, 16, 48, 576]);  bmm_190 = None
    view_1629: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_191, [8, 16, 576, 48]);  bmm_191 = None
    permute_1594: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1628, [0, 1, 3, 2]);  view_1628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_90: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1619, 0, 2);  view_1619 = None
    select_scatter_91: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1594, 0, 1);  permute_1594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_532: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_90, select_scatter_91);  select_scatter_90 = select_scatter_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1269: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1629, 0.14433756729740643);  view_1629 = None
    select_scatter_92: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1269, 0, 0);  mul_1269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_533: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_532, select_scatter_92);  add_532 = select_scatter_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1595: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_533, [1, 3, 0, 2, 4]);  add_533 = None
    clone_750: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1595, memory_format = torch.contiguous_format);  permute_1595 = None
    view_1630: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_750, [8, 576, 2304]);  clone_750 = None
    view_1631: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1630, [4608, 2304]);  view_1630 = None
    mm_456: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1631, permute_1596);  permute_1596 = None
    permute_1597: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1631, [1, 0])
    mm_457: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1597, view_121);  permute_1597 = view_121 = None
    permute_1598: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_457, [1, 0]);  mm_457 = None
    sum_580: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1631, [0], True);  view_1631 = None
    view_1632: "f32[2304]" = torch.ops.aten.reshape.default(sum_580, [2304]);  sum_580 = None
    permute_1599: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1598, [1, 0]);  permute_1598 = None
    view_1633: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_456, [8, 576, 768]);  mm_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1271: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1633, primals_177);  primals_177 = None
    mul_1272: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1271, 768)
    sum_581: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1271, [2], True)
    mul_1273: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1271, mul_60);  mul_1271 = None
    sum_582: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1273, [2], True);  mul_1273 = None
    mul_1274: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_60, sum_582);  sum_582 = None
    sub_336: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1272, sum_581);  mul_1272 = sum_581 = None
    sub_337: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_336, mul_1274);  sub_336 = mul_1274 = None
    mul_1275: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_100, sub_337);  div_100 = sub_337 = None
    mul_1276: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1633, mul_60);  mul_60 = None
    sum_583: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1276, [0, 1]);  mul_1276 = None
    sum_584: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1633, [0, 1]);  view_1633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_534: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_531, mul_1275);  add_531 = mul_1275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1277: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_534, primals_13);  primals_13 = None
    mul_1278: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_534, view_120);  view_120 = None
    sum_585: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1278, [0, 1], True);  mul_1278 = None
    view_1634: "f32[768]" = torch.ops.aten.reshape.default(sum_585, [768]);  sum_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1635: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1277, [4608, 768]);  mul_1277 = None
    mm_458: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1635, permute_1600);  permute_1600 = None
    permute_1601: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1635, [1, 0])
    mm_459: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1601, view_119);  permute_1601 = view_119 = None
    permute_1602: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_459, [1, 0]);  mm_459 = None
    sum_586: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1635, [0], True);  view_1635 = None
    view_1636: "f32[768]" = torch.ops.aten.reshape.default(sum_586, [768]);  sum_586 = None
    permute_1603: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1602, [1, 0]);  permute_1602 = None
    view_1637: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_458, [8, 576, 3072]);  mm_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1280: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_53, 0.5);  add_53 = None
    mul_1281: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_118, view_118)
    mul_1282: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1281, -0.5);  mul_1281 = None
    exp_68: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1282);  mul_1282 = None
    mul_1283: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_68, 0.3989422804014327);  exp_68 = None
    mul_1284: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_118, mul_1283);  view_118 = mul_1283 = None
    add_536: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1280, mul_1284);  mul_1280 = mul_1284 = None
    mul_1285: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1637, add_536);  view_1637 = add_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1638: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1285, [4608, 3072]);  mul_1285 = None
    mm_460: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1638, permute_1604);  permute_1604 = None
    permute_1605: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1638, [1, 0])
    mm_461: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1605, view_117);  permute_1605 = view_117 = None
    permute_1606: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_461, [1, 0]);  mm_461 = None
    sum_587: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1638, [0], True);  view_1638 = None
    view_1639: "f32[3072]" = torch.ops.aten.reshape.default(sum_587, [3072]);  sum_587 = None
    permute_1607: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1606, [1, 0]);  permute_1606 = None
    view_1640: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_460, [8, 576, 768]);  mm_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1287: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1640, primals_171);  primals_171 = None
    mul_1288: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1287, 768)
    sum_588: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1287, [2], True)
    mul_1289: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1287, mul_54);  mul_1287 = None
    sum_589: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1289, [2], True);  mul_1289 = None
    mul_1290: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_54, sum_589);  sum_589 = None
    sub_339: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1288, sum_588);  mul_1288 = sum_588 = None
    sub_340: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_339, mul_1290);  sub_339 = mul_1290 = None
    mul_1291: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_101, sub_340);  div_101 = sub_340 = None
    mul_1292: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1640, mul_54);  mul_54 = None
    sum_590: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1292, [0, 1]);  mul_1292 = None
    sum_591: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1640, [0, 1]);  view_1640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_537: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_534, mul_1291);  add_534 = mul_1291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1293: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_537, primals_12);  primals_12 = None
    mul_1294: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_537, view_116);  view_116 = None
    sum_592: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1294, [0, 1], True);  mul_1294 = None
    view_1641: "f32[768]" = torch.ops.aten.reshape.default(sum_592, [768]);  sum_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1642: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1293, [4608, 768]);  mul_1293 = None
    mm_462: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1642, permute_1608);  permute_1608 = None
    permute_1609: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1642, [1, 0])
    mm_463: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1609, view_115);  permute_1609 = view_115 = None
    permute_1610: "f32[768, 768]" = torch.ops.aten.permute.default(mm_463, [1, 0]);  mm_463 = None
    sum_593: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1642, [0], True);  view_1642 = None
    view_1643: "f32[768]" = torch.ops.aten.reshape.default(sum_593, [768]);  sum_593 = None
    permute_1611: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1610, [1, 0]);  permute_1610 = None
    view_1644: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_462, [8, 576, 768]);  mm_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1645: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1644, [8, 576, 16, 48]);  view_1644 = None
    permute_1612: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1645, [0, 2, 1, 3]);  view_1645 = None
    clone_753: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1612, memory_format = torch.contiguous_format);  permute_1612 = None
    view_1646: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_753, [128, 576, 48]);  clone_753 = None
    bmm_192: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1613, view_1646);  permute_1613 = None
    bmm_193: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1646, permute_1614);  view_1646 = permute_1614 = None
    view_1647: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_192, [8, 16, 576, 48]);  bmm_192 = None
    view_1648: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_193, [8, 16, 576, 576]);  bmm_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1615: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1648, [0, 2, 3, 1]);  view_1648 = None
    sum_594: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1615, [0, 1, 2], True)
    view_1649: "f32[16]" = torch.ops.aten.reshape.default(sum_594, [16]);  sum_594 = None
    clone_754: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1615, memory_format = torch.contiguous_format);  permute_1615 = None
    view_1650: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_754, [2654208, 16]);  clone_754 = None
    permute_1616: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1650, [1, 0])
    mm_464: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1616, view_109);  permute_1616 = view_109 = None
    permute_1617: "f32[16, 16]" = torch.ops.aten.permute.default(mm_464, [1, 0]);  mm_464 = None
    mm_465: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1650, permute_1618);  view_1650 = permute_1618 = None
    view_1651: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_465, [8, 576, 576, 16]);  mm_465 = None
    permute_1619: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1617, [1, 0]);  permute_1617 = None
    permute_1620: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1651, [0, 3, 1, 2]);  view_1651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1295: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1620, alias_70);  permute_1620 = None
    sum_595: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1295, [-1], True)
    mul_1296: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_70, sum_595);  alias_70 = sum_595 = None
    sub_341: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1295, mul_1296);  mul_1295 = mul_1296 = None
    clone_755: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_341, memory_format = torch.contiguous_format);  sub_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1621: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_755, [0, 2, 3, 1]);  clone_755 = None
    sum_596: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1621, [0, 1, 2], True)
    view_1652: "f32[16]" = torch.ops.aten.reshape.default(sum_596, [16]);  sum_596 = None
    clone_756: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1621, memory_format = torch.contiguous_format);  permute_1621 = None
    view_1653: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_756, [2654208, 16]);  clone_756 = None
    permute_1622: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1653, [1, 0])
    mm_466: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1622, view_107);  permute_1622 = view_107 = None
    permute_1623: "f32[16, 16]" = torch.ops.aten.permute.default(mm_466, [1, 0]);  mm_466 = None
    mm_467: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1653, permute_1624);  view_1653 = permute_1624 = None
    view_1654: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_467, [8, 576, 576, 16]);  mm_467 = None
    permute_1625: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1623, [1, 0]);  permute_1623 = None
    permute_1626: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1654, [0, 3, 1, 2]);  view_1654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_757: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1626, memory_format = torch.contiguous_format);  permute_1626 = None
    view_1655: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_757, [128, 576, 576]);  clone_757 = None
    bmm_194: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1627, view_1655);  permute_1627 = None
    bmm_195: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1655, permute_1628);  view_1655 = permute_1628 = None
    view_1656: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_194, [8, 16, 48, 576]);  bmm_194 = None
    view_1657: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_195, [8, 16, 576, 48]);  bmm_195 = None
    permute_1629: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1656, [0, 1, 3, 2]);  view_1656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_93: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1647, 0, 2);  view_1647 = None
    select_scatter_94: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1629, 0, 1);  permute_1629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_538: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_93, select_scatter_94);  select_scatter_93 = select_scatter_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1297: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1657, 0.14433756729740643);  view_1657 = None
    select_scatter_95: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1297, 0, 0);  mul_1297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_539: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_538, select_scatter_95);  add_538 = select_scatter_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1630: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_539, [1, 3, 0, 2, 4]);  add_539 = None
    clone_758: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1630, memory_format = torch.contiguous_format);  permute_1630 = None
    view_1658: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_758, [8, 576, 2304]);  clone_758 = None
    view_1659: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1658, [4608, 2304]);  view_1658 = None
    mm_468: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1659, permute_1631);  permute_1631 = None
    permute_1632: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1659, [1, 0])
    mm_469: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1632, view_101);  permute_1632 = view_101 = None
    permute_1633: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_469, [1, 0]);  mm_469 = None
    sum_597: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1659, [0], True);  view_1659 = None
    view_1660: "f32[2304]" = torch.ops.aten.reshape.default(sum_597, [2304]);  sum_597 = None
    permute_1634: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1633, [1, 0]);  permute_1633 = None
    view_1661: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_468, [8, 576, 768]);  mm_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1299: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1661, primals_161);  primals_161 = None
    mul_1300: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1299, 768)
    sum_598: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1299, [2], True)
    mul_1301: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1299, mul_50);  mul_1299 = None
    sum_599: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1301, [2], True);  mul_1301 = None
    mul_1302: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_599);  sum_599 = None
    sub_343: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1300, sum_598);  mul_1300 = sum_598 = None
    sub_344: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_343, mul_1302);  sub_343 = mul_1302 = None
    mul_1303: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_102, sub_344);  div_102 = sub_344 = None
    mul_1304: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1661, mul_50);  mul_50 = None
    sum_600: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1304, [0, 1]);  mul_1304 = None
    sum_601: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1661, [0, 1]);  view_1661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_540: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_537, mul_1303);  add_537 = mul_1303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1305: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_540, primals_11);  primals_11 = None
    mul_1306: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_540, view_100);  view_100 = None
    sum_602: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1306, [0, 1], True);  mul_1306 = None
    view_1662: "f32[768]" = torch.ops.aten.reshape.default(sum_602, [768]);  sum_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1663: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1305, [4608, 768]);  mul_1305 = None
    mm_470: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1663, permute_1635);  permute_1635 = None
    permute_1636: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1663, [1, 0])
    mm_471: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1636, view_99);  permute_1636 = view_99 = None
    permute_1637: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_471, [1, 0]);  mm_471 = None
    sum_603: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1663, [0], True);  view_1663 = None
    view_1664: "f32[768]" = torch.ops.aten.reshape.default(sum_603, [768]);  sum_603 = None
    permute_1638: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1637, [1, 0]);  permute_1637 = None
    view_1665: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_470, [8, 576, 3072]);  mm_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1308: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_44, 0.5);  add_44 = None
    mul_1309: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_98, view_98)
    mul_1310: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1309, -0.5);  mul_1309 = None
    exp_69: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1310);  mul_1310 = None
    mul_1311: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_69, 0.3989422804014327);  exp_69 = None
    mul_1312: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_98, mul_1311);  view_98 = mul_1311 = None
    add_542: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1308, mul_1312);  mul_1308 = mul_1312 = None
    mul_1313: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1665, add_542);  view_1665 = add_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1666: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1313, [4608, 3072]);  mul_1313 = None
    mm_472: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1666, permute_1639);  permute_1639 = None
    permute_1640: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1666, [1, 0])
    mm_473: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1640, view_97);  permute_1640 = view_97 = None
    permute_1641: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_473, [1, 0]);  mm_473 = None
    sum_604: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1666, [0], True);  view_1666 = None
    view_1667: "f32[3072]" = torch.ops.aten.reshape.default(sum_604, [3072]);  sum_604 = None
    permute_1642: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1641, [1, 0]);  permute_1641 = None
    view_1668: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_472, [8, 576, 768]);  mm_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1315: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1668, primals_155);  primals_155 = None
    mul_1316: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1315, 768)
    sum_605: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1315, [2], True)
    mul_1317: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1315, mul_44);  mul_1315 = None
    sum_606: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1317, [2], True);  mul_1317 = None
    mul_1318: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_44, sum_606);  sum_606 = None
    sub_346: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1316, sum_605);  mul_1316 = sum_605 = None
    sub_347: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_346, mul_1318);  sub_346 = mul_1318 = None
    mul_1319: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_103, sub_347);  div_103 = sub_347 = None
    mul_1320: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1668, mul_44);  mul_44 = None
    sum_607: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1320, [0, 1]);  mul_1320 = None
    sum_608: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1668, [0, 1]);  view_1668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_543: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_540, mul_1319);  add_540 = mul_1319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1321: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_543, primals_10);  primals_10 = None
    mul_1322: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_543, view_96);  view_96 = None
    sum_609: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1322, [0, 1], True);  mul_1322 = None
    view_1669: "f32[768]" = torch.ops.aten.reshape.default(sum_609, [768]);  sum_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1670: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1321, [4608, 768]);  mul_1321 = None
    mm_474: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1670, permute_1643);  permute_1643 = None
    permute_1644: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1670, [1, 0])
    mm_475: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1644, view_95);  permute_1644 = view_95 = None
    permute_1645: "f32[768, 768]" = torch.ops.aten.permute.default(mm_475, [1, 0]);  mm_475 = None
    sum_610: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1670, [0], True);  view_1670 = None
    view_1671: "f32[768]" = torch.ops.aten.reshape.default(sum_610, [768]);  sum_610 = None
    permute_1646: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1645, [1, 0]);  permute_1645 = None
    view_1672: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_474, [8, 576, 768]);  mm_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1673: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1672, [8, 576, 16, 48]);  view_1672 = None
    permute_1647: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1673, [0, 2, 1, 3]);  view_1673 = None
    clone_761: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1647, memory_format = torch.contiguous_format);  permute_1647 = None
    view_1674: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_761, [128, 576, 48]);  clone_761 = None
    bmm_196: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1648, view_1674);  permute_1648 = None
    bmm_197: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1674, permute_1649);  view_1674 = permute_1649 = None
    view_1675: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_196, [8, 16, 576, 48]);  bmm_196 = None
    view_1676: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_197, [8, 16, 576, 576]);  bmm_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1650: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1676, [0, 2, 3, 1]);  view_1676 = None
    sum_611: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1650, [0, 1, 2], True)
    view_1677: "f32[16]" = torch.ops.aten.reshape.default(sum_611, [16]);  sum_611 = None
    clone_762: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1650, memory_format = torch.contiguous_format);  permute_1650 = None
    view_1678: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_762, [2654208, 16]);  clone_762 = None
    permute_1651: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1678, [1, 0])
    mm_476: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1651, view_89);  permute_1651 = view_89 = None
    permute_1652: "f32[16, 16]" = torch.ops.aten.permute.default(mm_476, [1, 0]);  mm_476 = None
    mm_477: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1678, permute_1653);  view_1678 = permute_1653 = None
    view_1679: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_477, [8, 576, 576, 16]);  mm_477 = None
    permute_1654: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1652, [1, 0]);  permute_1652 = None
    permute_1655: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1679, [0, 3, 1, 2]);  view_1679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1323: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1655, alias_71);  permute_1655 = None
    sum_612: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1323, [-1], True)
    mul_1324: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_71, sum_612);  alias_71 = sum_612 = None
    sub_348: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1323, mul_1324);  mul_1323 = mul_1324 = None
    clone_763: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_348, memory_format = torch.contiguous_format);  sub_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1656: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_763, [0, 2, 3, 1]);  clone_763 = None
    sum_613: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1656, [0, 1, 2], True)
    view_1680: "f32[16]" = torch.ops.aten.reshape.default(sum_613, [16]);  sum_613 = None
    clone_764: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1656, memory_format = torch.contiguous_format);  permute_1656 = None
    view_1681: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_764, [2654208, 16]);  clone_764 = None
    permute_1657: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1681, [1, 0])
    mm_478: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1657, view_87);  permute_1657 = view_87 = None
    permute_1658: "f32[16, 16]" = torch.ops.aten.permute.default(mm_478, [1, 0]);  mm_478 = None
    mm_479: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1681, permute_1659);  view_1681 = permute_1659 = None
    view_1682: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_479, [8, 576, 576, 16]);  mm_479 = None
    permute_1660: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1658, [1, 0]);  permute_1658 = None
    permute_1661: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1682, [0, 3, 1, 2]);  view_1682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_765: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1661, memory_format = torch.contiguous_format);  permute_1661 = None
    view_1683: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_765, [128, 576, 576]);  clone_765 = None
    bmm_198: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1662, view_1683);  permute_1662 = None
    bmm_199: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1683, permute_1663);  view_1683 = permute_1663 = None
    view_1684: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_198, [8, 16, 48, 576]);  bmm_198 = None
    view_1685: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_199, [8, 16, 576, 48]);  bmm_199 = None
    permute_1664: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1684, [0, 1, 3, 2]);  view_1684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_96: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1675, 0, 2);  view_1675 = None
    select_scatter_97: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1664, 0, 1);  permute_1664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_544: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_96, select_scatter_97);  select_scatter_96 = select_scatter_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1325: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1685, 0.14433756729740643);  view_1685 = None
    select_scatter_98: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1325, 0, 0);  mul_1325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_545: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_544, select_scatter_98);  add_544 = select_scatter_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1665: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_545, [1, 3, 0, 2, 4]);  add_545 = None
    clone_766: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1665, memory_format = torch.contiguous_format);  permute_1665 = None
    view_1686: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_766, [8, 576, 2304]);  clone_766 = None
    view_1687: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1686, [4608, 2304]);  view_1686 = None
    mm_480: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1687, permute_1666);  permute_1666 = None
    permute_1667: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1687, [1, 0])
    mm_481: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1667, view_81);  permute_1667 = view_81 = None
    permute_1668: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_481, [1, 0]);  mm_481 = None
    sum_614: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1687, [0], True);  view_1687 = None
    view_1688: "f32[2304]" = torch.ops.aten.reshape.default(sum_614, [2304]);  sum_614 = None
    permute_1669: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1668, [1, 0]);  permute_1668 = None
    view_1689: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_480, [8, 576, 768]);  mm_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1327: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1689, primals_145);  primals_145 = None
    mul_1328: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1327, 768)
    sum_615: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1327, [2], True)
    mul_1329: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1327, mul_40);  mul_1327 = None
    sum_616: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1329, [2], True);  mul_1329 = None
    mul_1330: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_40, sum_616);  sum_616 = None
    sub_350: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1328, sum_615);  mul_1328 = sum_615 = None
    sub_351: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_350, mul_1330);  sub_350 = mul_1330 = None
    mul_1331: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_104, sub_351);  div_104 = sub_351 = None
    mul_1332: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1689, mul_40);  mul_40 = None
    sum_617: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1332, [0, 1]);  mul_1332 = None
    sum_618: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1689, [0, 1]);  view_1689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_546: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_543, mul_1331);  add_543 = mul_1331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1333: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_546, primals_9);  primals_9 = None
    mul_1334: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_546, view_80);  view_80 = None
    sum_619: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1334, [0, 1], True);  mul_1334 = None
    view_1690: "f32[768]" = torch.ops.aten.reshape.default(sum_619, [768]);  sum_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1691: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1333, [4608, 768]);  mul_1333 = None
    mm_482: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1691, permute_1670);  permute_1670 = None
    permute_1671: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1691, [1, 0])
    mm_483: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1671, view_79);  permute_1671 = view_79 = None
    permute_1672: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_483, [1, 0]);  mm_483 = None
    sum_620: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1691, [0], True);  view_1691 = None
    view_1692: "f32[768]" = torch.ops.aten.reshape.default(sum_620, [768]);  sum_620 = None
    permute_1673: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1672, [1, 0]);  permute_1672 = None
    view_1693: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_482, [8, 576, 3072]);  mm_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1336: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_35, 0.5);  add_35 = None
    mul_1337: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_78, view_78)
    mul_1338: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1337, -0.5);  mul_1337 = None
    exp_70: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1338);  mul_1338 = None
    mul_1339: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_70, 0.3989422804014327);  exp_70 = None
    mul_1340: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_78, mul_1339);  view_78 = mul_1339 = None
    add_548: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1336, mul_1340);  mul_1336 = mul_1340 = None
    mul_1341: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1693, add_548);  view_1693 = add_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1694: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1341, [4608, 3072]);  mul_1341 = None
    mm_484: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1694, permute_1674);  permute_1674 = None
    permute_1675: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1694, [1, 0])
    mm_485: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1675, view_77);  permute_1675 = view_77 = None
    permute_1676: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_485, [1, 0]);  mm_485 = None
    sum_621: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1694, [0], True);  view_1694 = None
    view_1695: "f32[3072]" = torch.ops.aten.reshape.default(sum_621, [3072]);  sum_621 = None
    permute_1677: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1676, [1, 0]);  permute_1676 = None
    view_1696: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_484, [8, 576, 768]);  mm_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1343: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1696, primals_139);  primals_139 = None
    mul_1344: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1343, 768)
    sum_622: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1343, [2], True)
    mul_1345: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1343, mul_34);  mul_1343 = None
    sum_623: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1345, [2], True);  mul_1345 = None
    mul_1346: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_34, sum_623);  sum_623 = None
    sub_353: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1344, sum_622);  mul_1344 = sum_622 = None
    sub_354: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_353, mul_1346);  sub_353 = mul_1346 = None
    mul_1347: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_105, sub_354);  div_105 = sub_354 = None
    mul_1348: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1696, mul_34);  mul_34 = None
    sum_624: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1348, [0, 1]);  mul_1348 = None
    sum_625: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1696, [0, 1]);  view_1696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_549: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_546, mul_1347);  add_546 = mul_1347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1349: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_549, primals_8);  primals_8 = None
    mul_1350: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_549, view_76);  view_76 = None
    sum_626: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1350, [0, 1], True);  mul_1350 = None
    view_1697: "f32[768]" = torch.ops.aten.reshape.default(sum_626, [768]);  sum_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1698: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1349, [4608, 768]);  mul_1349 = None
    mm_486: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1698, permute_1678);  permute_1678 = None
    permute_1679: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1698, [1, 0])
    mm_487: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1679, view_75);  permute_1679 = view_75 = None
    permute_1680: "f32[768, 768]" = torch.ops.aten.permute.default(mm_487, [1, 0]);  mm_487 = None
    sum_627: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1698, [0], True);  view_1698 = None
    view_1699: "f32[768]" = torch.ops.aten.reshape.default(sum_627, [768]);  sum_627 = None
    permute_1681: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1680, [1, 0]);  permute_1680 = None
    view_1700: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_486, [8, 576, 768]);  mm_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1701: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1700, [8, 576, 16, 48]);  view_1700 = None
    permute_1682: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1701, [0, 2, 1, 3]);  view_1701 = None
    clone_769: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1682, memory_format = torch.contiguous_format);  permute_1682 = None
    view_1702: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_769, [128, 576, 48]);  clone_769 = None
    bmm_200: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1683, view_1702);  permute_1683 = None
    bmm_201: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1702, permute_1684);  view_1702 = permute_1684 = None
    view_1703: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_200, [8, 16, 576, 48]);  bmm_200 = None
    view_1704: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_201, [8, 16, 576, 576]);  bmm_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1685: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1704, [0, 2, 3, 1]);  view_1704 = None
    sum_628: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1685, [0, 1, 2], True)
    view_1705: "f32[16]" = torch.ops.aten.reshape.default(sum_628, [16]);  sum_628 = None
    clone_770: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1685, memory_format = torch.contiguous_format);  permute_1685 = None
    view_1706: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_770, [2654208, 16]);  clone_770 = None
    permute_1686: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1706, [1, 0])
    mm_488: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1686, view_69);  permute_1686 = view_69 = None
    permute_1687: "f32[16, 16]" = torch.ops.aten.permute.default(mm_488, [1, 0]);  mm_488 = None
    mm_489: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1706, permute_1688);  view_1706 = permute_1688 = None
    view_1707: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_489, [8, 576, 576, 16]);  mm_489 = None
    permute_1689: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1687, [1, 0]);  permute_1687 = None
    permute_1690: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1707, [0, 3, 1, 2]);  view_1707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1351: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1690, alias_72);  permute_1690 = None
    sum_629: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1351, [-1], True)
    mul_1352: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_72, sum_629);  alias_72 = sum_629 = None
    sub_355: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1351, mul_1352);  mul_1351 = mul_1352 = None
    clone_771: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_355, memory_format = torch.contiguous_format);  sub_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1691: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_771, [0, 2, 3, 1]);  clone_771 = None
    sum_630: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1691, [0, 1, 2], True)
    view_1708: "f32[16]" = torch.ops.aten.reshape.default(sum_630, [16]);  sum_630 = None
    clone_772: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1691, memory_format = torch.contiguous_format);  permute_1691 = None
    view_1709: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_772, [2654208, 16]);  clone_772 = None
    permute_1692: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1709, [1, 0])
    mm_490: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1692, view_67);  permute_1692 = view_67 = None
    permute_1693: "f32[16, 16]" = torch.ops.aten.permute.default(mm_490, [1, 0]);  mm_490 = None
    mm_491: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1709, permute_1694);  view_1709 = permute_1694 = None
    view_1710: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_491, [8, 576, 576, 16]);  mm_491 = None
    permute_1695: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1693, [1, 0]);  permute_1693 = None
    permute_1696: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1710, [0, 3, 1, 2]);  view_1710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_773: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1696, memory_format = torch.contiguous_format);  permute_1696 = None
    view_1711: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_773, [128, 576, 576]);  clone_773 = None
    bmm_202: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1697, view_1711);  permute_1697 = None
    bmm_203: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1711, permute_1698);  view_1711 = permute_1698 = None
    view_1712: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_202, [8, 16, 48, 576]);  bmm_202 = None
    view_1713: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_203, [8, 16, 576, 48]);  bmm_203 = None
    permute_1699: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1712, [0, 1, 3, 2]);  view_1712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_99: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1703, 0, 2);  view_1703 = None
    select_scatter_100: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1699, 0, 1);  permute_1699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_550: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_99, select_scatter_100);  select_scatter_99 = select_scatter_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1353: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1713, 0.14433756729740643);  view_1713 = None
    select_scatter_101: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1353, 0, 0);  mul_1353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_551: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_550, select_scatter_101);  add_550 = select_scatter_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1700: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_551, [1, 3, 0, 2, 4]);  add_551 = None
    clone_774: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1700, memory_format = torch.contiguous_format);  permute_1700 = None
    view_1714: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_774, [8, 576, 2304]);  clone_774 = None
    view_1715: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1714, [4608, 2304]);  view_1714 = None
    mm_492: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1715, permute_1701);  permute_1701 = None
    permute_1702: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1715, [1, 0])
    mm_493: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1702, view_61);  permute_1702 = view_61 = None
    permute_1703: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_493, [1, 0]);  mm_493 = None
    sum_631: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1715, [0], True);  view_1715 = None
    view_1716: "f32[2304]" = torch.ops.aten.reshape.default(sum_631, [2304]);  sum_631 = None
    permute_1704: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1703, [1, 0]);  permute_1703 = None
    view_1717: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_492, [8, 576, 768]);  mm_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1355: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1717, primals_129);  primals_129 = None
    mul_1356: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1355, 768)
    sum_632: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1355, [2], True)
    mul_1357: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1355, mul_30);  mul_1355 = None
    sum_633: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1357, [2], True);  mul_1357 = None
    mul_1358: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_30, sum_633);  sum_633 = None
    sub_357: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1356, sum_632);  mul_1356 = sum_632 = None
    sub_358: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_357, mul_1358);  sub_357 = mul_1358 = None
    mul_1359: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_106, sub_358);  div_106 = sub_358 = None
    mul_1360: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1717, mul_30);  mul_30 = None
    sum_634: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1360, [0, 1]);  mul_1360 = None
    sum_635: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1717, [0, 1]);  view_1717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_552: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_549, mul_1359);  add_549 = mul_1359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1361: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_552, primals_7);  primals_7 = None
    mul_1362: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_552, view_60);  view_60 = None
    sum_636: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1362, [0, 1], True);  mul_1362 = None
    view_1718: "f32[768]" = torch.ops.aten.reshape.default(sum_636, [768]);  sum_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1719: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1361, [4608, 768]);  mul_1361 = None
    mm_494: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1719, permute_1705);  permute_1705 = None
    permute_1706: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1719, [1, 0])
    mm_495: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1706, view_59);  permute_1706 = view_59 = None
    permute_1707: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_495, [1, 0]);  mm_495 = None
    sum_637: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1719, [0], True);  view_1719 = None
    view_1720: "f32[768]" = torch.ops.aten.reshape.default(sum_637, [768]);  sum_637 = None
    permute_1708: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1707, [1, 0]);  permute_1707 = None
    view_1721: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_494, [8, 576, 3072]);  mm_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1364: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_26, 0.5);  add_26 = None
    mul_1365: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_58, view_58)
    mul_1366: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1365, -0.5);  mul_1365 = None
    exp_71: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1366);  mul_1366 = None
    mul_1367: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_71, 0.3989422804014327);  exp_71 = None
    mul_1368: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_58, mul_1367);  view_58 = mul_1367 = None
    add_554: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1364, mul_1368);  mul_1364 = mul_1368 = None
    mul_1369: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1721, add_554);  view_1721 = add_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1722: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1369, [4608, 3072]);  mul_1369 = None
    mm_496: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1722, permute_1709);  permute_1709 = None
    permute_1710: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1722, [1, 0])
    mm_497: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1710, view_57);  permute_1710 = view_57 = None
    permute_1711: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_497, [1, 0]);  mm_497 = None
    sum_638: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1722, [0], True);  view_1722 = None
    view_1723: "f32[3072]" = torch.ops.aten.reshape.default(sum_638, [3072]);  sum_638 = None
    permute_1712: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1711, [1, 0]);  permute_1711 = None
    view_1724: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_496, [8, 576, 768]);  mm_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1371: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1724, primals_123);  primals_123 = None
    mul_1372: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1371, 768)
    sum_639: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1371, [2], True)
    mul_1373: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1371, mul_24);  mul_1371 = None
    sum_640: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1373, [2], True);  mul_1373 = None
    mul_1374: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_24, sum_640);  sum_640 = None
    sub_360: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1372, sum_639);  mul_1372 = sum_639 = None
    sub_361: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_360, mul_1374);  sub_360 = mul_1374 = None
    mul_1375: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_107, sub_361);  div_107 = sub_361 = None
    mul_1376: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1724, mul_24);  mul_24 = None
    sum_641: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1376, [0, 1]);  mul_1376 = None
    sum_642: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1724, [0, 1]);  view_1724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_555: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_552, mul_1375);  add_552 = mul_1375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1377: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_555, primals_6);  primals_6 = None
    mul_1378: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_555, view_56);  view_56 = None
    sum_643: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1378, [0, 1], True);  mul_1378 = None
    view_1725: "f32[768]" = torch.ops.aten.reshape.default(sum_643, [768]);  sum_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1726: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1377, [4608, 768]);  mul_1377 = None
    mm_498: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1726, permute_1713);  permute_1713 = None
    permute_1714: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1726, [1, 0])
    mm_499: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1714, view_55);  permute_1714 = view_55 = None
    permute_1715: "f32[768, 768]" = torch.ops.aten.permute.default(mm_499, [1, 0]);  mm_499 = None
    sum_644: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1726, [0], True);  view_1726 = None
    view_1727: "f32[768]" = torch.ops.aten.reshape.default(sum_644, [768]);  sum_644 = None
    permute_1716: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1715, [1, 0]);  permute_1715 = None
    view_1728: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_498, [8, 576, 768]);  mm_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1729: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1728, [8, 576, 16, 48]);  view_1728 = None
    permute_1717: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1729, [0, 2, 1, 3]);  view_1729 = None
    clone_777: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1717, memory_format = torch.contiguous_format);  permute_1717 = None
    view_1730: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_777, [128, 576, 48]);  clone_777 = None
    bmm_204: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1718, view_1730);  permute_1718 = None
    bmm_205: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1730, permute_1719);  view_1730 = permute_1719 = None
    view_1731: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_204, [8, 16, 576, 48]);  bmm_204 = None
    view_1732: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_205, [8, 16, 576, 576]);  bmm_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1720: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1732, [0, 2, 3, 1]);  view_1732 = None
    sum_645: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1720, [0, 1, 2], True)
    view_1733: "f32[16]" = torch.ops.aten.reshape.default(sum_645, [16]);  sum_645 = None
    clone_778: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1720, memory_format = torch.contiguous_format);  permute_1720 = None
    view_1734: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_778, [2654208, 16]);  clone_778 = None
    permute_1721: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1734, [1, 0])
    mm_500: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1721, view_49);  permute_1721 = view_49 = None
    permute_1722: "f32[16, 16]" = torch.ops.aten.permute.default(mm_500, [1, 0]);  mm_500 = None
    mm_501: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1734, permute_1723);  view_1734 = permute_1723 = None
    view_1735: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_501, [8, 576, 576, 16]);  mm_501 = None
    permute_1724: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1722, [1, 0]);  permute_1722 = None
    permute_1725: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1735, [0, 3, 1, 2]);  view_1735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1379: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1725, alias_73);  permute_1725 = None
    sum_646: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1379, [-1], True)
    mul_1380: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_73, sum_646);  alias_73 = sum_646 = None
    sub_362: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1379, mul_1380);  mul_1379 = mul_1380 = None
    clone_779: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_362, memory_format = torch.contiguous_format);  sub_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1726: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_779, [0, 2, 3, 1]);  clone_779 = None
    sum_647: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1726, [0, 1, 2], True)
    view_1736: "f32[16]" = torch.ops.aten.reshape.default(sum_647, [16]);  sum_647 = None
    clone_780: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1726, memory_format = torch.contiguous_format);  permute_1726 = None
    view_1737: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_780, [2654208, 16]);  clone_780 = None
    permute_1727: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1737, [1, 0])
    mm_502: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1727, view_47);  permute_1727 = view_47 = None
    permute_1728: "f32[16, 16]" = torch.ops.aten.permute.default(mm_502, [1, 0]);  mm_502 = None
    mm_503: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1737, permute_1729);  view_1737 = permute_1729 = None
    view_1738: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_503, [8, 576, 576, 16]);  mm_503 = None
    permute_1730: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1728, [1, 0]);  permute_1728 = None
    permute_1731: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1738, [0, 3, 1, 2]);  view_1738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_781: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1731, memory_format = torch.contiguous_format);  permute_1731 = None
    view_1739: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_781, [128, 576, 576]);  clone_781 = None
    bmm_206: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1732, view_1739);  permute_1732 = None
    bmm_207: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1739, permute_1733);  view_1739 = permute_1733 = None
    view_1740: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_206, [8, 16, 48, 576]);  bmm_206 = None
    view_1741: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_207, [8, 16, 576, 48]);  bmm_207 = None
    permute_1734: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1740, [0, 1, 3, 2]);  view_1740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_102: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1731, 0, 2);  view_1731 = None
    select_scatter_103: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1734, 0, 1);  permute_1734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_556: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_102, select_scatter_103);  select_scatter_102 = select_scatter_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1381: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1741, 0.14433756729740643);  view_1741 = None
    select_scatter_104: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1381, 0, 0);  mul_1381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_557: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_556, select_scatter_104);  add_556 = select_scatter_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1735: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_557, [1, 3, 0, 2, 4]);  add_557 = None
    clone_782: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1735, memory_format = torch.contiguous_format);  permute_1735 = None
    view_1742: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_782, [8, 576, 2304]);  clone_782 = None
    view_1743: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1742, [4608, 2304]);  view_1742 = None
    mm_504: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1743, permute_1736);  permute_1736 = None
    permute_1737: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1743, [1, 0])
    mm_505: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1737, view_41);  permute_1737 = view_41 = None
    permute_1738: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_505, [1, 0]);  mm_505 = None
    sum_648: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1743, [0], True);  view_1743 = None
    view_1744: "f32[2304]" = torch.ops.aten.reshape.default(sum_648, [2304]);  sum_648 = None
    permute_1739: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1738, [1, 0]);  permute_1738 = None
    view_1745: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_504, [8, 576, 768]);  mm_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1383: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1745, primals_113);  primals_113 = None
    mul_1384: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1383, 768)
    sum_649: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1383, [2], True)
    mul_1385: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1383, mul_20);  mul_1383 = None
    sum_650: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1385, [2], True);  mul_1385 = None
    mul_1386: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_20, sum_650);  sum_650 = None
    sub_364: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1384, sum_649);  mul_1384 = sum_649 = None
    sub_365: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_364, mul_1386);  sub_364 = mul_1386 = None
    mul_1387: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_108, sub_365);  div_108 = sub_365 = None
    mul_1388: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1745, mul_20);  mul_20 = None
    sum_651: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1388, [0, 1]);  mul_1388 = None
    sum_652: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1745, [0, 1]);  view_1745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_558: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_555, mul_1387);  add_555 = mul_1387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1389: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_558, primals_5);  primals_5 = None
    mul_1390: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_558, view_40);  view_40 = None
    sum_653: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1390, [0, 1], True);  mul_1390 = None
    view_1746: "f32[768]" = torch.ops.aten.reshape.default(sum_653, [768]);  sum_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1747: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1389, [4608, 768]);  mul_1389 = None
    mm_506: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1747, permute_1740);  permute_1740 = None
    permute_1741: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1747, [1, 0])
    mm_507: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1741, view_39);  permute_1741 = view_39 = None
    permute_1742: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_507, [1, 0]);  mm_507 = None
    sum_654: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1747, [0], True);  view_1747 = None
    view_1748: "f32[768]" = torch.ops.aten.reshape.default(sum_654, [768]);  sum_654 = None
    permute_1743: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1742, [1, 0]);  permute_1742 = None
    view_1749: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_506, [8, 576, 3072]);  mm_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1392: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_17, 0.5);  add_17 = None
    mul_1393: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_38, view_38)
    mul_1394: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1393, -0.5);  mul_1393 = None
    exp_72: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1394);  mul_1394 = None
    mul_1395: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_72, 0.3989422804014327);  exp_72 = None
    mul_1396: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_38, mul_1395);  view_38 = mul_1395 = None
    add_560: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1392, mul_1396);  mul_1392 = mul_1396 = None
    mul_1397: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1749, add_560);  view_1749 = add_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1750: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1397, [4608, 3072]);  mul_1397 = None
    mm_508: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1750, permute_1744);  permute_1744 = None
    permute_1745: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1750, [1, 0])
    mm_509: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1745, view_37);  permute_1745 = view_37 = None
    permute_1746: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_509, [1, 0]);  mm_509 = None
    sum_655: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1750, [0], True);  view_1750 = None
    view_1751: "f32[3072]" = torch.ops.aten.reshape.default(sum_655, [3072]);  sum_655 = None
    permute_1747: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1746, [1, 0]);  permute_1746 = None
    view_1752: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_508, [8, 576, 768]);  mm_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1399: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1752, primals_107);  primals_107 = None
    mul_1400: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1399, 768)
    sum_656: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1399, [2], True)
    mul_1401: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1399, mul_14);  mul_1399 = None
    sum_657: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1401, [2], True);  mul_1401 = None
    mul_1402: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_14, sum_657);  sum_657 = None
    sub_367: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1400, sum_656);  mul_1400 = sum_656 = None
    sub_368: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_367, mul_1402);  sub_367 = mul_1402 = None
    mul_1403: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_109, sub_368);  div_109 = sub_368 = None
    mul_1404: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1752, mul_14);  mul_14 = None
    sum_658: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1404, [0, 1]);  mul_1404 = None
    sum_659: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1752, [0, 1]);  view_1752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_561: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_558, mul_1403);  add_558 = mul_1403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1405: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_561, primals_4);  primals_4 = None
    mul_1406: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_561, view_36);  view_36 = None
    sum_660: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1406, [0, 1], True);  mul_1406 = None
    view_1753: "f32[768]" = torch.ops.aten.reshape.default(sum_660, [768]);  sum_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1754: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1405, [4608, 768]);  mul_1405 = None
    mm_510: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1754, permute_1748);  permute_1748 = None
    permute_1749: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1754, [1, 0])
    mm_511: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1749, view_35);  permute_1749 = view_35 = None
    permute_1750: "f32[768, 768]" = torch.ops.aten.permute.default(mm_511, [1, 0]);  mm_511 = None
    sum_661: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1754, [0], True);  view_1754 = None
    view_1755: "f32[768]" = torch.ops.aten.reshape.default(sum_661, [768]);  sum_661 = None
    permute_1751: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1750, [1, 0]);  permute_1750 = None
    view_1756: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_510, [8, 576, 768]);  mm_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1757: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1756, [8, 576, 16, 48]);  view_1756 = None
    permute_1752: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1757, [0, 2, 1, 3]);  view_1757 = None
    clone_785: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1752, memory_format = torch.contiguous_format);  permute_1752 = None
    view_1758: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_785, [128, 576, 48]);  clone_785 = None
    bmm_208: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1753, view_1758);  permute_1753 = None
    bmm_209: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1758, permute_1754);  view_1758 = permute_1754 = None
    view_1759: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_208, [8, 16, 576, 48]);  bmm_208 = None
    view_1760: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_209, [8, 16, 576, 576]);  bmm_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1755: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1760, [0, 2, 3, 1]);  view_1760 = None
    sum_662: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1755, [0, 1, 2], True)
    view_1761: "f32[16]" = torch.ops.aten.reshape.default(sum_662, [16]);  sum_662 = None
    clone_786: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1755, memory_format = torch.contiguous_format);  permute_1755 = None
    view_1762: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_786, [2654208, 16]);  clone_786 = None
    permute_1756: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1762, [1, 0])
    mm_512: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1756, view_29);  permute_1756 = view_29 = None
    permute_1757: "f32[16, 16]" = torch.ops.aten.permute.default(mm_512, [1, 0]);  mm_512 = None
    mm_513: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1762, permute_1758);  view_1762 = permute_1758 = None
    view_1763: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_513, [8, 576, 576, 16]);  mm_513 = None
    permute_1759: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1757, [1, 0]);  permute_1757 = None
    permute_1760: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1763, [0, 3, 1, 2]);  view_1763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1407: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1760, alias_74);  permute_1760 = None
    sum_663: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1407, [-1], True)
    mul_1408: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_74, sum_663);  alias_74 = sum_663 = None
    sub_369: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1407, mul_1408);  mul_1407 = mul_1408 = None
    clone_787: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_369, memory_format = torch.contiguous_format);  sub_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1761: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_787, [0, 2, 3, 1]);  clone_787 = None
    sum_664: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1761, [0, 1, 2], True)
    view_1764: "f32[16]" = torch.ops.aten.reshape.default(sum_664, [16]);  sum_664 = None
    clone_788: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1761, memory_format = torch.contiguous_format);  permute_1761 = None
    view_1765: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_788, [2654208, 16]);  clone_788 = None
    permute_1762: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1765, [1, 0])
    mm_514: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1762, view_27);  permute_1762 = view_27 = None
    permute_1763: "f32[16, 16]" = torch.ops.aten.permute.default(mm_514, [1, 0]);  mm_514 = None
    mm_515: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1765, permute_1764);  view_1765 = permute_1764 = None
    view_1766: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_515, [8, 576, 576, 16]);  mm_515 = None
    permute_1765: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1763, [1, 0]);  permute_1763 = None
    permute_1766: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1766, [0, 3, 1, 2]);  view_1766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_789: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1766, memory_format = torch.contiguous_format);  permute_1766 = None
    view_1767: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_789, [128, 576, 576]);  clone_789 = None
    bmm_210: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1767, view_1767);  permute_1767 = None
    bmm_211: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1767, permute_1768);  view_1767 = permute_1768 = None
    view_1768: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_210, [8, 16, 48, 576]);  bmm_210 = None
    view_1769: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_211, [8, 16, 576, 48]);  bmm_211 = None
    permute_1769: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1768, [0, 1, 3, 2]);  view_1768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_105: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1759, 0, 2);  view_1759 = None
    select_scatter_106: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1769, 0, 1);  permute_1769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_562: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_105, select_scatter_106);  select_scatter_105 = select_scatter_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1409: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1769, 0.14433756729740643);  view_1769 = None
    select_scatter_107: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1409, 0, 0);  mul_1409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_563: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_562, select_scatter_107);  add_562 = select_scatter_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1770: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_563, [1, 3, 0, 2, 4]);  add_563 = None
    clone_790: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1770, memory_format = torch.contiguous_format);  permute_1770 = None
    view_1770: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_790, [8, 576, 2304]);  clone_790 = None
    view_1771: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1770, [4608, 2304]);  view_1770 = None
    mm_516: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1771, permute_1771);  permute_1771 = None
    permute_1772: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1771, [1, 0])
    mm_517: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1772, view_21);  permute_1772 = view_21 = None
    permute_1773: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_517, [1, 0]);  mm_517 = None
    sum_665: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1771, [0], True);  view_1771 = None
    view_1772: "f32[2304]" = torch.ops.aten.reshape.default(sum_665, [2304]);  sum_665 = None
    permute_1774: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1773, [1, 0]);  permute_1773 = None
    view_1773: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_516, [8, 576, 768]);  mm_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1411: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1773, primals_97);  primals_97 = None
    mul_1412: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1411, 768)
    sum_666: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1411, [2], True)
    mul_1413: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1411, mul_10);  mul_1411 = None
    sum_667: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1413, [2], True);  mul_1413 = None
    mul_1414: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_667);  sum_667 = None
    sub_371: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1412, sum_666);  mul_1412 = sum_666 = None
    sub_372: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_371, mul_1414);  sub_371 = mul_1414 = None
    mul_1415: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_110, sub_372);  div_110 = sub_372 = None
    mul_1416: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1773, mul_10);  mul_10 = None
    sum_668: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1416, [0, 1]);  mul_1416 = None
    sum_669: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1773, [0, 1]);  view_1773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_564: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_561, mul_1415);  add_561 = mul_1415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1417: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_564, primals_3);  primals_3 = None
    mul_1418: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_564, view_20);  view_20 = None
    sum_670: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1418, [0, 1], True);  mul_1418 = None
    view_1774: "f32[768]" = torch.ops.aten.reshape.default(sum_670, [768]);  sum_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1775: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1417, [4608, 768]);  mul_1417 = None
    mm_518: "f32[4608, 3072]" = torch.ops.aten.mm.default(view_1775, permute_1775);  permute_1775 = None
    permute_1776: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1775, [1, 0])
    mm_519: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1776, view_19);  permute_1776 = view_19 = None
    permute_1777: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_519, [1, 0]);  mm_519 = None
    sum_671: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1775, [0], True);  view_1775 = None
    view_1776: "f32[768]" = torch.ops.aten.reshape.default(sum_671, [768]);  sum_671 = None
    permute_1778: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1777, [1, 0]);  permute_1777 = None
    view_1777: "f32[8, 576, 3072]" = torch.ops.aten.reshape.default(mm_518, [8, 576, 3072]);  mm_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1420: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
    mul_1421: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_18, view_18)
    mul_1422: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_1421, -0.5);  mul_1421 = None
    exp_73: "f32[8, 576, 3072]" = torch.ops.aten.exp.default(mul_1422);  mul_1422 = None
    mul_1423: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(exp_73, 0.3989422804014327);  exp_73 = None
    mul_1424: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_18, mul_1423);  view_18 = mul_1423 = None
    add_566: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(mul_1420, mul_1424);  mul_1420 = mul_1424 = None
    mul_1425: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_1777, add_566);  view_1777 = add_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1778: "f32[4608, 3072]" = torch.ops.aten.reshape.default(mul_1425, [4608, 3072]);  mul_1425 = None
    mm_520: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1778, permute_1779);  permute_1779 = None
    permute_1780: "f32[3072, 4608]" = torch.ops.aten.permute.default(view_1778, [1, 0])
    mm_521: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1780, view_17);  permute_1780 = view_17 = None
    permute_1781: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_521, [1, 0]);  mm_521 = None
    sum_672: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1778, [0], True);  view_1778 = None
    view_1779: "f32[3072]" = torch.ops.aten.reshape.default(sum_672, [3072]);  sum_672 = None
    permute_1782: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1781, [1, 0]);  permute_1781 = None
    view_1780: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_520, [8, 576, 768]);  mm_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1427: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1780, primals_91);  primals_91 = None
    mul_1428: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1427, 768)
    sum_673: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1427, [2], True)
    mul_1429: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1427, mul_4);  mul_1427 = None
    sum_674: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1429, [2], True);  mul_1429 = None
    mul_1430: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_674);  sum_674 = None
    sub_374: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1428, sum_673);  mul_1428 = sum_673 = None
    sub_375: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_374, mul_1430);  sub_374 = mul_1430 = None
    mul_1431: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_111, sub_375);  div_111 = sub_375 = None
    mul_1432: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1780, mul_4);  mul_4 = None
    sum_675: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1432, [0, 1]);  mul_1432 = None
    sum_676: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1780, [0, 1]);  view_1780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    add_567: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_564, mul_1431);  add_564 = mul_1431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1433: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_567, primals_2);  primals_2 = None
    mul_1434: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(add_567, view_16);  view_16 = None
    sum_677: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1434, [0, 1], True);  mul_1434 = None
    view_1781: "f32[768]" = torch.ops.aten.reshape.default(sum_677, [768]);  sum_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_1782: "f32[4608, 768]" = torch.ops.aten.reshape.default(mul_1433, [4608, 768]);  mul_1433 = None
    mm_522: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1782, permute_1783);  permute_1783 = None
    permute_1784: "f32[768, 4608]" = torch.ops.aten.permute.default(view_1782, [1, 0])
    mm_523: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1784, view_15);  permute_1784 = view_15 = None
    permute_1785: "f32[768, 768]" = torch.ops.aten.permute.default(mm_523, [1, 0]);  mm_523 = None
    sum_678: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1782, [0], True);  view_1782 = None
    view_1783: "f32[768]" = torch.ops.aten.reshape.default(sum_678, [768]);  sum_678 = None
    permute_1786: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1785, [1, 0]);  permute_1785 = None
    view_1784: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_522, [8, 576, 768]);  mm_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_1785: "f32[8, 576, 16, 48]" = torch.ops.aten.reshape.default(view_1784, [8, 576, 16, 48]);  view_1784 = None
    permute_1787: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1785, [0, 2, 1, 3]);  view_1785 = None
    clone_793: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(permute_1787, memory_format = torch.contiguous_format);  permute_1787 = None
    view_1786: "f32[128, 576, 48]" = torch.ops.aten.reshape.default(clone_793, [128, 576, 48]);  clone_793 = None
    bmm_212: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(permute_1788, view_1786);  permute_1788 = None
    bmm_213: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_1786, permute_1789);  view_1786 = permute_1789 = None
    view_1787: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_212, [8, 16, 576, 48]);  bmm_212 = None
    view_1788: "f32[8, 16, 576, 576]" = torch.ops.aten.reshape.default(bmm_213, [8, 16, 576, 576]);  bmm_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1790: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_1788, [0, 2, 3, 1]);  view_1788 = None
    sum_679: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1790, [0, 1, 2], True)
    view_1789: "f32[16]" = torch.ops.aten.reshape.default(sum_679, [16]);  sum_679 = None
    clone_794: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1790, memory_format = torch.contiguous_format);  permute_1790 = None
    view_1790: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_794, [2654208, 16]);  clone_794 = None
    permute_1791: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1790, [1, 0])
    mm_524: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1791, view_9);  permute_1791 = view_9 = None
    permute_1792: "f32[16, 16]" = torch.ops.aten.permute.default(mm_524, [1, 0]);  mm_524 = None
    mm_525: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1790, permute_1793);  view_1790 = permute_1793 = None
    view_1791: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_525, [8, 576, 576, 16]);  mm_525 = None
    permute_1794: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1792, [1, 0]);  permute_1792 = None
    permute_1795: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1791, [0, 3, 1, 2]);  view_1791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    mul_1435: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(permute_1795, alias_75);  permute_1795 = None
    sum_680: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1435, [-1], True)
    mul_1436: "f32[8, 16, 576, 576]" = torch.ops.aten.mul.Tensor(alias_75, sum_680);  alias_75 = sum_680 = None
    sub_376: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(mul_1435, mul_1436);  mul_1435 = mul_1436 = None
    clone_795: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(sub_376, memory_format = torch.contiguous_format);  sub_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1796: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(clone_795, [0, 2, 3, 1]);  clone_795 = None
    sum_681: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_1796, [0, 1, 2], True)
    view_1792: "f32[16]" = torch.ops.aten.reshape.default(sum_681, [16]);  sum_681 = None
    clone_796: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_1796, memory_format = torch.contiguous_format);  permute_1796 = None
    view_1793: "f32[2654208, 16]" = torch.ops.aten.reshape.default(clone_796, [2654208, 16]);  clone_796 = None
    permute_1797: "f32[16, 2654208]" = torch.ops.aten.permute.default(view_1793, [1, 0])
    mm_526: "f32[16, 16]" = torch.ops.aten.mm.default(permute_1797, view_7);  permute_1797 = view_7 = None
    permute_1798: "f32[16, 16]" = torch.ops.aten.permute.default(mm_526, [1, 0]);  mm_526 = None
    mm_527: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_1793, permute_1799);  view_1793 = permute_1799 = None
    view_1794: "f32[8, 576, 576, 16]" = torch.ops.aten.reshape.default(mm_527, [8, 576, 576, 16]);  mm_527 = None
    permute_1800: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1798, [1, 0]);  permute_1798 = None
    permute_1801: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(view_1794, [0, 3, 1, 2]);  view_1794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    clone_797: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_1801, memory_format = torch.contiguous_format);  permute_1801 = None
    view_1795: "f32[128, 576, 576]" = torch.ops.aten.reshape.default(clone_797, [128, 576, 576]);  clone_797 = None
    bmm_214: "f32[128, 48, 576]" = torch.ops.aten.bmm.default(permute_1802, view_1795);  permute_1802 = None
    bmm_215: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_1795, permute_1803);  view_1795 = permute_1803 = None
    view_1796: "f32[8, 16, 48, 576]" = torch.ops.aten.reshape.default(bmm_214, [8, 16, 48, 576]);  bmm_214 = None
    view_1797: "f32[8, 16, 576, 48]" = torch.ops.aten.reshape.default(bmm_215, [8, 16, 576, 48]);  bmm_215 = None
    permute_1804: "f32[8, 16, 576, 48]" = torch.ops.aten.permute.default(view_1796, [0, 1, 3, 2]);  view_1796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_scatter_108: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, view_1787, 0, 2);  view_1787 = None
    select_scatter_109: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, permute_1804, 0, 1);  permute_1804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_568: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(select_scatter_108, select_scatter_109);  select_scatter_108 = select_scatter_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    mul_1437: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(view_1797, 0.14433756729740643);  view_1797 = None
    select_scatter_110: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.select_scatter.default(full_default_6, mul_1437, 0, 0);  full_default_6 = mul_1437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    add_569: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.add.Tensor(add_568, select_scatter_110);  add_568 = select_scatter_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1805: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.permute.default(add_569, [1, 3, 0, 2, 4]);  add_569 = None
    clone_798: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.clone.default(permute_1805, memory_format = torch.contiguous_format);  permute_1805 = None
    view_1798: "f32[8, 576, 2304]" = torch.ops.aten.reshape.default(clone_798, [8, 576, 2304]);  clone_798 = None
    view_1799: "f32[4608, 2304]" = torch.ops.aten.reshape.default(view_1798, [4608, 2304]);  view_1798 = None
    mm_528: "f32[4608, 768]" = torch.ops.aten.mm.default(view_1799, permute_1806);  permute_1806 = None
    permute_1807: "f32[2304, 4608]" = torch.ops.aten.permute.default(view_1799, [1, 0])
    mm_529: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_1807, view_1);  permute_1807 = view_1 = None
    permute_1808: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_529, [1, 0]);  mm_529 = None
    sum_682: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1799, [0], True);  view_1799 = None
    view_1800: "f32[2304]" = torch.ops.aten.reshape.default(sum_682, [2304]);  sum_682 = None
    permute_1809: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1808, [1, 0]);  permute_1808 = None
    view_1801: "f32[8, 576, 768]" = torch.ops.aten.reshape.default(mm_528, [8, 576, 768]);  mm_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1439: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1801, primals_81);  primals_81 = None
    mul_1440: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1439, 768)
    sum_683: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1439, [2], True)
    mul_1441: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_1439, mul);  mul_1439 = None
    sum_684: "f32[8, 576, 1]" = torch.ops.aten.sum.dim_IntList(mul_1441, [2], True);  mul_1441 = None
    mul_1442: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul, sum_684);  sum_684 = None
    sub_378: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(mul_1440, sum_683);  mul_1440 = sum_683 = None
    sub_379: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(sub_378, mul_1442);  sub_378 = mul_1442 = None
    mul_1443: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(div_112, sub_379);  div_112 = sub_379 = None
    mul_1444: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(view_1801, mul);  mul = None
    sum_685: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1444, [0, 1]);  mul_1444 = None
    sum_686: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1801, [0, 1]);  view_1801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    add_570: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_567, mul_1443);  add_567 = mul_1443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:341, code: x = x + self.pos_embed
    sum_687: "f32[1, 576, 768]" = torch.ops.aten.sum.dim_IntList(add_570, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_1810: "f32[8, 768, 576]" = torch.ops.aten.permute.default(add_570, [0, 2, 1]);  add_570 = None
    view_1802: "f32[8, 768, 24, 24]" = torch.ops.aten.reshape.default(permute_1810, [8, 768, 24, 24]);  permute_1810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(view_1802, primals_693, primals_79, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_1802 = primals_693 = primals_79 = None
    getitem_179: "f32[768, 3, 16, 16]" = convolution_backward[1]
    getitem_180: "f32[768]" = convolution_backward[2];  convolution_backward = None
    return [sum_687, view_1781, view_1774, view_1753, view_1746, view_1725, view_1718, view_1697, view_1690, view_1669, view_1662, view_1641, view_1634, view_1613, view_1606, view_1585, view_1578, view_1557, view_1550, view_1529, view_1522, view_1501, view_1494, view_1473, view_1466, view_1445, view_1438, view_1417, view_1410, view_1389, view_1382, view_1361, view_1354, view_1333, view_1326, view_1305, view_1298, view_1277, view_1270, view_1249, view_1242, view_1221, view_1214, view_1193, view_1186, view_1165, view_1158, view_1137, view_1130, view_1109, view_1102, view_1081, view_1074, view_1053, view_1046, view_1025, view_1018, view_997, view_990, view_969, view_962, view_941, view_934, view_913, view_906, view_885, view_878, view_857, view_850, view_829, view_822, view_801, view_794, sum_74, view_779, view_772, view_757, view_750, getitem_179, getitem_180, sum_685, sum_686, permute_1809, view_1800, permute_1800, view_1792, permute_1794, view_1789, permute_1786, view_1783, sum_675, sum_676, permute_1782, view_1779, permute_1778, view_1776, sum_668, sum_669, permute_1774, view_1772, permute_1765, view_1764, permute_1759, view_1761, permute_1751, view_1755, sum_658, sum_659, permute_1747, view_1751, permute_1743, view_1748, sum_651, sum_652, permute_1739, view_1744, permute_1730, view_1736, permute_1724, view_1733, permute_1716, view_1727, sum_641, sum_642, permute_1712, view_1723, permute_1708, view_1720, sum_634, sum_635, permute_1704, view_1716, permute_1695, view_1708, permute_1689, view_1705, permute_1681, view_1699, sum_624, sum_625, permute_1677, view_1695, permute_1673, view_1692, sum_617, sum_618, permute_1669, view_1688, permute_1660, view_1680, permute_1654, view_1677, permute_1646, view_1671, sum_607, sum_608, permute_1642, view_1667, permute_1638, view_1664, sum_600, sum_601, permute_1634, view_1660, permute_1625, view_1652, permute_1619, view_1649, permute_1611, view_1643, sum_590, sum_591, permute_1607, view_1639, permute_1603, view_1636, sum_583, sum_584, permute_1599, view_1632, permute_1590, view_1624, permute_1584, view_1621, permute_1576, view_1615, sum_573, sum_574, permute_1572, view_1611, permute_1568, view_1608, sum_566, sum_567, permute_1564, view_1604, permute_1555, view_1596, permute_1549, view_1593, permute_1541, view_1587, sum_556, sum_557, permute_1537, view_1583, permute_1533, view_1580, sum_549, sum_550, permute_1529, view_1576, permute_1520, view_1568, permute_1514, view_1565, permute_1506, view_1559, sum_539, sum_540, permute_1502, view_1555, permute_1498, view_1552, sum_532, sum_533, permute_1494, view_1548, permute_1485, view_1540, permute_1479, view_1537, permute_1471, view_1531, sum_522, sum_523, permute_1467, view_1527, permute_1463, view_1524, sum_515, sum_516, permute_1459, view_1520, permute_1450, view_1512, permute_1444, view_1509, permute_1436, view_1503, sum_505, sum_506, permute_1432, view_1499, permute_1428, view_1496, sum_498, sum_499, permute_1424, view_1492, permute_1415, view_1484, permute_1409, view_1481, permute_1401, view_1475, sum_488, sum_489, permute_1397, view_1471, permute_1393, view_1468, sum_481, sum_482, permute_1389, view_1464, permute_1380, view_1456, permute_1374, view_1453, permute_1366, view_1447, sum_471, sum_472, permute_1362, view_1443, permute_1358, view_1440, sum_464, sum_465, permute_1354, view_1436, permute_1345, view_1428, permute_1339, view_1425, permute_1331, view_1419, sum_454, sum_455, permute_1327, view_1415, permute_1323, view_1412, sum_447, sum_448, permute_1319, view_1408, permute_1310, view_1400, permute_1304, view_1397, permute_1296, view_1391, sum_437, sum_438, permute_1292, view_1387, permute_1288, view_1384, sum_430, sum_431, permute_1284, view_1380, permute_1275, view_1372, permute_1269, view_1369, permute_1261, view_1363, sum_420, sum_421, permute_1257, view_1359, permute_1253, view_1356, sum_413, sum_414, permute_1249, view_1352, permute_1240, view_1344, permute_1234, view_1341, permute_1226, view_1335, sum_403, sum_404, permute_1222, view_1331, permute_1218, view_1328, sum_396, sum_397, permute_1214, view_1324, permute_1205, view_1316, permute_1199, view_1313, permute_1191, view_1307, sum_386, sum_387, permute_1187, view_1303, permute_1183, view_1300, sum_379, sum_380, permute_1179, view_1296, permute_1170, view_1288, permute_1164, view_1285, permute_1156, view_1279, sum_369, sum_370, permute_1152, view_1275, permute_1148, view_1272, sum_362, sum_363, permute_1144, view_1268, permute_1135, view_1260, permute_1129, view_1257, permute_1121, view_1251, sum_352, sum_353, permute_1117, view_1247, permute_1113, view_1244, sum_345, sum_346, permute_1109, view_1240, permute_1100, view_1232, permute_1094, view_1229, permute_1086, view_1223, sum_335, sum_336, permute_1082, view_1219, permute_1078, view_1216, sum_328, sum_329, permute_1074, view_1212, permute_1065, view_1204, permute_1059, view_1201, permute_1051, view_1195, sum_318, sum_319, permute_1047, view_1191, permute_1043, view_1188, sum_311, sum_312, permute_1039, view_1184, permute_1030, view_1176, permute_1024, view_1173, permute_1016, view_1167, sum_301, sum_302, permute_1012, view_1163, permute_1008, view_1160, sum_294, sum_295, permute_1004, view_1156, permute_995, view_1148, permute_989, view_1145, permute_981, view_1139, sum_284, sum_285, permute_977, view_1135, permute_973, view_1132, sum_277, sum_278, permute_969, view_1128, permute_960, view_1120, permute_954, view_1117, permute_946, view_1111, sum_267, sum_268, permute_942, view_1107, permute_938, view_1104, sum_260, sum_261, permute_934, view_1100, permute_925, view_1092, permute_919, view_1089, permute_911, view_1083, sum_250, sum_251, permute_907, view_1079, permute_903, view_1076, sum_243, sum_244, permute_899, view_1072, permute_890, view_1064, permute_884, view_1061, permute_876, view_1055, sum_233, sum_234, permute_872, view_1051, permute_868, view_1048, sum_226, sum_227, permute_864, view_1044, permute_855, view_1036, permute_849, view_1033, permute_841, view_1027, sum_216, sum_217, permute_837, view_1023, permute_833, view_1020, sum_209, sum_210, permute_829, view_1016, permute_820, view_1008, permute_814, view_1005, permute_806, view_999, sum_199, sum_200, permute_802, view_995, permute_798, view_992, sum_192, sum_193, permute_794, view_988, permute_785, view_980, permute_779, view_977, permute_771, view_971, sum_182, sum_183, permute_767, view_967, permute_763, view_964, sum_175, sum_176, permute_759, view_960, permute_750, view_952, permute_744, view_949, permute_736, view_943, sum_165, sum_166, permute_732, view_939, permute_728, view_936, sum_158, sum_159, permute_724, view_932, permute_715, view_924, permute_709, view_921, permute_701, view_915, sum_148, sum_149, permute_697, view_911, permute_693, view_908, sum_141, sum_142, permute_689, view_904, permute_680, view_896, permute_674, view_893, permute_666, view_887, sum_131, sum_132, permute_662, view_883, permute_658, view_880, sum_124, sum_125, permute_654, view_876, permute_645, view_868, permute_639, view_865, permute_631, view_859, sum_114, sum_115, permute_627, view_855, permute_623, view_852, sum_107, sum_108, permute_619, view_848, permute_610, view_840, permute_604, view_837, permute_596, view_831, sum_97, sum_98, permute_592, view_827, permute_588, view_824, sum_90, sum_91, permute_584, view_820, permute_575, view_812, permute_569, view_809, permute_561, view_803, sum_80, sum_81, permute_557, view_799, permute_553, view_796, sum_72, sum_73, permute_549, view_793, permute_544, view_790, permute_539, view_786, permute_533, view_781, sum_63, sum_64, permute_529, view_777, permute_525, view_774, sum_56, sum_57, permute_521, view_771, permute_516, view_768, permute_511, view_764, permute_505, view_759, sum_47, sum_48, permute_501, view_755, permute_497, view_752, sum_40, sum_41, permute_493, view_749, None]
    