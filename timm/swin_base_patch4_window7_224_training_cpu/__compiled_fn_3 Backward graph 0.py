from __future__ import annotations



def forward(self, primals_25: "f32[128, 3, 4, 4]", primals_27: "f32[128]", primals_28: "f32[128]", primals_29: "f32[128]", primals_30: "f32[128]", primals_35: "f32[128]", primals_36: "f32[128]", primals_41: "f32[128]", primals_42: "f32[128]", primals_47: "f32[128]", primals_48: "f32[128]", primals_53: "f32[512]", primals_54: "f32[512]", primals_56: "f32[256]", primals_57: "f32[256]", primals_62: "f32[256]", primals_63: "f32[256]", primals_68: "f32[256]", primals_69: "f32[256]", primals_74: "f32[256]", primals_75: "f32[256]", primals_80: "f32[1024]", primals_81: "f32[1024]", primals_83: "f32[512]", primals_84: "f32[512]", primals_89: "f32[512]", primals_90: "f32[512]", primals_95: "f32[512]", primals_96: "f32[512]", primals_101: "f32[512]", primals_102: "f32[512]", primals_107: "f32[512]", primals_108: "f32[512]", primals_113: "f32[512]", primals_114: "f32[512]", primals_119: "f32[512]", primals_120: "f32[512]", primals_125: "f32[512]", primals_126: "f32[512]", primals_131: "f32[512]", primals_132: "f32[512]", primals_137: "f32[512]", primals_138: "f32[512]", primals_143: "f32[512]", primals_144: "f32[512]", primals_149: "f32[512]", primals_150: "f32[512]", primals_155: "f32[512]", primals_156: "f32[512]", primals_161: "f32[512]", primals_162: "f32[512]", primals_167: "f32[512]", primals_168: "f32[512]", primals_173: "f32[512]", primals_174: "f32[512]", primals_179: "f32[512]", primals_180: "f32[512]", primals_185: "f32[512]", primals_186: "f32[512]", primals_191: "f32[512]", primals_192: "f32[512]", primals_197: "f32[512]", primals_198: "f32[512]", primals_203: "f32[512]", primals_204: "f32[512]", primals_209: "f32[512]", primals_210: "f32[512]", primals_215: "f32[512]", primals_216: "f32[512]", primals_221: "f32[512]", primals_222: "f32[512]", primals_227: "f32[512]", primals_228: "f32[512]", primals_233: "f32[512]", primals_234: "f32[512]", primals_239: "f32[512]", primals_240: "f32[512]", primals_245: "f32[512]", primals_246: "f32[512]", primals_251: "f32[512]", primals_252: "f32[512]", primals_257: "f32[512]", primals_258: "f32[512]", primals_263: "f32[512]", primals_264: "f32[512]", primals_269: "f32[512]", primals_270: "f32[512]", primals_275: "f32[512]", primals_276: "f32[512]", primals_281: "f32[512]", primals_282: "f32[512]", primals_287: "f32[512]", primals_288: "f32[512]", primals_293: "f32[512]", primals_294: "f32[512]", primals_299: "f32[2048]", primals_300: "f32[2048]", primals_302: "f32[1024]", primals_303: "f32[1024]", primals_308: "f32[1024]", primals_309: "f32[1024]", primals_314: "f32[1024]", primals_315: "f32[1024]", primals_320: "f32[1024]", primals_321: "f32[1024]", primals_326: "f32[1024]", primals_327: "f32[1024]", primals_365: "f32[8, 3, 224, 224]", permute: "f32[8, 56, 56, 128]", getitem: "f32[8, 56, 56, 128]", getitem_1: "f32[8, 56, 56, 1]", getitem_2: "f32[8, 56, 56, 1]", getitem_4: "f32[8, 56, 56, 1]", getitem_5: "f32[8, 56, 56, 1]", view_3: "f32[25088, 128]", view_7: "i64[2401]", view_11: "f32[25088, 128]", view_16: "f32[8, 3136, 128]", getitem_10: "f32[8, 3136, 1]", getitem_11: "f32[8, 3136, 1]", view_17: "f32[25088, 128]", addmm_2: "f32[25088, 512]", view_19: "f32[25088, 512]", view_21: "f32[8, 56, 56, 128]", getitem_13: "f32[8, 56, 56, 1]", getitem_14: "f32[8, 56, 56, 1]", view_25: "f32[25088, 128]", view_29: "i64[2401]", view_35: "f32[25088, 128]", bernoulli: "f32[8, 1, 1, 1]", view_40: "f32[8, 3136, 128]", getitem_19: "f32[8, 3136, 1]", getitem_20: "f32[8, 3136, 1]", view_41: "f32[25088, 128]", addmm_6: "f32[25088, 512]", view_43: "f32[25088, 512]", bernoulli_1: "f32[8, 1, 1]", _unsafe_view_8: "f32[8, 28, 28, 512]", getitem_22: "f32[8, 28, 28, 1]", getitem_23: "f32[8, 28, 28, 1]", view_47: "f32[6272, 512]", view_48: "f32[8, 28, 28, 256]", getitem_25: "f32[8, 28, 28, 1]", getitem_26: "f32[8, 28, 28, 1]", view_52: "f32[6272, 256]", view_56: "i64[2401]", view_60: "f32[6272, 256]", bernoulli_2: "f32[8, 1, 1, 1]", view_65: "f32[8, 784, 256]", getitem_31: "f32[8, 784, 1]", getitem_32: "f32[8, 784, 1]", view_66: "f32[6272, 256]", addmm_10: "f32[6272, 1024]", view_68: "f32[6272, 1024]", bernoulli_3: "f32[8, 1, 1]", view_70: "f32[8, 28, 28, 256]", getitem_34: "f32[8, 28, 28, 1]", getitem_35: "f32[8, 28, 28, 1]", view_74: "f32[6272, 256]", view_78: "i64[2401]", view_84: "f32[6272, 256]", bernoulli_4: "f32[8, 1, 1, 1]", view_89: "f32[8, 784, 256]", getitem_40: "f32[8, 784, 1]", getitem_41: "f32[8, 784, 1]", view_90: "f32[6272, 256]", addmm_14: "f32[6272, 1024]", view_92: "f32[6272, 1024]", bernoulli_5: "f32[8, 1, 1]", _unsafe_view_17: "f32[8, 14, 14, 1024]", getitem_43: "f32[8, 14, 14, 1]", getitem_44: "f32[8, 14, 14, 1]", view_96: "f32[1568, 1024]", view_97: "f32[8, 14, 14, 512]", getitem_46: "f32[8, 14, 14, 1]", getitem_47: "f32[8, 14, 14, 1]", view_101: "f32[1568, 512]", view_105: "i64[2401]", view_109: "f32[1568, 512]", bernoulli_6: "f32[8, 1, 1, 1]", view_114: "f32[8, 196, 512]", getitem_52: "f32[8, 196, 1]", getitem_53: "f32[8, 196, 1]", view_115: "f32[1568, 512]", addmm_18: "f32[1568, 2048]", view_117: "f32[1568, 2048]", bernoulli_7: "f32[8, 1, 1]", view_119: "f32[8, 14, 14, 512]", getitem_55: "f32[8, 14, 14, 1]", getitem_56: "f32[8, 14, 14, 1]", view_123: "f32[1568, 512]", view_127: "i64[2401]", view_133: "f32[1568, 512]", bernoulli_8: "f32[8, 1, 1, 1]", view_138: "f32[8, 196, 512]", getitem_61: "f32[8, 196, 1]", getitem_62: "f32[8, 196, 1]", view_139: "f32[1568, 512]", addmm_22: "f32[1568, 2048]", view_141: "f32[1568, 2048]", bernoulli_9: "f32[8, 1, 1]", view_143: "f32[8, 14, 14, 512]", getitem_64: "f32[8, 14, 14, 1]", getitem_65: "f32[8, 14, 14, 1]", view_147: "f32[1568, 512]", view_151: "i64[2401]", view_155: "f32[1568, 512]", bernoulli_10: "f32[8, 1, 1, 1]", view_160: "f32[8, 196, 512]", getitem_70: "f32[8, 196, 1]", getitem_71: "f32[8, 196, 1]", view_161: "f32[1568, 512]", addmm_26: "f32[1568, 2048]", view_163: "f32[1568, 2048]", bernoulli_11: "f32[8, 1, 1]", view_165: "f32[8, 14, 14, 512]", getitem_73: "f32[8, 14, 14, 1]", getitem_74: "f32[8, 14, 14, 1]", view_169: "f32[1568, 512]", view_173: "i64[2401]", view_179: "f32[1568, 512]", bernoulli_12: "f32[8, 1, 1, 1]", view_184: "f32[8, 196, 512]", getitem_79: "f32[8, 196, 1]", getitem_80: "f32[8, 196, 1]", view_185: "f32[1568, 512]", addmm_30: "f32[1568, 2048]", view_187: "f32[1568, 2048]", bernoulli_13: "f32[8, 1, 1]", view_189: "f32[8, 14, 14, 512]", getitem_82: "f32[8, 14, 14, 1]", getitem_83: "f32[8, 14, 14, 1]", view_193: "f32[1568, 512]", view_197: "i64[2401]", view_201: "f32[1568, 512]", bernoulli_14: "f32[8, 1, 1, 1]", view_206: "f32[8, 196, 512]", getitem_88: "f32[8, 196, 1]", getitem_89: "f32[8, 196, 1]", view_207: "f32[1568, 512]", addmm_34: "f32[1568, 2048]", view_209: "f32[1568, 2048]", bernoulli_15: "f32[8, 1, 1]", view_211: "f32[8, 14, 14, 512]", getitem_91: "f32[8, 14, 14, 1]", getitem_92: "f32[8, 14, 14, 1]", view_215: "f32[1568, 512]", view_219: "i64[2401]", view_225: "f32[1568, 512]", bernoulli_16: "f32[8, 1, 1, 1]", view_230: "f32[8, 196, 512]", getitem_97: "f32[8, 196, 1]", getitem_98: "f32[8, 196, 1]", view_231: "f32[1568, 512]", addmm_38: "f32[1568, 2048]", view_233: "f32[1568, 2048]", bernoulli_17: "f32[8, 1, 1]", view_235: "f32[8, 14, 14, 512]", getitem_100: "f32[8, 14, 14, 1]", getitem_101: "f32[8, 14, 14, 1]", view_239: "f32[1568, 512]", view_243: "i64[2401]", view_247: "f32[1568, 512]", bernoulli_18: "f32[8, 1, 1, 1]", view_252: "f32[8, 196, 512]", getitem_106: "f32[8, 196, 1]", getitem_107: "f32[8, 196, 1]", view_253: "f32[1568, 512]", addmm_42: "f32[1568, 2048]", view_255: "f32[1568, 2048]", bernoulli_19: "f32[8, 1, 1]", view_257: "f32[8, 14, 14, 512]", getitem_109: "f32[8, 14, 14, 1]", getitem_110: "f32[8, 14, 14, 1]", view_261: "f32[1568, 512]", view_265: "i64[2401]", view_271: "f32[1568, 512]", bernoulli_20: "f32[8, 1, 1, 1]", view_276: "f32[8, 196, 512]", getitem_115: "f32[8, 196, 1]", getitem_116: "f32[8, 196, 1]", view_277: "f32[1568, 512]", addmm_46: "f32[1568, 2048]", view_279: "f32[1568, 2048]", bernoulli_21: "f32[8, 1, 1]", view_281: "f32[8, 14, 14, 512]", getitem_118: "f32[8, 14, 14, 1]", getitem_119: "f32[8, 14, 14, 1]", view_285: "f32[1568, 512]", view_289: "i64[2401]", view_293: "f32[1568, 512]", bernoulli_22: "f32[8, 1, 1, 1]", view_298: "f32[8, 196, 512]", getitem_124: "f32[8, 196, 1]", getitem_125: "f32[8, 196, 1]", view_299: "f32[1568, 512]", addmm_50: "f32[1568, 2048]", view_301: "f32[1568, 2048]", bernoulli_23: "f32[8, 1, 1]", view_303: "f32[8, 14, 14, 512]", getitem_127: "f32[8, 14, 14, 1]", getitem_128: "f32[8, 14, 14, 1]", view_307: "f32[1568, 512]", view_311: "i64[2401]", view_317: "f32[1568, 512]", bernoulli_24: "f32[8, 1, 1, 1]", view_322: "f32[8, 196, 512]", getitem_133: "f32[8, 196, 1]", getitem_134: "f32[8, 196, 1]", view_323: "f32[1568, 512]", addmm_54: "f32[1568, 2048]", view_325: "f32[1568, 2048]", bernoulli_25: "f32[8, 1, 1]", view_327: "f32[8, 14, 14, 512]", getitem_136: "f32[8, 14, 14, 1]", getitem_137: "f32[8, 14, 14, 1]", view_331: "f32[1568, 512]", view_335: "i64[2401]", view_339: "f32[1568, 512]", bernoulli_26: "f32[8, 1, 1, 1]", view_344: "f32[8, 196, 512]", getitem_142: "f32[8, 196, 1]", getitem_143: "f32[8, 196, 1]", view_345: "f32[1568, 512]", addmm_58: "f32[1568, 2048]", view_347: "f32[1568, 2048]", bernoulli_27: "f32[8, 1, 1]", view_349: "f32[8, 14, 14, 512]", getitem_145: "f32[8, 14, 14, 1]", getitem_146: "f32[8, 14, 14, 1]", view_353: "f32[1568, 512]", view_357: "i64[2401]", view_363: "f32[1568, 512]", bernoulli_28: "f32[8, 1, 1, 1]", view_368: "f32[8, 196, 512]", getitem_151: "f32[8, 196, 1]", getitem_152: "f32[8, 196, 1]", view_369: "f32[1568, 512]", addmm_62: "f32[1568, 2048]", view_371: "f32[1568, 2048]", bernoulli_29: "f32[8, 1, 1]", view_373: "f32[8, 14, 14, 512]", getitem_154: "f32[8, 14, 14, 1]", getitem_155: "f32[8, 14, 14, 1]", view_377: "f32[1568, 512]", view_381: "i64[2401]", view_385: "f32[1568, 512]", bernoulli_30: "f32[8, 1, 1, 1]", view_390: "f32[8, 196, 512]", getitem_160: "f32[8, 196, 1]", getitem_161: "f32[8, 196, 1]", view_391: "f32[1568, 512]", addmm_66: "f32[1568, 2048]", view_393: "f32[1568, 2048]", bernoulli_31: "f32[8, 1, 1]", view_395: "f32[8, 14, 14, 512]", getitem_163: "f32[8, 14, 14, 1]", getitem_164: "f32[8, 14, 14, 1]", view_399: "f32[1568, 512]", view_403: "i64[2401]", view_409: "f32[1568, 512]", bernoulli_32: "f32[8, 1, 1, 1]", view_414: "f32[8, 196, 512]", getitem_169: "f32[8, 196, 1]", getitem_170: "f32[8, 196, 1]", view_415: "f32[1568, 512]", addmm_70: "f32[1568, 2048]", view_417: "f32[1568, 2048]", bernoulli_33: "f32[8, 1, 1]", view_419: "f32[8, 14, 14, 512]", getitem_172: "f32[8, 14, 14, 1]", getitem_173: "f32[8, 14, 14, 1]", view_423: "f32[1568, 512]", view_427: "i64[2401]", view_431: "f32[1568, 512]", bernoulli_34: "f32[8, 1, 1, 1]", view_436: "f32[8, 196, 512]", getitem_178: "f32[8, 196, 1]", getitem_179: "f32[8, 196, 1]", view_437: "f32[1568, 512]", addmm_74: "f32[1568, 2048]", view_439: "f32[1568, 2048]", bernoulli_35: "f32[8, 1, 1]", view_441: "f32[8, 14, 14, 512]", getitem_181: "f32[8, 14, 14, 1]", getitem_182: "f32[8, 14, 14, 1]", view_445: "f32[1568, 512]", view_449: "i64[2401]", view_455: "f32[1568, 512]", bernoulli_36: "f32[8, 1, 1, 1]", view_460: "f32[8, 196, 512]", getitem_187: "f32[8, 196, 1]", getitem_188: "f32[8, 196, 1]", view_461: "f32[1568, 512]", addmm_78: "f32[1568, 2048]", view_463: "f32[1568, 2048]", bernoulli_37: "f32[8, 1, 1]", view_465: "f32[8, 14, 14, 512]", getitem_190: "f32[8, 14, 14, 1]", getitem_191: "f32[8, 14, 14, 1]", view_469: "f32[1568, 512]", view_473: "i64[2401]", view_477: "f32[1568, 512]", bernoulli_38: "f32[8, 1, 1, 1]", view_482: "f32[8, 196, 512]", getitem_196: "f32[8, 196, 1]", getitem_197: "f32[8, 196, 1]", view_483: "f32[1568, 512]", addmm_82: "f32[1568, 2048]", view_485: "f32[1568, 2048]", bernoulli_39: "f32[8, 1, 1]", view_487: "f32[8, 14, 14, 512]", getitem_199: "f32[8, 14, 14, 1]", getitem_200: "f32[8, 14, 14, 1]", view_491: "f32[1568, 512]", view_495: "i64[2401]", view_501: "f32[1568, 512]", bernoulli_40: "f32[8, 1, 1, 1]", view_506: "f32[8, 196, 512]", getitem_205: "f32[8, 196, 1]", getitem_206: "f32[8, 196, 1]", view_507: "f32[1568, 512]", addmm_86: "f32[1568, 2048]", view_509: "f32[1568, 2048]", bernoulli_41: "f32[8, 1, 1]", _unsafe_view_90: "f32[8, 7, 7, 2048]", getitem_208: "f32[8, 7, 7, 1]", getitem_209: "f32[8, 7, 7, 1]", view_513: "f32[392, 2048]", view_514: "f32[8, 7, 7, 1024]", getitem_211: "f32[8, 7, 7, 1]", getitem_212: "f32[8, 7, 7, 1]", view_518: "f32[392, 1024]", view_522: "i64[2401]", view_526: "f32[392, 1024]", bernoulli_42: "f32[8, 1, 1, 1]", view_531: "f32[8, 49, 1024]", getitem_217: "f32[8, 49, 1]", getitem_218: "f32[8, 49, 1]", view_532: "f32[392, 1024]", addmm_90: "f32[392, 4096]", view_534: "f32[392, 4096]", bernoulli_43: "f32[8, 1, 1]", view_536: "f32[8, 7, 7, 1024]", getitem_220: "f32[8, 7, 7, 1]", getitem_221: "f32[8, 7, 7, 1]", view_540: "f32[392, 1024]", view_544: "i64[2401]", view_548: "f32[392, 1024]", bernoulli_44: "f32[8, 1, 1, 1]", view_553: "f32[8, 49, 1024]", getitem_226: "f32[8, 49, 1]", getitem_227: "f32[8, 49, 1]", view_554: "f32[392, 1024]", addmm_94: "f32[392, 4096]", view_556: "f32[392, 4096]", bernoulli_45: "f32[8, 1, 1]", view_558: "f32[8, 7, 7, 1024]", getitem_229: "f32[8, 7, 7, 1]", getitem_230: "f32[8, 7, 7, 1]", clone_263: "f32[8, 1024]", t_100: "f32[1000, 1024]", t_104: "f32[1024, 4096]", t_108: "f32[4096, 1024]", t_112: "f32[1024, 1024]", transpose_49: "f32[256, 49, 49]", transpose_50: "f32[256, 32, 49]", detach_24: "f32[8, 32, 49, 49]", transpose_51: "f32[256, 32, 49]", transpose_52: "f32[256, 49, 32]", t_116: "f32[3072, 1024]", t_120: "f32[1024, 4096]", t_124: "f32[4096, 1024]", t_128: "f32[1024, 1024]", transpose_55: "f32[256, 49, 49]", transpose_56: "f32[256, 32, 49]", detach_25: "f32[8, 32, 49, 49]", transpose_57: "f32[256, 32, 49]", transpose_58: "f32[256, 49, 32]", t_132: "f32[3072, 1024]", t_138: "f32[1024, 2048]", t_140: "f32[512, 2048]", t_144: "f32[2048, 512]", t_148: "f32[512, 512]", transpose_61: "f32[512, 49, 49]", transpose_62: "f32[512, 32, 49]", detach_26: "f32[32, 16, 49, 49]", transpose_63: "f32[512, 32, 49]", transpose_64: "f32[512, 49, 32]", t_152: "f32[1536, 512]", t_156: "f32[512, 2048]", t_160: "f32[2048, 512]", t_164: "f32[512, 512]", transpose_67: "f32[512, 49, 49]", transpose_68: "f32[512, 32, 49]", detach_27: "f32[32, 16, 49, 49]", transpose_69: "f32[512, 32, 49]", transpose_70: "f32[512, 49, 32]", t_168: "f32[1536, 512]", t_172: "f32[512, 2048]", t_176: "f32[2048, 512]", t_180: "f32[512, 512]", transpose_73: "f32[512, 49, 49]", transpose_74: "f32[512, 32, 49]", detach_28: "f32[32, 16, 49, 49]", transpose_75: "f32[512, 32, 49]", transpose_76: "f32[512, 49, 32]", t_184: "f32[1536, 512]", t_188: "f32[512, 2048]", t_192: "f32[2048, 512]", t_196: "f32[512, 512]", transpose_79: "f32[512, 49, 49]", transpose_80: "f32[512, 32, 49]", detach_29: "f32[32, 16, 49, 49]", transpose_81: "f32[512, 32, 49]", transpose_82: "f32[512, 49, 32]", t_200: "f32[1536, 512]", t_204: "f32[512, 2048]", t_208: "f32[2048, 512]", t_212: "f32[512, 512]", transpose_85: "f32[512, 49, 49]", transpose_86: "f32[512, 32, 49]", detach_30: "f32[32, 16, 49, 49]", transpose_87: "f32[512, 32, 49]", transpose_88: "f32[512, 49, 32]", t_216: "f32[1536, 512]", t_220: "f32[512, 2048]", t_224: "f32[2048, 512]", t_228: "f32[512, 512]", transpose_91: "f32[512, 49, 49]", transpose_92: "f32[512, 32, 49]", detach_31: "f32[32, 16, 49, 49]", transpose_93: "f32[512, 32, 49]", transpose_94: "f32[512, 49, 32]", t_232: "f32[1536, 512]", t_236: "f32[512, 2048]", t_240: "f32[2048, 512]", t_244: "f32[512, 512]", transpose_97: "f32[512, 49, 49]", transpose_98: "f32[512, 32, 49]", detach_32: "f32[32, 16, 49, 49]", transpose_99: "f32[512, 32, 49]", transpose_100: "f32[512, 49, 32]", t_248: "f32[1536, 512]", t_252: "f32[512, 2048]", t_256: "f32[2048, 512]", t_260: "f32[512, 512]", transpose_103: "f32[512, 49, 49]", transpose_104: "f32[512, 32, 49]", detach_33: "f32[32, 16, 49, 49]", transpose_105: "f32[512, 32, 49]", transpose_106: "f32[512, 49, 32]", t_264: "f32[1536, 512]", t_268: "f32[512, 2048]", t_272: "f32[2048, 512]", t_276: "f32[512, 512]", transpose_109: "f32[512, 49, 49]", transpose_110: "f32[512, 32, 49]", detach_34: "f32[32, 16, 49, 49]", transpose_111: "f32[512, 32, 49]", transpose_112: "f32[512, 49, 32]", t_280: "f32[1536, 512]", t_284: "f32[512, 2048]", t_288: "f32[2048, 512]", t_292: "f32[512, 512]", transpose_115: "f32[512, 49, 49]", transpose_116: "f32[512, 32, 49]", detach_35: "f32[32, 16, 49, 49]", transpose_117: "f32[512, 32, 49]", transpose_118: "f32[512, 49, 32]", t_296: "f32[1536, 512]", t_300: "f32[512, 2048]", t_304: "f32[2048, 512]", t_308: "f32[512, 512]", transpose_121: "f32[512, 49, 49]", transpose_122: "f32[512, 32, 49]", detach_36: "f32[32, 16, 49, 49]", transpose_123: "f32[512, 32, 49]", transpose_124: "f32[512, 49, 32]", t_312: "f32[1536, 512]", t_316: "f32[512, 2048]", t_320: "f32[2048, 512]", t_324: "f32[512, 512]", transpose_127: "f32[512, 49, 49]", transpose_128: "f32[512, 32, 49]", detach_37: "f32[32, 16, 49, 49]", transpose_129: "f32[512, 32, 49]", transpose_130: "f32[512, 49, 32]", t_328: "f32[1536, 512]", t_332: "f32[512, 2048]", t_336: "f32[2048, 512]", t_340: "f32[512, 512]", transpose_133: "f32[512, 49, 49]", transpose_134: "f32[512, 32, 49]", detach_38: "f32[32, 16, 49, 49]", transpose_135: "f32[512, 32, 49]", transpose_136: "f32[512, 49, 32]", t_344: "f32[1536, 512]", t_348: "f32[512, 2048]", t_352: "f32[2048, 512]", t_356: "f32[512, 512]", transpose_139: "f32[512, 49, 49]", transpose_140: "f32[512, 32, 49]", detach_39: "f32[32, 16, 49, 49]", transpose_141: "f32[512, 32, 49]", transpose_142: "f32[512, 49, 32]", t_360: "f32[1536, 512]", t_364: "f32[512, 2048]", t_368: "f32[2048, 512]", t_372: "f32[512, 512]", transpose_145: "f32[512, 49, 49]", transpose_146: "f32[512, 32, 49]", detach_40: "f32[32, 16, 49, 49]", transpose_147: "f32[512, 32, 49]", transpose_148: "f32[512, 49, 32]", t_376: "f32[1536, 512]", t_380: "f32[512, 2048]", t_384: "f32[2048, 512]", t_388: "f32[512, 512]", transpose_151: "f32[512, 49, 49]", transpose_152: "f32[512, 32, 49]", detach_41: "f32[32, 16, 49, 49]", transpose_153: "f32[512, 32, 49]", transpose_154: "f32[512, 49, 32]", t_392: "f32[1536, 512]", t_396: "f32[512, 2048]", t_400: "f32[2048, 512]", t_404: "f32[512, 512]", transpose_157: "f32[512, 49, 49]", transpose_158: "f32[512, 32, 49]", detach_42: "f32[32, 16, 49, 49]", transpose_159: "f32[512, 32, 49]", transpose_160: "f32[512, 49, 32]", t_408: "f32[1536, 512]", t_412: "f32[512, 2048]", t_416: "f32[2048, 512]", t_420: "f32[512, 512]", transpose_163: "f32[512, 49, 49]", transpose_164: "f32[512, 32, 49]", detach_43: "f32[32, 16, 49, 49]", transpose_165: "f32[512, 32, 49]", transpose_166: "f32[512, 49, 32]", t_424: "f32[1536, 512]", t_430: "f32[512, 1024]", t_432: "f32[256, 1024]", t_436: "f32[1024, 256]", t_440: "f32[256, 256]", transpose_169: "f32[1024, 49, 49]", transpose_170: "f32[1024, 32, 49]", detach_44: "f32[128, 8, 49, 49]", transpose_171: "f32[1024, 32, 49]", transpose_172: "f32[1024, 49, 32]", t_444: "f32[768, 256]", t_448: "f32[256, 1024]", t_452: "f32[1024, 256]", t_456: "f32[256, 256]", transpose_175: "f32[1024, 49, 49]", transpose_176: "f32[1024, 32, 49]", detach_45: "f32[128, 8, 49, 49]", transpose_177: "f32[1024, 32, 49]", transpose_178: "f32[1024, 49, 32]", t_460: "f32[768, 256]", t_466: "f32[256, 512]", t_468: "f32[128, 512]", t_472: "f32[512, 128]", t_476: "f32[128, 128]", transpose_181: "f32[2048, 49, 49]", transpose_182: "f32[2048, 32, 49]", detach_46: "f32[512, 4, 49, 49]", transpose_183: "f32[2048, 32, 49]", transpose_184: "f32[2048, 49, 32]", t_480: "f32[384, 128]", t_484: "f32[128, 512]", t_488: "f32[512, 128]", t_492: "f32[128, 128]", transpose_187: "f32[2048, 49, 49]", transpose_188: "f32[2048, 32, 49]", detach_47: "f32[512, 4, 49, 49]", transpose_189: "f32[2048, 32, 49]", transpose_190: "f32[2048, 49, 32]", t_496: "f32[384, 128]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_18: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_2, [8, 3136, 512]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli, 0.9956521736457944);  bernoulli = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_42: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_6, [8, 3136, 512]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_1: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_1, 0.9956521736457944);  bernoulli_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_2: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_2, 0.9913043472915888);  bernoulli_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_67: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_10, [8, 784, 1024]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_3: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_3, 0.9913043472915888);  bernoulli_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_4: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_4, 0.9869565209373832);  bernoulli_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_91: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_14, [8, 784, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_5: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_5, 0.9869565209373832);  bernoulli_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_6: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_6, 0.9826086945831776);  bernoulli_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_116: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_18, [8, 196, 2048]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_7: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_7, 0.9826086945831776);  bernoulli_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_8: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_8, 0.9782608672976494);  bernoulli_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_140: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_22, [8, 196, 2048]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_9: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_9, 0.9782608672976494);  bernoulli_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_10: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_10, 0.9739130418747663);  bernoulli_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_162: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_26, [8, 196, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_11: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_11, 0.9739130418747663);  bernoulli_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_12: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_12, 0.9695652164518833);  bernoulli_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_186: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_30, [8, 196, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_13: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_13, 0.9695652164518833);  bernoulli_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_14: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_14, 0.9652173891663551);  bernoulli_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_208: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_34, [8, 196, 2048]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_15: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_15, 0.9652173891663551);  bernoulli_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_16: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_16, 0.960869561880827);  bernoulli_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_232: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_38, [8, 196, 2048]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_17: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_17, 0.960869561880827);  bernoulli_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_18: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_18, 0.9565217345952988);  bernoulli_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_254: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_42, [8, 196, 2048]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_19: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_19, 0.9565217345952988);  bernoulli_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_20: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_20, 0.9521739110350609);  bernoulli_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_278: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_46, [8, 196, 2048]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_21: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_21, 0.9521739110350609);  bernoulli_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_22: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_22, 0.947826087474823);  bernoulli_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_300: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_50, [8, 196, 2048]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_23: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_23, 0.947826087474823);  bernoulli_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_24: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_24, 0.9434782639145851);  bernoulli_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_324: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_54, [8, 196, 2048]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_25: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_25, 0.9434782639145851);  bernoulli_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_26: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_26, 0.9391304366290569);  bernoulli_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_346: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_58, [8, 196, 2048]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_27: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_27, 0.9391304366290569);  bernoulli_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_28: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_28, 0.9347826093435287);  bernoulli_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_370: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_62, [8, 196, 2048]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_29: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_29, 0.9347826093435287);  bernoulli_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_30: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_30, 0.9304347857832909);  bernoulli_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_392: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_66, [8, 196, 2048]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_31: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_31, 0.9304347857832909);  bernoulli_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_32: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_32, 0.9260869547724724);  bernoulli_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_416: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_70, [8, 196, 2048]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_33: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_33, 0.9260869547724724);  bernoulli_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_34: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_34, 0.9217391312122345);  bernoulli_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_438: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_74, [8, 196, 2048]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_35: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_35, 0.9217391312122345);  bernoulli_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_36: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_36, 0.917391300201416);  bernoulli_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_462: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_78, [8, 196, 2048]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_37: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_37, 0.917391300201416);  bernoulli_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_38: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_38, 0.9130434766411781);  bernoulli_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_484: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_82, [8, 196, 2048]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_39: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_39, 0.9130434766411781);  bernoulli_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_40: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_40, 0.9086956530809402);  bernoulli_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_508: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_86, [8, 196, 2048]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_41: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_41, 0.9086956530809402);  bernoulli_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_42: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_42, 0.9043478220701218);  bernoulli_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_533: "f32[8, 49, 4096]" = torch.ops.aten.view.default(addmm_90, [8, 49, 4096]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_43: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_43, 0.9043478220701218);  bernoulli_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_44: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_44, 0.8999999985098839);  bernoulli_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_555: "f32[8, 49, 4096]" = torch.ops.aten.view.default(addmm_94, [8, 49, 4096]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_45: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_45, 0.8999999985098839);  bernoulli_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm_3: "f32[8, 1024]" = torch.ops.aten.mm.default(tangents_1, t_100);  t_100 = None
    t_101: "f32[1000, 8]" = torch.ops.aten.t.default(tangents_1)
    mm_4: "f32[1000, 1024]" = torch.ops.aten.mm.default(t_101, clone_263);  t_101 = clone_263 = None
    t_102: "f32[1024, 1000]" = torch.ops.aten.t.default(mm_4);  mm_4 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_559: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    t_103: "f32[1000, 1024]" = torch.ops.aten.t.default(t_102);  t_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:65, code: return x.mean(self.dim, keepdim=not self.flatten)
    unsqueeze_46: "f32[8, 1, 1024]" = torch.ops.aten.unsqueeze.default(mm_3, 1);  mm_3 = None
    unsqueeze_47: "f32[8, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 2);  unsqueeze_46 = None
    expand_96: "f32[8, 7, 7, 1024]" = torch.ops.aten.expand.default(unsqueeze_47, [8, 7, 7, 1024]);  unsqueeze_47 = None
    div_46: "f32[8, 7, 7, 1024]" = torch.ops.aten.div.Scalar(expand_96, 49);  expand_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:610, code: x = self.norm(x)
    native_layer_norm_backward = torch.ops.aten.native_layer_norm_backward.default(div_46, view_558, [1024], getitem_229, getitem_230, primals_326, primals_327, [True, True, True]);  div_46 = view_558 = getitem_229 = getitem_230 = primals_326 = primals_327 = None
    getitem_231: "f32[8, 7, 7, 1024]" = native_layer_norm_backward[0]
    getitem_232: "f32[1024]" = native_layer_norm_backward[1]
    getitem_233: "f32[1024]" = native_layer_norm_backward[2];  native_layer_norm_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_560: "f32[8, 49, 1024]" = torch.ops.aten.view.default(getitem_231, [8, 49, 1024]);  getitem_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_70: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(view_560, div_45);  div_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_561: "f32[392, 1024]" = torch.ops.aten.view.default(mul_70, [392, 1024]);  mul_70 = None
    mm_5: "f32[392, 4096]" = torch.ops.aten.mm.default(view_561, t_104);  t_104 = None
    t_105: "f32[1024, 392]" = torch.ops.aten.t.default(view_561)
    mm_6: "f32[1024, 4096]" = torch.ops.aten.mm.default(t_105, view_556);  t_105 = view_556 = None
    t_106: "f32[4096, 1024]" = torch.ops.aten.t.default(mm_6);  mm_6 = None
    sum_2: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
    view_562: "f32[1024]" = torch.ops.aten.view.default(sum_2, [1024]);  sum_2 = None
    t_107: "f32[1024, 4096]" = torch.ops.aten.t.default(t_106);  t_106 = None
    view_563: "f32[8, 49, 4096]" = torch.ops.aten.view.default(mm_5, [8, 49, 4096]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward: "f32[8, 49, 4096]" = torch.ops.aten.gelu_backward.default(view_563, view_555);  view_563 = view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_564: "f32[392, 4096]" = torch.ops.aten.view.default(gelu_backward, [392, 4096]);  gelu_backward = None
    mm_7: "f32[392, 1024]" = torch.ops.aten.mm.default(view_564, t_108);  t_108 = None
    t_109: "f32[4096, 392]" = torch.ops.aten.t.default(view_564)
    mm_8: "f32[4096, 1024]" = torch.ops.aten.mm.default(t_109, view_554);  t_109 = view_554 = None
    t_110: "f32[1024, 4096]" = torch.ops.aten.t.default(mm_8);  mm_8 = None
    sum_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_564, [0], True);  view_564 = None
    view_565: "f32[4096]" = torch.ops.aten.view.default(sum_3, [4096]);  sum_3 = None
    t_111: "f32[4096, 1024]" = torch.ops.aten.t.default(t_110);  t_110 = None
    view_566: "f32[8, 49, 1024]" = torch.ops.aten.view.default(mm_7, [8, 49, 1024]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_1 = torch.ops.aten.native_layer_norm_backward.default(view_566, view_553, [1024], getitem_226, getitem_227, primals_320, primals_321, [True, True, True]);  view_566 = view_553 = getitem_226 = getitem_227 = primals_320 = primals_321 = None
    getitem_234: "f32[8, 49, 1024]" = native_layer_norm_backward_1[0]
    getitem_235: "f32[1024]" = native_layer_norm_backward_1[1]
    getitem_236: "f32[1024]" = native_layer_norm_backward_1[2];  native_layer_norm_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_83: "f32[8, 49, 1024]" = torch.ops.aten.add.Tensor(view_560, getitem_234);  view_560 = getitem_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_567: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(add_83, [8, 7, 7, 1024]);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_71: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_567, div_44);  div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward: "f32[8, 7, 7, 1024]" = torch.ops.aten.slice_backward.default(mul_71, [8, 7, 7, 1024], 3, 0, 9223372036854775807, 1);  mul_71 = None
    slice_backward_1: "f32[8, 7, 7, 1024]" = torch.ops.aten.slice_backward.default(slice_backward, [8, 7, 7, 1024], 0, 0, 9223372036854775807, 1);  slice_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_568: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.view.default(slice_backward_1, [8, 1, 7, 1, 7, 1024]);  slice_backward_1 = None
    permute_100: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_568, [0, 1, 3, 2, 4, 5]);  view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_569: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(permute_100, [8, 7, 7, 1024]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_570: "f32[8, 49, 1024]" = torch.ops.aten.view.default(view_569, [8, 49, 1024]);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_571: "f32[392, 1024]" = torch.ops.aten.view.default(view_570, [392, 1024]);  view_570 = None
    mm_9: "f32[392, 1024]" = torch.ops.aten.mm.default(view_571, t_112);  t_112 = None
    t_113: "f32[1024, 392]" = torch.ops.aten.t.default(view_571)
    mm_10: "f32[1024, 1024]" = torch.ops.aten.mm.default(t_113, view_548);  t_113 = view_548 = None
    t_114: "f32[1024, 1024]" = torch.ops.aten.t.default(mm_10);  mm_10 = None
    sum_4: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_571, [0], True);  view_571 = None
    view_572: "f32[1024]" = torch.ops.aten.view.default(sum_4, [1024]);  sum_4 = None
    t_115: "f32[1024, 1024]" = torch.ops.aten.t.default(t_114);  t_114 = None
    view_573: "f32[8, 49, 1024]" = torch.ops.aten.view.default(mm_9, [8, 49, 1024]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_574: "f32[8, 49, 32, 32]" = torch.ops.aten.view.default(view_573, [8, 49, 32, 32]);  view_573 = None
    transpose_48: "f32[8, 32, 49, 32]" = torch.ops.aten.transpose.int(view_574, 1, 2);  view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_264: "f32[8, 32, 49, 32]" = torch.ops.aten.clone.default(transpose_48, memory_format = torch.contiguous_format);  transpose_48 = None
    _unsafe_view_99: "f32[256, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_264, [256, 49, 32]);  clone_264 = None
    bmm_48: "f32[256, 49, 32]" = torch.ops.aten.bmm.default(transpose_49, _unsafe_view_99);  transpose_49 = None
    bmm_49: "f32[256, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_99, transpose_50);  _unsafe_view_99 = transpose_50 = None
    view_575: "f32[8, 32, 49, 32]" = torch.ops.aten.view.default(bmm_48, [8, 32, 49, 32]);  bmm_48 = None
    view_576: "f32[8, 32, 49, 49]" = torch.ops.aten.view.default(bmm_49, [8, 32, 49, 49]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data: "f32[8, 32, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_576, detach_24, -1, torch.float32);  view_576 = detach_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_5: "f32[1, 32, 49, 49]" = torch.ops.aten.sum.dim_IntList(_softmax_backward_data, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze: "f32[32, 49, 49]" = torch.ops.aten.squeeze.dim(sum_5, 0);  sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_101: "f32[49, 49, 32]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_577: "f32[2401, 32]" = torch.ops.aten.view.default(permute_101, [2401, 32]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros: "f32[169, 32]" = torch.ops.aten.new_zeros.default(view_577, [169, 32], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put: "f32[169, 32]" = torch.ops.aten.index_put.default(new_zeros, [view_544], view_577, True);  new_zeros = view_544 = view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_578: "f32[256, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data, [256, 49, 49]);  _softmax_backward_data = None
    bmm_50: "f32[256, 32, 49]" = torch.ops.aten.bmm.default(transpose_51, view_578);  transpose_51 = None
    bmm_51: "f32[256, 49, 32]" = torch.ops.aten.bmm.default(view_578, transpose_52);  view_578 = transpose_52 = None
    view_579: "f32[8, 32, 32, 49]" = torch.ops.aten.view.default(bmm_50, [8, 32, 32, 49]);  bmm_50 = None
    view_580: "f32[8, 32, 49, 32]" = torch.ops.aten.view.default(bmm_51, [8, 32, 49, 32]);  bmm_51 = None
    transpose_53: "f32[8, 32, 49, 32]" = torch.ops.aten.transpose.int(view_579, -2, -1);  view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_72: "f32[8, 32, 49, 32]" = torch.ops.aten.mul.Tensor(view_580, 0.1767766952966369);  view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack: "f32[3, 8, 32, 49, 32]" = torch.ops.aten.stack.default([mul_72, transpose_53, view_575]);  mul_72 = transpose_53 = view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_102: "f32[8, 49, 3, 32, 32]" = torch.ops.aten.permute.default(stack, [1, 3, 0, 2, 4]);  stack = None
    clone_265: "f32[8, 49, 3, 32, 32]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    _unsafe_view_100: "f32[8, 49, 3072]" = torch.ops.aten._unsafe_view.default(clone_265, [8, 49, 3072]);  clone_265 = None
    view_581: "f32[392, 3072]" = torch.ops.aten.view.default(_unsafe_view_100, [392, 3072]);  _unsafe_view_100 = None
    mm_11: "f32[392, 1024]" = torch.ops.aten.mm.default(view_581, t_116);  t_116 = None
    t_117: "f32[3072, 392]" = torch.ops.aten.t.default(view_581)
    mm_12: "f32[3072, 1024]" = torch.ops.aten.mm.default(t_117, view_540);  t_117 = view_540 = None
    t_118: "f32[1024, 3072]" = torch.ops.aten.t.default(mm_12);  mm_12 = None
    sum_6: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_581, [0], True);  view_581 = None
    view_582: "f32[3072]" = torch.ops.aten.view.default(sum_6, [3072]);  sum_6 = None
    t_119: "f32[3072, 1024]" = torch.ops.aten.t.default(t_118);  t_118 = None
    view_583: "f32[8, 49, 1024]" = torch.ops.aten.view.default(mm_11, [8, 49, 1024]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_584: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(view_583, [8, 7, 7, 1024]);  view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_585: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_584, [8, 1, 1, 7, 7, 1024]);  view_584 = None
    permute_103: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_585, [0, 1, 3, 2, 4, 5]);  view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_586: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(permute_103, [8, 7, 7, 1024]);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_24: "f32[8, 7, 7, 1024]" = torch.ops.aten.constant_pad_nd.default(view_586, [0, 0, 0, 0, 0, 0]);  view_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_2 = torch.ops.aten.native_layer_norm_backward.default(constant_pad_nd_24, view_536, [1024], getitem_220, getitem_221, primals_314, primals_315, [True, True, True]);  constant_pad_nd_24 = view_536 = getitem_220 = getitem_221 = primals_314 = primals_315 = None
    getitem_237: "f32[8, 7, 7, 1024]" = native_layer_norm_backward_2[0]
    getitem_238: "f32[1024]" = native_layer_norm_backward_2[1]
    getitem_239: "f32[1024]" = native_layer_norm_backward_2[2];  native_layer_norm_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_84: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_567, getitem_237);  view_567 = getitem_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_587: "f32[8, 49, 1024]" = torch.ops.aten.view.default(add_84, [8, 49, 1024]);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_73: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(view_587, div_43);  div_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_588: "f32[392, 1024]" = torch.ops.aten.view.default(mul_73, [392, 1024]);  mul_73 = None
    mm_13: "f32[392, 4096]" = torch.ops.aten.mm.default(view_588, t_120);  t_120 = None
    t_121: "f32[1024, 392]" = torch.ops.aten.t.default(view_588)
    mm_14: "f32[1024, 4096]" = torch.ops.aten.mm.default(t_121, view_534);  t_121 = view_534 = None
    t_122: "f32[4096, 1024]" = torch.ops.aten.t.default(mm_14);  mm_14 = None
    sum_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_588, [0], True);  view_588 = None
    view_589: "f32[1024]" = torch.ops.aten.view.default(sum_7, [1024]);  sum_7 = None
    t_123: "f32[1024, 4096]" = torch.ops.aten.t.default(t_122);  t_122 = None
    view_590: "f32[8, 49, 4096]" = torch.ops.aten.view.default(mm_13, [8, 49, 4096]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_1: "f32[8, 49, 4096]" = torch.ops.aten.gelu_backward.default(view_590, view_533);  view_590 = view_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_591: "f32[392, 4096]" = torch.ops.aten.view.default(gelu_backward_1, [392, 4096]);  gelu_backward_1 = None
    mm_15: "f32[392, 1024]" = torch.ops.aten.mm.default(view_591, t_124);  t_124 = None
    t_125: "f32[4096, 392]" = torch.ops.aten.t.default(view_591)
    mm_16: "f32[4096, 1024]" = torch.ops.aten.mm.default(t_125, view_532);  t_125 = view_532 = None
    t_126: "f32[1024, 4096]" = torch.ops.aten.t.default(mm_16);  mm_16 = None
    sum_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
    view_592: "f32[4096]" = torch.ops.aten.view.default(sum_8, [4096]);  sum_8 = None
    t_127: "f32[4096, 1024]" = torch.ops.aten.t.default(t_126);  t_126 = None
    view_593: "f32[8, 49, 1024]" = torch.ops.aten.view.default(mm_15, [8, 49, 1024]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_3 = torch.ops.aten.native_layer_norm_backward.default(view_593, view_531, [1024], getitem_217, getitem_218, primals_308, primals_309, [True, True, True]);  view_593 = view_531 = getitem_217 = getitem_218 = primals_308 = primals_309 = None
    getitem_240: "f32[8, 49, 1024]" = native_layer_norm_backward_3[0]
    getitem_241: "f32[1024]" = native_layer_norm_backward_3[1]
    getitem_242: "f32[1024]" = native_layer_norm_backward_3[2];  native_layer_norm_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_85: "f32[8, 49, 1024]" = torch.ops.aten.add.Tensor(view_587, getitem_240);  view_587 = getitem_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_594: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(add_85, [8, 7, 7, 1024]);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_74: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_594, div_42);  div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_2: "f32[8, 7, 7, 1024]" = torch.ops.aten.slice_backward.default(mul_74, [8, 7, 7, 1024], 3, 0, 9223372036854775807, 1);  mul_74 = None
    slice_backward_3: "f32[8, 7, 7, 1024]" = torch.ops.aten.slice_backward.default(slice_backward_2, [8, 7, 7, 1024], 0, 0, 9223372036854775807, 1);  slice_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_595: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.view.default(slice_backward_3, [8, 1, 7, 1, 7, 1024]);  slice_backward_3 = None
    permute_104: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_595, [0, 1, 3, 2, 4, 5]);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_596: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(permute_104, [8, 7, 7, 1024]);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_597: "f32[8, 49, 1024]" = torch.ops.aten.view.default(view_596, [8, 49, 1024]);  view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_598: "f32[392, 1024]" = torch.ops.aten.view.default(view_597, [392, 1024]);  view_597 = None
    mm_17: "f32[392, 1024]" = torch.ops.aten.mm.default(view_598, t_128);  t_128 = None
    t_129: "f32[1024, 392]" = torch.ops.aten.t.default(view_598)
    mm_18: "f32[1024, 1024]" = torch.ops.aten.mm.default(t_129, view_526);  t_129 = view_526 = None
    t_130: "f32[1024, 1024]" = torch.ops.aten.t.default(mm_18);  mm_18 = None
    sum_9: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_598, [0], True);  view_598 = None
    view_599: "f32[1024]" = torch.ops.aten.view.default(sum_9, [1024]);  sum_9 = None
    t_131: "f32[1024, 1024]" = torch.ops.aten.t.default(t_130);  t_130 = None
    view_600: "f32[8, 49, 1024]" = torch.ops.aten.view.default(mm_17, [8, 49, 1024]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_601: "f32[8, 49, 32, 32]" = torch.ops.aten.view.default(view_600, [8, 49, 32, 32]);  view_600 = None
    transpose_54: "f32[8, 32, 49, 32]" = torch.ops.aten.transpose.int(view_601, 1, 2);  view_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_266: "f32[8, 32, 49, 32]" = torch.ops.aten.clone.default(transpose_54, memory_format = torch.contiguous_format);  transpose_54 = None
    _unsafe_view_101: "f32[256, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_266, [256, 49, 32]);  clone_266 = None
    bmm_52: "f32[256, 49, 32]" = torch.ops.aten.bmm.default(transpose_55, _unsafe_view_101);  transpose_55 = None
    bmm_53: "f32[256, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_101, transpose_56);  _unsafe_view_101 = transpose_56 = None
    view_602: "f32[8, 32, 49, 32]" = torch.ops.aten.view.default(bmm_52, [8, 32, 49, 32]);  bmm_52 = None
    view_603: "f32[8, 32, 49, 49]" = torch.ops.aten.view.default(bmm_53, [8, 32, 49, 49]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_1: "f32[8, 32, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_603, detach_25, -1, torch.float32);  view_603 = detach_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_10: "f32[1, 32, 49, 49]" = torch.ops.aten.sum.dim_IntList(_softmax_backward_data_1, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_1: "f32[32, 49, 49]" = torch.ops.aten.squeeze.dim(sum_10, 0);  sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_105: "f32[49, 49, 32]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_604: "f32[2401, 32]" = torch.ops.aten.view.default(permute_105, [2401, 32]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_1: "f32[169, 32]" = torch.ops.aten.new_zeros.default(view_604, [169, 32], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_1: "f32[169, 32]" = torch.ops.aten.index_put.default(new_zeros_1, [view_522], view_604, True);  new_zeros_1 = view_522 = view_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_605: "f32[256, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_1, [256, 49, 49]);  _softmax_backward_data_1 = None
    bmm_54: "f32[256, 32, 49]" = torch.ops.aten.bmm.default(transpose_57, view_605);  transpose_57 = None
    bmm_55: "f32[256, 49, 32]" = torch.ops.aten.bmm.default(view_605, transpose_58);  view_605 = transpose_58 = None
    view_606: "f32[8, 32, 32, 49]" = torch.ops.aten.view.default(bmm_54, [8, 32, 32, 49]);  bmm_54 = None
    view_607: "f32[8, 32, 49, 32]" = torch.ops.aten.view.default(bmm_55, [8, 32, 49, 32]);  bmm_55 = None
    transpose_59: "f32[8, 32, 49, 32]" = torch.ops.aten.transpose.int(view_606, -2, -1);  view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_75: "f32[8, 32, 49, 32]" = torch.ops.aten.mul.Tensor(view_607, 0.1767766952966369);  view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_1: "f32[3, 8, 32, 49, 32]" = torch.ops.aten.stack.default([mul_75, transpose_59, view_602]);  mul_75 = transpose_59 = view_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_106: "f32[8, 49, 3, 32, 32]" = torch.ops.aten.permute.default(stack_1, [1, 3, 0, 2, 4]);  stack_1 = None
    clone_267: "f32[8, 49, 3, 32, 32]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    _unsafe_view_102: "f32[8, 49, 3072]" = torch.ops.aten._unsafe_view.default(clone_267, [8, 49, 3072]);  clone_267 = None
    view_608: "f32[392, 3072]" = torch.ops.aten.view.default(_unsafe_view_102, [392, 3072]);  _unsafe_view_102 = None
    mm_19: "f32[392, 1024]" = torch.ops.aten.mm.default(view_608, t_132);  t_132 = None
    t_133: "f32[3072, 392]" = torch.ops.aten.t.default(view_608)
    mm_20: "f32[3072, 1024]" = torch.ops.aten.mm.default(t_133, view_518);  t_133 = view_518 = None
    t_134: "f32[1024, 3072]" = torch.ops.aten.t.default(mm_20);  mm_20 = None
    sum_11: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_608, [0], True);  view_608 = None
    view_609: "f32[3072]" = torch.ops.aten.view.default(sum_11, [3072]);  sum_11 = None
    t_135: "f32[3072, 1024]" = torch.ops.aten.t.default(t_134);  t_134 = None
    view_610: "f32[8, 49, 1024]" = torch.ops.aten.view.default(mm_19, [8, 49, 1024]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_611: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(view_610, [8, 7, 7, 1024]);  view_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_612: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_611, [8, 1, 1, 7, 7, 1024]);  view_611 = None
    permute_107: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_612, [0, 1, 3, 2, 4, 5]);  view_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_613: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(permute_107, [8, 7, 7, 1024]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_25: "f32[8, 7, 7, 1024]" = torch.ops.aten.constant_pad_nd.default(view_613, [0, 0, 0, 0, 0, 0]);  view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_4 = torch.ops.aten.native_layer_norm_backward.default(constant_pad_nd_25, view_514, [1024], getitem_211, getitem_212, primals_302, primals_303, [True, True, True]);  constant_pad_nd_25 = view_514 = getitem_211 = getitem_212 = primals_302 = primals_303 = None
    getitem_243: "f32[8, 7, 7, 1024]" = native_layer_norm_backward_4[0]
    getitem_244: "f32[1024]" = native_layer_norm_backward_4[1]
    getitem_245: "f32[1024]" = native_layer_norm_backward_4[2];  native_layer_norm_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_86: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_594, getitem_243);  view_594 = getitem_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    view_614: "f32[392, 1024]" = torch.ops.aten.view.default(add_86, [392, 1024]);  add_86 = None
    t_136: "f32[1024, 392]" = torch.ops.aten.t.default(view_614)
    mm_21: "f32[1024, 2048]" = torch.ops.aten.mm.default(t_136, view_513);  t_136 = view_513 = None
    t_137: "f32[2048, 1024]" = torch.ops.aten.t.default(mm_21);  mm_21 = None
    mm_22: "f32[392, 2048]" = torch.ops.aten.mm.default(view_614, t_138);  view_614 = t_138 = None
    view_615: "f32[8, 7, 7, 2048]" = torch.ops.aten.view.default(mm_22, [8, 7, 7, 2048]);  mm_22 = None
    t_139: "f32[1024, 2048]" = torch.ops.aten.t.default(t_137);  t_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:357, code: x = self.norm(x)
    native_layer_norm_backward_5 = torch.ops.aten.native_layer_norm_backward.default(view_615, _unsafe_view_90, [2048], getitem_208, getitem_209, primals_299, primals_300, [True, True, True]);  view_615 = _unsafe_view_90 = getitem_208 = getitem_209 = primals_299 = primals_300 = None
    getitem_246: "f32[8, 7, 7, 2048]" = native_layer_norm_backward_5[0]
    getitem_247: "f32[2048]" = native_layer_norm_backward_5[1]
    getitem_248: "f32[2048]" = native_layer_norm_backward_5[2];  native_layer_norm_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:356, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    view_616: "f32[8, 7, 7, 2, 2, 512]" = torch.ops.aten.view.default(getitem_246, [8, 7, 7, 2, 2, 512]);  getitem_246 = None
    permute_108: "f32[8, 7, 2, 7, 2, 512]" = torch.ops.aten.permute.default(view_616, [0, 1, 4, 2, 3, 5]);  view_616 = None
    clone_268: "f32[8, 7, 2, 7, 2, 512]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
    _unsafe_view_103: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_268, [8, 14, 14, 512]);  clone_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_617: "f32[8, 196, 512]" = torch.ops.aten.view.default(_unsafe_view_103, [8, 196, 512]);  _unsafe_view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_76: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_617, div_41);  div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_618: "f32[1568, 512]" = torch.ops.aten.view.default(mul_76, [1568, 512]);  mul_76 = None
    mm_23: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_618, t_140);  t_140 = None
    t_141: "f32[512, 1568]" = torch.ops.aten.t.default(view_618)
    mm_24: "f32[512, 2048]" = torch.ops.aten.mm.default(t_141, view_509);  t_141 = view_509 = None
    t_142: "f32[2048, 512]" = torch.ops.aten.t.default(mm_24);  mm_24 = None
    sum_12: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_618, [0], True);  view_618 = None
    view_619: "f32[512]" = torch.ops.aten.view.default(sum_12, [512]);  sum_12 = None
    t_143: "f32[512, 2048]" = torch.ops.aten.t.default(t_142);  t_142 = None
    view_620: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_23, [8, 196, 2048]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_2: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_620, view_508);  view_620 = view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_621: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_2, [1568, 2048]);  gelu_backward_2 = None
    mm_25: "f32[1568, 512]" = torch.ops.aten.mm.default(view_621, t_144);  t_144 = None
    t_145: "f32[2048, 1568]" = torch.ops.aten.t.default(view_621)
    mm_26: "f32[2048, 512]" = torch.ops.aten.mm.default(t_145, view_507);  t_145 = view_507 = None
    t_146: "f32[512, 2048]" = torch.ops.aten.t.default(mm_26);  mm_26 = None
    sum_13: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_621, [0], True);  view_621 = None
    view_622: "f32[2048]" = torch.ops.aten.view.default(sum_13, [2048]);  sum_13 = None
    t_147: "f32[2048, 512]" = torch.ops.aten.t.default(t_146);  t_146 = None
    view_623: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_25, [8, 196, 512]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_6 = torch.ops.aten.native_layer_norm_backward.default(view_623, view_506, [512], getitem_205, getitem_206, primals_293, primals_294, [True, True, True]);  view_623 = view_506 = getitem_205 = getitem_206 = primals_293 = primals_294 = None
    getitem_249: "f32[8, 196, 512]" = native_layer_norm_backward_6[0]
    getitem_250: "f32[512]" = native_layer_norm_backward_6[1]
    getitem_251: "f32[512]" = native_layer_norm_backward_6[2];  native_layer_norm_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_87: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_617, getitem_249);  view_617 = getitem_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_624: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_87, [8, 14, 14, 512]);  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_77: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_624, div_40);  div_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_22: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_77, [-3, -3], [2, 1]);  mul_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_4: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(roll_22, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  roll_22 = None
    slice_backward_5: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_4, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_625: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_5, [8, 2, 7, 2, 7, 512]);  slice_backward_5 = None
    permute_109: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_625, [0, 1, 3, 2, 4, 5]);  view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_269: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
    _unsafe_view_104: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_269, [32, 7, 7, 512]);  clone_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_626: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_104, [32, 49, 512]);  _unsafe_view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_627: "f32[1568, 512]" = torch.ops.aten.view.default(view_626, [1568, 512]);  view_626 = None
    mm_27: "f32[1568, 512]" = torch.ops.aten.mm.default(view_627, t_148);  t_148 = None
    t_149: "f32[512, 1568]" = torch.ops.aten.t.default(view_627)
    mm_28: "f32[512, 512]" = torch.ops.aten.mm.default(t_149, view_501);  t_149 = view_501 = None
    t_150: "f32[512, 512]" = torch.ops.aten.t.default(mm_28);  mm_28 = None
    sum_14: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_627, [0], True);  view_627 = None
    view_628: "f32[512]" = torch.ops.aten.view.default(sum_14, [512]);  sum_14 = None
    t_151: "f32[512, 512]" = torch.ops.aten.t.default(t_150);  t_150 = None
    view_629: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_27, [32, 49, 512]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_630: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_629, [32, 49, 16, 32]);  view_629 = None
    transpose_60: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_630, 1, 2);  view_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_270: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_60, memory_format = torch.contiguous_format);  transpose_60 = None
    _unsafe_view_105: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_270, [512, 49, 32]);  clone_270 = None
    bmm_56: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_61, _unsafe_view_105);  transpose_61 = None
    bmm_57: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_105, transpose_62);  _unsafe_view_105 = transpose_62 = None
    view_631: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_56, [32, 16, 49, 32]);  bmm_56 = None
    view_632: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_57, [32, 16, 49, 49]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_2: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_632, detach_26, -1, torch.float32);  view_632 = detach_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_633: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_2, [8, 4, 16, 49, 49]);  _softmax_backward_data_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_634: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(view_633, [32, 16, 49, 49]);  view_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_15: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_634, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_2: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_15, 0);  sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_110: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_635: "f32[2401, 16]" = torch.ops.aten.view.default(permute_110, [2401, 16]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_2: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_635, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_2: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_2, [view_495], view_635, True);  new_zeros_2 = view_495 = view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_636: "f32[512, 49, 49]" = torch.ops.aten.view.default(view_634, [512, 49, 49]);  view_634 = None
    bmm_58: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_63, view_636);  transpose_63 = None
    bmm_59: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_636, transpose_64);  view_636 = transpose_64 = None
    view_637: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_58, [32, 16, 32, 49]);  bmm_58 = None
    view_638: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_59, [32, 16, 49, 32]);  bmm_59 = None
    transpose_65: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_637, -2, -1);  view_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_78: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_638, 0.1767766952966369);  view_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_2: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_78, transpose_65, view_631]);  mul_78 = transpose_65 = view_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_111: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_2, [1, 3, 0, 2, 4]);  stack_2 = None
    clone_271: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    _unsafe_view_106: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_271, [32, 49, 1536]);  clone_271 = None
    view_639: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_106, [1568, 1536]);  _unsafe_view_106 = None
    mm_29: "f32[1568, 512]" = torch.ops.aten.mm.default(view_639, t_152);  t_152 = None
    t_153: "f32[1536, 1568]" = torch.ops.aten.t.default(view_639)
    mm_30: "f32[1536, 512]" = torch.ops.aten.mm.default(t_153, view_491);  t_153 = view_491 = None
    t_154: "f32[512, 1536]" = torch.ops.aten.t.default(mm_30);  mm_30 = None
    sum_16: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_639, [0], True);  view_639 = None
    view_640: "f32[1536]" = torch.ops.aten.view.default(sum_16, [1536]);  sum_16 = None
    t_155: "f32[1536, 512]" = torch.ops.aten.t.default(t_154);  t_154 = None
    view_641: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_29, [32, 49, 512]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_642: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_641, [32, 7, 7, 512]);  view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_643: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_642, [8, 2, 2, 7, 7, 512]);  view_642 = None
    permute_112: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_643, [0, 1, 3, 2, 4, 5]);  view_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_272: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    _unsafe_view_107: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_272, [8, 14, 14, 512]);  clone_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_26: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_107, [0, 0, 0, 0, 0, 0]);  _unsafe_view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_23: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(constant_pad_nd_26, [3, 3], [2, 1]);  constant_pad_nd_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_7 = torch.ops.aten.native_layer_norm_backward.default(roll_23, view_487, [512], getitem_199, getitem_200, primals_287, primals_288, [True, True, True]);  roll_23 = view_487 = getitem_199 = getitem_200 = primals_287 = primals_288 = None
    getitem_252: "f32[8, 14, 14, 512]" = native_layer_norm_backward_7[0]
    getitem_253: "f32[512]" = native_layer_norm_backward_7[1]
    getitem_254: "f32[512]" = native_layer_norm_backward_7[2];  native_layer_norm_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_88: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_624, getitem_252);  view_624 = getitem_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_644: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_88, [8, 196, 512]);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_79: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_644, div_39);  div_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_645: "f32[1568, 512]" = torch.ops.aten.view.default(mul_79, [1568, 512]);  mul_79 = None
    mm_31: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_645, t_156);  t_156 = None
    t_157: "f32[512, 1568]" = torch.ops.aten.t.default(view_645)
    mm_32: "f32[512, 2048]" = torch.ops.aten.mm.default(t_157, view_485);  t_157 = view_485 = None
    t_158: "f32[2048, 512]" = torch.ops.aten.t.default(mm_32);  mm_32 = None
    sum_17: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_645, [0], True);  view_645 = None
    view_646: "f32[512]" = torch.ops.aten.view.default(sum_17, [512]);  sum_17 = None
    t_159: "f32[512, 2048]" = torch.ops.aten.t.default(t_158);  t_158 = None
    view_647: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_31, [8, 196, 2048]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_3: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_647, view_484);  view_647 = view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_648: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_3, [1568, 2048]);  gelu_backward_3 = None
    mm_33: "f32[1568, 512]" = torch.ops.aten.mm.default(view_648, t_160);  t_160 = None
    t_161: "f32[2048, 1568]" = torch.ops.aten.t.default(view_648)
    mm_34: "f32[2048, 512]" = torch.ops.aten.mm.default(t_161, view_483);  t_161 = view_483 = None
    t_162: "f32[512, 2048]" = torch.ops.aten.t.default(mm_34);  mm_34 = None
    sum_18: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_648, [0], True);  view_648 = None
    view_649: "f32[2048]" = torch.ops.aten.view.default(sum_18, [2048]);  sum_18 = None
    t_163: "f32[2048, 512]" = torch.ops.aten.t.default(t_162);  t_162 = None
    view_650: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_33, [8, 196, 512]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_8 = torch.ops.aten.native_layer_norm_backward.default(view_650, view_482, [512], getitem_196, getitem_197, primals_281, primals_282, [True, True, True]);  view_650 = view_482 = getitem_196 = getitem_197 = primals_281 = primals_282 = None
    getitem_255: "f32[8, 196, 512]" = native_layer_norm_backward_8[0]
    getitem_256: "f32[512]" = native_layer_norm_backward_8[1]
    getitem_257: "f32[512]" = native_layer_norm_backward_8[2];  native_layer_norm_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_89: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_644, getitem_255);  view_644 = getitem_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_651: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_89, [8, 14, 14, 512]);  add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_80: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_651, div_38);  div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_6: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(mul_80, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  mul_80 = None
    slice_backward_7: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_6, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_652: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_7, [8, 2, 7, 2, 7, 512]);  slice_backward_7 = None
    permute_113: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_652, [0, 1, 3, 2, 4, 5]);  view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_273: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    _unsafe_view_108: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_273, [32, 7, 7, 512]);  clone_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_653: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_108, [32, 49, 512]);  _unsafe_view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_654: "f32[1568, 512]" = torch.ops.aten.view.default(view_653, [1568, 512]);  view_653 = None
    mm_35: "f32[1568, 512]" = torch.ops.aten.mm.default(view_654, t_164);  t_164 = None
    t_165: "f32[512, 1568]" = torch.ops.aten.t.default(view_654)
    mm_36: "f32[512, 512]" = torch.ops.aten.mm.default(t_165, view_477);  t_165 = view_477 = None
    t_166: "f32[512, 512]" = torch.ops.aten.t.default(mm_36);  mm_36 = None
    sum_19: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_654, [0], True);  view_654 = None
    view_655: "f32[512]" = torch.ops.aten.view.default(sum_19, [512]);  sum_19 = None
    t_167: "f32[512, 512]" = torch.ops.aten.t.default(t_166);  t_166 = None
    view_656: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_35, [32, 49, 512]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_657: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_656, [32, 49, 16, 32]);  view_656 = None
    transpose_66: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_657, 1, 2);  view_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_274: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_66, memory_format = torch.contiguous_format);  transpose_66 = None
    _unsafe_view_109: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_274, [512, 49, 32]);  clone_274 = None
    bmm_60: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_67, _unsafe_view_109);  transpose_67 = None
    bmm_61: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_109, transpose_68);  _unsafe_view_109 = transpose_68 = None
    view_658: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_60, [32, 16, 49, 32]);  bmm_60 = None
    view_659: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_61, [32, 16, 49, 49]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_3: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_659, detach_27, -1, torch.float32);  view_659 = detach_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_20: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(_softmax_backward_data_3, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_3: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_20, 0);  sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_114: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_3, [1, 2, 0]);  squeeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_660: "f32[2401, 16]" = torch.ops.aten.view.default(permute_114, [2401, 16]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_3: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_660, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_3: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_3, [view_473], view_660, True);  new_zeros_3 = view_473 = view_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_661: "f32[512, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_3, [512, 49, 49]);  _softmax_backward_data_3 = None
    bmm_62: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_69, view_661);  transpose_69 = None
    bmm_63: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_661, transpose_70);  view_661 = transpose_70 = None
    view_662: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_62, [32, 16, 32, 49]);  bmm_62 = None
    view_663: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_63, [32, 16, 49, 32]);  bmm_63 = None
    transpose_71: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_662, -2, -1);  view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_81: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_663, 0.1767766952966369);  view_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_3: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_81, transpose_71, view_658]);  mul_81 = transpose_71 = view_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_115: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_3, [1, 3, 0, 2, 4]);  stack_3 = None
    clone_275: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    _unsafe_view_110: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_275, [32, 49, 1536]);  clone_275 = None
    view_664: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_110, [1568, 1536]);  _unsafe_view_110 = None
    mm_37: "f32[1568, 512]" = torch.ops.aten.mm.default(view_664, t_168);  t_168 = None
    t_169: "f32[1536, 1568]" = torch.ops.aten.t.default(view_664)
    mm_38: "f32[1536, 512]" = torch.ops.aten.mm.default(t_169, view_469);  t_169 = view_469 = None
    t_170: "f32[512, 1536]" = torch.ops.aten.t.default(mm_38);  mm_38 = None
    sum_21: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_664, [0], True);  view_664 = None
    view_665: "f32[1536]" = torch.ops.aten.view.default(sum_21, [1536]);  sum_21 = None
    t_171: "f32[1536, 512]" = torch.ops.aten.t.default(t_170);  t_170 = None
    view_666: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_37, [32, 49, 512]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_667: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_666, [32, 7, 7, 512]);  view_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_668: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_667, [8, 2, 2, 7, 7, 512]);  view_667 = None
    permute_116: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_668, [0, 1, 3, 2, 4, 5]);  view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_276: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
    _unsafe_view_111: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_276, [8, 14, 14, 512]);  clone_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_27: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_111, [0, 0, 0, 0, 0, 0]);  _unsafe_view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_9 = torch.ops.aten.native_layer_norm_backward.default(constant_pad_nd_27, view_465, [512], getitem_190, getitem_191, primals_275, primals_276, [True, True, True]);  constant_pad_nd_27 = view_465 = getitem_190 = getitem_191 = primals_275 = primals_276 = None
    getitem_258: "f32[8, 14, 14, 512]" = native_layer_norm_backward_9[0]
    getitem_259: "f32[512]" = native_layer_norm_backward_9[1]
    getitem_260: "f32[512]" = native_layer_norm_backward_9[2];  native_layer_norm_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_90: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_651, getitem_258);  view_651 = getitem_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_669: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_90, [8, 196, 512]);  add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_82: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_669, div_37);  div_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_670: "f32[1568, 512]" = torch.ops.aten.view.default(mul_82, [1568, 512]);  mul_82 = None
    mm_39: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_670, t_172);  t_172 = None
    t_173: "f32[512, 1568]" = torch.ops.aten.t.default(view_670)
    mm_40: "f32[512, 2048]" = torch.ops.aten.mm.default(t_173, view_463);  t_173 = view_463 = None
    t_174: "f32[2048, 512]" = torch.ops.aten.t.default(mm_40);  mm_40 = None
    sum_22: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True);  view_670 = None
    view_671: "f32[512]" = torch.ops.aten.view.default(sum_22, [512]);  sum_22 = None
    t_175: "f32[512, 2048]" = torch.ops.aten.t.default(t_174);  t_174 = None
    view_672: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_39, [8, 196, 2048]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_4: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_672, view_462);  view_672 = view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_673: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_4, [1568, 2048]);  gelu_backward_4 = None
    mm_41: "f32[1568, 512]" = torch.ops.aten.mm.default(view_673, t_176);  t_176 = None
    t_177: "f32[2048, 1568]" = torch.ops.aten.t.default(view_673)
    mm_42: "f32[2048, 512]" = torch.ops.aten.mm.default(t_177, view_461);  t_177 = view_461 = None
    t_178: "f32[512, 2048]" = torch.ops.aten.t.default(mm_42);  mm_42 = None
    sum_23: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_673, [0], True);  view_673 = None
    view_674: "f32[2048]" = torch.ops.aten.view.default(sum_23, [2048]);  sum_23 = None
    t_179: "f32[2048, 512]" = torch.ops.aten.t.default(t_178);  t_178 = None
    view_675: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_41, [8, 196, 512]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_10 = torch.ops.aten.native_layer_norm_backward.default(view_675, view_460, [512], getitem_187, getitem_188, primals_269, primals_270, [True, True, True]);  view_675 = view_460 = getitem_187 = getitem_188 = primals_269 = primals_270 = None
    getitem_261: "f32[8, 196, 512]" = native_layer_norm_backward_10[0]
    getitem_262: "f32[512]" = native_layer_norm_backward_10[1]
    getitem_263: "f32[512]" = native_layer_norm_backward_10[2];  native_layer_norm_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_91: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_669, getitem_261);  view_669 = getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_676: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_91, [8, 14, 14, 512]);  add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_83: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_676, div_36);  div_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_24: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_83, [-3, -3], [2, 1]);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_8: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(roll_24, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  roll_24 = None
    slice_backward_9: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_8, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_677: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_9, [8, 2, 7, 2, 7, 512]);  slice_backward_9 = None
    permute_117: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_677, [0, 1, 3, 2, 4, 5]);  view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_277: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    _unsafe_view_112: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_277, [32, 7, 7, 512]);  clone_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_678: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_112, [32, 49, 512]);  _unsafe_view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_679: "f32[1568, 512]" = torch.ops.aten.view.default(view_678, [1568, 512]);  view_678 = None
    mm_43: "f32[1568, 512]" = torch.ops.aten.mm.default(view_679, t_180);  t_180 = None
    t_181: "f32[512, 1568]" = torch.ops.aten.t.default(view_679)
    mm_44: "f32[512, 512]" = torch.ops.aten.mm.default(t_181, view_455);  t_181 = view_455 = None
    t_182: "f32[512, 512]" = torch.ops.aten.t.default(mm_44);  mm_44 = None
    sum_24: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_679, [0], True);  view_679 = None
    view_680: "f32[512]" = torch.ops.aten.view.default(sum_24, [512]);  sum_24 = None
    t_183: "f32[512, 512]" = torch.ops.aten.t.default(t_182);  t_182 = None
    view_681: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_43, [32, 49, 512]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_682: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_681, [32, 49, 16, 32]);  view_681 = None
    transpose_72: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_682, 1, 2);  view_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_278: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_72, memory_format = torch.contiguous_format);  transpose_72 = None
    _unsafe_view_113: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_278, [512, 49, 32]);  clone_278 = None
    bmm_64: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_73, _unsafe_view_113);  transpose_73 = None
    bmm_65: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_113, transpose_74);  _unsafe_view_113 = transpose_74 = None
    view_683: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_64, [32, 16, 49, 32]);  bmm_64 = None
    view_684: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_65, [32, 16, 49, 49]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_4: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_684, detach_28, -1, torch.float32);  view_684 = detach_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_685: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_4, [8, 4, 16, 49, 49]);  _softmax_backward_data_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_686: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(view_685, [32, 16, 49, 49]);  view_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_25: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_686, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_4: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_25, 0);  sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_118: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_4, [1, 2, 0]);  squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_687: "f32[2401, 16]" = torch.ops.aten.view.default(permute_118, [2401, 16]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_4: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_687, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_4: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_4, [view_449], view_687, True);  new_zeros_4 = view_449 = view_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_688: "f32[512, 49, 49]" = torch.ops.aten.view.default(view_686, [512, 49, 49]);  view_686 = None
    bmm_66: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_75, view_688);  transpose_75 = None
    bmm_67: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_688, transpose_76);  view_688 = transpose_76 = None
    view_689: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_66, [32, 16, 32, 49]);  bmm_66 = None
    view_690: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_67, [32, 16, 49, 32]);  bmm_67 = None
    transpose_77: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_689, -2, -1);  view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_84: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_690, 0.1767766952966369);  view_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_4: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_84, transpose_77, view_683]);  mul_84 = transpose_77 = view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_119: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_4, [1, 3, 0, 2, 4]);  stack_4 = None
    clone_279: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None
    _unsafe_view_114: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_279, [32, 49, 1536]);  clone_279 = None
    view_691: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_114, [1568, 1536]);  _unsafe_view_114 = None
    mm_45: "f32[1568, 512]" = torch.ops.aten.mm.default(view_691, t_184);  t_184 = None
    t_185: "f32[1536, 1568]" = torch.ops.aten.t.default(view_691)
    mm_46: "f32[1536, 512]" = torch.ops.aten.mm.default(t_185, view_445);  t_185 = view_445 = None
    t_186: "f32[512, 1536]" = torch.ops.aten.t.default(mm_46);  mm_46 = None
    sum_26: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_691, [0], True);  view_691 = None
    view_692: "f32[1536]" = torch.ops.aten.view.default(sum_26, [1536]);  sum_26 = None
    t_187: "f32[1536, 512]" = torch.ops.aten.t.default(t_186);  t_186 = None
    view_693: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_45, [32, 49, 512]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_694: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_693, [32, 7, 7, 512]);  view_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_695: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_694, [8, 2, 2, 7, 7, 512]);  view_694 = None
    permute_120: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_695, [0, 1, 3, 2, 4, 5]);  view_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_280: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
    _unsafe_view_115: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_280, [8, 14, 14, 512]);  clone_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_28: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_115, [0, 0, 0, 0, 0, 0]);  _unsafe_view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_25: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(constant_pad_nd_28, [3, 3], [2, 1]);  constant_pad_nd_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_11 = torch.ops.aten.native_layer_norm_backward.default(roll_25, view_441, [512], getitem_181, getitem_182, primals_263, primals_264, [True, True, True]);  roll_25 = view_441 = getitem_181 = getitem_182 = primals_263 = primals_264 = None
    getitem_264: "f32[8, 14, 14, 512]" = native_layer_norm_backward_11[0]
    getitem_265: "f32[512]" = native_layer_norm_backward_11[1]
    getitem_266: "f32[512]" = native_layer_norm_backward_11[2];  native_layer_norm_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_92: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_676, getitem_264);  view_676 = getitem_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_696: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_92, [8, 196, 512]);  add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_85: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_696, div_35);  div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_697: "f32[1568, 512]" = torch.ops.aten.view.default(mul_85, [1568, 512]);  mul_85 = None
    mm_47: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_697, t_188);  t_188 = None
    t_189: "f32[512, 1568]" = torch.ops.aten.t.default(view_697)
    mm_48: "f32[512, 2048]" = torch.ops.aten.mm.default(t_189, view_439);  t_189 = view_439 = None
    t_190: "f32[2048, 512]" = torch.ops.aten.t.default(mm_48);  mm_48 = None
    sum_27: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_697, [0], True);  view_697 = None
    view_698: "f32[512]" = torch.ops.aten.view.default(sum_27, [512]);  sum_27 = None
    t_191: "f32[512, 2048]" = torch.ops.aten.t.default(t_190);  t_190 = None
    view_699: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_47, [8, 196, 2048]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_5: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_699, view_438);  view_699 = view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_700: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_5, [1568, 2048]);  gelu_backward_5 = None
    mm_49: "f32[1568, 512]" = torch.ops.aten.mm.default(view_700, t_192);  t_192 = None
    t_193: "f32[2048, 1568]" = torch.ops.aten.t.default(view_700)
    mm_50: "f32[2048, 512]" = torch.ops.aten.mm.default(t_193, view_437);  t_193 = view_437 = None
    t_194: "f32[512, 2048]" = torch.ops.aten.t.default(mm_50);  mm_50 = None
    sum_28: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_700, [0], True);  view_700 = None
    view_701: "f32[2048]" = torch.ops.aten.view.default(sum_28, [2048]);  sum_28 = None
    t_195: "f32[2048, 512]" = torch.ops.aten.t.default(t_194);  t_194 = None
    view_702: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_49, [8, 196, 512]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_12 = torch.ops.aten.native_layer_norm_backward.default(view_702, view_436, [512], getitem_178, getitem_179, primals_257, primals_258, [True, True, True]);  view_702 = view_436 = getitem_178 = getitem_179 = primals_257 = primals_258 = None
    getitem_267: "f32[8, 196, 512]" = native_layer_norm_backward_12[0]
    getitem_268: "f32[512]" = native_layer_norm_backward_12[1]
    getitem_269: "f32[512]" = native_layer_norm_backward_12[2];  native_layer_norm_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_93: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_696, getitem_267);  view_696 = getitem_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_703: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_93, [8, 14, 14, 512]);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_86: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_703, div_34);  div_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_10: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(mul_86, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  mul_86 = None
    slice_backward_11: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_10, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_704: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_11, [8, 2, 7, 2, 7, 512]);  slice_backward_11 = None
    permute_121: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_704, [0, 1, 3, 2, 4, 5]);  view_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_281: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    _unsafe_view_116: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_281, [32, 7, 7, 512]);  clone_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_705: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_116, [32, 49, 512]);  _unsafe_view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_706: "f32[1568, 512]" = torch.ops.aten.view.default(view_705, [1568, 512]);  view_705 = None
    mm_51: "f32[1568, 512]" = torch.ops.aten.mm.default(view_706, t_196);  t_196 = None
    t_197: "f32[512, 1568]" = torch.ops.aten.t.default(view_706)
    mm_52: "f32[512, 512]" = torch.ops.aten.mm.default(t_197, view_431);  t_197 = view_431 = None
    t_198: "f32[512, 512]" = torch.ops.aten.t.default(mm_52);  mm_52 = None
    sum_29: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_706, [0], True);  view_706 = None
    view_707: "f32[512]" = torch.ops.aten.view.default(sum_29, [512]);  sum_29 = None
    t_199: "f32[512, 512]" = torch.ops.aten.t.default(t_198);  t_198 = None
    view_708: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_51, [32, 49, 512]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_709: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_708, [32, 49, 16, 32]);  view_708 = None
    transpose_78: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_709, 1, 2);  view_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_282: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_78, memory_format = torch.contiguous_format);  transpose_78 = None
    _unsafe_view_117: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_282, [512, 49, 32]);  clone_282 = None
    bmm_68: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_79, _unsafe_view_117);  transpose_79 = None
    bmm_69: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_117, transpose_80);  _unsafe_view_117 = transpose_80 = None
    view_710: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_68, [32, 16, 49, 32]);  bmm_68 = None
    view_711: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_69, [32, 16, 49, 49]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_5: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_711, detach_29, -1, torch.float32);  view_711 = detach_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_30: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(_softmax_backward_data_5, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_5: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_30, 0);  sum_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_122: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_5, [1, 2, 0]);  squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_712: "f32[2401, 16]" = torch.ops.aten.view.default(permute_122, [2401, 16]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_5: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_712, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_5: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_5, [view_427], view_712, True);  new_zeros_5 = view_427 = view_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_713: "f32[512, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_5, [512, 49, 49]);  _softmax_backward_data_5 = None
    bmm_70: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_81, view_713);  transpose_81 = None
    bmm_71: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_713, transpose_82);  view_713 = transpose_82 = None
    view_714: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_70, [32, 16, 32, 49]);  bmm_70 = None
    view_715: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_71, [32, 16, 49, 32]);  bmm_71 = None
    transpose_83: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_714, -2, -1);  view_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_87: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_715, 0.1767766952966369);  view_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_5: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_87, transpose_83, view_710]);  mul_87 = transpose_83 = view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_123: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_5, [1, 3, 0, 2, 4]);  stack_5 = None
    clone_283: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    _unsafe_view_118: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_283, [32, 49, 1536]);  clone_283 = None
    view_716: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_118, [1568, 1536]);  _unsafe_view_118 = None
    mm_53: "f32[1568, 512]" = torch.ops.aten.mm.default(view_716, t_200);  t_200 = None
    t_201: "f32[1536, 1568]" = torch.ops.aten.t.default(view_716)
    mm_54: "f32[1536, 512]" = torch.ops.aten.mm.default(t_201, view_423);  t_201 = view_423 = None
    t_202: "f32[512, 1536]" = torch.ops.aten.t.default(mm_54);  mm_54 = None
    sum_31: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_716, [0], True);  view_716 = None
    view_717: "f32[1536]" = torch.ops.aten.view.default(sum_31, [1536]);  sum_31 = None
    t_203: "f32[1536, 512]" = torch.ops.aten.t.default(t_202);  t_202 = None
    view_718: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_53, [32, 49, 512]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_719: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_718, [32, 7, 7, 512]);  view_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_720: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_719, [8, 2, 2, 7, 7, 512]);  view_719 = None
    permute_124: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_720, [0, 1, 3, 2, 4, 5]);  view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_284: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    _unsafe_view_119: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_284, [8, 14, 14, 512]);  clone_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_29: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_119, [0, 0, 0, 0, 0, 0]);  _unsafe_view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_13 = torch.ops.aten.native_layer_norm_backward.default(constant_pad_nd_29, view_419, [512], getitem_172, getitem_173, primals_251, primals_252, [True, True, True]);  constant_pad_nd_29 = view_419 = getitem_172 = getitem_173 = primals_251 = primals_252 = None
    getitem_270: "f32[8, 14, 14, 512]" = native_layer_norm_backward_13[0]
    getitem_271: "f32[512]" = native_layer_norm_backward_13[1]
    getitem_272: "f32[512]" = native_layer_norm_backward_13[2];  native_layer_norm_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_94: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_703, getitem_270);  view_703 = getitem_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_721: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_94, [8, 196, 512]);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_88: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_721, div_33);  div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_722: "f32[1568, 512]" = torch.ops.aten.view.default(mul_88, [1568, 512]);  mul_88 = None
    mm_55: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_722, t_204);  t_204 = None
    t_205: "f32[512, 1568]" = torch.ops.aten.t.default(view_722)
    mm_56: "f32[512, 2048]" = torch.ops.aten.mm.default(t_205, view_417);  t_205 = view_417 = None
    t_206: "f32[2048, 512]" = torch.ops.aten.t.default(mm_56);  mm_56 = None
    sum_32: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_722, [0], True);  view_722 = None
    view_723: "f32[512]" = torch.ops.aten.view.default(sum_32, [512]);  sum_32 = None
    t_207: "f32[512, 2048]" = torch.ops.aten.t.default(t_206);  t_206 = None
    view_724: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_55, [8, 196, 2048]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_6: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_724, view_416);  view_724 = view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_725: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_6, [1568, 2048]);  gelu_backward_6 = None
    mm_57: "f32[1568, 512]" = torch.ops.aten.mm.default(view_725, t_208);  t_208 = None
    t_209: "f32[2048, 1568]" = torch.ops.aten.t.default(view_725)
    mm_58: "f32[2048, 512]" = torch.ops.aten.mm.default(t_209, view_415);  t_209 = view_415 = None
    t_210: "f32[512, 2048]" = torch.ops.aten.t.default(mm_58);  mm_58 = None
    sum_33: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_725, [0], True);  view_725 = None
    view_726: "f32[2048]" = torch.ops.aten.view.default(sum_33, [2048]);  sum_33 = None
    t_211: "f32[2048, 512]" = torch.ops.aten.t.default(t_210);  t_210 = None
    view_727: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_57, [8, 196, 512]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_14 = torch.ops.aten.native_layer_norm_backward.default(view_727, view_414, [512], getitem_169, getitem_170, primals_245, primals_246, [True, True, True]);  view_727 = view_414 = getitem_169 = getitem_170 = primals_245 = primals_246 = None
    getitem_273: "f32[8, 196, 512]" = native_layer_norm_backward_14[0]
    getitem_274: "f32[512]" = native_layer_norm_backward_14[1]
    getitem_275: "f32[512]" = native_layer_norm_backward_14[2];  native_layer_norm_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_95: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_721, getitem_273);  view_721 = getitem_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_728: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_95, [8, 14, 14, 512]);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_89: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_728, div_32);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_26: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_89, [-3, -3], [2, 1]);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_12: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(roll_26, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  roll_26 = None
    slice_backward_13: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_12, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_729: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_13, [8, 2, 7, 2, 7, 512]);  slice_backward_13 = None
    permute_125: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_729, [0, 1, 3, 2, 4, 5]);  view_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_285: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    _unsafe_view_120: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_285, [32, 7, 7, 512]);  clone_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_730: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_120, [32, 49, 512]);  _unsafe_view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_731: "f32[1568, 512]" = torch.ops.aten.view.default(view_730, [1568, 512]);  view_730 = None
    mm_59: "f32[1568, 512]" = torch.ops.aten.mm.default(view_731, t_212);  t_212 = None
    t_213: "f32[512, 1568]" = torch.ops.aten.t.default(view_731)
    mm_60: "f32[512, 512]" = torch.ops.aten.mm.default(t_213, view_409);  t_213 = view_409 = None
    t_214: "f32[512, 512]" = torch.ops.aten.t.default(mm_60);  mm_60 = None
    sum_34: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_731, [0], True);  view_731 = None
    view_732: "f32[512]" = torch.ops.aten.view.default(sum_34, [512]);  sum_34 = None
    t_215: "f32[512, 512]" = torch.ops.aten.t.default(t_214);  t_214 = None
    view_733: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_59, [32, 49, 512]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_734: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_733, [32, 49, 16, 32]);  view_733 = None
    transpose_84: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_734, 1, 2);  view_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_286: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_84, memory_format = torch.contiguous_format);  transpose_84 = None
    _unsafe_view_121: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_286, [512, 49, 32]);  clone_286 = None
    bmm_72: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_85, _unsafe_view_121);  transpose_85 = None
    bmm_73: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_121, transpose_86);  _unsafe_view_121 = transpose_86 = None
    view_735: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_72, [32, 16, 49, 32]);  bmm_72 = None
    view_736: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_73, [32, 16, 49, 49]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_6: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_736, detach_30, -1, torch.float32);  view_736 = detach_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_737: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_6, [8, 4, 16, 49, 49]);  _softmax_backward_data_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_738: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(view_737, [32, 16, 49, 49]);  view_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_35: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_738, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_6: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_35, 0);  sum_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_126: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_6, [1, 2, 0]);  squeeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_739: "f32[2401, 16]" = torch.ops.aten.view.default(permute_126, [2401, 16]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_6: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_739, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_6: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_6, [view_403], view_739, True);  new_zeros_6 = view_403 = view_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_740: "f32[512, 49, 49]" = torch.ops.aten.view.default(view_738, [512, 49, 49]);  view_738 = None
    bmm_74: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_87, view_740);  transpose_87 = None
    bmm_75: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_740, transpose_88);  view_740 = transpose_88 = None
    view_741: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_74, [32, 16, 32, 49]);  bmm_74 = None
    view_742: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_75, [32, 16, 49, 32]);  bmm_75 = None
    transpose_89: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_741, -2, -1);  view_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_90: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_742, 0.1767766952966369);  view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_6: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_90, transpose_89, view_735]);  mul_90 = transpose_89 = view_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_127: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_6, [1, 3, 0, 2, 4]);  stack_6 = None
    clone_287: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    _unsafe_view_122: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_287, [32, 49, 1536]);  clone_287 = None
    view_743: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_122, [1568, 1536]);  _unsafe_view_122 = None
    mm_61: "f32[1568, 512]" = torch.ops.aten.mm.default(view_743, t_216);  t_216 = None
    t_217: "f32[1536, 1568]" = torch.ops.aten.t.default(view_743)
    mm_62: "f32[1536, 512]" = torch.ops.aten.mm.default(t_217, view_399);  t_217 = view_399 = None
    t_218: "f32[512, 1536]" = torch.ops.aten.t.default(mm_62);  mm_62 = None
    sum_36: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_743, [0], True);  view_743 = None
    view_744: "f32[1536]" = torch.ops.aten.view.default(sum_36, [1536]);  sum_36 = None
    t_219: "f32[1536, 512]" = torch.ops.aten.t.default(t_218);  t_218 = None
    view_745: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_61, [32, 49, 512]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_746: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_745, [32, 7, 7, 512]);  view_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_747: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_746, [8, 2, 2, 7, 7, 512]);  view_746 = None
    permute_128: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_747, [0, 1, 3, 2, 4, 5]);  view_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_288: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    _unsafe_view_123: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_288, [8, 14, 14, 512]);  clone_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_30: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_123, [0, 0, 0, 0, 0, 0]);  _unsafe_view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_27: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(constant_pad_nd_30, [3, 3], [2, 1]);  constant_pad_nd_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_15 = torch.ops.aten.native_layer_norm_backward.default(roll_27, view_395, [512], getitem_163, getitem_164, primals_239, primals_240, [True, True, True]);  roll_27 = view_395 = getitem_163 = getitem_164 = primals_239 = primals_240 = None
    getitem_276: "f32[8, 14, 14, 512]" = native_layer_norm_backward_15[0]
    getitem_277: "f32[512]" = native_layer_norm_backward_15[1]
    getitem_278: "f32[512]" = native_layer_norm_backward_15[2];  native_layer_norm_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_96: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_728, getitem_276);  view_728 = getitem_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_748: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_96, [8, 196, 512]);  add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_91: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_748, div_31);  div_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_749: "f32[1568, 512]" = torch.ops.aten.view.default(mul_91, [1568, 512]);  mul_91 = None
    mm_63: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_749, t_220);  t_220 = None
    t_221: "f32[512, 1568]" = torch.ops.aten.t.default(view_749)
    mm_64: "f32[512, 2048]" = torch.ops.aten.mm.default(t_221, view_393);  t_221 = view_393 = None
    t_222: "f32[2048, 512]" = torch.ops.aten.t.default(mm_64);  mm_64 = None
    sum_37: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_749, [0], True);  view_749 = None
    view_750: "f32[512]" = torch.ops.aten.view.default(sum_37, [512]);  sum_37 = None
    t_223: "f32[512, 2048]" = torch.ops.aten.t.default(t_222);  t_222 = None
    view_751: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_63, [8, 196, 2048]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_7: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_751, view_392);  view_751 = view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_752: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_7, [1568, 2048]);  gelu_backward_7 = None
    mm_65: "f32[1568, 512]" = torch.ops.aten.mm.default(view_752, t_224);  t_224 = None
    t_225: "f32[2048, 1568]" = torch.ops.aten.t.default(view_752)
    mm_66: "f32[2048, 512]" = torch.ops.aten.mm.default(t_225, view_391);  t_225 = view_391 = None
    t_226: "f32[512, 2048]" = torch.ops.aten.t.default(mm_66);  mm_66 = None
    sum_38: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_752, [0], True);  view_752 = None
    view_753: "f32[2048]" = torch.ops.aten.view.default(sum_38, [2048]);  sum_38 = None
    t_227: "f32[2048, 512]" = torch.ops.aten.t.default(t_226);  t_226 = None
    view_754: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_65, [8, 196, 512]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_16 = torch.ops.aten.native_layer_norm_backward.default(view_754, view_390, [512], getitem_160, getitem_161, primals_233, primals_234, [True, True, True]);  view_754 = view_390 = getitem_160 = getitem_161 = primals_233 = primals_234 = None
    getitem_279: "f32[8, 196, 512]" = native_layer_norm_backward_16[0]
    getitem_280: "f32[512]" = native_layer_norm_backward_16[1]
    getitem_281: "f32[512]" = native_layer_norm_backward_16[2];  native_layer_norm_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_97: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_748, getitem_279);  view_748 = getitem_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_755: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_97, [8, 14, 14, 512]);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_92: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_755, div_30);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_14: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(mul_92, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  mul_92 = None
    slice_backward_15: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_14, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_756: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_15, [8, 2, 7, 2, 7, 512]);  slice_backward_15 = None
    permute_129: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_756, [0, 1, 3, 2, 4, 5]);  view_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_289: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    _unsafe_view_124: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_289, [32, 7, 7, 512]);  clone_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_757: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_124, [32, 49, 512]);  _unsafe_view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_758: "f32[1568, 512]" = torch.ops.aten.view.default(view_757, [1568, 512]);  view_757 = None
    mm_67: "f32[1568, 512]" = torch.ops.aten.mm.default(view_758, t_228);  t_228 = None
    t_229: "f32[512, 1568]" = torch.ops.aten.t.default(view_758)
    mm_68: "f32[512, 512]" = torch.ops.aten.mm.default(t_229, view_385);  t_229 = view_385 = None
    t_230: "f32[512, 512]" = torch.ops.aten.t.default(mm_68);  mm_68 = None
    sum_39: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_758, [0], True);  view_758 = None
    view_759: "f32[512]" = torch.ops.aten.view.default(sum_39, [512]);  sum_39 = None
    t_231: "f32[512, 512]" = torch.ops.aten.t.default(t_230);  t_230 = None
    view_760: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_67, [32, 49, 512]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_761: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_760, [32, 49, 16, 32]);  view_760 = None
    transpose_90: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_761, 1, 2);  view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_290: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_90, memory_format = torch.contiguous_format);  transpose_90 = None
    _unsafe_view_125: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_290, [512, 49, 32]);  clone_290 = None
    bmm_76: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_91, _unsafe_view_125);  transpose_91 = None
    bmm_77: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_125, transpose_92);  _unsafe_view_125 = transpose_92 = None
    view_762: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_76, [32, 16, 49, 32]);  bmm_76 = None
    view_763: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_77, [32, 16, 49, 49]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_7: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_763, detach_31, -1, torch.float32);  view_763 = detach_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_40: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(_softmax_backward_data_7, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_7: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_40, 0);  sum_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_130: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_7, [1, 2, 0]);  squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_764: "f32[2401, 16]" = torch.ops.aten.view.default(permute_130, [2401, 16]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_7: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_764, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_7: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_7, [view_381], view_764, True);  new_zeros_7 = view_381 = view_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_765: "f32[512, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_7, [512, 49, 49]);  _softmax_backward_data_7 = None
    bmm_78: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_93, view_765);  transpose_93 = None
    bmm_79: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_765, transpose_94);  view_765 = transpose_94 = None
    view_766: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_78, [32, 16, 32, 49]);  bmm_78 = None
    view_767: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_79, [32, 16, 49, 32]);  bmm_79 = None
    transpose_95: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_766, -2, -1);  view_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_93: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_767, 0.1767766952966369);  view_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_7: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_93, transpose_95, view_762]);  mul_93 = transpose_95 = view_762 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_131: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_7, [1, 3, 0, 2, 4]);  stack_7 = None
    clone_291: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_131, memory_format = torch.contiguous_format);  permute_131 = None
    _unsafe_view_126: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_291, [32, 49, 1536]);  clone_291 = None
    view_768: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_126, [1568, 1536]);  _unsafe_view_126 = None
    mm_69: "f32[1568, 512]" = torch.ops.aten.mm.default(view_768, t_232);  t_232 = None
    t_233: "f32[1536, 1568]" = torch.ops.aten.t.default(view_768)
    mm_70: "f32[1536, 512]" = torch.ops.aten.mm.default(t_233, view_377);  t_233 = view_377 = None
    t_234: "f32[512, 1536]" = torch.ops.aten.t.default(mm_70);  mm_70 = None
    sum_41: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_768, [0], True);  view_768 = None
    view_769: "f32[1536]" = torch.ops.aten.view.default(sum_41, [1536]);  sum_41 = None
    t_235: "f32[1536, 512]" = torch.ops.aten.t.default(t_234);  t_234 = None
    view_770: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_69, [32, 49, 512]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_771: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_770, [32, 7, 7, 512]);  view_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_772: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_771, [8, 2, 2, 7, 7, 512]);  view_771 = None
    permute_132: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_772, [0, 1, 3, 2, 4, 5]);  view_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_292: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    _unsafe_view_127: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_292, [8, 14, 14, 512]);  clone_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_31: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_127, [0, 0, 0, 0, 0, 0]);  _unsafe_view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_17 = torch.ops.aten.native_layer_norm_backward.default(constant_pad_nd_31, view_373, [512], getitem_154, getitem_155, primals_227, primals_228, [True, True, True]);  constant_pad_nd_31 = view_373 = getitem_154 = getitem_155 = primals_227 = primals_228 = None
    getitem_282: "f32[8, 14, 14, 512]" = native_layer_norm_backward_17[0]
    getitem_283: "f32[512]" = native_layer_norm_backward_17[1]
    getitem_284: "f32[512]" = native_layer_norm_backward_17[2];  native_layer_norm_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_98: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_755, getitem_282);  view_755 = getitem_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_773: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_98, [8, 196, 512]);  add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_94: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_773, div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_774: "f32[1568, 512]" = torch.ops.aten.view.default(mul_94, [1568, 512]);  mul_94 = None
    mm_71: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_774, t_236);  t_236 = None
    t_237: "f32[512, 1568]" = torch.ops.aten.t.default(view_774)
    mm_72: "f32[512, 2048]" = torch.ops.aten.mm.default(t_237, view_371);  t_237 = view_371 = None
    t_238: "f32[2048, 512]" = torch.ops.aten.t.default(mm_72);  mm_72 = None
    sum_42: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_774, [0], True);  view_774 = None
    view_775: "f32[512]" = torch.ops.aten.view.default(sum_42, [512]);  sum_42 = None
    t_239: "f32[512, 2048]" = torch.ops.aten.t.default(t_238);  t_238 = None
    view_776: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_71, [8, 196, 2048]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_8: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_776, view_370);  view_776 = view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_777: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_8, [1568, 2048]);  gelu_backward_8 = None
    mm_73: "f32[1568, 512]" = torch.ops.aten.mm.default(view_777, t_240);  t_240 = None
    t_241: "f32[2048, 1568]" = torch.ops.aten.t.default(view_777)
    mm_74: "f32[2048, 512]" = torch.ops.aten.mm.default(t_241, view_369);  t_241 = view_369 = None
    t_242: "f32[512, 2048]" = torch.ops.aten.t.default(mm_74);  mm_74 = None
    sum_43: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_777, [0], True);  view_777 = None
    view_778: "f32[2048]" = torch.ops.aten.view.default(sum_43, [2048]);  sum_43 = None
    t_243: "f32[2048, 512]" = torch.ops.aten.t.default(t_242);  t_242 = None
    view_779: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_73, [8, 196, 512]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_18 = torch.ops.aten.native_layer_norm_backward.default(view_779, view_368, [512], getitem_151, getitem_152, primals_221, primals_222, [True, True, True]);  view_779 = view_368 = getitem_151 = getitem_152 = primals_221 = primals_222 = None
    getitem_285: "f32[8, 196, 512]" = native_layer_norm_backward_18[0]
    getitem_286: "f32[512]" = native_layer_norm_backward_18[1]
    getitem_287: "f32[512]" = native_layer_norm_backward_18[2];  native_layer_norm_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_99: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_773, getitem_285);  view_773 = getitem_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_780: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_99, [8, 14, 14, 512]);  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_95: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_780, div_28);  div_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_28: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_95, [-3, -3], [2, 1]);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_16: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(roll_28, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  roll_28 = None
    slice_backward_17: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_16, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_781: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_17, [8, 2, 7, 2, 7, 512]);  slice_backward_17 = None
    permute_133: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_781, [0, 1, 3, 2, 4, 5]);  view_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_293: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    _unsafe_view_128: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_293, [32, 7, 7, 512]);  clone_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_782: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_128, [32, 49, 512]);  _unsafe_view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_783: "f32[1568, 512]" = torch.ops.aten.view.default(view_782, [1568, 512]);  view_782 = None
    mm_75: "f32[1568, 512]" = torch.ops.aten.mm.default(view_783, t_244);  t_244 = None
    t_245: "f32[512, 1568]" = torch.ops.aten.t.default(view_783)
    mm_76: "f32[512, 512]" = torch.ops.aten.mm.default(t_245, view_363);  t_245 = view_363 = None
    t_246: "f32[512, 512]" = torch.ops.aten.t.default(mm_76);  mm_76 = None
    sum_44: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_783, [0], True);  view_783 = None
    view_784: "f32[512]" = torch.ops.aten.view.default(sum_44, [512]);  sum_44 = None
    t_247: "f32[512, 512]" = torch.ops.aten.t.default(t_246);  t_246 = None
    view_785: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_75, [32, 49, 512]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_786: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_785, [32, 49, 16, 32]);  view_785 = None
    transpose_96: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_786, 1, 2);  view_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_294: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_96, memory_format = torch.contiguous_format);  transpose_96 = None
    _unsafe_view_129: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_294, [512, 49, 32]);  clone_294 = None
    bmm_80: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_97, _unsafe_view_129);  transpose_97 = None
    bmm_81: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_129, transpose_98);  _unsafe_view_129 = transpose_98 = None
    view_787: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_80, [32, 16, 49, 32]);  bmm_80 = None
    view_788: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_81, [32, 16, 49, 49]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_8: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_788, detach_32, -1, torch.float32);  view_788 = detach_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_789: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_8, [8, 4, 16, 49, 49]);  _softmax_backward_data_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_790: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(view_789, [32, 16, 49, 49]);  view_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_45: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_790, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_8: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_45, 0);  sum_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_134: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_8, [1, 2, 0]);  squeeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_791: "f32[2401, 16]" = torch.ops.aten.view.default(permute_134, [2401, 16]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_8: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_791, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_8: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_8, [view_357], view_791, True);  new_zeros_8 = view_357 = view_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_792: "f32[512, 49, 49]" = torch.ops.aten.view.default(view_790, [512, 49, 49]);  view_790 = None
    bmm_82: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_99, view_792);  transpose_99 = None
    bmm_83: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_792, transpose_100);  view_792 = transpose_100 = None
    view_793: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_82, [32, 16, 32, 49]);  bmm_82 = None
    view_794: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_83, [32, 16, 49, 32]);  bmm_83 = None
    transpose_101: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_793, -2, -1);  view_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_96: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_794, 0.1767766952966369);  view_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_8: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_96, transpose_101, view_787]);  mul_96 = transpose_101 = view_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_135: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_8, [1, 3, 0, 2, 4]);  stack_8 = None
    clone_295: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
    _unsafe_view_130: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_295, [32, 49, 1536]);  clone_295 = None
    view_795: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_130, [1568, 1536]);  _unsafe_view_130 = None
    mm_77: "f32[1568, 512]" = torch.ops.aten.mm.default(view_795, t_248);  t_248 = None
    t_249: "f32[1536, 1568]" = torch.ops.aten.t.default(view_795)
    mm_78: "f32[1536, 512]" = torch.ops.aten.mm.default(t_249, view_353);  t_249 = view_353 = None
    t_250: "f32[512, 1536]" = torch.ops.aten.t.default(mm_78);  mm_78 = None
    sum_46: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_795, [0], True);  view_795 = None
    view_796: "f32[1536]" = torch.ops.aten.view.default(sum_46, [1536]);  sum_46 = None
    t_251: "f32[1536, 512]" = torch.ops.aten.t.default(t_250);  t_250 = None
    view_797: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_77, [32, 49, 512]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_798: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_797, [32, 7, 7, 512]);  view_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_799: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_798, [8, 2, 2, 7, 7, 512]);  view_798 = None
    permute_136: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_799, [0, 1, 3, 2, 4, 5]);  view_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_296: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    _unsafe_view_131: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_296, [8, 14, 14, 512]);  clone_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_32: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_131, [0, 0, 0, 0, 0, 0]);  _unsafe_view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_29: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(constant_pad_nd_32, [3, 3], [2, 1]);  constant_pad_nd_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_19 = torch.ops.aten.native_layer_norm_backward.default(roll_29, view_349, [512], getitem_145, getitem_146, primals_215, primals_216, [True, True, True]);  roll_29 = view_349 = getitem_145 = getitem_146 = primals_215 = primals_216 = None
    getitem_288: "f32[8, 14, 14, 512]" = native_layer_norm_backward_19[0]
    getitem_289: "f32[512]" = native_layer_norm_backward_19[1]
    getitem_290: "f32[512]" = native_layer_norm_backward_19[2];  native_layer_norm_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_100: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_780, getitem_288);  view_780 = getitem_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_800: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_100, [8, 196, 512]);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_97: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_800, div_27);  div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_801: "f32[1568, 512]" = torch.ops.aten.view.default(mul_97, [1568, 512]);  mul_97 = None
    mm_79: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_801, t_252);  t_252 = None
    t_253: "f32[512, 1568]" = torch.ops.aten.t.default(view_801)
    mm_80: "f32[512, 2048]" = torch.ops.aten.mm.default(t_253, view_347);  t_253 = view_347 = None
    t_254: "f32[2048, 512]" = torch.ops.aten.t.default(mm_80);  mm_80 = None
    sum_47: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_801, [0], True);  view_801 = None
    view_802: "f32[512]" = torch.ops.aten.view.default(sum_47, [512]);  sum_47 = None
    t_255: "f32[512, 2048]" = torch.ops.aten.t.default(t_254);  t_254 = None
    view_803: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_79, [8, 196, 2048]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_9: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_803, view_346);  view_803 = view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_804: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_9, [1568, 2048]);  gelu_backward_9 = None
    mm_81: "f32[1568, 512]" = torch.ops.aten.mm.default(view_804, t_256);  t_256 = None
    t_257: "f32[2048, 1568]" = torch.ops.aten.t.default(view_804)
    mm_82: "f32[2048, 512]" = torch.ops.aten.mm.default(t_257, view_345);  t_257 = view_345 = None
    t_258: "f32[512, 2048]" = torch.ops.aten.t.default(mm_82);  mm_82 = None
    sum_48: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_804, [0], True);  view_804 = None
    view_805: "f32[2048]" = torch.ops.aten.view.default(sum_48, [2048]);  sum_48 = None
    t_259: "f32[2048, 512]" = torch.ops.aten.t.default(t_258);  t_258 = None
    view_806: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_81, [8, 196, 512]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_20 = torch.ops.aten.native_layer_norm_backward.default(view_806, view_344, [512], getitem_142, getitem_143, primals_209, primals_210, [True, True, True]);  view_806 = view_344 = getitem_142 = getitem_143 = primals_209 = primals_210 = None
    getitem_291: "f32[8, 196, 512]" = native_layer_norm_backward_20[0]
    getitem_292: "f32[512]" = native_layer_norm_backward_20[1]
    getitem_293: "f32[512]" = native_layer_norm_backward_20[2];  native_layer_norm_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_101: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_800, getitem_291);  view_800 = getitem_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_807: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_101, [8, 14, 14, 512]);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_98: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_807, div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_18: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(mul_98, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  mul_98 = None
    slice_backward_19: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_18, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_808: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_19, [8, 2, 7, 2, 7, 512]);  slice_backward_19 = None
    permute_137: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_808, [0, 1, 3, 2, 4, 5]);  view_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_297: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    _unsafe_view_132: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_297, [32, 7, 7, 512]);  clone_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_809: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_132, [32, 49, 512]);  _unsafe_view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_810: "f32[1568, 512]" = torch.ops.aten.view.default(view_809, [1568, 512]);  view_809 = None
    mm_83: "f32[1568, 512]" = torch.ops.aten.mm.default(view_810, t_260);  t_260 = None
    t_261: "f32[512, 1568]" = torch.ops.aten.t.default(view_810)
    mm_84: "f32[512, 512]" = torch.ops.aten.mm.default(t_261, view_339);  t_261 = view_339 = None
    t_262: "f32[512, 512]" = torch.ops.aten.t.default(mm_84);  mm_84 = None
    sum_49: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_810, [0], True);  view_810 = None
    view_811: "f32[512]" = torch.ops.aten.view.default(sum_49, [512]);  sum_49 = None
    t_263: "f32[512, 512]" = torch.ops.aten.t.default(t_262);  t_262 = None
    view_812: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_83, [32, 49, 512]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_813: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_812, [32, 49, 16, 32]);  view_812 = None
    transpose_102: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_813, 1, 2);  view_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_298: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_102, memory_format = torch.contiguous_format);  transpose_102 = None
    _unsafe_view_133: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_298, [512, 49, 32]);  clone_298 = None
    bmm_84: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_103, _unsafe_view_133);  transpose_103 = None
    bmm_85: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_133, transpose_104);  _unsafe_view_133 = transpose_104 = None
    view_814: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_84, [32, 16, 49, 32]);  bmm_84 = None
    view_815: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_85, [32, 16, 49, 49]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_9: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_815, detach_33, -1, torch.float32);  view_815 = detach_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_50: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(_softmax_backward_data_9, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_9: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_50, 0);  sum_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_138: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_9, [1, 2, 0]);  squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_816: "f32[2401, 16]" = torch.ops.aten.view.default(permute_138, [2401, 16]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_9: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_816, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_9: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_9, [view_335], view_816, True);  new_zeros_9 = view_335 = view_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_817: "f32[512, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_9, [512, 49, 49]);  _softmax_backward_data_9 = None
    bmm_86: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_105, view_817);  transpose_105 = None
    bmm_87: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_817, transpose_106);  view_817 = transpose_106 = None
    view_818: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_86, [32, 16, 32, 49]);  bmm_86 = None
    view_819: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_87, [32, 16, 49, 32]);  bmm_87 = None
    transpose_107: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_818, -2, -1);  view_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_99: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_819, 0.1767766952966369);  view_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_9: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_99, transpose_107, view_814]);  mul_99 = transpose_107 = view_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_139: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_9, [1, 3, 0, 2, 4]);  stack_9 = None
    clone_299: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    _unsafe_view_134: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_299, [32, 49, 1536]);  clone_299 = None
    view_820: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_134, [1568, 1536]);  _unsafe_view_134 = None
    mm_85: "f32[1568, 512]" = torch.ops.aten.mm.default(view_820, t_264);  t_264 = None
    t_265: "f32[1536, 1568]" = torch.ops.aten.t.default(view_820)
    mm_86: "f32[1536, 512]" = torch.ops.aten.mm.default(t_265, view_331);  t_265 = view_331 = None
    t_266: "f32[512, 1536]" = torch.ops.aten.t.default(mm_86);  mm_86 = None
    sum_51: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_820, [0], True);  view_820 = None
    view_821: "f32[1536]" = torch.ops.aten.view.default(sum_51, [1536]);  sum_51 = None
    t_267: "f32[1536, 512]" = torch.ops.aten.t.default(t_266);  t_266 = None
    view_822: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_85, [32, 49, 512]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_823: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_822, [32, 7, 7, 512]);  view_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_824: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_823, [8, 2, 2, 7, 7, 512]);  view_823 = None
    permute_140: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_824, [0, 1, 3, 2, 4, 5]);  view_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_300: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    _unsafe_view_135: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_300, [8, 14, 14, 512]);  clone_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_33: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_135, [0, 0, 0, 0, 0, 0]);  _unsafe_view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_21 = torch.ops.aten.native_layer_norm_backward.default(constant_pad_nd_33, view_327, [512], getitem_136, getitem_137, primals_203, primals_204, [True, True, True]);  constant_pad_nd_33 = view_327 = getitem_136 = getitem_137 = primals_203 = primals_204 = None
    getitem_294: "f32[8, 14, 14, 512]" = native_layer_norm_backward_21[0]
    getitem_295: "f32[512]" = native_layer_norm_backward_21[1]
    getitem_296: "f32[512]" = native_layer_norm_backward_21[2];  native_layer_norm_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_102: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_807, getitem_294);  view_807 = getitem_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_825: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_102, [8, 196, 512]);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_100: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_825, div_25);  div_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_826: "f32[1568, 512]" = torch.ops.aten.view.default(mul_100, [1568, 512]);  mul_100 = None
    mm_87: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_826, t_268);  t_268 = None
    t_269: "f32[512, 1568]" = torch.ops.aten.t.default(view_826)
    mm_88: "f32[512, 2048]" = torch.ops.aten.mm.default(t_269, view_325);  t_269 = view_325 = None
    t_270: "f32[2048, 512]" = torch.ops.aten.t.default(mm_88);  mm_88 = None
    sum_52: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_826, [0], True);  view_826 = None
    view_827: "f32[512]" = torch.ops.aten.view.default(sum_52, [512]);  sum_52 = None
    t_271: "f32[512, 2048]" = torch.ops.aten.t.default(t_270);  t_270 = None
    view_828: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_87, [8, 196, 2048]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_10: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_828, view_324);  view_828 = view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_829: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_10, [1568, 2048]);  gelu_backward_10 = None
    mm_89: "f32[1568, 512]" = torch.ops.aten.mm.default(view_829, t_272);  t_272 = None
    t_273: "f32[2048, 1568]" = torch.ops.aten.t.default(view_829)
    mm_90: "f32[2048, 512]" = torch.ops.aten.mm.default(t_273, view_323);  t_273 = view_323 = None
    t_274: "f32[512, 2048]" = torch.ops.aten.t.default(mm_90);  mm_90 = None
    sum_53: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_829, [0], True);  view_829 = None
    view_830: "f32[2048]" = torch.ops.aten.view.default(sum_53, [2048]);  sum_53 = None
    t_275: "f32[2048, 512]" = torch.ops.aten.t.default(t_274);  t_274 = None
    view_831: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_89, [8, 196, 512]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_22 = torch.ops.aten.native_layer_norm_backward.default(view_831, view_322, [512], getitem_133, getitem_134, primals_197, primals_198, [True, True, True]);  view_831 = view_322 = getitem_133 = getitem_134 = primals_197 = primals_198 = None
    getitem_297: "f32[8, 196, 512]" = native_layer_norm_backward_22[0]
    getitem_298: "f32[512]" = native_layer_norm_backward_22[1]
    getitem_299: "f32[512]" = native_layer_norm_backward_22[2];  native_layer_norm_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_103: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_825, getitem_297);  view_825 = getitem_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_832: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_103, [8, 14, 14, 512]);  add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_101: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_832, div_24);  div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_30: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_101, [-3, -3], [2, 1]);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_20: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(roll_30, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  roll_30 = None
    slice_backward_21: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_20, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_833: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_21, [8, 2, 7, 2, 7, 512]);  slice_backward_21 = None
    permute_141: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_833, [0, 1, 3, 2, 4, 5]);  view_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_301: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
    _unsafe_view_136: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_301, [32, 7, 7, 512]);  clone_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_834: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_136, [32, 49, 512]);  _unsafe_view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_835: "f32[1568, 512]" = torch.ops.aten.view.default(view_834, [1568, 512]);  view_834 = None
    mm_91: "f32[1568, 512]" = torch.ops.aten.mm.default(view_835, t_276);  t_276 = None
    t_277: "f32[512, 1568]" = torch.ops.aten.t.default(view_835)
    mm_92: "f32[512, 512]" = torch.ops.aten.mm.default(t_277, view_317);  t_277 = view_317 = None
    t_278: "f32[512, 512]" = torch.ops.aten.t.default(mm_92);  mm_92 = None
    sum_54: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_835, [0], True);  view_835 = None
    view_836: "f32[512]" = torch.ops.aten.view.default(sum_54, [512]);  sum_54 = None
    t_279: "f32[512, 512]" = torch.ops.aten.t.default(t_278);  t_278 = None
    view_837: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_91, [32, 49, 512]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_838: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_837, [32, 49, 16, 32]);  view_837 = None
    transpose_108: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_838, 1, 2);  view_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_302: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_108, memory_format = torch.contiguous_format);  transpose_108 = None
    _unsafe_view_137: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_302, [512, 49, 32]);  clone_302 = None
    bmm_88: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_109, _unsafe_view_137);  transpose_109 = None
    bmm_89: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_137, transpose_110);  _unsafe_view_137 = transpose_110 = None
    view_839: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_88, [32, 16, 49, 32]);  bmm_88 = None
    view_840: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_89, [32, 16, 49, 49]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_10: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_840, detach_34, -1, torch.float32);  view_840 = detach_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_841: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_10, [8, 4, 16, 49, 49]);  _softmax_backward_data_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_842: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(view_841, [32, 16, 49, 49]);  view_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_55: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_842, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_10: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_55, 0);  sum_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_142: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_10, [1, 2, 0]);  squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_843: "f32[2401, 16]" = torch.ops.aten.view.default(permute_142, [2401, 16]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_10: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_843, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_10: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_10, [view_311], view_843, True);  new_zeros_10 = view_311 = view_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_844: "f32[512, 49, 49]" = torch.ops.aten.view.default(view_842, [512, 49, 49]);  view_842 = None
    bmm_90: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_111, view_844);  transpose_111 = None
    bmm_91: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_844, transpose_112);  view_844 = transpose_112 = None
    view_845: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_90, [32, 16, 32, 49]);  bmm_90 = None
    view_846: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_91, [32, 16, 49, 32]);  bmm_91 = None
    transpose_113: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_845, -2, -1);  view_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_102: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_846, 0.1767766952966369);  view_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_10: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_102, transpose_113, view_839]);  mul_102 = transpose_113 = view_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_143: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_10, [1, 3, 0, 2, 4]);  stack_10 = None
    clone_303: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
    _unsafe_view_138: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_303, [32, 49, 1536]);  clone_303 = None
    view_847: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_138, [1568, 1536]);  _unsafe_view_138 = None
    mm_93: "f32[1568, 512]" = torch.ops.aten.mm.default(view_847, t_280);  t_280 = None
    t_281: "f32[1536, 1568]" = torch.ops.aten.t.default(view_847)
    mm_94: "f32[1536, 512]" = torch.ops.aten.mm.default(t_281, view_307);  t_281 = view_307 = None
    t_282: "f32[512, 1536]" = torch.ops.aten.t.default(mm_94);  mm_94 = None
    sum_56: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_847, [0], True);  view_847 = None
    view_848: "f32[1536]" = torch.ops.aten.view.default(sum_56, [1536]);  sum_56 = None
    t_283: "f32[1536, 512]" = torch.ops.aten.t.default(t_282);  t_282 = None
    view_849: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_93, [32, 49, 512]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_850: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_849, [32, 7, 7, 512]);  view_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_851: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_850, [8, 2, 2, 7, 7, 512]);  view_850 = None
    permute_144: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_851, [0, 1, 3, 2, 4, 5]);  view_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_304: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    _unsafe_view_139: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_304, [8, 14, 14, 512]);  clone_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_34: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_139, [0, 0, 0, 0, 0, 0]);  _unsafe_view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_31: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(constant_pad_nd_34, [3, 3], [2, 1]);  constant_pad_nd_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_23 = torch.ops.aten.native_layer_norm_backward.default(roll_31, view_303, [512], getitem_127, getitem_128, primals_191, primals_192, [True, True, True]);  roll_31 = view_303 = getitem_127 = getitem_128 = primals_191 = primals_192 = None
    getitem_300: "f32[8, 14, 14, 512]" = native_layer_norm_backward_23[0]
    getitem_301: "f32[512]" = native_layer_norm_backward_23[1]
    getitem_302: "f32[512]" = native_layer_norm_backward_23[2];  native_layer_norm_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_104: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_832, getitem_300);  view_832 = getitem_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_852: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_104, [8, 196, 512]);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_103: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_852, div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_853: "f32[1568, 512]" = torch.ops.aten.view.default(mul_103, [1568, 512]);  mul_103 = None
    mm_95: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_853, t_284);  t_284 = None
    t_285: "f32[512, 1568]" = torch.ops.aten.t.default(view_853)
    mm_96: "f32[512, 2048]" = torch.ops.aten.mm.default(t_285, view_301);  t_285 = view_301 = None
    t_286: "f32[2048, 512]" = torch.ops.aten.t.default(mm_96);  mm_96 = None
    sum_57: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_853, [0], True);  view_853 = None
    view_854: "f32[512]" = torch.ops.aten.view.default(sum_57, [512]);  sum_57 = None
    t_287: "f32[512, 2048]" = torch.ops.aten.t.default(t_286);  t_286 = None
    view_855: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_95, [8, 196, 2048]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_11: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_855, view_300);  view_855 = view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_856: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_11, [1568, 2048]);  gelu_backward_11 = None
    mm_97: "f32[1568, 512]" = torch.ops.aten.mm.default(view_856, t_288);  t_288 = None
    t_289: "f32[2048, 1568]" = torch.ops.aten.t.default(view_856)
    mm_98: "f32[2048, 512]" = torch.ops.aten.mm.default(t_289, view_299);  t_289 = view_299 = None
    t_290: "f32[512, 2048]" = torch.ops.aten.t.default(mm_98);  mm_98 = None
    sum_58: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_856, [0], True);  view_856 = None
    view_857: "f32[2048]" = torch.ops.aten.view.default(sum_58, [2048]);  sum_58 = None
    t_291: "f32[2048, 512]" = torch.ops.aten.t.default(t_290);  t_290 = None
    view_858: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_97, [8, 196, 512]);  mm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_24 = torch.ops.aten.native_layer_norm_backward.default(view_858, view_298, [512], getitem_124, getitem_125, primals_185, primals_186, [True, True, True]);  view_858 = view_298 = getitem_124 = getitem_125 = primals_185 = primals_186 = None
    getitem_303: "f32[8, 196, 512]" = native_layer_norm_backward_24[0]
    getitem_304: "f32[512]" = native_layer_norm_backward_24[1]
    getitem_305: "f32[512]" = native_layer_norm_backward_24[2];  native_layer_norm_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_105: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_852, getitem_303);  view_852 = getitem_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_859: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_105, [8, 14, 14, 512]);  add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_104: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_859, div_22);  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_22: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(mul_104, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  mul_104 = None
    slice_backward_23: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_22, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_860: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_23, [8, 2, 7, 2, 7, 512]);  slice_backward_23 = None
    permute_145: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_860, [0, 1, 3, 2, 4, 5]);  view_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_305: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    _unsafe_view_140: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_305, [32, 7, 7, 512]);  clone_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_861: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_140, [32, 49, 512]);  _unsafe_view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_862: "f32[1568, 512]" = torch.ops.aten.view.default(view_861, [1568, 512]);  view_861 = None
    mm_99: "f32[1568, 512]" = torch.ops.aten.mm.default(view_862, t_292);  t_292 = None
    t_293: "f32[512, 1568]" = torch.ops.aten.t.default(view_862)
    mm_100: "f32[512, 512]" = torch.ops.aten.mm.default(t_293, view_293);  t_293 = view_293 = None
    t_294: "f32[512, 512]" = torch.ops.aten.t.default(mm_100);  mm_100 = None
    sum_59: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_862, [0], True);  view_862 = None
    view_863: "f32[512]" = torch.ops.aten.view.default(sum_59, [512]);  sum_59 = None
    t_295: "f32[512, 512]" = torch.ops.aten.t.default(t_294);  t_294 = None
    view_864: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_99, [32, 49, 512]);  mm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_865: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_864, [32, 49, 16, 32]);  view_864 = None
    transpose_114: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_865, 1, 2);  view_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_306: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_114, memory_format = torch.contiguous_format);  transpose_114 = None
    _unsafe_view_141: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_306, [512, 49, 32]);  clone_306 = None
    bmm_92: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_115, _unsafe_view_141);  transpose_115 = None
    bmm_93: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_141, transpose_116);  _unsafe_view_141 = transpose_116 = None
    view_866: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_92, [32, 16, 49, 32]);  bmm_92 = None
    view_867: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_93, [32, 16, 49, 49]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_11: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_867, detach_35, -1, torch.float32);  view_867 = detach_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_60: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(_softmax_backward_data_11, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_11: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_60, 0);  sum_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_146: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_11, [1, 2, 0]);  squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_868: "f32[2401, 16]" = torch.ops.aten.view.default(permute_146, [2401, 16]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_11: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_868, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_11: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_11, [view_289], view_868, True);  new_zeros_11 = view_289 = view_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_869: "f32[512, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_11, [512, 49, 49]);  _softmax_backward_data_11 = None
    bmm_94: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_117, view_869);  transpose_117 = None
    bmm_95: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_869, transpose_118);  view_869 = transpose_118 = None
    view_870: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_94, [32, 16, 32, 49]);  bmm_94 = None
    view_871: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_95, [32, 16, 49, 32]);  bmm_95 = None
    transpose_119: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_870, -2, -1);  view_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_105: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_871, 0.1767766952966369);  view_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_11: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_105, transpose_119, view_866]);  mul_105 = transpose_119 = view_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_147: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_11, [1, 3, 0, 2, 4]);  stack_11 = None
    clone_307: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    _unsafe_view_142: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_307, [32, 49, 1536]);  clone_307 = None
    view_872: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_142, [1568, 1536]);  _unsafe_view_142 = None
    mm_101: "f32[1568, 512]" = torch.ops.aten.mm.default(view_872, t_296);  t_296 = None
    t_297: "f32[1536, 1568]" = torch.ops.aten.t.default(view_872)
    mm_102: "f32[1536, 512]" = torch.ops.aten.mm.default(t_297, view_285);  t_297 = view_285 = None
    t_298: "f32[512, 1536]" = torch.ops.aten.t.default(mm_102);  mm_102 = None
    sum_61: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_872, [0], True);  view_872 = None
    view_873: "f32[1536]" = torch.ops.aten.view.default(sum_61, [1536]);  sum_61 = None
    t_299: "f32[1536, 512]" = torch.ops.aten.t.default(t_298);  t_298 = None
    view_874: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_101, [32, 49, 512]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_875: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_874, [32, 7, 7, 512]);  view_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_876: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_875, [8, 2, 2, 7, 7, 512]);  view_875 = None
    permute_148: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_876, [0, 1, 3, 2, 4, 5]);  view_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_308: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    _unsafe_view_143: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_308, [8, 14, 14, 512]);  clone_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_35: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_143, [0, 0, 0, 0, 0, 0]);  _unsafe_view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_25 = torch.ops.aten.native_layer_norm_backward.default(constant_pad_nd_35, view_281, [512], getitem_118, getitem_119, primals_179, primals_180, [True, True, True]);  constant_pad_nd_35 = view_281 = getitem_118 = getitem_119 = primals_179 = primals_180 = None
    getitem_306: "f32[8, 14, 14, 512]" = native_layer_norm_backward_25[0]
    getitem_307: "f32[512]" = native_layer_norm_backward_25[1]
    getitem_308: "f32[512]" = native_layer_norm_backward_25[2];  native_layer_norm_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_106: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_859, getitem_306);  view_859 = getitem_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_877: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_106, [8, 196, 512]);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_106: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_877, div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_878: "f32[1568, 512]" = torch.ops.aten.view.default(mul_106, [1568, 512]);  mul_106 = None
    mm_103: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_878, t_300);  t_300 = None
    t_301: "f32[512, 1568]" = torch.ops.aten.t.default(view_878)
    mm_104: "f32[512, 2048]" = torch.ops.aten.mm.default(t_301, view_279);  t_301 = view_279 = None
    t_302: "f32[2048, 512]" = torch.ops.aten.t.default(mm_104);  mm_104 = None
    sum_62: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_878, [0], True);  view_878 = None
    view_879: "f32[512]" = torch.ops.aten.view.default(sum_62, [512]);  sum_62 = None
    t_303: "f32[512, 2048]" = torch.ops.aten.t.default(t_302);  t_302 = None
    view_880: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_103, [8, 196, 2048]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_12: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_880, view_278);  view_880 = view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_881: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_12, [1568, 2048]);  gelu_backward_12 = None
    mm_105: "f32[1568, 512]" = torch.ops.aten.mm.default(view_881, t_304);  t_304 = None
    t_305: "f32[2048, 1568]" = torch.ops.aten.t.default(view_881)
    mm_106: "f32[2048, 512]" = torch.ops.aten.mm.default(t_305, view_277);  t_305 = view_277 = None
    t_306: "f32[512, 2048]" = torch.ops.aten.t.default(mm_106);  mm_106 = None
    sum_63: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_881, [0], True);  view_881 = None
    view_882: "f32[2048]" = torch.ops.aten.view.default(sum_63, [2048]);  sum_63 = None
    t_307: "f32[2048, 512]" = torch.ops.aten.t.default(t_306);  t_306 = None
    view_883: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_105, [8, 196, 512]);  mm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_26 = torch.ops.aten.native_layer_norm_backward.default(view_883, view_276, [512], getitem_115, getitem_116, primals_173, primals_174, [True, True, True]);  view_883 = view_276 = getitem_115 = getitem_116 = primals_173 = primals_174 = None
    getitem_309: "f32[8, 196, 512]" = native_layer_norm_backward_26[0]
    getitem_310: "f32[512]" = native_layer_norm_backward_26[1]
    getitem_311: "f32[512]" = native_layer_norm_backward_26[2];  native_layer_norm_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_107: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_877, getitem_309);  view_877 = getitem_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_884: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_107, [8, 14, 14, 512]);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_107: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_884, div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_32: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_107, [-3, -3], [2, 1]);  mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_24: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(roll_32, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  roll_32 = None
    slice_backward_25: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_24, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_885: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_25, [8, 2, 7, 2, 7, 512]);  slice_backward_25 = None
    permute_149: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_885, [0, 1, 3, 2, 4, 5]);  view_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_309: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
    _unsafe_view_144: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_309, [32, 7, 7, 512]);  clone_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_886: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_144, [32, 49, 512]);  _unsafe_view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_887: "f32[1568, 512]" = torch.ops.aten.view.default(view_886, [1568, 512]);  view_886 = None
    mm_107: "f32[1568, 512]" = torch.ops.aten.mm.default(view_887, t_308);  t_308 = None
    t_309: "f32[512, 1568]" = torch.ops.aten.t.default(view_887)
    mm_108: "f32[512, 512]" = torch.ops.aten.mm.default(t_309, view_271);  t_309 = view_271 = None
    t_310: "f32[512, 512]" = torch.ops.aten.t.default(mm_108);  mm_108 = None
    sum_64: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_887, [0], True);  view_887 = None
    view_888: "f32[512]" = torch.ops.aten.view.default(sum_64, [512]);  sum_64 = None
    t_311: "f32[512, 512]" = torch.ops.aten.t.default(t_310);  t_310 = None
    view_889: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_107, [32, 49, 512]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_890: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_889, [32, 49, 16, 32]);  view_889 = None
    transpose_120: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_890, 1, 2);  view_890 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_310: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_120, memory_format = torch.contiguous_format);  transpose_120 = None
    _unsafe_view_145: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_310, [512, 49, 32]);  clone_310 = None
    bmm_96: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_121, _unsafe_view_145);  transpose_121 = None
    bmm_97: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_145, transpose_122);  _unsafe_view_145 = transpose_122 = None
    view_891: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_96, [32, 16, 49, 32]);  bmm_96 = None
    view_892: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_97, [32, 16, 49, 49]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_12: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_892, detach_36, -1, torch.float32);  view_892 = detach_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_893: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_12, [8, 4, 16, 49, 49]);  _softmax_backward_data_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_894: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(view_893, [32, 16, 49, 49]);  view_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_65: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_894, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_12: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_65, 0);  sum_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_150: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_12, [1, 2, 0]);  squeeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_895: "f32[2401, 16]" = torch.ops.aten.view.default(permute_150, [2401, 16]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_12: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_895, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_12: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_12, [view_265], view_895, True);  new_zeros_12 = view_265 = view_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_896: "f32[512, 49, 49]" = torch.ops.aten.view.default(view_894, [512, 49, 49]);  view_894 = None
    bmm_98: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_123, view_896);  transpose_123 = None
    bmm_99: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_896, transpose_124);  view_896 = transpose_124 = None
    view_897: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_98, [32, 16, 32, 49]);  bmm_98 = None
    view_898: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_99, [32, 16, 49, 32]);  bmm_99 = None
    transpose_125: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_897, -2, -1);  view_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_108: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_898, 0.1767766952966369);  view_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_12: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_108, transpose_125, view_891]);  mul_108 = transpose_125 = view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_151: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_12, [1, 3, 0, 2, 4]);  stack_12 = None
    clone_311: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_151, memory_format = torch.contiguous_format);  permute_151 = None
    _unsafe_view_146: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_311, [32, 49, 1536]);  clone_311 = None
    view_899: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_146, [1568, 1536]);  _unsafe_view_146 = None
    mm_109: "f32[1568, 512]" = torch.ops.aten.mm.default(view_899, t_312);  t_312 = None
    t_313: "f32[1536, 1568]" = torch.ops.aten.t.default(view_899)
    mm_110: "f32[1536, 512]" = torch.ops.aten.mm.default(t_313, view_261);  t_313 = view_261 = None
    t_314: "f32[512, 1536]" = torch.ops.aten.t.default(mm_110);  mm_110 = None
    sum_66: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_899, [0], True);  view_899 = None
    view_900: "f32[1536]" = torch.ops.aten.view.default(sum_66, [1536]);  sum_66 = None
    t_315: "f32[1536, 512]" = torch.ops.aten.t.default(t_314);  t_314 = None
    view_901: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_109, [32, 49, 512]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_902: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_901, [32, 7, 7, 512]);  view_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_903: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_902, [8, 2, 2, 7, 7, 512]);  view_902 = None
    permute_152: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_903, [0, 1, 3, 2, 4, 5]);  view_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_312: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
    _unsafe_view_147: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_312, [8, 14, 14, 512]);  clone_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_36: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_147, [0, 0, 0, 0, 0, 0]);  _unsafe_view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_33: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(constant_pad_nd_36, [3, 3], [2, 1]);  constant_pad_nd_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_27 = torch.ops.aten.native_layer_norm_backward.default(roll_33, view_257, [512], getitem_109, getitem_110, primals_167, primals_168, [True, True, True]);  roll_33 = view_257 = getitem_109 = getitem_110 = primals_167 = primals_168 = None
    getitem_312: "f32[8, 14, 14, 512]" = native_layer_norm_backward_27[0]
    getitem_313: "f32[512]" = native_layer_norm_backward_27[1]
    getitem_314: "f32[512]" = native_layer_norm_backward_27[2];  native_layer_norm_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_108: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_884, getitem_312);  view_884 = getitem_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_904: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_108, [8, 196, 512]);  add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_109: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_904, div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_905: "f32[1568, 512]" = torch.ops.aten.view.default(mul_109, [1568, 512]);  mul_109 = None
    mm_111: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_905, t_316);  t_316 = None
    t_317: "f32[512, 1568]" = torch.ops.aten.t.default(view_905)
    mm_112: "f32[512, 2048]" = torch.ops.aten.mm.default(t_317, view_255);  t_317 = view_255 = None
    t_318: "f32[2048, 512]" = torch.ops.aten.t.default(mm_112);  mm_112 = None
    sum_67: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_905, [0], True);  view_905 = None
    view_906: "f32[512]" = torch.ops.aten.view.default(sum_67, [512]);  sum_67 = None
    t_319: "f32[512, 2048]" = torch.ops.aten.t.default(t_318);  t_318 = None
    view_907: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_111, [8, 196, 2048]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_13: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_907, view_254);  view_907 = view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_908: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_13, [1568, 2048]);  gelu_backward_13 = None
    mm_113: "f32[1568, 512]" = torch.ops.aten.mm.default(view_908, t_320);  t_320 = None
    t_321: "f32[2048, 1568]" = torch.ops.aten.t.default(view_908)
    mm_114: "f32[2048, 512]" = torch.ops.aten.mm.default(t_321, view_253);  t_321 = view_253 = None
    t_322: "f32[512, 2048]" = torch.ops.aten.t.default(mm_114);  mm_114 = None
    sum_68: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_908, [0], True);  view_908 = None
    view_909: "f32[2048]" = torch.ops.aten.view.default(sum_68, [2048]);  sum_68 = None
    t_323: "f32[2048, 512]" = torch.ops.aten.t.default(t_322);  t_322 = None
    view_910: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_113, [8, 196, 512]);  mm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_28 = torch.ops.aten.native_layer_norm_backward.default(view_910, view_252, [512], getitem_106, getitem_107, primals_161, primals_162, [True, True, True]);  view_910 = view_252 = getitem_106 = getitem_107 = primals_161 = primals_162 = None
    getitem_315: "f32[8, 196, 512]" = native_layer_norm_backward_28[0]
    getitem_316: "f32[512]" = native_layer_norm_backward_28[1]
    getitem_317: "f32[512]" = native_layer_norm_backward_28[2];  native_layer_norm_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_109: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_904, getitem_315);  view_904 = getitem_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_911: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_109, [8, 14, 14, 512]);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_110: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_911, div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_26: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(mul_110, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  mul_110 = None
    slice_backward_27: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_26, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_912: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_27, [8, 2, 7, 2, 7, 512]);  slice_backward_27 = None
    permute_153: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_912, [0, 1, 3, 2, 4, 5]);  view_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_313: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
    _unsafe_view_148: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_313, [32, 7, 7, 512]);  clone_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_913: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_148, [32, 49, 512]);  _unsafe_view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_914: "f32[1568, 512]" = torch.ops.aten.view.default(view_913, [1568, 512]);  view_913 = None
    mm_115: "f32[1568, 512]" = torch.ops.aten.mm.default(view_914, t_324);  t_324 = None
    t_325: "f32[512, 1568]" = torch.ops.aten.t.default(view_914)
    mm_116: "f32[512, 512]" = torch.ops.aten.mm.default(t_325, view_247);  t_325 = view_247 = None
    t_326: "f32[512, 512]" = torch.ops.aten.t.default(mm_116);  mm_116 = None
    sum_69: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_914, [0], True);  view_914 = None
    view_915: "f32[512]" = torch.ops.aten.view.default(sum_69, [512]);  sum_69 = None
    t_327: "f32[512, 512]" = torch.ops.aten.t.default(t_326);  t_326 = None
    view_916: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_115, [32, 49, 512]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_917: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_916, [32, 49, 16, 32]);  view_916 = None
    transpose_126: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_917, 1, 2);  view_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_314: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_126, memory_format = torch.contiguous_format);  transpose_126 = None
    _unsafe_view_149: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_314, [512, 49, 32]);  clone_314 = None
    bmm_100: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_127, _unsafe_view_149);  transpose_127 = None
    bmm_101: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_149, transpose_128);  _unsafe_view_149 = transpose_128 = None
    view_918: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_100, [32, 16, 49, 32]);  bmm_100 = None
    view_919: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_101, [32, 16, 49, 49]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_13: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_919, detach_37, -1, torch.float32);  view_919 = detach_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_70: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(_softmax_backward_data_13, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_13: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_70, 0);  sum_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_154: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_13, [1, 2, 0]);  squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_920: "f32[2401, 16]" = torch.ops.aten.view.default(permute_154, [2401, 16]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_13: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_920, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_13: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_13, [view_243], view_920, True);  new_zeros_13 = view_243 = view_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_921: "f32[512, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_13, [512, 49, 49]);  _softmax_backward_data_13 = None
    bmm_102: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_129, view_921);  transpose_129 = None
    bmm_103: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_921, transpose_130);  view_921 = transpose_130 = None
    view_922: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_102, [32, 16, 32, 49]);  bmm_102 = None
    view_923: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_103, [32, 16, 49, 32]);  bmm_103 = None
    transpose_131: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_922, -2, -1);  view_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_111: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_923, 0.1767766952966369);  view_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_13: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_111, transpose_131, view_918]);  mul_111 = transpose_131 = view_918 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_155: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_13, [1, 3, 0, 2, 4]);  stack_13 = None
    clone_315: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
    _unsafe_view_150: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_315, [32, 49, 1536]);  clone_315 = None
    view_924: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_150, [1568, 1536]);  _unsafe_view_150 = None
    mm_117: "f32[1568, 512]" = torch.ops.aten.mm.default(view_924, t_328);  t_328 = None
    t_329: "f32[1536, 1568]" = torch.ops.aten.t.default(view_924)
    mm_118: "f32[1536, 512]" = torch.ops.aten.mm.default(t_329, view_239);  t_329 = view_239 = None
    t_330: "f32[512, 1536]" = torch.ops.aten.t.default(mm_118);  mm_118 = None
    sum_71: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_924, [0], True);  view_924 = None
    view_925: "f32[1536]" = torch.ops.aten.view.default(sum_71, [1536]);  sum_71 = None
    t_331: "f32[1536, 512]" = torch.ops.aten.t.default(t_330);  t_330 = None
    view_926: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_117, [32, 49, 512]);  mm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_927: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_926, [32, 7, 7, 512]);  view_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_928: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_927, [8, 2, 2, 7, 7, 512]);  view_927 = None
    permute_156: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_928, [0, 1, 3, 2, 4, 5]);  view_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_316: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    _unsafe_view_151: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_316, [8, 14, 14, 512]);  clone_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_37: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_151, [0, 0, 0, 0, 0, 0]);  _unsafe_view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_29 = torch.ops.aten.native_layer_norm_backward.default(constant_pad_nd_37, view_235, [512], getitem_100, getitem_101, primals_155, primals_156, [True, True, True]);  constant_pad_nd_37 = view_235 = getitem_100 = getitem_101 = primals_155 = primals_156 = None
    getitem_318: "f32[8, 14, 14, 512]" = native_layer_norm_backward_29[0]
    getitem_319: "f32[512]" = native_layer_norm_backward_29[1]
    getitem_320: "f32[512]" = native_layer_norm_backward_29[2];  native_layer_norm_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_110: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_911, getitem_318);  view_911 = getitem_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_929: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_110, [8, 196, 512]);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_112: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_929, div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_930: "f32[1568, 512]" = torch.ops.aten.view.default(mul_112, [1568, 512]);  mul_112 = None
    mm_119: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_930, t_332);  t_332 = None
    t_333: "f32[512, 1568]" = torch.ops.aten.t.default(view_930)
    mm_120: "f32[512, 2048]" = torch.ops.aten.mm.default(t_333, view_233);  t_333 = view_233 = None
    t_334: "f32[2048, 512]" = torch.ops.aten.t.default(mm_120);  mm_120 = None
    sum_72: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_930, [0], True);  view_930 = None
    view_931: "f32[512]" = torch.ops.aten.view.default(sum_72, [512]);  sum_72 = None
    t_335: "f32[512, 2048]" = torch.ops.aten.t.default(t_334);  t_334 = None
    view_932: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_119, [8, 196, 2048]);  mm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_14: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_932, view_232);  view_932 = view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_933: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_14, [1568, 2048]);  gelu_backward_14 = None
    mm_121: "f32[1568, 512]" = torch.ops.aten.mm.default(view_933, t_336);  t_336 = None
    t_337: "f32[2048, 1568]" = torch.ops.aten.t.default(view_933)
    mm_122: "f32[2048, 512]" = torch.ops.aten.mm.default(t_337, view_231);  t_337 = view_231 = None
    t_338: "f32[512, 2048]" = torch.ops.aten.t.default(mm_122);  mm_122 = None
    sum_73: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_933, [0], True);  view_933 = None
    view_934: "f32[2048]" = torch.ops.aten.view.default(sum_73, [2048]);  sum_73 = None
    t_339: "f32[2048, 512]" = torch.ops.aten.t.default(t_338);  t_338 = None
    view_935: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_121, [8, 196, 512]);  mm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_30 = torch.ops.aten.native_layer_norm_backward.default(view_935, view_230, [512], getitem_97, getitem_98, primals_149, primals_150, [True, True, True]);  view_935 = view_230 = getitem_97 = getitem_98 = primals_149 = primals_150 = None
    getitem_321: "f32[8, 196, 512]" = native_layer_norm_backward_30[0]
    getitem_322: "f32[512]" = native_layer_norm_backward_30[1]
    getitem_323: "f32[512]" = native_layer_norm_backward_30[2];  native_layer_norm_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_111: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_929, getitem_321);  view_929 = getitem_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_936: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_111, [8, 14, 14, 512]);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_113: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_936, div_16);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_34: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_113, [-3, -3], [2, 1]);  mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_28: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(roll_34, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  roll_34 = None
    slice_backward_29: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_28, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_937: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_29, [8, 2, 7, 2, 7, 512]);  slice_backward_29 = None
    permute_157: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_937, [0, 1, 3, 2, 4, 5]);  view_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_317: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    _unsafe_view_152: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_317, [32, 7, 7, 512]);  clone_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_938: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_152, [32, 49, 512]);  _unsafe_view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_939: "f32[1568, 512]" = torch.ops.aten.view.default(view_938, [1568, 512]);  view_938 = None
    mm_123: "f32[1568, 512]" = torch.ops.aten.mm.default(view_939, t_340);  t_340 = None
    t_341: "f32[512, 1568]" = torch.ops.aten.t.default(view_939)
    mm_124: "f32[512, 512]" = torch.ops.aten.mm.default(t_341, view_225);  t_341 = view_225 = None
    t_342: "f32[512, 512]" = torch.ops.aten.t.default(mm_124);  mm_124 = None
    sum_74: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_939, [0], True);  view_939 = None
    view_940: "f32[512]" = torch.ops.aten.view.default(sum_74, [512]);  sum_74 = None
    t_343: "f32[512, 512]" = torch.ops.aten.t.default(t_342);  t_342 = None
    view_941: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_123, [32, 49, 512]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_942: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_941, [32, 49, 16, 32]);  view_941 = None
    transpose_132: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_942, 1, 2);  view_942 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_318: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_132, memory_format = torch.contiguous_format);  transpose_132 = None
    _unsafe_view_153: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_318, [512, 49, 32]);  clone_318 = None
    bmm_104: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_133, _unsafe_view_153);  transpose_133 = None
    bmm_105: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_153, transpose_134);  _unsafe_view_153 = transpose_134 = None
    view_943: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_104, [32, 16, 49, 32]);  bmm_104 = None
    view_944: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_105, [32, 16, 49, 49]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_14: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_944, detach_38, -1, torch.float32);  view_944 = detach_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_945: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_14, [8, 4, 16, 49, 49]);  _softmax_backward_data_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_946: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(view_945, [32, 16, 49, 49]);  view_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_75: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_946, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_14: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_75, 0);  sum_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_158: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_14, [1, 2, 0]);  squeeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_947: "f32[2401, 16]" = torch.ops.aten.view.default(permute_158, [2401, 16]);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_14: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_947, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_14: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_14, [view_219], view_947, True);  new_zeros_14 = view_219 = view_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_948: "f32[512, 49, 49]" = torch.ops.aten.view.default(view_946, [512, 49, 49]);  view_946 = None
    bmm_106: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_135, view_948);  transpose_135 = None
    bmm_107: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_948, transpose_136);  view_948 = transpose_136 = None
    view_949: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_106, [32, 16, 32, 49]);  bmm_106 = None
    view_950: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_107, [32, 16, 49, 32]);  bmm_107 = None
    transpose_137: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_949, -2, -1);  view_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_114: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_950, 0.1767766952966369);  view_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_14: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_114, transpose_137, view_943]);  mul_114 = transpose_137 = view_943 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_159: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_14, [1, 3, 0, 2, 4]);  stack_14 = None
    clone_319: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    _unsafe_view_154: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_319, [32, 49, 1536]);  clone_319 = None
    view_951: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_154, [1568, 1536]);  _unsafe_view_154 = None
    mm_125: "f32[1568, 512]" = torch.ops.aten.mm.default(view_951, t_344);  t_344 = None
    t_345: "f32[1536, 1568]" = torch.ops.aten.t.default(view_951)
    mm_126: "f32[1536, 512]" = torch.ops.aten.mm.default(t_345, view_215);  t_345 = view_215 = None
    t_346: "f32[512, 1536]" = torch.ops.aten.t.default(mm_126);  mm_126 = None
    sum_76: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_951, [0], True);  view_951 = None
    view_952: "f32[1536]" = torch.ops.aten.view.default(sum_76, [1536]);  sum_76 = None
    t_347: "f32[1536, 512]" = torch.ops.aten.t.default(t_346);  t_346 = None
    view_953: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_125, [32, 49, 512]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_954: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_953, [32, 7, 7, 512]);  view_953 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_955: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_954, [8, 2, 2, 7, 7, 512]);  view_954 = None
    permute_160: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_955, [0, 1, 3, 2, 4, 5]);  view_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_320: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    _unsafe_view_155: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_320, [8, 14, 14, 512]);  clone_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_38: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_155, [0, 0, 0, 0, 0, 0]);  _unsafe_view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_35: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(constant_pad_nd_38, [3, 3], [2, 1]);  constant_pad_nd_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_31 = torch.ops.aten.native_layer_norm_backward.default(roll_35, view_211, [512], getitem_91, getitem_92, primals_143, primals_144, [True, True, True]);  roll_35 = view_211 = getitem_91 = getitem_92 = primals_143 = primals_144 = None
    getitem_324: "f32[8, 14, 14, 512]" = native_layer_norm_backward_31[0]
    getitem_325: "f32[512]" = native_layer_norm_backward_31[1]
    getitem_326: "f32[512]" = native_layer_norm_backward_31[2];  native_layer_norm_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_112: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_936, getitem_324);  view_936 = getitem_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_956: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_112, [8, 196, 512]);  add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_115: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_956, div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_957: "f32[1568, 512]" = torch.ops.aten.view.default(mul_115, [1568, 512]);  mul_115 = None
    mm_127: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_957, t_348);  t_348 = None
    t_349: "f32[512, 1568]" = torch.ops.aten.t.default(view_957)
    mm_128: "f32[512, 2048]" = torch.ops.aten.mm.default(t_349, view_209);  t_349 = view_209 = None
    t_350: "f32[2048, 512]" = torch.ops.aten.t.default(mm_128);  mm_128 = None
    sum_77: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_957, [0], True);  view_957 = None
    view_958: "f32[512]" = torch.ops.aten.view.default(sum_77, [512]);  sum_77 = None
    t_351: "f32[512, 2048]" = torch.ops.aten.t.default(t_350);  t_350 = None
    view_959: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_127, [8, 196, 2048]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_15: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_959, view_208);  view_959 = view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_960: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_15, [1568, 2048]);  gelu_backward_15 = None
    mm_129: "f32[1568, 512]" = torch.ops.aten.mm.default(view_960, t_352);  t_352 = None
    t_353: "f32[2048, 1568]" = torch.ops.aten.t.default(view_960)
    mm_130: "f32[2048, 512]" = torch.ops.aten.mm.default(t_353, view_207);  t_353 = view_207 = None
    t_354: "f32[512, 2048]" = torch.ops.aten.t.default(mm_130);  mm_130 = None
    sum_78: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_960, [0], True);  view_960 = None
    view_961: "f32[2048]" = torch.ops.aten.view.default(sum_78, [2048]);  sum_78 = None
    t_355: "f32[2048, 512]" = torch.ops.aten.t.default(t_354);  t_354 = None
    view_962: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_129, [8, 196, 512]);  mm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_32 = torch.ops.aten.native_layer_norm_backward.default(view_962, view_206, [512], getitem_88, getitem_89, primals_137, primals_138, [True, True, True]);  view_962 = view_206 = getitem_88 = getitem_89 = primals_137 = primals_138 = None
    getitem_327: "f32[8, 196, 512]" = native_layer_norm_backward_32[0]
    getitem_328: "f32[512]" = native_layer_norm_backward_32[1]
    getitem_329: "f32[512]" = native_layer_norm_backward_32[2];  native_layer_norm_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_113: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_956, getitem_327);  view_956 = getitem_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_963: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_113, [8, 14, 14, 512]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_116: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_963, div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_30: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(mul_116, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  mul_116 = None
    slice_backward_31: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_30, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_964: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_31, [8, 2, 7, 2, 7, 512]);  slice_backward_31 = None
    permute_161: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_964, [0, 1, 3, 2, 4, 5]);  view_964 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_321: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    _unsafe_view_156: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_321, [32, 7, 7, 512]);  clone_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_965: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_156, [32, 49, 512]);  _unsafe_view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_966: "f32[1568, 512]" = torch.ops.aten.view.default(view_965, [1568, 512]);  view_965 = None
    mm_131: "f32[1568, 512]" = torch.ops.aten.mm.default(view_966, t_356);  t_356 = None
    t_357: "f32[512, 1568]" = torch.ops.aten.t.default(view_966)
    mm_132: "f32[512, 512]" = torch.ops.aten.mm.default(t_357, view_201);  t_357 = view_201 = None
    t_358: "f32[512, 512]" = torch.ops.aten.t.default(mm_132);  mm_132 = None
    sum_79: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_966, [0], True);  view_966 = None
    view_967: "f32[512]" = torch.ops.aten.view.default(sum_79, [512]);  sum_79 = None
    t_359: "f32[512, 512]" = torch.ops.aten.t.default(t_358);  t_358 = None
    view_968: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_131, [32, 49, 512]);  mm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_969: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_968, [32, 49, 16, 32]);  view_968 = None
    transpose_138: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_969, 1, 2);  view_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_322: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_138, memory_format = torch.contiguous_format);  transpose_138 = None
    _unsafe_view_157: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_322, [512, 49, 32]);  clone_322 = None
    bmm_108: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_139, _unsafe_view_157);  transpose_139 = None
    bmm_109: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_157, transpose_140);  _unsafe_view_157 = transpose_140 = None
    view_970: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_108, [32, 16, 49, 32]);  bmm_108 = None
    view_971: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_109, [32, 16, 49, 49]);  bmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_15: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_971, detach_39, -1, torch.float32);  view_971 = detach_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_80: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(_softmax_backward_data_15, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_15: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_80, 0);  sum_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_162: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_15, [1, 2, 0]);  squeeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_972: "f32[2401, 16]" = torch.ops.aten.view.default(permute_162, [2401, 16]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_15: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_972, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_15: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_15, [view_197], view_972, True);  new_zeros_15 = view_197 = view_972 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_973: "f32[512, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_15, [512, 49, 49]);  _softmax_backward_data_15 = None
    bmm_110: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_141, view_973);  transpose_141 = None
    bmm_111: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_973, transpose_142);  view_973 = transpose_142 = None
    view_974: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_110, [32, 16, 32, 49]);  bmm_110 = None
    view_975: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_111, [32, 16, 49, 32]);  bmm_111 = None
    transpose_143: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_974, -2, -1);  view_974 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_117: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_975, 0.1767766952966369);  view_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_15: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_117, transpose_143, view_970]);  mul_117 = transpose_143 = view_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_163: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_15, [1, 3, 0, 2, 4]);  stack_15 = None
    clone_323: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    _unsafe_view_158: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_323, [32, 49, 1536]);  clone_323 = None
    view_976: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_158, [1568, 1536]);  _unsafe_view_158 = None
    mm_133: "f32[1568, 512]" = torch.ops.aten.mm.default(view_976, t_360);  t_360 = None
    t_361: "f32[1536, 1568]" = torch.ops.aten.t.default(view_976)
    mm_134: "f32[1536, 512]" = torch.ops.aten.mm.default(t_361, view_193);  t_361 = view_193 = None
    t_362: "f32[512, 1536]" = torch.ops.aten.t.default(mm_134);  mm_134 = None
    sum_81: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_976, [0], True);  view_976 = None
    view_977: "f32[1536]" = torch.ops.aten.view.default(sum_81, [1536]);  sum_81 = None
    t_363: "f32[1536, 512]" = torch.ops.aten.t.default(t_362);  t_362 = None
    view_978: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_133, [32, 49, 512]);  mm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_979: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_978, [32, 7, 7, 512]);  view_978 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_980: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_979, [8, 2, 2, 7, 7, 512]);  view_979 = None
    permute_164: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_980, [0, 1, 3, 2, 4, 5]);  view_980 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_324: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    _unsafe_view_159: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_324, [8, 14, 14, 512]);  clone_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_39: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_159, [0, 0, 0, 0, 0, 0]);  _unsafe_view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_33 = torch.ops.aten.native_layer_norm_backward.default(constant_pad_nd_39, view_189, [512], getitem_82, getitem_83, primals_131, primals_132, [True, True, True]);  constant_pad_nd_39 = view_189 = getitem_82 = getitem_83 = primals_131 = primals_132 = None
    getitem_330: "f32[8, 14, 14, 512]" = native_layer_norm_backward_33[0]
    getitem_331: "f32[512]" = native_layer_norm_backward_33[1]
    getitem_332: "f32[512]" = native_layer_norm_backward_33[2];  native_layer_norm_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_114: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_963, getitem_330);  view_963 = getitem_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_981: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_114, [8, 196, 512]);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_118: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_981, div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_982: "f32[1568, 512]" = torch.ops.aten.view.default(mul_118, [1568, 512]);  mul_118 = None
    mm_135: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_982, t_364);  t_364 = None
    t_365: "f32[512, 1568]" = torch.ops.aten.t.default(view_982)
    mm_136: "f32[512, 2048]" = torch.ops.aten.mm.default(t_365, view_187);  t_365 = view_187 = None
    t_366: "f32[2048, 512]" = torch.ops.aten.t.default(mm_136);  mm_136 = None
    sum_82: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_982, [0], True);  view_982 = None
    view_983: "f32[512]" = torch.ops.aten.view.default(sum_82, [512]);  sum_82 = None
    t_367: "f32[512, 2048]" = torch.ops.aten.t.default(t_366);  t_366 = None
    view_984: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_135, [8, 196, 2048]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_16: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_984, view_186);  view_984 = view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_985: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_16, [1568, 2048]);  gelu_backward_16 = None
    mm_137: "f32[1568, 512]" = torch.ops.aten.mm.default(view_985, t_368);  t_368 = None
    t_369: "f32[2048, 1568]" = torch.ops.aten.t.default(view_985)
    mm_138: "f32[2048, 512]" = torch.ops.aten.mm.default(t_369, view_185);  t_369 = view_185 = None
    t_370: "f32[512, 2048]" = torch.ops.aten.t.default(mm_138);  mm_138 = None
    sum_83: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_985, [0], True);  view_985 = None
    view_986: "f32[2048]" = torch.ops.aten.view.default(sum_83, [2048]);  sum_83 = None
    t_371: "f32[2048, 512]" = torch.ops.aten.t.default(t_370);  t_370 = None
    view_987: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_137, [8, 196, 512]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_34 = torch.ops.aten.native_layer_norm_backward.default(view_987, view_184, [512], getitem_79, getitem_80, primals_125, primals_126, [True, True, True]);  view_987 = view_184 = getitem_79 = getitem_80 = primals_125 = primals_126 = None
    getitem_333: "f32[8, 196, 512]" = native_layer_norm_backward_34[0]
    getitem_334: "f32[512]" = native_layer_norm_backward_34[1]
    getitem_335: "f32[512]" = native_layer_norm_backward_34[2];  native_layer_norm_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_115: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_981, getitem_333);  view_981 = getitem_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_988: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_115, [8, 14, 14, 512]);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_119: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_988, div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_36: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_119, [-3, -3], [2, 1]);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_32: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(roll_36, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  roll_36 = None
    slice_backward_33: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_32, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_989: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_33, [8, 2, 7, 2, 7, 512]);  slice_backward_33 = None
    permute_165: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_989, [0, 1, 3, 2, 4, 5]);  view_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_325: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
    _unsafe_view_160: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_325, [32, 7, 7, 512]);  clone_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_990: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_160, [32, 49, 512]);  _unsafe_view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_991: "f32[1568, 512]" = torch.ops.aten.view.default(view_990, [1568, 512]);  view_990 = None
    mm_139: "f32[1568, 512]" = torch.ops.aten.mm.default(view_991, t_372);  t_372 = None
    t_373: "f32[512, 1568]" = torch.ops.aten.t.default(view_991)
    mm_140: "f32[512, 512]" = torch.ops.aten.mm.default(t_373, view_179);  t_373 = view_179 = None
    t_374: "f32[512, 512]" = torch.ops.aten.t.default(mm_140);  mm_140 = None
    sum_84: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_991, [0], True);  view_991 = None
    view_992: "f32[512]" = torch.ops.aten.view.default(sum_84, [512]);  sum_84 = None
    t_375: "f32[512, 512]" = torch.ops.aten.t.default(t_374);  t_374 = None
    view_993: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_139, [32, 49, 512]);  mm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_994: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_993, [32, 49, 16, 32]);  view_993 = None
    transpose_144: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_994, 1, 2);  view_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_326: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_144, memory_format = torch.contiguous_format);  transpose_144 = None
    _unsafe_view_161: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_326, [512, 49, 32]);  clone_326 = None
    bmm_112: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_145, _unsafe_view_161);  transpose_145 = None
    bmm_113: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_161, transpose_146);  _unsafe_view_161 = transpose_146 = None
    view_995: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_112, [32, 16, 49, 32]);  bmm_112 = None
    view_996: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_113, [32, 16, 49, 49]);  bmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_16: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_996, detach_40, -1, torch.float32);  view_996 = detach_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_997: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_16, [8, 4, 16, 49, 49]);  _softmax_backward_data_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_998: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(view_997, [32, 16, 49, 49]);  view_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_85: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_998, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_16: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_85, 0);  sum_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_166: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_16, [1, 2, 0]);  squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_999: "f32[2401, 16]" = torch.ops.aten.view.default(permute_166, [2401, 16]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_16: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_999, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_16: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_16, [view_173], view_999, True);  new_zeros_16 = view_173 = view_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1000: "f32[512, 49, 49]" = torch.ops.aten.view.default(view_998, [512, 49, 49]);  view_998 = None
    bmm_114: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_147, view_1000);  transpose_147 = None
    bmm_115: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1000, transpose_148);  view_1000 = transpose_148 = None
    view_1001: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_114, [32, 16, 32, 49]);  bmm_114 = None
    view_1002: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_115, [32, 16, 49, 32]);  bmm_115 = None
    transpose_149: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_1001, -2, -1);  view_1001 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_120: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1002, 0.1767766952966369);  view_1002 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_16: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_120, transpose_149, view_995]);  mul_120 = transpose_149 = view_995 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_167: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_16, [1, 3, 0, 2, 4]);  stack_16 = None
    clone_327: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
    _unsafe_view_162: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_327, [32, 49, 1536]);  clone_327 = None
    view_1003: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_162, [1568, 1536]);  _unsafe_view_162 = None
    mm_141: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1003, t_376);  t_376 = None
    t_377: "f32[1536, 1568]" = torch.ops.aten.t.default(view_1003)
    mm_142: "f32[1536, 512]" = torch.ops.aten.mm.default(t_377, view_169);  t_377 = view_169 = None
    t_378: "f32[512, 1536]" = torch.ops.aten.t.default(mm_142);  mm_142 = None
    sum_86: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1003, [0], True);  view_1003 = None
    view_1004: "f32[1536]" = torch.ops.aten.view.default(sum_86, [1536]);  sum_86 = None
    t_379: "f32[1536, 512]" = torch.ops.aten.t.default(t_378);  t_378 = None
    view_1005: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_141, [32, 49, 512]);  mm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1006: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1005, [32, 7, 7, 512]);  view_1005 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1007: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1006, [8, 2, 2, 7, 7, 512]);  view_1006 = None
    permute_168: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1007, [0, 1, 3, 2, 4, 5]);  view_1007 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_328: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    _unsafe_view_163: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_328, [8, 14, 14, 512]);  clone_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_40: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_163, [0, 0, 0, 0, 0, 0]);  _unsafe_view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_37: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(constant_pad_nd_40, [3, 3], [2, 1]);  constant_pad_nd_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_35 = torch.ops.aten.native_layer_norm_backward.default(roll_37, view_165, [512], getitem_73, getitem_74, primals_119, primals_120, [True, True, True]);  roll_37 = view_165 = getitem_73 = getitem_74 = primals_119 = primals_120 = None
    getitem_336: "f32[8, 14, 14, 512]" = native_layer_norm_backward_35[0]
    getitem_337: "f32[512]" = native_layer_norm_backward_35[1]
    getitem_338: "f32[512]" = native_layer_norm_backward_35[2];  native_layer_norm_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_116: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_988, getitem_336);  view_988 = getitem_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1008: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_116, [8, 196, 512]);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_121: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1008, div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1009: "f32[1568, 512]" = torch.ops.aten.view.default(mul_121, [1568, 512]);  mul_121 = None
    mm_143: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_1009, t_380);  t_380 = None
    t_381: "f32[512, 1568]" = torch.ops.aten.t.default(view_1009)
    mm_144: "f32[512, 2048]" = torch.ops.aten.mm.default(t_381, view_163);  t_381 = view_163 = None
    t_382: "f32[2048, 512]" = torch.ops.aten.t.default(mm_144);  mm_144 = None
    sum_87: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1009, [0], True);  view_1009 = None
    view_1010: "f32[512]" = torch.ops.aten.view.default(sum_87, [512]);  sum_87 = None
    t_383: "f32[512, 2048]" = torch.ops.aten.t.default(t_382);  t_382 = None
    view_1011: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_143, [8, 196, 2048]);  mm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_17: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_1011, view_162);  view_1011 = view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1012: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_17, [1568, 2048]);  gelu_backward_17 = None
    mm_145: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1012, t_384);  t_384 = None
    t_385: "f32[2048, 1568]" = torch.ops.aten.t.default(view_1012)
    mm_146: "f32[2048, 512]" = torch.ops.aten.mm.default(t_385, view_161);  t_385 = view_161 = None
    t_386: "f32[512, 2048]" = torch.ops.aten.t.default(mm_146);  mm_146 = None
    sum_88: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1012, [0], True);  view_1012 = None
    view_1013: "f32[2048]" = torch.ops.aten.view.default(sum_88, [2048]);  sum_88 = None
    t_387: "f32[2048, 512]" = torch.ops.aten.t.default(t_386);  t_386 = None
    view_1014: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_145, [8, 196, 512]);  mm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_36 = torch.ops.aten.native_layer_norm_backward.default(view_1014, view_160, [512], getitem_70, getitem_71, primals_113, primals_114, [True, True, True]);  view_1014 = view_160 = getitem_70 = getitem_71 = primals_113 = primals_114 = None
    getitem_339: "f32[8, 196, 512]" = native_layer_norm_backward_36[0]
    getitem_340: "f32[512]" = native_layer_norm_backward_36[1]
    getitem_341: "f32[512]" = native_layer_norm_backward_36[2];  native_layer_norm_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_117: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1008, getitem_339);  view_1008 = getitem_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1015: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_117, [8, 14, 14, 512]);  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_122: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1015, div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_34: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(mul_122, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  mul_122 = None
    slice_backward_35: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_34, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1016: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_35, [8, 2, 7, 2, 7, 512]);  slice_backward_35 = None
    permute_169: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1016, [0, 1, 3, 2, 4, 5]);  view_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_329: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
    _unsafe_view_164: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_329, [32, 7, 7, 512]);  clone_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1017: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_164, [32, 49, 512]);  _unsafe_view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1018: "f32[1568, 512]" = torch.ops.aten.view.default(view_1017, [1568, 512]);  view_1017 = None
    mm_147: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1018, t_388);  t_388 = None
    t_389: "f32[512, 1568]" = torch.ops.aten.t.default(view_1018)
    mm_148: "f32[512, 512]" = torch.ops.aten.mm.default(t_389, view_155);  t_389 = view_155 = None
    t_390: "f32[512, 512]" = torch.ops.aten.t.default(mm_148);  mm_148 = None
    sum_89: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1018, [0], True);  view_1018 = None
    view_1019: "f32[512]" = torch.ops.aten.view.default(sum_89, [512]);  sum_89 = None
    t_391: "f32[512, 512]" = torch.ops.aten.t.default(t_390);  t_390 = None
    view_1020: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_147, [32, 49, 512]);  mm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1021: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_1020, [32, 49, 16, 32]);  view_1020 = None
    transpose_150: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_1021, 1, 2);  view_1021 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_330: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_150, memory_format = torch.contiguous_format);  transpose_150 = None
    _unsafe_view_165: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_330, [512, 49, 32]);  clone_330 = None
    bmm_116: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_151, _unsafe_view_165);  transpose_151 = None
    bmm_117: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_165, transpose_152);  _unsafe_view_165 = transpose_152 = None
    view_1022: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_116, [32, 16, 49, 32]);  bmm_116 = None
    view_1023: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_117, [32, 16, 49, 49]);  bmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_17: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_1023, detach_41, -1, torch.float32);  view_1023 = detach_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_90: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(_softmax_backward_data_17, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_17: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_90, 0);  sum_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_170: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_17, [1, 2, 0]);  squeeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1024: "f32[2401, 16]" = torch.ops.aten.view.default(permute_170, [2401, 16]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_17: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_1024, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_17: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_17, [view_151], view_1024, True);  new_zeros_17 = view_151 = view_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1025: "f32[512, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_17, [512, 49, 49]);  _softmax_backward_data_17 = None
    bmm_118: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_153, view_1025);  transpose_153 = None
    bmm_119: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1025, transpose_154);  view_1025 = transpose_154 = None
    view_1026: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_118, [32, 16, 32, 49]);  bmm_118 = None
    view_1027: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_119, [32, 16, 49, 32]);  bmm_119 = None
    transpose_155: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_1026, -2, -1);  view_1026 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_123: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1027, 0.1767766952966369);  view_1027 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_17: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_123, transpose_155, view_1022]);  mul_123 = transpose_155 = view_1022 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_171: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_17, [1, 3, 0, 2, 4]);  stack_17 = None
    clone_331: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    _unsafe_view_166: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_331, [32, 49, 1536]);  clone_331 = None
    view_1028: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_166, [1568, 1536]);  _unsafe_view_166 = None
    mm_149: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1028, t_392);  t_392 = None
    t_393: "f32[1536, 1568]" = torch.ops.aten.t.default(view_1028)
    mm_150: "f32[1536, 512]" = torch.ops.aten.mm.default(t_393, view_147);  t_393 = view_147 = None
    t_394: "f32[512, 1536]" = torch.ops.aten.t.default(mm_150);  mm_150 = None
    sum_91: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1028, [0], True);  view_1028 = None
    view_1029: "f32[1536]" = torch.ops.aten.view.default(sum_91, [1536]);  sum_91 = None
    t_395: "f32[1536, 512]" = torch.ops.aten.t.default(t_394);  t_394 = None
    view_1030: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_149, [32, 49, 512]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1031: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1030, [32, 7, 7, 512]);  view_1030 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1032: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1031, [8, 2, 2, 7, 7, 512]);  view_1031 = None
    permute_172: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1032, [0, 1, 3, 2, 4, 5]);  view_1032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_332: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    _unsafe_view_167: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_332, [8, 14, 14, 512]);  clone_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_41: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_167, [0, 0, 0, 0, 0, 0]);  _unsafe_view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_37 = torch.ops.aten.native_layer_norm_backward.default(constant_pad_nd_41, view_143, [512], getitem_64, getitem_65, primals_107, primals_108, [True, True, True]);  constant_pad_nd_41 = view_143 = getitem_64 = getitem_65 = primals_107 = primals_108 = None
    getitem_342: "f32[8, 14, 14, 512]" = native_layer_norm_backward_37[0]
    getitem_343: "f32[512]" = native_layer_norm_backward_37[1]
    getitem_344: "f32[512]" = native_layer_norm_backward_37[2];  native_layer_norm_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_118: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1015, getitem_342);  view_1015 = getitem_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1033: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_118, [8, 196, 512]);  add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_124: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1033, div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1034: "f32[1568, 512]" = torch.ops.aten.view.default(mul_124, [1568, 512]);  mul_124 = None
    mm_151: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_1034, t_396);  t_396 = None
    t_397: "f32[512, 1568]" = torch.ops.aten.t.default(view_1034)
    mm_152: "f32[512, 2048]" = torch.ops.aten.mm.default(t_397, view_141);  t_397 = view_141 = None
    t_398: "f32[2048, 512]" = torch.ops.aten.t.default(mm_152);  mm_152 = None
    sum_92: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1034, [0], True);  view_1034 = None
    view_1035: "f32[512]" = torch.ops.aten.view.default(sum_92, [512]);  sum_92 = None
    t_399: "f32[512, 2048]" = torch.ops.aten.t.default(t_398);  t_398 = None
    view_1036: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_151, [8, 196, 2048]);  mm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_18: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_1036, view_140);  view_1036 = view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1037: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_18, [1568, 2048]);  gelu_backward_18 = None
    mm_153: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1037, t_400);  t_400 = None
    t_401: "f32[2048, 1568]" = torch.ops.aten.t.default(view_1037)
    mm_154: "f32[2048, 512]" = torch.ops.aten.mm.default(t_401, view_139);  t_401 = view_139 = None
    t_402: "f32[512, 2048]" = torch.ops.aten.t.default(mm_154);  mm_154 = None
    sum_93: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True);  view_1037 = None
    view_1038: "f32[2048]" = torch.ops.aten.view.default(sum_93, [2048]);  sum_93 = None
    t_403: "f32[2048, 512]" = torch.ops.aten.t.default(t_402);  t_402 = None
    view_1039: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_153, [8, 196, 512]);  mm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_38 = torch.ops.aten.native_layer_norm_backward.default(view_1039, view_138, [512], getitem_61, getitem_62, primals_101, primals_102, [True, True, True]);  view_1039 = view_138 = getitem_61 = getitem_62 = primals_101 = primals_102 = None
    getitem_345: "f32[8, 196, 512]" = native_layer_norm_backward_38[0]
    getitem_346: "f32[512]" = native_layer_norm_backward_38[1]
    getitem_347: "f32[512]" = native_layer_norm_backward_38[2];  native_layer_norm_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_119: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1033, getitem_345);  view_1033 = getitem_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1040: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_119, [8, 14, 14, 512]);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_125: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1040, div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_38: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_125, [-3, -3], [2, 1]);  mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_36: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(roll_38, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  roll_38 = None
    slice_backward_37: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_36, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1041: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_37, [8, 2, 7, 2, 7, 512]);  slice_backward_37 = None
    permute_173: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1041, [0, 1, 3, 2, 4, 5]);  view_1041 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_333: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
    _unsafe_view_168: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_333, [32, 7, 7, 512]);  clone_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1042: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_168, [32, 49, 512]);  _unsafe_view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1043: "f32[1568, 512]" = torch.ops.aten.view.default(view_1042, [1568, 512]);  view_1042 = None
    mm_155: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1043, t_404);  t_404 = None
    t_405: "f32[512, 1568]" = torch.ops.aten.t.default(view_1043)
    mm_156: "f32[512, 512]" = torch.ops.aten.mm.default(t_405, view_133);  t_405 = view_133 = None
    t_406: "f32[512, 512]" = torch.ops.aten.t.default(mm_156);  mm_156 = None
    sum_94: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1043, [0], True);  view_1043 = None
    view_1044: "f32[512]" = torch.ops.aten.view.default(sum_94, [512]);  sum_94 = None
    t_407: "f32[512, 512]" = torch.ops.aten.t.default(t_406);  t_406 = None
    view_1045: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_155, [32, 49, 512]);  mm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1046: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_1045, [32, 49, 16, 32]);  view_1045 = None
    transpose_156: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_1046, 1, 2);  view_1046 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_334: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_156, memory_format = torch.contiguous_format);  transpose_156 = None
    _unsafe_view_169: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_334, [512, 49, 32]);  clone_334 = None
    bmm_120: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_157, _unsafe_view_169);  transpose_157 = None
    bmm_121: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_169, transpose_158);  _unsafe_view_169 = transpose_158 = None
    view_1047: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_120, [32, 16, 49, 32]);  bmm_120 = None
    view_1048: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_121, [32, 16, 49, 49]);  bmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_18: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_1048, detach_42, -1, torch.float32);  view_1048 = detach_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_1049: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_18, [8, 4, 16, 49, 49]);  _softmax_backward_data_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_1050: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(view_1049, [32, 16, 49, 49]);  view_1049 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_95: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_1050, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_18: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_95, 0);  sum_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_174: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_18, [1, 2, 0]);  squeeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1051: "f32[2401, 16]" = torch.ops.aten.view.default(permute_174, [2401, 16]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_18: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_1051, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_18: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_18, [view_127], view_1051, True);  new_zeros_18 = view_127 = view_1051 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1052: "f32[512, 49, 49]" = torch.ops.aten.view.default(view_1050, [512, 49, 49]);  view_1050 = None
    bmm_122: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_159, view_1052);  transpose_159 = None
    bmm_123: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1052, transpose_160);  view_1052 = transpose_160 = None
    view_1053: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_122, [32, 16, 32, 49]);  bmm_122 = None
    view_1054: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_123, [32, 16, 49, 32]);  bmm_123 = None
    transpose_161: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_1053, -2, -1);  view_1053 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_126: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1054, 0.1767766952966369);  view_1054 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_18: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_126, transpose_161, view_1047]);  mul_126 = transpose_161 = view_1047 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_175: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_18, [1, 3, 0, 2, 4]);  stack_18 = None
    clone_335: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
    _unsafe_view_170: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_335, [32, 49, 1536]);  clone_335 = None
    view_1055: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_170, [1568, 1536]);  _unsafe_view_170 = None
    mm_157: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1055, t_408);  t_408 = None
    t_409: "f32[1536, 1568]" = torch.ops.aten.t.default(view_1055)
    mm_158: "f32[1536, 512]" = torch.ops.aten.mm.default(t_409, view_123);  t_409 = view_123 = None
    t_410: "f32[512, 1536]" = torch.ops.aten.t.default(mm_158);  mm_158 = None
    sum_96: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1055, [0], True);  view_1055 = None
    view_1056: "f32[1536]" = torch.ops.aten.view.default(sum_96, [1536]);  sum_96 = None
    t_411: "f32[1536, 512]" = torch.ops.aten.t.default(t_410);  t_410 = None
    view_1057: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_157, [32, 49, 512]);  mm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1058: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1057, [32, 7, 7, 512]);  view_1057 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1059: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1058, [8, 2, 2, 7, 7, 512]);  view_1058 = None
    permute_176: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1059, [0, 1, 3, 2, 4, 5]);  view_1059 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_336: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
    _unsafe_view_171: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_336, [8, 14, 14, 512]);  clone_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_42: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_171, [0, 0, 0, 0, 0, 0]);  _unsafe_view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_39: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(constant_pad_nd_42, [3, 3], [2, 1]);  constant_pad_nd_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_39 = torch.ops.aten.native_layer_norm_backward.default(roll_39, view_119, [512], getitem_55, getitem_56, primals_95, primals_96, [True, True, True]);  roll_39 = view_119 = getitem_55 = getitem_56 = primals_95 = primals_96 = None
    getitem_348: "f32[8, 14, 14, 512]" = native_layer_norm_backward_39[0]
    getitem_349: "f32[512]" = native_layer_norm_backward_39[1]
    getitem_350: "f32[512]" = native_layer_norm_backward_39[2];  native_layer_norm_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_120: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1040, getitem_348);  view_1040 = getitem_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1060: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_120, [8, 196, 512]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_127: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1060, div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1061: "f32[1568, 512]" = torch.ops.aten.view.default(mul_127, [1568, 512]);  mul_127 = None
    mm_159: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_1061, t_412);  t_412 = None
    t_413: "f32[512, 1568]" = torch.ops.aten.t.default(view_1061)
    mm_160: "f32[512, 2048]" = torch.ops.aten.mm.default(t_413, view_117);  t_413 = view_117 = None
    t_414: "f32[2048, 512]" = torch.ops.aten.t.default(mm_160);  mm_160 = None
    sum_97: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1061, [0], True);  view_1061 = None
    view_1062: "f32[512]" = torch.ops.aten.view.default(sum_97, [512]);  sum_97 = None
    t_415: "f32[512, 2048]" = torch.ops.aten.t.default(t_414);  t_414 = None
    view_1063: "f32[8, 196, 2048]" = torch.ops.aten.view.default(mm_159, [8, 196, 2048]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_19: "f32[8, 196, 2048]" = torch.ops.aten.gelu_backward.default(view_1063, view_116);  view_1063 = view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1064: "f32[1568, 2048]" = torch.ops.aten.view.default(gelu_backward_19, [1568, 2048]);  gelu_backward_19 = None
    mm_161: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1064, t_416);  t_416 = None
    t_417: "f32[2048, 1568]" = torch.ops.aten.t.default(view_1064)
    mm_162: "f32[2048, 512]" = torch.ops.aten.mm.default(t_417, view_115);  t_417 = view_115 = None
    t_418: "f32[512, 2048]" = torch.ops.aten.t.default(mm_162);  mm_162 = None
    sum_98: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1064, [0], True);  view_1064 = None
    view_1065: "f32[2048]" = torch.ops.aten.view.default(sum_98, [2048]);  sum_98 = None
    t_419: "f32[2048, 512]" = torch.ops.aten.t.default(t_418);  t_418 = None
    view_1066: "f32[8, 196, 512]" = torch.ops.aten.view.default(mm_161, [8, 196, 512]);  mm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_40 = torch.ops.aten.native_layer_norm_backward.default(view_1066, view_114, [512], getitem_52, getitem_53, primals_89, primals_90, [True, True, True]);  view_1066 = view_114 = getitem_52 = getitem_53 = primals_89 = primals_90 = None
    getitem_351: "f32[8, 196, 512]" = native_layer_norm_backward_40[0]
    getitem_352: "f32[512]" = native_layer_norm_backward_40[1]
    getitem_353: "f32[512]" = native_layer_norm_backward_40[2];  native_layer_norm_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_121: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1060, getitem_351);  view_1060 = getitem_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1067: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_121, [8, 14, 14, 512]);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_128: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1067, div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_38: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(mul_128, [8, 14, 14, 512], 3, 0, 9223372036854775807, 1);  mul_128 = None
    slice_backward_39: "f32[8, 14, 14, 512]" = torch.ops.aten.slice_backward.default(slice_backward_38, [8, 14, 14, 512], 0, 0, 9223372036854775807, 1);  slice_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1068: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(slice_backward_39, [8, 2, 7, 2, 7, 512]);  slice_backward_39 = None
    permute_177: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1068, [0, 1, 3, 2, 4, 5]);  view_1068 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_337: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    _unsafe_view_172: "f32[32, 7, 7, 512]" = torch.ops.aten._unsafe_view.default(clone_337, [32, 7, 7, 512]);  clone_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1069: "f32[32, 49, 512]" = torch.ops.aten.view.default(_unsafe_view_172, [32, 49, 512]);  _unsafe_view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1070: "f32[1568, 512]" = torch.ops.aten.view.default(view_1069, [1568, 512]);  view_1069 = None
    mm_163: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1070, t_420);  t_420 = None
    t_421: "f32[512, 1568]" = torch.ops.aten.t.default(view_1070)
    mm_164: "f32[512, 512]" = torch.ops.aten.mm.default(t_421, view_109);  t_421 = view_109 = None
    t_422: "f32[512, 512]" = torch.ops.aten.t.default(mm_164);  mm_164 = None
    sum_99: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1070, [0], True);  view_1070 = None
    view_1071: "f32[512]" = torch.ops.aten.view.default(sum_99, [512]);  sum_99 = None
    t_423: "f32[512, 512]" = torch.ops.aten.t.default(t_422);  t_422 = None
    view_1072: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_163, [32, 49, 512]);  mm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1073: "f32[32, 49, 16, 32]" = torch.ops.aten.view.default(view_1072, [32, 49, 16, 32]);  view_1072 = None
    transpose_162: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_1073, 1, 2);  view_1073 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_338: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(transpose_162, memory_format = torch.contiguous_format);  transpose_162 = None
    _unsafe_view_173: "f32[512, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_338, [512, 49, 32]);  clone_338 = None
    bmm_124: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(transpose_163, _unsafe_view_173);  transpose_163 = None
    bmm_125: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_173, transpose_164);  _unsafe_view_173 = transpose_164 = None
    view_1074: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_124, [32, 16, 49, 32]);  bmm_124 = None
    view_1075: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_125, [32, 16, 49, 49]);  bmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_19: "f32[32, 16, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_1075, detach_43, -1, torch.float32);  view_1075 = detach_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_100: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(_softmax_backward_data_19, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_19: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_100, 0);  sum_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_178: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_19, [1, 2, 0]);  squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1076: "f32[2401, 16]" = torch.ops.aten.view.default(permute_178, [2401, 16]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_19: "f32[169, 16]" = torch.ops.aten.new_zeros.default(view_1076, [169, 16], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_19: "f32[169, 16]" = torch.ops.aten.index_put.default(new_zeros_19, [view_105], view_1076, True);  new_zeros_19 = view_105 = view_1076 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1077: "f32[512, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_19, [512, 49, 49]);  _softmax_backward_data_19 = None
    bmm_126: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(transpose_165, view_1077);  transpose_165 = None
    bmm_127: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1077, transpose_166);  view_1077 = transpose_166 = None
    view_1078: "f32[32, 16, 32, 49]" = torch.ops.aten.view.default(bmm_126, [32, 16, 32, 49]);  bmm_126 = None
    view_1079: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_127, [32, 16, 49, 32]);  bmm_127 = None
    transpose_167: "f32[32, 16, 49, 32]" = torch.ops.aten.transpose.int(view_1078, -2, -1);  view_1078 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_129: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1079, 0.1767766952966369);  view_1079 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_19: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.stack.default([mul_129, transpose_167, view_1074]);  mul_129 = transpose_167 = view_1074 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_179: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(stack_19, [1, 3, 0, 2, 4]);  stack_19 = None
    clone_339: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    _unsafe_view_174: "f32[32, 49, 1536]" = torch.ops.aten._unsafe_view.default(clone_339, [32, 49, 1536]);  clone_339 = None
    view_1080: "f32[1568, 1536]" = torch.ops.aten.view.default(_unsafe_view_174, [1568, 1536]);  _unsafe_view_174 = None
    mm_165: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1080, t_424);  t_424 = None
    t_425: "f32[1536, 1568]" = torch.ops.aten.t.default(view_1080)
    mm_166: "f32[1536, 512]" = torch.ops.aten.mm.default(t_425, view_101);  t_425 = view_101 = None
    t_426: "f32[512, 1536]" = torch.ops.aten.t.default(mm_166);  mm_166 = None
    sum_101: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1080, [0], True);  view_1080 = None
    view_1081: "f32[1536]" = torch.ops.aten.view.default(sum_101, [1536]);  sum_101 = None
    t_427: "f32[1536, 512]" = torch.ops.aten.t.default(t_426);  t_426 = None
    view_1082: "f32[32, 49, 512]" = torch.ops.aten.view.default(mm_165, [32, 49, 512]);  mm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1083: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1082, [32, 7, 7, 512]);  view_1082 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1084: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1083, [8, 2, 2, 7, 7, 512]);  view_1083 = None
    permute_180: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1084, [0, 1, 3, 2, 4, 5]);  view_1084 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_340: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
    _unsafe_view_175: "f32[8, 14, 14, 512]" = torch.ops.aten._unsafe_view.default(clone_340, [8, 14, 14, 512]);  clone_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_43: "f32[8, 14, 14, 512]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_175, [0, 0, 0, 0, 0, 0]);  _unsafe_view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_41 = torch.ops.aten.native_layer_norm_backward.default(constant_pad_nd_43, view_97, [512], getitem_46, getitem_47, primals_83, primals_84, [True, True, True]);  constant_pad_nd_43 = view_97 = getitem_46 = getitem_47 = primals_83 = primals_84 = None
    getitem_354: "f32[8, 14, 14, 512]" = native_layer_norm_backward_41[0]
    getitem_355: "f32[512]" = native_layer_norm_backward_41[1]
    getitem_356: "f32[512]" = native_layer_norm_backward_41[2];  native_layer_norm_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_122: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1067, getitem_354);  view_1067 = getitem_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    view_1085: "f32[1568, 512]" = torch.ops.aten.view.default(add_122, [1568, 512]);  add_122 = None
    t_428: "f32[512, 1568]" = torch.ops.aten.t.default(view_1085)
    mm_167: "f32[512, 1024]" = torch.ops.aten.mm.default(t_428, view_96);  t_428 = view_96 = None
    t_429: "f32[1024, 512]" = torch.ops.aten.t.default(mm_167);  mm_167 = None
    mm_168: "f32[1568, 1024]" = torch.ops.aten.mm.default(view_1085, t_430);  view_1085 = t_430 = None
    view_1086: "f32[8, 14, 14, 1024]" = torch.ops.aten.view.default(mm_168, [8, 14, 14, 1024]);  mm_168 = None
    t_431: "f32[512, 1024]" = torch.ops.aten.t.default(t_429);  t_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:357, code: x = self.norm(x)
    native_layer_norm_backward_42 = torch.ops.aten.native_layer_norm_backward.default(view_1086, _unsafe_view_17, [1024], getitem_43, getitem_44, primals_80, primals_81, [True, True, True]);  view_1086 = _unsafe_view_17 = getitem_43 = getitem_44 = primals_80 = primals_81 = None
    getitem_357: "f32[8, 14, 14, 1024]" = native_layer_norm_backward_42[0]
    getitem_358: "f32[1024]" = native_layer_norm_backward_42[1]
    getitem_359: "f32[1024]" = native_layer_norm_backward_42[2];  native_layer_norm_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:356, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    view_1087: "f32[8, 14, 14, 2, 2, 256]" = torch.ops.aten.view.default(getitem_357, [8, 14, 14, 2, 2, 256]);  getitem_357 = None
    permute_181: "f32[8, 14, 2, 14, 2, 256]" = torch.ops.aten.permute.default(view_1087, [0, 1, 4, 2, 3, 5]);  view_1087 = None
    clone_341: "f32[8, 14, 2, 14, 2, 256]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    _unsafe_view_176: "f32[8, 28, 28, 256]" = torch.ops.aten._unsafe_view.default(clone_341, [8, 28, 28, 256]);  clone_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1088: "f32[8, 784, 256]" = torch.ops.aten.view.default(_unsafe_view_176, [8, 784, 256]);  _unsafe_view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_130: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(view_1088, div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1089: "f32[6272, 256]" = torch.ops.aten.view.default(mul_130, [6272, 256]);  mul_130 = None
    mm_169: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_1089, t_432);  t_432 = None
    t_433: "f32[256, 6272]" = torch.ops.aten.t.default(view_1089)
    mm_170: "f32[256, 1024]" = torch.ops.aten.mm.default(t_433, view_92);  t_433 = view_92 = None
    t_434: "f32[1024, 256]" = torch.ops.aten.t.default(mm_170);  mm_170 = None
    sum_102: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1089, [0], True);  view_1089 = None
    view_1090: "f32[256]" = torch.ops.aten.view.default(sum_102, [256]);  sum_102 = None
    t_435: "f32[256, 1024]" = torch.ops.aten.t.default(t_434);  t_434 = None
    view_1091: "f32[8, 784, 1024]" = torch.ops.aten.view.default(mm_169, [8, 784, 1024]);  mm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_20: "f32[8, 784, 1024]" = torch.ops.aten.gelu_backward.default(view_1091, view_91);  view_1091 = view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1092: "f32[6272, 1024]" = torch.ops.aten.view.default(gelu_backward_20, [6272, 1024]);  gelu_backward_20 = None
    mm_171: "f32[6272, 256]" = torch.ops.aten.mm.default(view_1092, t_436);  t_436 = None
    t_437: "f32[1024, 6272]" = torch.ops.aten.t.default(view_1092)
    mm_172: "f32[1024, 256]" = torch.ops.aten.mm.default(t_437, view_90);  t_437 = view_90 = None
    t_438: "f32[256, 1024]" = torch.ops.aten.t.default(mm_172);  mm_172 = None
    sum_103: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1092, [0], True);  view_1092 = None
    view_1093: "f32[1024]" = torch.ops.aten.view.default(sum_103, [1024]);  sum_103 = None
    t_439: "f32[1024, 256]" = torch.ops.aten.t.default(t_438);  t_438 = None
    view_1094: "f32[8, 784, 256]" = torch.ops.aten.view.default(mm_171, [8, 784, 256]);  mm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_43 = torch.ops.aten.native_layer_norm_backward.default(view_1094, view_89, [256], getitem_40, getitem_41, primals_74, primals_75, [True, True, True]);  view_1094 = view_89 = getitem_40 = getitem_41 = primals_74 = primals_75 = None
    getitem_360: "f32[8, 784, 256]" = native_layer_norm_backward_43[0]
    getitem_361: "f32[256]" = native_layer_norm_backward_43[1]
    getitem_362: "f32[256]" = native_layer_norm_backward_43[2];  native_layer_norm_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_123: "f32[8, 784, 256]" = torch.ops.aten.add.Tensor(view_1088, getitem_360);  view_1088 = getitem_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1095: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(add_123, [8, 28, 28, 256]);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_131: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_1095, div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_40: "f32[8, 28, 28, 256]" = torch.ops.aten.roll.default(mul_131, [-3, -3], [2, 1]);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_40: "f32[8, 28, 28, 256]" = torch.ops.aten.slice_backward.default(roll_40, [8, 28, 28, 256], 3, 0, 9223372036854775807, 1);  roll_40 = None
    slice_backward_41: "f32[8, 28, 28, 256]" = torch.ops.aten.slice_backward.default(slice_backward_40, [8, 28, 28, 256], 0, 0, 9223372036854775807, 1);  slice_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1096: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(slice_backward_41, [8, 4, 7, 4, 7, 256]);  slice_backward_41 = None
    permute_182: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_1096, [0, 1, 3, 2, 4, 5]);  view_1096 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_342: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
    _unsafe_view_177: "f32[128, 7, 7, 256]" = torch.ops.aten._unsafe_view.default(clone_342, [128, 7, 7, 256]);  clone_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1097: "f32[128, 49, 256]" = torch.ops.aten.view.default(_unsafe_view_177, [128, 49, 256]);  _unsafe_view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1098: "f32[6272, 256]" = torch.ops.aten.view.default(view_1097, [6272, 256]);  view_1097 = None
    mm_173: "f32[6272, 256]" = torch.ops.aten.mm.default(view_1098, t_440);  t_440 = None
    t_441: "f32[256, 6272]" = torch.ops.aten.t.default(view_1098)
    mm_174: "f32[256, 256]" = torch.ops.aten.mm.default(t_441, view_84);  t_441 = view_84 = None
    t_442: "f32[256, 256]" = torch.ops.aten.t.default(mm_174);  mm_174 = None
    sum_104: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1098, [0], True);  view_1098 = None
    view_1099: "f32[256]" = torch.ops.aten.view.default(sum_104, [256]);  sum_104 = None
    t_443: "f32[256, 256]" = torch.ops.aten.t.default(t_442);  t_442 = None
    view_1100: "f32[128, 49, 256]" = torch.ops.aten.view.default(mm_173, [128, 49, 256]);  mm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1101: "f32[128, 49, 8, 32]" = torch.ops.aten.view.default(view_1100, [128, 49, 8, 32]);  view_1100 = None
    transpose_168: "f32[128, 8, 49, 32]" = torch.ops.aten.transpose.int(view_1101, 1, 2);  view_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_343: "f32[128, 8, 49, 32]" = torch.ops.aten.clone.default(transpose_168, memory_format = torch.contiguous_format);  transpose_168 = None
    _unsafe_view_178: "f32[1024, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_343, [1024, 49, 32]);  clone_343 = None
    bmm_128: "f32[1024, 49, 32]" = torch.ops.aten.bmm.default(transpose_169, _unsafe_view_178);  transpose_169 = None
    bmm_129: "f32[1024, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_178, transpose_170);  _unsafe_view_178 = transpose_170 = None
    view_1102: "f32[128, 8, 49, 32]" = torch.ops.aten.view.default(bmm_128, [128, 8, 49, 32]);  bmm_128 = None
    view_1103: "f32[128, 8, 49, 49]" = torch.ops.aten.view.default(bmm_129, [128, 8, 49, 49]);  bmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_20: "f32[128, 8, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_1103, detach_44, -1, torch.float32);  view_1103 = detach_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_1104: "f32[8, 16, 8, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_20, [8, 16, 8, 49, 49]);  _softmax_backward_data_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_1105: "f32[128, 8, 49, 49]" = torch.ops.aten.view.default(view_1104, [128, 8, 49, 49]);  view_1104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_105: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_1105, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_20: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_105, 0);  sum_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_183: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze_20, [1, 2, 0]);  squeeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1106: "f32[2401, 8]" = torch.ops.aten.view.default(permute_183, [2401, 8]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_20: "f32[169, 8]" = torch.ops.aten.new_zeros.default(view_1106, [169, 8], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_20: "f32[169, 8]" = torch.ops.aten.index_put.default(new_zeros_20, [view_78], view_1106, True);  new_zeros_20 = view_78 = view_1106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1107: "f32[1024, 49, 49]" = torch.ops.aten.view.default(view_1105, [1024, 49, 49]);  view_1105 = None
    bmm_130: "f32[1024, 32, 49]" = torch.ops.aten.bmm.default(transpose_171, view_1107);  transpose_171 = None
    bmm_131: "f32[1024, 49, 32]" = torch.ops.aten.bmm.default(view_1107, transpose_172);  view_1107 = transpose_172 = None
    view_1108: "f32[128, 8, 32, 49]" = torch.ops.aten.view.default(bmm_130, [128, 8, 32, 49]);  bmm_130 = None
    view_1109: "f32[128, 8, 49, 32]" = torch.ops.aten.view.default(bmm_131, [128, 8, 49, 32]);  bmm_131 = None
    transpose_173: "f32[128, 8, 49, 32]" = torch.ops.aten.transpose.int(view_1108, -2, -1);  view_1108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_132: "f32[128, 8, 49, 32]" = torch.ops.aten.mul.Tensor(view_1109, 0.1767766952966369);  view_1109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_20: "f32[3, 128, 8, 49, 32]" = torch.ops.aten.stack.default([mul_132, transpose_173, view_1102]);  mul_132 = transpose_173 = view_1102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_184: "f32[128, 49, 3, 8, 32]" = torch.ops.aten.permute.default(stack_20, [1, 3, 0, 2, 4]);  stack_20 = None
    clone_344: "f32[128, 49, 3, 8, 32]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
    _unsafe_view_179: "f32[128, 49, 768]" = torch.ops.aten._unsafe_view.default(clone_344, [128, 49, 768]);  clone_344 = None
    view_1110: "f32[6272, 768]" = torch.ops.aten.view.default(_unsafe_view_179, [6272, 768]);  _unsafe_view_179 = None
    mm_175: "f32[6272, 256]" = torch.ops.aten.mm.default(view_1110, t_444);  t_444 = None
    t_445: "f32[768, 6272]" = torch.ops.aten.t.default(view_1110)
    mm_176: "f32[768, 256]" = torch.ops.aten.mm.default(t_445, view_74);  t_445 = view_74 = None
    t_446: "f32[256, 768]" = torch.ops.aten.t.default(mm_176);  mm_176 = None
    sum_106: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1110, [0], True);  view_1110 = None
    view_1111: "f32[768]" = torch.ops.aten.view.default(sum_106, [768]);  sum_106 = None
    t_447: "f32[768, 256]" = torch.ops.aten.t.default(t_446);  t_446 = None
    view_1112: "f32[128, 49, 256]" = torch.ops.aten.view.default(mm_175, [128, 49, 256]);  mm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1113: "f32[128, 7, 7, 256]" = torch.ops.aten.view.default(view_1112, [128, 7, 7, 256]);  view_1112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1114: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_1113, [8, 4, 4, 7, 7, 256]);  view_1113 = None
    permute_185: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_1114, [0, 1, 3, 2, 4, 5]);  view_1114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_345: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
    _unsafe_view_180: "f32[8, 28, 28, 256]" = torch.ops.aten._unsafe_view.default(clone_345, [8, 28, 28, 256]);  clone_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_44: "f32[8, 28, 28, 256]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_180, [0, 0, 0, 0, 0, 0]);  _unsafe_view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_41: "f32[8, 28, 28, 256]" = torch.ops.aten.roll.default(constant_pad_nd_44, [3, 3], [2, 1]);  constant_pad_nd_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_44 = torch.ops.aten.native_layer_norm_backward.default(roll_41, view_70, [256], getitem_34, getitem_35, primals_68, primals_69, [True, True, True]);  roll_41 = view_70 = getitem_34 = getitem_35 = primals_68 = primals_69 = None
    getitem_363: "f32[8, 28, 28, 256]" = native_layer_norm_backward_44[0]
    getitem_364: "f32[256]" = native_layer_norm_backward_44[1]
    getitem_365: "f32[256]" = native_layer_norm_backward_44[2];  native_layer_norm_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_124: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_1095, getitem_363);  view_1095 = getitem_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1115: "f32[8, 784, 256]" = torch.ops.aten.view.default(add_124, [8, 784, 256]);  add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_133: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(view_1115, div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1116: "f32[6272, 256]" = torch.ops.aten.view.default(mul_133, [6272, 256]);  mul_133 = None
    mm_177: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_1116, t_448);  t_448 = None
    t_449: "f32[256, 6272]" = torch.ops.aten.t.default(view_1116)
    mm_178: "f32[256, 1024]" = torch.ops.aten.mm.default(t_449, view_68);  t_449 = view_68 = None
    t_450: "f32[1024, 256]" = torch.ops.aten.t.default(mm_178);  mm_178 = None
    sum_107: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1116, [0], True);  view_1116 = None
    view_1117: "f32[256]" = torch.ops.aten.view.default(sum_107, [256]);  sum_107 = None
    t_451: "f32[256, 1024]" = torch.ops.aten.t.default(t_450);  t_450 = None
    view_1118: "f32[8, 784, 1024]" = torch.ops.aten.view.default(mm_177, [8, 784, 1024]);  mm_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_21: "f32[8, 784, 1024]" = torch.ops.aten.gelu_backward.default(view_1118, view_67);  view_1118 = view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1119: "f32[6272, 1024]" = torch.ops.aten.view.default(gelu_backward_21, [6272, 1024]);  gelu_backward_21 = None
    mm_179: "f32[6272, 256]" = torch.ops.aten.mm.default(view_1119, t_452);  t_452 = None
    t_453: "f32[1024, 6272]" = torch.ops.aten.t.default(view_1119)
    mm_180: "f32[1024, 256]" = torch.ops.aten.mm.default(t_453, view_66);  t_453 = view_66 = None
    t_454: "f32[256, 1024]" = torch.ops.aten.t.default(mm_180);  mm_180 = None
    sum_108: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1119, [0], True);  view_1119 = None
    view_1120: "f32[1024]" = torch.ops.aten.view.default(sum_108, [1024]);  sum_108 = None
    t_455: "f32[1024, 256]" = torch.ops.aten.t.default(t_454);  t_454 = None
    view_1121: "f32[8, 784, 256]" = torch.ops.aten.view.default(mm_179, [8, 784, 256]);  mm_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_45 = torch.ops.aten.native_layer_norm_backward.default(view_1121, view_65, [256], getitem_31, getitem_32, primals_62, primals_63, [True, True, True]);  view_1121 = view_65 = getitem_31 = getitem_32 = primals_62 = primals_63 = None
    getitem_366: "f32[8, 784, 256]" = native_layer_norm_backward_45[0]
    getitem_367: "f32[256]" = native_layer_norm_backward_45[1]
    getitem_368: "f32[256]" = native_layer_norm_backward_45[2];  native_layer_norm_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_125: "f32[8, 784, 256]" = torch.ops.aten.add.Tensor(view_1115, getitem_366);  view_1115 = getitem_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1122: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(add_125, [8, 28, 28, 256]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_134: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_1122, div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_42: "f32[8, 28, 28, 256]" = torch.ops.aten.slice_backward.default(mul_134, [8, 28, 28, 256], 3, 0, 9223372036854775807, 1);  mul_134 = None
    slice_backward_43: "f32[8, 28, 28, 256]" = torch.ops.aten.slice_backward.default(slice_backward_42, [8, 28, 28, 256], 0, 0, 9223372036854775807, 1);  slice_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1123: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(slice_backward_43, [8, 4, 7, 4, 7, 256]);  slice_backward_43 = None
    permute_186: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_1123, [0, 1, 3, 2, 4, 5]);  view_1123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_346: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    _unsafe_view_181: "f32[128, 7, 7, 256]" = torch.ops.aten._unsafe_view.default(clone_346, [128, 7, 7, 256]);  clone_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1124: "f32[128, 49, 256]" = torch.ops.aten.view.default(_unsafe_view_181, [128, 49, 256]);  _unsafe_view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1125: "f32[6272, 256]" = torch.ops.aten.view.default(view_1124, [6272, 256]);  view_1124 = None
    mm_181: "f32[6272, 256]" = torch.ops.aten.mm.default(view_1125, t_456);  t_456 = None
    t_457: "f32[256, 6272]" = torch.ops.aten.t.default(view_1125)
    mm_182: "f32[256, 256]" = torch.ops.aten.mm.default(t_457, view_60);  t_457 = view_60 = None
    t_458: "f32[256, 256]" = torch.ops.aten.t.default(mm_182);  mm_182 = None
    sum_109: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1125, [0], True);  view_1125 = None
    view_1126: "f32[256]" = torch.ops.aten.view.default(sum_109, [256]);  sum_109 = None
    t_459: "f32[256, 256]" = torch.ops.aten.t.default(t_458);  t_458 = None
    view_1127: "f32[128, 49, 256]" = torch.ops.aten.view.default(mm_181, [128, 49, 256]);  mm_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1128: "f32[128, 49, 8, 32]" = torch.ops.aten.view.default(view_1127, [128, 49, 8, 32]);  view_1127 = None
    transpose_174: "f32[128, 8, 49, 32]" = torch.ops.aten.transpose.int(view_1128, 1, 2);  view_1128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_347: "f32[128, 8, 49, 32]" = torch.ops.aten.clone.default(transpose_174, memory_format = torch.contiguous_format);  transpose_174 = None
    _unsafe_view_182: "f32[1024, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_347, [1024, 49, 32]);  clone_347 = None
    bmm_132: "f32[1024, 49, 32]" = torch.ops.aten.bmm.default(transpose_175, _unsafe_view_182);  transpose_175 = None
    bmm_133: "f32[1024, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_182, transpose_176);  _unsafe_view_182 = transpose_176 = None
    view_1129: "f32[128, 8, 49, 32]" = torch.ops.aten.view.default(bmm_132, [128, 8, 49, 32]);  bmm_132 = None
    view_1130: "f32[128, 8, 49, 49]" = torch.ops.aten.view.default(bmm_133, [128, 8, 49, 49]);  bmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_21: "f32[128, 8, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_1130, detach_45, -1, torch.float32);  view_1130 = detach_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_110: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(_softmax_backward_data_21, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_21: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_110, 0);  sum_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_187: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze_21, [1, 2, 0]);  squeeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1131: "f32[2401, 8]" = torch.ops.aten.view.default(permute_187, [2401, 8]);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_21: "f32[169, 8]" = torch.ops.aten.new_zeros.default(view_1131, [169, 8], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_21: "f32[169, 8]" = torch.ops.aten.index_put.default(new_zeros_21, [view_56], view_1131, True);  new_zeros_21 = view_56 = view_1131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1132: "f32[1024, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_21, [1024, 49, 49]);  _softmax_backward_data_21 = None
    bmm_134: "f32[1024, 32, 49]" = torch.ops.aten.bmm.default(transpose_177, view_1132);  transpose_177 = None
    bmm_135: "f32[1024, 49, 32]" = torch.ops.aten.bmm.default(view_1132, transpose_178);  view_1132 = transpose_178 = None
    view_1133: "f32[128, 8, 32, 49]" = torch.ops.aten.view.default(bmm_134, [128, 8, 32, 49]);  bmm_134 = None
    view_1134: "f32[128, 8, 49, 32]" = torch.ops.aten.view.default(bmm_135, [128, 8, 49, 32]);  bmm_135 = None
    transpose_179: "f32[128, 8, 49, 32]" = torch.ops.aten.transpose.int(view_1133, -2, -1);  view_1133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_135: "f32[128, 8, 49, 32]" = torch.ops.aten.mul.Tensor(view_1134, 0.1767766952966369);  view_1134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_21: "f32[3, 128, 8, 49, 32]" = torch.ops.aten.stack.default([mul_135, transpose_179, view_1129]);  mul_135 = transpose_179 = view_1129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_188: "f32[128, 49, 3, 8, 32]" = torch.ops.aten.permute.default(stack_21, [1, 3, 0, 2, 4]);  stack_21 = None
    clone_348: "f32[128, 49, 3, 8, 32]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    _unsafe_view_183: "f32[128, 49, 768]" = torch.ops.aten._unsafe_view.default(clone_348, [128, 49, 768]);  clone_348 = None
    view_1135: "f32[6272, 768]" = torch.ops.aten.view.default(_unsafe_view_183, [6272, 768]);  _unsafe_view_183 = None
    mm_183: "f32[6272, 256]" = torch.ops.aten.mm.default(view_1135, t_460);  t_460 = None
    t_461: "f32[768, 6272]" = torch.ops.aten.t.default(view_1135)
    mm_184: "f32[768, 256]" = torch.ops.aten.mm.default(t_461, view_52);  t_461 = view_52 = None
    t_462: "f32[256, 768]" = torch.ops.aten.t.default(mm_184);  mm_184 = None
    sum_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1135, [0], True);  view_1135 = None
    view_1136: "f32[768]" = torch.ops.aten.view.default(sum_111, [768]);  sum_111 = None
    t_463: "f32[768, 256]" = torch.ops.aten.t.default(t_462);  t_462 = None
    view_1137: "f32[128, 49, 256]" = torch.ops.aten.view.default(mm_183, [128, 49, 256]);  mm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1138: "f32[128, 7, 7, 256]" = torch.ops.aten.view.default(view_1137, [128, 7, 7, 256]);  view_1137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1139: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_1138, [8, 4, 4, 7, 7, 256]);  view_1138 = None
    permute_189: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_1139, [0, 1, 3, 2, 4, 5]);  view_1139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_349: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    _unsafe_view_184: "f32[8, 28, 28, 256]" = torch.ops.aten._unsafe_view.default(clone_349, [8, 28, 28, 256]);  clone_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_45: "f32[8, 28, 28, 256]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_184, [0, 0, 0, 0, 0, 0]);  _unsafe_view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_46 = torch.ops.aten.native_layer_norm_backward.default(constant_pad_nd_45, view_48, [256], getitem_25, getitem_26, primals_56, primals_57, [True, True, True]);  constant_pad_nd_45 = view_48 = getitem_25 = getitem_26 = primals_56 = primals_57 = None
    getitem_369: "f32[8, 28, 28, 256]" = native_layer_norm_backward_46[0]
    getitem_370: "f32[256]" = native_layer_norm_backward_46[1]
    getitem_371: "f32[256]" = native_layer_norm_backward_46[2];  native_layer_norm_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_126: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_1122, getitem_369);  view_1122 = getitem_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    view_1140: "f32[6272, 256]" = torch.ops.aten.view.default(add_126, [6272, 256]);  add_126 = None
    t_464: "f32[256, 6272]" = torch.ops.aten.t.default(view_1140)
    mm_185: "f32[256, 512]" = torch.ops.aten.mm.default(t_464, view_47);  t_464 = view_47 = None
    t_465: "f32[512, 256]" = torch.ops.aten.t.default(mm_185);  mm_185 = None
    mm_186: "f32[6272, 512]" = torch.ops.aten.mm.default(view_1140, t_466);  view_1140 = t_466 = None
    view_1141: "f32[8, 28, 28, 512]" = torch.ops.aten.view.default(mm_186, [8, 28, 28, 512]);  mm_186 = None
    t_467: "f32[256, 512]" = torch.ops.aten.t.default(t_465);  t_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:357, code: x = self.norm(x)
    native_layer_norm_backward_47 = torch.ops.aten.native_layer_norm_backward.default(view_1141, _unsafe_view_8, [512], getitem_22, getitem_23, primals_53, primals_54, [True, True, True]);  view_1141 = _unsafe_view_8 = getitem_22 = getitem_23 = primals_53 = primals_54 = None
    getitem_372: "f32[8, 28, 28, 512]" = native_layer_norm_backward_47[0]
    getitem_373: "f32[512]" = native_layer_norm_backward_47[1]
    getitem_374: "f32[512]" = native_layer_norm_backward_47[2];  native_layer_norm_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:356, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    view_1142: "f32[8, 28, 28, 2, 2, 128]" = torch.ops.aten.view.default(getitem_372, [8, 28, 28, 2, 2, 128]);  getitem_372 = None
    permute_190: "f32[8, 28, 2, 28, 2, 128]" = torch.ops.aten.permute.default(view_1142, [0, 1, 4, 2, 3, 5]);  view_1142 = None
    clone_350: "f32[8, 28, 2, 28, 2, 128]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
    _unsafe_view_185: "f32[8, 56, 56, 128]" = torch.ops.aten._unsafe_view.default(clone_350, [8, 56, 56, 128]);  clone_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1143: "f32[8, 3136, 128]" = torch.ops.aten.view.default(_unsafe_view_185, [8, 3136, 128]);  _unsafe_view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_136: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(view_1143, div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1144: "f32[25088, 128]" = torch.ops.aten.view.default(mul_136, [25088, 128]);  mul_136 = None
    mm_187: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1144, t_468);  t_468 = None
    t_469: "f32[128, 25088]" = torch.ops.aten.t.default(view_1144)
    mm_188: "f32[128, 512]" = torch.ops.aten.mm.default(t_469, view_43);  t_469 = view_43 = None
    t_470: "f32[512, 128]" = torch.ops.aten.t.default(mm_188);  mm_188 = None
    sum_112: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1144, [0], True);  view_1144 = None
    view_1145: "f32[128]" = torch.ops.aten.view.default(sum_112, [128]);  sum_112 = None
    t_471: "f32[128, 512]" = torch.ops.aten.t.default(t_470);  t_470 = None
    view_1146: "f32[8, 3136, 512]" = torch.ops.aten.view.default(mm_187, [8, 3136, 512]);  mm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_22: "f32[8, 3136, 512]" = torch.ops.aten.gelu_backward.default(view_1146, view_42);  view_1146 = view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1147: "f32[25088, 512]" = torch.ops.aten.view.default(gelu_backward_22, [25088, 512]);  gelu_backward_22 = None
    mm_189: "f32[25088, 128]" = torch.ops.aten.mm.default(view_1147, t_472);  t_472 = None
    t_473: "f32[512, 25088]" = torch.ops.aten.t.default(view_1147)
    mm_190: "f32[512, 128]" = torch.ops.aten.mm.default(t_473, view_41);  t_473 = view_41 = None
    t_474: "f32[128, 512]" = torch.ops.aten.t.default(mm_190);  mm_190 = None
    sum_113: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1147, [0], True);  view_1147 = None
    view_1148: "f32[512]" = torch.ops.aten.view.default(sum_113, [512]);  sum_113 = None
    t_475: "f32[512, 128]" = torch.ops.aten.t.default(t_474);  t_474 = None
    view_1149: "f32[8, 3136, 128]" = torch.ops.aten.view.default(mm_189, [8, 3136, 128]);  mm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_48 = torch.ops.aten.native_layer_norm_backward.default(view_1149, view_40, [128], getitem_19, getitem_20, primals_47, primals_48, [True, True, True]);  view_1149 = view_40 = getitem_19 = getitem_20 = primals_47 = primals_48 = None
    getitem_375: "f32[8, 3136, 128]" = native_layer_norm_backward_48[0]
    getitem_376: "f32[128]" = native_layer_norm_backward_48[1]
    getitem_377: "f32[128]" = native_layer_norm_backward_48[2];  native_layer_norm_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_127: "f32[8, 3136, 128]" = torch.ops.aten.add.Tensor(view_1143, getitem_375);  view_1143 = getitem_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1150: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(add_127, [8, 56, 56, 128]);  add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_137: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_1150, div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_42: "f32[8, 56, 56, 128]" = torch.ops.aten.roll.default(mul_137, [-3, -3], [2, 1]);  mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_44: "f32[8, 56, 56, 128]" = torch.ops.aten.slice_backward.default(roll_42, [8, 56, 56, 128], 3, 0, 9223372036854775807, 1);  roll_42 = None
    slice_backward_45: "f32[8, 56, 56, 128]" = torch.ops.aten.slice_backward.default(slice_backward_44, [8, 56, 56, 128], 0, 0, 9223372036854775807, 1);  slice_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1151: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(slice_backward_45, [8, 8, 7, 8, 7, 128]);  slice_backward_45 = None
    permute_191: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_1151, [0, 1, 3, 2, 4, 5]);  view_1151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_351: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
    _unsafe_view_186: "f32[512, 7, 7, 128]" = torch.ops.aten._unsafe_view.default(clone_351, [512, 7, 7, 128]);  clone_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1152: "f32[512, 49, 128]" = torch.ops.aten.view.default(_unsafe_view_186, [512, 49, 128]);  _unsafe_view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1153: "f32[25088, 128]" = torch.ops.aten.view.default(view_1152, [25088, 128]);  view_1152 = None
    mm_191: "f32[25088, 128]" = torch.ops.aten.mm.default(view_1153, t_476);  t_476 = None
    t_477: "f32[128, 25088]" = torch.ops.aten.t.default(view_1153)
    mm_192: "f32[128, 128]" = torch.ops.aten.mm.default(t_477, view_35);  t_477 = view_35 = None
    t_478: "f32[128, 128]" = torch.ops.aten.t.default(mm_192);  mm_192 = None
    sum_114: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1153, [0], True);  view_1153 = None
    view_1154: "f32[128]" = torch.ops.aten.view.default(sum_114, [128]);  sum_114 = None
    t_479: "f32[128, 128]" = torch.ops.aten.t.default(t_478);  t_478 = None
    view_1155: "f32[512, 49, 128]" = torch.ops.aten.view.default(mm_191, [512, 49, 128]);  mm_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1156: "f32[512, 49, 4, 32]" = torch.ops.aten.view.default(view_1155, [512, 49, 4, 32]);  view_1155 = None
    transpose_180: "f32[512, 4, 49, 32]" = torch.ops.aten.transpose.int(view_1156, 1, 2);  view_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_352: "f32[512, 4, 49, 32]" = torch.ops.aten.clone.default(transpose_180, memory_format = torch.contiguous_format);  transpose_180 = None
    _unsafe_view_187: "f32[2048, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_352, [2048, 49, 32]);  clone_352 = None
    bmm_136: "f32[2048, 49, 32]" = torch.ops.aten.bmm.default(transpose_181, _unsafe_view_187);  transpose_181 = None
    bmm_137: "f32[2048, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_187, transpose_182);  _unsafe_view_187 = transpose_182 = None
    view_1157: "f32[512, 4, 49, 32]" = torch.ops.aten.view.default(bmm_136, [512, 4, 49, 32]);  bmm_136 = None
    view_1158: "f32[512, 4, 49, 49]" = torch.ops.aten.view.default(bmm_137, [512, 4, 49, 49]);  bmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_22: "f32[512, 4, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_1158, detach_46, -1, torch.float32);  view_1158 = detach_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_1159: "f32[8, 64, 4, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_22, [8, 64, 4, 49, 49]);  _softmax_backward_data_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_1160: "f32[512, 4, 49, 49]" = torch.ops.aten.view.default(view_1159, [512, 4, 49, 49]);  view_1159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_115: "f32[1, 4, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_1160, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_22: "f32[4, 49, 49]" = torch.ops.aten.squeeze.dim(sum_115, 0);  sum_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_192: "f32[49, 49, 4]" = torch.ops.aten.permute.default(squeeze_22, [1, 2, 0]);  squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1161: "f32[2401, 4]" = torch.ops.aten.view.default(permute_192, [2401, 4]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_22: "f32[169, 4]" = torch.ops.aten.new_zeros.default(view_1161, [169, 4], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_22: "f32[169, 4]" = torch.ops.aten.index_put.default(new_zeros_22, [view_29], view_1161, True);  new_zeros_22 = view_29 = view_1161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1162: "f32[2048, 49, 49]" = torch.ops.aten.view.default(view_1160, [2048, 49, 49]);  view_1160 = None
    bmm_138: "f32[2048, 32, 49]" = torch.ops.aten.bmm.default(transpose_183, view_1162);  transpose_183 = None
    bmm_139: "f32[2048, 49, 32]" = torch.ops.aten.bmm.default(view_1162, transpose_184);  view_1162 = transpose_184 = None
    view_1163: "f32[512, 4, 32, 49]" = torch.ops.aten.view.default(bmm_138, [512, 4, 32, 49]);  bmm_138 = None
    view_1164: "f32[512, 4, 49, 32]" = torch.ops.aten.view.default(bmm_139, [512, 4, 49, 32]);  bmm_139 = None
    transpose_185: "f32[512, 4, 49, 32]" = torch.ops.aten.transpose.int(view_1163, -2, -1);  view_1163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_138: "f32[512, 4, 49, 32]" = torch.ops.aten.mul.Tensor(view_1164, 0.1767766952966369);  view_1164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_22: "f32[3, 512, 4, 49, 32]" = torch.ops.aten.stack.default([mul_138, transpose_185, view_1157]);  mul_138 = transpose_185 = view_1157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_193: "f32[512, 49, 3, 4, 32]" = torch.ops.aten.permute.default(stack_22, [1, 3, 0, 2, 4]);  stack_22 = None
    clone_353: "f32[512, 49, 3, 4, 32]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    _unsafe_view_188: "f32[512, 49, 384]" = torch.ops.aten._unsafe_view.default(clone_353, [512, 49, 384]);  clone_353 = None
    view_1165: "f32[25088, 384]" = torch.ops.aten.view.default(_unsafe_view_188, [25088, 384]);  _unsafe_view_188 = None
    mm_193: "f32[25088, 128]" = torch.ops.aten.mm.default(view_1165, t_480);  t_480 = None
    t_481: "f32[384, 25088]" = torch.ops.aten.t.default(view_1165)
    mm_194: "f32[384, 128]" = torch.ops.aten.mm.default(t_481, view_25);  t_481 = view_25 = None
    t_482: "f32[128, 384]" = torch.ops.aten.t.default(mm_194);  mm_194 = None
    sum_116: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1165, [0], True);  view_1165 = None
    view_1166: "f32[384]" = torch.ops.aten.view.default(sum_116, [384]);  sum_116 = None
    t_483: "f32[384, 128]" = torch.ops.aten.t.default(t_482);  t_482 = None
    view_1167: "f32[512, 49, 128]" = torch.ops.aten.view.default(mm_193, [512, 49, 128]);  mm_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1168: "f32[512, 7, 7, 128]" = torch.ops.aten.view.default(view_1167, [512, 7, 7, 128]);  view_1167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1169: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_1168, [8, 8, 8, 7, 7, 128]);  view_1168 = None
    permute_194: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_1169, [0, 1, 3, 2, 4, 5]);  view_1169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_354: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    _unsafe_view_189: "f32[8, 56, 56, 128]" = torch.ops.aten._unsafe_view.default(clone_354, [8, 56, 56, 128]);  clone_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_46: "f32[8, 56, 56, 128]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_189, [0, 0, 0, 0, 0, 0]);  _unsafe_view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_43: "f32[8, 56, 56, 128]" = torch.ops.aten.roll.default(constant_pad_nd_46, [3, 3], [2, 1]);  constant_pad_nd_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_49 = torch.ops.aten.native_layer_norm_backward.default(roll_43, view_21, [128], getitem_13, getitem_14, primals_41, primals_42, [True, True, True]);  roll_43 = view_21 = getitem_13 = getitem_14 = primals_41 = primals_42 = None
    getitem_378: "f32[8, 56, 56, 128]" = native_layer_norm_backward_49[0]
    getitem_379: "f32[128]" = native_layer_norm_backward_49[1]
    getitem_380: "f32[128]" = native_layer_norm_backward_49[2];  native_layer_norm_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_128: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_1150, getitem_378);  view_1150 = getitem_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1170: "f32[8, 3136, 128]" = torch.ops.aten.view.default(add_128, [8, 3136, 128]);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1171: "f32[25088, 128]" = torch.ops.aten.view.default(view_1170, [25088, 128])
    mm_195: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1171, t_484);  t_484 = None
    t_485: "f32[128, 25088]" = torch.ops.aten.t.default(view_1171)
    mm_196: "f32[128, 512]" = torch.ops.aten.mm.default(t_485, view_19);  t_485 = view_19 = None
    t_486: "f32[512, 128]" = torch.ops.aten.t.default(mm_196);  mm_196 = None
    sum_117: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1171, [0], True);  view_1171 = None
    view_1172: "f32[128]" = torch.ops.aten.view.default(sum_117, [128]);  sum_117 = None
    t_487: "f32[128, 512]" = torch.ops.aten.t.default(t_486);  t_486 = None
    view_1173: "f32[8, 3136, 512]" = torch.ops.aten.view.default(mm_195, [8, 3136, 512]);  mm_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_23: "f32[8, 3136, 512]" = torch.ops.aten.gelu_backward.default(view_1173, view_18);  view_1173 = view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1174: "f32[25088, 512]" = torch.ops.aten.view.default(gelu_backward_23, [25088, 512]);  gelu_backward_23 = None
    mm_197: "f32[25088, 128]" = torch.ops.aten.mm.default(view_1174, t_488);  t_488 = None
    t_489: "f32[512, 25088]" = torch.ops.aten.t.default(view_1174)
    mm_198: "f32[512, 128]" = torch.ops.aten.mm.default(t_489, view_17);  t_489 = view_17 = None
    t_490: "f32[128, 512]" = torch.ops.aten.t.default(mm_198);  mm_198 = None
    sum_118: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1174, [0], True);  view_1174 = None
    view_1175: "f32[512]" = torch.ops.aten.view.default(sum_118, [512]);  sum_118 = None
    t_491: "f32[512, 128]" = torch.ops.aten.t.default(t_490);  t_490 = None
    view_1176: "f32[8, 3136, 128]" = torch.ops.aten.view.default(mm_197, [8, 3136, 128]);  mm_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    native_layer_norm_backward_50 = torch.ops.aten.native_layer_norm_backward.default(view_1176, view_16, [128], getitem_10, getitem_11, primals_35, primals_36, [True, True, True]);  view_1176 = view_16 = getitem_10 = getitem_11 = primals_35 = primals_36 = None
    getitem_381: "f32[8, 3136, 128]" = native_layer_norm_backward_50[0]
    getitem_382: "f32[128]" = native_layer_norm_backward_50[1]
    getitem_383: "f32[128]" = native_layer_norm_backward_50[2];  native_layer_norm_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_129: "f32[8, 3136, 128]" = torch.ops.aten.add.Tensor(view_1170, getitem_381);  view_1170 = getitem_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1177: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(add_129, [8, 56, 56, 128]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    slice_backward_46: "f32[8, 56, 56, 128]" = torch.ops.aten.slice_backward.default(view_1177, [8, 56, 56, 128], 3, 0, 9223372036854775807, 1)
    slice_backward_47: "f32[8, 56, 56, 128]" = torch.ops.aten.slice_backward.default(slice_backward_46, [8, 56, 56, 128], 0, 0, 9223372036854775807, 1);  slice_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1178: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(slice_backward_47, [8, 8, 7, 8, 7, 128]);  slice_backward_47 = None
    permute_195: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_1178, [0, 1, 3, 2, 4, 5]);  view_1178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_355: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
    _unsafe_view_190: "f32[512, 7, 7, 128]" = torch.ops.aten._unsafe_view.default(clone_355, [512, 7, 7, 128]);  clone_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1179: "f32[512, 49, 128]" = torch.ops.aten.view.default(_unsafe_view_190, [512, 49, 128]);  _unsafe_view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1180: "f32[25088, 128]" = torch.ops.aten.view.default(view_1179, [25088, 128]);  view_1179 = None
    mm_199: "f32[25088, 128]" = torch.ops.aten.mm.default(view_1180, t_492);  t_492 = None
    t_493: "f32[128, 25088]" = torch.ops.aten.t.default(view_1180)
    mm_200: "f32[128, 128]" = torch.ops.aten.mm.default(t_493, view_11);  t_493 = view_11 = None
    t_494: "f32[128, 128]" = torch.ops.aten.t.default(mm_200);  mm_200 = None
    sum_119: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1180, [0], True);  view_1180 = None
    view_1181: "f32[128]" = torch.ops.aten.view.default(sum_119, [128]);  sum_119 = None
    t_495: "f32[128, 128]" = torch.ops.aten.t.default(t_494);  t_494 = None
    view_1182: "f32[512, 49, 128]" = torch.ops.aten.view.default(mm_199, [512, 49, 128]);  mm_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1183: "f32[512, 49, 4, 32]" = torch.ops.aten.view.default(view_1182, [512, 49, 4, 32]);  view_1182 = None
    transpose_186: "f32[512, 4, 49, 32]" = torch.ops.aten.transpose.int(view_1183, 1, 2);  view_1183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_356: "f32[512, 4, 49, 32]" = torch.ops.aten.clone.default(transpose_186, memory_format = torch.contiguous_format);  transpose_186 = None
    _unsafe_view_191: "f32[2048, 49, 32]" = torch.ops.aten._unsafe_view.default(clone_356, [2048, 49, 32]);  clone_356 = None
    bmm_140: "f32[2048, 49, 32]" = torch.ops.aten.bmm.default(transpose_187, _unsafe_view_191);  transpose_187 = None
    bmm_141: "f32[2048, 49, 49]" = torch.ops.aten.bmm.default(_unsafe_view_191, transpose_188);  _unsafe_view_191 = transpose_188 = None
    view_1184: "f32[512, 4, 49, 32]" = torch.ops.aten.view.default(bmm_140, [512, 4, 49, 32]);  bmm_140 = None
    view_1185: "f32[512, 4, 49, 49]" = torch.ops.aten.view.default(bmm_141, [512, 4, 49, 49]);  bmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    _softmax_backward_data_23: "f32[512, 4, 49, 49]" = torch.ops.aten._softmax_backward_data.default(view_1185, detach_47, -1, torch.float32);  view_1185 = detach_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_120: "f32[1, 4, 49, 49]" = torch.ops.aten.sum.dim_IntList(_softmax_backward_data_23, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_23: "f32[4, 49, 49]" = torch.ops.aten.squeeze.dim(sum_120, 0);  sum_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_196: "f32[49, 49, 4]" = torch.ops.aten.permute.default(squeeze_23, [1, 2, 0]);  squeeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1186: "f32[2401, 4]" = torch.ops.aten.view.default(permute_196, [2401, 4]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    new_zeros_23: "f32[169, 4]" = torch.ops.aten.new_zeros.default(view_1186, [169, 4], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_23: "f32[169, 4]" = torch.ops.aten.index_put.default(new_zeros_23, [view_7], view_1186, True);  new_zeros_23 = view_7 = view_1186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1187: "f32[2048, 49, 49]" = torch.ops.aten.view.default(_softmax_backward_data_23, [2048, 49, 49]);  _softmax_backward_data_23 = None
    bmm_142: "f32[2048, 32, 49]" = torch.ops.aten.bmm.default(transpose_189, view_1187);  transpose_189 = None
    bmm_143: "f32[2048, 49, 32]" = torch.ops.aten.bmm.default(view_1187, transpose_190);  view_1187 = transpose_190 = None
    view_1188: "f32[512, 4, 32, 49]" = torch.ops.aten.view.default(bmm_142, [512, 4, 32, 49]);  bmm_142 = None
    view_1189: "f32[512, 4, 49, 32]" = torch.ops.aten.view.default(bmm_143, [512, 4, 49, 32]);  bmm_143 = None
    transpose_191: "f32[512, 4, 49, 32]" = torch.ops.aten.transpose.int(view_1188, -2, -1);  view_1188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_139: "f32[512, 4, 49, 32]" = torch.ops.aten.mul.Tensor(view_1189, 0.1767766952966369);  view_1189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    stack_23: "f32[3, 512, 4, 49, 32]" = torch.ops.aten.stack.default([mul_139, transpose_191, view_1184]);  mul_139 = transpose_191 = view_1184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_197: "f32[512, 49, 3, 4, 32]" = torch.ops.aten.permute.default(stack_23, [1, 3, 0, 2, 4]);  stack_23 = None
    clone_357: "f32[512, 49, 3, 4, 32]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
    _unsafe_view_192: "f32[512, 49, 384]" = torch.ops.aten._unsafe_view.default(clone_357, [512, 49, 384]);  clone_357 = None
    view_1190: "f32[25088, 384]" = torch.ops.aten.view.default(_unsafe_view_192, [25088, 384]);  _unsafe_view_192 = None
    mm_201: "f32[25088, 128]" = torch.ops.aten.mm.default(view_1190, t_496);  t_496 = None
    t_497: "f32[384, 25088]" = torch.ops.aten.t.default(view_1190)
    mm_202: "f32[384, 128]" = torch.ops.aten.mm.default(t_497, view_3);  t_497 = view_3 = None
    t_498: "f32[128, 384]" = torch.ops.aten.t.default(mm_202);  mm_202 = None
    sum_121: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1190, [0], True);  view_1190 = None
    view_1191: "f32[384]" = torch.ops.aten.view.default(sum_121, [384]);  sum_121 = None
    t_499: "f32[384, 128]" = torch.ops.aten.t.default(t_498);  t_498 = None
    view_1192: "f32[512, 49, 128]" = torch.ops.aten.view.default(mm_201, [512, 49, 128]);  mm_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1193: "f32[512, 7, 7, 128]" = torch.ops.aten.view.default(view_1192, [512, 7, 7, 128]);  view_1192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1194: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_1193, [8, 8, 8, 7, 7, 128]);  view_1193 = None
    permute_198: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_1194, [0, 1, 3, 2, 4, 5]);  view_1194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_358: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
    _unsafe_view_193: "f32[8, 56, 56, 128]" = torch.ops.aten._unsafe_view.default(clone_358, [8, 56, 56, 128]);  clone_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    constant_pad_nd_47: "f32[8, 56, 56, 128]" = torch.ops.aten.constant_pad_nd.default(_unsafe_view_193, [0, 0, 0, 0, 0, 0]);  _unsafe_view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    native_layer_norm_backward_51 = torch.ops.aten.native_layer_norm_backward.default(constant_pad_nd_47, getitem, [128], getitem_4, getitem_5, primals_29, primals_30, [True, True, True]);  constant_pad_nd_47 = getitem = getitem_4 = getitem_5 = primals_29 = primals_30 = None
    getitem_384: "f32[8, 56, 56, 128]" = native_layer_norm_backward_51[0]
    getitem_385: "f32[128]" = native_layer_norm_backward_51[1]
    getitem_386: "f32[128]" = native_layer_norm_backward_51[2];  native_layer_norm_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_130: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_1177, getitem_384);  view_1177 = getitem_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    native_layer_norm_backward_52 = torch.ops.aten.native_layer_norm_backward.default(add_130, permute, [128], getitem_1, getitem_2, primals_27, primals_28, [True, True, True]);  add_130 = permute = getitem_1 = getitem_2 = primals_27 = primals_28 = None
    getitem_387: "f32[8, 56, 56, 128]" = native_layer_norm_backward_52[0]
    getitem_388: "f32[128]" = native_layer_norm_backward_52[1]
    getitem_389: "f32[128]" = native_layer_norm_backward_52[2];  native_layer_norm_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/format.py:43, code: x = x.permute(0, 2, 3, 1)
    permute_199: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(getitem_387, [0, 3, 1, 2]);  getitem_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(permute_199, primals_365, primals_25, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  permute_199 = primals_365 = primals_25 = None
    getitem_391: "f32[128, 3, 4, 4]" = convolution_backward[1]
    getitem_392: "f32[128]" = convolution_backward[2];  convolution_backward = None
    return [index_put_23, index_put_22, index_put_21, index_put_20, index_put_19, index_put_18, index_put_17, index_put_16, index_put_15, index_put_14, index_put_13, index_put_12, index_put_11, index_put_10, index_put_9, index_put_8, index_put_7, index_put_6, index_put_5, index_put_4, index_put_3, index_put_2, index_put_1, index_put, getitem_391, getitem_392, getitem_388, getitem_389, getitem_385, getitem_386, t_499, view_1191, t_495, view_1181, getitem_382, getitem_383, t_491, view_1175, t_487, view_1172, getitem_379, getitem_380, t_483, view_1166, t_479, view_1154, getitem_376, getitem_377, t_475, view_1148, t_471, view_1145, getitem_373, getitem_374, t_467, getitem_370, getitem_371, t_463, view_1136, t_459, view_1126, getitem_367, getitem_368, t_455, view_1120, t_451, view_1117, getitem_364, getitem_365, t_447, view_1111, t_443, view_1099, getitem_361, getitem_362, t_439, view_1093, t_435, view_1090, getitem_358, getitem_359, t_431, getitem_355, getitem_356, t_427, view_1081, t_423, view_1071, getitem_352, getitem_353, t_419, view_1065, t_415, view_1062, getitem_349, getitem_350, t_411, view_1056, t_407, view_1044, getitem_346, getitem_347, t_403, view_1038, t_399, view_1035, getitem_343, getitem_344, t_395, view_1029, t_391, view_1019, getitem_340, getitem_341, t_387, view_1013, t_383, view_1010, getitem_337, getitem_338, t_379, view_1004, t_375, view_992, getitem_334, getitem_335, t_371, view_986, t_367, view_983, getitem_331, getitem_332, t_363, view_977, t_359, view_967, getitem_328, getitem_329, t_355, view_961, t_351, view_958, getitem_325, getitem_326, t_347, view_952, t_343, view_940, getitem_322, getitem_323, t_339, view_934, t_335, view_931, getitem_319, getitem_320, t_331, view_925, t_327, view_915, getitem_316, getitem_317, t_323, view_909, t_319, view_906, getitem_313, getitem_314, t_315, view_900, t_311, view_888, getitem_310, getitem_311, t_307, view_882, t_303, view_879, getitem_307, getitem_308, t_299, view_873, t_295, view_863, getitem_304, getitem_305, t_291, view_857, t_287, view_854, getitem_301, getitem_302, t_283, view_848, t_279, view_836, getitem_298, getitem_299, t_275, view_830, t_271, view_827, getitem_295, getitem_296, t_267, view_821, t_263, view_811, getitem_292, getitem_293, t_259, view_805, t_255, view_802, getitem_289, getitem_290, t_251, view_796, t_247, view_784, getitem_286, getitem_287, t_243, view_778, t_239, view_775, getitem_283, getitem_284, t_235, view_769, t_231, view_759, getitem_280, getitem_281, t_227, view_753, t_223, view_750, getitem_277, getitem_278, t_219, view_744, t_215, view_732, getitem_274, getitem_275, t_211, view_726, t_207, view_723, getitem_271, getitem_272, t_203, view_717, t_199, view_707, getitem_268, getitem_269, t_195, view_701, t_191, view_698, getitem_265, getitem_266, t_187, view_692, t_183, view_680, getitem_262, getitem_263, t_179, view_674, t_175, view_671, getitem_259, getitem_260, t_171, view_665, t_167, view_655, getitem_256, getitem_257, t_163, view_649, t_159, view_646, getitem_253, getitem_254, t_155, view_640, t_151, view_628, getitem_250, getitem_251, t_147, view_622, t_143, view_619, getitem_247, getitem_248, t_139, getitem_244, getitem_245, t_135, view_609, t_131, view_599, getitem_241, getitem_242, t_127, view_592, t_123, view_589, getitem_238, getitem_239, t_119, view_582, t_115, view_572, getitem_235, getitem_236, t_111, view_565, t_107, view_562, getitem_232, getitem_233, t_103, view_559, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    