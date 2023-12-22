from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[64]", primals_4: "f32[128, 64, 1, 1]", primals_5: "f32[128]", primals_7: "f32[32, 4, 3, 3]", primals_8: "f32[32]", primals_10: "f32[32, 4, 3, 3]", primals_11: "f32[32]", primals_13: "f32[32, 4, 3, 3]", primals_14: "f32[32]", primals_16: "f32[256, 128, 1, 1]", primals_17: "f32[256]", primals_19: "f32[256, 64, 1, 1]", primals_20: "f32[256]", primals_22: "f32[128, 256, 1, 1]", primals_23: "f32[128]", primals_25: "f32[32, 4, 3, 3]", primals_26: "f32[32]", primals_28: "f32[32, 4, 3, 3]", primals_29: "f32[32]", primals_31: "f32[32, 4, 3, 3]", primals_32: "f32[32]", primals_34: "f32[256, 128, 1, 1]", primals_35: "f32[256]", primals_37: "f32[128, 256, 1, 1]", primals_38: "f32[128]", primals_40: "f32[32, 4, 3, 3]", primals_41: "f32[32]", primals_43: "f32[32, 4, 3, 3]", primals_44: "f32[32]", primals_46: "f32[32, 4, 3, 3]", primals_47: "f32[32]", primals_49: "f32[256, 128, 1, 1]", primals_50: "f32[256]", primals_52: "f32[256, 256, 1, 1]", primals_53: "f32[256]", primals_55: "f32[64, 8, 3, 3]", primals_56: "f32[64]", primals_58: "f32[64, 8, 3, 3]", primals_59: "f32[64]", primals_61: "f32[64, 8, 3, 3]", primals_62: "f32[64]", primals_64: "f32[512, 256, 1, 1]", primals_65: "f32[512]", primals_67: "f32[512, 256, 1, 1]", primals_68: "f32[512]", primals_70: "f32[256, 512, 1, 1]", primals_71: "f32[256]", primals_73: "f32[64, 8, 3, 3]", primals_74: "f32[64]", primals_76: "f32[64, 8, 3, 3]", primals_77: "f32[64]", primals_79: "f32[64, 8, 3, 3]", primals_80: "f32[64]", primals_82: "f32[512, 256, 1, 1]", primals_83: "f32[512]", primals_85: "f32[256, 512, 1, 1]", primals_86: "f32[256]", primals_88: "f32[64, 8, 3, 3]", primals_89: "f32[64]", primals_91: "f32[64, 8, 3, 3]", primals_92: "f32[64]", primals_94: "f32[64, 8, 3, 3]", primals_95: "f32[64]", primals_97: "f32[512, 256, 1, 1]", primals_98: "f32[512]", primals_100: "f32[256, 512, 1, 1]", primals_101: "f32[256]", primals_103: "f32[64, 8, 3, 3]", primals_104: "f32[64]", primals_106: "f32[64, 8, 3, 3]", primals_107: "f32[64]", primals_109: "f32[64, 8, 3, 3]", primals_110: "f32[64]", primals_112: "f32[512, 256, 1, 1]", primals_113: "f32[512]", primals_115: "f32[512, 512, 1, 1]", primals_116: "f32[512]", primals_118: "f32[128, 16, 3, 3]", primals_119: "f32[128]", primals_121: "f32[128, 16, 3, 3]", primals_122: "f32[128]", primals_124: "f32[128, 16, 3, 3]", primals_125: "f32[128]", primals_127: "f32[1024, 512, 1, 1]", primals_128: "f32[1024]", primals_130: "f32[1024, 512, 1, 1]", primals_131: "f32[1024]", primals_133: "f32[512, 1024, 1, 1]", primals_134: "f32[512]", primals_136: "f32[128, 16, 3, 3]", primals_137: "f32[128]", primals_139: "f32[128, 16, 3, 3]", primals_140: "f32[128]", primals_142: "f32[128, 16, 3, 3]", primals_143: "f32[128]", primals_145: "f32[1024, 512, 1, 1]", primals_146: "f32[1024]", primals_148: "f32[512, 1024, 1, 1]", primals_149: "f32[512]", primals_151: "f32[128, 16, 3, 3]", primals_152: "f32[128]", primals_154: "f32[128, 16, 3, 3]", primals_155: "f32[128]", primals_157: "f32[128, 16, 3, 3]", primals_158: "f32[128]", primals_160: "f32[1024, 512, 1, 1]", primals_161: "f32[1024]", primals_163: "f32[512, 1024, 1, 1]", primals_164: "f32[512]", primals_166: "f32[128, 16, 3, 3]", primals_167: "f32[128]", primals_169: "f32[128, 16, 3, 3]", primals_170: "f32[128]", primals_172: "f32[128, 16, 3, 3]", primals_173: "f32[128]", primals_175: "f32[1024, 512, 1, 1]", primals_176: "f32[1024]", primals_178: "f32[512, 1024, 1, 1]", primals_179: "f32[512]", primals_181: "f32[128, 16, 3, 3]", primals_182: "f32[128]", primals_184: "f32[128, 16, 3, 3]", primals_185: "f32[128]", primals_187: "f32[128, 16, 3, 3]", primals_188: "f32[128]", primals_190: "f32[1024, 512, 1, 1]", primals_191: "f32[1024]", primals_193: "f32[512, 1024, 1, 1]", primals_194: "f32[512]", primals_196: "f32[128, 16, 3, 3]", primals_197: "f32[128]", primals_199: "f32[128, 16, 3, 3]", primals_200: "f32[128]", primals_202: "f32[128, 16, 3, 3]", primals_203: "f32[128]", primals_205: "f32[1024, 512, 1, 1]", primals_206: "f32[1024]", primals_208: "f32[1024, 1024, 1, 1]", primals_209: "f32[1024]", primals_211: "f32[256, 32, 3, 3]", primals_212: "f32[256]", primals_214: "f32[256, 32, 3, 3]", primals_215: "f32[256]", primals_217: "f32[256, 32, 3, 3]", primals_218: "f32[256]", primals_220: "f32[2048, 1024, 1, 1]", primals_221: "f32[2048]", primals_223: "f32[2048, 1024, 1, 1]", primals_224: "f32[2048]", primals_226: "f32[1024, 2048, 1, 1]", primals_227: "f32[1024]", primals_229: "f32[256, 32, 3, 3]", primals_230: "f32[256]", primals_232: "f32[256, 32, 3, 3]", primals_233: "f32[256]", primals_235: "f32[256, 32, 3, 3]", primals_236: "f32[256]", primals_238: "f32[2048, 1024, 1, 1]", primals_239: "f32[2048]", primals_241: "f32[1024, 2048, 1, 1]", primals_242: "f32[1024]", primals_244: "f32[256, 32, 3, 3]", primals_245: "f32[256]", primals_247: "f32[256, 32, 3, 3]", primals_248: "f32[256]", primals_250: "f32[256, 32, 3, 3]", primals_251: "f32[256]", primals_253: "f32[2048, 1024, 1, 1]", primals_254: "f32[2048]", primals_513: "f32[8, 3, 224, 224]", convolution: "f32[8, 64, 112, 112]", squeeze_1: "f32[64]", relu: "f32[8, 64, 112, 112]", getitem_2: "f32[8, 64, 56, 56]", getitem_3: "i64[8, 64, 56, 56]", convolution_1: "f32[8, 128, 56, 56]", squeeze_4: "f32[128]", getitem_10: "f32[8, 32, 56, 56]", convolution_2: "f32[8, 32, 56, 56]", squeeze_7: "f32[32]", getitem_17: "f32[8, 32, 56, 56]", convolution_3: "f32[8, 32, 56, 56]", squeeze_10: "f32[32]", getitem_24: "f32[8, 32, 56, 56]", convolution_4: "f32[8, 32, 56, 56]", squeeze_13: "f32[32]", getitem_31: "f32[8, 32, 56, 56]", cat: "f32[8, 128, 56, 56]", convolution_5: "f32[8, 256, 56, 56]", squeeze_16: "f32[256]", convolution_6: "f32[8, 256, 56, 56]", squeeze_19: "f32[256]", relu_5: "f32[8, 256, 56, 56]", convolution_7: "f32[8, 128, 56, 56]", squeeze_22: "f32[128]", getitem_42: "f32[8, 32, 56, 56]", convolution_8: "f32[8, 32, 56, 56]", squeeze_25: "f32[32]", add_46: "f32[8, 32, 56, 56]", convolution_9: "f32[8, 32, 56, 56]", squeeze_28: "f32[32]", add_52: "f32[8, 32, 56, 56]", convolution_10: "f32[8, 32, 56, 56]", squeeze_31: "f32[32]", cat_1: "f32[8, 128, 56, 56]", convolution_11: "f32[8, 256, 56, 56]", squeeze_34: "f32[256]", relu_10: "f32[8, 256, 56, 56]", convolution_12: "f32[8, 128, 56, 56]", squeeze_37: "f32[128]", getitem_72: "f32[8, 32, 56, 56]", convolution_13: "f32[8, 32, 56, 56]", squeeze_40: "f32[32]", add_74: "f32[8, 32, 56, 56]", convolution_14: "f32[8, 32, 56, 56]", squeeze_43: "f32[32]", add_80: "f32[8, 32, 56, 56]", convolution_15: "f32[8, 32, 56, 56]", squeeze_46: "f32[32]", cat_2: "f32[8, 128, 56, 56]", convolution_16: "f32[8, 256, 56, 56]", squeeze_49: "f32[256]", relu_15: "f32[8, 256, 56, 56]", convolution_17: "f32[8, 256, 56, 56]", squeeze_52: "f32[256]", getitem_102: "f32[8, 64, 56, 56]", convolution_18: "f32[8, 64, 28, 28]", squeeze_55: "f32[64]", getitem_109: "f32[8, 64, 56, 56]", convolution_19: "f32[8, 64, 28, 28]", squeeze_58: "f32[64]", getitem_116: "f32[8, 64, 56, 56]", convolution_20: "f32[8, 64, 28, 28]", squeeze_61: "f32[64]", getitem_123: "f32[8, 64, 56, 56]", cat_3: "f32[8, 256, 28, 28]", convolution_21: "f32[8, 512, 28, 28]", squeeze_64: "f32[512]", convolution_22: "f32[8, 512, 28, 28]", squeeze_67: "f32[512]", relu_20: "f32[8, 512, 28, 28]", convolution_23: "f32[8, 256, 28, 28]", squeeze_70: "f32[256]", getitem_134: "f32[8, 64, 28, 28]", convolution_24: "f32[8, 64, 28, 28]", squeeze_73: "f32[64]", add_133: "f32[8, 64, 28, 28]", convolution_25: "f32[8, 64, 28, 28]", squeeze_76: "f32[64]", add_139: "f32[8, 64, 28, 28]", convolution_26: "f32[8, 64, 28, 28]", squeeze_79: "f32[64]", cat_4: "f32[8, 256, 28, 28]", convolution_27: "f32[8, 512, 28, 28]", squeeze_82: "f32[512]", relu_25: "f32[8, 512, 28, 28]", convolution_28: "f32[8, 256, 28, 28]", squeeze_85: "f32[256]", getitem_164: "f32[8, 64, 28, 28]", convolution_29: "f32[8, 64, 28, 28]", squeeze_88: "f32[64]", add_161: "f32[8, 64, 28, 28]", convolution_30: "f32[8, 64, 28, 28]", squeeze_91: "f32[64]", add_167: "f32[8, 64, 28, 28]", convolution_31: "f32[8, 64, 28, 28]", squeeze_94: "f32[64]", cat_5: "f32[8, 256, 28, 28]", convolution_32: "f32[8, 512, 28, 28]", squeeze_97: "f32[512]", relu_30: "f32[8, 512, 28, 28]", convolution_33: "f32[8, 256, 28, 28]", squeeze_100: "f32[256]", getitem_194: "f32[8, 64, 28, 28]", convolution_34: "f32[8, 64, 28, 28]", squeeze_103: "f32[64]", add_189: "f32[8, 64, 28, 28]", convolution_35: "f32[8, 64, 28, 28]", squeeze_106: "f32[64]", add_195: "f32[8, 64, 28, 28]", convolution_36: "f32[8, 64, 28, 28]", squeeze_109: "f32[64]", cat_6: "f32[8, 256, 28, 28]", convolution_37: "f32[8, 512, 28, 28]", squeeze_112: "f32[512]", relu_35: "f32[8, 512, 28, 28]", convolution_38: "f32[8, 512, 28, 28]", squeeze_115: "f32[512]", getitem_224: "f32[8, 128, 28, 28]", convolution_39: "f32[8, 128, 14, 14]", squeeze_118: "f32[128]", getitem_231: "f32[8, 128, 28, 28]", convolution_40: "f32[8, 128, 14, 14]", squeeze_121: "f32[128]", getitem_238: "f32[8, 128, 28, 28]", convolution_41: "f32[8, 128, 14, 14]", squeeze_124: "f32[128]", getitem_245: "f32[8, 128, 28, 28]", cat_7: "f32[8, 512, 14, 14]", convolution_42: "f32[8, 1024, 14, 14]", squeeze_127: "f32[1024]", convolution_43: "f32[8, 1024, 14, 14]", squeeze_130: "f32[1024]", relu_40: "f32[8, 1024, 14, 14]", convolution_44: "f32[8, 512, 14, 14]", squeeze_133: "f32[512]", getitem_256: "f32[8, 128, 14, 14]", convolution_45: "f32[8, 128, 14, 14]", squeeze_136: "f32[128]", add_248: "f32[8, 128, 14, 14]", convolution_46: "f32[8, 128, 14, 14]", squeeze_139: "f32[128]", add_254: "f32[8, 128, 14, 14]", convolution_47: "f32[8, 128, 14, 14]", squeeze_142: "f32[128]", cat_8: "f32[8, 512, 14, 14]", convolution_48: "f32[8, 1024, 14, 14]", squeeze_145: "f32[1024]", relu_45: "f32[8, 1024, 14, 14]", convolution_49: "f32[8, 512, 14, 14]", squeeze_148: "f32[512]", getitem_286: "f32[8, 128, 14, 14]", convolution_50: "f32[8, 128, 14, 14]", squeeze_151: "f32[128]", add_276: "f32[8, 128, 14, 14]", convolution_51: "f32[8, 128, 14, 14]", squeeze_154: "f32[128]", add_282: "f32[8, 128, 14, 14]", convolution_52: "f32[8, 128, 14, 14]", squeeze_157: "f32[128]", cat_9: "f32[8, 512, 14, 14]", convolution_53: "f32[8, 1024, 14, 14]", squeeze_160: "f32[1024]", relu_50: "f32[8, 1024, 14, 14]", convolution_54: "f32[8, 512, 14, 14]", squeeze_163: "f32[512]", getitem_316: "f32[8, 128, 14, 14]", convolution_55: "f32[8, 128, 14, 14]", squeeze_166: "f32[128]", add_304: "f32[8, 128, 14, 14]", convolution_56: "f32[8, 128, 14, 14]", squeeze_169: "f32[128]", add_310: "f32[8, 128, 14, 14]", convolution_57: "f32[8, 128, 14, 14]", squeeze_172: "f32[128]", cat_10: "f32[8, 512, 14, 14]", convolution_58: "f32[8, 1024, 14, 14]", squeeze_175: "f32[1024]", relu_55: "f32[8, 1024, 14, 14]", convolution_59: "f32[8, 512, 14, 14]", squeeze_178: "f32[512]", getitem_346: "f32[8, 128, 14, 14]", convolution_60: "f32[8, 128, 14, 14]", squeeze_181: "f32[128]", add_332: "f32[8, 128, 14, 14]", convolution_61: "f32[8, 128, 14, 14]", squeeze_184: "f32[128]", add_338: "f32[8, 128, 14, 14]", convolution_62: "f32[8, 128, 14, 14]", squeeze_187: "f32[128]", cat_11: "f32[8, 512, 14, 14]", convolution_63: "f32[8, 1024, 14, 14]", squeeze_190: "f32[1024]", relu_60: "f32[8, 1024, 14, 14]", convolution_64: "f32[8, 512, 14, 14]", squeeze_193: "f32[512]", getitem_376: "f32[8, 128, 14, 14]", convolution_65: "f32[8, 128, 14, 14]", squeeze_196: "f32[128]", add_360: "f32[8, 128, 14, 14]", convolution_66: "f32[8, 128, 14, 14]", squeeze_199: "f32[128]", add_366: "f32[8, 128, 14, 14]", convolution_67: "f32[8, 128, 14, 14]", squeeze_202: "f32[128]", cat_12: "f32[8, 512, 14, 14]", convolution_68: "f32[8, 1024, 14, 14]", squeeze_205: "f32[1024]", relu_65: "f32[8, 1024, 14, 14]", convolution_69: "f32[8, 1024, 14, 14]", squeeze_208: "f32[1024]", getitem_406: "f32[8, 256, 14, 14]", convolution_70: "f32[8, 256, 7, 7]", squeeze_211: "f32[256]", getitem_413: "f32[8, 256, 14, 14]", convolution_71: "f32[8, 256, 7, 7]", squeeze_214: "f32[256]", getitem_420: "f32[8, 256, 14, 14]", convolution_72: "f32[8, 256, 7, 7]", squeeze_217: "f32[256]", getitem_427: "f32[8, 256, 14, 14]", cat_13: "f32[8, 1024, 7, 7]", convolution_73: "f32[8, 2048, 7, 7]", squeeze_220: "f32[2048]", convolution_74: "f32[8, 2048, 7, 7]", squeeze_223: "f32[2048]", relu_70: "f32[8, 2048, 7, 7]", convolution_75: "f32[8, 1024, 7, 7]", squeeze_226: "f32[1024]", getitem_438: "f32[8, 256, 7, 7]", convolution_76: "f32[8, 256, 7, 7]", squeeze_229: "f32[256]", add_419: "f32[8, 256, 7, 7]", convolution_77: "f32[8, 256, 7, 7]", squeeze_232: "f32[256]", add_425: "f32[8, 256, 7, 7]", convolution_78: "f32[8, 256, 7, 7]", squeeze_235: "f32[256]", cat_14: "f32[8, 1024, 7, 7]", convolution_79: "f32[8, 2048, 7, 7]", squeeze_238: "f32[2048]", relu_75: "f32[8, 2048, 7, 7]", convolution_80: "f32[8, 1024, 7, 7]", squeeze_241: "f32[1024]", getitem_468: "f32[8, 256, 7, 7]", convolution_81: "f32[8, 256, 7, 7]", squeeze_244: "f32[256]", add_447: "f32[8, 256, 7, 7]", convolution_82: "f32[8, 256, 7, 7]", squeeze_247: "f32[256]", add_453: "f32[8, 256, 7, 7]", convolution_83: "f32[8, 256, 7, 7]", squeeze_250: "f32[256]", cat_15: "f32[8, 1024, 7, 7]", convolution_84: "f32[8, 2048, 7, 7]", squeeze_253: "f32[2048]", view: "f32[8, 2048]", permute_1: "f32[1000, 2048]", le: "b8[8, 2048, 7, 7]", unsqueeze_342: "f32[1, 2048, 1, 1]", le_1: "b8[8, 256, 7, 7]", unsqueeze_354: "f32[1, 256, 1, 1]", le_2: "b8[8, 256, 7, 7]", unsqueeze_366: "f32[1, 256, 1, 1]", le_3: "b8[8, 256, 7, 7]", unsqueeze_378: "f32[1, 256, 1, 1]", le_4: "b8[8, 1024, 7, 7]", unsqueeze_390: "f32[1, 1024, 1, 1]", unsqueeze_402: "f32[1, 2048, 1, 1]", le_6: "b8[8, 256, 7, 7]", unsqueeze_414: "f32[1, 256, 1, 1]", le_7: "b8[8, 256, 7, 7]", unsqueeze_426: "f32[1, 256, 1, 1]", le_8: "b8[8, 256, 7, 7]", unsqueeze_438: "f32[1, 256, 1, 1]", le_9: "b8[8, 1024, 7, 7]", unsqueeze_450: "f32[1, 1024, 1, 1]", unsqueeze_462: "f32[1, 2048, 1, 1]", unsqueeze_474: "f32[1, 2048, 1, 1]", le_11: "b8[8, 256, 7, 7]", unsqueeze_486: "f32[1, 256, 1, 1]", le_12: "b8[8, 256, 7, 7]", unsqueeze_498: "f32[1, 256, 1, 1]", le_13: "b8[8, 256, 7, 7]", unsqueeze_510: "f32[1, 256, 1, 1]", le_14: "b8[8, 1024, 14, 14]", unsqueeze_522: "f32[1, 1024, 1, 1]", unsqueeze_534: "f32[1, 1024, 1, 1]", le_16: "b8[8, 128, 14, 14]", unsqueeze_546: "f32[1, 128, 1, 1]", le_17: "b8[8, 128, 14, 14]", unsqueeze_558: "f32[1, 128, 1, 1]", le_18: "b8[8, 128, 14, 14]", unsqueeze_570: "f32[1, 128, 1, 1]", le_19: "b8[8, 512, 14, 14]", unsqueeze_582: "f32[1, 512, 1, 1]", unsqueeze_594: "f32[1, 1024, 1, 1]", le_21: "b8[8, 128, 14, 14]", unsqueeze_606: "f32[1, 128, 1, 1]", le_22: "b8[8, 128, 14, 14]", unsqueeze_618: "f32[1, 128, 1, 1]", le_23: "b8[8, 128, 14, 14]", unsqueeze_630: "f32[1, 128, 1, 1]", le_24: "b8[8, 512, 14, 14]", unsqueeze_642: "f32[1, 512, 1, 1]", unsqueeze_654: "f32[1, 1024, 1, 1]", le_26: "b8[8, 128, 14, 14]", unsqueeze_666: "f32[1, 128, 1, 1]", le_27: "b8[8, 128, 14, 14]", unsqueeze_678: "f32[1, 128, 1, 1]", le_28: "b8[8, 128, 14, 14]", unsqueeze_690: "f32[1, 128, 1, 1]", le_29: "b8[8, 512, 14, 14]", unsqueeze_702: "f32[1, 512, 1, 1]", unsqueeze_714: "f32[1, 1024, 1, 1]", le_31: "b8[8, 128, 14, 14]", unsqueeze_726: "f32[1, 128, 1, 1]", le_32: "b8[8, 128, 14, 14]", unsqueeze_738: "f32[1, 128, 1, 1]", le_33: "b8[8, 128, 14, 14]", unsqueeze_750: "f32[1, 128, 1, 1]", le_34: "b8[8, 512, 14, 14]", unsqueeze_762: "f32[1, 512, 1, 1]", unsqueeze_774: "f32[1, 1024, 1, 1]", le_36: "b8[8, 128, 14, 14]", unsqueeze_786: "f32[1, 128, 1, 1]", le_37: "b8[8, 128, 14, 14]", unsqueeze_798: "f32[1, 128, 1, 1]", le_38: "b8[8, 128, 14, 14]", unsqueeze_810: "f32[1, 128, 1, 1]", le_39: "b8[8, 512, 14, 14]", unsqueeze_822: "f32[1, 512, 1, 1]", unsqueeze_834: "f32[1, 1024, 1, 1]", unsqueeze_846: "f32[1, 1024, 1, 1]", le_41: "b8[8, 128, 14, 14]", unsqueeze_858: "f32[1, 128, 1, 1]", le_42: "b8[8, 128, 14, 14]", unsqueeze_870: "f32[1, 128, 1, 1]", le_43: "b8[8, 128, 14, 14]", unsqueeze_882: "f32[1, 128, 1, 1]", le_44: "b8[8, 512, 28, 28]", unsqueeze_894: "f32[1, 512, 1, 1]", unsqueeze_906: "f32[1, 512, 1, 1]", le_46: "b8[8, 64, 28, 28]", unsqueeze_918: "f32[1, 64, 1, 1]", le_47: "b8[8, 64, 28, 28]", unsqueeze_930: "f32[1, 64, 1, 1]", le_48: "b8[8, 64, 28, 28]", unsqueeze_942: "f32[1, 64, 1, 1]", le_49: "b8[8, 256, 28, 28]", unsqueeze_954: "f32[1, 256, 1, 1]", unsqueeze_966: "f32[1, 512, 1, 1]", le_51: "b8[8, 64, 28, 28]", unsqueeze_978: "f32[1, 64, 1, 1]", le_52: "b8[8, 64, 28, 28]", unsqueeze_990: "f32[1, 64, 1, 1]", le_53: "b8[8, 64, 28, 28]", unsqueeze_1002: "f32[1, 64, 1, 1]", le_54: "b8[8, 256, 28, 28]", unsqueeze_1014: "f32[1, 256, 1, 1]", unsqueeze_1026: "f32[1, 512, 1, 1]", le_56: "b8[8, 64, 28, 28]", unsqueeze_1038: "f32[1, 64, 1, 1]", le_57: "b8[8, 64, 28, 28]", unsqueeze_1050: "f32[1, 64, 1, 1]", le_58: "b8[8, 64, 28, 28]", unsqueeze_1062: "f32[1, 64, 1, 1]", le_59: "b8[8, 256, 28, 28]", unsqueeze_1074: "f32[1, 256, 1, 1]", unsqueeze_1086: "f32[1, 512, 1, 1]", unsqueeze_1098: "f32[1, 512, 1, 1]", le_61: "b8[8, 64, 28, 28]", unsqueeze_1110: "f32[1, 64, 1, 1]", le_62: "b8[8, 64, 28, 28]", unsqueeze_1122: "f32[1, 64, 1, 1]", le_63: "b8[8, 64, 28, 28]", unsqueeze_1134: "f32[1, 64, 1, 1]", le_64: "b8[8, 256, 56, 56]", unsqueeze_1146: "f32[1, 256, 1, 1]", unsqueeze_1158: "f32[1, 256, 1, 1]", le_66: "b8[8, 32, 56, 56]", unsqueeze_1170: "f32[1, 32, 1, 1]", le_67: "b8[8, 32, 56, 56]", unsqueeze_1182: "f32[1, 32, 1, 1]", le_68: "b8[8, 32, 56, 56]", unsqueeze_1194: "f32[1, 32, 1, 1]", le_69: "b8[8, 128, 56, 56]", unsqueeze_1206: "f32[1, 128, 1, 1]", unsqueeze_1218: "f32[1, 256, 1, 1]", le_71: "b8[8, 32, 56, 56]", unsqueeze_1230: "f32[1, 32, 1, 1]", le_72: "b8[8, 32, 56, 56]", unsqueeze_1242: "f32[1, 32, 1, 1]", le_73: "b8[8, 32, 56, 56]", unsqueeze_1254: "f32[1, 32, 1, 1]", le_74: "b8[8, 128, 56, 56]", unsqueeze_1266: "f32[1, 128, 1, 1]", unsqueeze_1278: "f32[1, 256, 1, 1]", unsqueeze_1290: "f32[1, 256, 1, 1]", le_76: "b8[8, 32, 56, 56]", unsqueeze_1302: "f32[1, 32, 1, 1]", le_77: "b8[8, 32, 56, 56]", unsqueeze_1314: "f32[1, 32, 1, 1]", le_78: "b8[8, 32, 56, 56]", unsqueeze_1326: "f32[1, 32, 1, 1]", le_79: "b8[8, 128, 56, 56]", unsqueeze_1338: "f32[1, 128, 1, 1]", unsqueeze_1350: "f32[1, 64, 1, 1]", tangents_1: "f32[8, 1000]"):
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
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_2: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_85: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_342);  convolution_84 = unsqueeze_342 = None
    mul_595: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_85)
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
    mul_601: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_348);  sub_85 = unsqueeze_348 = None
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
    where_1: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, slice_3);  le_1 = slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_89: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_354);  convolution_83 = unsqueeze_354 = None
    mul_604: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_89)
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
    mul_610: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_360);  sub_89 = unsqueeze_360 = None
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
    where_2: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, add_465);  le_2 = add_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_6: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_93: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_366);  convolution_82 = unsqueeze_366 = None
    mul_613: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_93)
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
    mul_619: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_372);  sub_93 = unsqueeze_372 = None
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
    where_3: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, add_466);  le_3 = add_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_8: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_97: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_378);  convolution_81 = unsqueeze_378 = None
    mul_622: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_97)
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
    mul_628: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_384);  sub_97 = unsqueeze_384 = None
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
    where_4: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, cat_16);  le_4 = cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_101: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_390);  convolution_80 = unsqueeze_390 = None
    mul_631: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_101)
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
    mul_637: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_396);  sub_101 = unsqueeze_396 = None
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
    le_5: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
    where_5: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, add_467);  le_5 = add_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_12: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_105: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_402);  convolution_79 = unsqueeze_402 = None
    mul_640: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_105)
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
    mul_646: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_408);  sub_105 = unsqueeze_408 = None
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
    where_6: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, slice_7);  le_6 = slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_14: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_109: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_414);  convolution_78 = unsqueeze_414 = None
    mul_649: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_109)
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
    mul_655: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_420);  sub_109 = unsqueeze_420 = None
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
    where_7: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, add_468);  le_7 = add_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_16: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_113: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_426);  convolution_77 = unsqueeze_426 = None
    mul_658: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_113)
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
    mul_664: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_432);  sub_113 = unsqueeze_432 = None
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
    where_8: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_8, full_default, add_469);  le_8 = add_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_18: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_117: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_438);  convolution_76 = unsqueeze_438 = None
    mul_667: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_117)
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
    mul_673: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_444);  sub_117 = unsqueeze_444 = None
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
    where_9: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le_9, full_default, cat_17);  le_9 = cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_20: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_121: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_450);  convolution_75 = unsqueeze_450 = None
    mul_676: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_121)
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
    mul_682: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_456);  sub_121 = unsqueeze_456 = None
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
    le_10: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
    where_10: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le_10, full_default, add_470);  le_10 = add_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    sum_22: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_125: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_462);  convolution_74 = unsqueeze_462 = None
    mul_685: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_125)
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
    mul_691: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_468);  sub_125 = unsqueeze_468 = None
    sub_127: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_10, mul_691);  mul_691 = None
    sub_128: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_465);  sub_127 = None
    mul_692: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_471);  sub_128 = unsqueeze_471 = None
    mul_693: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_223);  sum_23 = squeeze_223 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_692, relu_65, primals_223, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_692 = primals_223 = None
    getitem_522: "f32[8, 1024, 14, 14]" = convolution_backward_10[0]
    getitem_523: "f32[2048, 1024, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sub_129: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_474);  convolution_73 = unsqueeze_474 = None
    mul_694: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_129)
    sum_25: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_694, [0, 2, 3]);  mul_694 = None
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
    mul_700: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_480);  sub_129 = unsqueeze_480 = None
    sub_131: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_10, mul_700);  where_10 = mul_700 = None
    sub_132: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_465);  sub_131 = unsqueeze_465 = None
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
    where_11: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_11, full_default, slice_11);  le_11 = slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_26: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_133: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_486);  convolution_72 = unsqueeze_486 = None
    mul_703: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_11, sub_133)
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
    mul_709: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_492);  sub_133 = unsqueeze_492 = None
    sub_135: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_11, mul_709);  where_11 = mul_709 = None
    sub_136: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_489);  sub_135 = unsqueeze_489 = None
    mul_710: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_495);  sub_136 = unsqueeze_495 = None
    mul_711: "f32[256]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_217);  sum_27 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_710, getitem_420, primals_217, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_710 = getitem_420 = primals_217 = None
    getitem_528: "f32[8, 256, 14, 14]" = convolution_backward_12[0]
    getitem_529: "f32[256, 32, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_12: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_12, full_default, slice_10);  le_12 = slice_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_28: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_137: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_498);  convolution_71 = unsqueeze_498 = None
    mul_712: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_12, sub_137)
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
    mul_718: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_504);  sub_137 = unsqueeze_504 = None
    sub_139: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_12, mul_718);  where_12 = mul_718 = None
    sub_140: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_501);  sub_139 = unsqueeze_501 = None
    mul_719: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_507);  sub_140 = unsqueeze_507 = None
    mul_720: "f32[256]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_214);  sum_29 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_719, getitem_413, primals_214, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_719 = getitem_413 = primals_214 = None
    getitem_531: "f32[8, 256, 14, 14]" = convolution_backward_13[0]
    getitem_532: "f32[256, 32, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_13: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_13, full_default, slice_9);  le_13 = slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_30: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_141: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_510);  convolution_70 = unsqueeze_510 = None
    mul_721: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, sub_141)
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
    mul_727: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_516);  sub_141 = unsqueeze_516 = None
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
    where_14: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, cat_18);  le_14 = cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_32: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_145: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_522);  convolution_69 = unsqueeze_522 = None
    mul_730: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_145)
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
    mul_736: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_528);  sub_145 = unsqueeze_528 = None
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
    le_15: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
    where_15: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, add_471);  le_15 = add_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_149: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_534);  convolution_68 = unsqueeze_534 = None
    mul_739: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_149)
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
    mul_745: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_540);  sub_149 = unsqueeze_540 = None
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
    where_16: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, slice_15);  le_16 = slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_36: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_153: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_546);  convolution_67 = unsqueeze_546 = None
    mul_748: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_153)
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
    mul_754: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_552);  sub_153 = unsqueeze_552 = None
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
    where_17: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_17, full_default, add_472);  le_17 = add_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_38: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_157: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_558);  convolution_66 = unsqueeze_558 = None
    mul_757: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_157)
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
    mul_763: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_564);  sub_157 = unsqueeze_564 = None
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
    where_18: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, add_473);  le_18 = add_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_40: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_161: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_570);  convolution_65 = unsqueeze_570 = None
    mul_766: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_161)
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
    mul_772: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_576);  sub_161 = unsqueeze_576 = None
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
    where_19: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, cat_19);  le_19 = cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_42: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_165: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_582);  convolution_64 = unsqueeze_582 = None
    mul_775: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_165)
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
    mul_781: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_588);  sub_165 = unsqueeze_588 = None
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
    le_20: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
    where_20: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_20, full_default, add_474);  le_20 = add_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_44: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_169: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_594);  convolution_63 = unsqueeze_594 = None
    mul_784: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_169)
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
    mul_790: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_600);  sub_169 = unsqueeze_600 = None
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
    where_21: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_21, full_default, slice_19);  le_21 = slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_46: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_173: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_606);  convolution_62 = unsqueeze_606 = None
    mul_793: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_173)
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
    mul_799: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_612);  sub_173 = unsqueeze_612 = None
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
    where_22: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_22, full_default, add_475);  le_22 = add_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_48: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_177: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_618);  convolution_61 = unsqueeze_618 = None
    mul_802: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_177)
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
    mul_808: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_624);  sub_177 = unsqueeze_624 = None
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
    where_23: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_23, full_default, add_476);  le_23 = add_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_50: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_181: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_630);  convolution_60 = unsqueeze_630 = None
    mul_811: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_181)
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
    mul_817: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_636);  sub_181 = unsqueeze_636 = None
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
    where_24: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_24, full_default, cat_20);  le_24 = cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_52: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_185: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_642);  convolution_59 = unsqueeze_642 = None
    mul_820: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_185)
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
    mul_826: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_648);  sub_185 = unsqueeze_648 = None
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
    le_25: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
    where_25: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_25, full_default, add_477);  le_25 = add_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_54: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_189: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_654);  convolution_58 = unsqueeze_654 = None
    mul_829: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_189)
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
    mul_835: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_660);  sub_189 = unsqueeze_660 = None
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
    where_26: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_26, full_default, slice_23);  le_26 = slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_56: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_193: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_666);  convolution_57 = unsqueeze_666 = None
    mul_838: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_193)
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
    mul_844: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_672);  sub_193 = unsqueeze_672 = None
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
    where_27: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_27, full_default, add_478);  le_27 = add_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_58: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_197: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_678);  convolution_56 = unsqueeze_678 = None
    mul_847: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_197)
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
    mul_853: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_684);  sub_197 = unsqueeze_684 = None
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
    where_28: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_28, full_default, add_479);  le_28 = add_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_60: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_201: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_690);  convolution_55 = unsqueeze_690 = None
    mul_856: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_201)
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
    mul_862: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_696);  sub_201 = unsqueeze_696 = None
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
    where_29: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_29, full_default, cat_21);  le_29 = cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_62: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_205: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_702);  convolution_54 = unsqueeze_702 = None
    mul_865: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_29, sub_205)
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
    mul_871: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_708);  sub_205 = unsqueeze_708 = None
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
    le_30: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
    where_30: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_30, full_default, add_480);  le_30 = add_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_64: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_209: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_714);  convolution_53 = unsqueeze_714 = None
    mul_874: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_209)
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
    mul_880: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_720);  sub_209 = unsqueeze_720 = None
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
    where_31: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_31, full_default, slice_27);  le_31 = slice_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_66: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_213: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_726);  convolution_52 = unsqueeze_726 = None
    mul_883: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_213)
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
    mul_889: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_732);  sub_213 = unsqueeze_732 = None
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
    where_32: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_32, full_default, add_481);  le_32 = add_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_68: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_217: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_738);  convolution_51 = unsqueeze_738 = None
    mul_892: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_217)
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
    mul_898: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_744);  sub_217 = unsqueeze_744 = None
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
    where_33: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_33, full_default, add_482);  le_33 = add_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_70: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_221: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_750);  convolution_50 = unsqueeze_750 = None
    mul_901: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_221)
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
    mul_907: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_756);  sub_221 = unsqueeze_756 = None
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
    where_34: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_34, full_default, cat_22);  le_34 = cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_72: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_225: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_762);  convolution_49 = unsqueeze_762 = None
    mul_910: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_225)
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
    mul_916: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_768);  sub_225 = unsqueeze_768 = None
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
    le_35: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
    where_35: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_35, full_default, add_483);  le_35 = add_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_74: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_229: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_774);  convolution_48 = unsqueeze_774 = None
    mul_919: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_229)
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
    mul_925: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_780);  sub_229 = unsqueeze_780 = None
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
    where_36: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_36, full_default, slice_31);  le_36 = slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_76: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_233: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_786);  convolution_47 = unsqueeze_786 = None
    mul_928: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_233)
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
    mul_934: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_792);  sub_233 = unsqueeze_792 = None
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
    where_37: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_37, full_default, add_484);  le_37 = add_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_78: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_237: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_798);  convolution_46 = unsqueeze_798 = None
    mul_937: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_237)
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
    mul_943: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_804);  sub_237 = unsqueeze_804 = None
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
    where_38: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_38, full_default, add_485);  le_38 = add_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_80: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_241: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_810);  convolution_45 = unsqueeze_810 = None
    mul_946: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_241)
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
    mul_952: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_816);  sub_241 = unsqueeze_816 = None
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
    where_39: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_39, full_default, cat_23);  le_39 = cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_82: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_245: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_822);  convolution_44 = unsqueeze_822 = None
    mul_955: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_245)
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
    mul_961: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_828);  sub_245 = unsqueeze_828 = None
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
    le_40: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
    where_40: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_40, full_default, add_486);  le_40 = add_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    sum_84: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_249: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_834);  convolution_43 = unsqueeze_834 = None
    mul_964: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_249)
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
    mul_970: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_840);  sub_249 = unsqueeze_840 = None
    sub_251: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_970);  mul_970 = None
    sub_252: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_837);  sub_251 = None
    mul_971: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_843);  sub_252 = unsqueeze_843 = None
    mul_972: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_130);  sum_85 = squeeze_130 = None
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_971, relu_35, primals_130, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_971 = primals_130 = None
    getitem_615: "f32[8, 512, 28, 28]" = convolution_backward_41[0]
    getitem_616: "f32[1024, 512, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sub_253: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_846);  convolution_42 = unsqueeze_846 = None
    mul_973: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_253)
    sum_87: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_973, [0, 2, 3]);  mul_973 = None
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
    mul_979: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_852);  sub_253 = unsqueeze_852 = None
    sub_255: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_979);  where_40 = mul_979 = None
    sub_256: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_837);  sub_255 = unsqueeze_837 = None
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
    where_41: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_41, full_default, slice_35);  le_41 = slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_88: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_257: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_858);  convolution_41 = unsqueeze_858 = None
    mul_982: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, sub_257)
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
    mul_988: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_864);  sub_257 = unsqueeze_864 = None
    sub_259: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_41, mul_988);  where_41 = mul_988 = None
    sub_260: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_861);  sub_259 = unsqueeze_861 = None
    mul_989: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_867);  sub_260 = unsqueeze_867 = None
    mul_990: "f32[128]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_124);  sum_89 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_989, getitem_238, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_989 = getitem_238 = primals_124 = None
    getitem_621: "f32[8, 128, 28, 28]" = convolution_backward_43[0]
    getitem_622: "f32[128, 16, 3, 3]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_42: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_42, full_default, slice_34);  le_42 = slice_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_90: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_261: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_870);  convolution_40 = unsqueeze_870 = None
    mul_991: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_261)
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
    mul_997: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_876);  sub_261 = unsqueeze_876 = None
    sub_263: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_42, mul_997);  where_42 = mul_997 = None
    sub_264: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_873);  sub_263 = unsqueeze_873 = None
    mul_998: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_879);  sub_264 = unsqueeze_879 = None
    mul_999: "f32[128]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_121);  sum_91 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_998, getitem_231, primals_121, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_998 = getitem_231 = primals_121 = None
    getitem_624: "f32[8, 128, 28, 28]" = convolution_backward_44[0]
    getitem_625: "f32[128, 16, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_43: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_43, full_default, slice_33);  le_43 = slice_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_92: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_265: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_882);  convolution_39 = unsqueeze_882 = None
    mul_1000: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_265)
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
    mul_1006: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_888);  sub_265 = unsqueeze_888 = None
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
    where_44: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_44, full_default, cat_24);  le_44 = cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_94: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_269: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_894);  convolution_38 = unsqueeze_894 = None
    mul_1009: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_44, sub_269)
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
    mul_1015: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_900);  sub_269 = unsqueeze_900 = None
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
    le_45: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
    where_45: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_45, full_default, add_487);  le_45 = add_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_96: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_273: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_906);  convolution_37 = unsqueeze_906 = None
    mul_1018: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_45, sub_273)
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
    mul_1024: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_912);  sub_273 = unsqueeze_912 = None
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
    where_46: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_46, full_default, slice_39);  le_46 = slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_98: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_277: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_918);  convolution_36 = unsqueeze_918 = None
    mul_1027: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_46, sub_277)
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
    mul_1033: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_924);  sub_277 = unsqueeze_924 = None
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
    where_47: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_47, full_default, add_488);  le_47 = add_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_100: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_281: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_930);  convolution_35 = unsqueeze_930 = None
    mul_1036: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_47, sub_281)
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
    mul_1042: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_936);  sub_281 = unsqueeze_936 = None
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
    where_48: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_48, full_default, add_489);  le_48 = add_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_102: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_285: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_942);  convolution_34 = unsqueeze_942 = None
    mul_1045: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_48, sub_285)
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
    mul_1051: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_948);  sub_285 = unsqueeze_948 = None
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
    where_49: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_49, full_default, cat_25);  le_49 = cat_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_104: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_289: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_954);  convolution_33 = unsqueeze_954 = None
    mul_1054: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_49, sub_289)
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
    mul_1060: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_960);  sub_289 = unsqueeze_960 = None
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
    le_50: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
    where_50: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_50, full_default, add_490);  le_50 = add_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_106: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_293: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_966);  convolution_32 = unsqueeze_966 = None
    mul_1063: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_50, sub_293)
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
    mul_1069: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_972);  sub_293 = unsqueeze_972 = None
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
    where_51: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_51, full_default, slice_43);  le_51 = slice_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_108: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_297: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_978);  convolution_31 = unsqueeze_978 = None
    mul_1072: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_51, sub_297)
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
    mul_1078: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_984);  sub_297 = unsqueeze_984 = None
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
    where_52: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_52, full_default, add_491);  le_52 = add_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_110: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_301: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_990);  convolution_30 = unsqueeze_990 = None
    mul_1081: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_52, sub_301)
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
    mul_1087: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_996);  sub_301 = unsqueeze_996 = None
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
    where_53: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_53, full_default, add_492);  le_53 = add_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_112: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_305: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1002);  convolution_29 = unsqueeze_1002 = None
    mul_1090: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_53, sub_305)
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
    mul_1096: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_1008);  sub_305 = unsqueeze_1008 = None
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
    where_54: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_54, full_default, cat_26);  le_54 = cat_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_114: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_309: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1014);  convolution_28 = unsqueeze_1014 = None
    mul_1099: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_54, sub_309)
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
    mul_1105: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_1020);  sub_309 = unsqueeze_1020 = None
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
    le_55: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    where_55: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_55, full_default, add_493);  le_55 = add_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_116: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_313: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1026);  convolution_27 = unsqueeze_1026 = None
    mul_1108: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_55, sub_313)
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
    mul_1114: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_1032);  sub_313 = unsqueeze_1032 = None
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
    where_56: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_56, full_default, slice_47);  le_56 = slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_118: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_317: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1038);  convolution_26 = unsqueeze_1038 = None
    mul_1117: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_56, sub_317)
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
    mul_1123: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_1044);  sub_317 = unsqueeze_1044 = None
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
    where_57: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_57, full_default, add_494);  le_57 = add_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_120: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_321: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_1050);  convolution_25 = unsqueeze_1050 = None
    mul_1126: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_57, sub_321)
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
    mul_1132: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_1056);  sub_321 = unsqueeze_1056 = None
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
    where_58: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_58, full_default, add_495);  le_58 = add_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_122: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_325: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1062);  convolution_24 = unsqueeze_1062 = None
    mul_1135: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_58, sub_325)
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
    mul_1141: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_1068);  sub_325 = unsqueeze_1068 = None
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
    where_59: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_59, full_default, cat_27);  le_59 = cat_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_124: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_329: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1074);  convolution_23 = unsqueeze_1074 = None
    mul_1144: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_59, sub_329)
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
    mul_1150: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_1080);  sub_329 = unsqueeze_1080 = None
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
    le_60: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    where_60: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_60, full_default, add_496);  le_60 = add_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    sum_126: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_333: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1086);  convolution_22 = unsqueeze_1086 = None
    mul_1153: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, sub_333)
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
    mul_1159: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_1092);  sub_333 = unsqueeze_1092 = None
    sub_335: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_60, mul_1159);  mul_1159 = None
    sub_336: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_335, unsqueeze_1089);  sub_335 = None
    mul_1160: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_1095);  sub_336 = unsqueeze_1095 = None
    mul_1161: "f32[512]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_67);  sum_127 = squeeze_67 = None
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1160, relu_15, primals_67, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1160 = primals_67 = None
    getitem_678: "f32[8, 256, 56, 56]" = convolution_backward_62[0]
    getitem_679: "f32[512, 256, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sub_337: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1098);  convolution_21 = unsqueeze_1098 = None
    mul_1162: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, sub_337)
    sum_129: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1162, [0, 2, 3]);  mul_1162 = None
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
    mul_1168: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_1104);  sub_337 = unsqueeze_1104 = None
    sub_339: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_60, mul_1168);  where_60 = mul_1168 = None
    sub_340: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_339, unsqueeze_1089);  sub_339 = unsqueeze_1089 = None
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
    where_61: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_61, full_default, slice_51);  le_61 = slice_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_130: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_341: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_1110);  convolution_20 = unsqueeze_1110 = None
    mul_1171: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_61, sub_341)
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
    mul_1177: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_1116);  sub_341 = unsqueeze_1116 = None
    sub_343: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_61, mul_1177);  where_61 = mul_1177 = None
    sub_344: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_343, unsqueeze_1113);  sub_343 = unsqueeze_1113 = None
    mul_1178: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_1119);  sub_344 = unsqueeze_1119 = None
    mul_1179: "f32[64]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_61);  sum_131 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1178, getitem_116, primals_61, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1178 = getitem_116 = primals_61 = None
    getitem_684: "f32[8, 64, 56, 56]" = convolution_backward_64[0]
    getitem_685: "f32[64, 8, 3, 3]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_62: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_62, full_default, slice_50);  le_62 = slice_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_132: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_345: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1122);  convolution_19 = unsqueeze_1122 = None
    mul_1180: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_62, sub_345)
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
    mul_1186: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_1128);  sub_345 = unsqueeze_1128 = None
    sub_347: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_62, mul_1186);  where_62 = mul_1186 = None
    sub_348: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_347, unsqueeze_1125);  sub_347 = unsqueeze_1125 = None
    mul_1187: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_1131);  sub_348 = unsqueeze_1131 = None
    mul_1188: "f32[64]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_58);  sum_133 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1187, getitem_109, primals_58, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1187 = getitem_109 = primals_58 = None
    getitem_687: "f32[8, 64, 56, 56]" = convolution_backward_65[0]
    getitem_688: "f32[64, 8, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_63: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_63, full_default, slice_49);  le_63 = slice_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_134: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_349: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1134);  convolution_18 = unsqueeze_1134 = None
    mul_1189: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_63, sub_349)
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
    mul_1195: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_1140);  sub_349 = unsqueeze_1140 = None
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
    where_64: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_64, full_default, cat_28);  le_64 = cat_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_136: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_353: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1146);  convolution_17 = unsqueeze_1146 = None
    mul_1198: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_64, sub_353)
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
    mul_1204: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_353, unsqueeze_1152);  sub_353 = unsqueeze_1152 = None
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
    le_65: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_65: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_65, full_default, add_497);  le_65 = add_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_138: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_357: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1158);  convolution_16 = unsqueeze_1158 = None
    mul_1207: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_65, sub_357)
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
    mul_1213: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_357, unsqueeze_1164);  sub_357 = unsqueeze_1164 = None
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
    where_66: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_66, full_default, slice_55);  le_66 = slice_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_140: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_361: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1170);  convolution_15 = unsqueeze_1170 = None
    mul_1216: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_66, sub_361)
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
    mul_1222: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_361, unsqueeze_1176);  sub_361 = unsqueeze_1176 = None
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
    where_67: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_67, full_default, add_498);  le_67 = add_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_142: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_365: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1182);  convolution_14 = unsqueeze_1182 = None
    mul_1225: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_67, sub_365)
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
    mul_1231: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_365, unsqueeze_1188);  sub_365 = unsqueeze_1188 = None
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
    where_68: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_68, full_default, add_499);  le_68 = add_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_144: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_369: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1194);  convolution_13 = unsqueeze_1194 = None
    mul_1234: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_68, sub_369)
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
    mul_1240: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_369, unsqueeze_1200);  sub_369 = unsqueeze_1200 = None
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
    where_69: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_69, full_default, cat_29);  le_69 = cat_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_146: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_373: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1206);  convolution_12 = unsqueeze_1206 = None
    mul_1243: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_69, sub_373)
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
    mul_1249: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_373, unsqueeze_1212);  sub_373 = unsqueeze_1212 = None
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
    le_70: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_70: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_70, full_default, add_500);  le_70 = add_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sum_148: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_377: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1218);  convolution_11 = unsqueeze_1218 = None
    mul_1252: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_70, sub_377)
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
    mul_1258: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_377, unsqueeze_1224);  sub_377 = unsqueeze_1224 = None
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
    where_71: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_71, full_default, slice_59);  le_71 = slice_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_150: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_381: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1230);  convolution_10 = unsqueeze_1230 = None
    mul_1261: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_71, sub_381)
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
    mul_1267: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_381, unsqueeze_1236);  sub_381 = unsqueeze_1236 = None
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
    where_72: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_72, full_default, add_501);  le_72 = add_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_152: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_385: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1242);  convolution_9 = unsqueeze_1242 = None
    mul_1270: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_72, sub_385)
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
    mul_1276: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_385, unsqueeze_1248);  sub_385 = unsqueeze_1248 = None
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
    where_73: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_73, full_default, add_502);  le_73 = add_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_154: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_389: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1254);  convolution_8 = unsqueeze_1254 = None
    mul_1279: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_73, sub_389)
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
    mul_1285: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_389, unsqueeze_1260);  sub_389 = unsqueeze_1260 = None
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
    where_74: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_74, full_default, cat_30);  le_74 = cat_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_156: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_393: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1266);  convolution_7 = unsqueeze_1266 = None
    mul_1288: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_74, sub_393)
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
    mul_1294: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_393, unsqueeze_1272);  sub_393 = unsqueeze_1272 = None
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
    le_75: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_75: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_75, full_default, add_503);  le_75 = add_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    sum_158: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_397: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1278);  convolution_6 = unsqueeze_1278 = None
    mul_1297: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_75, sub_397)
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
    mul_1303: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_397, unsqueeze_1284);  sub_397 = unsqueeze_1284 = None
    sub_399: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_75, mul_1303);  mul_1303 = None
    sub_400: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_399, unsqueeze_1281);  sub_399 = None
    mul_1304: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_1287);  sub_400 = unsqueeze_1287 = None
    mul_1305: "f32[256]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_19);  sum_159 = squeeze_19 = None
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1304, getitem_2, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1304 = primals_19 = None
    getitem_726: "f32[8, 64, 56, 56]" = convolution_backward_78[0]
    getitem_727: "f32[256, 64, 1, 1]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    sub_401: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1290);  convolution_5 = unsqueeze_1290 = None
    mul_1306: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_75, sub_401)
    sum_161: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1306, [0, 2, 3]);  mul_1306 = None
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
    mul_1312: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_401, unsqueeze_1296);  sub_401 = unsqueeze_1296 = None
    sub_403: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_75, mul_1312);  where_75 = mul_1312 = None
    sub_404: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_403, unsqueeze_1281);  sub_403 = unsqueeze_1281 = None
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
    where_76: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_76, full_default, slice_63);  le_76 = slice_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_162: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_405: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1302);  convolution_4 = unsqueeze_1302 = None
    mul_1315: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_76, sub_405)
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
    mul_1321: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_405, unsqueeze_1308);  sub_405 = unsqueeze_1308 = None
    sub_407: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_76, mul_1321);  where_76 = mul_1321 = None
    sub_408: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_407, unsqueeze_1305);  sub_407 = unsqueeze_1305 = None
    mul_1322: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_408, unsqueeze_1311);  sub_408 = unsqueeze_1311 = None
    mul_1323: "f32[32]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_13);  sum_163 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1322, getitem_24, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1322 = getitem_24 = primals_13 = None
    getitem_732: "f32[8, 32, 56, 56]" = convolution_backward_80[0]
    getitem_733: "f32[32, 4, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_77: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_77, full_default, slice_62);  le_77 = slice_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_164: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_409: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1314);  convolution_3 = unsqueeze_1314 = None
    mul_1324: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_77, sub_409)
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
    mul_1330: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_409, unsqueeze_1320);  sub_409 = unsqueeze_1320 = None
    sub_411: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_77, mul_1330);  where_77 = mul_1330 = None
    sub_412: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_411, unsqueeze_1317);  sub_411 = unsqueeze_1317 = None
    mul_1331: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_412, unsqueeze_1323);  sub_412 = unsqueeze_1323 = None
    mul_1332: "f32[32]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_10);  sum_165 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1331, getitem_17, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1331 = getitem_17 = primals_10 = None
    getitem_735: "f32[8, 32, 56, 56]" = convolution_backward_81[0]
    getitem_736: "f32[32, 4, 3, 3]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    where_78: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_78, full_default, slice_61);  le_78 = slice_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sum_166: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_413: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1326);  convolution_2 = unsqueeze_1326 = None
    mul_1333: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_78, sub_413)
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
    mul_1339: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_413, unsqueeze_1332);  sub_413 = unsqueeze_1332 = None
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
    where_79: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_79, full_default, cat_31);  le_79 = cat_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    sum_168: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_417: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1338);  convolution_1 = unsqueeze_1338 = None
    mul_1342: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_79, sub_417)
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
    mul_1348: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_417, unsqueeze_1344);  sub_417 = unsqueeze_1344 = None
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
    le_80: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_80: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_80, full_default, max_pool2d_with_indices_backward);  le_80 = full_default = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    sum_170: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_421: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1350);  convolution = unsqueeze_1350 = None
    mul_1351: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_80, sub_421)
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
    mul_1357: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_421, unsqueeze_1356);  sub_421 = unsqueeze_1356 = None
    sub_423: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_80, mul_1357);  where_80 = mul_1357 = None
    sub_424: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_423, unsqueeze_1353);  sub_423 = unsqueeze_1353 = None
    mul_1358: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_424, unsqueeze_1359);  sub_424 = unsqueeze_1359 = None
    mul_1359: "f32[64]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_1);  sum_171 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1358, primals_513, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1358 = primals_513 = primals_1 = None
    getitem_745: "f32[64, 3, 7, 7]" = convolution_backward_84[1];  convolution_backward_84 = None
    return [getitem_745, mul_1359, sum_170, getitem_742, mul_1350, sum_168, getitem_739, mul_1341, sum_166, getitem_736, mul_1332, sum_164, getitem_733, mul_1323, sum_162, getitem_730, mul_1314, sum_158, getitem_727, mul_1305, sum_158, getitem_724, mul_1296, sum_156, getitem_721, mul_1287, sum_154, getitem_718, mul_1278, sum_152, getitem_715, mul_1269, sum_150, getitem_712, mul_1260, sum_148, getitem_709, mul_1251, sum_146, getitem_706, mul_1242, sum_144, getitem_703, mul_1233, sum_142, getitem_700, mul_1224, sum_140, getitem_697, mul_1215, sum_138, getitem_694, mul_1206, sum_136, getitem_691, mul_1197, sum_134, getitem_688, mul_1188, sum_132, getitem_685, mul_1179, sum_130, getitem_682, mul_1170, sum_126, getitem_679, mul_1161, sum_126, getitem_676, mul_1152, sum_124, getitem_673, mul_1143, sum_122, getitem_670, mul_1134, sum_120, getitem_667, mul_1125, sum_118, getitem_664, mul_1116, sum_116, getitem_661, mul_1107, sum_114, getitem_658, mul_1098, sum_112, getitem_655, mul_1089, sum_110, getitem_652, mul_1080, sum_108, getitem_649, mul_1071, sum_106, getitem_646, mul_1062, sum_104, getitem_643, mul_1053, sum_102, getitem_640, mul_1044, sum_100, getitem_637, mul_1035, sum_98, getitem_634, mul_1026, sum_96, getitem_631, mul_1017, sum_94, getitem_628, mul_1008, sum_92, getitem_625, mul_999, sum_90, getitem_622, mul_990, sum_88, getitem_619, mul_981, sum_84, getitem_616, mul_972, sum_84, getitem_613, mul_963, sum_82, getitem_610, mul_954, sum_80, getitem_607, mul_945, sum_78, getitem_604, mul_936, sum_76, getitem_601, mul_927, sum_74, getitem_598, mul_918, sum_72, getitem_595, mul_909, sum_70, getitem_592, mul_900, sum_68, getitem_589, mul_891, sum_66, getitem_586, mul_882, sum_64, getitem_583, mul_873, sum_62, getitem_580, mul_864, sum_60, getitem_577, mul_855, sum_58, getitem_574, mul_846, sum_56, getitem_571, mul_837, sum_54, getitem_568, mul_828, sum_52, getitem_565, mul_819, sum_50, getitem_562, mul_810, sum_48, getitem_559, mul_801, sum_46, getitem_556, mul_792, sum_44, getitem_553, mul_783, sum_42, getitem_550, mul_774, sum_40, getitem_547, mul_765, sum_38, getitem_544, mul_756, sum_36, getitem_541, mul_747, sum_34, getitem_538, mul_738, sum_32, getitem_535, mul_729, sum_30, getitem_532, mul_720, sum_28, getitem_529, mul_711, sum_26, getitem_526, mul_702, sum_22, getitem_523, mul_693, sum_22, getitem_520, mul_684, sum_20, getitem_517, mul_675, sum_18, getitem_514, mul_666, sum_16, getitem_511, mul_657, sum_14, getitem_508, mul_648, sum_12, getitem_505, mul_639, sum_10, getitem_502, mul_630, sum_8, getitem_499, mul_621, sum_6, getitem_496, mul_612, sum_4, getitem_493, mul_603, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    