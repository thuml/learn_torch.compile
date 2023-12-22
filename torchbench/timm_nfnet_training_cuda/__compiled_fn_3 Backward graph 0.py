from __future__ import annotations



def forward(self, primals_1: "f32[16, 3, 3, 3]", primals_2: "f32[16, 1, 1, 1]", primals_4: "f32[32, 16, 3, 3]", primals_5: "f32[32, 1, 1, 1]", primals_7: "f32[64, 32, 3, 3]", primals_8: "f32[64, 1, 1, 1]", primals_10: "f32[128, 64, 3, 3]", primals_11: "f32[128, 1, 1, 1]", primals_13: "f32[256, 128, 1, 1]", primals_14: "f32[256, 1, 1, 1]", primals_16: "f32[128, 128, 1, 1]", primals_17: "f32[128, 1, 1, 1]", primals_19: "f32[128, 128, 3, 3]", primals_20: "f32[128, 1, 1, 1]", primals_22: "f32[128, 128, 3, 3]", primals_23: "f32[128, 1, 1, 1]", primals_25: "f32[256, 128, 1, 1]", primals_26: "f32[256, 1, 1, 1]", primals_28: "f32[]", primals_29: "f32[512, 256, 1, 1]", primals_30: "f32[512, 1, 1, 1]", primals_32: "f32[256, 256, 1, 1]", primals_33: "f32[256, 1, 1, 1]", primals_35: "f32[256, 128, 3, 3]", primals_36: "f32[256, 1, 1, 1]", primals_38: "f32[256, 128, 3, 3]", primals_39: "f32[256, 1, 1, 1]", primals_41: "f32[512, 256, 1, 1]", primals_42: "f32[512, 1, 1, 1]", primals_44: "f32[]", primals_45: "f32[256, 512, 1, 1]", primals_46: "f32[256, 1, 1, 1]", primals_48: "f32[256, 128, 3, 3]", primals_49: "f32[256, 1, 1, 1]", primals_51: "f32[256, 128, 3, 3]", primals_52: "f32[256, 1, 1, 1]", primals_54: "f32[512, 256, 1, 1]", primals_55: "f32[512, 1, 1, 1]", primals_57: "f32[]", primals_58: "f32[1536, 512, 1, 1]", primals_59: "f32[1536, 1, 1, 1]", primals_61: "f32[768, 512, 1, 1]", primals_62: "f32[768, 1, 1, 1]", primals_64: "f32[768, 128, 3, 3]", primals_65: "f32[768, 1, 1, 1]", primals_67: "f32[768, 128, 3, 3]", primals_68: "f32[768, 1, 1, 1]", primals_70: "f32[1536, 768, 1, 1]", primals_71: "f32[1536, 1, 1, 1]", primals_73: "f32[]", primals_74: "f32[768, 1536, 1, 1]", primals_75: "f32[768, 1, 1, 1]", primals_77: "f32[768, 128, 3, 3]", primals_78: "f32[768, 1, 1, 1]", primals_80: "f32[768, 128, 3, 3]", primals_81: "f32[768, 1, 1, 1]", primals_83: "f32[1536, 768, 1, 1]", primals_84: "f32[1536, 1, 1, 1]", primals_86: "f32[]", primals_87: "f32[768, 1536, 1, 1]", primals_88: "f32[768, 1, 1, 1]", primals_90: "f32[768, 128, 3, 3]", primals_91: "f32[768, 1, 1, 1]", primals_93: "f32[768, 128, 3, 3]", primals_94: "f32[768, 1, 1, 1]", primals_96: "f32[1536, 768, 1, 1]", primals_97: "f32[1536, 1, 1, 1]", primals_99: "f32[]", primals_100: "f32[768, 1536, 1, 1]", primals_101: "f32[768, 1, 1, 1]", primals_103: "f32[768, 128, 3, 3]", primals_104: "f32[768, 1, 1, 1]", primals_106: "f32[768, 128, 3, 3]", primals_107: "f32[768, 1, 1, 1]", primals_109: "f32[1536, 768, 1, 1]", primals_110: "f32[1536, 1, 1, 1]", primals_112: "f32[]", primals_113: "f32[768, 1536, 1, 1]", primals_114: "f32[768, 1, 1, 1]", primals_116: "f32[768, 128, 3, 3]", primals_117: "f32[768, 1, 1, 1]", primals_119: "f32[768, 128, 3, 3]", primals_120: "f32[768, 1, 1, 1]", primals_122: "f32[1536, 768, 1, 1]", primals_123: "f32[1536, 1, 1, 1]", primals_125: "f32[]", primals_126: "f32[768, 1536, 1, 1]", primals_127: "f32[768, 1, 1, 1]", primals_129: "f32[768, 128, 3, 3]", primals_130: "f32[768, 1, 1, 1]", primals_132: "f32[768, 128, 3, 3]", primals_133: "f32[768, 1, 1, 1]", primals_135: "f32[1536, 768, 1, 1]", primals_136: "f32[1536, 1, 1, 1]", primals_138: "f32[]", primals_139: "f32[1536, 1536, 1, 1]", primals_140: "f32[1536, 1, 1, 1]", primals_142: "f32[768, 1536, 1, 1]", primals_143: "f32[768, 1, 1, 1]", primals_145: "f32[768, 128, 3, 3]", primals_146: "f32[768, 1, 1, 1]", primals_148: "f32[768, 128, 3, 3]", primals_149: "f32[768, 1, 1, 1]", primals_151: "f32[1536, 768, 1, 1]", primals_152: "f32[1536, 1, 1, 1]", primals_154: "f32[]", primals_155: "f32[768, 1536, 1, 1]", primals_156: "f32[768, 1, 1, 1]", primals_158: "f32[768, 128, 3, 3]", primals_159: "f32[768, 1, 1, 1]", primals_161: "f32[768, 128, 3, 3]", primals_162: "f32[768, 1, 1, 1]", primals_164: "f32[1536, 768, 1, 1]", primals_165: "f32[1536, 1, 1, 1]", primals_167: "f32[]", primals_168: "f32[768, 1536, 1, 1]", primals_169: "f32[768, 1, 1, 1]", primals_171: "f32[768, 128, 3, 3]", primals_172: "f32[768, 1, 1, 1]", primals_174: "f32[768, 128, 3, 3]", primals_175: "f32[768, 1, 1, 1]", primals_177: "f32[1536, 768, 1, 1]", primals_178: "f32[1536, 1, 1, 1]", primals_180: "f32[]", primals_181: "f32[3072, 1536, 1, 1]", primals_182: "f32[3072, 1, 1, 1]", primals_184: "f32[128, 256, 1, 1]", primals_186: "f32[256, 128, 1, 1]", primals_188: "f32[256, 512, 1, 1]", primals_190: "f32[512, 256, 1, 1]", primals_192: "f32[256, 512, 1, 1]", primals_194: "f32[512, 256, 1, 1]", primals_196: "f32[768, 1536, 1, 1]", primals_198: "f32[1536, 768, 1, 1]", primals_200: "f32[768, 1536, 1, 1]", primals_202: "f32[1536, 768, 1, 1]", primals_204: "f32[768, 1536, 1, 1]", primals_206: "f32[1536, 768, 1, 1]", primals_208: "f32[768, 1536, 1, 1]", primals_210: "f32[1536, 768, 1, 1]", primals_212: "f32[768, 1536, 1, 1]", primals_214: "f32[1536, 768, 1, 1]", primals_216: "f32[768, 1536, 1, 1]", primals_218: "f32[1536, 768, 1, 1]", primals_220: "f32[768, 1536, 1, 1]", primals_222: "f32[1536, 768, 1, 1]", primals_224: "f32[768, 1536, 1, 1]", primals_226: "f32[1536, 768, 1, 1]", primals_228: "f32[768, 1536, 1, 1]", primals_230: "f32[1536, 768, 1, 1]", constant_pad_nd: "f32[4, 3, 193, 193]", squeeze_1: "f32[16]", view_2: "f32[16, 3, 3, 3]", convolution: "f32[4, 16, 96, 96]", mul_6: "f32[4, 16, 96, 96]", squeeze_3: "f32[32]", view_5: "f32[32, 16, 3, 3]", convolution_1: "f32[4, 32, 96, 96]", mul_13: "f32[4, 32, 96, 96]", squeeze_5: "f32[64]", view_8: "f32[64, 32, 3, 3]", convolution_2: "f32[4, 64, 96, 96]", constant_pad_nd_1: "f32[4, 64, 97, 97]", squeeze_7: "f32[128]", view_11: "f32[128, 64, 3, 3]", convolution_3: "f32[4, 128, 48, 48]", mul_28: "f32[4, 128, 48, 48]", squeeze_9: "f32[256]", view_14: "f32[256, 128, 1, 1]", convolution_4: "f32[4, 256, 48, 48]", squeeze_11: "f32[128]", view_17: "f32[128, 128, 1, 1]", convolution_5: "f32[4, 128, 48, 48]", mul_38: "f32[4, 128, 48, 48]", squeeze_13: "f32[128]", view_20: "f32[128, 128, 3, 3]", convolution_6: "f32[4, 128, 48, 48]", mul_45: "f32[4, 128, 48, 48]", squeeze_15: "f32[128]", view_23: "f32[128, 128, 3, 3]", convolution_7: "f32[4, 128, 48, 48]", mul_52: "f32[4, 128, 48, 48]", squeeze_17: "f32[256]", view_26: "f32[256, 128, 1, 1]", convolution_8: "f32[4, 256, 48, 48]", mean: "f32[4, 256, 1, 1]", relu: "f32[4, 128, 1, 1]", convolution_10: "f32[4, 256, 1, 1]", mul_64: "f32[4, 256, 48, 48]", avg_pool2d: "f32[4, 256, 24, 24]", squeeze_19: "f32[512]", view_29: "f32[512, 256, 1, 1]", convolution_11: "f32[4, 512, 24, 24]", squeeze_21: "f32[256]", view_32: "f32[256, 256, 1, 1]", convolution_12: "f32[4, 256, 48, 48]", constant_pad_nd_2: "f32[4, 256, 49, 49]", squeeze_23: "f32[256]", view_35: "f32[256, 128, 3, 3]", convolution_13: "f32[4, 256, 24, 24]", mul_81: "f32[4, 256, 24, 24]", squeeze_25: "f32[256]", view_38: "f32[256, 128, 3, 3]", convolution_14: "f32[4, 256, 24, 24]", mul_88: "f32[4, 256, 24, 24]", squeeze_27: "f32[512]", view_41: "f32[512, 256, 1, 1]", convolution_15: "f32[4, 512, 24, 24]", mean_1: "f32[4, 512, 1, 1]", relu_1: "f32[4, 256, 1, 1]", convolution_17: "f32[4, 512, 1, 1]", mul_100: "f32[4, 512, 24, 24]", squeeze_29: "f32[256]", view_44: "f32[256, 512, 1, 1]", convolution_18: "f32[4, 256, 24, 24]", mul_107: "f32[4, 256, 24, 24]", squeeze_31: "f32[256]", view_47: "f32[256, 128, 3, 3]", convolution_19: "f32[4, 256, 24, 24]", mul_114: "f32[4, 256, 24, 24]", squeeze_33: "f32[256]", view_50: "f32[256, 128, 3, 3]", convolution_20: "f32[4, 256, 24, 24]", mul_121: "f32[4, 256, 24, 24]", squeeze_35: "f32[512]", view_53: "f32[512, 256, 1, 1]", convolution_21: "f32[4, 512, 24, 24]", mean_2: "f32[4, 512, 1, 1]", relu_2: "f32[4, 256, 1, 1]", convolution_23: "f32[4, 512, 1, 1]", mul_133: "f32[4, 512, 24, 24]", avg_pool2d_1: "f32[4, 512, 12, 12]", squeeze_37: "f32[1536]", view_56: "f32[1536, 512, 1, 1]", convolution_24: "f32[4, 1536, 12, 12]", squeeze_39: "f32[768]", view_59: "f32[768, 512, 1, 1]", convolution_25: "f32[4, 768, 24, 24]", constant_pad_nd_3: "f32[4, 768, 25, 25]", squeeze_41: "f32[768]", view_62: "f32[768, 128, 3, 3]", convolution_26: "f32[4, 768, 12, 12]", mul_150: "f32[4, 768, 12, 12]", squeeze_43: "f32[768]", view_65: "f32[768, 128, 3, 3]", convolution_27: "f32[4, 768, 12, 12]", mul_157: "f32[4, 768, 12, 12]", squeeze_45: "f32[1536]", view_68: "f32[1536, 768, 1, 1]", convolution_28: "f32[4, 1536, 12, 12]", mean_3: "f32[4, 1536, 1, 1]", relu_3: "f32[4, 768, 1, 1]", convolution_30: "f32[4, 1536, 1, 1]", mul_169: "f32[4, 1536, 12, 12]", squeeze_47: "f32[768]", view_71: "f32[768, 1536, 1, 1]", convolution_31: "f32[4, 768, 12, 12]", mul_176: "f32[4, 768, 12, 12]", squeeze_49: "f32[768]", view_74: "f32[768, 128, 3, 3]", convolution_32: "f32[4, 768, 12, 12]", mul_183: "f32[4, 768, 12, 12]", squeeze_51: "f32[768]", view_77: "f32[768, 128, 3, 3]", convolution_33: "f32[4, 768, 12, 12]", mul_190: "f32[4, 768, 12, 12]", squeeze_53: "f32[1536]", view_80: "f32[1536, 768, 1, 1]", convolution_34: "f32[4, 1536, 12, 12]", mean_4: "f32[4, 1536, 1, 1]", relu_4: "f32[4, 768, 1, 1]", convolution_36: "f32[4, 1536, 1, 1]", mul_202: "f32[4, 1536, 12, 12]", squeeze_55: "f32[768]", view_83: "f32[768, 1536, 1, 1]", convolution_37: "f32[4, 768, 12, 12]", mul_209: "f32[4, 768, 12, 12]", squeeze_57: "f32[768]", view_86: "f32[768, 128, 3, 3]", convolution_38: "f32[4, 768, 12, 12]", mul_216: "f32[4, 768, 12, 12]", squeeze_59: "f32[768]", view_89: "f32[768, 128, 3, 3]", convolution_39: "f32[4, 768, 12, 12]", mul_223: "f32[4, 768, 12, 12]", squeeze_61: "f32[1536]", view_92: "f32[1536, 768, 1, 1]", convolution_40: "f32[4, 1536, 12, 12]", mean_5: "f32[4, 1536, 1, 1]", relu_5: "f32[4, 768, 1, 1]", convolution_42: "f32[4, 1536, 1, 1]", mul_235: "f32[4, 1536, 12, 12]", squeeze_63: "f32[768]", view_95: "f32[768, 1536, 1, 1]", convolution_43: "f32[4, 768, 12, 12]", mul_242: "f32[4, 768, 12, 12]", squeeze_65: "f32[768]", view_98: "f32[768, 128, 3, 3]", convolution_44: "f32[4, 768, 12, 12]", mul_249: "f32[4, 768, 12, 12]", squeeze_67: "f32[768]", view_101: "f32[768, 128, 3, 3]", convolution_45: "f32[4, 768, 12, 12]", mul_256: "f32[4, 768, 12, 12]", squeeze_69: "f32[1536]", view_104: "f32[1536, 768, 1, 1]", convolution_46: "f32[4, 1536, 12, 12]", mean_6: "f32[4, 1536, 1, 1]", relu_6: "f32[4, 768, 1, 1]", convolution_48: "f32[4, 1536, 1, 1]", mul_268: "f32[4, 1536, 12, 12]", squeeze_71: "f32[768]", view_107: "f32[768, 1536, 1, 1]", convolution_49: "f32[4, 768, 12, 12]", mul_275: "f32[4, 768, 12, 12]", squeeze_73: "f32[768]", view_110: "f32[768, 128, 3, 3]", convolution_50: "f32[4, 768, 12, 12]", mul_282: "f32[4, 768, 12, 12]", squeeze_75: "f32[768]", view_113: "f32[768, 128, 3, 3]", convolution_51: "f32[4, 768, 12, 12]", mul_289: "f32[4, 768, 12, 12]", squeeze_77: "f32[1536]", view_116: "f32[1536, 768, 1, 1]", convolution_52: "f32[4, 1536, 12, 12]", mean_7: "f32[4, 1536, 1, 1]", relu_7: "f32[4, 768, 1, 1]", convolution_54: "f32[4, 1536, 1, 1]", mul_301: "f32[4, 1536, 12, 12]", squeeze_79: "f32[768]", view_119: "f32[768, 1536, 1, 1]", convolution_55: "f32[4, 768, 12, 12]", mul_308: "f32[4, 768, 12, 12]", squeeze_81: "f32[768]", view_122: "f32[768, 128, 3, 3]", convolution_56: "f32[4, 768, 12, 12]", mul_315: "f32[4, 768, 12, 12]", squeeze_83: "f32[768]", view_125: "f32[768, 128, 3, 3]", convolution_57: "f32[4, 768, 12, 12]", mul_322: "f32[4, 768, 12, 12]", squeeze_85: "f32[1536]", view_128: "f32[1536, 768, 1, 1]", convolution_58: "f32[4, 1536, 12, 12]", mean_8: "f32[4, 1536, 1, 1]", relu_8: "f32[4, 768, 1, 1]", convolution_60: "f32[4, 1536, 1, 1]", mul_334: "f32[4, 1536, 12, 12]", avg_pool2d_2: "f32[4, 1536, 6, 6]", squeeze_87: "f32[1536]", view_131: "f32[1536, 1536, 1, 1]", convolution_61: "f32[4, 1536, 6, 6]", squeeze_89: "f32[768]", view_134: "f32[768, 1536, 1, 1]", convolution_62: "f32[4, 768, 12, 12]", constant_pad_nd_4: "f32[4, 768, 13, 13]", squeeze_91: "f32[768]", view_137: "f32[768, 128, 3, 3]", convolution_63: "f32[4, 768, 6, 6]", mul_351: "f32[4, 768, 6, 6]", squeeze_93: "f32[768]", view_140: "f32[768, 128, 3, 3]", convolution_64: "f32[4, 768, 6, 6]", mul_358: "f32[4, 768, 6, 6]", squeeze_95: "f32[1536]", view_143: "f32[1536, 768, 1, 1]", convolution_65: "f32[4, 1536, 6, 6]", mean_9: "f32[4, 1536, 1, 1]", relu_9: "f32[4, 768, 1, 1]", convolution_67: "f32[4, 1536, 1, 1]", mul_370: "f32[4, 1536, 6, 6]", squeeze_97: "f32[768]", view_146: "f32[768, 1536, 1, 1]", convolution_68: "f32[4, 768, 6, 6]", mul_377: "f32[4, 768, 6, 6]", squeeze_99: "f32[768]", view_149: "f32[768, 128, 3, 3]", convolution_69: "f32[4, 768, 6, 6]", mul_384: "f32[4, 768, 6, 6]", squeeze_101: "f32[768]", view_152: "f32[768, 128, 3, 3]", convolution_70: "f32[4, 768, 6, 6]", mul_391: "f32[4, 768, 6, 6]", squeeze_103: "f32[1536]", view_155: "f32[1536, 768, 1, 1]", convolution_71: "f32[4, 1536, 6, 6]", mean_10: "f32[4, 1536, 1, 1]", relu_10: "f32[4, 768, 1, 1]", convolution_73: "f32[4, 1536, 1, 1]", mul_403: "f32[4, 1536, 6, 6]", squeeze_105: "f32[768]", view_158: "f32[768, 1536, 1, 1]", convolution_74: "f32[4, 768, 6, 6]", mul_410: "f32[4, 768, 6, 6]", squeeze_107: "f32[768]", view_161: "f32[768, 128, 3, 3]", convolution_75: "f32[4, 768, 6, 6]", mul_417: "f32[4, 768, 6, 6]", squeeze_109: "f32[768]", view_164: "f32[768, 128, 3, 3]", convolution_76: "f32[4, 768, 6, 6]", mul_424: "f32[4, 768, 6, 6]", squeeze_111: "f32[1536]", view_167: "f32[1536, 768, 1, 1]", convolution_77: "f32[4, 1536, 6, 6]", mean_11: "f32[4, 1536, 1, 1]", relu_11: "f32[4, 768, 1, 1]", convolution_79: "f32[4, 1536, 1, 1]", add_118: "f32[4, 1536, 6, 6]", squeeze_113: "f32[3072]", view_170: "f32[3072, 1536, 1, 1]", convolution_80: "f32[4, 3072, 6, 6]", clone_12: "f32[4, 3072]", permute_1: "f32[1000, 3072]", unsqueeze_58: "f32[1, 3072, 1]", unsqueeze_66: "f32[1, 1536, 1]", unsqueeze_74: "f32[1, 768, 1]", unsqueeze_82: "f32[1, 768, 1]", unsqueeze_90: "f32[1, 768, 1]", unsqueeze_98: "f32[1, 1536, 1]", unsqueeze_106: "f32[1, 768, 1]", unsqueeze_114: "f32[1, 768, 1]", unsqueeze_122: "f32[1, 768, 1]", unsqueeze_130: "f32[1, 1536, 1]", unsqueeze_138: "f32[1, 768, 1]", unsqueeze_146: "f32[1, 768, 1]", unsqueeze_154: "f32[1, 768, 1]", unsqueeze_162: "f32[1, 1536, 1]", unsqueeze_170: "f32[1, 1536, 1]", unsqueeze_178: "f32[1, 768, 1]", unsqueeze_186: "f32[1, 768, 1]", unsqueeze_194: "f32[1, 768, 1]", unsqueeze_202: "f32[1, 1536, 1]", unsqueeze_210: "f32[1, 768, 1]", unsqueeze_218: "f32[1, 768, 1]", unsqueeze_226: "f32[1, 768, 1]", unsqueeze_234: "f32[1, 1536, 1]", unsqueeze_242: "f32[1, 768, 1]", unsqueeze_250: "f32[1, 768, 1]", unsqueeze_258: "f32[1, 768, 1]", unsqueeze_266: "f32[1, 1536, 1]", unsqueeze_274: "f32[1, 768, 1]", unsqueeze_282: "f32[1, 768, 1]", unsqueeze_290: "f32[1, 768, 1]", unsqueeze_298: "f32[1, 1536, 1]", unsqueeze_306: "f32[1, 768, 1]", unsqueeze_314: "f32[1, 768, 1]", unsqueeze_322: "f32[1, 768, 1]", unsqueeze_330: "f32[1, 1536, 1]", unsqueeze_338: "f32[1, 768, 1]", unsqueeze_346: "f32[1, 768, 1]", unsqueeze_354: "f32[1, 768, 1]", unsqueeze_362: "f32[1, 1536, 1]", unsqueeze_370: "f32[1, 512, 1]", unsqueeze_378: "f32[1, 256, 1]", unsqueeze_386: "f32[1, 256, 1]", unsqueeze_394: "f32[1, 256, 1]", unsqueeze_402: "f32[1, 512, 1]", unsqueeze_410: "f32[1, 256, 1]", unsqueeze_418: "f32[1, 256, 1]", unsqueeze_426: "f32[1, 256, 1]", unsqueeze_434: "f32[1, 512, 1]", unsqueeze_442: "f32[1, 256, 1]", unsqueeze_450: "f32[1, 128, 1]", unsqueeze_458: "f32[1, 128, 1]", unsqueeze_466: "f32[1, 128, 1]", unsqueeze_474: "f32[1, 256, 1]", unsqueeze_482: "f32[1, 128, 1]", unsqueeze_490: "f32[1, 64, 1]", unsqueeze_498: "f32[1, 32, 1]", unsqueeze_506: "f32[1, 16, 1]", tangents_1: "f32[4, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view: "f32[1, 16, 27]" = torch.ops.aten.view.default(primals_1, [1, 16, -1]);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_2, 0.19245008972987526);  primals_2 = None
    view_1: "f32[16]" = torch.ops.aten.view.default(mul, [-1]);  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_4: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(convolution, 0.7071067811865476)
    erf: "f32[4, 16, 96, 96]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
    add_1: "f32[4, 16, 96, 96]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_3: "f32[1, 32, 144]" = torch.ops.aten.view.default(primals_4, [1, 32, -1]);  primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_7: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_5, 0.08333333333333333);  primals_5 = None
    view_4: "f32[32]" = torch.ops.aten.view.default(mul_7, [-1]);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_11: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_1, 0.7071067811865476)
    erf_1: "f32[4, 32, 96, 96]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
    add_3: "f32[4, 32, 96, 96]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_6: "f32[1, 64, 288]" = torch.ops.aten.view.default(primals_7, [1, 64, -1]);  primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_14: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_8, 0.05892556509887896);  primals_8 = None
    view_7: "f32[64]" = torch.ops.aten.view.default(mul_14, [-1]);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_18: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_2, 0.7071067811865476)
    erf_2: "f32[4, 64, 96, 96]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_5: "f32[4, 64, 96, 96]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_9: "f32[1, 128, 576]" = torch.ops.aten.view.default(primals_10, [1, 128, -1]);  primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_21: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_11, 0.041666666666666664);  primals_11 = None
    view_10: "f32[128]" = torch.ops.aten.view.default(mul_21, [-1]);  mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_25: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_3, 0.7071067811865476)
    erf_3: "f32[4, 128, 48, 48]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
    add_7: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_12: "f32[1, 256, 128]" = torch.ops.aten.view.default(primals_13, [1, 256, -1]);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_29: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_14, 0.08838834764831845);  primals_14 = None
    view_13: "f32[256]" = torch.ops.aten.view.default(mul_29, [-1]);  mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_15: "f32[1, 128, 128]" = torch.ops.aten.view.default(primals_16, [1, 128, -1]);  primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_32: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_17, 0.08838834764831845);  primals_17 = None
    view_16: "f32[128]" = torch.ops.aten.view.default(mul_32, [-1]);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_36: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_5, 0.7071067811865476)
    erf_4: "f32[4, 128, 48, 48]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
    add_10: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_18: "f32[1, 128, 1152]" = torch.ops.aten.view.default(primals_19, [1, 128, -1]);  primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_39: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_20, 0.02946278254943948);  primals_20 = None
    view_19: "f32[128]" = torch.ops.aten.view.default(mul_39, [-1]);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_43: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_6, 0.7071067811865476)
    erf_5: "f32[4, 128, 48, 48]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
    add_12: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_21: "f32[1, 128, 1152]" = torch.ops.aten.view.default(primals_22, [1, 128, -1]);  primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_46: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_23, 0.02946278254943948);  primals_23 = None
    view_22: "f32[128]" = torch.ops.aten.view.default(mul_46, [-1]);  mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_50: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_7, 0.7071067811865476)
    erf_6: "f32[4, 128, 48, 48]" = torch.ops.aten.erf.default(mul_50);  mul_50 = None
    add_14: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_24: "f32[1, 256, 128]" = torch.ops.aten.view.default(primals_25, [1, 256, -1]);  primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_53: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_26, 0.08838834764831845);  primals_26 = None
    view_25: "f32[256]" = torch.ops.aten.view.default(mul_53, [-1]);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid: "f32[4, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
    alias_1: "f32[4, 256, 1, 1]" = torch.ops.aten.alias.default(sigmoid)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_56: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_8, sigmoid)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_57: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_56, 2.0);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone: "f32[4, 256, 48, 48]" = torch.ops.aten.clone.default(mul_57)
    mul_58: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_57, primals_28);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_59: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_58, 0.2);  mul_58 = None
    add_16: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul_59, convolution_4);  mul_59 = convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_61: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_16, 0.7071067811865476)
    erf_7: "f32[4, 256, 48, 48]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_17: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_27: "f32[1, 512, 256]" = torch.ops.aten.view.default(primals_29, [1, 512, -1]);  primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_65: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_30, 0.0625);  primals_30 = None
    view_28: "f32[512]" = torch.ops.aten.view.default(mul_65, [-1]);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_30: "f32[1, 256, 256]" = torch.ops.aten.view.default(primals_32, [1, 256, -1]);  primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_68: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_33, 0.0625);  primals_33 = None
    view_31: "f32[256]" = torch.ops.aten.view.default(mul_68, [-1]);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_72: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_12, 0.7071067811865476)
    erf_8: "f32[4, 256, 48, 48]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_20: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_33: "f32[1, 256, 1152]" = torch.ops.aten.view.default(primals_35, [1, 256, -1]);  primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_75: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_36, 0.02946278254943948);  primals_36 = None
    view_34: "f32[256]" = torch.ops.aten.view.default(mul_75, [-1]);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_79: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_13, 0.7071067811865476)
    erf_9: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
    add_22: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_36: "f32[1, 256, 1152]" = torch.ops.aten.view.default(primals_38, [1, 256, -1]);  primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_82: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_39, 0.02946278254943948);  primals_39 = None
    view_37: "f32[256]" = torch.ops.aten.view.default(mul_82, [-1]);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_86: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_14, 0.7071067811865476)
    erf_10: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_86);  mul_86 = None
    add_24: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_39: "f32[1, 512, 256]" = torch.ops.aten.view.default(primals_41, [1, 512, -1]);  primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_89: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_42, 0.0625);  primals_42 = None
    view_40: "f32[512]" = torch.ops.aten.view.default(mul_89, [-1]);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_1: "f32[4, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None
    alias_3: "f32[4, 512, 1, 1]" = torch.ops.aten.alias.default(sigmoid_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_92: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_15, sigmoid_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_93: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_92, 2.0);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_1: "f32[4, 512, 24, 24]" = torch.ops.aten.clone.default(mul_93)
    mul_94: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_93, primals_44);  mul_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_95: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_94, 0.2);  mul_94 = None
    add_26: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_95, convolution_11);  mul_95 = convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_97: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_26, 0.7071067811865476)
    erf_11: "f32[4, 512, 24, 24]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
    add_27: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_42: "f32[1, 256, 512]" = torch.ops.aten.view.default(primals_45, [1, 256, -1]);  primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_101: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_46, 0.04419417382415922);  primals_46 = None
    view_43: "f32[256]" = torch.ops.aten.view.default(mul_101, [-1]);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_105: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476)
    erf_12: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_29: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_45: "f32[1, 256, 1152]" = torch.ops.aten.view.default(primals_48, [1, 256, -1]);  primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_108: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_49, 0.02946278254943948);  primals_49 = None
    view_46: "f32[256]" = torch.ops.aten.view.default(mul_108, [-1]);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_112: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_19, 0.7071067811865476)
    erf_13: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
    add_31: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_48: "f32[1, 256, 1152]" = torch.ops.aten.view.default(primals_51, [1, 256, -1]);  primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_115: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_52, 0.02946278254943948);  primals_52 = None
    view_49: "f32[256]" = torch.ops.aten.view.default(mul_115, [-1]);  mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_119: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_20, 0.7071067811865476)
    erf_14: "f32[4, 256, 24, 24]" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
    add_33: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_51: "f32[1, 512, 256]" = torch.ops.aten.view.default(primals_54, [1, 512, -1]);  primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_122: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_55, 0.0625);  primals_55 = None
    view_52: "f32[512]" = torch.ops.aten.view.default(mul_122, [-1]);  mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2: "f32[4, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    alias_5: "f32[4, 512, 1, 1]" = torch.ops.aten.alias.default(sigmoid_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_125: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_21, sigmoid_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_126: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_125, 2.0);  mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_2: "f32[4, 512, 24, 24]" = torch.ops.aten.clone.default(mul_126)
    mul_127: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_126, primals_57);  mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_128: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_127, 0.2);  mul_127 = None
    add_35: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_128, add_26);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_130: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_35, 0.7071067811865476)
    erf_15: "f32[4, 512, 24, 24]" = torch.ops.aten.erf.default(mul_130);  mul_130 = None
    add_36: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_54: "f32[1, 1536, 512]" = torch.ops.aten.view.default(primals_58, [1, 1536, -1]);  primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_134: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_59, 0.04419417382415922);  primals_59 = None
    view_55: "f32[1536]" = torch.ops.aten.view.default(mul_134, [-1]);  mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_57: "f32[1, 768, 512]" = torch.ops.aten.view.default(primals_61, [1, 768, -1]);  primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_137: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_62, 0.04419417382415922);  primals_62 = None
    view_58: "f32[768]" = torch.ops.aten.view.default(mul_137, [-1]);  mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_141: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_25, 0.7071067811865476)
    erf_16: "f32[4, 768, 24, 24]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
    add_39: "f32[4, 768, 24, 24]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_60: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_64, [1, 768, -1]);  primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_144: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_65, 0.02946278254943948);  primals_65 = None
    view_61: "f32[768]" = torch.ops.aten.view.default(mul_144, [-1]);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_148: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_26, 0.7071067811865476)
    erf_17: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
    add_41: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_63: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_67, [1, 768, -1]);  primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_151: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_68, 0.02946278254943948);  primals_68 = None
    view_64: "f32[768]" = torch.ops.aten.view.default(mul_151, [-1]);  mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_155: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_27, 0.7071067811865476)
    erf_18: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_43: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_66: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_70, [1, 1536, -1]);  primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_158: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_71, 0.03608439182435161);  primals_71 = None
    view_67: "f32[1536]" = torch.ops.aten.view.default(mul_158, [-1]);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_3: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_30);  convolution_30 = None
    alias_7: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_161: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_28, sigmoid_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_162: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_161, 2.0);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_3: "f32[4, 1536, 12, 12]" = torch.ops.aten.clone.default(mul_162)
    mul_163: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_162, primals_73);  mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_164: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_163, 0.2);  mul_163 = None
    add_45: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_164, convolution_24);  mul_164 = convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_166: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_45, 0.7071067811865476)
    erf_19: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_166);  mul_166 = None
    add_46: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_69: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_74, [1, 768, -1]);  primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_170: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_75, 0.02551551815399144);  primals_75 = None
    view_70: "f32[768]" = torch.ops.aten.view.default(mul_170, [-1]);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_174: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_31, 0.7071067811865476)
    erf_20: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
    add_48: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_72: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_77, [1, 768, -1]);  primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_177: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_78, 0.02946278254943948);  primals_78 = None
    view_73: "f32[768]" = torch.ops.aten.view.default(mul_177, [-1]);  mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_181: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_32, 0.7071067811865476)
    erf_21: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
    add_50: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_75: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_80, [1, 768, -1]);  primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_184: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_81, 0.02946278254943948);  primals_81 = None
    view_76: "f32[768]" = torch.ops.aten.view.default(mul_184, [-1]);  mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_188: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_33, 0.7071067811865476)
    erf_22: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_188);  mul_188 = None
    add_52: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_78: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_83, [1, 1536, -1]);  primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_191: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_84, 0.03608439182435161);  primals_84 = None
    view_79: "f32[1536]" = torch.ops.aten.view.default(mul_191, [-1]);  mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None
    alias_9: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_194: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_34, sigmoid_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_195: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_194, 2.0);  mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_4: "f32[4, 1536, 12, 12]" = torch.ops.aten.clone.default(mul_195)
    mul_196: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_195, primals_86);  mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_197: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_196, 0.2);  mul_196 = None
    add_54: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_197, add_45);  mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_199: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_54, 0.7071067811865476)
    erf_23: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_199);  mul_199 = None
    add_55: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_81: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_87, [1, 768, -1]);  primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_203: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_88, 0.02551551815399144);  primals_88 = None
    view_82: "f32[768]" = torch.ops.aten.view.default(mul_203, [-1]);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_207: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_37, 0.7071067811865476)
    erf_24: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
    add_57: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_84: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_90, [1, 768, -1]);  primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_210: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_91, 0.02946278254943948);  primals_91 = None
    view_85: "f32[768]" = torch.ops.aten.view.default(mul_210, [-1]);  mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_214: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_38, 0.7071067811865476)
    erf_25: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_214);  mul_214 = None
    add_59: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_87: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_93, [1, 768, -1]);  primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_217: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_94, 0.02946278254943948);  primals_94 = None
    view_88: "f32[768]" = torch.ops.aten.view.default(mul_217, [-1]);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_221: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_39, 0.7071067811865476)
    erf_26: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
    add_61: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_90: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_96, [1, 1536, -1]);  primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_224: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_97, 0.03608439182435161);  primals_97 = None
    view_91: "f32[1536]" = torch.ops.aten.view.default(mul_224, [-1]);  mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42);  convolution_42 = None
    alias_11: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_227: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_40, sigmoid_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_228: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_227, 2.0);  mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_5: "f32[4, 1536, 12, 12]" = torch.ops.aten.clone.default(mul_228)
    mul_229: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_228, primals_99);  mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_230: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_229, 0.2);  mul_229 = None
    add_63: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_230, add_54);  mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_232: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_63, 0.7071067811865476)
    erf_27: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_232);  mul_232 = None
    add_64: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_93: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_100, [1, 768, -1]);  primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_236: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_101, 0.02551551815399144);  primals_101 = None
    view_94: "f32[768]" = torch.ops.aten.view.default(mul_236, [-1]);  mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_240: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_43, 0.7071067811865476)
    erf_28: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_240);  mul_240 = None
    add_66: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_96: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_103, [1, 768, -1]);  primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_243: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_104, 0.02946278254943948);  primals_104 = None
    view_97: "f32[768]" = torch.ops.aten.view.default(mul_243, [-1]);  mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_247: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_44, 0.7071067811865476)
    erf_29: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_247);  mul_247 = None
    add_68: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_99: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_106, [1, 768, -1]);  primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_250: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_107, 0.02946278254943948);  primals_107 = None
    view_100: "f32[768]" = torch.ops.aten.view.default(mul_250, [-1]);  mul_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_254: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_45, 0.7071067811865476)
    erf_30: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_254);  mul_254 = None
    add_70: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_102: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_109, [1, 1536, -1]);  primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_257: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_110, 0.03608439182435161);  primals_110 = None
    view_103: "f32[1536]" = torch.ops.aten.view.default(mul_257, [-1]);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_6: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
    alias_13: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_260: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_46, sigmoid_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_261: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_260, 2.0);  mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_6: "f32[4, 1536, 12, 12]" = torch.ops.aten.clone.default(mul_261)
    mul_262: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_261, primals_112);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_263: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_262, 0.2);  mul_262 = None
    add_72: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_263, add_63);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_265: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_72, 0.7071067811865476)
    erf_31: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_265);  mul_265 = None
    add_73: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_105: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_113, [1, 768, -1]);  primals_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_269: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_114, 0.02551551815399144);  primals_114 = None
    view_106: "f32[768]" = torch.ops.aten.view.default(mul_269, [-1]);  mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_273: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_49, 0.7071067811865476)
    erf_32: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_273);  mul_273 = None
    add_75: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_108: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_116, [1, 768, -1]);  primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_276: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_117, 0.02946278254943948);  primals_117 = None
    view_109: "f32[768]" = torch.ops.aten.view.default(mul_276, [-1]);  mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_280: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_50, 0.7071067811865476)
    erf_33: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_280);  mul_280 = None
    add_77: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_111: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_119, [1, 768, -1]);  primals_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_283: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_120, 0.02946278254943948);  primals_120 = None
    view_112: "f32[768]" = torch.ops.aten.view.default(mul_283, [-1]);  mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_287: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_51, 0.7071067811865476)
    erf_34: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_287);  mul_287 = None
    add_79: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_114: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_122, [1, 1536, -1]);  primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_290: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_123, 0.03608439182435161);  primals_123 = None
    view_115: "f32[1536]" = torch.ops.aten.view.default(mul_290, [-1]);  mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_54);  convolution_54 = None
    alias_15: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_293: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_52, sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_294: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_293, 2.0);  mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_7: "f32[4, 1536, 12, 12]" = torch.ops.aten.clone.default(mul_294)
    mul_295: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_294, primals_125);  mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_296: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_295, 0.2);  mul_295 = None
    add_81: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_296, add_72);  mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_298: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_81, 0.7071067811865476)
    erf_35: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_298);  mul_298 = None
    add_82: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_117: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_126, [1, 768, -1]);  primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_302: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_127, 0.02551551815399144);  primals_127 = None
    view_118: "f32[768]" = torch.ops.aten.view.default(mul_302, [-1]);  mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_306: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_55, 0.7071067811865476)
    erf_36: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_306);  mul_306 = None
    add_84: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_120: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_129, [1, 768, -1]);  primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_309: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_130, 0.02946278254943948);  primals_130 = None
    view_121: "f32[768]" = torch.ops.aten.view.default(mul_309, [-1]);  mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_313: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_56, 0.7071067811865476)
    erf_37: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_313);  mul_313 = None
    add_86: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_123: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_132, [1, 768, -1]);  primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_316: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_133, 0.02946278254943948);  primals_133 = None
    view_124: "f32[768]" = torch.ops.aten.view.default(mul_316, [-1]);  mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_320: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_57, 0.7071067811865476)
    erf_38: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_320);  mul_320 = None
    add_88: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_126: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_135, [1, 1536, -1]);  primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_323: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_136, 0.03608439182435161);  primals_136 = None
    view_127: "f32[1536]" = torch.ops.aten.view.default(mul_323, [-1]);  mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_8: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60);  convolution_60 = None
    alias_17: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_326: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_58, sigmoid_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_327: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_326, 2.0);  mul_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_8: "f32[4, 1536, 12, 12]" = torch.ops.aten.clone.default(mul_327)
    mul_328: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_327, primals_138);  mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_329: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_328, 0.2);  mul_328 = None
    add_90: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_329, add_81);  mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_331: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_90, 0.7071067811865476)
    erf_39: "f32[4, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_331);  mul_331 = None
    add_91: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_129: "f32[1, 1536, 1536]" = torch.ops.aten.view.default(primals_139, [1, 1536, -1]);  primals_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_335: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_140, 0.02551551815399144);  primals_140 = None
    view_130: "f32[1536]" = torch.ops.aten.view.default(mul_335, [-1]);  mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_132: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_142, [1, 768, -1]);  primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_338: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_143, 0.02551551815399144);  primals_143 = None
    view_133: "f32[768]" = torch.ops.aten.view.default(mul_338, [-1]);  mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_342: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_62, 0.7071067811865476)
    erf_40: "f32[4, 768, 12, 12]" = torch.ops.aten.erf.default(mul_342);  mul_342 = None
    add_94: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_135: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_145, [1, 768, -1]);  primals_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_345: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_146, 0.02946278254943948);  primals_146 = None
    view_136: "f32[768]" = torch.ops.aten.view.default(mul_345, [-1]);  mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_349: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_63, 0.7071067811865476)
    erf_41: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_349);  mul_349 = None
    add_96: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_138: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_148, [1, 768, -1]);  primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_352: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_149, 0.02946278254943948);  primals_149 = None
    view_139: "f32[768]" = torch.ops.aten.view.default(mul_352, [-1]);  mul_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_356: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_64, 0.7071067811865476)
    erf_42: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_356);  mul_356 = None
    add_98: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_141: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_151, [1, 1536, -1]);  primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_359: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_152, 0.03608439182435161);  primals_152 = None
    view_142: "f32[1536]" = torch.ops.aten.view.default(mul_359, [-1]);  mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None
    alias_19: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_362: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_65, sigmoid_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_363: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_362, 2.0);  mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_9: "f32[4, 1536, 6, 6]" = torch.ops.aten.clone.default(mul_363)
    mul_364: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_363, primals_154);  mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_365: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_364, 0.2);  mul_364 = None
    add_100: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_365, convolution_61);  mul_365 = convolution_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_367: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_100, 0.7071067811865476)
    erf_43: "f32[4, 1536, 6, 6]" = torch.ops.aten.erf.default(mul_367);  mul_367 = None
    add_101: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_144: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_155, [1, 768, -1]);  primals_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_371: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_156, 0.02551551815399144);  primals_156 = None
    view_145: "f32[768]" = torch.ops.aten.view.default(mul_371, [-1]);  mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_375: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_68, 0.7071067811865476)
    erf_44: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_375);  mul_375 = None
    add_103: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_147: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_158, [1, 768, -1]);  primals_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_378: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_159, 0.02946278254943948);  primals_159 = None
    view_148: "f32[768]" = torch.ops.aten.view.default(mul_378, [-1]);  mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_382: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_69, 0.7071067811865476)
    erf_45: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_382);  mul_382 = None
    add_105: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_150: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_161, [1, 768, -1]);  primals_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_385: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_162, 0.02946278254943948);  primals_162 = None
    view_151: "f32[768]" = torch.ops.aten.view.default(mul_385, [-1]);  mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_389: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_70, 0.7071067811865476)
    erf_46: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_389);  mul_389 = None
    add_107: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_153: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_164, [1, 1536, -1]);  primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_392: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_165, 0.03608439182435161);  primals_165 = None
    view_154: "f32[1536]" = torch.ops.aten.view.default(mul_392, [-1]);  mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_10: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
    alias_21: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_395: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_71, sigmoid_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_396: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_395, 2.0);  mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_10: "f32[4, 1536, 6, 6]" = torch.ops.aten.clone.default(mul_396)
    mul_397: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_396, primals_167);  mul_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_398: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_397, 0.2);  mul_397 = None
    add_109: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_398, add_100);  mul_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_400: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_109, 0.7071067811865476)
    erf_47: "f32[4, 1536, 6, 6]" = torch.ops.aten.erf.default(mul_400);  mul_400 = None
    add_110: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_156: "f32[1, 768, 1536]" = torch.ops.aten.view.default(primals_168, [1, 768, -1]);  primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_404: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_169, 0.02551551815399144);  primals_169 = None
    view_157: "f32[768]" = torch.ops.aten.view.default(mul_404, [-1]);  mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_408: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_74, 0.7071067811865476)
    erf_48: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_408);  mul_408 = None
    add_112: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_159: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_171, [1, 768, -1]);  primals_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_411: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_172, 0.02946278254943948);  primals_172 = None
    view_160: "f32[768]" = torch.ops.aten.view.default(mul_411, [-1]);  mul_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_415: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_75, 0.7071067811865476)
    erf_49: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_415);  mul_415 = None
    add_114: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_162: "f32[1, 768, 1152]" = torch.ops.aten.view.default(primals_174, [1, 768, -1]);  primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_418: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_175, 0.02946278254943948);  primals_175 = None
    view_163: "f32[768]" = torch.ops.aten.view.default(mul_418, [-1]);  mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_422: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_76, 0.7071067811865476)
    erf_50: "f32[4, 768, 6, 6]" = torch.ops.aten.erf.default(mul_422);  mul_422 = None
    add_116: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_165: "f32[1, 1536, 768]" = torch.ops.aten.view.default(primals_177, [1, 1536, -1]);  primals_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_425: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_178, 0.03608439182435161);  primals_178 = None
    view_166: "f32[1536]" = torch.ops.aten.view.default(mul_425, [-1]);  mul_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_11: "f32[4, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_79);  convolution_79 = None
    alias_23: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(sigmoid_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_428: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_429: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_428, 2.0);  mul_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    clone_11: "f32[4, 1536, 6, 6]" = torch.ops.aten.clone.default(mul_429);  mul_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_168: "f32[1, 3072, 1536]" = torch.ops.aten.view.default(primals_181, [1, 3072, -1]);  primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    mul_432: "f32[3072, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_182, 0.02551551815399144);  primals_182 = None
    view_169: "f32[3072]" = torch.ops.aten.view.default(mul_432, [-1]);  mul_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_436: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, 0.7071067811865476)
    erf_51: "f32[4, 3072, 6, 6]" = torch.ops.aten.erf.default(mul_436);  mul_436 = None
    add_120: "f32[4, 3072, 6, 6]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm: "f32[4, 3072]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 3072]" = torch.ops.aten.mm.default(permute_2, clone_12);  permute_2 = clone_12 = None
    permute_3: "f32[3072, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_172: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 3072]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_173: "f32[4, 3072, 1, 1]" = torch.ops.aten.view.default(mm, [4, 3072, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[4, 3072, 6, 6]" = torch.ops.aten.expand.default(view_173, [4, 3072, 6, 6]);  view_173 = None
    div: "f32[4, 3072, 6, 6]" = torch.ops.aten.div.Scalar(expand, 36);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_439: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(div, 1.7015043497085571);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_441: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(add_120, 0.5);  add_120 = None
    mul_442: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, convolution_80)
    mul_443: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_442, -0.5);  mul_442 = None
    exp: "f32[4, 3072, 6, 6]" = torch.ops.aten.exp.default(mul_443);  mul_443 = None
    mul_444: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_445: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, mul_444);  convolution_80 = mul_444 = None
    add_122: "f32[4, 3072, 6, 6]" = torch.ops.aten.add.Tensor(mul_441, mul_445);  mul_441 = mul_445 = None
    mul_446: "f32[4, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_439, add_122);  mul_439 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_2: "f32[3072]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_446, add_118, view_170, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_446 = add_118 = view_170 = None
    getitem_114: "f32[4, 1536, 6, 6]" = convolution_backward[0]
    getitem_115: "f32[3072, 1536, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_174: "f32[1, 3072, 1536]" = torch.ops.aten.view.default(getitem_115, [1, 3072, 1536]);  getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_3: "f32[3072]" = torch.ops.aten.sum.dim_IntList(view_174, [0, 2])
    sub_57: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(view_168, unsqueeze_58);  view_168 = unsqueeze_58 = None
    mul_447: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(view_174, sub_57)
    sum_4: "f32[3072]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 2]);  mul_447 = None
    mul_448: "f32[3072]" = torch.ops.aten.mul.Tensor(sum_3, 0.0006510416666666666);  sum_3 = None
    unsqueeze_59: "f32[1, 3072]" = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
    unsqueeze_60: "f32[1, 3072, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 2);  unsqueeze_59 = None
    mul_449: "f32[3072]" = torch.ops.aten.mul.Tensor(sum_4, 0.0006510416666666666)
    mul_450: "f32[3072]" = torch.ops.aten.mul.Tensor(squeeze_113, squeeze_113)
    mul_451: "f32[3072]" = torch.ops.aten.mul.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    unsqueeze_61: "f32[1, 3072]" = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
    unsqueeze_62: "f32[1, 3072, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 2);  unsqueeze_61 = None
    mul_452: "f32[3072]" = torch.ops.aten.mul.Tensor(squeeze_113, view_169);  view_169 = None
    unsqueeze_63: "f32[1, 3072]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_64: "f32[1, 3072, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 2);  unsqueeze_63 = None
    mul_453: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_62);  sub_57 = unsqueeze_62 = None
    sub_59: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(view_174, mul_453);  view_174 = mul_453 = None
    sub_60: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(sub_59, unsqueeze_60);  sub_59 = unsqueeze_60 = None
    mul_454: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_64);  sub_60 = unsqueeze_64 = None
    mul_455: "f32[3072]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_113);  sum_4 = squeeze_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_175: "f32[3072, 1, 1, 1]" = torch.ops.aten.view.default(mul_455, [3072, 1, 1, 1]);  mul_455 = None
    mul_456: "f32[3072, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_175, 0.02551551815399144);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_176: "f32[3072, 1536, 1, 1]" = torch.ops.aten.view.default(mul_454, [3072, 1536, 1, 1]);  mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_457: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_114, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_458: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_457, clone_11);  clone_11 = None
    mul_459: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_457, primals_180);  mul_457 = primals_180 = None
    sum_5: "f32[]" = torch.ops.aten.sum.default(mul_458);  mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_460: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_459, 2.0);  mul_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_461: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_460, convolution_77);  convolution_77 = None
    mul_462: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_460, sigmoid_11);  mul_460 = sigmoid_11 = None
    sum_6: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2, 3], True);  mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_24: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    sub_61: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_24)
    mul_463: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_24, sub_61);  alias_24 = sub_61 = None
    mul_464: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, mul_463);  sum_6 = mul_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_7: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_464, relu_11, primals_230, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_464 = primals_230 = None
    getitem_117: "f32[4, 768, 1, 1]" = convolution_backward_1[0]
    getitem_118: "f32[1536, 768, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_26: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_27: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    le: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_27, 0);  alias_27 = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le, full_default, getitem_117);  le = getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where, mean_11, primals_228, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = mean_11 = primals_228 = None
    getitem_120: "f32[4, 1536, 1, 1]" = convolution_backward_2[0]
    getitem_121: "f32[768, 1536, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[4, 1536, 6, 6]" = torch.ops.aten.expand.default(getitem_120, [4, 1536, 6, 6]);  getitem_120 = None
    div_1: "f32[4, 1536, 6, 6]" = torch.ops.aten.div.Scalar(expand_1, 36);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_123: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_462, div_1);  mul_462 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_9: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(add_123, mul_424, view_167, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_123 = mul_424 = view_167 = None
    getitem_123: "f32[4, 768, 6, 6]" = convolution_backward_3[0]
    getitem_124: "f32[1536, 768, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_177: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_124, [1, 1536, 768]);  getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_10: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_177, [0, 2])
    sub_62: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_165, unsqueeze_66);  view_165 = unsqueeze_66 = None
    mul_465: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_177, sub_62)
    sum_11: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_465, [0, 2]);  mul_465 = None
    mul_466: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_10, 0.0013020833333333333);  sum_10 = None
    unsqueeze_67: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_466, 0);  mul_466 = None
    unsqueeze_68: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 2);  unsqueeze_67 = None
    mul_467: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_11, 0.0013020833333333333)
    mul_468: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_111, squeeze_111)
    mul_469: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_467, mul_468);  mul_467 = mul_468 = None
    unsqueeze_69: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
    unsqueeze_70: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 2);  unsqueeze_69 = None
    mul_470: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_111, view_166);  view_166 = None
    unsqueeze_71: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_470, 0);  mul_470 = None
    unsqueeze_72: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_71, 2);  unsqueeze_71 = None
    mul_471: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_70);  sub_62 = unsqueeze_70 = None
    sub_64: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_177, mul_471);  view_177 = mul_471 = None
    sub_65: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_64, unsqueeze_68);  sub_64 = unsqueeze_68 = None
    mul_472: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_72);  sub_65 = unsqueeze_72 = None
    mul_473: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_111);  sum_11 = squeeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_178: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_473, [1536, 1, 1, 1]);  mul_473 = None
    mul_474: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_178, 0.03608439182435161);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_179: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_472, [1536, 768, 1, 1]);  mul_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_475: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_123, 1.7015043497085571);  getitem_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_477: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_116, 0.5);  add_116 = None
    mul_478: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_76, convolution_76)
    mul_479: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_478, -0.5);  mul_478 = None
    exp_1: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_479);  mul_479 = None
    mul_480: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_481: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_76, mul_480);  convolution_76 = mul_480 = None
    add_125: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_477, mul_481);  mul_477 = mul_481 = None
    mul_482: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_475, add_125);  mul_475 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 2, 3])
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_482, mul_417, view_164, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_482 = mul_417 = view_164 = None
    getitem_126: "f32[4, 768, 6, 6]" = convolution_backward_4[0]
    getitem_127: "f32[768, 128, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_180: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_127, [1, 768, 1152]);  getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_180, [0, 2])
    sub_66: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_162, unsqueeze_74);  view_162 = unsqueeze_74 = None
    mul_483: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_180, sub_66)
    sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_483, [0, 2]);  mul_483 = None
    mul_484: "f32[768]" = torch.ops.aten.mul.Tensor(sum_13, 0.0008680555555555555);  sum_13 = None
    unsqueeze_75: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
    unsqueeze_76: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 2);  unsqueeze_75 = None
    mul_485: "f32[768]" = torch.ops.aten.mul.Tensor(sum_14, 0.0008680555555555555)
    mul_486: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_487: "f32[768]" = torch.ops.aten.mul.Tensor(mul_485, mul_486);  mul_485 = mul_486 = None
    unsqueeze_77: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
    unsqueeze_78: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_77, 2);  unsqueeze_77 = None
    mul_488: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_109, view_163);  view_163 = None
    unsqueeze_79: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_488, 0);  mul_488 = None
    unsqueeze_80: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 2);  unsqueeze_79 = None
    mul_489: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_78);  sub_66 = unsqueeze_78 = None
    sub_68: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_180, mul_489);  view_180 = mul_489 = None
    sub_69: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_76);  sub_68 = unsqueeze_76 = None
    mul_490: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_80);  sub_69 = unsqueeze_80 = None
    mul_491: "f32[768]" = torch.ops.aten.mul.Tensor(sum_14, squeeze_109);  sum_14 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_181: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_491, [768, 1, 1, 1]);  mul_491 = None
    mul_492: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_181, 0.02946278254943948);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_182: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_490, [768, 128, 3, 3]);  mul_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_493: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_126, 1.7015043497085571);  getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_495: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_114, 0.5);  add_114 = None
    mul_496: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_75, convolution_75)
    mul_497: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_496, -0.5);  mul_496 = None
    exp_2: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_497);  mul_497 = None
    mul_498: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_499: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_75, mul_498);  convolution_75 = mul_498 = None
    add_127: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_495, mul_499);  mul_495 = mul_499 = None
    mul_500: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_493, add_127);  mul_493 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3])
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_500, mul_410, view_161, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_500 = mul_410 = view_161 = None
    getitem_129: "f32[4, 768, 6, 6]" = convolution_backward_5[0]
    getitem_130: "f32[768, 128, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_183: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_130, [1, 768, 1152]);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_183, [0, 2])
    sub_70: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_159, unsqueeze_82);  view_159 = unsqueeze_82 = None
    mul_501: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_183, sub_70)
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_501, [0, 2]);  mul_501 = None
    mul_502: "f32[768]" = torch.ops.aten.mul.Tensor(sum_16, 0.0008680555555555555);  sum_16 = None
    unsqueeze_83: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
    unsqueeze_84: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_83, 2);  unsqueeze_83 = None
    mul_503: "f32[768]" = torch.ops.aten.mul.Tensor(sum_17, 0.0008680555555555555)
    mul_504: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_107, squeeze_107)
    mul_505: "f32[768]" = torch.ops.aten.mul.Tensor(mul_503, mul_504);  mul_503 = mul_504 = None
    unsqueeze_85: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
    unsqueeze_86: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 2);  unsqueeze_85 = None
    mul_506: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_107, view_160);  view_160 = None
    unsqueeze_87: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_506, 0);  mul_506 = None
    unsqueeze_88: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
    mul_507: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_86);  sub_70 = unsqueeze_86 = None
    sub_72: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_183, mul_507);  view_183 = mul_507 = None
    sub_73: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_72, unsqueeze_84);  sub_72 = unsqueeze_84 = None
    mul_508: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_88);  sub_73 = unsqueeze_88 = None
    mul_509: "f32[768]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_107);  sum_17 = squeeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_184: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_509, [768, 1, 1, 1]);  mul_509 = None
    mul_510: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_184, 0.02946278254943948);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_185: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_508, [768, 128, 3, 3]);  mul_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_511: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_129, 1.7015043497085571);  getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_513: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_112, 0.5);  add_112 = None
    mul_514: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_74, convolution_74)
    mul_515: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_514, -0.5);  mul_514 = None
    exp_3: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_515);  mul_515 = None
    mul_516: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_517: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_74, mul_516);  convolution_74 = mul_516 = None
    add_129: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_513, mul_517);  mul_513 = mul_517 = None
    mul_518: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_511, add_129);  mul_511 = add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3])
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_518, mul_403, view_158, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_518 = mul_403 = view_158 = None
    getitem_132: "f32[4, 1536, 6, 6]" = convolution_backward_6[0]
    getitem_133: "f32[768, 1536, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_186: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_133, [1, 768, 1536]);  getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_186, [0, 2])
    sub_74: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_156, unsqueeze_90);  view_156 = unsqueeze_90 = None
    mul_519: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_186, sub_74)
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2]);  mul_519 = None
    mul_520: "f32[768]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006510416666666666);  sum_19 = None
    unsqueeze_91: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_92: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 2);  unsqueeze_91 = None
    mul_521: "f32[768]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006510416666666666)
    mul_522: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_105, squeeze_105)
    mul_523: "f32[768]" = torch.ops.aten.mul.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
    unsqueeze_93: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
    unsqueeze_94: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None
    mul_524: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_105, view_157);  view_157 = None
    unsqueeze_95: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
    unsqueeze_96: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
    mul_525: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_94);  sub_74 = unsqueeze_94 = None
    sub_76: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_186, mul_525);  view_186 = mul_525 = None
    sub_77: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_76, unsqueeze_92);  sub_76 = unsqueeze_92 = None
    mul_526: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_96);  sub_77 = unsqueeze_96 = None
    mul_527: "f32[768]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_105);  sum_20 = squeeze_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_187: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_527, [768, 1, 1, 1]);  mul_527 = None
    mul_528: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_187, 0.02551551815399144);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_188: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_526, [768, 1536, 1, 1]);  mul_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_529: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_132, 0.9622504486493761);  getitem_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_530: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_529, 1.7015043497085571);  mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_532: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_110, 0.5);  add_110 = None
    mul_533: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_109, add_109)
    mul_534: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_533, -0.5);  mul_533 = None
    exp_4: "f32[4, 1536, 6, 6]" = torch.ops.aten.exp.default(mul_534);  mul_534 = None
    mul_535: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_536: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_109, mul_535);  add_109 = mul_535 = None
    add_131: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_532, mul_536);  mul_532 = mul_536 = None
    mul_537: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_530, add_131);  mul_530 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_132: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(getitem_114, mul_537);  getitem_114 = mul_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_538: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_132, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_539: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_538, clone_10);  clone_10 = None
    mul_540: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_538, primals_167);  mul_538 = primals_167 = None
    sum_21: "f32[]" = torch.ops.aten.sum.default(mul_539);  mul_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_541: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_540, 2.0);  mul_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_542: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_541, convolution_71);  convolution_71 = None
    mul_543: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_541, sigmoid_10);  mul_541 = sigmoid_10 = None
    sum_22: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_542, [2, 3], True);  mul_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_28: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    sub_78: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_28)
    mul_544: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_28, sub_78);  alias_28 = sub_78 = None
    mul_545: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_22, mul_544);  sum_22 = mul_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_23: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_545, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_545, relu_10, primals_226, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_545 = primals_226 = None
    getitem_135: "f32[4, 768, 1, 1]" = convolution_backward_7[0]
    getitem_136: "f32[1536, 768, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_30: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_31: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    le_1: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_31, 0);  alias_31 = None
    where_1: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_1, full_default, getitem_135);  le_1 = getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_1, mean_10, primals_224, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = mean_10 = primals_224 = None
    getitem_138: "f32[4, 1536, 1, 1]" = convolution_backward_8[0]
    getitem_139: "f32[768, 1536, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[4, 1536, 6, 6]" = torch.ops.aten.expand.default(getitem_138, [4, 1536, 6, 6]);  getitem_138 = None
    div_2: "f32[4, 1536, 6, 6]" = torch.ops.aten.div.Scalar(expand_2, 36);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_133: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_543, div_2);  mul_543 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_25: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 2, 3])
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(add_133, mul_391, view_155, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_133 = mul_391 = view_155 = None
    getitem_141: "f32[4, 768, 6, 6]" = convolution_backward_9[0]
    getitem_142: "f32[1536, 768, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_189: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_142, [1, 1536, 768]);  getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_26: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_189, [0, 2])
    sub_79: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_153, unsqueeze_98);  view_153 = unsqueeze_98 = None
    mul_546: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_189, sub_79)
    sum_27: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_546, [0, 2]);  mul_546 = None
    mul_547: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_26, 0.0013020833333333333);  sum_26 = None
    unsqueeze_99: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
    unsqueeze_100: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None
    mul_548: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_27, 0.0013020833333333333)
    mul_549: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_550: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_548, mul_549);  mul_548 = mul_549 = None
    unsqueeze_101: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_550, 0);  mul_550 = None
    unsqueeze_102: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 2);  unsqueeze_101 = None
    mul_551: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_103, view_154);  view_154 = None
    unsqueeze_103: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
    unsqueeze_104: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 2);  unsqueeze_103 = None
    mul_552: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_102);  sub_79 = unsqueeze_102 = None
    sub_81: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_189, mul_552);  view_189 = mul_552 = None
    sub_82: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_100);  sub_81 = unsqueeze_100 = None
    mul_553: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_104);  sub_82 = unsqueeze_104 = None
    mul_554: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_103);  sum_27 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_190: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_554, [1536, 1, 1, 1]);  mul_554 = None
    mul_555: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_190, 0.03608439182435161);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_191: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_553, [1536, 768, 1, 1]);  mul_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_556: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_141, 1.7015043497085571);  getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_558: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_107, 0.5);  add_107 = None
    mul_559: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_70, convolution_70)
    mul_560: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_559, -0.5);  mul_559 = None
    exp_5: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_560);  mul_560 = None
    mul_561: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_562: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_70, mul_561);  convolution_70 = mul_561 = None
    add_135: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_558, mul_562);  mul_558 = mul_562 = None
    mul_563: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_556, add_135);  mul_556 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 2, 3])
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_563, mul_384, view_152, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_563 = mul_384 = view_152 = None
    getitem_144: "f32[4, 768, 6, 6]" = convolution_backward_10[0]
    getitem_145: "f32[768, 128, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_192: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_145, [1, 768, 1152]);  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_192, [0, 2])
    sub_83: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_150, unsqueeze_106);  view_150 = unsqueeze_106 = None
    mul_564: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_192, sub_83)
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_564, [0, 2]);  mul_564 = None
    mul_565: "f32[768]" = torch.ops.aten.mul.Tensor(sum_29, 0.0008680555555555555);  sum_29 = None
    unsqueeze_107: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
    unsqueeze_108: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 2);  unsqueeze_107 = None
    mul_566: "f32[768]" = torch.ops.aten.mul.Tensor(sum_30, 0.0008680555555555555)
    mul_567: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_101, squeeze_101)
    mul_568: "f32[768]" = torch.ops.aten.mul.Tensor(mul_566, mul_567);  mul_566 = mul_567 = None
    unsqueeze_109: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
    unsqueeze_110: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 2);  unsqueeze_109 = None
    mul_569: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_101, view_151);  view_151 = None
    unsqueeze_111: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_569, 0);  mul_569 = None
    unsqueeze_112: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
    mul_570: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_110);  sub_83 = unsqueeze_110 = None
    sub_85: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_192, mul_570);  view_192 = mul_570 = None
    sub_86: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_108);  sub_85 = unsqueeze_108 = None
    mul_571: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_112);  sub_86 = unsqueeze_112 = None
    mul_572: "f32[768]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_101);  sum_30 = squeeze_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_193: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_572, [768, 1, 1, 1]);  mul_572 = None
    mul_573: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_193, 0.02946278254943948);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_194: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_571, [768, 128, 3, 3]);  mul_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_574: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_144, 1.7015043497085571);  getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_576: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_105, 0.5);  add_105 = None
    mul_577: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_69, convolution_69)
    mul_578: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_577, -0.5);  mul_577 = None
    exp_6: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_578);  mul_578 = None
    mul_579: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_580: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_69, mul_579);  convolution_69 = mul_579 = None
    add_137: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_576, mul_580);  mul_576 = mul_580 = None
    mul_581: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_574, add_137);  mul_574 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 2, 3])
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_581, mul_377, view_149, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_581 = mul_377 = view_149 = None
    getitem_147: "f32[4, 768, 6, 6]" = convolution_backward_11[0]
    getitem_148: "f32[768, 128, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_195: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_148, [1, 768, 1152]);  getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_195, [0, 2])
    sub_87: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_147, unsqueeze_114);  view_147 = unsqueeze_114 = None
    mul_582: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_195, sub_87)
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_582, [0, 2]);  mul_582 = None
    mul_583: "f32[768]" = torch.ops.aten.mul.Tensor(sum_32, 0.0008680555555555555);  sum_32 = None
    unsqueeze_115: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
    unsqueeze_116: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 2);  unsqueeze_115 = None
    mul_584: "f32[768]" = torch.ops.aten.mul.Tensor(sum_33, 0.0008680555555555555)
    mul_585: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_99, squeeze_99)
    mul_586: "f32[768]" = torch.ops.aten.mul.Tensor(mul_584, mul_585);  mul_584 = mul_585 = None
    unsqueeze_117: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
    unsqueeze_118: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
    mul_587: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_99, view_148);  view_148 = None
    unsqueeze_119: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_587, 0);  mul_587 = None
    unsqueeze_120: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
    mul_588: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_118);  sub_87 = unsqueeze_118 = None
    sub_89: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_195, mul_588);  view_195 = mul_588 = None
    sub_90: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_116);  sub_89 = unsqueeze_116 = None
    mul_589: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_120);  sub_90 = unsqueeze_120 = None
    mul_590: "f32[768]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_99);  sum_33 = squeeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_196: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_590, [768, 1, 1, 1]);  mul_590 = None
    mul_591: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_196, 0.02946278254943948);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_197: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_589, [768, 128, 3, 3]);  mul_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_592: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_147, 1.7015043497085571);  getitem_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_594: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_103, 0.5);  add_103 = None
    mul_595: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_68, convolution_68)
    mul_596: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_595, -0.5);  mul_595 = None
    exp_7: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_596);  mul_596 = None
    mul_597: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_598: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_68, mul_597);  convolution_68 = mul_597 = None
    add_139: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_594, mul_598);  mul_594 = mul_598 = None
    mul_599: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_592, add_139);  mul_592 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_599, mul_370, view_146, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_599 = mul_370 = view_146 = None
    getitem_150: "f32[4, 1536, 6, 6]" = convolution_backward_12[0]
    getitem_151: "f32[768, 1536, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_198: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_151, [1, 768, 1536]);  getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_198, [0, 2])
    sub_91: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_144, unsqueeze_122);  view_144 = unsqueeze_122 = None
    mul_600: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_198, sub_91)
    sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_600, [0, 2]);  mul_600 = None
    mul_601: "f32[768]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006510416666666666);  sum_35 = None
    unsqueeze_123: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
    unsqueeze_124: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
    mul_602: "f32[768]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006510416666666666)
    mul_603: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_604: "f32[768]" = torch.ops.aten.mul.Tensor(mul_602, mul_603);  mul_602 = mul_603 = None
    unsqueeze_125: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
    unsqueeze_126: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 2);  unsqueeze_125 = None
    mul_605: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_97, view_145);  view_145 = None
    unsqueeze_127: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_605, 0);  mul_605 = None
    unsqueeze_128: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
    mul_606: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_126);  sub_91 = unsqueeze_126 = None
    sub_93: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_198, mul_606);  view_198 = mul_606 = None
    sub_94: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_124);  sub_93 = unsqueeze_124 = None
    mul_607: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_128);  sub_94 = unsqueeze_128 = None
    mul_608: "f32[768]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_97);  sum_36 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_199: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_608, [768, 1, 1, 1]);  mul_608 = None
    mul_609: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_199, 0.02551551815399144);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_200: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_607, [768, 1536, 1, 1]);  mul_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_610: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_150, 0.9805806756909201);  getitem_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_611: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_610, 1.7015043497085571);  mul_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_613: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_101, 0.5);  add_101 = None
    mul_614: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_100, add_100)
    mul_615: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_614, -0.5);  mul_614 = None
    exp_8: "f32[4, 1536, 6, 6]" = torch.ops.aten.exp.default(mul_615);  mul_615 = None
    mul_616: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_617: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_100, mul_616);  add_100 = mul_616 = None
    add_141: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_613, mul_617);  mul_613 = mul_617 = None
    mul_618: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_611, add_141);  mul_611 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_142: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(add_132, mul_618);  add_132 = mul_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_619: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_142, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_620: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_619, clone_9);  clone_9 = None
    mul_621: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_619, primals_154);  mul_619 = primals_154 = None
    sum_37: "f32[]" = torch.ops.aten.sum.default(mul_620);  mul_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_622: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_621, 2.0);  mul_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_623: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_622, convolution_65);  convolution_65 = None
    mul_624: "f32[4, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_622, sigmoid_9);  mul_622 = sigmoid_9 = None
    sum_38: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_623, [2, 3], True);  mul_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_32: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    sub_95: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_32)
    mul_625: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_32, sub_95);  alias_32 = sub_95 = None
    mul_626: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_38, mul_625);  sum_38 = mul_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_39: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_626, relu_9, primals_222, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_626 = primals_222 = None
    getitem_153: "f32[4, 768, 1, 1]" = convolution_backward_13[0]
    getitem_154: "f32[1536, 768, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_34: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_35: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    le_2: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_35, 0);  alias_35 = None
    where_2: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_2, full_default, getitem_153);  le_2 = getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_2, mean_9, primals_220, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = mean_9 = primals_220 = None
    getitem_156: "f32[4, 1536, 1, 1]" = convolution_backward_14[0]
    getitem_157: "f32[768, 1536, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[4, 1536, 6, 6]" = torch.ops.aten.expand.default(getitem_156, [4, 1536, 6, 6]);  getitem_156 = None
    div_3: "f32[4, 1536, 6, 6]" = torch.ops.aten.div.Scalar(expand_3, 36);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_143: "f32[4, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_624, div_3);  mul_624 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_41: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 2, 3])
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(add_143, mul_358, view_143, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_143 = mul_358 = view_143 = None
    getitem_159: "f32[4, 768, 6, 6]" = convolution_backward_15[0]
    getitem_160: "f32[1536, 768, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_201: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_160, [1, 1536, 768]);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_42: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_201, [0, 2])
    sub_96: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_141, unsqueeze_130);  view_141 = unsqueeze_130 = None
    mul_627: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_201, sub_96)
    sum_43: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 2]);  mul_627 = None
    mul_628: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_42, 0.0013020833333333333);  sum_42 = None
    unsqueeze_131: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
    unsqueeze_132: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
    mul_629: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_43, 0.0013020833333333333)
    mul_630: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_95, squeeze_95)
    mul_631: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_629, mul_630);  mul_629 = mul_630 = None
    unsqueeze_133: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
    unsqueeze_134: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
    mul_632: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_95, view_142);  view_142 = None
    unsqueeze_135: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_136: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
    mul_633: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_134);  sub_96 = unsqueeze_134 = None
    sub_98: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_201, mul_633);  view_201 = mul_633 = None
    sub_99: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_132);  sub_98 = unsqueeze_132 = None
    mul_634: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_136);  sub_99 = unsqueeze_136 = None
    mul_635: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_95);  sum_43 = squeeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_202: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_635, [1536, 1, 1, 1]);  mul_635 = None
    mul_636: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_202, 0.03608439182435161);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_203: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_634, [1536, 768, 1, 1]);  mul_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_637: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_159, 1.7015043497085571);  getitem_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_639: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_98, 0.5);  add_98 = None
    mul_640: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_64, convolution_64)
    mul_641: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_640, -0.5);  mul_640 = None
    exp_9: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_641);  mul_641 = None
    mul_642: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_643: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_64, mul_642);  convolution_64 = mul_642 = None
    add_145: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_639, mul_643);  mul_639 = mul_643 = None
    mul_644: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_637, add_145);  mul_637 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_644, [0, 2, 3])
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_644, mul_351, view_140, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_644 = mul_351 = view_140 = None
    getitem_162: "f32[4, 768, 6, 6]" = convolution_backward_16[0]
    getitem_163: "f32[768, 128, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_204: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_163, [1, 768, 1152]);  getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_204, [0, 2])
    sub_100: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_138, unsqueeze_138);  view_138 = unsqueeze_138 = None
    mul_645: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_204, sub_100)
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_645, [0, 2]);  mul_645 = None
    mul_646: "f32[768]" = torch.ops.aten.mul.Tensor(sum_45, 0.0008680555555555555);  sum_45 = None
    unsqueeze_139: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
    unsqueeze_140: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
    mul_647: "f32[768]" = torch.ops.aten.mul.Tensor(sum_46, 0.0008680555555555555)
    mul_648: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_93, squeeze_93)
    mul_649: "f32[768]" = torch.ops.aten.mul.Tensor(mul_647, mul_648);  mul_647 = mul_648 = None
    unsqueeze_141: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
    unsqueeze_142: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None
    mul_650: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_93, view_139);  view_139 = None
    unsqueeze_143: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_650, 0);  mul_650 = None
    unsqueeze_144: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
    mul_651: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_142);  sub_100 = unsqueeze_142 = None
    sub_102: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_204, mul_651);  view_204 = mul_651 = None
    sub_103: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_140);  sub_102 = unsqueeze_140 = None
    mul_652: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_144);  sub_103 = unsqueeze_144 = None
    mul_653: "f32[768]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_93);  sum_46 = squeeze_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_205: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_653, [768, 1, 1, 1]);  mul_653 = None
    mul_654: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_205, 0.02946278254943948);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_206: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_652, [768, 128, 3, 3]);  mul_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_655: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_162, 1.7015043497085571);  getitem_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_657: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
    mul_658: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_63, convolution_63)
    mul_659: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_658, -0.5);  mul_658 = None
    exp_10: "f32[4, 768, 6, 6]" = torch.ops.aten.exp.default(mul_659);  mul_659 = None
    mul_660: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_661: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_63, mul_660);  convolution_63 = mul_660 = None
    add_147: "f32[4, 768, 6, 6]" = torch.ops.aten.add.Tensor(mul_657, mul_661);  mul_657 = mul_661 = None
    mul_662: "f32[4, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_655, add_147);  mul_655 = add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_662, constant_pad_nd_4, view_137, [768], [2, 2], [0, 0], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_662 = constant_pad_nd_4 = view_137 = None
    getitem_165: "f32[4, 768, 13, 13]" = convolution_backward_17[0]
    getitem_166: "f32[768, 128, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_207: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_166, [1, 768, 1152]);  getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_207, [0, 2])
    sub_104: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_135, unsqueeze_146);  view_135 = unsqueeze_146 = None
    mul_663: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_207, sub_104)
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_663, [0, 2]);  mul_663 = None
    mul_664: "f32[768]" = torch.ops.aten.mul.Tensor(sum_48, 0.0008680555555555555);  sum_48 = None
    unsqueeze_147: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
    unsqueeze_148: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 2);  unsqueeze_147 = None
    mul_665: "f32[768]" = torch.ops.aten.mul.Tensor(sum_49, 0.0008680555555555555)
    mul_666: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_667: "f32[768]" = torch.ops.aten.mul.Tensor(mul_665, mul_666);  mul_665 = mul_666 = None
    unsqueeze_149: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_667, 0);  mul_667 = None
    unsqueeze_150: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
    mul_668: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_91, view_136);  view_136 = None
    unsqueeze_151: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_668, 0);  mul_668 = None
    unsqueeze_152: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
    mul_669: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_150);  sub_104 = unsqueeze_150 = None
    sub_106: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_207, mul_669);  view_207 = mul_669 = None
    sub_107: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_148);  sub_106 = unsqueeze_148 = None
    mul_670: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_152);  sub_107 = unsqueeze_152 = None
    mul_671: "f32[768]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_91);  sum_49 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_208: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_671, [768, 1, 1, 1]);  mul_671 = None
    mul_672: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_208, 0.02946278254943948);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_209: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_670, [768, 128, 3, 3]);  mul_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_5: "f32[4, 768, 12, 12]" = torch.ops.aten.constant_pad_nd.default(getitem_165, [0, -1, 0, -1]);  getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_673: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(constant_pad_nd_5, 1.7015043497085571);  constant_pad_nd_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_675: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_94, 0.5);  add_94 = None
    mul_676: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_62, convolution_62)
    mul_677: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_676, -0.5);  mul_676 = None
    exp_11: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_677);  mul_677 = None
    mul_678: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_679: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_62, mul_678);  convolution_62 = mul_678 = None
    add_149: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_675, mul_679);  mul_675 = mul_679 = None
    mul_680: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_673, add_149);  mul_673 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_680, mul_334, view_134, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_680 = view_134 = None
    getitem_168: "f32[4, 1536, 12, 12]" = convolution_backward_18[0]
    getitem_169: "f32[768, 1536, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_210: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_169, [1, 768, 1536]);  getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_210, [0, 2])
    sub_108: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_132, unsqueeze_154);  view_132 = unsqueeze_154 = None
    mul_681: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_210, sub_108)
    sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_681, [0, 2]);  mul_681 = None
    mul_682: "f32[768]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006510416666666666);  sum_51 = None
    unsqueeze_155: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
    unsqueeze_156: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
    mul_683: "f32[768]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006510416666666666)
    mul_684: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_89, squeeze_89)
    mul_685: "f32[768]" = torch.ops.aten.mul.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
    unsqueeze_157: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_158: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
    mul_686: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_89, view_133);  view_133 = None
    unsqueeze_159: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
    unsqueeze_160: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
    mul_687: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_158);  sub_108 = unsqueeze_158 = None
    sub_110: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_210, mul_687);  view_210 = mul_687 = None
    sub_111: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_156);  sub_110 = unsqueeze_156 = None
    mul_688: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_160);  sub_111 = unsqueeze_160 = None
    mul_689: "f32[768]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_89);  sum_52 = squeeze_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_211: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_689, [768, 1, 1, 1]);  mul_689 = None
    mul_690: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_211, 0.02551551815399144);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_212: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_688, [768, 1536, 1, 1]);  mul_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_53: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 2, 3])
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(add_142, avg_pool2d_2, view_131, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_142 = avg_pool2d_2 = view_131 = None
    getitem_171: "f32[4, 1536, 6, 6]" = convolution_backward_19[0]
    getitem_172: "f32[1536, 1536, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_213: "f32[1, 1536, 1536]" = torch.ops.aten.view.default(getitem_172, [1, 1536, 1536]);  getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_54: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_213, [0, 2])
    sub_112: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_129, unsqueeze_162);  view_129 = unsqueeze_162 = None
    mul_691: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(view_213, sub_112)
    sum_55: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 2]);  mul_691 = None
    mul_692: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006510416666666666);  sum_54 = None
    unsqueeze_163: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
    unsqueeze_164: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
    mul_693: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006510416666666666)
    mul_694: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_87, squeeze_87)
    mul_695: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_693, mul_694);  mul_693 = mul_694 = None
    unsqueeze_165: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    unsqueeze_166: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
    mul_696: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_87, view_130);  view_130 = None
    unsqueeze_167: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
    unsqueeze_168: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
    mul_697: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_166);  sub_112 = unsqueeze_166 = None
    sub_114: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_213, mul_697);  view_213 = mul_697 = None
    sub_115: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_164);  sub_114 = unsqueeze_164 = None
    mul_698: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_168);  sub_115 = unsqueeze_168 = None
    mul_699: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_87);  sum_55 = squeeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_214: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_699, [1536, 1, 1, 1]);  mul_699 = None
    mul_700: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_214, 0.02551551815399144);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_215: "f32[1536, 1536, 1, 1]" = torch.ops.aten.view.default(mul_698, [1536, 1536, 1, 1]);  mul_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_backward: "f32[4, 1536, 12, 12]" = torch.ops.aten.avg_pool2d_backward.default(getitem_171, mul_334, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_171 = mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    add_150: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(getitem_168, avg_pool2d_backward);  getitem_168 = avg_pool2d_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_701: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_150, 0.8980265101338745);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_702: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_701, 1.7015043497085571);  mul_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_704: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_91, 0.5);  add_91 = None
    mul_705: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_90, add_90)
    mul_706: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_705, -0.5);  mul_705 = None
    exp_12: "f32[4, 1536, 12, 12]" = torch.ops.aten.exp.default(mul_706);  mul_706 = None
    mul_707: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_708: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_90, mul_707);  add_90 = mul_707 = None
    add_152: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_704, mul_708);  mul_704 = mul_708 = None
    mul_709: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_702, add_152);  mul_702 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_710: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_709, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_711: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_710, clone_8);  clone_8 = None
    mul_712: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_710, primals_138);  mul_710 = primals_138 = None
    sum_56: "f32[]" = torch.ops.aten.sum.default(mul_711);  mul_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_713: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_712, 2.0);  mul_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_714: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_713, convolution_58);  convolution_58 = None
    mul_715: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_713, sigmoid_8);  mul_713 = sigmoid_8 = None
    sum_57: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_714, [2, 3], True);  mul_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_36: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    sub_116: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_36)
    mul_716: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_36, sub_116);  alias_36 = sub_116 = None
    mul_717: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_57, mul_716);  sum_57 = mul_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_58: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3])
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_717, relu_8, primals_218, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_717 = primals_218 = None
    getitem_174: "f32[4, 768, 1, 1]" = convolution_backward_20[0]
    getitem_175: "f32[1536, 768, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_38: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_39: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    le_3: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_39, 0);  alias_39 = None
    where_3: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_3, full_default, getitem_174);  le_3 = getitem_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(where_3, mean_8, primals_216, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = mean_8 = primals_216 = None
    getitem_177: "f32[4, 1536, 1, 1]" = convolution_backward_21[0]
    getitem_178: "f32[768, 1536, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[4, 1536, 12, 12]" = torch.ops.aten.expand.default(getitem_177, [4, 1536, 12, 12]);  getitem_177 = None
    div_4: "f32[4, 1536, 12, 12]" = torch.ops.aten.div.Scalar(expand_4, 144);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_153: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_715, div_4);  mul_715 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_60: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(add_153, mul_322, view_128, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_153 = mul_322 = view_128 = None
    getitem_180: "f32[4, 768, 12, 12]" = convolution_backward_22[0]
    getitem_181: "f32[1536, 768, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_216: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_181, [1, 1536, 768]);  getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_61: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_216, [0, 2])
    sub_117: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_126, unsqueeze_170);  view_126 = unsqueeze_170 = None
    mul_718: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_216, sub_117)
    sum_62: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_718, [0, 2]);  mul_718 = None
    mul_719: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_61, 0.0013020833333333333);  sum_61 = None
    unsqueeze_171: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
    unsqueeze_172: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
    mul_720: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_62, 0.0013020833333333333)
    mul_721: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_722: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_720, mul_721);  mul_720 = mul_721 = None
    unsqueeze_173: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_174: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
    mul_723: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_85, view_127);  view_127 = None
    unsqueeze_175: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_723, 0);  mul_723 = None
    unsqueeze_176: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
    mul_724: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_174);  sub_117 = unsqueeze_174 = None
    sub_119: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_216, mul_724);  view_216 = mul_724 = None
    sub_120: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_172);  sub_119 = unsqueeze_172 = None
    mul_725: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_176);  sub_120 = unsqueeze_176 = None
    mul_726: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_62, squeeze_85);  sum_62 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_217: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_726, [1536, 1, 1, 1]);  mul_726 = None
    mul_727: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_217, 0.03608439182435161);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_218: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_725, [1536, 768, 1, 1]);  mul_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_728: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_180, 1.7015043497085571);  getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_730: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
    mul_731: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_57, convolution_57)
    mul_732: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_731, -0.5);  mul_731 = None
    exp_13: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_732);  mul_732 = None
    mul_733: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_734: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_57, mul_733);  convolution_57 = mul_733 = None
    add_155: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_730, mul_734);  mul_730 = mul_734 = None
    mul_735: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_728, add_155);  mul_728 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_735, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_735, mul_315, view_125, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_735 = mul_315 = view_125 = None
    getitem_183: "f32[4, 768, 12, 12]" = convolution_backward_23[0]
    getitem_184: "f32[768, 128, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_219: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_184, [1, 768, 1152]);  getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_219, [0, 2])
    sub_121: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_123, unsqueeze_178);  view_123 = unsqueeze_178 = None
    mul_736: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_219, sub_121)
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_736, [0, 2]);  mul_736 = None
    mul_737: "f32[768]" = torch.ops.aten.mul.Tensor(sum_64, 0.0008680555555555555);  sum_64 = None
    unsqueeze_179: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
    unsqueeze_180: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    mul_738: "f32[768]" = torch.ops.aten.mul.Tensor(sum_65, 0.0008680555555555555)
    mul_739: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_83, squeeze_83)
    mul_740: "f32[768]" = torch.ops.aten.mul.Tensor(mul_738, mul_739);  mul_738 = mul_739 = None
    unsqueeze_181: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    unsqueeze_182: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
    mul_741: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_83, view_124);  view_124 = None
    unsqueeze_183: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
    unsqueeze_184: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
    mul_742: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_182);  sub_121 = unsqueeze_182 = None
    sub_123: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_219, mul_742);  view_219 = mul_742 = None
    sub_124: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_180);  sub_123 = unsqueeze_180 = None
    mul_743: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_184);  sub_124 = unsqueeze_184 = None
    mul_744: "f32[768]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_83);  sum_65 = squeeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_220: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_744, [768, 1, 1, 1]);  mul_744 = None
    mul_745: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_220, 0.02946278254943948);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_221: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_743, [768, 128, 3, 3]);  mul_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_746: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_183, 1.7015043497085571);  getitem_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_748: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_86, 0.5);  add_86 = None
    mul_749: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_56, convolution_56)
    mul_750: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_749, -0.5);  mul_749 = None
    exp_14: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_750);  mul_750 = None
    mul_751: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_752: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_56, mul_751);  convolution_56 = mul_751 = None
    add_157: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_748, mul_752);  mul_748 = mul_752 = None
    mul_753: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_746, add_157);  mul_746 = add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_753, [0, 2, 3])
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_753, mul_308, view_122, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_753 = mul_308 = view_122 = None
    getitem_186: "f32[4, 768, 12, 12]" = convolution_backward_24[0]
    getitem_187: "f32[768, 128, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_222: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_187, [1, 768, 1152]);  getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_222, [0, 2])
    sub_125: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_120, unsqueeze_186);  view_120 = unsqueeze_186 = None
    mul_754: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_222, sub_125)
    sum_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_754, [0, 2]);  mul_754 = None
    mul_755: "f32[768]" = torch.ops.aten.mul.Tensor(sum_67, 0.0008680555555555555);  sum_67 = None
    unsqueeze_187: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_188: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
    mul_756: "f32[768]" = torch.ops.aten.mul.Tensor(sum_68, 0.0008680555555555555)
    mul_757: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_81, squeeze_81)
    mul_758: "f32[768]" = torch.ops.aten.mul.Tensor(mul_756, mul_757);  mul_756 = mul_757 = None
    unsqueeze_189: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_190: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
    mul_759: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_81, view_121);  view_121 = None
    unsqueeze_191: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_759, 0);  mul_759 = None
    unsqueeze_192: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    mul_760: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_190);  sub_125 = unsqueeze_190 = None
    sub_127: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_222, mul_760);  view_222 = mul_760 = None
    sub_128: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_188);  sub_127 = unsqueeze_188 = None
    mul_761: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_192);  sub_128 = unsqueeze_192 = None
    mul_762: "f32[768]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_81);  sum_68 = squeeze_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_223: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_762, [768, 1, 1, 1]);  mul_762 = None
    mul_763: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_223, 0.02946278254943948);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_224: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_761, [768, 128, 3, 3]);  mul_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_764: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_186, 1.7015043497085571);  getitem_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_766: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_84, 0.5);  add_84 = None
    mul_767: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_55, convolution_55)
    mul_768: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_767, -0.5);  mul_767 = None
    exp_15: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_768);  mul_768 = None
    mul_769: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_770: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_55, mul_769);  convolution_55 = mul_769 = None
    add_159: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_766, mul_770);  mul_766 = mul_770 = None
    mul_771: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_764, add_159);  mul_764 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3])
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_771, mul_301, view_119, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_771 = mul_301 = view_119 = None
    getitem_189: "f32[4, 1536, 12, 12]" = convolution_backward_25[0]
    getitem_190: "f32[768, 1536, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_225: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_190, [1, 768, 1536]);  getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_225, [0, 2])
    sub_129: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_117, unsqueeze_194);  view_117 = unsqueeze_194 = None
    mul_772: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_225, sub_129)
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_772, [0, 2]);  mul_772 = None
    mul_773: "f32[768]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006510416666666666);  sum_70 = None
    unsqueeze_195: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_196: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
    mul_774: "f32[768]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006510416666666666)
    mul_775: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_776: "f32[768]" = torch.ops.aten.mul.Tensor(mul_774, mul_775);  mul_774 = mul_775 = None
    unsqueeze_197: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_198: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    mul_777: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, view_118);  view_118 = None
    unsqueeze_199: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
    unsqueeze_200: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
    mul_778: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_198);  sub_129 = unsqueeze_198 = None
    sub_131: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_225, mul_778);  view_225 = mul_778 = None
    sub_132: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_196);  sub_131 = unsqueeze_196 = None
    mul_779: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_200);  sub_132 = unsqueeze_200 = None
    mul_780: "f32[768]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_79);  sum_71 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_226: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_780, [768, 1, 1, 1]);  mul_780 = None
    mul_781: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_226, 0.02551551815399144);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_227: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_779, [768, 1536, 1, 1]);  mul_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_782: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_189, 0.9128709291752768);  getitem_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_783: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_782, 1.7015043497085571);  mul_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_785: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
    mul_786: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_81, add_81)
    mul_787: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_786, -0.5);  mul_786 = None
    exp_16: "f32[4, 1536, 12, 12]" = torch.ops.aten.exp.default(mul_787);  mul_787 = None
    mul_788: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_789: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_81, mul_788);  add_81 = mul_788 = None
    add_161: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_785, mul_789);  mul_785 = mul_789 = None
    mul_790: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_783, add_161);  mul_783 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_162: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_709, mul_790);  mul_709 = mul_790 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_791: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_162, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_792: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_791, clone_7);  clone_7 = None
    mul_793: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_791, primals_125);  mul_791 = primals_125 = None
    sum_72: "f32[]" = torch.ops.aten.sum.default(mul_792);  mul_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_794: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_793, 2.0);  mul_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_795: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_794, convolution_52);  convolution_52 = None
    mul_796: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_794, sigmoid_7);  mul_794 = sigmoid_7 = None
    sum_73: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_795, [2, 3], True);  mul_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_40: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    sub_133: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_40)
    mul_797: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_40, sub_133);  alias_40 = sub_133 = None
    mul_798: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_73, mul_797);  sum_73 = mul_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_74: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_798, [0, 2, 3])
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_798, relu_7, primals_214, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_798 = primals_214 = None
    getitem_192: "f32[4, 768, 1, 1]" = convolution_backward_26[0]
    getitem_193: "f32[1536, 768, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_42: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_43: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_4: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_43, 0);  alias_43 = None
    where_4: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_4, full_default, getitem_192);  le_4 = getitem_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_4, mean_7, primals_212, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = mean_7 = primals_212 = None
    getitem_195: "f32[4, 1536, 1, 1]" = convolution_backward_27[0]
    getitem_196: "f32[768, 1536, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[4, 1536, 12, 12]" = torch.ops.aten.expand.default(getitem_195, [4, 1536, 12, 12]);  getitem_195 = None
    div_5: "f32[4, 1536, 12, 12]" = torch.ops.aten.div.Scalar(expand_5, 144);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_163: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_796, div_5);  mul_796 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_76: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_163, [0, 2, 3])
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(add_163, mul_289, view_116, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_163 = mul_289 = view_116 = None
    getitem_198: "f32[4, 768, 12, 12]" = convolution_backward_28[0]
    getitem_199: "f32[1536, 768, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_228: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_199, [1, 1536, 768]);  getitem_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_77: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_228, [0, 2])
    sub_134: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_114, unsqueeze_202);  view_114 = unsqueeze_202 = None
    mul_799: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_228, sub_134)
    sum_78: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_799, [0, 2]);  mul_799 = None
    mul_800: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_77, 0.0013020833333333333);  sum_77 = None
    unsqueeze_203: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_204: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    mul_801: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_78, 0.0013020833333333333)
    mul_802: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_77, squeeze_77)
    mul_803: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_801, mul_802);  mul_801 = mul_802 = None
    unsqueeze_205: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    unsqueeze_206: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    mul_804: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_77, view_115);  view_115 = None
    unsqueeze_207: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
    unsqueeze_208: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
    mul_805: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_206);  sub_134 = unsqueeze_206 = None
    sub_136: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_228, mul_805);  view_228 = mul_805 = None
    sub_137: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_136, unsqueeze_204);  sub_136 = unsqueeze_204 = None
    mul_806: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_208);  sub_137 = unsqueeze_208 = None
    mul_807: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_77);  sum_78 = squeeze_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_229: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_807, [1536, 1, 1, 1]);  mul_807 = None
    mul_808: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_229, 0.03608439182435161);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_230: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_806, [1536, 768, 1, 1]);  mul_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_809: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_198, 1.7015043497085571);  getitem_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_811: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_79, 0.5);  add_79 = None
    mul_812: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_51, convolution_51)
    mul_813: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_812, -0.5);  mul_812 = None
    exp_17: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_813);  mul_813 = None
    mul_814: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_815: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_51, mul_814);  convolution_51 = mul_814 = None
    add_165: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_811, mul_815);  mul_811 = mul_815 = None
    mul_816: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_809, add_165);  mul_809 = add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_816, [0, 2, 3])
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_816, mul_282, view_113, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_816 = mul_282 = view_113 = None
    getitem_201: "f32[4, 768, 12, 12]" = convolution_backward_29[0]
    getitem_202: "f32[768, 128, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_231: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_202, [1, 768, 1152]);  getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_231, [0, 2])
    sub_138: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_111, unsqueeze_210);  view_111 = unsqueeze_210 = None
    mul_817: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_231, sub_138)
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_817, [0, 2]);  mul_817 = None
    mul_818: "f32[768]" = torch.ops.aten.mul.Tensor(sum_80, 0.0008680555555555555);  sum_80 = None
    unsqueeze_211: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    unsqueeze_212: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    mul_819: "f32[768]" = torch.ops.aten.mul.Tensor(sum_81, 0.0008680555555555555)
    mul_820: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_75, squeeze_75)
    mul_821: "f32[768]" = torch.ops.aten.mul.Tensor(mul_819, mul_820);  mul_819 = mul_820 = None
    unsqueeze_213: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    unsqueeze_214: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
    mul_822: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_75, view_112);  view_112 = None
    unsqueeze_215: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
    unsqueeze_216: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
    mul_823: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_214);  sub_138 = unsqueeze_214 = None
    sub_140: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_231, mul_823);  view_231 = mul_823 = None
    sub_141: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_140, unsqueeze_212);  sub_140 = unsqueeze_212 = None
    mul_824: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_216);  sub_141 = unsqueeze_216 = None
    mul_825: "f32[768]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_75);  sum_81 = squeeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_232: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_825, [768, 1, 1, 1]);  mul_825 = None
    mul_826: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_232, 0.02946278254943948);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_233: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_824, [768, 128, 3, 3]);  mul_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_827: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_201, 1.7015043497085571);  getitem_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_829: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_77, 0.5);  add_77 = None
    mul_830: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_50, convolution_50)
    mul_831: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_830, -0.5);  mul_830 = None
    exp_18: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_831);  mul_831 = None
    mul_832: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_833: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_50, mul_832);  convolution_50 = mul_832 = None
    add_167: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_829, mul_833);  mul_829 = mul_833 = None
    mul_834: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_827, add_167);  mul_827 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_834, [0, 2, 3])
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_834, mul_275, view_110, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_834 = mul_275 = view_110 = None
    getitem_204: "f32[4, 768, 12, 12]" = convolution_backward_30[0]
    getitem_205: "f32[768, 128, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_234: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_205, [1, 768, 1152]);  getitem_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_234, [0, 2])
    sub_142: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_108, unsqueeze_218);  view_108 = unsqueeze_218 = None
    mul_835: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_234, sub_142)
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_835, [0, 2]);  mul_835 = None
    mul_836: "f32[768]" = torch.ops.aten.mul.Tensor(sum_83, 0.0008680555555555555);  sum_83 = None
    unsqueeze_219: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_836, 0);  mul_836 = None
    unsqueeze_220: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
    mul_837: "f32[768]" = torch.ops.aten.mul.Tensor(sum_84, 0.0008680555555555555)
    mul_838: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_839: "f32[768]" = torch.ops.aten.mul.Tensor(mul_837, mul_838);  mul_837 = mul_838 = None
    unsqueeze_221: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_839, 0);  mul_839 = None
    unsqueeze_222: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    mul_840: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, view_109);  view_109 = None
    unsqueeze_223: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
    unsqueeze_224: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    mul_841: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_222);  sub_142 = unsqueeze_222 = None
    sub_144: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_234, mul_841);  view_234 = mul_841 = None
    sub_145: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_144, unsqueeze_220);  sub_144 = unsqueeze_220 = None
    mul_842: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_224);  sub_145 = unsqueeze_224 = None
    mul_843: "f32[768]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_73);  sum_84 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_235: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_843, [768, 1, 1, 1]);  mul_843 = None
    mul_844: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_235, 0.02946278254943948);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_236: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_842, [768, 128, 3, 3]);  mul_842 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_845: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_204, 1.7015043497085571);  getitem_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_847: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_75, 0.5);  add_75 = None
    mul_848: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_49, convolution_49)
    mul_849: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_848, -0.5);  mul_848 = None
    exp_19: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_849);  mul_849 = None
    mul_850: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_851: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_49, mul_850);  convolution_49 = mul_850 = None
    add_169: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_847, mul_851);  mul_847 = mul_851 = None
    mul_852: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_845, add_169);  mul_845 = add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_852, [0, 2, 3])
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_852, mul_268, view_107, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_852 = mul_268 = view_107 = None
    getitem_207: "f32[4, 1536, 12, 12]" = convolution_backward_31[0]
    getitem_208: "f32[768, 1536, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_237: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_208, [1, 768, 1536]);  getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_237, [0, 2])
    sub_146: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_105, unsqueeze_226);  view_105 = unsqueeze_226 = None
    mul_853: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_237, sub_146)
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_853, [0, 2]);  mul_853 = None
    mul_854: "f32[768]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006510416666666666);  sum_86 = None
    unsqueeze_227: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_854, 0);  mul_854 = None
    unsqueeze_228: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    mul_855: "f32[768]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006510416666666666)
    mul_856: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_71, squeeze_71)
    mul_857: "f32[768]" = torch.ops.aten.mul.Tensor(mul_855, mul_856);  mul_855 = mul_856 = None
    unsqueeze_229: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_857, 0);  mul_857 = None
    unsqueeze_230: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    mul_858: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_71, view_106);  view_106 = None
    unsqueeze_231: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_858, 0);  mul_858 = None
    unsqueeze_232: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
    mul_859: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_230);  sub_146 = unsqueeze_230 = None
    sub_148: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_237, mul_859);  view_237 = mul_859 = None
    sub_149: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_148, unsqueeze_228);  sub_148 = unsqueeze_228 = None
    mul_860: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_232);  sub_149 = unsqueeze_232 = None
    mul_861: "f32[768]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_71);  sum_87 = squeeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_238: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_861, [768, 1, 1, 1]);  mul_861 = None
    mul_862: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_238, 0.02551551815399144);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_239: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_860, [768, 1536, 1, 1]);  mul_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_863: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_207, 0.9284766908852592);  getitem_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_864: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_863, 1.7015043497085571);  mul_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_866: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_73, 0.5);  add_73 = None
    mul_867: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_72, add_72)
    mul_868: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_867, -0.5);  mul_867 = None
    exp_20: "f32[4, 1536, 12, 12]" = torch.ops.aten.exp.default(mul_868);  mul_868 = None
    mul_869: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_870: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_72, mul_869);  add_72 = mul_869 = None
    add_171: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_866, mul_870);  mul_866 = mul_870 = None
    mul_871: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_864, add_171);  mul_864 = add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_172: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(add_162, mul_871);  add_162 = mul_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_872: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_172, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_873: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_872, clone_6);  clone_6 = None
    mul_874: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_872, primals_112);  mul_872 = primals_112 = None
    sum_88: "f32[]" = torch.ops.aten.sum.default(mul_873);  mul_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_875: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_874, 2.0);  mul_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_876: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_875, convolution_46);  convolution_46 = None
    mul_877: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_875, sigmoid_6);  mul_875 = sigmoid_6 = None
    sum_89: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_876, [2, 3], True);  mul_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_44: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    sub_150: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_44)
    mul_878: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_44, sub_150);  alias_44 = sub_150 = None
    mul_879: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_89, mul_878);  sum_89 = mul_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_90: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_879, [0, 2, 3])
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_879, relu_6, primals_210, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_879 = primals_210 = None
    getitem_210: "f32[4, 768, 1, 1]" = convolution_backward_32[0]
    getitem_211: "f32[1536, 768, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_46: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_47: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    le_5: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_47, 0);  alias_47 = None
    where_5: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_5, full_default, getitem_210);  le_5 = getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_5, mean_6, primals_208, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_6 = primals_208 = None
    getitem_213: "f32[4, 1536, 1, 1]" = convolution_backward_33[0]
    getitem_214: "f32[768, 1536, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[4, 1536, 12, 12]" = torch.ops.aten.expand.default(getitem_213, [4, 1536, 12, 12]);  getitem_213 = None
    div_6: "f32[4, 1536, 12, 12]" = torch.ops.aten.div.Scalar(expand_6, 144);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_173: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_877, div_6);  mul_877 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_92: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 2, 3])
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(add_173, mul_256, view_104, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_173 = mul_256 = view_104 = None
    getitem_216: "f32[4, 768, 12, 12]" = convolution_backward_34[0]
    getitem_217: "f32[1536, 768, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_240: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_217, [1, 1536, 768]);  getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_93: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_240, [0, 2])
    sub_151: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_102, unsqueeze_234);  view_102 = unsqueeze_234 = None
    mul_880: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_240, sub_151)
    sum_94: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_880, [0, 2]);  mul_880 = None
    mul_881: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_93, 0.0013020833333333333);  sum_93 = None
    unsqueeze_235: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    unsqueeze_236: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    mul_882: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_94, 0.0013020833333333333)
    mul_883: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_69, squeeze_69)
    mul_884: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_882, mul_883);  mul_882 = mul_883 = None
    unsqueeze_237: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_238: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
    mul_885: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_69, view_103);  view_103 = None
    unsqueeze_239: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_885, 0);  mul_885 = None
    unsqueeze_240: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    mul_886: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_238);  sub_151 = unsqueeze_238 = None
    sub_153: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_240, mul_886);  view_240 = mul_886 = None
    sub_154: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_153, unsqueeze_236);  sub_153 = unsqueeze_236 = None
    mul_887: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_240);  sub_154 = unsqueeze_240 = None
    mul_888: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_69);  sum_94 = squeeze_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_241: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_888, [1536, 1, 1, 1]);  mul_888 = None
    mul_889: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_241, 0.03608439182435161);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_242: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_887, [1536, 768, 1, 1]);  mul_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_890: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_216, 1.7015043497085571);  getitem_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_892: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_70, 0.5);  add_70 = None
    mul_893: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_45, convolution_45)
    mul_894: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_893, -0.5);  mul_893 = None
    exp_21: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_894);  mul_894 = None
    mul_895: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_896: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_45, mul_895);  convolution_45 = mul_895 = None
    add_175: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_892, mul_896);  mul_892 = mul_896 = None
    mul_897: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_890, add_175);  mul_890 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 2, 3])
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_897, mul_249, view_101, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_897 = mul_249 = view_101 = None
    getitem_219: "f32[4, 768, 12, 12]" = convolution_backward_35[0]
    getitem_220: "f32[768, 128, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_243: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_220, [1, 768, 1152]);  getitem_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_243, [0, 2])
    sub_155: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_99, unsqueeze_242);  view_99 = unsqueeze_242 = None
    mul_898: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_243, sub_155)
    sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_898, [0, 2]);  mul_898 = None
    mul_899: "f32[768]" = torch.ops.aten.mul.Tensor(sum_96, 0.0008680555555555555);  sum_96 = None
    unsqueeze_243: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
    unsqueeze_244: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
    mul_900: "f32[768]" = torch.ops.aten.mul.Tensor(sum_97, 0.0008680555555555555)
    mul_901: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_902: "f32[768]" = torch.ops.aten.mul.Tensor(mul_900, mul_901);  mul_900 = mul_901 = None
    unsqueeze_245: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
    unsqueeze_246: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    mul_903: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, view_100);  view_100 = None
    unsqueeze_247: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_903, 0);  mul_903 = None
    unsqueeze_248: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    mul_904: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_246);  sub_155 = unsqueeze_246 = None
    sub_157: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_243, mul_904);  view_243 = mul_904 = None
    sub_158: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_157, unsqueeze_244);  sub_157 = unsqueeze_244 = None
    mul_905: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_248);  sub_158 = unsqueeze_248 = None
    mul_906: "f32[768]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_67);  sum_97 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_244: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_906, [768, 1, 1, 1]);  mul_906 = None
    mul_907: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_244, 0.02946278254943948);  view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_245: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_905, [768, 128, 3, 3]);  mul_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_908: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_219, 1.7015043497085571);  getitem_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_910: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_68, 0.5);  add_68 = None
    mul_911: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_44, convolution_44)
    mul_912: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_911, -0.5);  mul_911 = None
    exp_22: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_912);  mul_912 = None
    mul_913: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_914: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_44, mul_913);  convolution_44 = mul_913 = None
    add_177: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_910, mul_914);  mul_910 = mul_914 = None
    mul_915: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_908, add_177);  mul_908 = add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_915, [0, 2, 3])
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_915, mul_242, view_98, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_915 = mul_242 = view_98 = None
    getitem_222: "f32[4, 768, 12, 12]" = convolution_backward_36[0]
    getitem_223: "f32[768, 128, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_246: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_223, [1, 768, 1152]);  getitem_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_246, [0, 2])
    sub_159: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_96, unsqueeze_250);  view_96 = unsqueeze_250 = None
    mul_916: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_246, sub_159)
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_916, [0, 2]);  mul_916 = None
    mul_917: "f32[768]" = torch.ops.aten.mul.Tensor(sum_99, 0.0008680555555555555);  sum_99 = None
    unsqueeze_251: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_917, 0);  mul_917 = None
    unsqueeze_252: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    mul_918: "f32[768]" = torch.ops.aten.mul.Tensor(sum_100, 0.0008680555555555555)
    mul_919: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_65, squeeze_65)
    mul_920: "f32[768]" = torch.ops.aten.mul.Tensor(mul_918, mul_919);  mul_918 = mul_919 = None
    unsqueeze_253: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
    unsqueeze_254: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    mul_921: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_65, view_97);  view_97 = None
    unsqueeze_255: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_921, 0);  mul_921 = None
    unsqueeze_256: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
    mul_922: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_254);  sub_159 = unsqueeze_254 = None
    sub_161: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_246, mul_922);  view_246 = mul_922 = None
    sub_162: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_252);  sub_161 = unsqueeze_252 = None
    mul_923: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_256);  sub_162 = unsqueeze_256 = None
    mul_924: "f32[768]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_65);  sum_100 = squeeze_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_247: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_924, [768, 1, 1, 1]);  mul_924 = None
    mul_925: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_247, 0.02946278254943948);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_248: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_923, [768, 128, 3, 3]);  mul_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_926: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_222, 1.7015043497085571);  getitem_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_928: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_66, 0.5);  add_66 = None
    mul_929: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_43, convolution_43)
    mul_930: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_929, -0.5);  mul_929 = None
    exp_23: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_930);  mul_930 = None
    mul_931: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_932: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_43, mul_931);  convolution_43 = mul_931 = None
    add_179: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_928, mul_932);  mul_928 = mul_932 = None
    mul_933: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_926, add_179);  mul_926 = add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_933, [0, 2, 3])
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_933, mul_235, view_95, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_933 = mul_235 = view_95 = None
    getitem_225: "f32[4, 1536, 12, 12]" = convolution_backward_37[0]
    getitem_226: "f32[768, 1536, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_249: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_226, [1, 768, 1536]);  getitem_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_102: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_249, [0, 2])
    sub_163: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_93, unsqueeze_258);  view_93 = unsqueeze_258 = None
    mul_934: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_249, sub_163)
    sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_934, [0, 2]);  mul_934 = None
    mul_935: "f32[768]" = torch.ops.aten.mul.Tensor(sum_102, 0.0006510416666666666);  sum_102 = None
    unsqueeze_259: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
    unsqueeze_260: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    mul_936: "f32[768]" = torch.ops.aten.mul.Tensor(sum_103, 0.0006510416666666666)
    mul_937: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_63, squeeze_63)
    mul_938: "f32[768]" = torch.ops.aten.mul.Tensor(mul_936, mul_937);  mul_936 = mul_937 = None
    unsqueeze_261: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
    unsqueeze_262: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
    mul_939: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_63, view_94);  view_94 = None
    unsqueeze_263: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_939, 0);  mul_939 = None
    unsqueeze_264: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    mul_940: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_262);  sub_163 = unsqueeze_262 = None
    sub_165: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_249, mul_940);  view_249 = mul_940 = None
    sub_166: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_260);  sub_165 = unsqueeze_260 = None
    mul_941: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_264);  sub_166 = unsqueeze_264 = None
    mul_942: "f32[768]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_63);  sum_103 = squeeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_250: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_942, [768, 1, 1, 1]);  mul_942 = None
    mul_943: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_250, 0.02551551815399144);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_251: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_941, [768, 1536, 1, 1]);  mul_941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_944: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_225, 0.9449111825230679);  getitem_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_945: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_944, 1.7015043497085571);  mul_944 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_947: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
    mul_948: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_63, add_63)
    mul_949: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_948, -0.5);  mul_948 = None
    exp_24: "f32[4, 1536, 12, 12]" = torch.ops.aten.exp.default(mul_949);  mul_949 = None
    mul_950: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_951: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_63, mul_950);  add_63 = mul_950 = None
    add_181: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_947, mul_951);  mul_947 = mul_951 = None
    mul_952: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_945, add_181);  mul_945 = add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_182: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(add_172, mul_952);  add_172 = mul_952 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_953: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_182, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_954: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_953, clone_5);  clone_5 = None
    mul_955: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_953, primals_99);  mul_953 = primals_99 = None
    sum_104: "f32[]" = torch.ops.aten.sum.default(mul_954);  mul_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_956: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_955, 2.0);  mul_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_957: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_956, convolution_40);  convolution_40 = None
    mul_958: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_956, sigmoid_5);  mul_956 = sigmoid_5 = None
    sum_105: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_957, [2, 3], True);  mul_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_48: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    sub_167: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_48)
    mul_959: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_48, sub_167);  alias_48 = sub_167 = None
    mul_960: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_105, mul_959);  sum_105 = mul_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_106: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_960, [0, 2, 3])
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_960, relu_5, primals_206, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_960 = primals_206 = None
    getitem_228: "f32[4, 768, 1, 1]" = convolution_backward_38[0]
    getitem_229: "f32[1536, 768, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_50: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_51: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    le_6: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_51, 0);  alias_51 = None
    where_6: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_6, full_default, getitem_228);  le_6 = getitem_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(where_6, mean_5, primals_204, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = mean_5 = primals_204 = None
    getitem_231: "f32[4, 1536, 1, 1]" = convolution_backward_39[0]
    getitem_232: "f32[768, 1536, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[4, 1536, 12, 12]" = torch.ops.aten.expand.default(getitem_231, [4, 1536, 12, 12]);  getitem_231 = None
    div_7: "f32[4, 1536, 12, 12]" = torch.ops.aten.div.Scalar(expand_7, 144);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_183: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_958, div_7);  mul_958 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_108: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_183, [0, 2, 3])
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(add_183, mul_223, view_92, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_183 = mul_223 = view_92 = None
    getitem_234: "f32[4, 768, 12, 12]" = convolution_backward_40[0]
    getitem_235: "f32[1536, 768, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_252: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_235, [1, 1536, 768]);  getitem_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_109: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_252, [0, 2])
    sub_168: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_90, unsqueeze_266);  view_90 = unsqueeze_266 = None
    mul_961: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_252, sub_168)
    sum_110: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_961, [0, 2]);  mul_961 = None
    mul_962: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_109, 0.0013020833333333333);  sum_109 = None
    unsqueeze_267: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_962, 0);  mul_962 = None
    unsqueeze_268: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
    mul_963: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_110, 0.0013020833333333333)
    mul_964: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_965: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_963, mul_964);  mul_963 = mul_964 = None
    unsqueeze_269: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_965, 0);  mul_965 = None
    unsqueeze_270: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    mul_966: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_61, view_91);  view_91 = None
    unsqueeze_271: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_966, 0);  mul_966 = None
    unsqueeze_272: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    mul_967: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_270);  sub_168 = unsqueeze_270 = None
    sub_170: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_252, mul_967);  view_252 = mul_967 = None
    sub_171: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_268);  sub_170 = unsqueeze_268 = None
    mul_968: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_272);  sub_171 = unsqueeze_272 = None
    mul_969: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_61);  sum_110 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_253: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_969, [1536, 1, 1, 1]);  mul_969 = None
    mul_970: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_253, 0.03608439182435161);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_254: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_968, [1536, 768, 1, 1]);  mul_968 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_971: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_234, 1.7015043497085571);  getitem_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_973: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_61, 0.5);  add_61 = None
    mul_974: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_39, convolution_39)
    mul_975: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_974, -0.5);  mul_974 = None
    exp_25: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_975);  mul_975 = None
    mul_976: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_977: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_39, mul_976);  convolution_39 = mul_976 = None
    add_185: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_973, mul_977);  mul_973 = mul_977 = None
    mul_978: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_971, add_185);  mul_971 = add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_978, [0, 2, 3])
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_978, mul_216, view_89, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_978 = mul_216 = view_89 = None
    getitem_237: "f32[4, 768, 12, 12]" = convolution_backward_41[0]
    getitem_238: "f32[768, 128, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_255: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_238, [1, 768, 1152]);  getitem_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_255, [0, 2])
    sub_172: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_87, unsqueeze_274);  view_87 = unsqueeze_274 = None
    mul_979: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_255, sub_172)
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_979, [0, 2]);  mul_979 = None
    mul_980: "f32[768]" = torch.ops.aten.mul.Tensor(sum_112, 0.0008680555555555555);  sum_112 = None
    unsqueeze_275: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_980, 0);  mul_980 = None
    unsqueeze_276: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    mul_981: "f32[768]" = torch.ops.aten.mul.Tensor(sum_113, 0.0008680555555555555)
    mul_982: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_59, squeeze_59)
    mul_983: "f32[768]" = torch.ops.aten.mul.Tensor(mul_981, mul_982);  mul_981 = mul_982 = None
    unsqueeze_277: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_278: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    mul_984: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_59, view_88);  view_88 = None
    unsqueeze_279: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_984, 0);  mul_984 = None
    unsqueeze_280: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
    mul_985: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_278);  sub_172 = unsqueeze_278 = None
    sub_174: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_255, mul_985);  view_255 = mul_985 = None
    sub_175: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_276);  sub_174 = unsqueeze_276 = None
    mul_986: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_280);  sub_175 = unsqueeze_280 = None
    mul_987: "f32[768]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_59);  sum_113 = squeeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_256: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_987, [768, 1, 1, 1]);  mul_987 = None
    mul_988: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_256, 0.02946278254943948);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_257: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_986, [768, 128, 3, 3]);  mul_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_989: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_237, 1.7015043497085571);  getitem_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_991: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_59, 0.5);  add_59 = None
    mul_992: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_38, convolution_38)
    mul_993: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_992, -0.5);  mul_992 = None
    exp_26: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_993);  mul_993 = None
    mul_994: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_995: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_38, mul_994);  convolution_38 = mul_994 = None
    add_187: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_991, mul_995);  mul_991 = mul_995 = None
    mul_996: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_989, add_187);  mul_989 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_996, [0, 2, 3])
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_996, mul_209, view_86, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_996 = mul_209 = view_86 = None
    getitem_240: "f32[4, 768, 12, 12]" = convolution_backward_42[0]
    getitem_241: "f32[768, 128, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_258: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_241, [1, 768, 1152]);  getitem_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_258, [0, 2])
    sub_176: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_84, unsqueeze_282);  view_84 = unsqueeze_282 = None
    mul_997: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_258, sub_176)
    sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_997, [0, 2]);  mul_997 = None
    mul_998: "f32[768]" = torch.ops.aten.mul.Tensor(sum_115, 0.0008680555555555555);  sum_115 = None
    unsqueeze_283: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_998, 0);  mul_998 = None
    unsqueeze_284: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    mul_999: "f32[768]" = torch.ops.aten.mul.Tensor(sum_116, 0.0008680555555555555)
    mul_1000: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_57, squeeze_57)
    mul_1001: "f32[768]" = torch.ops.aten.mul.Tensor(mul_999, mul_1000);  mul_999 = mul_1000 = None
    unsqueeze_285: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1001, 0);  mul_1001 = None
    unsqueeze_286: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
    mul_1002: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_57, view_85);  view_85 = None
    unsqueeze_287: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1002, 0);  mul_1002 = None
    unsqueeze_288: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    mul_1003: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_286);  sub_176 = unsqueeze_286 = None
    sub_178: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_258, mul_1003);  view_258 = mul_1003 = None
    sub_179: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_284);  sub_178 = unsqueeze_284 = None
    mul_1004: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_288);  sub_179 = unsqueeze_288 = None
    mul_1005: "f32[768]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_57);  sum_116 = squeeze_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_259: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1005, [768, 1, 1, 1]);  mul_1005 = None
    mul_1006: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_259, 0.02946278254943948);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_260: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_1004, [768, 128, 3, 3]);  mul_1004 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1007: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_240, 1.7015043497085571);  getitem_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1009: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_57, 0.5);  add_57 = None
    mul_1010: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_37, convolution_37)
    mul_1011: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1010, -0.5);  mul_1010 = None
    exp_27: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_1011);  mul_1011 = None
    mul_1012: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_1013: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_37, mul_1012);  convolution_37 = mul_1012 = None
    add_189: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_1009, mul_1013);  mul_1009 = mul_1013 = None
    mul_1014: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1007, add_189);  mul_1007 = add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1014, [0, 2, 3])
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1014, mul_202, view_83, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1014 = mul_202 = view_83 = None
    getitem_243: "f32[4, 1536, 12, 12]" = convolution_backward_43[0]
    getitem_244: "f32[768, 1536, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_261: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_244, [1, 768, 1536]);  getitem_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_261, [0, 2])
    sub_180: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_81, unsqueeze_290);  view_81 = unsqueeze_290 = None
    mul_1015: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_261, sub_180)
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1015, [0, 2]);  mul_1015 = None
    mul_1016: "f32[768]" = torch.ops.aten.mul.Tensor(sum_118, 0.0006510416666666666);  sum_118 = None
    unsqueeze_291: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1016, 0);  mul_1016 = None
    unsqueeze_292: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
    mul_1017: "f32[768]" = torch.ops.aten.mul.Tensor(sum_119, 0.0006510416666666666)
    mul_1018: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1019: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1017, mul_1018);  mul_1017 = mul_1018 = None
    unsqueeze_293: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
    unsqueeze_294: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    mul_1020: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, view_82);  view_82 = None
    unsqueeze_295: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1020, 0);  mul_1020 = None
    unsqueeze_296: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    mul_1021: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_294);  sub_180 = unsqueeze_294 = None
    sub_182: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_261, mul_1021);  view_261 = mul_1021 = None
    sub_183: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_292);  sub_182 = unsqueeze_292 = None
    mul_1022: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_296);  sub_183 = unsqueeze_296 = None
    mul_1023: "f32[768]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_55);  sum_119 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_262: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1023, [768, 1, 1, 1]);  mul_1023 = None
    mul_1024: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_262, 0.02551551815399144);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_263: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_1022, [768, 1536, 1, 1]);  mul_1022 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1025: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_243, 0.9622504486493761);  getitem_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1026: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1025, 1.7015043497085571);  mul_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1028: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
    mul_1029: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_54, add_54)
    mul_1030: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1029, -0.5);  mul_1029 = None
    exp_28: "f32[4, 1536, 12, 12]" = torch.ops.aten.exp.default(mul_1030);  mul_1030 = None
    mul_1031: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_1032: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_54, mul_1031);  add_54 = mul_1031 = None
    add_191: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_1028, mul_1032);  mul_1028 = mul_1032 = None
    mul_1033: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1026, add_191);  mul_1026 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_192: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(add_182, mul_1033);  add_182 = mul_1033 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_1034: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_192, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_1035: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1034, clone_4);  clone_4 = None
    mul_1036: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1034, primals_86);  mul_1034 = primals_86 = None
    sum_120: "f32[]" = torch.ops.aten.sum.default(mul_1035);  mul_1035 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_1037: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1036, 2.0);  mul_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_1038: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1037, convolution_34);  convolution_34 = None
    mul_1039: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1037, sigmoid_4);  mul_1037 = sigmoid_4 = None
    sum_121: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1038, [2, 3], True);  mul_1038 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_52: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    sub_184: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_52)
    mul_1040: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_52, sub_184);  alias_52 = sub_184 = None
    mul_1041: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_121, mul_1040);  sum_121 = mul_1040 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_122: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1041, [0, 2, 3])
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1041, relu_4, primals_202, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1041 = primals_202 = None
    getitem_246: "f32[4, 768, 1, 1]" = convolution_backward_44[0]
    getitem_247: "f32[1536, 768, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_54: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_55: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_7: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    where_7: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_7, full_default, getitem_246);  le_7 = getitem_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(where_7, mean_4, primals_200, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_7 = mean_4 = primals_200 = None
    getitem_249: "f32[4, 1536, 1, 1]" = convolution_backward_45[0]
    getitem_250: "f32[768, 1536, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[4, 1536, 12, 12]" = torch.ops.aten.expand.default(getitem_249, [4, 1536, 12, 12]);  getitem_249 = None
    div_8: "f32[4, 1536, 12, 12]" = torch.ops.aten.div.Scalar(expand_8, 144);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_193: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_1039, div_8);  mul_1039 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_124: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_193, [0, 2, 3])
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(add_193, mul_190, view_80, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_193 = mul_190 = view_80 = None
    getitem_252: "f32[4, 768, 12, 12]" = convolution_backward_46[0]
    getitem_253: "f32[1536, 768, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_264: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_253, [1, 1536, 768]);  getitem_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_125: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_264, [0, 2])
    sub_185: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_78, unsqueeze_298);  view_78 = unsqueeze_298 = None
    mul_1042: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_264, sub_185)
    sum_126: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1042, [0, 2]);  mul_1042 = None
    mul_1043: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_125, 0.0013020833333333333);  sum_125 = None
    unsqueeze_299: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1043, 0);  mul_1043 = None
    unsqueeze_300: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    mul_1044: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_126, 0.0013020833333333333)
    mul_1045: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_53, squeeze_53)
    mul_1046: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_1044, mul_1045);  mul_1044 = mul_1045 = None
    unsqueeze_301: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
    unsqueeze_302: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    mul_1047: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_53, view_79);  view_79 = None
    unsqueeze_303: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1047, 0);  mul_1047 = None
    unsqueeze_304: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
    mul_1048: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_302);  sub_185 = unsqueeze_302 = None
    sub_187: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_264, mul_1048);  view_264 = mul_1048 = None
    sub_188: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_300);  sub_187 = unsqueeze_300 = None
    mul_1049: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_304);  sub_188 = unsqueeze_304 = None
    mul_1050: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_126, squeeze_53);  sum_126 = squeeze_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_265: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_1050, [1536, 1, 1, 1]);  mul_1050 = None
    mul_1051: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_265, 0.03608439182435161);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_266: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_1049, [1536, 768, 1, 1]);  mul_1049 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1052: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_252, 1.7015043497085571);  getitem_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1054: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_52, 0.5);  add_52 = None
    mul_1055: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_33, convolution_33)
    mul_1056: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1055, -0.5);  mul_1055 = None
    exp_29: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_1056);  mul_1056 = None
    mul_1057: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_1058: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_33, mul_1057);  convolution_33 = mul_1057 = None
    add_195: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_1054, mul_1058);  mul_1054 = mul_1058 = None
    mul_1059: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1052, add_195);  mul_1052 = add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1059, [0, 2, 3])
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1059, mul_183, view_77, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_1059 = mul_183 = view_77 = None
    getitem_255: "f32[4, 768, 12, 12]" = convolution_backward_47[0]
    getitem_256: "f32[768, 128, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_267: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_256, [1, 768, 1152]);  getitem_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_128: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_267, [0, 2])
    sub_189: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_75, unsqueeze_306);  view_75 = unsqueeze_306 = None
    mul_1060: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_267, sub_189)
    sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1060, [0, 2]);  mul_1060 = None
    mul_1061: "f32[768]" = torch.ops.aten.mul.Tensor(sum_128, 0.0008680555555555555);  sum_128 = None
    unsqueeze_307: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1061, 0);  mul_1061 = None
    unsqueeze_308: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    mul_1062: "f32[768]" = torch.ops.aten.mul.Tensor(sum_129, 0.0008680555555555555)
    mul_1063: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_51, squeeze_51)
    mul_1064: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1062, mul_1063);  mul_1062 = mul_1063 = None
    unsqueeze_309: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
    unsqueeze_310: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
    mul_1065: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_51, view_76);  view_76 = None
    unsqueeze_311: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1065, 0);  mul_1065 = None
    unsqueeze_312: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    mul_1066: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_310);  sub_189 = unsqueeze_310 = None
    sub_191: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_267, mul_1066);  view_267 = mul_1066 = None
    sub_192: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_308);  sub_191 = unsqueeze_308 = None
    mul_1067: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_312);  sub_192 = unsqueeze_312 = None
    mul_1068: "f32[768]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_51);  sum_129 = squeeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_268: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1068, [768, 1, 1, 1]);  mul_1068 = None
    mul_1069: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_268, 0.02946278254943948);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_269: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_1067, [768, 128, 3, 3]);  mul_1067 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1070: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_255, 1.7015043497085571);  getitem_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1072: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_50, 0.5);  add_50 = None
    mul_1073: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_32, convolution_32)
    mul_1074: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1073, -0.5);  mul_1073 = None
    exp_30: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_1074);  mul_1074 = None
    mul_1075: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_1076: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_32, mul_1075);  convolution_32 = mul_1075 = None
    add_197: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_1072, mul_1076);  mul_1072 = mul_1076 = None
    mul_1077: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1070, add_197);  mul_1070 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1077, [0, 2, 3])
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1077, mul_176, view_74, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_1077 = mul_176 = view_74 = None
    getitem_258: "f32[4, 768, 12, 12]" = convolution_backward_48[0]
    getitem_259: "f32[768, 128, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_270: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_259, [1, 768, 1152]);  getitem_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_270, [0, 2])
    sub_193: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_72, unsqueeze_314);  view_72 = unsqueeze_314 = None
    mul_1078: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_270, sub_193)
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1078, [0, 2]);  mul_1078 = None
    mul_1079: "f32[768]" = torch.ops.aten.mul.Tensor(sum_131, 0.0008680555555555555);  sum_131 = None
    unsqueeze_315: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1079, 0);  mul_1079 = None
    unsqueeze_316: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
    mul_1080: "f32[768]" = torch.ops.aten.mul.Tensor(sum_132, 0.0008680555555555555)
    mul_1081: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1082: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1080, mul_1081);  mul_1080 = mul_1081 = None
    unsqueeze_317: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
    unsqueeze_318: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    mul_1083: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_49, view_73);  view_73 = None
    unsqueeze_319: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1083, 0);  mul_1083 = None
    unsqueeze_320: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    mul_1084: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_318);  sub_193 = unsqueeze_318 = None
    sub_195: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_270, mul_1084);  view_270 = mul_1084 = None
    sub_196: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_316);  sub_195 = unsqueeze_316 = None
    mul_1085: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_320);  sub_196 = unsqueeze_320 = None
    mul_1086: "f32[768]" = torch.ops.aten.mul.Tensor(sum_132, squeeze_49);  sum_132 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_271: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1086, [768, 1, 1, 1]);  mul_1086 = None
    mul_1087: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_271, 0.02946278254943948);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_272: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_1085, [768, 128, 3, 3]);  mul_1085 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1088: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_258, 1.7015043497085571);  getitem_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1090: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_1091: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_31, convolution_31)
    mul_1092: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1091, -0.5);  mul_1091 = None
    exp_31: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_1092);  mul_1092 = None
    mul_1093: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_1094: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_31, mul_1093);  convolution_31 = mul_1093 = None
    add_199: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_1090, mul_1094);  mul_1090 = mul_1094 = None
    mul_1095: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1088, add_199);  mul_1088 = add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1095, [0, 2, 3])
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1095, mul_169, view_71, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1095 = mul_169 = view_71 = None
    getitem_261: "f32[4, 1536, 12, 12]" = convolution_backward_49[0]
    getitem_262: "f32[768, 1536, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_273: "f32[1, 768, 1536]" = torch.ops.aten.view.default(getitem_262, [1, 768, 1536]);  getitem_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_134: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 2])
    sub_197: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_69, unsqueeze_322);  view_69 = unsqueeze_322 = None
    mul_1096: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(view_273, sub_197)
    sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1096, [0, 2]);  mul_1096 = None
    mul_1097: "f32[768]" = torch.ops.aten.mul.Tensor(sum_134, 0.0006510416666666666);  sum_134 = None
    unsqueeze_323: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1097, 0);  mul_1097 = None
    unsqueeze_324: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    mul_1098: "f32[768]" = torch.ops.aten.mul.Tensor(sum_135, 0.0006510416666666666)
    mul_1099: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_47, squeeze_47)
    mul_1100: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1098, mul_1099);  mul_1098 = mul_1099 = None
    unsqueeze_325: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1100, 0);  mul_1100 = None
    unsqueeze_326: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    mul_1101: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_47, view_70);  view_70 = None
    unsqueeze_327: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1101, 0);  mul_1101 = None
    unsqueeze_328: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
    mul_1102: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_326);  sub_197 = unsqueeze_326 = None
    sub_199: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(view_273, mul_1102);  view_273 = mul_1102 = None
    sub_200: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_324);  sub_199 = unsqueeze_324 = None
    mul_1103: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_328);  sub_200 = unsqueeze_328 = None
    mul_1104: "f32[768]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_47);  sum_135 = squeeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_274: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1104, [768, 1, 1, 1]);  mul_1104 = None
    mul_1105: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_274, 0.02551551815399144);  view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_275: "f32[768, 1536, 1, 1]" = torch.ops.aten.view.default(mul_1103, [768, 1536, 1, 1]);  mul_1103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1106: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_261, 0.9805806756909201);  getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1107: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1106, 1.7015043497085571);  mul_1106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1109: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_46, 0.5);  add_46 = None
    mul_1110: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_45, add_45)
    mul_1111: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1110, -0.5);  mul_1110 = None
    exp_32: "f32[4, 1536, 12, 12]" = torch.ops.aten.exp.default(mul_1111);  mul_1111 = None
    mul_1112: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_1113: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_45, mul_1112);  add_45 = mul_1112 = None
    add_201: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_1109, mul_1113);  mul_1109 = mul_1113 = None
    mul_1114: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1107, add_201);  mul_1107 = add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_202: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(add_192, mul_1114);  add_192 = mul_1114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_1115: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(add_202, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_1116: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1115, clone_3);  clone_3 = None
    mul_1117: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1115, primals_73);  mul_1115 = primals_73 = None
    sum_136: "f32[]" = torch.ops.aten.sum.default(mul_1116);  mul_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_1118: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1117, 2.0);  mul_1117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_1119: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1118, convolution_28);  convolution_28 = None
    mul_1120: "f32[4, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1118, sigmoid_3);  mul_1118 = sigmoid_3 = None
    sum_137: "f32[4, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1119, [2, 3], True);  mul_1119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_56: "f32[4, 1536, 1, 1]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    sub_201: "f32[4, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_56)
    mul_1121: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(alias_56, sub_201);  alias_56 = sub_201 = None
    mul_1122: "f32[4, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_137, mul_1121);  sum_137 = mul_1121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_138: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1122, [0, 2, 3])
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1122, relu_3, primals_198, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1122 = primals_198 = None
    getitem_264: "f32[4, 768, 1, 1]" = convolution_backward_50[0]
    getitem_265: "f32[1536, 768, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_58: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_59: "f32[4, 768, 1, 1]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    le_8: "b8[4, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_59, 0);  alias_59 = None
    where_8: "f32[4, 768, 1, 1]" = torch.ops.aten.where.self(le_8, full_default, getitem_264);  le_8 = getitem_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_139: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(where_8, mean_3, primals_196, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = mean_3 = primals_196 = None
    getitem_267: "f32[4, 1536, 1, 1]" = convolution_backward_51[0]
    getitem_268: "f32[768, 1536, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[4, 1536, 12, 12]" = torch.ops.aten.expand.default(getitem_267, [4, 1536, 12, 12]);  getitem_267 = None
    div_9: "f32[4, 1536, 12, 12]" = torch.ops.aten.div.Scalar(expand_9, 144);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_203: "f32[4, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_1120, div_9);  mul_1120 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_140: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_203, [0, 2, 3])
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(add_203, mul_157, view_68, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_203 = mul_157 = view_68 = None
    getitem_270: "f32[4, 768, 12, 12]" = convolution_backward_52[0]
    getitem_271: "f32[1536, 768, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_276: "f32[1, 1536, 768]" = torch.ops.aten.view.default(getitem_271, [1, 1536, 768]);  getitem_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_141: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_276, [0, 2])
    sub_202: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_66, unsqueeze_330);  view_66 = unsqueeze_330 = None
    mul_1123: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(view_276, sub_202)
    sum_142: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1123, [0, 2]);  mul_1123 = None
    mul_1124: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_141, 0.0013020833333333333);  sum_141 = None
    unsqueeze_331: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1124, 0);  mul_1124 = None
    unsqueeze_332: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    mul_1125: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_142, 0.0013020833333333333)
    mul_1126: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_45, squeeze_45)
    mul_1127: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_1125, mul_1126);  mul_1125 = mul_1126 = None
    unsqueeze_333: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1127, 0);  mul_1127 = None
    unsqueeze_334: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
    mul_1128: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_45, view_67);  view_67 = None
    unsqueeze_335: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1128, 0);  mul_1128 = None
    unsqueeze_336: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    mul_1129: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_334);  sub_202 = unsqueeze_334 = None
    sub_204: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_276, mul_1129);  view_276 = mul_1129 = None
    sub_205: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_204, unsqueeze_332);  sub_204 = unsqueeze_332 = None
    mul_1130: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_336);  sub_205 = unsqueeze_336 = None
    mul_1131: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_142, squeeze_45);  sum_142 = squeeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_277: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_1131, [1536, 1, 1, 1]);  mul_1131 = None
    mul_1132: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_277, 0.03608439182435161);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_278: "f32[1536, 768, 1, 1]" = torch.ops.aten.view.default(mul_1130, [1536, 768, 1, 1]);  mul_1130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1133: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_270, 1.7015043497085571);  getitem_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1135: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_43, 0.5);  add_43 = None
    mul_1136: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_27, convolution_27)
    mul_1137: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1136, -0.5);  mul_1136 = None
    exp_33: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_1137);  mul_1137 = None
    mul_1138: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_1139: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_27, mul_1138);  convolution_27 = mul_1138 = None
    add_205: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_1135, mul_1139);  mul_1135 = mul_1139 = None
    mul_1140: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1133, add_205);  mul_1133 = add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1140, [0, 2, 3])
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1140, mul_150, view_65, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_1140 = mul_150 = view_65 = None
    getitem_273: "f32[4, 768, 12, 12]" = convolution_backward_53[0]
    getitem_274: "f32[768, 128, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_279: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_274, [1, 768, 1152]);  getitem_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_144: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_279, [0, 2])
    sub_206: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_63, unsqueeze_338);  view_63 = unsqueeze_338 = None
    mul_1141: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_279, sub_206)
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1141, [0, 2]);  mul_1141 = None
    mul_1142: "f32[768]" = torch.ops.aten.mul.Tensor(sum_144, 0.0008680555555555555);  sum_144 = None
    unsqueeze_339: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1142, 0);  mul_1142 = None
    unsqueeze_340: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
    mul_1143: "f32[768]" = torch.ops.aten.mul.Tensor(sum_145, 0.0008680555555555555)
    mul_1144: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1145: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1143, mul_1144);  mul_1143 = mul_1144 = None
    unsqueeze_341: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1145, 0);  mul_1145 = None
    unsqueeze_342: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    mul_1146: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_43, view_64);  view_64 = None
    unsqueeze_343: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1146, 0);  mul_1146 = None
    unsqueeze_344: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    mul_1147: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_342);  sub_206 = unsqueeze_342 = None
    sub_208: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_279, mul_1147);  view_279 = mul_1147 = None
    sub_209: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_208, unsqueeze_340);  sub_208 = unsqueeze_340 = None
    mul_1148: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_344);  sub_209 = unsqueeze_344 = None
    mul_1149: "f32[768]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_43);  sum_145 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_280: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1149, [768, 1, 1, 1]);  mul_1149 = None
    mul_1150: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_280, 0.02946278254943948);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_281: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_1148, [768, 128, 3, 3]);  mul_1148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1151: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_273, 1.7015043497085571);  getitem_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1153: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
    mul_1154: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_26, convolution_26)
    mul_1155: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1154, -0.5);  mul_1154 = None
    exp_34: "f32[4, 768, 12, 12]" = torch.ops.aten.exp.default(mul_1155);  mul_1155 = None
    mul_1156: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_1157: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_26, mul_1156);  convolution_26 = mul_1156 = None
    add_207: "f32[4, 768, 12, 12]" = torch.ops.aten.add.Tensor(mul_1153, mul_1157);  mul_1153 = mul_1157 = None
    mul_1158: "f32[4, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1151, add_207);  mul_1151 = add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1158, [0, 2, 3])
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1158, constant_pad_nd_3, view_62, [768], [2, 2], [0, 0], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_1158 = constant_pad_nd_3 = view_62 = None
    getitem_276: "f32[4, 768, 25, 25]" = convolution_backward_54[0]
    getitem_277: "f32[768, 128, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_282: "f32[1, 768, 1152]" = torch.ops.aten.view.default(getitem_277, [1, 768, 1152]);  getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_147: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_282, [0, 2])
    sub_210: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_60, unsqueeze_346);  view_60 = unsqueeze_346 = None
    mul_1159: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(view_282, sub_210)
    sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1159, [0, 2]);  mul_1159 = None
    mul_1160: "f32[768]" = torch.ops.aten.mul.Tensor(sum_147, 0.0008680555555555555);  sum_147 = None
    unsqueeze_347: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1160, 0);  mul_1160 = None
    unsqueeze_348: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    mul_1161: "f32[768]" = torch.ops.aten.mul.Tensor(sum_148, 0.0008680555555555555)
    mul_1162: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_41, squeeze_41)
    mul_1163: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1161, mul_1162);  mul_1161 = mul_1162 = None
    unsqueeze_349: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1163, 0);  mul_1163 = None
    unsqueeze_350: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    mul_1164: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_41, view_61);  view_61 = None
    unsqueeze_351: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1164, 0);  mul_1164 = None
    unsqueeze_352: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
    mul_1165: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_350);  sub_210 = unsqueeze_350 = None
    sub_212: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view_282, mul_1165);  view_282 = mul_1165 = None
    sub_213: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_212, unsqueeze_348);  sub_212 = unsqueeze_348 = None
    mul_1166: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_352);  sub_213 = unsqueeze_352 = None
    mul_1167: "f32[768]" = torch.ops.aten.mul.Tensor(sum_148, squeeze_41);  sum_148 = squeeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_283: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1167, [768, 1, 1, 1]);  mul_1167 = None
    mul_1168: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_283, 0.02946278254943948);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_284: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_1166, [768, 128, 3, 3]);  mul_1166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_6: "f32[4, 768, 24, 24]" = torch.ops.aten.constant_pad_nd.default(getitem_276, [0, -1, 0, -1]);  getitem_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1169: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(constant_pad_nd_6, 1.7015043497085571);  constant_pad_nd_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1171: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(add_39, 0.5);  add_39 = None
    mul_1172: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_25, convolution_25)
    mul_1173: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1172, -0.5);  mul_1172 = None
    exp_35: "f32[4, 768, 24, 24]" = torch.ops.aten.exp.default(mul_1173);  mul_1173 = None
    mul_1174: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_1175: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_25, mul_1174);  convolution_25 = mul_1174 = None
    add_209: "f32[4, 768, 24, 24]" = torch.ops.aten.add.Tensor(mul_1171, mul_1175);  mul_1171 = mul_1175 = None
    mul_1176: "f32[4, 768, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1169, add_209);  mul_1169 = add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_149: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1176, [0, 2, 3])
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1176, mul_133, view_59, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1176 = view_59 = None
    getitem_279: "f32[4, 512, 24, 24]" = convolution_backward_55[0]
    getitem_280: "f32[768, 512, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_285: "f32[1, 768, 512]" = torch.ops.aten.view.default(getitem_280, [1, 768, 512]);  getitem_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_150: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_285, [0, 2])
    sub_214: "f32[1, 768, 512]" = torch.ops.aten.sub.Tensor(view_57, unsqueeze_354);  view_57 = unsqueeze_354 = None
    mul_1177: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(view_285, sub_214)
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1177, [0, 2]);  mul_1177 = None
    mul_1178: "f32[768]" = torch.ops.aten.mul.Tensor(sum_150, 0.001953125);  sum_150 = None
    unsqueeze_355: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1178, 0);  mul_1178 = None
    unsqueeze_356: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    mul_1179: "f32[768]" = torch.ops.aten.mul.Tensor(sum_151, 0.001953125)
    mul_1180: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_39, squeeze_39)
    mul_1181: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1179, mul_1180);  mul_1179 = mul_1180 = None
    unsqueeze_357: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1181, 0);  mul_1181 = None
    unsqueeze_358: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    mul_1182: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_39, view_58);  view_58 = None
    unsqueeze_359: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1182, 0);  mul_1182 = None
    unsqueeze_360: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    mul_1183: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_358);  sub_214 = unsqueeze_358 = None
    sub_216: "f32[1, 768, 512]" = torch.ops.aten.sub.Tensor(view_285, mul_1183);  view_285 = mul_1183 = None
    sub_217: "f32[1, 768, 512]" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_356);  sub_216 = unsqueeze_356 = None
    mul_1184: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_360);  sub_217 = unsqueeze_360 = None
    mul_1185: "f32[768]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_39);  sum_151 = squeeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_286: "f32[768, 1, 1, 1]" = torch.ops.aten.view.default(mul_1185, [768, 1, 1, 1]);  mul_1185 = None
    mul_1186: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_286, 0.04419417382415922);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_287: "f32[768, 512, 1, 1]" = torch.ops.aten.view.default(mul_1184, [768, 512, 1, 1]);  mul_1184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_152: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_202, [0, 2, 3])
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(add_202, avg_pool2d_1, view_56, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_202 = avg_pool2d_1 = view_56 = None
    getitem_282: "f32[4, 512, 12, 12]" = convolution_backward_56[0]
    getitem_283: "f32[1536, 512, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_288: "f32[1, 1536, 512]" = torch.ops.aten.view.default(getitem_283, [1, 1536, 512]);  getitem_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_153: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_288, [0, 2])
    sub_218: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_54, unsqueeze_362);  view_54 = unsqueeze_362 = None
    mul_1187: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(view_288, sub_218)
    sum_154: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1187, [0, 2]);  mul_1187 = None
    mul_1188: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_153, 0.001953125);  sum_153 = None
    unsqueeze_363: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1188, 0);  mul_1188 = None
    unsqueeze_364: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
    mul_1189: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_154, 0.001953125)
    mul_1190: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1191: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_1189, mul_1190);  mul_1189 = mul_1190 = None
    unsqueeze_365: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1191, 0);  mul_1191 = None
    unsqueeze_366: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    mul_1192: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_37, view_55);  view_55 = None
    unsqueeze_367: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1192, 0);  mul_1192 = None
    unsqueeze_368: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    mul_1193: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_366);  sub_218 = unsqueeze_366 = None
    sub_220: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_288, mul_1193);  view_288 = mul_1193 = None
    sub_221: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(sub_220, unsqueeze_364);  sub_220 = unsqueeze_364 = None
    mul_1194: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_368);  sub_221 = unsqueeze_368 = None
    mul_1195: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_154, squeeze_37);  sum_154 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_289: "f32[1536, 1, 1, 1]" = torch.ops.aten.view.default(mul_1195, [1536, 1, 1, 1]);  mul_1195 = None
    mul_1196: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_289, 0.04419417382415922);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_290: "f32[1536, 512, 1, 1]" = torch.ops.aten.view.default(mul_1194, [1536, 512, 1, 1]);  mul_1194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_backward_1: "f32[4, 512, 24, 24]" = torch.ops.aten.avg_pool2d_backward.default(getitem_282, mul_133, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_282 = mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    add_210: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(getitem_279, avg_pool2d_backward_1);  getitem_279 = avg_pool2d_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1197: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_210, 0.9622504486493761);  add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1198: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1197, 1.7015043497085571);  mul_1197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1200: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_36, 0.5);  add_36 = None
    mul_1201: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_35, add_35)
    mul_1202: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1201, -0.5);  mul_1201 = None
    exp_36: "f32[4, 512, 24, 24]" = torch.ops.aten.exp.default(mul_1202);  mul_1202 = None
    mul_1203: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_1204: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_35, mul_1203);  add_35 = mul_1203 = None
    add_212: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_1200, mul_1204);  mul_1200 = mul_1204 = None
    mul_1205: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1198, add_212);  mul_1198 = add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_1206: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1205, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_1207: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1206, clone_2);  clone_2 = None
    mul_1208: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1206, primals_57);  mul_1206 = primals_57 = None
    sum_155: "f32[]" = torch.ops.aten.sum.default(mul_1207);  mul_1207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_1209: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1208, 2.0);  mul_1208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_1210: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1209, convolution_21);  convolution_21 = None
    mul_1211: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1209, sigmoid_2);  mul_1209 = sigmoid_2 = None
    sum_156: "f32[4, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1210, [2, 3], True);  mul_1210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_60: "f32[4, 512, 1, 1]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    sub_222: "f32[4, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_60)
    mul_1212: "f32[4, 512, 1, 1]" = torch.ops.aten.mul.Tensor(alias_60, sub_222);  alias_60 = sub_222 = None
    mul_1213: "f32[4, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_156, mul_1212);  sum_156 = mul_1212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_157: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1213, [0, 2, 3])
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1213, relu_2, primals_194, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1213 = primals_194 = None
    getitem_285: "f32[4, 256, 1, 1]" = convolution_backward_57[0]
    getitem_286: "f32[512, 256, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_62: "f32[4, 256, 1, 1]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_63: "f32[4, 256, 1, 1]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    le_9: "b8[4, 256, 1, 1]" = torch.ops.aten.le.Scalar(alias_63, 0);  alias_63 = None
    where_9: "f32[4, 256, 1, 1]" = torch.ops.aten.where.self(le_9, full_default, getitem_285);  le_9 = getitem_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_158: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(where_9, mean_2, primals_192, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = mean_2 = primals_192 = None
    getitem_288: "f32[4, 512, 1, 1]" = convolution_backward_58[0]
    getitem_289: "f32[256, 512, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[4, 512, 24, 24]" = torch.ops.aten.expand.default(getitem_288, [4, 512, 24, 24]);  getitem_288 = None
    div_10: "f32[4, 512, 24, 24]" = torch.ops.aten.div.Scalar(expand_10, 576);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_213: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_1211, div_10);  mul_1211 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_159: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_213, [0, 2, 3])
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(add_213, mul_121, view_53, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_213 = mul_121 = view_53 = None
    getitem_291: "f32[4, 256, 24, 24]" = convolution_backward_59[0]
    getitem_292: "f32[512, 256, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_291: "f32[1, 512, 256]" = torch.ops.aten.view.default(getitem_292, [1, 512, 256]);  getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_160: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_291, [0, 2])
    sub_223: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_51, unsqueeze_370);  view_51 = unsqueeze_370 = None
    mul_1214: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_291, sub_223)
    sum_161: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1214, [0, 2]);  mul_1214 = None
    mul_1215: "f32[512]" = torch.ops.aten.mul.Tensor(sum_160, 0.00390625);  sum_160 = None
    unsqueeze_371: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1215, 0);  mul_1215 = None
    unsqueeze_372: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    mul_1216: "f32[512]" = torch.ops.aten.mul.Tensor(sum_161, 0.00390625)
    mul_1217: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_35, squeeze_35)
    mul_1218: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1216, mul_1217);  mul_1216 = mul_1217 = None
    unsqueeze_373: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1218, 0);  mul_1218 = None
    unsqueeze_374: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    mul_1219: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_35, view_52);  view_52 = None
    unsqueeze_375: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1219, 0);  mul_1219 = None
    unsqueeze_376: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
    mul_1220: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_374);  sub_223 = unsqueeze_374 = None
    sub_225: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_291, mul_1220);  view_291 = mul_1220 = None
    sub_226: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_225, unsqueeze_372);  sub_225 = unsqueeze_372 = None
    mul_1221: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_376);  sub_226 = unsqueeze_376 = None
    mul_1222: "f32[512]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_35);  sum_161 = squeeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_292: "f32[512, 1, 1, 1]" = torch.ops.aten.view.default(mul_1222, [512, 1, 1, 1]);  mul_1222 = None
    mul_1223: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_292, 0.0625);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_293: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_1221, [512, 256, 1, 1]);  mul_1221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1224: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_291, 1.7015043497085571);  getitem_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1226: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(add_33, 0.5);  add_33 = None
    mul_1227: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_20, convolution_20)
    mul_1228: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1227, -0.5);  mul_1227 = None
    exp_37: "f32[4, 256, 24, 24]" = torch.ops.aten.exp.default(mul_1228);  mul_1228 = None
    mul_1229: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_1230: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_20, mul_1229);  convolution_20 = mul_1229 = None
    add_215: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(mul_1226, mul_1230);  mul_1226 = mul_1230 = None
    mul_1231: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1224, add_215);  mul_1224 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_162: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1231, [0, 2, 3])
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1231, mul_114, view_50, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1231 = mul_114 = view_50 = None
    getitem_294: "f32[4, 256, 24, 24]" = convolution_backward_60[0]
    getitem_295: "f32[256, 128, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_294: "f32[1, 256, 1152]" = torch.ops.aten.view.default(getitem_295, [1, 256, 1152]);  getitem_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_163: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_294, [0, 2])
    sub_227: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_48, unsqueeze_378);  view_48 = unsqueeze_378 = None
    mul_1232: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(view_294, sub_227)
    sum_164: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1232, [0, 2]);  mul_1232 = None
    mul_1233: "f32[256]" = torch.ops.aten.mul.Tensor(sum_163, 0.0008680555555555555);  sum_163 = None
    unsqueeze_379: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1233, 0);  mul_1233 = None
    unsqueeze_380: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    mul_1234: "f32[256]" = torch.ops.aten.mul.Tensor(sum_164, 0.0008680555555555555)
    mul_1235: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, squeeze_33)
    mul_1236: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1234, mul_1235);  mul_1234 = mul_1235 = None
    unsqueeze_381: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1236, 0);  mul_1236 = None
    unsqueeze_382: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
    mul_1237: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, view_49);  view_49 = None
    unsqueeze_383: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1237, 0);  mul_1237 = None
    unsqueeze_384: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    mul_1238: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_382);  sub_227 = unsqueeze_382 = None
    sub_229: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_294, mul_1238);  view_294 = mul_1238 = None
    sub_230: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(sub_229, unsqueeze_380);  sub_229 = unsqueeze_380 = None
    mul_1239: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_384);  sub_230 = unsqueeze_384 = None
    mul_1240: "f32[256]" = torch.ops.aten.mul.Tensor(sum_164, squeeze_33);  sum_164 = squeeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_295: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1240, [256, 1, 1, 1]);  mul_1240 = None
    mul_1241: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_295, 0.02946278254943948);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_296: "f32[256, 128, 3, 3]" = torch.ops.aten.view.default(mul_1239, [256, 128, 3, 3]);  mul_1239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1242: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_294, 1.7015043497085571);  getitem_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1244: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(add_31, 0.5);  add_31 = None
    mul_1245: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_19, convolution_19)
    mul_1246: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1245, -0.5);  mul_1245 = None
    exp_38: "f32[4, 256, 24, 24]" = torch.ops.aten.exp.default(mul_1246);  mul_1246 = None
    mul_1247: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_1248: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_19, mul_1247);  convolution_19 = mul_1247 = None
    add_217: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(mul_1244, mul_1248);  mul_1244 = mul_1248 = None
    mul_1249: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1242, add_217);  mul_1242 = add_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_165: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1249, [0, 2, 3])
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1249, mul_107, view_47, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1249 = mul_107 = view_47 = None
    getitem_297: "f32[4, 256, 24, 24]" = convolution_backward_61[0]
    getitem_298: "f32[256, 128, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_297: "f32[1, 256, 1152]" = torch.ops.aten.view.default(getitem_298, [1, 256, 1152]);  getitem_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_166: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_297, [0, 2])
    sub_231: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_45, unsqueeze_386);  view_45 = unsqueeze_386 = None
    mul_1250: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(view_297, sub_231)
    sum_167: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1250, [0, 2]);  mul_1250 = None
    mul_1251: "f32[256]" = torch.ops.aten.mul.Tensor(sum_166, 0.0008680555555555555);  sum_166 = None
    unsqueeze_387: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1251, 0);  mul_1251 = None
    unsqueeze_388: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
    mul_1252: "f32[256]" = torch.ops.aten.mul.Tensor(sum_167, 0.0008680555555555555)
    mul_1253: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1254: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1252, mul_1253);  mul_1252 = mul_1253 = None
    unsqueeze_389: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1254, 0);  mul_1254 = None
    unsqueeze_390: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    mul_1255: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, view_46);  view_46 = None
    unsqueeze_391: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1255, 0);  mul_1255 = None
    unsqueeze_392: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    mul_1256: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_390);  sub_231 = unsqueeze_390 = None
    sub_233: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_297, mul_1256);  view_297 = mul_1256 = None
    sub_234: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(sub_233, unsqueeze_388);  sub_233 = unsqueeze_388 = None
    mul_1257: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_392);  sub_234 = unsqueeze_392 = None
    mul_1258: "f32[256]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_31);  sum_167 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_298: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1258, [256, 1, 1, 1]);  mul_1258 = None
    mul_1259: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_298, 0.02946278254943948);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_299: "f32[256, 128, 3, 3]" = torch.ops.aten.view.default(mul_1257, [256, 128, 3, 3]);  mul_1257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1260: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_297, 1.7015043497085571);  getitem_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1262: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(add_29, 0.5);  add_29 = None
    mul_1263: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_18, convolution_18)
    mul_1264: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1263, -0.5);  mul_1263 = None
    exp_39: "f32[4, 256, 24, 24]" = torch.ops.aten.exp.default(mul_1264);  mul_1264 = None
    mul_1265: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_1266: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_18, mul_1265);  convolution_18 = mul_1265 = None
    add_219: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(mul_1262, mul_1266);  mul_1262 = mul_1266 = None
    mul_1267: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1260, add_219);  mul_1260 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_168: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1267, [0, 2, 3])
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1267, mul_100, view_44, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1267 = mul_100 = view_44 = None
    getitem_300: "f32[4, 512, 24, 24]" = convolution_backward_62[0]
    getitem_301: "f32[256, 512, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_300: "f32[1, 256, 512]" = torch.ops.aten.view.default(getitem_301, [1, 256, 512]);  getitem_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_169: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_300, [0, 2])
    sub_235: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(view_42, unsqueeze_394);  view_42 = unsqueeze_394 = None
    mul_1268: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(view_300, sub_235)
    sum_170: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1268, [0, 2]);  mul_1268 = None
    mul_1269: "f32[256]" = torch.ops.aten.mul.Tensor(sum_169, 0.001953125);  sum_169 = None
    unsqueeze_395: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1269, 0);  mul_1269 = None
    unsqueeze_396: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    mul_1270: "f32[256]" = torch.ops.aten.mul.Tensor(sum_170, 0.001953125)
    mul_1271: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_29, squeeze_29)
    mul_1272: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1270, mul_1271);  mul_1270 = mul_1271 = None
    unsqueeze_397: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1272, 0);  mul_1272 = None
    unsqueeze_398: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    mul_1273: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_29, view_43);  view_43 = None
    unsqueeze_399: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1273, 0);  mul_1273 = None
    unsqueeze_400: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    mul_1274: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_398);  sub_235 = unsqueeze_398 = None
    sub_237: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(view_300, mul_1274);  view_300 = mul_1274 = None
    sub_238: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_396);  sub_237 = unsqueeze_396 = None
    mul_1275: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_400);  sub_238 = unsqueeze_400 = None
    mul_1276: "f32[256]" = torch.ops.aten.mul.Tensor(sum_170, squeeze_29);  sum_170 = squeeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_301: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1276, [256, 1, 1, 1]);  mul_1276 = None
    mul_1277: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_301, 0.04419417382415922);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_302: "f32[256, 512, 1, 1]" = torch.ops.aten.view.default(mul_1275, [256, 512, 1, 1]);  mul_1275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1278: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_300, 0.9805806756909201);  getitem_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1279: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1278, 1.7015043497085571);  mul_1278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1281: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
    mul_1282: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_26, add_26)
    mul_1283: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1282, -0.5);  mul_1282 = None
    exp_40: "f32[4, 512, 24, 24]" = torch.ops.aten.exp.default(mul_1283);  mul_1283 = None
    mul_1284: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
    mul_1285: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_26, mul_1284);  add_26 = mul_1284 = None
    add_221: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_1281, mul_1285);  mul_1281 = mul_1285 = None
    mul_1286: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1279, add_221);  mul_1279 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    add_222: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_1205, mul_1286);  mul_1205 = mul_1286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_1287: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_222, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_1288: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1287, clone_1);  clone_1 = None
    mul_1289: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1287, primals_44);  mul_1287 = primals_44 = None
    sum_171: "f32[]" = torch.ops.aten.sum.default(mul_1288);  mul_1288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_1290: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1289, 2.0);  mul_1289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_1291: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1290, convolution_15);  convolution_15 = None
    mul_1292: "f32[4, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1290, sigmoid_1);  mul_1290 = sigmoid_1 = None
    sum_172: "f32[4, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1291, [2, 3], True);  mul_1291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_64: "f32[4, 512, 1, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    sub_239: "f32[4, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_64)
    mul_1293: "f32[4, 512, 1, 1]" = torch.ops.aten.mul.Tensor(alias_64, sub_239);  alias_64 = sub_239 = None
    mul_1294: "f32[4, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_172, mul_1293);  sum_172 = mul_1293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_173: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1294, [0, 2, 3])
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1294, relu_1, primals_190, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1294 = primals_190 = None
    getitem_303: "f32[4, 256, 1, 1]" = convolution_backward_63[0]
    getitem_304: "f32[512, 256, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_66: "f32[4, 256, 1, 1]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_67: "f32[4, 256, 1, 1]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    le_10: "b8[4, 256, 1, 1]" = torch.ops.aten.le.Scalar(alias_67, 0);  alias_67 = None
    where_10: "f32[4, 256, 1, 1]" = torch.ops.aten.where.self(le_10, full_default, getitem_303);  le_10 = getitem_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_174: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(where_10, mean_1, primals_188, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = mean_1 = primals_188 = None
    getitem_306: "f32[4, 512, 1, 1]" = convolution_backward_64[0]
    getitem_307: "f32[256, 512, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[4, 512, 24, 24]" = torch.ops.aten.expand.default(getitem_306, [4, 512, 24, 24]);  getitem_306 = None
    div_11: "f32[4, 512, 24, 24]" = torch.ops.aten.div.Scalar(expand_11, 576);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_223: "f32[4, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_1292, div_11);  mul_1292 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_175: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_223, [0, 2, 3])
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(add_223, mul_88, view_41, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_223 = mul_88 = view_41 = None
    getitem_309: "f32[4, 256, 24, 24]" = convolution_backward_65[0]
    getitem_310: "f32[512, 256, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_303: "f32[1, 512, 256]" = torch.ops.aten.view.default(getitem_310, [1, 512, 256]);  getitem_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_176: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_303, [0, 2])
    sub_240: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_39, unsqueeze_402);  view_39 = unsqueeze_402 = None
    mul_1295: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_303, sub_240)
    sum_177: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1295, [0, 2]);  mul_1295 = None
    mul_1296: "f32[512]" = torch.ops.aten.mul.Tensor(sum_176, 0.00390625);  sum_176 = None
    unsqueeze_403: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1296, 0);  mul_1296 = None
    unsqueeze_404: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    mul_1297: "f32[512]" = torch.ops.aten.mul.Tensor(sum_177, 0.00390625)
    mul_1298: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_27, squeeze_27)
    mul_1299: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1297, mul_1298);  mul_1297 = mul_1298 = None
    unsqueeze_405: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1299, 0);  mul_1299 = None
    unsqueeze_406: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    mul_1300: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_27, view_40);  view_40 = None
    unsqueeze_407: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1300, 0);  mul_1300 = None
    unsqueeze_408: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    mul_1301: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_406);  sub_240 = unsqueeze_406 = None
    sub_242: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_303, mul_1301);  view_303 = mul_1301 = None
    sub_243: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_404);  sub_242 = unsqueeze_404 = None
    mul_1302: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_408);  sub_243 = unsqueeze_408 = None
    mul_1303: "f32[512]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_27);  sum_177 = squeeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_304: "f32[512, 1, 1, 1]" = torch.ops.aten.view.default(mul_1303, [512, 1, 1, 1]);  mul_1303 = None
    mul_1304: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_304, 0.0625);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_305: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_1302, [512, 256, 1, 1]);  mul_1302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1305: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_309, 1.7015043497085571);  getitem_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1307: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
    mul_1308: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_14, convolution_14)
    mul_1309: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1308, -0.5);  mul_1308 = None
    exp_41: "f32[4, 256, 24, 24]" = torch.ops.aten.exp.default(mul_1309);  mul_1309 = None
    mul_1310: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
    mul_1311: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_14, mul_1310);  convolution_14 = mul_1310 = None
    add_225: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(mul_1307, mul_1311);  mul_1307 = mul_1311 = None
    mul_1312: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1305, add_225);  mul_1305 = add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_178: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1312, [0, 2, 3])
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1312, mul_81, view_38, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1312 = mul_81 = view_38 = None
    getitem_312: "f32[4, 256, 24, 24]" = convolution_backward_66[0]
    getitem_313: "f32[256, 128, 3, 3]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_306: "f32[1, 256, 1152]" = torch.ops.aten.view.default(getitem_313, [1, 256, 1152]);  getitem_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_179: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_306, [0, 2])
    sub_244: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_36, unsqueeze_410);  view_36 = unsqueeze_410 = None
    mul_1313: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(view_306, sub_244)
    sum_180: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1313, [0, 2]);  mul_1313 = None
    mul_1314: "f32[256]" = torch.ops.aten.mul.Tensor(sum_179, 0.0008680555555555555);  sum_179 = None
    unsqueeze_411: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1314, 0);  mul_1314 = None
    unsqueeze_412: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    mul_1315: "f32[256]" = torch.ops.aten.mul.Tensor(sum_180, 0.0008680555555555555)
    mul_1316: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1317: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1315, mul_1316);  mul_1315 = mul_1316 = None
    unsqueeze_413: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1317, 0);  mul_1317 = None
    unsqueeze_414: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    mul_1318: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, view_37);  view_37 = None
    unsqueeze_415: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1318, 0);  mul_1318 = None
    unsqueeze_416: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    mul_1319: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_414);  sub_244 = unsqueeze_414 = None
    sub_246: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_306, mul_1319);  view_306 = mul_1319 = None
    sub_247: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_412);  sub_246 = unsqueeze_412 = None
    mul_1320: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_416);  sub_247 = unsqueeze_416 = None
    mul_1321: "f32[256]" = torch.ops.aten.mul.Tensor(sum_180, squeeze_25);  sum_180 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_307: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1321, [256, 1, 1, 1]);  mul_1321 = None
    mul_1322: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_307, 0.02946278254943948);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_308: "f32[256, 128, 3, 3]" = torch.ops.aten.view.default(mul_1320, [256, 128, 3, 3]);  mul_1320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1323: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_312, 1.7015043497085571);  getitem_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1325: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(add_22, 0.5);  add_22 = None
    mul_1326: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_13, convolution_13)
    mul_1327: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1326, -0.5);  mul_1326 = None
    exp_42: "f32[4, 256, 24, 24]" = torch.ops.aten.exp.default(mul_1327);  mul_1327 = None
    mul_1328: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
    mul_1329: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_13, mul_1328);  convolution_13 = mul_1328 = None
    add_227: "f32[4, 256, 24, 24]" = torch.ops.aten.add.Tensor(mul_1325, mul_1329);  mul_1325 = mul_1329 = None
    mul_1330: "f32[4, 256, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1323, add_227);  mul_1323 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_181: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1330, [0, 2, 3])
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1330, constant_pad_nd_2, view_35, [256], [2, 2], [0, 0], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1330 = constant_pad_nd_2 = view_35 = None
    getitem_315: "f32[4, 256, 49, 49]" = convolution_backward_67[0]
    getitem_316: "f32[256, 128, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_309: "f32[1, 256, 1152]" = torch.ops.aten.view.default(getitem_316, [1, 256, 1152]);  getitem_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_182: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_309, [0, 2])
    sub_248: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_33, unsqueeze_418);  view_33 = unsqueeze_418 = None
    mul_1331: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(view_309, sub_248)
    sum_183: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1331, [0, 2]);  mul_1331 = None
    mul_1332: "f32[256]" = torch.ops.aten.mul.Tensor(sum_182, 0.0008680555555555555);  sum_182 = None
    unsqueeze_419: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1332, 0);  mul_1332 = None
    unsqueeze_420: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    mul_1333: "f32[256]" = torch.ops.aten.mul.Tensor(sum_183, 0.0008680555555555555)
    mul_1334: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, squeeze_23)
    mul_1335: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1333, mul_1334);  mul_1333 = mul_1334 = None
    unsqueeze_421: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1335, 0);  mul_1335 = None
    unsqueeze_422: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    mul_1336: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, view_34);  view_34 = None
    unsqueeze_423: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1336, 0);  mul_1336 = None
    unsqueeze_424: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    mul_1337: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_422);  sub_248 = unsqueeze_422 = None
    sub_250: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(view_309, mul_1337);  view_309 = mul_1337 = None
    sub_251: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_420);  sub_250 = unsqueeze_420 = None
    mul_1338: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_424);  sub_251 = unsqueeze_424 = None
    mul_1339: "f32[256]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_23);  sum_183 = squeeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_310: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1339, [256, 1, 1, 1]);  mul_1339 = None
    mul_1340: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_310, 0.02946278254943948);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_311: "f32[256, 128, 3, 3]" = torch.ops.aten.view.default(mul_1338, [256, 128, 3, 3]);  mul_1338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_7: "f32[4, 256, 48, 48]" = torch.ops.aten.constant_pad_nd.default(getitem_315, [0, -1, 0, -1]);  getitem_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1341: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(constant_pad_nd_7, 1.7015043497085571);  constant_pad_nd_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1343: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_20, 0.5);  add_20 = None
    mul_1344: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_12, convolution_12)
    mul_1345: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1344, -0.5);  mul_1344 = None
    exp_43: "f32[4, 256, 48, 48]" = torch.ops.aten.exp.default(mul_1345);  mul_1345 = None
    mul_1346: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
    mul_1347: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_12, mul_1346);  convolution_12 = mul_1346 = None
    add_229: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul_1343, mul_1347);  mul_1343 = mul_1347 = None
    mul_1348: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1341, add_229);  mul_1341 = add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_184: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1348, [0, 2, 3])
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1348, mul_64, view_32, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1348 = view_32 = None
    getitem_318: "f32[4, 256, 48, 48]" = convolution_backward_68[0]
    getitem_319: "f32[256, 256, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_312: "f32[1, 256, 256]" = torch.ops.aten.view.default(getitem_319, [1, 256, 256]);  getitem_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_185: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_312, [0, 2])
    sub_252: "f32[1, 256, 256]" = torch.ops.aten.sub.Tensor(view_30, unsqueeze_426);  view_30 = unsqueeze_426 = None
    mul_1349: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(view_312, sub_252)
    sum_186: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1349, [0, 2]);  mul_1349 = None
    mul_1350: "f32[256]" = torch.ops.aten.mul.Tensor(sum_185, 0.00390625);  sum_185 = None
    unsqueeze_427: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1350, 0);  mul_1350 = None
    unsqueeze_428: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    mul_1351: "f32[256]" = torch.ops.aten.mul.Tensor(sum_186, 0.00390625)
    mul_1352: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, squeeze_21)
    mul_1353: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1351, mul_1352);  mul_1351 = mul_1352 = None
    unsqueeze_429: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1353, 0);  mul_1353 = None
    unsqueeze_430: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    mul_1354: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, view_31);  view_31 = None
    unsqueeze_431: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1354, 0);  mul_1354 = None
    unsqueeze_432: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    mul_1355: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_430);  sub_252 = unsqueeze_430 = None
    sub_254: "f32[1, 256, 256]" = torch.ops.aten.sub.Tensor(view_312, mul_1355);  view_312 = mul_1355 = None
    sub_255: "f32[1, 256, 256]" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_428);  sub_254 = unsqueeze_428 = None
    mul_1356: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_432);  sub_255 = unsqueeze_432 = None
    mul_1357: "f32[256]" = torch.ops.aten.mul.Tensor(sum_186, squeeze_21);  sum_186 = squeeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_313: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1357, [256, 1, 1, 1]);  mul_1357 = None
    mul_1358: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_313, 0.0625);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_314: "f32[256, 256, 1, 1]" = torch.ops.aten.view.default(mul_1356, [256, 256, 1, 1]);  mul_1356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_187: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_222, [0, 2, 3])
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(add_222, avg_pool2d, view_29, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_222 = avg_pool2d = view_29 = None
    getitem_321: "f32[4, 256, 24, 24]" = convolution_backward_69[0]
    getitem_322: "f32[512, 256, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_315: "f32[1, 512, 256]" = torch.ops.aten.view.default(getitem_322, [1, 512, 256]);  getitem_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_188: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_315, [0, 2])
    sub_256: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_27, unsqueeze_434);  view_27 = unsqueeze_434 = None
    mul_1359: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_315, sub_256)
    sum_189: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1359, [0, 2]);  mul_1359 = None
    mul_1360: "f32[512]" = torch.ops.aten.mul.Tensor(sum_188, 0.00390625);  sum_188 = None
    unsqueeze_435: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1360, 0);  mul_1360 = None
    unsqueeze_436: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
    mul_1361: "f32[512]" = torch.ops.aten.mul.Tensor(sum_189, 0.00390625)
    mul_1362: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1363: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1361, mul_1362);  mul_1361 = mul_1362 = None
    unsqueeze_437: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1363, 0);  mul_1363 = None
    unsqueeze_438: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    mul_1364: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, view_28);  view_28 = None
    unsqueeze_439: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1364, 0);  mul_1364 = None
    unsqueeze_440: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    mul_1365: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_438);  sub_256 = unsqueeze_438 = None
    sub_258: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_315, mul_1365);  view_315 = mul_1365 = None
    sub_259: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_436);  sub_258 = unsqueeze_436 = None
    mul_1366: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_440);  sub_259 = unsqueeze_440 = None
    mul_1367: "f32[512]" = torch.ops.aten.mul.Tensor(sum_189, squeeze_19);  sum_189 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_316: "f32[512, 1, 1, 1]" = torch.ops.aten.view.default(mul_1367, [512, 1, 1, 1]);  mul_1367 = None
    mul_1368: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_316, 0.0625);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_317: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_1366, [512, 256, 1, 1]);  mul_1366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_backward_2: "f32[4, 256, 48, 48]" = torch.ops.aten.avg_pool2d_backward.default(getitem_321, mul_64, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_321 = mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    add_230: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(getitem_318, avg_pool2d_backward_2);  getitem_318 = avg_pool2d_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1369: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_230, 0.9805806756909201);  add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1370: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1369, 1.7015043497085571);  mul_1369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1372: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_17, 0.5);  add_17 = None
    mul_1373: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_16, add_16)
    mul_1374: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1373, -0.5);  mul_1373 = None
    exp_44: "f32[4, 256, 48, 48]" = torch.ops.aten.exp.default(mul_1374);  mul_1374 = None
    mul_1375: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
    mul_1376: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(add_16, mul_1375);  add_16 = mul_1375 = None
    add_232: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul_1372, mul_1376);  mul_1372 = mul_1376 = None
    mul_1377: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1370, add_232);  mul_1370 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_1378: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1377, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:198, code: out.mul_(self.skipinit_gain)
    mul_1379: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1378, clone);  clone = None
    mul_1380: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1378, primals_28);  mul_1378 = primals_28 = None
    sum_190: "f32[]" = torch.ops.aten.sum.default(mul_1379);  mul_1379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_1381: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1380, 2.0);  mul_1380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_1382: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1381, convolution_8);  convolution_8 = None
    mul_1383: "f32[4, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1381, sigmoid);  mul_1381 = sigmoid = None
    sum_191: "f32[4, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1382, [2, 3], True);  mul_1382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_68: "f32[4, 256, 1, 1]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    sub_260: "f32[4, 256, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_68)
    mul_1384: "f32[4, 256, 1, 1]" = torch.ops.aten.mul.Tensor(alias_68, sub_260);  alias_68 = sub_260 = None
    mul_1385: "f32[4, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sum_191, mul_1384);  sum_191 = mul_1384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_192: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1385, [0, 2, 3])
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1385, relu, primals_186, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1385 = primals_186 = None
    getitem_324: "f32[4, 128, 1, 1]" = convolution_backward_70[0]
    getitem_325: "f32[256, 128, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_70: "f32[4, 128, 1, 1]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_71: "f32[4, 128, 1, 1]" = torch.ops.aten.alias.default(alias_70);  alias_70 = None
    le_11: "b8[4, 128, 1, 1]" = torch.ops.aten.le.Scalar(alias_71, 0);  alias_71 = None
    where_11: "f32[4, 128, 1, 1]" = torch.ops.aten.where.self(le_11, full_default, getitem_324);  le_11 = full_default = getitem_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_193: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(where_11, mean, primals_184, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = mean = primals_184 = None
    getitem_327: "f32[4, 256, 1, 1]" = convolution_backward_71[0]
    getitem_328: "f32[128, 256, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[4, 256, 48, 48]" = torch.ops.aten.expand.default(getitem_327, [4, 256, 48, 48]);  getitem_327 = None
    div_12: "f32[4, 256, 48, 48]" = torch.ops.aten.div.Scalar(expand_12, 2304);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_233: "f32[4, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul_1383, div_12);  mul_1383 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_194: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 2, 3])
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(add_233, mul_52, view_26, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_233 = mul_52 = view_26 = None
    getitem_330: "f32[4, 128, 48, 48]" = convolution_backward_72[0]
    getitem_331: "f32[256, 128, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_318: "f32[1, 256, 128]" = torch.ops.aten.view.default(getitem_331, [1, 256, 128]);  getitem_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_195: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_318, [0, 2])
    sub_261: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_24, unsqueeze_442);  view_24 = unsqueeze_442 = None
    mul_1386: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(view_318, sub_261)
    sum_196: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1386, [0, 2]);  mul_1386 = None
    mul_1387: "f32[256]" = torch.ops.aten.mul.Tensor(sum_195, 0.0078125);  sum_195 = None
    unsqueeze_443: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1387, 0);  mul_1387 = None
    unsqueeze_444: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    mul_1388: "f32[256]" = torch.ops.aten.mul.Tensor(sum_196, 0.0078125)
    mul_1389: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, squeeze_17)
    mul_1390: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1388, mul_1389);  mul_1388 = mul_1389 = None
    unsqueeze_445: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1390, 0);  mul_1390 = None
    unsqueeze_446: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    mul_1391: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, view_25);  view_25 = None
    unsqueeze_447: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1391, 0);  mul_1391 = None
    unsqueeze_448: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
    mul_1392: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_446);  sub_261 = unsqueeze_446 = None
    sub_263: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_318, mul_1392);  view_318 = mul_1392 = None
    sub_264: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_444);  sub_263 = unsqueeze_444 = None
    mul_1393: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_448);  sub_264 = unsqueeze_448 = None
    mul_1394: "f32[256]" = torch.ops.aten.mul.Tensor(sum_196, squeeze_17);  sum_196 = squeeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_319: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1394, [256, 1, 1, 1]);  mul_1394 = None
    mul_1395: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_319, 0.08838834764831845);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_320: "f32[256, 128, 1, 1]" = torch.ops.aten.view.default(mul_1393, [256, 128, 1, 1]);  mul_1393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1396: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(getitem_330, 1.7015043497085571);  getitem_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1398: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_14, 0.5);  add_14 = None
    mul_1399: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_7, convolution_7)
    mul_1400: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1399, -0.5);  mul_1399 = None
    exp_45: "f32[4, 128, 48, 48]" = torch.ops.aten.exp.default(mul_1400);  mul_1400 = None
    mul_1401: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
    mul_1402: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_7, mul_1401);  convolution_7 = mul_1401 = None
    add_235: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(mul_1398, mul_1402);  mul_1398 = mul_1402 = None
    mul_1403: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1396, add_235);  mul_1396 = add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_197: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1403, [0, 2, 3])
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1403, mul_45, view_23, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1403 = mul_45 = view_23 = None
    getitem_333: "f32[4, 128, 48, 48]" = convolution_backward_73[0]
    getitem_334: "f32[128, 128, 3, 3]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_321: "f32[1, 128, 1152]" = torch.ops.aten.view.default(getitem_334, [1, 128, 1152]);  getitem_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_198: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_321, [0, 2])
    sub_265: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_21, unsqueeze_450);  view_21 = unsqueeze_450 = None
    mul_1404: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(view_321, sub_265)
    sum_199: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1404, [0, 2]);  mul_1404 = None
    mul_1405: "f32[128]" = torch.ops.aten.mul.Tensor(sum_198, 0.0008680555555555555);  sum_198 = None
    unsqueeze_451: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1405, 0);  mul_1405 = None
    unsqueeze_452: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    mul_1406: "f32[128]" = torch.ops.aten.mul.Tensor(sum_199, 0.0008680555555555555)
    mul_1407: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_15, squeeze_15)
    mul_1408: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1406, mul_1407);  mul_1406 = mul_1407 = None
    unsqueeze_453: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1408, 0);  mul_1408 = None
    unsqueeze_454: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
    mul_1409: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_15, view_22);  view_22 = None
    unsqueeze_455: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1409, 0);  mul_1409 = None
    unsqueeze_456: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    mul_1410: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_454);  sub_265 = unsqueeze_454 = None
    sub_267: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_321, mul_1410);  view_321 = mul_1410 = None
    sub_268: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_452);  sub_267 = unsqueeze_452 = None
    mul_1411: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_456);  sub_268 = unsqueeze_456 = None
    mul_1412: "f32[128]" = torch.ops.aten.mul.Tensor(sum_199, squeeze_15);  sum_199 = squeeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_322: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_1412, [128, 1, 1, 1]);  mul_1412 = None
    mul_1413: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_322, 0.02946278254943948);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_323: "f32[128, 128, 3, 3]" = torch.ops.aten.view.default(mul_1411, [128, 128, 3, 3]);  mul_1411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1414: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(getitem_333, 1.7015043497085571);  getitem_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1416: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_12, 0.5);  add_12 = None
    mul_1417: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_6, convolution_6)
    mul_1418: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1417, -0.5);  mul_1417 = None
    exp_46: "f32[4, 128, 48, 48]" = torch.ops.aten.exp.default(mul_1418);  mul_1418 = None
    mul_1419: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
    mul_1420: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_6, mul_1419);  convolution_6 = mul_1419 = None
    add_237: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(mul_1416, mul_1420);  mul_1416 = mul_1420 = None
    mul_1421: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1414, add_237);  mul_1414 = add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_200: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1421, [0, 2, 3])
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1421, mul_38, view_20, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1421 = mul_38 = view_20 = None
    getitem_336: "f32[4, 128, 48, 48]" = convolution_backward_74[0]
    getitem_337: "f32[128, 128, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_324: "f32[1, 128, 1152]" = torch.ops.aten.view.default(getitem_337, [1, 128, 1152]);  getitem_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_201: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_324, [0, 2])
    sub_269: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_18, unsqueeze_458);  view_18 = unsqueeze_458 = None
    mul_1422: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(view_324, sub_269)
    sum_202: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1422, [0, 2]);  mul_1422 = None
    mul_1423: "f32[128]" = torch.ops.aten.mul.Tensor(sum_201, 0.0008680555555555555);  sum_201 = None
    unsqueeze_459: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1423, 0);  mul_1423 = None
    unsqueeze_460: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
    mul_1424: "f32[128]" = torch.ops.aten.mul.Tensor(sum_202, 0.0008680555555555555)
    mul_1425: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1426: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1424, mul_1425);  mul_1424 = mul_1425 = None
    unsqueeze_461: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1426, 0);  mul_1426 = None
    unsqueeze_462: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    mul_1427: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, view_19);  view_19 = None
    unsqueeze_463: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1427, 0);  mul_1427 = None
    unsqueeze_464: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    mul_1428: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_462);  sub_269 = unsqueeze_462 = None
    sub_271: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(view_324, mul_1428);  view_324 = mul_1428 = None
    sub_272: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_460);  sub_271 = unsqueeze_460 = None
    mul_1429: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_464);  sub_272 = unsqueeze_464 = None
    mul_1430: "f32[128]" = torch.ops.aten.mul.Tensor(sum_202, squeeze_13);  sum_202 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_325: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_1430, [128, 1, 1, 1]);  mul_1430 = None
    mul_1431: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_325, 0.02946278254943948);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_326: "f32[128, 128, 3, 3]" = torch.ops.aten.view.default(mul_1429, [128, 128, 3, 3]);  mul_1429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1432: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(getitem_336, 1.7015043497085571);  getitem_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1434: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
    mul_1435: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_5, convolution_5)
    mul_1436: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1435, -0.5);  mul_1435 = None
    exp_47: "f32[4, 128, 48, 48]" = torch.ops.aten.exp.default(mul_1436);  mul_1436 = None
    mul_1437: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
    mul_1438: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_5, mul_1437);  convolution_5 = mul_1437 = None
    add_239: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(mul_1434, mul_1438);  mul_1434 = mul_1438 = None
    mul_1439: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1432, add_239);  mul_1432 = add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_203: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1439, [0, 2, 3])
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1439, mul_28, view_17, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1439 = view_17 = None
    getitem_339: "f32[4, 128, 48, 48]" = convolution_backward_75[0]
    getitem_340: "f32[128, 128, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_327: "f32[1, 128, 128]" = torch.ops.aten.view.default(getitem_340, [1, 128, 128]);  getitem_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_204: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_327, [0, 2])
    sub_273: "f32[1, 128, 128]" = torch.ops.aten.sub.Tensor(view_15, unsqueeze_466);  view_15 = unsqueeze_466 = None
    mul_1440: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_327, sub_273)
    sum_205: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1440, [0, 2]);  mul_1440 = None
    mul_1441: "f32[128]" = torch.ops.aten.mul.Tensor(sum_204, 0.0078125);  sum_204 = None
    unsqueeze_467: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1441, 0);  mul_1441 = None
    unsqueeze_468: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    mul_1442: "f32[128]" = torch.ops.aten.mul.Tensor(sum_205, 0.0078125)
    mul_1443: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_11, squeeze_11)
    mul_1444: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1442, mul_1443);  mul_1442 = mul_1443 = None
    unsqueeze_469: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1444, 0);  mul_1444 = None
    unsqueeze_470: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    mul_1445: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_11, view_16);  view_16 = None
    unsqueeze_471: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1445, 0);  mul_1445 = None
    unsqueeze_472: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
    mul_1446: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_470);  sub_273 = unsqueeze_470 = None
    sub_275: "f32[1, 128, 128]" = torch.ops.aten.sub.Tensor(view_327, mul_1446);  view_327 = mul_1446 = None
    sub_276: "f32[1, 128, 128]" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_468);  sub_275 = unsqueeze_468 = None
    mul_1447: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_472);  sub_276 = unsqueeze_472 = None
    mul_1448: "f32[128]" = torch.ops.aten.mul.Tensor(sum_205, squeeze_11);  sum_205 = squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_328: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_1448, [128, 1, 1, 1]);  mul_1448 = None
    mul_1449: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_328, 0.08838834764831845);  view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_329: "f32[128, 128, 1, 1]" = torch.ops.aten.view.default(mul_1447, [128, 128, 1, 1]);  mul_1447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_206: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1377, [0, 2, 3])
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1377, mul_28, view_14, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1377 = mul_28 = view_14 = None
    getitem_342: "f32[4, 128, 48, 48]" = convolution_backward_76[0]
    getitem_343: "f32[256, 128, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    add_240: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(getitem_339, getitem_342);  getitem_339 = getitem_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_330: "f32[1, 256, 128]" = torch.ops.aten.view.default(getitem_343, [1, 256, 128]);  getitem_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_207: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_330, [0, 2])
    sub_277: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_12, unsqueeze_474);  view_12 = unsqueeze_474 = None
    mul_1450: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(view_330, sub_277)
    sum_208: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1450, [0, 2]);  mul_1450 = None
    mul_1451: "f32[256]" = torch.ops.aten.mul.Tensor(sum_207, 0.0078125);  sum_207 = None
    unsqueeze_475: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1451, 0);  mul_1451 = None
    unsqueeze_476: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    mul_1452: "f32[256]" = torch.ops.aten.mul.Tensor(sum_208, 0.0078125)
    mul_1453: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_9, squeeze_9)
    mul_1454: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1452, mul_1453);  mul_1452 = mul_1453 = None
    unsqueeze_477: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1454, 0);  mul_1454 = None
    unsqueeze_478: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
    mul_1455: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_9, view_13);  view_13 = None
    unsqueeze_479: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1455, 0);  mul_1455 = None
    unsqueeze_480: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    mul_1456: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_478);  sub_277 = unsqueeze_478 = None
    sub_279: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_330, mul_1456);  view_330 = mul_1456 = None
    sub_280: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_476);  sub_279 = unsqueeze_476 = None
    mul_1457: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_480);  sub_280 = unsqueeze_480 = None
    mul_1458: "f32[256]" = torch.ops.aten.mul.Tensor(sum_208, squeeze_9);  sum_208 = squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_331: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_1458, [256, 1, 1, 1]);  mul_1458 = None
    mul_1459: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_331, 0.08838834764831845);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_332: "f32[256, 128, 1, 1]" = torch.ops.aten.view.default(mul_1457, [256, 128, 1, 1]);  mul_1457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1460: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_240, 1.0);  add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1461: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1460, 1.7015043497085571);  mul_1460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1463: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
    mul_1464: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_3, convolution_3)
    mul_1465: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1464, -0.5);  mul_1464 = None
    exp_48: "f32[4, 128, 48, 48]" = torch.ops.aten.exp.default(mul_1465);  mul_1465 = None
    mul_1466: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
    mul_1467: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_3, mul_1466);  convolution_3 = mul_1466 = None
    add_242: "f32[4, 128, 48, 48]" = torch.ops.aten.add.Tensor(mul_1463, mul_1467);  mul_1463 = mul_1467 = None
    mul_1468: "f32[4, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1461, add_242);  mul_1461 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_209: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1468, [0, 2, 3])
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1468, constant_pad_nd_1, view_11, [128], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1468 = constant_pad_nd_1 = view_11 = None
    getitem_345: "f32[4, 64, 97, 97]" = convolution_backward_77[0]
    getitem_346: "f32[128, 64, 3, 3]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_333: "f32[1, 128, 576]" = torch.ops.aten.view.default(getitem_346, [1, 128, 576]);  getitem_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_210: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_333, [0, 2])
    sub_281: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_9, unsqueeze_482);  view_9 = unsqueeze_482 = None
    mul_1469: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(view_333, sub_281)
    sum_211: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1469, [0, 2]);  mul_1469 = None
    mul_1470: "f32[128]" = torch.ops.aten.mul.Tensor(sum_210, 0.001736111111111111);  sum_210 = None
    unsqueeze_483: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1470, 0);  mul_1470 = None
    unsqueeze_484: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
    mul_1471: "f32[128]" = torch.ops.aten.mul.Tensor(sum_211, 0.001736111111111111)
    mul_1472: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1473: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1471, mul_1472);  mul_1471 = mul_1472 = None
    unsqueeze_485: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1473, 0);  mul_1473 = None
    unsqueeze_486: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    mul_1474: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, view_10);  view_10 = None
    unsqueeze_487: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1474, 0);  mul_1474 = None
    unsqueeze_488: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    mul_1475: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_486);  sub_281 = unsqueeze_486 = None
    sub_283: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_333, mul_1475);  view_333 = mul_1475 = None
    sub_284: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_484);  sub_283 = unsqueeze_484 = None
    mul_1476: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_488);  sub_284 = unsqueeze_488 = None
    mul_1477: "f32[128]" = torch.ops.aten.mul.Tensor(sum_211, squeeze_7);  sum_211 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_334: "f32[128, 1, 1, 1]" = torch.ops.aten.view.default(mul_1477, [128, 1, 1, 1]);  mul_1477 = None
    mul_1478: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_334, 0.041666666666666664);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_335: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_1476, [128, 64, 3, 3]);  mul_1476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_8: "f32[4, 64, 96, 96]" = torch.ops.aten.constant_pad_nd.default(getitem_345, [0, -1, 0, -1]);  getitem_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1479: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(constant_pad_nd_8, 1.7015043497085571);  constant_pad_nd_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1481: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(add_5, 0.5);  add_5 = None
    mul_1482: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_2, convolution_2)
    mul_1483: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1482, -0.5);  mul_1482 = None
    exp_49: "f32[4, 64, 96, 96]" = torch.ops.aten.exp.default(mul_1483);  mul_1483 = None
    mul_1484: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
    mul_1485: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_2, mul_1484);  convolution_2 = mul_1484 = None
    add_244: "f32[4, 64, 96, 96]" = torch.ops.aten.add.Tensor(mul_1481, mul_1485);  mul_1481 = mul_1485 = None
    mul_1486: "f32[4, 64, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1479, add_244);  mul_1479 = add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_212: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1486, [0, 2, 3])
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1486, mul_13, view_8, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1486 = mul_13 = view_8 = None
    getitem_348: "f32[4, 32, 96, 96]" = convolution_backward_78[0]
    getitem_349: "f32[64, 32, 3, 3]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_336: "f32[1, 64, 288]" = torch.ops.aten.view.default(getitem_349, [1, 64, 288]);  getitem_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_213: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_336, [0, 2])
    sub_285: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_6, unsqueeze_490);  view_6 = unsqueeze_490 = None
    mul_1487: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(view_336, sub_285)
    sum_214: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1487, [0, 2]);  mul_1487 = None
    mul_1488: "f32[64]" = torch.ops.aten.mul.Tensor(sum_213, 0.003472222222222222);  sum_213 = None
    unsqueeze_491: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1488, 0);  mul_1488 = None
    unsqueeze_492: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    mul_1489: "f32[64]" = torch.ops.aten.mul.Tensor(sum_214, 0.003472222222222222)
    mul_1490: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, squeeze_5)
    mul_1491: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1489, mul_1490);  mul_1489 = mul_1490 = None
    unsqueeze_493: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1491, 0);  mul_1491 = None
    unsqueeze_494: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    mul_1492: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, view_7);  view_7 = None
    unsqueeze_495: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1492, 0);  mul_1492 = None
    unsqueeze_496: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
    mul_1493: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_494);  sub_285 = unsqueeze_494 = None
    sub_287: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_336, mul_1493);  view_336 = mul_1493 = None
    sub_288: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_492);  sub_287 = unsqueeze_492 = None
    mul_1494: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_496);  sub_288 = unsqueeze_496 = None
    mul_1495: "f32[64]" = torch.ops.aten.mul.Tensor(sum_214, squeeze_5);  sum_214 = squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_337: "f32[64, 1, 1, 1]" = torch.ops.aten.view.default(mul_1495, [64, 1, 1, 1]);  mul_1495 = None
    mul_1496: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_337, 0.05892556509887896);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_338: "f32[64, 32, 3, 3]" = torch.ops.aten.view.default(mul_1494, [64, 32, 3, 3]);  mul_1494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1497: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(getitem_348, 1.7015043497085571);  getitem_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1499: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
    mul_1500: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_1, convolution_1)
    mul_1501: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1500, -0.5);  mul_1500 = None
    exp_50: "f32[4, 32, 96, 96]" = torch.ops.aten.exp.default(mul_1501);  mul_1501 = None
    mul_1502: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
    mul_1503: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_1, mul_1502);  convolution_1 = mul_1502 = None
    add_246: "f32[4, 32, 96, 96]" = torch.ops.aten.add.Tensor(mul_1499, mul_1503);  mul_1499 = mul_1503 = None
    mul_1504: "f32[4, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1497, add_246);  mul_1497 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_215: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1504, [0, 2, 3])
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1504, mul_6, view_5, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1504 = mul_6 = view_5 = None
    getitem_351: "f32[4, 16, 96, 96]" = convolution_backward_79[0]
    getitem_352: "f32[32, 16, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_339: "f32[1, 32, 144]" = torch.ops.aten.view.default(getitem_352, [1, 32, 144]);  getitem_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_216: "f32[32]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 2])
    sub_289: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_3, unsqueeze_498);  view_3 = unsqueeze_498 = None
    mul_1505: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(view_339, sub_289)
    sum_217: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1505, [0, 2]);  mul_1505 = None
    mul_1506: "f32[32]" = torch.ops.aten.mul.Tensor(sum_216, 0.006944444444444444);  sum_216 = None
    unsqueeze_499: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1506, 0);  mul_1506 = None
    unsqueeze_500: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    mul_1507: "f32[32]" = torch.ops.aten.mul.Tensor(sum_217, 0.006944444444444444)
    mul_1508: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, squeeze_3)
    mul_1509: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1507, mul_1508);  mul_1507 = mul_1508 = None
    unsqueeze_501: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1509, 0);  mul_1509 = None
    unsqueeze_502: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
    mul_1510: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, view_4);  view_4 = None
    unsqueeze_503: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1510, 0);  mul_1510 = None
    unsqueeze_504: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    mul_1511: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_502);  sub_289 = unsqueeze_502 = None
    sub_291: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_339, mul_1511);  view_339 = mul_1511 = None
    sub_292: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_500);  sub_291 = unsqueeze_500 = None
    mul_1512: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_504);  sub_292 = unsqueeze_504 = None
    mul_1513: "f32[32]" = torch.ops.aten.mul.Tensor(sum_217, squeeze_3);  sum_217 = squeeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_340: "f32[32, 1, 1, 1]" = torch.ops.aten.view.default(mul_1513, [32, 1, 1, 1]);  mul_1513 = None
    mul_1514: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_340, 0.08333333333333333);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_341: "f32[32, 16, 3, 3]" = torch.ops.aten.view.default(mul_1512, [32, 16, 3, 3]);  mul_1512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:71, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
    mul_1515: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(getitem_351, 1.7015043497085571);  getitem_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:135, code: return F.gelu(x)
    mul_1517: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
    mul_1518: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(convolution, convolution)
    mul_1519: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1518, -0.5);  mul_1518 = None
    exp_51: "f32[4, 16, 96, 96]" = torch.ops.aten.exp.default(mul_1519);  mul_1519 = None
    mul_1520: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(exp_51, 0.3989422804014327);  exp_51 = None
    mul_1521: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(convolution, mul_1520);  convolution = mul_1520 = None
    add_248: "f32[4, 16, 96, 96]" = torch.ops.aten.add.Tensor(mul_1517, mul_1521);  mul_1517 = mul_1521 = None
    mul_1522: "f32[4, 16, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1515, add_248);  mul_1515 = add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:133, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_218: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1522, [0, 2, 3])
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1522, constant_pad_nd, view_2, [16], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1522 = constant_pad_nd = view_2 = None
    getitem_355: "f32[16, 3, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:132, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_342: "f32[1, 16, 27]" = torch.ops.aten.view.default(getitem_355, [1, 16, 27]);  getitem_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:129, code: weight = F.batch_norm(
    sum_219: "f32[16]" = torch.ops.aten.sum.dim_IntList(view_342, [0, 2])
    sub_293: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view, unsqueeze_506);  view = unsqueeze_506 = None
    mul_1523: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(view_342, sub_293)
    sum_220: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1523, [0, 2]);  mul_1523 = None
    mul_1524: "f32[16]" = torch.ops.aten.mul.Tensor(sum_219, 0.037037037037037035);  sum_219 = None
    unsqueeze_507: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1524, 0);  mul_1524 = None
    unsqueeze_508: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
    mul_1525: "f32[16]" = torch.ops.aten.mul.Tensor(sum_220, 0.037037037037037035)
    mul_1526: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1527: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1525, mul_1526);  mul_1525 = mul_1526 = None
    unsqueeze_509: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1527, 0);  mul_1527 = None
    unsqueeze_510: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    mul_1528: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, view_1);  view_1 = None
    unsqueeze_511: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1528, 0);  mul_1528 = None
    unsqueeze_512: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    mul_1529: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_510);  sub_293 = unsqueeze_510 = None
    sub_295: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view_342, mul_1529);  view_342 = mul_1529 = None
    sub_296: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_508);  sub_295 = unsqueeze_508 = None
    mul_1530: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_512);  sub_296 = unsqueeze_512 = None
    mul_1531: "f32[16]" = torch.ops.aten.mul.Tensor(sum_220, squeeze_1);  sum_220 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:131, code: weight=(self.gain * self.scale).view(-1),
    view_343: "f32[16, 1, 1, 1]" = torch.ops.aten.view.default(mul_1531, [16, 1, 1, 1]);  mul_1531 = None
    mul_1532: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_343, 0.19245008972987526);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:130, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_344: "f32[16, 3, 3, 3]" = torch.ops.aten.view.default(mul_1530, [16, 3, 3, 3]);  mul_1530 = None
    return [view_344, mul_1532, sum_218, view_341, mul_1514, sum_215, view_338, mul_1496, sum_212, view_335, mul_1478, sum_209, view_332, mul_1459, sum_206, view_329, mul_1449, sum_203, view_326, mul_1431, sum_200, view_323, mul_1413, sum_197, view_320, mul_1395, sum_194, sum_190, view_317, mul_1368, sum_187, view_314, mul_1358, sum_184, view_311, mul_1340, sum_181, view_308, mul_1322, sum_178, view_305, mul_1304, sum_175, sum_171, view_302, mul_1277, sum_168, view_299, mul_1259, sum_165, view_296, mul_1241, sum_162, view_293, mul_1223, sum_159, sum_155, view_290, mul_1196, sum_152, view_287, mul_1186, sum_149, view_284, mul_1168, sum_146, view_281, mul_1150, sum_143, view_278, mul_1132, sum_140, sum_136, view_275, mul_1105, sum_133, view_272, mul_1087, sum_130, view_269, mul_1069, sum_127, view_266, mul_1051, sum_124, sum_120, view_263, mul_1024, sum_117, view_260, mul_1006, sum_114, view_257, mul_988, sum_111, view_254, mul_970, sum_108, sum_104, view_251, mul_943, sum_101, view_248, mul_925, sum_98, view_245, mul_907, sum_95, view_242, mul_889, sum_92, sum_88, view_239, mul_862, sum_85, view_236, mul_844, sum_82, view_233, mul_826, sum_79, view_230, mul_808, sum_76, sum_72, view_227, mul_781, sum_69, view_224, mul_763, sum_66, view_221, mul_745, sum_63, view_218, mul_727, sum_60, sum_56, view_215, mul_700, sum_53, view_212, mul_690, sum_50, view_209, mul_672, sum_47, view_206, mul_654, sum_44, view_203, mul_636, sum_41, sum_37, view_200, mul_609, sum_34, view_197, mul_591, sum_31, view_194, mul_573, sum_28, view_191, mul_555, sum_25, sum_21, view_188, mul_528, sum_18, view_185, mul_510, sum_15, view_182, mul_492, sum_12, view_179, mul_474, sum_9, sum_5, view_176, mul_456, sum_2, getitem_328, sum_193, getitem_325, sum_192, getitem_307, sum_174, getitem_304, sum_173, getitem_289, sum_158, getitem_286, sum_157, getitem_268, sum_139, getitem_265, sum_138, getitem_250, sum_123, getitem_247, sum_122, getitem_232, sum_107, getitem_229, sum_106, getitem_214, sum_91, getitem_211, sum_90, getitem_196, sum_75, getitem_193, sum_74, getitem_178, sum_59, getitem_175, sum_58, getitem_157, sum_40, getitem_154, sum_39, getitem_139, sum_24, getitem_136, sum_23, getitem_121, sum_8, getitem_118, sum_7, permute_4, view_172, None]
    