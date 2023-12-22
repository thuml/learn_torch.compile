from __future__ import annotations



def forward(self, primals_1: "f32[16, 3, 3, 3]", primals_2: "f32[16, 1, 1, 1]", primals_4: "f32[32, 16, 3, 3]", primals_5: "f32[32, 1, 1, 1]", primals_7: "f32[64, 32, 3, 3]", primals_8: "f32[64, 1, 1, 1]", primals_10: "f32[128, 64, 3, 3]", primals_11: "f32[128, 1, 1, 1]", primals_13: "f32[256, 128, 1, 1]", primals_14: "f32[256, 1, 1, 1]", primals_16: "f32[64, 128, 1, 1]", primals_17: "f32[64, 1, 1, 1]", primals_19: "f32[64, 64, 3, 3]", primals_20: "f32[64, 1, 1, 1]", primals_22: "f32[64, 64, 3, 3]", primals_23: "f32[64, 1, 1, 1]", primals_25: "f32[256, 64, 1, 1]", primals_26: "f32[256, 1, 1, 1]", primals_28: "f32[512, 256, 1, 1]", primals_29: "f32[512, 1, 1, 1]", primals_31: "f32[128, 256, 1, 1]", primals_32: "f32[128, 1, 1, 1]", primals_34: "f32[128, 64, 3, 3]", primals_35: "f32[128, 1, 1, 1]", primals_37: "f32[128, 64, 3, 3]", primals_38: "f32[128, 1, 1, 1]", primals_40: "f32[512, 128, 1, 1]", primals_41: "f32[512, 1, 1, 1]", primals_43: "f32[128, 512, 1, 1]", primals_44: "f32[128, 1, 1, 1]", primals_46: "f32[128, 64, 3, 3]", primals_47: "f32[128, 1, 1, 1]", primals_49: "f32[128, 64, 3, 3]", primals_50: "f32[128, 1, 1, 1]", primals_52: "f32[512, 128, 1, 1]", primals_53: "f32[512, 1, 1, 1]", primals_55: "f32[1536, 512, 1, 1]", primals_56: "f32[1536, 1, 1, 1]", primals_58: "f32[384, 512, 1, 1]", primals_59: "f32[384, 1, 1, 1]", primals_61: "f32[384, 64, 3, 3]", primals_62: "f32[384, 1, 1, 1]", primals_64: "f32[384, 64, 3, 3]", primals_65: "f32[384, 1, 1, 1]", primals_67: "f32[1536, 384, 1, 1]", primals_68: "f32[1536, 1, 1, 1]", primals_70: "f32[384, 1536, 1, 1]", primals_71: "f32[384, 1, 1, 1]", primals_73: "f32[384, 64, 3, 3]", primals_74: "f32[384, 1, 1, 1]", primals_76: "f32[384, 64, 3, 3]", primals_77: "f32[384, 1, 1, 1]", primals_79: "f32[1536, 384, 1, 1]", primals_80: "f32[1536, 1, 1, 1]", primals_82: "f32[384, 1536, 1, 1]", primals_83: "f32[384, 1, 1, 1]", primals_85: "f32[384, 64, 3, 3]", primals_86: "f32[384, 1, 1, 1]", primals_88: "f32[384, 64, 3, 3]", primals_89: "f32[384, 1, 1, 1]", primals_91: "f32[1536, 384, 1, 1]", primals_92: "f32[1536, 1, 1, 1]", primals_94: "f32[384, 1536, 1, 1]", primals_95: "f32[384, 1, 1, 1]", primals_97: "f32[384, 64, 3, 3]", primals_98: "f32[384, 1, 1, 1]", primals_100: "f32[384, 64, 3, 3]", primals_101: "f32[384, 1, 1, 1]", primals_103: "f32[1536, 384, 1, 1]", primals_104: "f32[1536, 1, 1, 1]", primals_106: "f32[384, 1536, 1, 1]", primals_107: "f32[384, 1, 1, 1]", primals_109: "f32[384, 64, 3, 3]", primals_110: "f32[384, 1, 1, 1]", primals_112: "f32[384, 64, 3, 3]", primals_113: "f32[384, 1, 1, 1]", primals_115: "f32[1536, 384, 1, 1]", primals_116: "f32[1536, 1, 1, 1]", primals_118: "f32[384, 1536, 1, 1]", primals_119: "f32[384, 1, 1, 1]", primals_121: "f32[384, 64, 3, 3]", primals_122: "f32[384, 1, 1, 1]", primals_124: "f32[384, 64, 3, 3]", primals_125: "f32[384, 1, 1, 1]", primals_127: "f32[1536, 384, 1, 1]", primals_128: "f32[1536, 1, 1, 1]", primals_130: "f32[1536, 1536, 1, 1]", primals_131: "f32[1536, 1, 1, 1]", primals_133: "f32[384, 1536, 1, 1]", primals_134: "f32[384, 1, 1, 1]", primals_136: "f32[384, 64, 3, 3]", primals_137: "f32[384, 1, 1, 1]", primals_139: "f32[384, 64, 3, 3]", primals_140: "f32[384, 1, 1, 1]", primals_142: "f32[1536, 384, 1, 1]", primals_143: "f32[1536, 1, 1, 1]", primals_145: "f32[384, 1536, 1, 1]", primals_146: "f32[384, 1, 1, 1]", primals_148: "f32[384, 64, 3, 3]", primals_149: "f32[384, 1, 1, 1]", primals_151: "f32[384, 64, 3, 3]", primals_152: "f32[384, 1, 1, 1]", primals_154: "f32[1536, 384, 1, 1]", primals_155: "f32[1536, 1, 1, 1]", primals_157: "f32[384, 1536, 1, 1]", primals_158: "f32[384, 1, 1, 1]", primals_160: "f32[384, 64, 3, 3]", primals_161: "f32[384, 1, 1, 1]", primals_163: "f32[384, 64, 3, 3]", primals_164: "f32[384, 1, 1, 1]", primals_166: "f32[1536, 384, 1, 1]", primals_167: "f32[1536, 1, 1, 1]", primals_169: "f32[2304, 1536, 1, 1]", primals_170: "f32[2304, 1, 1, 1]", primals_172: "f32[64, 256, 1, 1]", primals_174: "f32[256, 64, 1, 1]", primals_176: "f32[128, 512, 1, 1]", primals_178: "f32[512, 128, 1, 1]", primals_180: "f32[128, 512, 1, 1]", primals_182: "f32[512, 128, 1, 1]", primals_184: "f32[384, 1536, 1, 1]", primals_186: "f32[1536, 384, 1, 1]", primals_188: "f32[384, 1536, 1, 1]", primals_190: "f32[1536, 384, 1, 1]", primals_192: "f32[384, 1536, 1, 1]", primals_194: "f32[1536, 384, 1, 1]", primals_196: "f32[384, 1536, 1, 1]", primals_198: "f32[1536, 384, 1, 1]", primals_200: "f32[384, 1536, 1, 1]", primals_202: "f32[1536, 384, 1, 1]", primals_204: "f32[384, 1536, 1, 1]", primals_206: "f32[1536, 384, 1, 1]", primals_208: "f32[384, 1536, 1, 1]", primals_210: "f32[1536, 384, 1, 1]", primals_212: "f32[384, 1536, 1, 1]", primals_214: "f32[1536, 384, 1, 1]", primals_216: "f32[384, 1536, 1, 1]", primals_218: "f32[1536, 384, 1, 1]", primals_222: "f32[8, 3, 224, 224]", squeeze_1: "f32[16]", view_2: "f32[16, 3, 3, 3]", convolution: "f32[8, 16, 112, 112]", mul_3: "f32[8, 16, 112, 112]", squeeze_3: "f32[32]", view_5: "f32[32, 16, 3, 3]", convolution_1: "f32[8, 32, 112, 112]", mul_7: "f32[8, 32, 112, 112]", squeeze_5: "f32[64]", view_8: "f32[64, 32, 3, 3]", convolution_2: "f32[8, 64, 112, 112]", mul_11: "f32[8, 64, 112, 112]", squeeze_7: "f32[128]", view_11: "f32[128, 64, 3, 3]", convolution_3: "f32[8, 128, 56, 56]", mul_16: "f32[8, 128, 56, 56]", squeeze_9: "f32[256]", view_14: "f32[256, 128, 1, 1]", squeeze_11: "f32[64]", view_17: "f32[64, 128, 1, 1]", convolution_5: "f32[8, 64, 56, 56]", mul_23: "f32[8, 64, 56, 56]", squeeze_13: "f32[64]", view_20: "f32[64, 64, 3, 3]", convolution_6: "f32[8, 64, 56, 56]", mul_27: "f32[8, 64, 56, 56]", squeeze_15: "f32[64]", view_23: "f32[64, 64, 3, 3]", convolution_7: "f32[8, 64, 56, 56]", mul_31: "f32[8, 64, 56, 56]", squeeze_17: "f32[256]", view_26: "f32[256, 64, 1, 1]", convolution_8: "f32[8, 256, 56, 56]", mean: "f32[8, 256, 1, 1]", relu: "f32[8, 64, 1, 1]", convolution_10: "f32[8, 256, 1, 1]", mul_39: "f32[8, 256, 56, 56]", avg_pool2d: "f32[8, 256, 28, 28]", squeeze_19: "f32[512]", view_29: "f32[512, 256, 1, 1]", squeeze_21: "f32[128]", view_32: "f32[128, 256, 1, 1]", convolution_12: "f32[8, 128, 56, 56]", mul_46: "f32[8, 128, 56, 56]", squeeze_23: "f32[128]", view_35: "f32[128, 64, 3, 3]", convolution_13: "f32[8, 128, 28, 28]", mul_50: "f32[8, 128, 28, 28]", squeeze_25: "f32[128]", view_38: "f32[128, 64, 3, 3]", convolution_14: "f32[8, 128, 28, 28]", mul_54: "f32[8, 128, 28, 28]", squeeze_27: "f32[512]", view_41: "f32[512, 128, 1, 1]", convolution_15: "f32[8, 512, 28, 28]", mean_1: "f32[8, 512, 1, 1]", relu_1: "f32[8, 128, 1, 1]", convolution_17: "f32[8, 512, 1, 1]", mul_62: "f32[8, 512, 28, 28]", squeeze_29: "f32[128]", view_44: "f32[128, 512, 1, 1]", convolution_18: "f32[8, 128, 28, 28]", mul_66: "f32[8, 128, 28, 28]", squeeze_31: "f32[128]", view_47: "f32[128, 64, 3, 3]", convolution_19: "f32[8, 128, 28, 28]", mul_70: "f32[8, 128, 28, 28]", squeeze_33: "f32[128]", view_50: "f32[128, 64, 3, 3]", convolution_20: "f32[8, 128, 28, 28]", mul_74: "f32[8, 128, 28, 28]", squeeze_35: "f32[512]", view_53: "f32[512, 128, 1, 1]", convolution_21: "f32[8, 512, 28, 28]", mean_2: "f32[8, 512, 1, 1]", relu_2: "f32[8, 128, 1, 1]", convolution_23: "f32[8, 512, 1, 1]", mul_82: "f32[8, 512, 28, 28]", avg_pool2d_1: "f32[8, 512, 14, 14]", squeeze_37: "f32[1536]", view_56: "f32[1536, 512, 1, 1]", squeeze_39: "f32[384]", view_59: "f32[384, 512, 1, 1]", convolution_25: "f32[8, 384, 28, 28]", mul_89: "f32[8, 384, 28, 28]", squeeze_41: "f32[384]", view_62: "f32[384, 64, 3, 3]", convolution_26: "f32[8, 384, 14, 14]", mul_93: "f32[8, 384, 14, 14]", squeeze_43: "f32[384]", view_65: "f32[384, 64, 3, 3]", convolution_27: "f32[8, 384, 14, 14]", mul_97: "f32[8, 384, 14, 14]", squeeze_45: "f32[1536]", view_68: "f32[1536, 384, 1, 1]", convolution_28: "f32[8, 1536, 14, 14]", mean_3: "f32[8, 1536, 1, 1]", relu_3: "f32[8, 384, 1, 1]", convolution_30: "f32[8, 1536, 1, 1]", mul_105: "f32[8, 1536, 14, 14]", squeeze_47: "f32[384]", view_71: "f32[384, 1536, 1, 1]", convolution_31: "f32[8, 384, 14, 14]", mul_109: "f32[8, 384, 14, 14]", squeeze_49: "f32[384]", view_74: "f32[384, 64, 3, 3]", convolution_32: "f32[8, 384, 14, 14]", mul_113: "f32[8, 384, 14, 14]", squeeze_51: "f32[384]", view_77: "f32[384, 64, 3, 3]", convolution_33: "f32[8, 384, 14, 14]", mul_117: "f32[8, 384, 14, 14]", squeeze_53: "f32[1536]", view_80: "f32[1536, 384, 1, 1]", convolution_34: "f32[8, 1536, 14, 14]", mean_4: "f32[8, 1536, 1, 1]", relu_4: "f32[8, 384, 1, 1]", convolution_36: "f32[8, 1536, 1, 1]", mul_125: "f32[8, 1536, 14, 14]", squeeze_55: "f32[384]", view_83: "f32[384, 1536, 1, 1]", convolution_37: "f32[8, 384, 14, 14]", mul_129: "f32[8, 384, 14, 14]", squeeze_57: "f32[384]", view_86: "f32[384, 64, 3, 3]", convolution_38: "f32[8, 384, 14, 14]", mul_133: "f32[8, 384, 14, 14]", squeeze_59: "f32[384]", view_89: "f32[384, 64, 3, 3]", convolution_39: "f32[8, 384, 14, 14]", mul_137: "f32[8, 384, 14, 14]", squeeze_61: "f32[1536]", view_92: "f32[1536, 384, 1, 1]", convolution_40: "f32[8, 1536, 14, 14]", mean_5: "f32[8, 1536, 1, 1]", relu_5: "f32[8, 384, 1, 1]", convolution_42: "f32[8, 1536, 1, 1]", mul_145: "f32[8, 1536, 14, 14]", squeeze_63: "f32[384]", view_95: "f32[384, 1536, 1, 1]", convolution_43: "f32[8, 384, 14, 14]", mul_149: "f32[8, 384, 14, 14]", squeeze_65: "f32[384]", view_98: "f32[384, 64, 3, 3]", convolution_44: "f32[8, 384, 14, 14]", mul_153: "f32[8, 384, 14, 14]", squeeze_67: "f32[384]", view_101: "f32[384, 64, 3, 3]", convolution_45: "f32[8, 384, 14, 14]", mul_157: "f32[8, 384, 14, 14]", squeeze_69: "f32[1536]", view_104: "f32[1536, 384, 1, 1]", convolution_46: "f32[8, 1536, 14, 14]", mean_6: "f32[8, 1536, 1, 1]", relu_6: "f32[8, 384, 1, 1]", convolution_48: "f32[8, 1536, 1, 1]", mul_165: "f32[8, 1536, 14, 14]", squeeze_71: "f32[384]", view_107: "f32[384, 1536, 1, 1]", convolution_49: "f32[8, 384, 14, 14]", mul_169: "f32[8, 384, 14, 14]", squeeze_73: "f32[384]", view_110: "f32[384, 64, 3, 3]", convolution_50: "f32[8, 384, 14, 14]", mul_173: "f32[8, 384, 14, 14]", squeeze_75: "f32[384]", view_113: "f32[384, 64, 3, 3]", convolution_51: "f32[8, 384, 14, 14]", mul_177: "f32[8, 384, 14, 14]", squeeze_77: "f32[1536]", view_116: "f32[1536, 384, 1, 1]", convolution_52: "f32[8, 1536, 14, 14]", mean_7: "f32[8, 1536, 1, 1]", relu_7: "f32[8, 384, 1, 1]", convolution_54: "f32[8, 1536, 1, 1]", mul_185: "f32[8, 1536, 14, 14]", squeeze_79: "f32[384]", view_119: "f32[384, 1536, 1, 1]", convolution_55: "f32[8, 384, 14, 14]", mul_189: "f32[8, 384, 14, 14]", squeeze_81: "f32[384]", view_122: "f32[384, 64, 3, 3]", convolution_56: "f32[8, 384, 14, 14]", mul_193: "f32[8, 384, 14, 14]", squeeze_83: "f32[384]", view_125: "f32[384, 64, 3, 3]", convolution_57: "f32[8, 384, 14, 14]", mul_197: "f32[8, 384, 14, 14]", squeeze_85: "f32[1536]", view_128: "f32[1536, 384, 1, 1]", convolution_58: "f32[8, 1536, 14, 14]", mean_8: "f32[8, 1536, 1, 1]", relu_8: "f32[8, 384, 1, 1]", convolution_60: "f32[8, 1536, 1, 1]", mul_205: "f32[8, 1536, 14, 14]", avg_pool2d_2: "f32[8, 1536, 7, 7]", squeeze_87: "f32[1536]", view_131: "f32[1536, 1536, 1, 1]", squeeze_89: "f32[384]", view_134: "f32[384, 1536, 1, 1]", convolution_62: "f32[8, 384, 14, 14]", mul_212: "f32[8, 384, 14, 14]", squeeze_91: "f32[384]", view_137: "f32[384, 64, 3, 3]", convolution_63: "f32[8, 384, 7, 7]", mul_216: "f32[8, 384, 7, 7]", squeeze_93: "f32[384]", view_140: "f32[384, 64, 3, 3]", convolution_64: "f32[8, 384, 7, 7]", mul_220: "f32[8, 384, 7, 7]", squeeze_95: "f32[1536]", view_143: "f32[1536, 384, 1, 1]", convolution_65: "f32[8, 1536, 7, 7]", mean_9: "f32[8, 1536, 1, 1]", relu_9: "f32[8, 384, 1, 1]", convolution_67: "f32[8, 1536, 1, 1]", mul_228: "f32[8, 1536, 7, 7]", squeeze_97: "f32[384]", view_146: "f32[384, 1536, 1, 1]", convolution_68: "f32[8, 384, 7, 7]", mul_232: "f32[8, 384, 7, 7]", squeeze_99: "f32[384]", view_149: "f32[384, 64, 3, 3]", convolution_69: "f32[8, 384, 7, 7]", mul_236: "f32[8, 384, 7, 7]", squeeze_101: "f32[384]", view_152: "f32[384, 64, 3, 3]", convolution_70: "f32[8, 384, 7, 7]", mul_240: "f32[8, 384, 7, 7]", squeeze_103: "f32[1536]", view_155: "f32[1536, 384, 1, 1]", convolution_71: "f32[8, 1536, 7, 7]", mean_10: "f32[8, 1536, 1, 1]", relu_10: "f32[8, 384, 1, 1]", convolution_73: "f32[8, 1536, 1, 1]", mul_248: "f32[8, 1536, 7, 7]", squeeze_105: "f32[384]", view_158: "f32[384, 1536, 1, 1]", convolution_74: "f32[8, 384, 7, 7]", mul_252: "f32[8, 384, 7, 7]", squeeze_107: "f32[384]", view_161: "f32[384, 64, 3, 3]", convolution_75: "f32[8, 384, 7, 7]", mul_256: "f32[8, 384, 7, 7]", squeeze_109: "f32[384]", view_164: "f32[384, 64, 3, 3]", convolution_76: "f32[8, 384, 7, 7]", mul_260: "f32[8, 384, 7, 7]", squeeze_111: "f32[1536]", view_167: "f32[1536, 384, 1, 1]", convolution_77: "f32[8, 1536, 7, 7]", mean_11: "f32[8, 1536, 1, 1]", relu_11: "f32[8, 384, 1, 1]", convolution_79: "f32[8, 1536, 1, 1]", add_67: "f32[8, 1536, 7, 7]", squeeze_113: "f32[2304]", view_170: "f32[2304, 1536, 1, 1]", convolution_80: "f32[8, 2304, 7, 7]", clone_28: "f32[8, 2304]", permute_1: "f32[1000, 2304]", unsqueeze_58: "f32[1, 2304, 1]", unsqueeze_66: "f32[1, 1536, 1]", unsqueeze_74: "f32[1, 384, 1]", unsqueeze_82: "f32[1, 384, 1]", unsqueeze_90: "f32[1, 384, 1]", mul_341: "f32[8, 1536, 7, 7]", unsqueeze_98: "f32[1, 1536, 1]", unsqueeze_106: "f32[1, 384, 1]", unsqueeze_114: "f32[1, 384, 1]", unsqueeze_122: "f32[1, 384, 1]", mul_400: "f32[8, 1536, 7, 7]", unsqueeze_130: "f32[1, 1536, 1]", unsqueeze_138: "f32[1, 384, 1]", unsqueeze_146: "f32[1, 384, 1]", unsqueeze_154: "f32[1, 384, 1]", unsqueeze_162: "f32[1, 1536, 1]", mul_469: "f32[8, 1536, 14, 14]", unsqueeze_170: "f32[1, 1536, 1]", unsqueeze_178: "f32[1, 384, 1]", unsqueeze_186: "f32[1, 384, 1]", unsqueeze_194: "f32[1, 384, 1]", mul_528: "f32[8, 1536, 14, 14]", unsqueeze_202: "f32[1, 1536, 1]", unsqueeze_210: "f32[1, 384, 1]", unsqueeze_218: "f32[1, 384, 1]", unsqueeze_226: "f32[1, 384, 1]", mul_587: "f32[8, 1536, 14, 14]", unsqueeze_234: "f32[1, 1536, 1]", unsqueeze_242: "f32[1, 384, 1]", unsqueeze_250: "f32[1, 384, 1]", unsqueeze_258: "f32[1, 384, 1]", mul_646: "f32[8, 1536, 14, 14]", unsqueeze_266: "f32[1, 1536, 1]", unsqueeze_274: "f32[1, 384, 1]", unsqueeze_282: "f32[1, 384, 1]", unsqueeze_290: "f32[1, 384, 1]", mul_705: "f32[8, 1536, 14, 14]", unsqueeze_298: "f32[1, 1536, 1]", unsqueeze_306: "f32[1, 384, 1]", unsqueeze_314: "f32[1, 384, 1]", unsqueeze_322: "f32[1, 384, 1]", mul_764: "f32[8, 1536, 14, 14]", unsqueeze_330: "f32[1, 1536, 1]", unsqueeze_338: "f32[1, 384, 1]", unsqueeze_346: "f32[1, 384, 1]", unsqueeze_354: "f32[1, 384, 1]", unsqueeze_362: "f32[1, 1536, 1]", mul_833: "f32[8, 512, 28, 28]", unsqueeze_370: "f32[1, 512, 1]", unsqueeze_378: "f32[1, 128, 1]", unsqueeze_386: "f32[1, 128, 1]", unsqueeze_394: "f32[1, 128, 1]", mul_892: "f32[8, 512, 28, 28]", unsqueeze_402: "f32[1, 512, 1]", unsqueeze_410: "f32[1, 128, 1]", unsqueeze_418: "f32[1, 128, 1]", unsqueeze_426: "f32[1, 128, 1]", unsqueeze_434: "f32[1, 512, 1]", mul_961: "f32[8, 256, 56, 56]", unsqueeze_442: "f32[1, 256, 1]", unsqueeze_450: "f32[1, 64, 1]", unsqueeze_458: "f32[1, 64, 1]", unsqueeze_466: "f32[1, 64, 1]", unsqueeze_474: "f32[1, 256, 1]", unsqueeze_482: "f32[1, 128, 1]", unsqueeze_490: "f32[1, 64, 1]", unsqueeze_498: "f32[1, 32, 1]", unsqueeze_506: "f32[1, 16, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view: "f32[1, 16, 27]" = torch.ops.aten.reshape.default(primals_1, [1, 16, -1]);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_2, 0.34412564994580647);  primals_2 = None
    view_1: "f32[16]" = torch.ops.aten.reshape.default(mul, [-1]);  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_3: "f32[1, 32, 144]" = torch.ops.aten.reshape.default(primals_4, [1, 32, -1]);  primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_4: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_5, 0.1490107774734497);  primals_5 = None
    view_4: "f32[32]" = torch.ops.aten.reshape.default(mul_4, [-1]);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_6: "f32[1, 64, 288]" = torch.ops.aten.reshape.default(primals_7, [1, 64, -1]);  primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_8: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_8, 0.10536653122135592);  primals_8 = None
    view_7: "f32[64]" = torch.ops.aten.reshape.default(mul_8, [-1]);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_9: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(primals_10, [1, 128, -1]);  primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_12: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_11, 0.07450538873672485);  primals_11 = None
    view_10: "f32[128]" = torch.ops.aten.reshape.default(mul_12, [-1]);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    sigmoid_3: "f32[8, 128, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_12: "f32[1, 256, 128]" = torch.ops.aten.reshape.default(primals_13, [1, 256, -1]);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_17: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_14, 0.1580497968320339);  primals_14 = None
    view_13: "f32[256]" = torch.ops.aten.reshape.default(mul_17, [-1]);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_15: "f32[1, 64, 128]" = torch.ops.aten.reshape.default(primals_16, [1, 64, -1]);  primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_20: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_17, 0.1580497968320339);  primals_17 = None
    view_16: "f32[64]" = torch.ops.aten.reshape.default(mul_20, [-1]);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_18: "f32[1, 64, 576]" = torch.ops.aten.reshape.default(primals_19, [1, 64, -1]);  primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_24: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_20, 0.07450538873672485);  primals_20 = None
    view_19: "f32[64]" = torch.ops.aten.reshape.default(mul_24, [-1]);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_21: "f32[1, 64, 576]" = torch.ops.aten.reshape.default(primals_22, [1, 64, -1]);  primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_28: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_23, 0.07450538873672485);  primals_23 = None
    view_22: "f32[64]" = torch.ops.aten.reshape.default(mul_28, [-1]);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_6: "f32[8, 64, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_24: "f32[1, 256, 64]" = torch.ops.aten.reshape.default(primals_25, [1, 256, -1]);  primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_32: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_26, 0.22351616621017456);  primals_26 = None
    view_25: "f32[256]" = torch.ops.aten.reshape.default(mul_32, [-1]);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_27: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(primals_28, [1, 512, -1]);  primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_40: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_29, 0.11175808310508728);  primals_29 = None
    view_28: "f32[512]" = torch.ops.aten.reshape.default(mul_40, [-1]);  mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_30: "f32[1, 128, 256]" = torch.ops.aten.reshape.default(primals_31, [1, 128, -1]);  primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_43: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_32, 0.11175808310508728);  primals_32 = None
    view_31: "f32[128]" = torch.ops.aten.reshape.default(mul_43, [-1]);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_33: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(primals_34, [1, 128, -1]);  primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_47: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_35, 0.07450538873672485);  primals_35 = None
    view_34: "f32[128]" = torch.ops.aten.reshape.default(mul_47, [-1]);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_36: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(primals_37, [1, 128, -1]);  primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_51: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_38, 0.07450538873672485);  primals_38 = None
    view_37: "f32[128]" = torch.ops.aten.reshape.default(mul_51, [-1]);  mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_11: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_39: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(primals_40, [1, 512, -1]);  primals_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_55: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_41, 0.1580497968320339);  primals_41 = None
    view_40: "f32[512]" = torch.ops.aten.reshape.default(mul_55, [-1]);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_12: "f32[8, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_42: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(primals_43, [1, 128, -1]);  primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_63: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_44, 0.07902489841601695);  primals_44 = None
    view_43: "f32[128]" = torch.ops.aten.reshape.default(mul_63, [-1]);  mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_45: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(primals_46, [1, 128, -1]);  primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_67: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_47, 0.07450538873672485);  primals_47 = None
    view_46: "f32[128]" = torch.ops.aten.reshape.default(mul_67, [-1]);  mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_48: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(primals_49, [1, 128, -1]);  primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_71: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_50, 0.07450538873672485);  primals_50 = None
    view_49: "f32[128]" = torch.ops.aten.reshape.default(mul_71, [-1]);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_16: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_51: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(primals_52, [1, 512, -1]);  primals_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_75: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_53, 0.1580497968320339);  primals_53 = None
    view_52: "f32[512]" = torch.ops.aten.reshape.default(mul_75, [-1]);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_17: "f32[8, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_54: "f32[1, 1536, 512]" = torch.ops.aten.reshape.default(primals_55, [1, 1536, -1]);  primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_83: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_56, 0.07902489841601695);  primals_56 = None
    view_55: "f32[1536]" = torch.ops.aten.reshape.default(mul_83, [-1]);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_57: "f32[1, 384, 512]" = torch.ops.aten.reshape.default(primals_58, [1, 384, -1]);  primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_86: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_59, 0.07902489841601695);  primals_59 = None
    view_58: "f32[384]" = torch.ops.aten.reshape.default(mul_86, [-1]);  mul_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_60: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_61, [1, 384, -1]);  primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_90: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_62, 0.07450538873672485);  primals_62 = None
    view_61: "f32[384]" = torch.ops.aten.reshape.default(mul_90, [-1]);  mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_63: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_64, [1, 384, -1]);  primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_94: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_65, 0.07450538873672485);  primals_65 = None
    view_64: "f32[384]" = torch.ops.aten.reshape.default(mul_94, [-1]);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_21: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_66: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(primals_67, [1, 1536, -1]);  primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_98: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_68, 0.09125009274634042);  primals_68 = None
    view_67: "f32[1536]" = torch.ops.aten.reshape.default(mul_98, [-1]);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_22: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_30);  convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_69: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(primals_70, [1, 384, -1]);  primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_106: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_71, 0.04562504637317021);  primals_71 = None
    view_70: "f32[384]" = torch.ops.aten.reshape.default(mul_106, [-1]);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_72: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_73, [1, 384, -1]);  primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_110: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_74, 0.07450538873672485);  primals_74 = None
    view_73: "f32[384]" = torch.ops.aten.reshape.default(mul_110, [-1]);  mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_75: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_76, [1, 384, -1]);  primals_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_114: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_77, 0.07450538873672485);  primals_77 = None
    view_76: "f32[384]" = torch.ops.aten.reshape.default(mul_114, [-1]);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_26: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_78: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(primals_79, [1, 1536, -1]);  primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_118: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_80, 0.09125009274634042);  primals_80 = None
    view_79: "f32[1536]" = torch.ops.aten.reshape.default(mul_118, [-1]);  mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_27: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_81: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(primals_82, [1, 384, -1]);  primals_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_126: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_83, 0.04562504637317021);  primals_83 = None
    view_82: "f32[384]" = torch.ops.aten.reshape.default(mul_126, [-1]);  mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_84: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_85, [1, 384, -1]);  primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_130: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_86, 0.07450538873672485);  primals_86 = None
    view_85: "f32[384]" = torch.ops.aten.reshape.default(mul_130, [-1]);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_87: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_88, [1, 384, -1]);  primals_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_134: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_89, 0.07450538873672485);  primals_89 = None
    view_88: "f32[384]" = torch.ops.aten.reshape.default(mul_134, [-1]);  mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_31: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_90: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(primals_91, [1, 1536, -1]);  primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_138: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_92, 0.09125009274634042);  primals_92 = None
    view_91: "f32[1536]" = torch.ops.aten.reshape.default(mul_138, [-1]);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_32: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42);  convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_93: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(primals_94, [1, 384, -1]);  primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_146: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_95, 0.04562504637317021);  primals_95 = None
    view_94: "f32[384]" = torch.ops.aten.reshape.default(mul_146, [-1]);  mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_96: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_97, [1, 384, -1]);  primals_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_150: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_98, 0.07450538873672485);  primals_98 = None
    view_97: "f32[384]" = torch.ops.aten.reshape.default(mul_150, [-1]);  mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_99: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_100, [1, 384, -1]);  primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_154: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_101, 0.07450538873672485);  primals_101 = None
    view_100: "f32[384]" = torch.ops.aten.reshape.default(mul_154, [-1]);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_36: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_102: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(primals_103, [1, 1536, -1]);  primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_158: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_104, 0.09125009274634042);  primals_104 = None
    view_103: "f32[1536]" = torch.ops.aten.reshape.default(mul_158, [-1]);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_37: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_105: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(primals_106, [1, 384, -1]);  primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_166: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_107, 0.04562504637317021);  primals_107 = None
    view_106: "f32[384]" = torch.ops.aten.reshape.default(mul_166, [-1]);  mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_108: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_109, [1, 384, -1]);  primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_170: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_110, 0.07450538873672485);  primals_110 = None
    view_109: "f32[384]" = torch.ops.aten.reshape.default(mul_170, [-1]);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_111: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_112, [1, 384, -1]);  primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_174: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_113, 0.07450538873672485);  primals_113 = None
    view_112: "f32[384]" = torch.ops.aten.reshape.default(mul_174, [-1]);  mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_41: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_114: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(primals_115, [1, 1536, -1]);  primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_178: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_116, 0.09125009274634042);  primals_116 = None
    view_115: "f32[1536]" = torch.ops.aten.reshape.default(mul_178, [-1]);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_42: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_54);  convolution_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_117: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(primals_118, [1, 384, -1]);  primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_186: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_119, 0.04562504637317021);  primals_119 = None
    view_118: "f32[384]" = torch.ops.aten.reshape.default(mul_186, [-1]);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_120: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_121, [1, 384, -1]);  primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_190: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_122, 0.07450538873672485);  primals_122 = None
    view_121: "f32[384]" = torch.ops.aten.reshape.default(mul_190, [-1]);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_123: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_124, [1, 384, -1]);  primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_194: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_125, 0.07450538873672485);  primals_125 = None
    view_124: "f32[384]" = torch.ops.aten.reshape.default(mul_194, [-1]);  mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_46: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_126: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(primals_127, [1, 1536, -1]);  primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_198: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_128, 0.09125009274634042);  primals_128 = None
    view_127: "f32[1536]" = torch.ops.aten.reshape.default(mul_198, [-1]);  mul_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_47: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60);  convolution_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_129: "f32[1, 1536, 1536]" = torch.ops.aten.reshape.default(primals_130, [1, 1536, -1]);  primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_206: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_131, 0.04562504637317021);  primals_131 = None
    view_130: "f32[1536]" = torch.ops.aten.reshape.default(mul_206, [-1]);  mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_132: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(primals_133, [1, 384, -1]);  primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_209: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_134, 0.04562504637317021);  primals_134 = None
    view_133: "f32[384]" = torch.ops.aten.reshape.default(mul_209, [-1]);  mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_135: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_136, [1, 384, -1]);  primals_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_213: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_137, 0.07450538873672485);  primals_137 = None
    view_136: "f32[384]" = torch.ops.aten.reshape.default(mul_213, [-1]);  mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_138: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_139, [1, 384, -1]);  primals_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_217: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_140, 0.07450538873672485);  primals_140 = None
    view_139: "f32[384]" = torch.ops.aten.reshape.default(mul_217, [-1]);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_51: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_141: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(primals_142, [1, 1536, -1]);  primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_221: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_143, 0.09125009274634042);  primals_143 = None
    view_142: "f32[1536]" = torch.ops.aten.reshape.default(mul_221, [-1]);  mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_52: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_144: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(primals_145, [1, 384, -1]);  primals_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_229: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_146, 0.04562504637317021);  primals_146 = None
    view_145: "f32[384]" = torch.ops.aten.reshape.default(mul_229, [-1]);  mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_147: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_148, [1, 384, -1]);  primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_233: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_149, 0.07450538873672485);  primals_149 = None
    view_148: "f32[384]" = torch.ops.aten.reshape.default(mul_233, [-1]);  mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_150: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_151, [1, 384, -1]);  primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_237: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_152, 0.07450538873672485);  primals_152 = None
    view_151: "f32[384]" = torch.ops.aten.reshape.default(mul_237, [-1]);  mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_56: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_70)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_153: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(primals_154, [1, 1536, -1]);  primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_241: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_155, 0.09125009274634042);  primals_155 = None
    view_154: "f32[1536]" = torch.ops.aten.reshape.default(mul_241, [-1]);  mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_57: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_156: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(primals_157, [1, 384, -1]);  primals_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_249: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_158, 0.04562504637317021);  primals_158 = None
    view_157: "f32[384]" = torch.ops.aten.reshape.default(mul_249, [-1]);  mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_159: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_160, [1, 384, -1]);  primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_253: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_161, 0.07450538873672485);  primals_161 = None
    view_160: "f32[384]" = torch.ops.aten.reshape.default(mul_253, [-1]);  mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_162: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(primals_163, [1, 384, -1]);  primals_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_257: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_164, 0.07450538873672485);  primals_164 = None
    view_163: "f32[384]" = torch.ops.aten.reshape.default(mul_257, [-1]);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sigmoid_61: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_76)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_165: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(primals_166, [1, 1536, -1]);  primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_261: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_167, 0.09125009274634042);  primals_167 = None
    view_166: "f32[1536]" = torch.ops.aten.reshape.default(mul_261, [-1]);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_62: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_79);  convolution_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_168: "f32[1, 2304, 1536]" = torch.ops.aten.reshape.default(primals_169, [1, 2304, -1]);  primals_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    mul_267: "f32[2304, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_170, 0.04562504637317021);  primals_170 = None
    view_169: "f32[2304]" = torch.ops.aten.reshape.default(mul_267, [-1]);  mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm: "f32[8, 2304]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 2304]" = torch.ops.aten.mm.default(permute_2, clone_28);  permute_2 = clone_28 = None
    permute_3: "f32[2304, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_172: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 2304]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_173: "f32[8, 2304, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 2304, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 2304, 7, 7]" = torch.ops.aten.expand.default(view_173, [8, 2304, 7, 7]);  view_173 = None
    div: "f32[8, 2304, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:445, code: x = self.final_act(x)
    sigmoid_64: "f32[8, 2304, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_80)
    full_default: "f32[8, 2304, 7, 7]" = torch.ops.aten.full.default([8, 2304, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_57: "f32[8, 2304, 7, 7]" = torch.ops.aten.sub.Tensor(full_default, sigmoid_64);  full_default = None
    mul_271: "f32[8, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_80, sub_57);  convolution_80 = sub_57 = None
    add_69: "f32[8, 2304, 7, 7]" = torch.ops.aten.add.Scalar(mul_271, 1);  mul_271 = None
    mul_272: "f32[8, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_64, add_69);  sigmoid_64 = add_69 = None
    mul_273: "f32[8, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(div, mul_272);  div = mul_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_2: "f32[2304]" = torch.ops.aten.sum.dim_IntList(mul_273, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_273, add_67, view_170, [2304], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_273 = add_67 = view_170 = None
    getitem_114: "f32[8, 1536, 7, 7]" = convolution_backward[0]
    getitem_115: "f32[2304, 1536, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_174: "f32[1, 2304, 1536]" = torch.ops.aten.reshape.default(getitem_115, [1, 2304, 1536]);  getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_3: "f32[2304]" = torch.ops.aten.sum.dim_IntList(view_174, [0, 2])
    sub_58: "f32[1, 2304, 1536]" = torch.ops.aten.sub.Tensor(view_168, unsqueeze_58);  view_168 = unsqueeze_58 = None
    mul_274: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(view_174, sub_58)
    sum_4: "f32[2304]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 2]);  mul_274 = None
    mul_275: "f32[2304]" = torch.ops.aten.mul.Tensor(sum_3, 0.0006510416666666666);  sum_3 = None
    unsqueeze_59: "f32[1, 2304]" = torch.ops.aten.unsqueeze.default(mul_275, 0);  mul_275 = None
    unsqueeze_60: "f32[1, 2304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 2);  unsqueeze_59 = None
    mul_276: "f32[2304]" = torch.ops.aten.mul.Tensor(sum_4, 0.0006510416666666666)
    mul_277: "f32[2304]" = torch.ops.aten.mul.Tensor(squeeze_113, squeeze_113)
    mul_278: "f32[2304]" = torch.ops.aten.mul.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
    unsqueeze_61: "f32[1, 2304]" = torch.ops.aten.unsqueeze.default(mul_278, 0);  mul_278 = None
    unsqueeze_62: "f32[1, 2304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 2);  unsqueeze_61 = None
    mul_279: "f32[2304]" = torch.ops.aten.mul.Tensor(squeeze_113, view_169);  view_169 = None
    unsqueeze_63: "f32[1, 2304]" = torch.ops.aten.unsqueeze.default(mul_279, 0);  mul_279 = None
    unsqueeze_64: "f32[1, 2304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 2);  unsqueeze_63 = None
    mul_280: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_62);  sub_58 = unsqueeze_62 = None
    sub_60: "f32[1, 2304, 1536]" = torch.ops.aten.sub.Tensor(view_174, mul_280);  view_174 = mul_280 = None
    sub_61: "f32[1, 2304, 1536]" = torch.ops.aten.sub.Tensor(sub_60, unsqueeze_60);  sub_60 = unsqueeze_60 = None
    mul_281: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_64);  sub_61 = unsqueeze_64 = None
    mul_282: "f32[2304]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_113);  sum_4 = squeeze_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_175: "f32[2304, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_282, [2304, 1, 1, 1]);  mul_282 = None
    mul_283: "f32[2304, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_175, 0.04562504637317021);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_176: "f32[2304, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_281, [2304, 1536, 1, 1]);  mul_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_284: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_114, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_285: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_284, 2.0);  mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_286: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_285, convolution_77);  convolution_77 = None
    mul_287: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_285, sigmoid_62);  mul_285 = None
    sum_5: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2, 3], True);  mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_62: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_62)
    mul_288: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_62, sub_62);  sigmoid_62 = sub_62 = None
    mul_289: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_5, mul_288);  sum_5 = mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_6: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_289, relu_11, primals_218, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_289 = primals_218 = None
    getitem_117: "f32[8, 384, 1, 1]" = convolution_backward_1[0]
    getitem_118: "f32[1536, 384, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le, full_default_1, getitem_117);  le = getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_7: "f32[384]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where, mean_11, primals_216, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = mean_11 = primals_216 = None
    getitem_120: "f32[8, 1536, 1, 1]" = convolution_backward_2[0]
    getitem_121: "f32[384, 1536, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 1536, 7, 7]" = torch.ops.aten.expand.default(getitem_120, [8, 1536, 7, 7]);  getitem_120 = None
    div_1: "f32[8, 1536, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_70: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_287, div_1);  mul_287 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_8: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_70, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(add_70, mul_260, view_167, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_70 = mul_260 = view_167 = None
    getitem_123: "f32[8, 384, 7, 7]" = convolution_backward_3[0]
    getitem_124: "f32[1536, 384, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_177: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(getitem_124, [1, 1536, 384]);  getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_9: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_177, [0, 2])
    sub_63: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_165, unsqueeze_66);  view_165 = unsqueeze_66 = None
    mul_290: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_177, sub_63)
    sum_10: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 2]);  mul_290 = None
    mul_291: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_9, 0.0026041666666666665);  sum_9 = None
    unsqueeze_67: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_291, 0);  mul_291 = None
    unsqueeze_68: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 2);  unsqueeze_67 = None
    mul_292: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_10, 0.0026041666666666665)
    mul_293: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_111, squeeze_111)
    mul_294: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
    unsqueeze_69: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_294, 0);  mul_294 = None
    unsqueeze_70: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 2);  unsqueeze_69 = None
    mul_295: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_111, view_166);  view_166 = None
    unsqueeze_71: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_295, 0);  mul_295 = None
    unsqueeze_72: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_71, 2);  unsqueeze_71 = None
    mul_296: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_70);  sub_63 = unsqueeze_70 = None
    sub_65: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_177, mul_296);  view_177 = mul_296 = None
    sub_66: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_68);  sub_65 = unsqueeze_68 = None
    mul_297: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_72);  sub_66 = unsqueeze_72 = None
    mul_298: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_111);  sum_10 = squeeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_178: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_298, [1536, 1, 1, 1]);  mul_298 = None
    mul_299: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_178, 0.09125009274634042);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_179: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_297, [1536, 384, 1, 1]);  mul_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    full_default_2: "f32[8, 384, 7, 7]" = torch.ops.aten.full.default([8, 384, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_67: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_61)
    mul_300: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_76, sub_67);  convolution_76 = sub_67 = None
    add_71: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_300, 1);  mul_300 = None
    mul_301: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_61, add_71);  sigmoid_61 = add_71 = None
    mul_302: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_123, mul_301);  getitem_123 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_11: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 2, 3])
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_302, mul_256, view_164, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_302 = mul_256 = view_164 = None
    getitem_126: "f32[8, 384, 7, 7]" = convolution_backward_4[0]
    getitem_127: "f32[384, 64, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_180: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_127, [1, 384, 576]);  getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_12: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_180, [0, 2])
    sub_68: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_162, unsqueeze_74);  view_162 = unsqueeze_74 = None
    mul_303: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_180, sub_68)
    sum_13: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 2]);  mul_303 = None
    mul_304: "f32[384]" = torch.ops.aten.mul.Tensor(sum_12, 0.001736111111111111);  sum_12 = None
    unsqueeze_75: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_304, 0);  mul_304 = None
    unsqueeze_76: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 2);  unsqueeze_75 = None
    mul_305: "f32[384]" = torch.ops.aten.mul.Tensor(sum_13, 0.001736111111111111)
    mul_306: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_307: "f32[384]" = torch.ops.aten.mul.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_77: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_307, 0);  mul_307 = None
    unsqueeze_78: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_77, 2);  unsqueeze_77 = None
    mul_308: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_109, view_163);  view_163 = None
    unsqueeze_79: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_308, 0);  mul_308 = None
    unsqueeze_80: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 2);  unsqueeze_79 = None
    mul_309: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_78);  sub_68 = unsqueeze_78 = None
    sub_70: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_180, mul_309);  view_180 = mul_309 = None
    sub_71: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_76);  sub_70 = unsqueeze_76 = None
    mul_310: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_80);  sub_71 = unsqueeze_80 = None
    mul_311: "f32[384]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_109);  sum_13 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_181: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_311, [384, 1, 1, 1]);  mul_311 = None
    mul_312: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_181, 0.07450538873672485);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_182: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_310, [384, 64, 3, 3]);  mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_66: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_75)
    sub_72: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_66)
    mul_313: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_75, sub_72);  convolution_75 = sub_72 = None
    add_72: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_313, 1);  mul_313 = None
    mul_314: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_66, add_72);  sigmoid_66 = add_72 = None
    mul_315: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_126, mul_314);  getitem_126 = mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_14: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 2, 3])
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_315, mul_252, view_161, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_315 = mul_252 = view_161 = None
    getitem_129: "f32[8, 384, 7, 7]" = convolution_backward_5[0]
    getitem_130: "f32[384, 64, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_183: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_130, [1, 384, 576]);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_15: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_183, [0, 2])
    sub_73: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_159, unsqueeze_82);  view_159 = unsqueeze_82 = None
    mul_316: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_183, sub_73)
    sum_16: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 2]);  mul_316 = None
    mul_317: "f32[384]" = torch.ops.aten.mul.Tensor(sum_15, 0.001736111111111111);  sum_15 = None
    unsqueeze_83: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_317, 0);  mul_317 = None
    unsqueeze_84: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_83, 2);  unsqueeze_83 = None
    mul_318: "f32[384]" = torch.ops.aten.mul.Tensor(sum_16, 0.001736111111111111)
    mul_319: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_107, squeeze_107)
    mul_320: "f32[384]" = torch.ops.aten.mul.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
    unsqueeze_85: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_320, 0);  mul_320 = None
    unsqueeze_86: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 2);  unsqueeze_85 = None
    mul_321: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_107, view_160);  view_160 = None
    unsqueeze_87: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_321, 0);  mul_321 = None
    unsqueeze_88: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
    mul_322: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_86);  sub_73 = unsqueeze_86 = None
    sub_75: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_183, mul_322);  view_183 = mul_322 = None
    sub_76: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_84);  sub_75 = unsqueeze_84 = None
    mul_323: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_88);  sub_76 = unsqueeze_88 = None
    mul_324: "f32[384]" = torch.ops.aten.mul.Tensor(sum_16, squeeze_107);  sum_16 = squeeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_184: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_324, [384, 1, 1, 1]);  mul_324 = None
    mul_325: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_184, 0.07450538873672485);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_185: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_323, [384, 64, 3, 3]);  mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_67: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_74)
    sub_77: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_67)
    mul_326: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_74, sub_77);  convolution_74 = sub_77 = None
    add_73: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_326, 1);  mul_326 = None
    mul_327: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_67, add_73);  sigmoid_67 = add_73 = None
    mul_328: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_129, mul_327);  getitem_129 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_17: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 2, 3])
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_328, mul_248, view_158, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_328 = mul_248 = view_158 = None
    getitem_132: "f32[8, 1536, 7, 7]" = convolution_backward_6[0]
    getitem_133: "f32[384, 1536, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_186: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(getitem_133, [1, 384, 1536]);  getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_18: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_186, [0, 2])
    sub_78: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_156, unsqueeze_90);  view_156 = unsqueeze_90 = None
    mul_329: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_186, sub_78)
    sum_19: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 2]);  mul_329 = None
    mul_330: "f32[384]" = torch.ops.aten.mul.Tensor(sum_18, 0.0006510416666666666);  sum_18 = None
    unsqueeze_91: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_330, 0);  mul_330 = None
    unsqueeze_92: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 2);  unsqueeze_91 = None
    mul_331: "f32[384]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006510416666666666)
    mul_332: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_105, squeeze_105)
    mul_333: "f32[384]" = torch.ops.aten.mul.Tensor(mul_331, mul_332);  mul_331 = mul_332 = None
    unsqueeze_93: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_333, 0);  mul_333 = None
    unsqueeze_94: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None
    mul_334: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_105, view_157);  view_157 = None
    unsqueeze_95: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_334, 0);  mul_334 = None
    unsqueeze_96: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
    mul_335: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_94);  sub_78 = unsqueeze_94 = None
    sub_80: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_186, mul_335);  view_186 = mul_335 = None
    sub_81: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_80, unsqueeze_92);  sub_80 = unsqueeze_92 = None
    mul_336: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_96);  sub_81 = unsqueeze_96 = None
    mul_337: "f32[384]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_105);  sum_19 = squeeze_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_187: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_337, [384, 1, 1, 1]);  mul_337 = None
    mul_338: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_187, 0.04562504637317021);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_188: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_336, [384, 1536, 1, 1]);  mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_339: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_132, 0.9622504486493761);  getitem_132 = None
    mul_342: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_339, mul_341);  mul_339 = mul_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_75: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(getitem_114, mul_342);  getitem_114 = mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_343: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_75, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_344: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_343, 2.0);  mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_345: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_344, convolution_71);  convolution_71 = None
    mul_346: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_344, sigmoid_57);  mul_344 = None
    sum_20: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_345, [2, 3], True);  mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_83: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_57)
    mul_347: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_57, sub_83);  sigmoid_57 = sub_83 = None
    mul_348: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_20, mul_347);  sum_20 = mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_21: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_348, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_348, relu_10, primals_214, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_348 = primals_214 = None
    getitem_135: "f32[8, 384, 1, 1]" = convolution_backward_7[0]
    getitem_136: "f32[1536, 384, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_1: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_1: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_1, full_default_1, getitem_135);  le_1 = getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_22: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_1, mean_10, primals_212, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = mean_10 = primals_212 = None
    getitem_138: "f32[8, 1536, 1, 1]" = convolution_backward_8[0]
    getitem_139: "f32[384, 1536, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 1536, 7, 7]" = torch.ops.aten.expand.default(getitem_138, [8, 1536, 7, 7]);  getitem_138 = None
    div_2: "f32[8, 1536, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_76: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_346, div_2);  mul_346 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_23: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_76, [0, 2, 3])
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(add_76, mul_240, view_155, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_76 = mul_240 = view_155 = None
    getitem_141: "f32[8, 384, 7, 7]" = convolution_backward_9[0]
    getitem_142: "f32[1536, 384, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_189: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(getitem_142, [1, 1536, 384]);  getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_24: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_189, [0, 2])
    sub_84: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_153, unsqueeze_98);  view_153 = unsqueeze_98 = None
    mul_349: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_189, sub_84)
    sum_25: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_349, [0, 2]);  mul_349 = None
    mul_350: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_24, 0.0026041666666666665);  sum_24 = None
    unsqueeze_99: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_350, 0);  mul_350 = None
    unsqueeze_100: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None
    mul_351: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_25, 0.0026041666666666665)
    mul_352: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_353: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    unsqueeze_101: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
    unsqueeze_102: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 2);  unsqueeze_101 = None
    mul_354: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_103, view_154);  view_154 = None
    unsqueeze_103: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_354, 0);  mul_354 = None
    unsqueeze_104: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 2);  unsqueeze_103 = None
    mul_355: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_102);  sub_84 = unsqueeze_102 = None
    sub_86: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_189, mul_355);  view_189 = mul_355 = None
    sub_87: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_100);  sub_86 = unsqueeze_100 = None
    mul_356: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_104);  sub_87 = unsqueeze_104 = None
    mul_357: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_103);  sum_25 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_190: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_357, [1536, 1, 1, 1]);  mul_357 = None
    mul_358: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_190, 0.09125009274634042);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_191: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_356, [1536, 384, 1, 1]);  mul_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sub_88: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_56)
    mul_359: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_70, sub_88);  convolution_70 = sub_88 = None
    add_77: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_359, 1);  mul_359 = None
    mul_360: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_56, add_77);  sigmoid_56 = add_77 = None
    mul_361: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_141, mul_360);  getitem_141 = mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_26: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 2, 3])
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_361, mul_236, view_152, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_361 = mul_236 = view_152 = None
    getitem_144: "f32[8, 384, 7, 7]" = convolution_backward_10[0]
    getitem_145: "f32[384, 64, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_192: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_145, [1, 384, 576]);  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_27: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_192, [0, 2])
    sub_89: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_150, unsqueeze_106);  view_150 = unsqueeze_106 = None
    mul_362: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_192, sub_89)
    sum_28: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 2]);  mul_362 = None
    mul_363: "f32[384]" = torch.ops.aten.mul.Tensor(sum_27, 0.001736111111111111);  sum_27 = None
    unsqueeze_107: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_363, 0);  mul_363 = None
    unsqueeze_108: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 2);  unsqueeze_107 = None
    mul_364: "f32[384]" = torch.ops.aten.mul.Tensor(sum_28, 0.001736111111111111)
    mul_365: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_101, squeeze_101)
    mul_366: "f32[384]" = torch.ops.aten.mul.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
    unsqueeze_109: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_366, 0);  mul_366 = None
    unsqueeze_110: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 2);  unsqueeze_109 = None
    mul_367: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_101, view_151);  view_151 = None
    unsqueeze_111: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_367, 0);  mul_367 = None
    unsqueeze_112: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
    mul_368: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_110);  sub_89 = unsqueeze_110 = None
    sub_91: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_192, mul_368);  view_192 = mul_368 = None
    sub_92: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_108);  sub_91 = unsqueeze_108 = None
    mul_369: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_112);  sub_92 = unsqueeze_112 = None
    mul_370: "f32[384]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_101);  sum_28 = squeeze_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_193: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_370, [384, 1, 1, 1]);  mul_370 = None
    mul_371: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_193, 0.07450538873672485);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_194: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_369, [384, 64, 3, 3]);  mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_70: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_69)
    sub_93: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_70)
    mul_372: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_69, sub_93);  convolution_69 = sub_93 = None
    add_78: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_372, 1);  mul_372 = None
    mul_373: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_70, add_78);  sigmoid_70 = add_78 = None
    mul_374: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_144, mul_373);  getitem_144 = mul_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_29: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 2, 3])
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_374, mul_232, view_149, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_374 = mul_232 = view_149 = None
    getitem_147: "f32[8, 384, 7, 7]" = convolution_backward_11[0]
    getitem_148: "f32[384, 64, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_195: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_148, [1, 384, 576]);  getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_30: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_195, [0, 2])
    sub_94: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_147, unsqueeze_114);  view_147 = unsqueeze_114 = None
    mul_375: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_195, sub_94)
    sum_31: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 2]);  mul_375 = None
    mul_376: "f32[384]" = torch.ops.aten.mul.Tensor(sum_30, 0.001736111111111111);  sum_30 = None
    unsqueeze_115: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
    unsqueeze_116: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 2);  unsqueeze_115 = None
    mul_377: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, 0.001736111111111111)
    mul_378: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_99, squeeze_99)
    mul_379: "f32[384]" = torch.ops.aten.mul.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
    unsqueeze_117: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_379, 0);  mul_379 = None
    unsqueeze_118: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
    mul_380: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_99, view_148);  view_148 = None
    unsqueeze_119: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_380, 0);  mul_380 = None
    unsqueeze_120: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
    mul_381: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_118);  sub_94 = unsqueeze_118 = None
    sub_96: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_195, mul_381);  view_195 = mul_381 = None
    sub_97: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_116);  sub_96 = unsqueeze_116 = None
    mul_382: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_120);  sub_97 = unsqueeze_120 = None
    mul_383: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_99);  sum_31 = squeeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_196: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_383, [384, 1, 1, 1]);  mul_383 = None
    mul_384: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_196, 0.07450538873672485);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_197: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_382, [384, 64, 3, 3]);  mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_71: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_68)
    sub_98: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_71)
    mul_385: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_68, sub_98);  convolution_68 = sub_98 = None
    add_79: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_385, 1);  mul_385 = None
    mul_386: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_71, add_79);  sigmoid_71 = add_79 = None
    mul_387: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_147, mul_386);  getitem_147 = mul_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_32: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_387, mul_228, view_146, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_387 = mul_228 = view_146 = None
    getitem_150: "f32[8, 1536, 7, 7]" = convolution_backward_12[0]
    getitem_151: "f32[384, 1536, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_198: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(getitem_151, [1, 384, 1536]);  getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_33: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_198, [0, 2])
    sub_99: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_144, unsqueeze_122);  view_144 = unsqueeze_122 = None
    mul_388: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_198, sub_99)
    sum_34: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 2]);  mul_388 = None
    mul_389: "f32[384]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006510416666666666);  sum_33 = None
    unsqueeze_123: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_389, 0);  mul_389 = None
    unsqueeze_124: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
    mul_390: "f32[384]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006510416666666666)
    mul_391: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_392: "f32[384]" = torch.ops.aten.mul.Tensor(mul_390, mul_391);  mul_390 = mul_391 = None
    unsqueeze_125: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_392, 0);  mul_392 = None
    unsqueeze_126: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 2);  unsqueeze_125 = None
    mul_393: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_97, view_145);  view_145 = None
    unsqueeze_127: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_393, 0);  mul_393 = None
    unsqueeze_128: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
    mul_394: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_126);  sub_99 = unsqueeze_126 = None
    sub_101: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_198, mul_394);  view_198 = mul_394 = None
    sub_102: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_124);  sub_101 = unsqueeze_124 = None
    mul_395: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_128);  sub_102 = unsqueeze_128 = None
    mul_396: "f32[384]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_97);  sum_34 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_199: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_396, [384, 1, 1, 1]);  mul_396 = None
    mul_397: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_199, 0.04562504637317021);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_200: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_395, [384, 1536, 1, 1]);  mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_398: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_150, 0.9805806756909201);  getitem_150 = None
    mul_401: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_398, mul_400);  mul_398 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_81: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(add_75, mul_401);  add_75 = mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_402: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_81, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_403: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_402, 2.0);  mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_404: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_403, convolution_65);  convolution_65 = None
    mul_405: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_403, sigmoid_52);  mul_403 = None
    sum_35: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2, 3], True);  mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_104: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_52)
    mul_406: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_52, sub_104);  sigmoid_52 = sub_104 = None
    mul_407: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_35, mul_406);  sum_35 = mul_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_36: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_407, relu_9, primals_210, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_407 = primals_210 = None
    getitem_153: "f32[8, 384, 1, 1]" = convolution_backward_13[0]
    getitem_154: "f32[1536, 384, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_2: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_2: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_2, full_default_1, getitem_153);  le_2 = getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_37: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_2, mean_9, primals_208, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = mean_9 = primals_208 = None
    getitem_156: "f32[8, 1536, 1, 1]" = convolution_backward_14[0]
    getitem_157: "f32[384, 1536, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 1536, 7, 7]" = torch.ops.aten.expand.default(getitem_156, [8, 1536, 7, 7]);  getitem_156 = None
    div_3: "f32[8, 1536, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_82: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_405, div_3);  mul_405 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_38: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_82, [0, 2, 3])
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(add_82, mul_220, view_143, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_82 = mul_220 = view_143 = None
    getitem_159: "f32[8, 384, 7, 7]" = convolution_backward_15[0]
    getitem_160: "f32[1536, 384, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_201: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(getitem_160, [1, 1536, 384]);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_39: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_201, [0, 2])
    sub_105: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_141, unsqueeze_130);  view_141 = unsqueeze_130 = None
    mul_408: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_201, sub_105)
    sum_40: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 2]);  mul_408 = None
    mul_409: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_39, 0.0026041666666666665);  sum_39 = None
    unsqueeze_131: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
    unsqueeze_132: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
    mul_410: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_40, 0.0026041666666666665)
    mul_411: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_95, squeeze_95)
    mul_412: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    unsqueeze_133: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
    unsqueeze_134: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
    mul_413: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_95, view_142);  view_142 = None
    unsqueeze_135: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
    unsqueeze_136: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
    mul_414: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_134);  sub_105 = unsqueeze_134 = None
    sub_107: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_201, mul_414);  view_201 = mul_414 = None
    sub_108: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_132);  sub_107 = unsqueeze_132 = None
    mul_415: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_136);  sub_108 = unsqueeze_136 = None
    mul_416: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_95);  sum_40 = squeeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_202: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_416, [1536, 1, 1, 1]);  mul_416 = None
    mul_417: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_202, 0.09125009274634042);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_203: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_415, [1536, 384, 1, 1]);  mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sub_109: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_51)
    mul_418: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_64, sub_109);  convolution_64 = sub_109 = None
    add_83: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_418, 1);  mul_418 = None
    mul_419: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_51, add_83);  sigmoid_51 = add_83 = None
    mul_420: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_159, mul_419);  getitem_159 = mul_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_41: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 2, 3])
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_420, mul_216, view_140, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_420 = mul_216 = view_140 = None
    getitem_162: "f32[8, 384, 7, 7]" = convolution_backward_16[0]
    getitem_163: "f32[384, 64, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_204: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_163, [1, 384, 576]);  getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_42: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_204, [0, 2])
    sub_110: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_138, unsqueeze_138);  view_138 = unsqueeze_138 = None
    mul_421: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_204, sub_110)
    sum_43: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 2]);  mul_421 = None
    mul_422: "f32[384]" = torch.ops.aten.mul.Tensor(sum_42, 0.001736111111111111);  sum_42 = None
    unsqueeze_139: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_140: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
    mul_423: "f32[384]" = torch.ops.aten.mul.Tensor(sum_43, 0.001736111111111111)
    mul_424: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_93, squeeze_93)
    mul_425: "f32[384]" = torch.ops.aten.mul.Tensor(mul_423, mul_424);  mul_423 = mul_424 = None
    unsqueeze_141: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_425, 0);  mul_425 = None
    unsqueeze_142: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None
    mul_426: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_93, view_139);  view_139 = None
    unsqueeze_143: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_426, 0);  mul_426 = None
    unsqueeze_144: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
    mul_427: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_142);  sub_110 = unsqueeze_142 = None
    sub_112: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_204, mul_427);  view_204 = mul_427 = None
    sub_113: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_112, unsqueeze_140);  sub_112 = unsqueeze_140 = None
    mul_428: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_144);  sub_113 = unsqueeze_144 = None
    mul_429: "f32[384]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_93);  sum_43 = squeeze_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_205: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_429, [384, 1, 1, 1]);  mul_429 = None
    mul_430: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_205, 0.07450538873672485);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_206: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_428, [384, 64, 3, 3]);  mul_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_74: "f32[8, 384, 7, 7]" = torch.ops.aten.sigmoid.default(convolution_63)
    sub_114: "f32[8, 384, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_74);  full_default_2 = None
    mul_431: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_63, sub_114);  convolution_63 = sub_114 = None
    add_84: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Scalar(mul_431, 1);  mul_431 = None
    mul_432: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_74, add_84);  sigmoid_74 = add_84 = None
    mul_433: "f32[8, 384, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_162, mul_432);  getitem_162 = mul_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_44: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_433, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_433, mul_212, view_137, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_433 = mul_212 = view_137 = None
    getitem_165: "f32[8, 384, 14, 14]" = convolution_backward_17[0]
    getitem_166: "f32[384, 64, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_207: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_166, [1, 384, 576]);  getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_45: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_207, [0, 2])
    sub_115: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_135, unsqueeze_146);  view_135 = unsqueeze_146 = None
    mul_434: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_207, sub_115)
    sum_46: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_434, [0, 2]);  mul_434 = None
    mul_435: "f32[384]" = torch.ops.aten.mul.Tensor(sum_45, 0.001736111111111111);  sum_45 = None
    unsqueeze_147: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
    unsqueeze_148: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 2);  unsqueeze_147 = None
    mul_436: "f32[384]" = torch.ops.aten.mul.Tensor(sum_46, 0.001736111111111111)
    mul_437: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_438: "f32[384]" = torch.ops.aten.mul.Tensor(mul_436, mul_437);  mul_436 = mul_437 = None
    unsqueeze_149: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_438, 0);  mul_438 = None
    unsqueeze_150: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
    mul_439: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_91, view_136);  view_136 = None
    unsqueeze_151: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_439, 0);  mul_439 = None
    unsqueeze_152: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
    mul_440: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_150);  sub_115 = unsqueeze_150 = None
    sub_117: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_207, mul_440);  view_207 = mul_440 = None
    sub_118: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_117, unsqueeze_148);  sub_117 = unsqueeze_148 = None
    mul_441: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_152);  sub_118 = unsqueeze_152 = None
    mul_442: "f32[384]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_91);  sum_46 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_208: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_442, [384, 1, 1, 1]);  mul_442 = None
    mul_443: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_208, 0.07450538873672485);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_209: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_441, [384, 64, 3, 3]);  mul_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_75: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_62)
    full_default_14: "f32[8, 384, 14, 14]" = torch.ops.aten.full.default([8, 384, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_119: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_75)
    mul_444: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_62, sub_119);  convolution_62 = sub_119 = None
    add_85: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_444, 1);  mul_444 = None
    mul_445: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_75, add_85);  sigmoid_75 = add_85 = None
    mul_446: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_165, mul_445);  getitem_165 = mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_47: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_446, mul_205, view_134, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_446 = view_134 = None
    getitem_168: "f32[8, 1536, 14, 14]" = convolution_backward_18[0]
    getitem_169: "f32[384, 1536, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_210: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(getitem_169, [1, 384, 1536]);  getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_48: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_210, [0, 2])
    sub_120: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_132, unsqueeze_154);  view_132 = unsqueeze_154 = None
    mul_447: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_210, sub_120)
    sum_49: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 2]);  mul_447 = None
    mul_448: "f32[384]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006510416666666666);  sum_48 = None
    unsqueeze_155: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
    unsqueeze_156: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
    mul_449: "f32[384]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006510416666666666)
    mul_450: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_89, squeeze_89)
    mul_451: "f32[384]" = torch.ops.aten.mul.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    unsqueeze_157: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
    unsqueeze_158: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
    mul_452: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_89, view_133);  view_133 = None
    unsqueeze_159: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_160: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
    mul_453: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_158);  sub_120 = unsqueeze_158 = None
    sub_122: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_210, mul_453);  view_210 = mul_453 = None
    sub_123: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_156);  sub_122 = unsqueeze_156 = None
    mul_454: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_160);  sub_123 = unsqueeze_160 = None
    mul_455: "f32[384]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_89);  sum_49 = squeeze_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_211: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_455, [384, 1, 1, 1]);  mul_455 = None
    mul_456: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_211, 0.04562504637317021);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_212: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_454, [384, 1536, 1, 1]);  mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_50: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_81, [0, 2, 3])
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(add_81, avg_pool2d_2, view_131, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_81 = avg_pool2d_2 = view_131 = None
    getitem_171: "f32[8, 1536, 7, 7]" = convolution_backward_19[0]
    getitem_172: "f32[1536, 1536, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_213: "f32[1, 1536, 1536]" = torch.ops.aten.reshape.default(getitem_172, [1, 1536, 1536]);  getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_51: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_213, [0, 2])
    sub_124: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_129, unsqueeze_162);  view_129 = unsqueeze_162 = None
    mul_457: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(view_213, sub_124)
    sum_52: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 2]);  mul_457 = None
    mul_458: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006510416666666666);  sum_51 = None
    unsqueeze_163: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_164: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
    mul_459: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006510416666666666)
    mul_460: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_87, squeeze_87)
    mul_461: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
    unsqueeze_165: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_461, 0);  mul_461 = None
    unsqueeze_166: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
    mul_462: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_87, view_130);  view_130 = None
    unsqueeze_167: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_168: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
    mul_463: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_166);  sub_124 = unsqueeze_166 = None
    sub_126: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_213, mul_463);  view_213 = mul_463 = None
    sub_127: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_164);  sub_126 = unsqueeze_164 = None
    mul_464: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_168);  sub_127 = unsqueeze_168 = None
    mul_465: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_87);  sum_52 = squeeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_214: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_465, [1536, 1, 1, 1]);  mul_465 = None
    mul_466: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_214, 0.04562504637317021);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_215: "f32[1536, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_464, [1536, 1536, 1, 1]);  mul_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_backward: "f32[8, 1536, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(getitem_171, mul_205, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_171 = mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    add_86: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(getitem_168, avg_pool2d_backward);  getitem_168 = avg_pool2d_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_467: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_86, 0.8980265101338745);  add_86 = None
    mul_470: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_467, mul_469);  mul_467 = mul_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_471: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_470, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_472: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_471, 2.0);  mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_473: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_472, convolution_58);  convolution_58 = None
    mul_474: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_472, sigmoid_47);  mul_472 = None
    sum_53: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2, 3], True);  mul_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_129: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_47)
    mul_475: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_47, sub_129);  sigmoid_47 = sub_129 = None
    mul_476: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_53, mul_475);  sum_53 = mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_54: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 2, 3])
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_476, relu_8, primals_206, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_476 = primals_206 = None
    getitem_174: "f32[8, 384, 1, 1]" = convolution_backward_20[0]
    getitem_175: "f32[1536, 384, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_3: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_3: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_3, full_default_1, getitem_174);  le_3 = getitem_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_55: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(where_3, mean_8, primals_204, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = mean_8 = primals_204 = None
    getitem_177: "f32[8, 1536, 1, 1]" = convolution_backward_21[0]
    getitem_178: "f32[384, 1536, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 1536, 14, 14]" = torch.ops.aten.expand.default(getitem_177, [8, 1536, 14, 14]);  getitem_177 = None
    div_4: "f32[8, 1536, 14, 14]" = torch.ops.aten.div.Scalar(expand_4, 196);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_88: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_474, div_4);  mul_474 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_56: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_88, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(add_88, mul_197, view_128, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_88 = mul_197 = view_128 = None
    getitem_180: "f32[8, 384, 14, 14]" = convolution_backward_22[0]
    getitem_181: "f32[1536, 384, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_216: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(getitem_181, [1, 1536, 384]);  getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_57: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_216, [0, 2])
    sub_130: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_126, unsqueeze_170);  view_126 = unsqueeze_170 = None
    mul_477: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_216, sub_130)
    sum_58: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_477, [0, 2]);  mul_477 = None
    mul_478: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_57, 0.0026041666666666665);  sum_57 = None
    unsqueeze_171: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_478, 0);  mul_478 = None
    unsqueeze_172: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
    mul_479: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_58, 0.0026041666666666665)
    mul_480: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_481: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_479, mul_480);  mul_479 = mul_480 = None
    unsqueeze_173: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
    unsqueeze_174: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
    mul_482: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_85, view_127);  view_127 = None
    unsqueeze_175: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_482, 0);  mul_482 = None
    unsqueeze_176: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
    mul_483: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_174);  sub_130 = unsqueeze_174 = None
    sub_132: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_216, mul_483);  view_216 = mul_483 = None
    sub_133: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_132, unsqueeze_172);  sub_132 = unsqueeze_172 = None
    mul_484: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_176);  sub_133 = unsqueeze_176 = None
    mul_485: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_58, squeeze_85);  sum_58 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_217: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_485, [1536, 1, 1, 1]);  mul_485 = None
    mul_486: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_217, 0.09125009274634042);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_218: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_484, [1536, 384, 1, 1]);  mul_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sub_134: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_46)
    mul_487: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_57, sub_134);  convolution_57 = sub_134 = None
    add_89: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_487, 1);  mul_487 = None
    mul_488: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_46, add_89);  sigmoid_46 = add_89 = None
    mul_489: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_180, mul_488);  getitem_180 = mul_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_59: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_489, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_489, mul_193, view_125, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_489 = mul_193 = view_125 = None
    getitem_183: "f32[8, 384, 14, 14]" = convolution_backward_23[0]
    getitem_184: "f32[384, 64, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_219: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_184, [1, 384, 576]);  getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_60: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_219, [0, 2])
    sub_135: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_123, unsqueeze_178);  view_123 = unsqueeze_178 = None
    mul_490: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_219, sub_135)
    sum_61: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2]);  mul_490 = None
    mul_491: "f32[384]" = torch.ops.aten.mul.Tensor(sum_60, 0.001736111111111111);  sum_60 = None
    unsqueeze_179: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_491, 0);  mul_491 = None
    unsqueeze_180: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    mul_492: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, 0.001736111111111111)
    mul_493: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_83, squeeze_83)
    mul_494: "f32[384]" = torch.ops.aten.mul.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
    unsqueeze_181: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    unsqueeze_182: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
    mul_495: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_83, view_124);  view_124 = None
    unsqueeze_183: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_184: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
    mul_496: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_182);  sub_135 = unsqueeze_182 = None
    sub_137: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_219, mul_496);  view_219 = mul_496 = None
    sub_138: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_137, unsqueeze_180);  sub_137 = unsqueeze_180 = None
    mul_497: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_184);  sub_138 = unsqueeze_184 = None
    mul_498: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_83);  sum_61 = squeeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_220: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_498, [384, 1, 1, 1]);  mul_498 = None
    mul_499: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_220, 0.07450538873672485);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_221: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_497, [384, 64, 3, 3]);  mul_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_78: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_56)
    sub_139: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_78)
    mul_500: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_56, sub_139);  convolution_56 = sub_139 = None
    add_90: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_500, 1);  mul_500 = None
    mul_501: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_78, add_90);  sigmoid_78 = add_90 = None
    mul_502: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_183, mul_501);  getitem_183 = mul_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_62: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_502, [0, 2, 3])
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_502, mul_189, view_122, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_502 = mul_189 = view_122 = None
    getitem_186: "f32[8, 384, 14, 14]" = convolution_backward_24[0]
    getitem_187: "f32[384, 64, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_222: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_187, [1, 384, 576]);  getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_63: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_222, [0, 2])
    sub_140: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_120, unsqueeze_186);  view_120 = unsqueeze_186 = None
    mul_503: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_222, sub_140)
    sum_64: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 2]);  mul_503 = None
    mul_504: "f32[384]" = torch.ops.aten.mul.Tensor(sum_63, 0.001736111111111111);  sum_63 = None
    unsqueeze_187: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_188: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
    mul_505: "f32[384]" = torch.ops.aten.mul.Tensor(sum_64, 0.001736111111111111)
    mul_506: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_81, squeeze_81)
    mul_507: "f32[384]" = torch.ops.aten.mul.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    unsqueeze_189: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
    unsqueeze_190: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
    mul_508: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_81, view_121);  view_121 = None
    unsqueeze_191: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_192: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    mul_509: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_190);  sub_140 = unsqueeze_190 = None
    sub_142: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_222, mul_509);  view_222 = mul_509 = None
    sub_143: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_188);  sub_142 = unsqueeze_188 = None
    mul_510: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_192);  sub_143 = unsqueeze_192 = None
    mul_511: "f32[384]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_81);  sum_64 = squeeze_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_223: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_511, [384, 1, 1, 1]);  mul_511 = None
    mul_512: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_223, 0.07450538873672485);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_224: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_510, [384, 64, 3, 3]);  mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_79: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_55)
    sub_144: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_79)
    mul_513: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_55, sub_144);  convolution_55 = sub_144 = None
    add_91: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_513, 1);  mul_513 = None
    mul_514: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_79, add_91);  sigmoid_79 = add_91 = None
    mul_515: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_186, mul_514);  getitem_186 = mul_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_65: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_515, [0, 2, 3])
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_515, mul_185, view_119, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_515 = mul_185 = view_119 = None
    getitem_189: "f32[8, 1536, 14, 14]" = convolution_backward_25[0]
    getitem_190: "f32[384, 1536, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_225: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(getitem_190, [1, 384, 1536]);  getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_66: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_225, [0, 2])
    sub_145: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_117, unsqueeze_194);  view_117 = unsqueeze_194 = None
    mul_516: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_225, sub_145)
    sum_67: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 2]);  mul_516 = None
    mul_517: "f32[384]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006510416666666666);  sum_66 = None
    unsqueeze_195: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
    unsqueeze_196: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
    mul_518: "f32[384]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006510416666666666)
    mul_519: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_520: "f32[384]" = torch.ops.aten.mul.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
    unsqueeze_197: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_198: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    mul_521: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_79, view_118);  view_118 = None
    unsqueeze_199: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
    unsqueeze_200: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
    mul_522: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_198);  sub_145 = unsqueeze_198 = None
    sub_147: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_225, mul_522);  view_225 = mul_522 = None
    sub_148: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_196);  sub_147 = unsqueeze_196 = None
    mul_523: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_200);  sub_148 = unsqueeze_200 = None
    mul_524: "f32[384]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_79);  sum_67 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_226: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_524, [384, 1, 1, 1]);  mul_524 = None
    mul_525: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_226, 0.04562504637317021);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_227: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_523, [384, 1536, 1, 1]);  mul_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_526: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_189, 0.9128709291752768);  getitem_189 = None
    mul_529: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_526, mul_528);  mul_526 = mul_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_93: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_470, mul_529);  mul_470 = mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_530: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_93, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_531: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_530, 2.0);  mul_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_532: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_531, convolution_52);  convolution_52 = None
    mul_533: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_531, sigmoid_42);  mul_531 = None
    sum_68: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_532, [2, 3], True);  mul_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_150: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_42)
    mul_534: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_42, sub_150);  sigmoid_42 = sub_150 = None
    mul_535: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_68, mul_534);  sum_68 = mul_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_69: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3])
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_535, relu_7, primals_202, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_535 = primals_202 = None
    getitem_192: "f32[8, 384, 1, 1]" = convolution_backward_26[0]
    getitem_193: "f32[1536, 384, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_4: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_4: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_4, full_default_1, getitem_192);  le_4 = getitem_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_70: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_4, mean_7, primals_200, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = mean_7 = primals_200 = None
    getitem_195: "f32[8, 1536, 1, 1]" = convolution_backward_27[0]
    getitem_196: "f32[384, 1536, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[8, 1536, 14, 14]" = torch.ops.aten.expand.default(getitem_195, [8, 1536, 14, 14]);  getitem_195 = None
    div_5: "f32[8, 1536, 14, 14]" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_94: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_533, div_5);  mul_533 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_71: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_94, [0, 2, 3])
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(add_94, mul_177, view_116, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_94 = mul_177 = view_116 = None
    getitem_198: "f32[8, 384, 14, 14]" = convolution_backward_28[0]
    getitem_199: "f32[1536, 384, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_228: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(getitem_199, [1, 1536, 384]);  getitem_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_72: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_228, [0, 2])
    sub_151: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_114, unsqueeze_202);  view_114 = unsqueeze_202 = None
    mul_536: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_228, sub_151)
    sum_73: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 2]);  mul_536 = None
    mul_537: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_72, 0.0026041666666666665);  sum_72 = None
    unsqueeze_203: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
    unsqueeze_204: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    mul_538: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_73, 0.0026041666666666665)
    mul_539: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_77, squeeze_77)
    mul_540: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_538, mul_539);  mul_538 = mul_539 = None
    unsqueeze_205: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_206: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    mul_541: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_77, view_115);  view_115 = None
    unsqueeze_207: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
    unsqueeze_208: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
    mul_542: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_206);  sub_151 = unsqueeze_206 = None
    sub_153: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_228, mul_542);  view_228 = mul_542 = None
    sub_154: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_153, unsqueeze_204);  sub_153 = unsqueeze_204 = None
    mul_543: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_208);  sub_154 = unsqueeze_208 = None
    mul_544: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_77);  sum_73 = squeeze_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_229: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_544, [1536, 1, 1, 1]);  mul_544 = None
    mul_545: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_229, 0.09125009274634042);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_230: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_543, [1536, 384, 1, 1]);  mul_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sub_155: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_41)
    mul_546: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_51, sub_155);  convolution_51 = sub_155 = None
    add_95: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_546, 1);  mul_546 = None
    mul_547: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_41, add_95);  sigmoid_41 = add_95 = None
    mul_548: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_198, mul_547);  getitem_198 = mul_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_74: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_548, [0, 2, 3])
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_548, mul_173, view_113, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_548 = mul_173 = view_113 = None
    getitem_201: "f32[8, 384, 14, 14]" = convolution_backward_29[0]
    getitem_202: "f32[384, 64, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_231: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_202, [1, 384, 576]);  getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_75: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_231, [0, 2])
    sub_156: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_111, unsqueeze_210);  view_111 = unsqueeze_210 = None
    mul_549: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_231, sub_156)
    sum_76: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_549, [0, 2]);  mul_549 = None
    mul_550: "f32[384]" = torch.ops.aten.mul.Tensor(sum_75, 0.001736111111111111);  sum_75 = None
    unsqueeze_211: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_550, 0);  mul_550 = None
    unsqueeze_212: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    mul_551: "f32[384]" = torch.ops.aten.mul.Tensor(sum_76, 0.001736111111111111)
    mul_552: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_75, squeeze_75)
    mul_553: "f32[384]" = torch.ops.aten.mul.Tensor(mul_551, mul_552);  mul_551 = mul_552 = None
    unsqueeze_213: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
    unsqueeze_214: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
    mul_554: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_75, view_112);  view_112 = None
    unsqueeze_215: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
    unsqueeze_216: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
    mul_555: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_214);  sub_156 = unsqueeze_214 = None
    sub_158: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_231, mul_555);  view_231 = mul_555 = None
    sub_159: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_212);  sub_158 = unsqueeze_212 = None
    mul_556: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_216);  sub_159 = unsqueeze_216 = None
    mul_557: "f32[384]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_75);  sum_76 = squeeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_232: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_557, [384, 1, 1, 1]);  mul_557 = None
    mul_558: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_232, 0.07450538873672485);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_233: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_556, [384, 64, 3, 3]);  mul_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_82: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_50)
    sub_160: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_82)
    mul_559: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_50, sub_160);  convolution_50 = sub_160 = None
    add_96: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_559, 1);  mul_559 = None
    mul_560: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_82, add_96);  sigmoid_82 = add_96 = None
    mul_561: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_201, mul_560);  getitem_201 = mul_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_77: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_561, [0, 2, 3])
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_561, mul_169, view_110, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_561 = mul_169 = view_110 = None
    getitem_204: "f32[8, 384, 14, 14]" = convolution_backward_30[0]
    getitem_205: "f32[384, 64, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_234: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_205, [1, 384, 576]);  getitem_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_78: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_234, [0, 2])
    sub_161: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_108, unsqueeze_218);  view_108 = unsqueeze_218 = None
    mul_562: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_234, sub_161)
    sum_79: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 2]);  mul_562 = None
    mul_563: "f32[384]" = torch.ops.aten.mul.Tensor(sum_78, 0.001736111111111111);  sum_78 = None
    unsqueeze_219: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    unsqueeze_220: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
    mul_564: "f32[384]" = torch.ops.aten.mul.Tensor(sum_79, 0.001736111111111111)
    mul_565: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_566: "f32[384]" = torch.ops.aten.mul.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_221: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
    unsqueeze_222: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    mul_567: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_73, view_109);  view_109 = None
    unsqueeze_223: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_224: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    mul_568: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_222);  sub_161 = unsqueeze_222 = None
    sub_163: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_234, mul_568);  view_234 = mul_568 = None
    sub_164: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_220);  sub_163 = unsqueeze_220 = None
    mul_569: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_224);  sub_164 = unsqueeze_224 = None
    mul_570: "f32[384]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_73);  sum_79 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_235: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_570, [384, 1, 1, 1]);  mul_570 = None
    mul_571: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_235, 0.07450538873672485);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_236: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_569, [384, 64, 3, 3]);  mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_83: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_49)
    sub_165: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_83)
    mul_572: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_49, sub_165);  convolution_49 = sub_165 = None
    add_97: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_572, 1);  mul_572 = None
    mul_573: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_83, add_97);  sigmoid_83 = add_97 = None
    mul_574: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_204, mul_573);  getitem_204 = mul_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_80: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_574, [0, 2, 3])
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_574, mul_165, view_107, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_574 = mul_165 = view_107 = None
    getitem_207: "f32[8, 1536, 14, 14]" = convolution_backward_31[0]
    getitem_208: "f32[384, 1536, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_237: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(getitem_208, [1, 384, 1536]);  getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_81: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_237, [0, 2])
    sub_166: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_105, unsqueeze_226);  view_105 = unsqueeze_226 = None
    mul_575: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_237, sub_166)
    sum_82: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 2]);  mul_575 = None
    mul_576: "f32[384]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006510416666666666);  sum_81 = None
    unsqueeze_227: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_228: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    mul_577: "f32[384]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006510416666666666)
    mul_578: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_71, squeeze_71)
    mul_579: "f32[384]" = torch.ops.aten.mul.Tensor(mul_577, mul_578);  mul_577 = mul_578 = None
    unsqueeze_229: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_230: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    mul_580: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_71, view_106);  view_106 = None
    unsqueeze_231: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
    unsqueeze_232: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
    mul_581: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_230);  sub_166 = unsqueeze_230 = None
    sub_168: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_237, mul_581);  view_237 = mul_581 = None
    sub_169: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_228);  sub_168 = unsqueeze_228 = None
    mul_582: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_232);  sub_169 = unsqueeze_232 = None
    mul_583: "f32[384]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_71);  sum_82 = squeeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_238: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_583, [384, 1, 1, 1]);  mul_583 = None
    mul_584: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_238, 0.04562504637317021);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_239: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_582, [384, 1536, 1, 1]);  mul_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_585: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_207, 0.9284766908852592);  getitem_207 = None
    mul_588: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_585, mul_587);  mul_585 = mul_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_99: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(add_93, mul_588);  add_93 = mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_589: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_99, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_590: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_589, 2.0);  mul_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_591: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_590, convolution_46);  convolution_46 = None
    mul_592: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_590, sigmoid_37);  mul_590 = None
    sum_83: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_591, [2, 3], True);  mul_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_171: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_37)
    mul_593: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_37, sub_171);  sigmoid_37 = sub_171 = None
    mul_594: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_83, mul_593);  sum_83 = mul_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_84: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_594, [0, 2, 3])
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_594, relu_6, primals_198, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_594 = primals_198 = None
    getitem_210: "f32[8, 384, 1, 1]" = convolution_backward_32[0]
    getitem_211: "f32[1536, 384, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_5: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_5: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_5, full_default_1, getitem_210);  le_5 = getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_85: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_5, mean_6, primals_196, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_6 = primals_196 = None
    getitem_213: "f32[8, 1536, 1, 1]" = convolution_backward_33[0]
    getitem_214: "f32[384, 1536, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[8, 1536, 14, 14]" = torch.ops.aten.expand.default(getitem_213, [8, 1536, 14, 14]);  getitem_213 = None
    div_6: "f32[8, 1536, 14, 14]" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_100: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_592, div_6);  mul_592 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_86: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_100, [0, 2, 3])
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(add_100, mul_157, view_104, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_100 = mul_157 = view_104 = None
    getitem_216: "f32[8, 384, 14, 14]" = convolution_backward_34[0]
    getitem_217: "f32[1536, 384, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_240: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(getitem_217, [1, 1536, 384]);  getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_87: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_240, [0, 2])
    sub_172: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_102, unsqueeze_234);  view_102 = unsqueeze_234 = None
    mul_595: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_240, sub_172)
    sum_88: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 2]);  mul_595 = None
    mul_596: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_87, 0.0026041666666666665);  sum_87 = None
    unsqueeze_235: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_596, 0);  mul_596 = None
    unsqueeze_236: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    mul_597: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_88, 0.0026041666666666665)
    mul_598: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_69, squeeze_69)
    mul_599: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_597, mul_598);  mul_597 = mul_598 = None
    unsqueeze_237: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_599, 0);  mul_599 = None
    unsqueeze_238: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
    mul_600: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_69, view_103);  view_103 = None
    unsqueeze_239: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_240: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    mul_601: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_238);  sub_172 = unsqueeze_238 = None
    sub_174: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_240, mul_601);  view_240 = mul_601 = None
    sub_175: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_236);  sub_174 = unsqueeze_236 = None
    mul_602: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_240);  sub_175 = unsqueeze_240 = None
    mul_603: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_88, squeeze_69);  sum_88 = squeeze_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_241: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_603, [1536, 1, 1, 1]);  mul_603 = None
    mul_604: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_241, 0.09125009274634042);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_242: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_602, [1536, 384, 1, 1]);  mul_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sub_176: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_36)
    mul_605: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_45, sub_176);  convolution_45 = sub_176 = None
    add_101: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_605, 1);  mul_605 = None
    mul_606: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_36, add_101);  sigmoid_36 = add_101 = None
    mul_607: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_216, mul_606);  getitem_216 = mul_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_89: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3])
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_607, mul_153, view_101, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_607 = mul_153 = view_101 = None
    getitem_219: "f32[8, 384, 14, 14]" = convolution_backward_35[0]
    getitem_220: "f32[384, 64, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_243: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_220, [1, 384, 576]);  getitem_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_90: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_243, [0, 2])
    sub_177: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_99, unsqueeze_242);  view_99 = unsqueeze_242 = None
    mul_608: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_243, sub_177)
    sum_91: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2]);  mul_608 = None
    mul_609: "f32[384]" = torch.ops.aten.mul.Tensor(sum_90, 0.001736111111111111);  sum_90 = None
    unsqueeze_243: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
    unsqueeze_244: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
    mul_610: "f32[384]" = torch.ops.aten.mul.Tensor(sum_91, 0.001736111111111111)
    mul_611: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_612: "f32[384]" = torch.ops.aten.mul.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
    unsqueeze_245: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_246: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    mul_613: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_67, view_100);  view_100 = None
    unsqueeze_247: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
    unsqueeze_248: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    mul_614: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_246);  sub_177 = unsqueeze_246 = None
    sub_179: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_243, mul_614);  view_243 = mul_614 = None
    sub_180: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_244);  sub_179 = unsqueeze_244 = None
    mul_615: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_248);  sub_180 = unsqueeze_248 = None
    mul_616: "f32[384]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_67);  sum_91 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_244: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_616, [384, 1, 1, 1]);  mul_616 = None
    mul_617: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_244, 0.07450538873672485);  view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_245: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_615, [384, 64, 3, 3]);  mul_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_86: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_44)
    sub_181: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_86)
    mul_618: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_44, sub_181);  convolution_44 = sub_181 = None
    add_102: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_618, 1);  mul_618 = None
    mul_619: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_86, add_102);  sigmoid_86 = add_102 = None
    mul_620: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_219, mul_619);  getitem_219 = mul_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_92: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 2, 3])
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_620, mul_149, view_98, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_620 = mul_149 = view_98 = None
    getitem_222: "f32[8, 384, 14, 14]" = convolution_backward_36[0]
    getitem_223: "f32[384, 64, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_246: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_223, [1, 384, 576]);  getitem_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_93: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_246, [0, 2])
    sub_182: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_96, unsqueeze_250);  view_96 = unsqueeze_250 = None
    mul_621: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_246, sub_182)
    sum_94: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_621, [0, 2]);  mul_621 = None
    mul_622: "f32[384]" = torch.ops.aten.mul.Tensor(sum_93, 0.001736111111111111);  sum_93 = None
    unsqueeze_251: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
    unsqueeze_252: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    mul_623: "f32[384]" = torch.ops.aten.mul.Tensor(sum_94, 0.001736111111111111)
    mul_624: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_65, squeeze_65)
    mul_625: "f32[384]" = torch.ops.aten.mul.Tensor(mul_623, mul_624);  mul_623 = mul_624 = None
    unsqueeze_253: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
    unsqueeze_254: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    mul_626: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_65, view_97);  view_97 = None
    unsqueeze_255: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_626, 0);  mul_626 = None
    unsqueeze_256: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
    mul_627: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_254);  sub_182 = unsqueeze_254 = None
    sub_184: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_246, mul_627);  view_246 = mul_627 = None
    sub_185: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_252);  sub_184 = unsqueeze_252 = None
    mul_628: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_256);  sub_185 = unsqueeze_256 = None
    mul_629: "f32[384]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_65);  sum_94 = squeeze_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_247: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_629, [384, 1, 1, 1]);  mul_629 = None
    mul_630: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_247, 0.07450538873672485);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_248: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_628, [384, 64, 3, 3]);  mul_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_87: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_43)
    sub_186: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_87)
    mul_631: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_43, sub_186);  convolution_43 = sub_186 = None
    add_103: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_631, 1);  mul_631 = None
    mul_632: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_87, add_103);  sigmoid_87 = add_103 = None
    mul_633: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_222, mul_632);  getitem_222 = mul_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_95: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_633, [0, 2, 3])
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_633, mul_145, view_95, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_633 = mul_145 = view_95 = None
    getitem_225: "f32[8, 1536, 14, 14]" = convolution_backward_37[0]
    getitem_226: "f32[384, 1536, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_249: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(getitem_226, [1, 384, 1536]);  getitem_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_96: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_249, [0, 2])
    sub_187: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_93, unsqueeze_258);  view_93 = unsqueeze_258 = None
    mul_634: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_249, sub_187)
    sum_97: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_634, [0, 2]);  mul_634 = None
    mul_635: "f32[384]" = torch.ops.aten.mul.Tensor(sum_96, 0.0006510416666666666);  sum_96 = None
    unsqueeze_259: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    unsqueeze_260: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    mul_636: "f32[384]" = torch.ops.aten.mul.Tensor(sum_97, 0.0006510416666666666)
    mul_637: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_63, squeeze_63)
    mul_638: "f32[384]" = torch.ops.aten.mul.Tensor(mul_636, mul_637);  mul_636 = mul_637 = None
    unsqueeze_261: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    unsqueeze_262: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
    mul_639: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_63, view_94);  view_94 = None
    unsqueeze_263: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    unsqueeze_264: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    mul_640: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_262);  sub_187 = unsqueeze_262 = None
    sub_189: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_249, mul_640);  view_249 = mul_640 = None
    sub_190: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_260);  sub_189 = unsqueeze_260 = None
    mul_641: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_264);  sub_190 = unsqueeze_264 = None
    mul_642: "f32[384]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_63);  sum_97 = squeeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_250: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_642, [384, 1, 1, 1]);  mul_642 = None
    mul_643: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_250, 0.04562504637317021);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_251: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_641, [384, 1536, 1, 1]);  mul_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_644: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_225, 0.9449111825230679);  getitem_225 = None
    mul_647: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_644, mul_646);  mul_644 = mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_105: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(add_99, mul_647);  add_99 = mul_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_648: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_105, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_649: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_648, 2.0);  mul_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_650: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_649, convolution_40);  convolution_40 = None
    mul_651: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_649, sigmoid_32);  mul_649 = None
    sum_98: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_650, [2, 3], True);  mul_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_192: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_32)
    mul_652: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_32, sub_192);  sigmoid_32 = sub_192 = None
    mul_653: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_98, mul_652);  sum_98 = mul_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_99: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_653, [0, 2, 3])
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_653, relu_5, primals_194, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_653 = primals_194 = None
    getitem_228: "f32[8, 384, 1, 1]" = convolution_backward_38[0]
    getitem_229: "f32[1536, 384, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_6: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_6: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_6, full_default_1, getitem_228);  le_6 = getitem_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_100: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(where_6, mean_5, primals_192, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = mean_5 = primals_192 = None
    getitem_231: "f32[8, 1536, 1, 1]" = convolution_backward_39[0]
    getitem_232: "f32[384, 1536, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[8, 1536, 14, 14]" = torch.ops.aten.expand.default(getitem_231, [8, 1536, 14, 14]);  getitem_231 = None
    div_7: "f32[8, 1536, 14, 14]" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_106: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_651, div_7);  mul_651 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_101: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_106, [0, 2, 3])
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(add_106, mul_137, view_92, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_106 = mul_137 = view_92 = None
    getitem_234: "f32[8, 384, 14, 14]" = convolution_backward_40[0]
    getitem_235: "f32[1536, 384, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_252: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(getitem_235, [1, 1536, 384]);  getitem_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_102: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_252, [0, 2])
    sub_193: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_90, unsqueeze_266);  view_90 = unsqueeze_266 = None
    mul_654: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_252, sub_193)
    sum_103: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_654, [0, 2]);  mul_654 = None
    mul_655: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_102, 0.0026041666666666665);  sum_102 = None
    unsqueeze_267: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
    unsqueeze_268: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
    mul_656: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_103, 0.0026041666666666665)
    mul_657: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_658: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_656, mul_657);  mul_656 = mul_657 = None
    unsqueeze_269: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
    unsqueeze_270: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    mul_659: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_61, view_91);  view_91 = None
    unsqueeze_271: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_659, 0);  mul_659 = None
    unsqueeze_272: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    mul_660: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_270);  sub_193 = unsqueeze_270 = None
    sub_195: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_252, mul_660);  view_252 = mul_660 = None
    sub_196: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_268);  sub_195 = unsqueeze_268 = None
    mul_661: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_272);  sub_196 = unsqueeze_272 = None
    mul_662: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_61);  sum_103 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_253: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_662, [1536, 1, 1, 1]);  mul_662 = None
    mul_663: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_253, 0.09125009274634042);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_254: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_661, [1536, 384, 1, 1]);  mul_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sub_197: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_31)
    mul_664: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_39, sub_197);  convolution_39 = sub_197 = None
    add_107: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_664, 1);  mul_664 = None
    mul_665: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_31, add_107);  sigmoid_31 = add_107 = None
    mul_666: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_234, mul_665);  getitem_234 = mul_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_104: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_666, [0, 2, 3])
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_666, mul_133, view_89, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_666 = mul_133 = view_89 = None
    getitem_237: "f32[8, 384, 14, 14]" = convolution_backward_41[0]
    getitem_238: "f32[384, 64, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_255: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_238, [1, 384, 576]);  getitem_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_105: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_255, [0, 2])
    sub_198: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_87, unsqueeze_274);  view_87 = unsqueeze_274 = None
    mul_667: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_255, sub_198)
    sum_106: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 2]);  mul_667 = None
    mul_668: "f32[384]" = torch.ops.aten.mul.Tensor(sum_105, 0.001736111111111111);  sum_105 = None
    unsqueeze_275: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_668, 0);  mul_668 = None
    unsqueeze_276: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    mul_669: "f32[384]" = torch.ops.aten.mul.Tensor(sum_106, 0.001736111111111111)
    mul_670: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_59, squeeze_59)
    mul_671: "f32[384]" = torch.ops.aten.mul.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    unsqueeze_277: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
    unsqueeze_278: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    mul_672: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_59, view_88);  view_88 = None
    unsqueeze_279: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_280: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
    mul_673: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_278);  sub_198 = unsqueeze_278 = None
    sub_200: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_255, mul_673);  view_255 = mul_673 = None
    sub_201: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_276);  sub_200 = unsqueeze_276 = None
    mul_674: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_280);  sub_201 = unsqueeze_280 = None
    mul_675: "f32[384]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_59);  sum_106 = squeeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_256: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_675, [384, 1, 1, 1]);  mul_675 = None
    mul_676: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_256, 0.07450538873672485);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_257: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_674, [384, 64, 3, 3]);  mul_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_90: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_38)
    sub_202: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_90)
    mul_677: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, sub_202);  convolution_38 = sub_202 = None
    add_108: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_677, 1);  mul_677 = None
    mul_678: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_90, add_108);  sigmoid_90 = add_108 = None
    mul_679: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_237, mul_678);  getitem_237 = mul_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_107: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 2, 3])
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_679, mul_129, view_86, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_679 = mul_129 = view_86 = None
    getitem_240: "f32[8, 384, 14, 14]" = convolution_backward_42[0]
    getitem_241: "f32[384, 64, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_258: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_241, [1, 384, 576]);  getitem_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_108: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_258, [0, 2])
    sub_203: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_84, unsqueeze_282);  view_84 = unsqueeze_282 = None
    mul_680: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_258, sub_203)
    sum_109: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2]);  mul_680 = None
    mul_681: "f32[384]" = torch.ops.aten.mul.Tensor(sum_108, 0.001736111111111111);  sum_108 = None
    unsqueeze_283: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
    unsqueeze_284: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    mul_682: "f32[384]" = torch.ops.aten.mul.Tensor(sum_109, 0.001736111111111111)
    mul_683: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_57, squeeze_57)
    mul_684: "f32[384]" = torch.ops.aten.mul.Tensor(mul_682, mul_683);  mul_682 = mul_683 = None
    unsqueeze_285: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_286: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
    mul_685: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_57, view_85);  view_85 = None
    unsqueeze_287: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_288: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    mul_686: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_286);  sub_203 = unsqueeze_286 = None
    sub_205: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_258, mul_686);  view_258 = mul_686 = None
    sub_206: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_284);  sub_205 = unsqueeze_284 = None
    mul_687: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_288);  sub_206 = unsqueeze_288 = None
    mul_688: "f32[384]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_57);  sum_109 = squeeze_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_259: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_688, [384, 1, 1, 1]);  mul_688 = None
    mul_689: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_259, 0.07450538873672485);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_260: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_687, [384, 64, 3, 3]);  mul_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_91: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_37)
    sub_207: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_91)
    mul_690: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_37, sub_207);  convolution_37 = sub_207 = None
    add_109: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_690, 1);  mul_690 = None
    mul_691: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_91, add_109);  sigmoid_91 = add_109 = None
    mul_692: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_240, mul_691);  getitem_240 = mul_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_110: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_692, [0, 2, 3])
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_692, mul_125, view_83, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_692 = mul_125 = view_83 = None
    getitem_243: "f32[8, 1536, 14, 14]" = convolution_backward_43[0]
    getitem_244: "f32[384, 1536, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_261: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(getitem_244, [1, 384, 1536]);  getitem_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_111: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_261, [0, 2])
    sub_208: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_81, unsqueeze_290);  view_81 = unsqueeze_290 = None
    mul_693: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_261, sub_208)
    sum_112: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_693, [0, 2]);  mul_693 = None
    mul_694: "f32[384]" = torch.ops.aten.mul.Tensor(sum_111, 0.0006510416666666666);  sum_111 = None
    unsqueeze_291: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_292: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
    mul_695: "f32[384]" = torch.ops.aten.mul.Tensor(sum_112, 0.0006510416666666666)
    mul_696: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_697: "f32[384]" = torch.ops.aten.mul.Tensor(mul_695, mul_696);  mul_695 = mul_696 = None
    unsqueeze_293: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_697, 0);  mul_697 = None
    unsqueeze_294: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    mul_698: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_55, view_82);  view_82 = None
    unsqueeze_295: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
    unsqueeze_296: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    mul_699: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_294);  sub_208 = unsqueeze_294 = None
    sub_210: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_261, mul_699);  view_261 = mul_699 = None
    sub_211: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_292);  sub_210 = unsqueeze_292 = None
    mul_700: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_296);  sub_211 = unsqueeze_296 = None
    mul_701: "f32[384]" = torch.ops.aten.mul.Tensor(sum_112, squeeze_55);  sum_112 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_262: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_701, [384, 1, 1, 1]);  mul_701 = None
    mul_702: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_262, 0.04562504637317021);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_263: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_700, [384, 1536, 1, 1]);  mul_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_703: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_243, 0.9622504486493761);  getitem_243 = None
    mul_706: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_703, mul_705);  mul_703 = mul_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_111: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(add_105, mul_706);  add_105 = mul_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_707: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_111, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_708: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_707, 2.0);  mul_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_709: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_708, convolution_34);  convolution_34 = None
    mul_710: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_708, sigmoid_27);  mul_708 = None
    sum_113: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_709, [2, 3], True);  mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_213: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_27)
    mul_711: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_27, sub_213);  sigmoid_27 = sub_213 = None
    mul_712: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_113, mul_711);  sum_113 = mul_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_114: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_712, [0, 2, 3])
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_712, relu_4, primals_190, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_712 = primals_190 = None
    getitem_246: "f32[8, 384, 1, 1]" = convolution_backward_44[0]
    getitem_247: "f32[1536, 384, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_7: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_7: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_7, full_default_1, getitem_246);  le_7 = getitem_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_115: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(where_7, mean_4, primals_188, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_7 = mean_4 = primals_188 = None
    getitem_249: "f32[8, 1536, 1, 1]" = convolution_backward_45[0]
    getitem_250: "f32[384, 1536, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[8, 1536, 14, 14]" = torch.ops.aten.expand.default(getitem_249, [8, 1536, 14, 14]);  getitem_249 = None
    div_8: "f32[8, 1536, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_112: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_710, div_8);  mul_710 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_116: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_112, [0, 2, 3])
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(add_112, mul_117, view_80, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_112 = mul_117 = view_80 = None
    getitem_252: "f32[8, 384, 14, 14]" = convolution_backward_46[0]
    getitem_253: "f32[1536, 384, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_264: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(getitem_253, [1, 1536, 384]);  getitem_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_117: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_264, [0, 2])
    sub_214: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_78, unsqueeze_298);  view_78 = unsqueeze_298 = None
    mul_713: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_264, sub_214)
    sum_118: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_713, [0, 2]);  mul_713 = None
    mul_714: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_117, 0.0026041666666666665);  sum_117 = None
    unsqueeze_299: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_714, 0);  mul_714 = None
    unsqueeze_300: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    mul_715: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_118, 0.0026041666666666665)
    mul_716: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_53, squeeze_53)
    mul_717: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_715, mul_716);  mul_715 = mul_716 = None
    unsqueeze_301: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
    unsqueeze_302: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    mul_718: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_53, view_79);  view_79 = None
    unsqueeze_303: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
    unsqueeze_304: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
    mul_719: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_302);  sub_214 = unsqueeze_302 = None
    sub_216: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_264, mul_719);  view_264 = mul_719 = None
    sub_217: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_300);  sub_216 = unsqueeze_300 = None
    mul_720: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_304);  sub_217 = unsqueeze_304 = None
    mul_721: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_53);  sum_118 = squeeze_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_265: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_721, [1536, 1, 1, 1]);  mul_721 = None
    mul_722: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_265, 0.09125009274634042);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_266: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_720, [1536, 384, 1, 1]);  mul_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sub_218: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_26)
    mul_723: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_33, sub_218);  convolution_33 = sub_218 = None
    add_113: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_723, 1);  mul_723 = None
    mul_724: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_26, add_113);  sigmoid_26 = add_113 = None
    mul_725: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_252, mul_724);  getitem_252 = mul_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_119: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_725, [0, 2, 3])
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_725, mul_113, view_77, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_725 = mul_113 = view_77 = None
    getitem_255: "f32[8, 384, 14, 14]" = convolution_backward_47[0]
    getitem_256: "f32[384, 64, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_267: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_256, [1, 384, 576]);  getitem_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_120: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_267, [0, 2])
    sub_219: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_75, unsqueeze_306);  view_75 = unsqueeze_306 = None
    mul_726: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_267, sub_219)
    sum_121: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_726, [0, 2]);  mul_726 = None
    mul_727: "f32[384]" = torch.ops.aten.mul.Tensor(sum_120, 0.001736111111111111);  sum_120 = None
    unsqueeze_307: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
    unsqueeze_308: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    mul_728: "f32[384]" = torch.ops.aten.mul.Tensor(sum_121, 0.001736111111111111)
    mul_729: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_51, squeeze_51)
    mul_730: "f32[384]" = torch.ops.aten.mul.Tensor(mul_728, mul_729);  mul_728 = mul_729 = None
    unsqueeze_309: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_310: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
    mul_731: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_51, view_76);  view_76 = None
    unsqueeze_311: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
    unsqueeze_312: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    mul_732: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_310);  sub_219 = unsqueeze_310 = None
    sub_221: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_267, mul_732);  view_267 = mul_732 = None
    sub_222: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_308);  sub_221 = unsqueeze_308 = None
    mul_733: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_312);  sub_222 = unsqueeze_312 = None
    mul_734: "f32[384]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_51);  sum_121 = squeeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_268: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_734, [384, 1, 1, 1]);  mul_734 = None
    mul_735: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_268, 0.07450538873672485);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_269: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_733, [384, 64, 3, 3]);  mul_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_94: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_32)
    sub_223: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_94)
    mul_736: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_32, sub_223);  convolution_32 = sub_223 = None
    add_114: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_736, 1);  mul_736 = None
    mul_737: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_94, add_114);  sigmoid_94 = add_114 = None
    mul_738: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_255, mul_737);  getitem_255 = mul_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_122: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_738, [0, 2, 3])
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_738, mul_109, view_74, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_738 = mul_109 = view_74 = None
    getitem_258: "f32[8, 384, 14, 14]" = convolution_backward_48[0]
    getitem_259: "f32[384, 64, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_270: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_259, [1, 384, 576]);  getitem_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_123: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_270, [0, 2])
    sub_224: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_72, unsqueeze_314);  view_72 = unsqueeze_314 = None
    mul_739: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_270, sub_224)
    sum_124: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_739, [0, 2]);  mul_739 = None
    mul_740: "f32[384]" = torch.ops.aten.mul.Tensor(sum_123, 0.001736111111111111);  sum_123 = None
    unsqueeze_315: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    unsqueeze_316: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
    mul_741: "f32[384]" = torch.ops.aten.mul.Tensor(sum_124, 0.001736111111111111)
    mul_742: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_743: "f32[384]" = torch.ops.aten.mul.Tensor(mul_741, mul_742);  mul_741 = mul_742 = None
    unsqueeze_317: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_318: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    mul_744: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_49, view_73);  view_73 = None
    unsqueeze_319: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
    unsqueeze_320: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    mul_745: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_318);  sub_224 = unsqueeze_318 = None
    sub_226: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_270, mul_745);  view_270 = mul_745 = None
    sub_227: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_316);  sub_226 = unsqueeze_316 = None
    mul_746: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_320);  sub_227 = unsqueeze_320 = None
    mul_747: "f32[384]" = torch.ops.aten.mul.Tensor(sum_124, squeeze_49);  sum_124 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_271: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_747, [384, 1, 1, 1]);  mul_747 = None
    mul_748: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_271, 0.07450538873672485);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_272: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_746, [384, 64, 3, 3]);  mul_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_95: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_31)
    sub_228: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_95)
    mul_749: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_31, sub_228);  convolution_31 = sub_228 = None
    add_115: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_749, 1);  mul_749 = None
    mul_750: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_95, add_115);  sigmoid_95 = add_115 = None
    mul_751: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_258, mul_750);  getitem_258 = mul_750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_125: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_751, [0, 2, 3])
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_751, mul_105, view_71, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_751 = mul_105 = view_71 = None
    getitem_261: "f32[8, 1536, 14, 14]" = convolution_backward_49[0]
    getitem_262: "f32[384, 1536, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_273: "f32[1, 384, 1536]" = torch.ops.aten.reshape.default(getitem_262, [1, 384, 1536]);  getitem_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_126: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 2])
    sub_229: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_69, unsqueeze_322);  view_69 = unsqueeze_322 = None
    mul_752: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(view_273, sub_229)
    sum_127: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 2]);  mul_752 = None
    mul_753: "f32[384]" = torch.ops.aten.mul.Tensor(sum_126, 0.0006510416666666666);  sum_126 = None
    unsqueeze_323: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
    unsqueeze_324: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    mul_754: "f32[384]" = torch.ops.aten.mul.Tensor(sum_127, 0.0006510416666666666)
    mul_755: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_47, squeeze_47)
    mul_756: "f32[384]" = torch.ops.aten.mul.Tensor(mul_754, mul_755);  mul_754 = mul_755 = None
    unsqueeze_325: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
    unsqueeze_326: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    mul_757: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_47, view_70);  view_70 = None
    unsqueeze_327: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_757, 0);  mul_757 = None
    unsqueeze_328: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
    mul_758: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_326);  sub_229 = unsqueeze_326 = None
    sub_231: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_273, mul_758);  view_273 = mul_758 = None
    sub_232: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_324);  sub_231 = unsqueeze_324 = None
    mul_759: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_328);  sub_232 = unsqueeze_328 = None
    mul_760: "f32[384]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_47);  sum_127 = squeeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_274: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_760, [384, 1, 1, 1]);  mul_760 = None
    mul_761: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_274, 0.04562504637317021);  view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_275: "f32[384, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_759, [384, 1536, 1, 1]);  mul_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_762: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_261, 0.9805806756909201);  getitem_261 = None
    mul_765: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_762, mul_764);  mul_762 = mul_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_117: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(add_111, mul_765);  add_111 = mul_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_766: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_117, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_767: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_766, 2.0);  mul_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_768: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_767, convolution_28);  convolution_28 = None
    mul_769: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_767, sigmoid_22);  mul_767 = None
    sum_128: "f32[8, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_768, [2, 3], True);  mul_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_234: "f32[8, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_22)
    mul_770: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_22, sub_234);  sigmoid_22 = sub_234 = None
    mul_771: "f32[8, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_128, mul_770);  sum_128 = mul_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_129: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3])
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_771, relu_3, primals_186, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_771 = primals_186 = None
    getitem_264: "f32[8, 384, 1, 1]" = convolution_backward_50[0]
    getitem_265: "f32[1536, 384, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_8: "b8[8, 384, 1, 1]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_8: "f32[8, 384, 1, 1]" = torch.ops.aten.where.self(le_8, full_default_1, getitem_264);  le_8 = getitem_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_130: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(where_8, mean_3, primals_184, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = mean_3 = primals_184 = None
    getitem_267: "f32[8, 1536, 1, 1]" = convolution_backward_51[0]
    getitem_268: "f32[384, 1536, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[8, 1536, 14, 14]" = torch.ops.aten.expand.default(getitem_267, [8, 1536, 14, 14]);  getitem_267 = None
    div_9: "f32[8, 1536, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_118: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_769, div_9);  mul_769 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_131: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_118, [0, 2, 3])
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(add_118, mul_97, view_68, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_118 = mul_97 = view_68 = None
    getitem_270: "f32[8, 384, 14, 14]" = convolution_backward_52[0]
    getitem_271: "f32[1536, 384, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_276: "f32[1, 1536, 384]" = torch.ops.aten.reshape.default(getitem_271, [1, 1536, 384]);  getitem_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_132: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_276, [0, 2])
    sub_235: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_66, unsqueeze_330);  view_66 = unsqueeze_330 = None
    mul_772: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(view_276, sub_235)
    sum_133: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_772, [0, 2]);  mul_772 = None
    mul_773: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_132, 0.0026041666666666665);  sum_132 = None
    unsqueeze_331: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_332: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    mul_774: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_133, 0.0026041666666666665)
    mul_775: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_45, squeeze_45)
    mul_776: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_774, mul_775);  mul_774 = mul_775 = None
    unsqueeze_333: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_334: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
    mul_777: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_45, view_67);  view_67 = None
    unsqueeze_335: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
    unsqueeze_336: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    mul_778: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_334);  sub_235 = unsqueeze_334 = None
    sub_237: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_276, mul_778);  view_276 = mul_778 = None
    sub_238: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_332);  sub_237 = unsqueeze_332 = None
    mul_779: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_336);  sub_238 = unsqueeze_336 = None
    mul_780: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_45);  sum_133 = squeeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_277: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_780, [1536, 1, 1, 1]);  mul_780 = None
    mul_781: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_277, 0.09125009274634042);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_278: "f32[1536, 384, 1, 1]" = torch.ops.aten.reshape.default(mul_779, [1536, 384, 1, 1]);  mul_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sub_239: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_21)
    mul_782: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_27, sub_239);  convolution_27 = sub_239 = None
    add_119: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_782, 1);  mul_782 = None
    mul_783: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_21, add_119);  sigmoid_21 = add_119 = None
    mul_784: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_270, mul_783);  getitem_270 = mul_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_134: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_784, [0, 2, 3])
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_784, mul_93, view_65, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_784 = mul_93 = view_65 = None
    getitem_273: "f32[8, 384, 14, 14]" = convolution_backward_53[0]
    getitem_274: "f32[384, 64, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_279: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_274, [1, 384, 576]);  getitem_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_135: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_279, [0, 2])
    sub_240: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_63, unsqueeze_338);  view_63 = unsqueeze_338 = None
    mul_785: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_279, sub_240)
    sum_136: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_785, [0, 2]);  mul_785 = None
    mul_786: "f32[384]" = torch.ops.aten.mul.Tensor(sum_135, 0.001736111111111111);  sum_135 = None
    unsqueeze_339: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
    unsqueeze_340: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
    mul_787: "f32[384]" = torch.ops.aten.mul.Tensor(sum_136, 0.001736111111111111)
    mul_788: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_789: "f32[384]" = torch.ops.aten.mul.Tensor(mul_787, mul_788);  mul_787 = mul_788 = None
    unsqueeze_341: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
    unsqueeze_342: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    mul_790: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_43, view_64);  view_64 = None
    unsqueeze_343: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
    unsqueeze_344: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    mul_791: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_342);  sub_240 = unsqueeze_342 = None
    sub_242: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_279, mul_791);  view_279 = mul_791 = None
    sub_243: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_340);  sub_242 = unsqueeze_340 = None
    mul_792: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_344);  sub_243 = unsqueeze_344 = None
    mul_793: "f32[384]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_43);  sum_136 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_280: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_793, [384, 1, 1, 1]);  mul_793 = None
    mul_794: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_280, 0.07450538873672485);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_281: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_792, [384, 64, 3, 3]);  mul_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_98: "f32[8, 384, 14, 14]" = torch.ops.aten.sigmoid.default(convolution_26)
    sub_244: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_98);  full_default_14 = None
    mul_795: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, sub_244);  convolution_26 = sub_244 = None
    add_120: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Scalar(mul_795, 1);  mul_795 = None
    mul_796: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_98, add_120);  sigmoid_98 = add_120 = None
    mul_797: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_273, mul_796);  getitem_273 = mul_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_137: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_797, [0, 2, 3])
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_797, mul_89, view_62, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_797 = mul_89 = view_62 = None
    getitem_276: "f32[8, 384, 28, 28]" = convolution_backward_54[0]
    getitem_277: "f32[384, 64, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_282: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(getitem_277, [1, 384, 576]);  getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_138: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_282, [0, 2])
    sub_245: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_60, unsqueeze_346);  view_60 = unsqueeze_346 = None
    mul_798: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(view_282, sub_245)
    sum_139: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_798, [0, 2]);  mul_798 = None
    mul_799: "f32[384]" = torch.ops.aten.mul.Tensor(sum_138, 0.001736111111111111);  sum_138 = None
    unsqueeze_347: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
    unsqueeze_348: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    mul_800: "f32[384]" = torch.ops.aten.mul.Tensor(sum_139, 0.001736111111111111)
    mul_801: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_41, squeeze_41)
    mul_802: "f32[384]" = torch.ops.aten.mul.Tensor(mul_800, mul_801);  mul_800 = mul_801 = None
    unsqueeze_349: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
    unsqueeze_350: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    mul_803: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_41, view_61);  view_61 = None
    unsqueeze_351: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    unsqueeze_352: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
    mul_804: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_350);  sub_245 = unsqueeze_350 = None
    sub_247: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_282, mul_804);  view_282 = mul_804 = None
    sub_248: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_348);  sub_247 = unsqueeze_348 = None
    mul_805: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_352);  sub_248 = unsqueeze_352 = None
    mul_806: "f32[384]" = torch.ops.aten.mul.Tensor(sum_139, squeeze_41);  sum_139 = squeeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_283: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_806, [384, 1, 1, 1]);  mul_806 = None
    mul_807: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_283, 0.07450538873672485);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_284: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_805, [384, 64, 3, 3]);  mul_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_99: "f32[8, 384, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_25)
    full_default_44: "f32[8, 384, 28, 28]" = torch.ops.aten.full.default([8, 384, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_249: "f32[8, 384, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_44, sigmoid_99);  full_default_44 = None
    mul_808: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_25, sub_249);  convolution_25 = sub_249 = None
    add_121: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Scalar(mul_808, 1);  mul_808 = None
    mul_809: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_99, add_121);  sigmoid_99 = add_121 = None
    mul_810: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_276, mul_809);  getitem_276 = mul_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_140: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_810, [0, 2, 3])
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_810, mul_82, view_59, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_810 = view_59 = None
    getitem_279: "f32[8, 512, 28, 28]" = convolution_backward_55[0]
    getitem_280: "f32[384, 512, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_285: "f32[1, 384, 512]" = torch.ops.aten.reshape.default(getitem_280, [1, 384, 512]);  getitem_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_141: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_285, [0, 2])
    sub_250: "f32[1, 384, 512]" = torch.ops.aten.sub.Tensor(view_57, unsqueeze_354);  view_57 = unsqueeze_354 = None
    mul_811: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(view_285, sub_250)
    sum_142: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_811, [0, 2]);  mul_811 = None
    mul_812: "f32[384]" = torch.ops.aten.mul.Tensor(sum_141, 0.001953125);  sum_141 = None
    unsqueeze_355: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    unsqueeze_356: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    mul_813: "f32[384]" = torch.ops.aten.mul.Tensor(sum_142, 0.001953125)
    mul_814: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_39, squeeze_39)
    mul_815: "f32[384]" = torch.ops.aten.mul.Tensor(mul_813, mul_814);  mul_813 = mul_814 = None
    unsqueeze_357: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
    unsqueeze_358: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    mul_816: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_39, view_58);  view_58 = None
    unsqueeze_359: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
    unsqueeze_360: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    mul_817: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_358);  sub_250 = unsqueeze_358 = None
    sub_252: "f32[1, 384, 512]" = torch.ops.aten.sub.Tensor(view_285, mul_817);  view_285 = mul_817 = None
    sub_253: "f32[1, 384, 512]" = torch.ops.aten.sub.Tensor(sub_252, unsqueeze_356);  sub_252 = unsqueeze_356 = None
    mul_818: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_360);  sub_253 = unsqueeze_360 = None
    mul_819: "f32[384]" = torch.ops.aten.mul.Tensor(sum_142, squeeze_39);  sum_142 = squeeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_286: "f32[384, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_819, [384, 1, 1, 1]);  mul_819 = None
    mul_820: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_286, 0.07902489841601695);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_287: "f32[384, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_818, [384, 512, 1, 1]);  mul_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_143: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 2, 3])
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(add_117, avg_pool2d_1, view_56, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_117 = avg_pool2d_1 = view_56 = None
    getitem_282: "f32[8, 512, 14, 14]" = convolution_backward_56[0]
    getitem_283: "f32[1536, 512, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_288: "f32[1, 1536, 512]" = torch.ops.aten.reshape.default(getitem_283, [1, 1536, 512]);  getitem_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_144: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_288, [0, 2])
    sub_254: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_54, unsqueeze_362);  view_54 = unsqueeze_362 = None
    mul_821: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(view_288, sub_254)
    sum_145: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_821, [0, 2]);  mul_821 = None
    mul_822: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_144, 0.001953125);  sum_144 = None
    unsqueeze_363: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
    unsqueeze_364: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
    mul_823: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_145, 0.001953125)
    mul_824: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_825: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_823, mul_824);  mul_823 = mul_824 = None
    unsqueeze_365: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
    unsqueeze_366: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    mul_826: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_37, view_55);  view_55 = None
    unsqueeze_367: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_826, 0);  mul_826 = None
    unsqueeze_368: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    mul_827: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_366);  sub_254 = unsqueeze_366 = None
    sub_256: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_288, mul_827);  view_288 = mul_827 = None
    sub_257: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(sub_256, unsqueeze_364);  sub_256 = unsqueeze_364 = None
    mul_828: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_368);  sub_257 = unsqueeze_368 = None
    mul_829: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_37);  sum_145 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_289: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_829, [1536, 1, 1, 1]);  mul_829 = None
    mul_830: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_289, 0.07902489841601695);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_290: "f32[1536, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_828, [1536, 512, 1, 1]);  mul_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_backward_1: "f32[8, 512, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(getitem_282, mul_82, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_282 = mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    add_122: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(getitem_279, avg_pool2d_backward_1);  getitem_279 = avg_pool2d_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_831: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_122, 0.9622504486493761);  add_122 = None
    mul_834: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_831, mul_833);  mul_831 = mul_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_835: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_834, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_836: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_835, 2.0);  mul_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_837: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_836, convolution_21);  convolution_21 = None
    mul_838: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_836, sigmoid_17);  mul_836 = None
    sum_146: "f32[8, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_837, [2, 3], True);  mul_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_259: "f32[8, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_17)
    mul_839: "f32[8, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_17, sub_259);  sigmoid_17 = sub_259 = None
    mul_840: "f32[8, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_146, mul_839);  sum_146 = mul_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_147: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_840, [0, 2, 3])
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_840, relu_2, primals_182, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_840 = primals_182 = None
    getitem_285: "f32[8, 128, 1, 1]" = convolution_backward_57[0]
    getitem_286: "f32[512, 128, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_9: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_9: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_9, full_default_1, getitem_285);  le_9 = getitem_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_148: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(where_9, mean_2, primals_180, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = mean_2 = primals_180 = None
    getitem_288: "f32[8, 512, 1, 1]" = convolution_backward_58[0]
    getitem_289: "f32[128, 512, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[8, 512, 28, 28]" = torch.ops.aten.expand.default(getitem_288, [8, 512, 28, 28]);  getitem_288 = None
    div_10: "f32[8, 512, 28, 28]" = torch.ops.aten.div.Scalar(expand_10, 784);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_124: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_838, div_10);  mul_838 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_149: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_124, [0, 2, 3])
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(add_124, mul_74, view_53, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_124 = mul_74 = view_53 = None
    getitem_291: "f32[8, 128, 28, 28]" = convolution_backward_59[0]
    getitem_292: "f32[512, 128, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_291: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(getitem_292, [1, 512, 128]);  getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_150: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_291, [0, 2])
    sub_260: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_51, unsqueeze_370);  view_51 = unsqueeze_370 = None
    mul_841: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_291, sub_260)
    sum_151: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_841, [0, 2]);  mul_841 = None
    mul_842: "f32[512]" = torch.ops.aten.mul.Tensor(sum_150, 0.0078125);  sum_150 = None
    unsqueeze_371: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_842, 0);  mul_842 = None
    unsqueeze_372: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    mul_843: "f32[512]" = torch.ops.aten.mul.Tensor(sum_151, 0.0078125)
    mul_844: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_35, squeeze_35)
    mul_845: "f32[512]" = torch.ops.aten.mul.Tensor(mul_843, mul_844);  mul_843 = mul_844 = None
    unsqueeze_373: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_845, 0);  mul_845 = None
    unsqueeze_374: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    mul_846: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_35, view_52);  view_52 = None
    unsqueeze_375: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
    unsqueeze_376: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
    mul_847: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_374);  sub_260 = unsqueeze_374 = None
    sub_262: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_291, mul_847);  view_291 = mul_847 = None
    sub_263: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_372);  sub_262 = unsqueeze_372 = None
    mul_848: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_376);  sub_263 = unsqueeze_376 = None
    mul_849: "f32[512]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_35);  sum_151 = squeeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_292: "f32[512, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_849, [512, 1, 1, 1]);  mul_849 = None
    mul_850: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_292, 0.1580497968320339);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_293: "f32[512, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_848, [512, 128, 1, 1]);  mul_848 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    full_default_47: "f32[8, 128, 28, 28]" = torch.ops.aten.full.default([8, 128, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_264: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_47, sigmoid_16)
    mul_851: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, sub_264);  convolution_20 = sub_264 = None
    add_125: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Scalar(mul_851, 1);  mul_851 = None
    mul_852: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_16, add_125);  sigmoid_16 = add_125 = None
    mul_853: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_291, mul_852);  getitem_291 = mul_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_152: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_853, [0, 2, 3])
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_853, mul_70, view_50, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_853 = mul_70 = view_50 = None
    getitem_294: "f32[8, 128, 28, 28]" = convolution_backward_60[0]
    getitem_295: "f32[128, 64, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_294: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(getitem_295, [1, 128, 576]);  getitem_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_153: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_294, [0, 2])
    sub_265: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_48, unsqueeze_378);  view_48 = unsqueeze_378 = None
    mul_854: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(view_294, sub_265)
    sum_154: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_854, [0, 2]);  mul_854 = None
    mul_855: "f32[128]" = torch.ops.aten.mul.Tensor(sum_153, 0.001736111111111111);  sum_153 = None
    unsqueeze_379: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
    unsqueeze_380: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    mul_856: "f32[128]" = torch.ops.aten.mul.Tensor(sum_154, 0.001736111111111111)
    mul_857: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_33, squeeze_33)
    mul_858: "f32[128]" = torch.ops.aten.mul.Tensor(mul_856, mul_857);  mul_856 = mul_857 = None
    unsqueeze_381: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_858, 0);  mul_858 = None
    unsqueeze_382: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
    mul_859: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_33, view_49);  view_49 = None
    unsqueeze_383: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_859, 0);  mul_859 = None
    unsqueeze_384: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    mul_860: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_382);  sub_265 = unsqueeze_382 = None
    sub_267: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_294, mul_860);  view_294 = mul_860 = None
    sub_268: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_380);  sub_267 = unsqueeze_380 = None
    mul_861: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_384);  sub_268 = unsqueeze_384 = None
    mul_862: "f32[128]" = torch.ops.aten.mul.Tensor(sum_154, squeeze_33);  sum_154 = squeeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_295: "f32[128, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_862, [128, 1, 1, 1]);  mul_862 = None
    mul_863: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_295, 0.07450538873672485);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_296: "f32[128, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_861, [128, 64, 3, 3]);  mul_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_102: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_19)
    sub_269: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_47, sigmoid_102)
    mul_864: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_19, sub_269);  convolution_19 = sub_269 = None
    add_126: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Scalar(mul_864, 1);  mul_864 = None
    mul_865: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_102, add_126);  sigmoid_102 = add_126 = None
    mul_866: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_294, mul_865);  getitem_294 = mul_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_155: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_866, [0, 2, 3])
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_866, mul_66, view_47, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_866 = mul_66 = view_47 = None
    getitem_297: "f32[8, 128, 28, 28]" = convolution_backward_61[0]
    getitem_298: "f32[128, 64, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_297: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(getitem_298, [1, 128, 576]);  getitem_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_156: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_297, [0, 2])
    sub_270: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_45, unsqueeze_386);  view_45 = unsqueeze_386 = None
    mul_867: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(view_297, sub_270)
    sum_157: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_867, [0, 2]);  mul_867 = None
    mul_868: "f32[128]" = torch.ops.aten.mul.Tensor(sum_156, 0.001736111111111111);  sum_156 = None
    unsqueeze_387: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_868, 0);  mul_868 = None
    unsqueeze_388: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
    mul_869: "f32[128]" = torch.ops.aten.mul.Tensor(sum_157, 0.001736111111111111)
    mul_870: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_871: "f32[128]" = torch.ops.aten.mul.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
    unsqueeze_389: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
    unsqueeze_390: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    mul_872: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, view_46);  view_46 = None
    unsqueeze_391: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
    unsqueeze_392: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    mul_873: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_390);  sub_270 = unsqueeze_390 = None
    sub_272: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_297, mul_873);  view_297 = mul_873 = None
    sub_273: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(sub_272, unsqueeze_388);  sub_272 = unsqueeze_388 = None
    mul_874: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_392);  sub_273 = unsqueeze_392 = None
    mul_875: "f32[128]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_31);  sum_157 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_298: "f32[128, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_875, [128, 1, 1, 1]);  mul_875 = None
    mul_876: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_298, 0.07450538873672485);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_299: "f32[128, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_874, [128, 64, 3, 3]);  mul_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_103: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_18)
    sub_274: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_47, sigmoid_103)
    mul_877: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, sub_274);  convolution_18 = sub_274 = None
    add_127: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Scalar(mul_877, 1);  mul_877 = None
    mul_878: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_103, add_127);  sigmoid_103 = add_127 = None
    mul_879: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_297, mul_878);  getitem_297 = mul_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_158: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_879, [0, 2, 3])
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_879, mul_62, view_44, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_879 = mul_62 = view_44 = None
    getitem_300: "f32[8, 512, 28, 28]" = convolution_backward_62[0]
    getitem_301: "f32[128, 512, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_300: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(getitem_301, [1, 128, 512]);  getitem_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_159: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_300, [0, 2])
    sub_275: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(view_42, unsqueeze_394);  view_42 = unsqueeze_394 = None
    mul_880: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_300, sub_275)
    sum_160: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_880, [0, 2]);  mul_880 = None
    mul_881: "f32[128]" = torch.ops.aten.mul.Tensor(sum_159, 0.001953125);  sum_159 = None
    unsqueeze_395: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    unsqueeze_396: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    mul_882: "f32[128]" = torch.ops.aten.mul.Tensor(sum_160, 0.001953125)
    mul_883: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_29, squeeze_29)
    mul_884: "f32[128]" = torch.ops.aten.mul.Tensor(mul_882, mul_883);  mul_882 = mul_883 = None
    unsqueeze_397: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_398: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    mul_885: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_29, view_43);  view_43 = None
    unsqueeze_399: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_885, 0);  mul_885 = None
    unsqueeze_400: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    mul_886: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_398);  sub_275 = unsqueeze_398 = None
    sub_277: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(view_300, mul_886);  view_300 = mul_886 = None
    sub_278: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(sub_277, unsqueeze_396);  sub_277 = unsqueeze_396 = None
    mul_887: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_400);  sub_278 = unsqueeze_400 = None
    mul_888: "f32[128]" = torch.ops.aten.mul.Tensor(sum_160, squeeze_29);  sum_160 = squeeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_301: "f32[128, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_888, [128, 1, 1, 1]);  mul_888 = None
    mul_889: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_301, 0.07902489841601695);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_302: "f32[128, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_887, [128, 512, 1, 1]);  mul_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_890: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_300, 0.9805806756909201);  getitem_300 = None
    mul_893: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_890, mul_892);  mul_890 = mul_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    add_129: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_834, mul_893);  mul_834 = mul_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_894: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_129, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_895: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_894, 2.0);  mul_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_896: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_895, convolution_15);  convolution_15 = None
    mul_897: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_895, sigmoid_12);  mul_895 = None
    sum_161: "f32[8, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_896, [2, 3], True);  mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_280: "f32[8, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_12)
    mul_898: "f32[8, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_12, sub_280);  sigmoid_12 = sub_280 = None
    mul_899: "f32[8, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_161, mul_898);  sum_161 = mul_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_162: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_899, [0, 2, 3])
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_899, relu_1, primals_178, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_899 = primals_178 = None
    getitem_303: "f32[8, 128, 1, 1]" = convolution_backward_63[0]
    getitem_304: "f32[512, 128, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_10: "b8[8, 128, 1, 1]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_10: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(le_10, full_default_1, getitem_303);  le_10 = getitem_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_163: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(where_10, mean_1, primals_176, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = mean_1 = primals_176 = None
    getitem_306: "f32[8, 512, 1, 1]" = convolution_backward_64[0]
    getitem_307: "f32[128, 512, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[8, 512, 28, 28]" = torch.ops.aten.expand.default(getitem_306, [8, 512, 28, 28]);  getitem_306 = None
    div_11: "f32[8, 512, 28, 28]" = torch.ops.aten.div.Scalar(expand_11, 784);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_130: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_897, div_11);  mul_897 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_164: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_130, [0, 2, 3])
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(add_130, mul_54, view_41, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_130 = mul_54 = view_41 = None
    getitem_309: "f32[8, 128, 28, 28]" = convolution_backward_65[0]
    getitem_310: "f32[512, 128, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_303: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(getitem_310, [1, 512, 128]);  getitem_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_165: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_303, [0, 2])
    sub_281: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_39, unsqueeze_402);  view_39 = unsqueeze_402 = None
    mul_900: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_303, sub_281)
    sum_166: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_900, [0, 2]);  mul_900 = None
    mul_901: "f32[512]" = torch.ops.aten.mul.Tensor(sum_165, 0.0078125);  sum_165 = None
    unsqueeze_403: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
    unsqueeze_404: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    mul_902: "f32[512]" = torch.ops.aten.mul.Tensor(sum_166, 0.0078125)
    mul_903: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_27, squeeze_27)
    mul_904: "f32[512]" = torch.ops.aten.mul.Tensor(mul_902, mul_903);  mul_902 = mul_903 = None
    unsqueeze_405: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_904, 0);  mul_904 = None
    unsqueeze_406: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    mul_905: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_27, view_40);  view_40 = None
    unsqueeze_407: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_905, 0);  mul_905 = None
    unsqueeze_408: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    mul_906: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_406);  sub_281 = unsqueeze_406 = None
    sub_283: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_303, mul_906);  view_303 = mul_906 = None
    sub_284: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_404);  sub_283 = unsqueeze_404 = None
    mul_907: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_408);  sub_284 = unsqueeze_408 = None
    mul_908: "f32[512]" = torch.ops.aten.mul.Tensor(sum_166, squeeze_27);  sum_166 = squeeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_304: "f32[512, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_908, [512, 1, 1, 1]);  mul_908 = None
    mul_909: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_304, 0.1580497968320339);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_305: "f32[512, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_907, [512, 128, 1, 1]);  mul_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    sub_285: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_47, sigmoid_11)
    mul_910: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, sub_285);  convolution_14 = sub_285 = None
    add_131: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Scalar(mul_910, 1);  mul_910 = None
    mul_911: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_11, add_131);  sigmoid_11 = add_131 = None
    mul_912: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_309, mul_911);  getitem_309 = mul_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_167: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_912, [0, 2, 3])
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_912, mul_50, view_38, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_912 = mul_50 = view_38 = None
    getitem_312: "f32[8, 128, 28, 28]" = convolution_backward_66[0]
    getitem_313: "f32[128, 64, 3, 3]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_306: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(getitem_313, [1, 128, 576]);  getitem_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_168: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_306, [0, 2])
    sub_286: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_36, unsqueeze_410);  view_36 = unsqueeze_410 = None
    mul_913: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(view_306, sub_286)
    sum_169: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_913, [0, 2]);  mul_913 = None
    mul_914: "f32[128]" = torch.ops.aten.mul.Tensor(sum_168, 0.001736111111111111);  sum_168 = None
    unsqueeze_411: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
    unsqueeze_412: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    mul_915: "f32[128]" = torch.ops.aten.mul.Tensor(sum_169, 0.001736111111111111)
    mul_916: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_917: "f32[128]" = torch.ops.aten.mul.Tensor(mul_915, mul_916);  mul_915 = mul_916 = None
    unsqueeze_413: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_917, 0);  mul_917 = None
    unsqueeze_414: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    mul_918: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_25, view_37);  view_37 = None
    unsqueeze_415: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
    unsqueeze_416: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    mul_919: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_414);  sub_286 = unsqueeze_414 = None
    sub_288: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_306, mul_919);  view_306 = mul_919 = None
    sub_289: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(sub_288, unsqueeze_412);  sub_288 = unsqueeze_412 = None
    mul_920: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_416);  sub_289 = unsqueeze_416 = None
    mul_921: "f32[128]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_25);  sum_169 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_307: "f32[128, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_921, [128, 1, 1, 1]);  mul_921 = None
    mul_922: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_307, 0.07450538873672485);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_308: "f32[128, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_920, [128, 64, 3, 3]);  mul_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_106: "f32[8, 128, 28, 28]" = torch.ops.aten.sigmoid.default(convolution_13)
    sub_290: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_47, sigmoid_106);  full_default_47 = None
    mul_923: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_13, sub_290);  convolution_13 = sub_290 = None
    add_132: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Scalar(mul_923, 1);  mul_923 = None
    mul_924: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_106, add_132);  sigmoid_106 = add_132 = None
    mul_925: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_312, mul_924);  getitem_312 = mul_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_170: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_925, [0, 2, 3])
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_925, mul_46, view_35, [128], [2, 2], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_925 = mul_46 = view_35 = None
    getitem_315: "f32[8, 128, 56, 56]" = convolution_backward_67[0]
    getitem_316: "f32[128, 64, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_309: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(getitem_316, [1, 128, 576]);  getitem_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_171: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_309, [0, 2])
    sub_291: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_33, unsqueeze_418);  view_33 = unsqueeze_418 = None
    mul_926: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(view_309, sub_291)
    sum_172: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_926, [0, 2]);  mul_926 = None
    mul_927: "f32[128]" = torch.ops.aten.mul.Tensor(sum_171, 0.001736111111111111);  sum_171 = None
    unsqueeze_419: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
    unsqueeze_420: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    mul_928: "f32[128]" = torch.ops.aten.mul.Tensor(sum_172, 0.001736111111111111)
    mul_929: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_23, squeeze_23)
    mul_930: "f32[128]" = torch.ops.aten.mul.Tensor(mul_928, mul_929);  mul_928 = mul_929 = None
    unsqueeze_421: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_930, 0);  mul_930 = None
    unsqueeze_422: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    mul_931: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_23, view_34);  view_34 = None
    unsqueeze_423: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
    unsqueeze_424: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    mul_932: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_422);  sub_291 = unsqueeze_422 = None
    sub_293: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_309, mul_932);  view_309 = mul_932 = None
    sub_294: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(sub_293, unsqueeze_420);  sub_293 = unsqueeze_420 = None
    mul_933: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_424);  sub_294 = unsqueeze_424 = None
    mul_934: "f32[128]" = torch.ops.aten.mul.Tensor(sum_172, squeeze_23);  sum_172 = squeeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_310: "f32[128, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_934, [128, 1, 1, 1]);  mul_934 = None
    mul_935: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_310, 0.07450538873672485);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_311: "f32[128, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_933, [128, 64, 3, 3]);  mul_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_107: "f32[8, 128, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_12)
    full_default_54: "f32[8, 128, 56, 56]" = torch.ops.aten.full.default([8, 128, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_295: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(full_default_54, sigmoid_107)
    mul_936: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_12, sub_295);  convolution_12 = sub_295 = None
    add_133: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Scalar(mul_936, 1);  mul_936 = None
    mul_937: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_107, add_133);  sigmoid_107 = add_133 = None
    mul_938: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_315, mul_937);  getitem_315 = mul_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_173: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_938, [0, 2, 3])
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_938, mul_39, view_32, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_938 = view_32 = None
    getitem_318: "f32[8, 256, 56, 56]" = convolution_backward_68[0]
    getitem_319: "f32[128, 256, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_312: "f32[1, 128, 256]" = torch.ops.aten.reshape.default(getitem_319, [1, 128, 256]);  getitem_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_174: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_312, [0, 2])
    sub_296: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(view_30, unsqueeze_426);  view_30 = unsqueeze_426 = None
    mul_939: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(view_312, sub_296)
    sum_175: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_939, [0, 2]);  mul_939 = None
    mul_940: "f32[128]" = torch.ops.aten.mul.Tensor(sum_174, 0.00390625);  sum_174 = None
    unsqueeze_427: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_940, 0);  mul_940 = None
    unsqueeze_428: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    mul_941: "f32[128]" = torch.ops.aten.mul.Tensor(sum_175, 0.00390625)
    mul_942: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_21, squeeze_21)
    mul_943: "f32[128]" = torch.ops.aten.mul.Tensor(mul_941, mul_942);  mul_941 = mul_942 = None
    unsqueeze_429: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_943, 0);  mul_943 = None
    unsqueeze_430: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    mul_944: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_21, view_31);  view_31 = None
    unsqueeze_431: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
    unsqueeze_432: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    mul_945: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_430);  sub_296 = unsqueeze_430 = None
    sub_298: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(view_312, mul_945);  view_312 = mul_945 = None
    sub_299: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(sub_298, unsqueeze_428);  sub_298 = unsqueeze_428 = None
    mul_946: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_432);  sub_299 = unsqueeze_432 = None
    mul_947: "f32[128]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_21);  sum_175 = squeeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_313: "f32[128, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_947, [128, 1, 1, 1]);  mul_947 = None
    mul_948: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_313, 0.11175808310508728);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_314: "f32[128, 256, 1, 1]" = torch.ops.aten.reshape.default(mul_946, [128, 256, 1, 1]);  mul_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_176: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 2, 3])
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(add_129, avg_pool2d, view_29, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_129 = avg_pool2d = view_29 = None
    getitem_321: "f32[8, 256, 28, 28]" = convolution_backward_69[0]
    getitem_322: "f32[512, 256, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_315: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(getitem_322, [1, 512, 256]);  getitem_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_177: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_315, [0, 2])
    sub_300: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_27, unsqueeze_434);  view_27 = unsqueeze_434 = None
    mul_949: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_315, sub_300)
    sum_178: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_949, [0, 2]);  mul_949 = None
    mul_950: "f32[512]" = torch.ops.aten.mul.Tensor(sum_177, 0.00390625);  sum_177 = None
    unsqueeze_435: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
    unsqueeze_436: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
    mul_951: "f32[512]" = torch.ops.aten.mul.Tensor(sum_178, 0.00390625)
    mul_952: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_953: "f32[512]" = torch.ops.aten.mul.Tensor(mul_951, mul_952);  mul_951 = mul_952 = None
    unsqueeze_437: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_953, 0);  mul_953 = None
    unsqueeze_438: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    mul_954: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, view_28);  view_28 = None
    unsqueeze_439: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
    unsqueeze_440: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    mul_955: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_438);  sub_300 = unsqueeze_438 = None
    sub_302: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_315, mul_955);  view_315 = mul_955 = None
    sub_303: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_436);  sub_302 = unsqueeze_436 = None
    mul_956: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_440);  sub_303 = unsqueeze_440 = None
    mul_957: "f32[512]" = torch.ops.aten.mul.Tensor(sum_178, squeeze_19);  sum_178 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_316: "f32[512, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_957, [512, 1, 1, 1]);  mul_957 = None
    mul_958: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_316, 0.11175808310508728);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_317: "f32[512, 256, 1, 1]" = torch.ops.aten.reshape.default(mul_956, [512, 256, 1, 1]);  mul_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    avg_pool2d_backward_2: "f32[8, 256, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(getitem_321, mul_39, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_321 = mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:101, code: return self.conv(self.pool(x))
    add_134: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(getitem_318, avg_pool2d_backward_2);  getitem_318 = avg_pool2d_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_959: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_134, 0.9805806756909201);  add_134 = None
    mul_962: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_959, mul_961);  mul_959 = mul_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:199, code: out = out * self.alpha + shortcut
    mul_963: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_962, 0.2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:194, code: out = self.attn_gain * self.attn_last(out)
    mul_964: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_963, 2.0);  mul_963 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_965: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_964, convolution_8);  convolution_8 = None
    mul_966: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_964, sigmoid_7);  mul_964 = None
    sum_179: "f32[8, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_965, [2, 3], True);  mul_965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_305: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_7)
    mul_967: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_7, sub_305);  sigmoid_7 = sub_305 = None
    mul_968: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sum_179, mul_967);  sum_179 = mul_967 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_180: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_968, [0, 2, 3])
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_968, relu, primals_174, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_968 = primals_174 = None
    getitem_324: "f32[8, 64, 1, 1]" = convolution_backward_70[0]
    getitem_325: "f32[256, 64, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_11: "b8[8, 64, 1, 1]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_11: "f32[8, 64, 1, 1]" = torch.ops.aten.where.self(le_11, full_default_1, getitem_324);  le_11 = full_default_1 = getitem_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_181: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(where_11, mean, primals_172, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = mean = primals_172 = None
    getitem_327: "f32[8, 256, 1, 1]" = convolution_backward_71[0]
    getitem_328: "f32[64, 256, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[8, 256, 56, 56]" = torch.ops.aten.expand.default(getitem_327, [8, 256, 56, 56]);  getitem_327 = None
    div_12: "f32[8, 256, 56, 56]" = torch.ops.aten.div.Scalar(expand_12, 3136);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_136: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_966, div_12);  mul_966 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_182: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_136, [0, 2, 3])
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(add_136, mul_31, view_26, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_136 = mul_31 = view_26 = None
    getitem_330: "f32[8, 64, 56, 56]" = convolution_backward_72[0]
    getitem_331: "f32[256, 64, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_318: "f32[1, 256, 64]" = torch.ops.aten.reshape.default(getitem_331, [1, 256, 64]);  getitem_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_183: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_318, [0, 2])
    sub_306: "f32[1, 256, 64]" = torch.ops.aten.sub.Tensor(view_24, unsqueeze_442);  view_24 = unsqueeze_442 = None
    mul_969: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(view_318, sub_306)
    sum_184: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_969, [0, 2]);  mul_969 = None
    mul_970: "f32[256]" = torch.ops.aten.mul.Tensor(sum_183, 0.015625);  sum_183 = None
    unsqueeze_443: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_970, 0);  mul_970 = None
    unsqueeze_444: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    mul_971: "f32[256]" = torch.ops.aten.mul.Tensor(sum_184, 0.015625)
    mul_972: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, squeeze_17)
    mul_973: "f32[256]" = torch.ops.aten.mul.Tensor(mul_971, mul_972);  mul_971 = mul_972 = None
    unsqueeze_445: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_973, 0);  mul_973 = None
    unsqueeze_446: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    mul_974: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, view_25);  view_25 = None
    unsqueeze_447: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
    unsqueeze_448: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
    mul_975: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_446);  sub_306 = unsqueeze_446 = None
    sub_308: "f32[1, 256, 64]" = torch.ops.aten.sub.Tensor(view_318, mul_975);  view_318 = mul_975 = None
    sub_309: "f32[1, 256, 64]" = torch.ops.aten.sub.Tensor(sub_308, unsqueeze_444);  sub_308 = unsqueeze_444 = None
    mul_976: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_448);  sub_309 = unsqueeze_448 = None
    mul_977: "f32[256]" = torch.ops.aten.mul.Tensor(sum_184, squeeze_17);  sum_184 = squeeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_319: "f32[256, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_977, [256, 1, 1, 1]);  mul_977 = None
    mul_978: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_319, 0.22351616621017456);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_320: "f32[256, 64, 1, 1]" = torch.ops.aten.reshape.default(mul_976, [256, 64, 1, 1]);  mul_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:192, code: out = self.conv3(self.act3(out))
    full_default_57: "f32[8, 64, 56, 56]" = torch.ops.aten.full.default([8, 64, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_310: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(full_default_57, sigmoid_6)
    mul_979: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_7, sub_310);  convolution_7 = sub_310 = None
    add_137: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Scalar(mul_979, 1);  mul_979 = None
    mul_980: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_6, add_137);  sigmoid_6 = add_137 = None
    mul_981: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_330, mul_980);  getitem_330 = mul_980 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_185: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_981, [0, 2, 3])
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_981, mul_27, view_23, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_981 = mul_27 = view_23 = None
    getitem_333: "f32[8, 64, 56, 56]" = convolution_backward_73[0]
    getitem_334: "f32[64, 64, 3, 3]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_321: "f32[1, 64, 576]" = torch.ops.aten.reshape.default(getitem_334, [1, 64, 576]);  getitem_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_186: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_321, [0, 2])
    sub_311: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_21, unsqueeze_450);  view_21 = unsqueeze_450 = None
    mul_982: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(view_321, sub_311)
    sum_187: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_982, [0, 2]);  mul_982 = None
    mul_983: "f32[64]" = torch.ops.aten.mul.Tensor(sum_186, 0.001736111111111111);  sum_186 = None
    unsqueeze_451: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_452: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    mul_984: "f32[64]" = torch.ops.aten.mul.Tensor(sum_187, 0.001736111111111111)
    mul_985: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_15, squeeze_15)
    mul_986: "f32[64]" = torch.ops.aten.mul.Tensor(mul_984, mul_985);  mul_984 = mul_985 = None
    unsqueeze_453: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_986, 0);  mul_986 = None
    unsqueeze_454: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
    mul_987: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_15, view_22);  view_22 = None
    unsqueeze_455: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_987, 0);  mul_987 = None
    unsqueeze_456: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    mul_988: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_454);  sub_311 = unsqueeze_454 = None
    sub_313: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_321, mul_988);  view_321 = mul_988 = None
    sub_314: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(sub_313, unsqueeze_452);  sub_313 = unsqueeze_452 = None
    mul_989: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_456);  sub_314 = unsqueeze_456 = None
    mul_990: "f32[64]" = torch.ops.aten.mul.Tensor(sum_187, squeeze_15);  sum_187 = squeeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_322: "f32[64, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_990, [64, 1, 1, 1]);  mul_990 = None
    mul_991: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_322, 0.07450538873672485);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_323: "f32[64, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_989, [64, 64, 3, 3]);  mul_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:189, code: out = self.conv2b(self.act2b(out))
    sigmoid_110: "f32[8, 64, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_6)
    sub_315: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(full_default_57, sigmoid_110)
    mul_992: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_6, sub_315);  convolution_6 = sub_315 = None
    add_138: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Scalar(mul_992, 1);  mul_992 = None
    mul_993: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_110, add_138);  sigmoid_110 = add_138 = None
    mul_994: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_333, mul_993);  getitem_333 = mul_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_188: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_994, [0, 2, 3])
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_994, mul_23, view_20, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_994 = mul_23 = view_20 = None
    getitem_336: "f32[8, 64, 56, 56]" = convolution_backward_74[0]
    getitem_337: "f32[64, 64, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_324: "f32[1, 64, 576]" = torch.ops.aten.reshape.default(getitem_337, [1, 64, 576]);  getitem_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_189: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_324, [0, 2])
    sub_316: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_18, unsqueeze_458);  view_18 = unsqueeze_458 = None
    mul_995: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(view_324, sub_316)
    sum_190: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_995, [0, 2]);  mul_995 = None
    mul_996: "f32[64]" = torch.ops.aten.mul.Tensor(sum_189, 0.001736111111111111);  sum_189 = None
    unsqueeze_459: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_996, 0);  mul_996 = None
    unsqueeze_460: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
    mul_997: "f32[64]" = torch.ops.aten.mul.Tensor(sum_190, 0.001736111111111111)
    mul_998: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_999: "f32[64]" = torch.ops.aten.mul.Tensor(mul_997, mul_998);  mul_997 = mul_998 = None
    unsqueeze_461: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_999, 0);  mul_999 = None
    unsqueeze_462: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    mul_1000: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, view_19);  view_19 = None
    unsqueeze_463: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
    unsqueeze_464: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    mul_1001: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_462);  sub_316 = unsqueeze_462 = None
    sub_318: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_324, mul_1001);  view_324 = mul_1001 = None
    sub_319: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(sub_318, unsqueeze_460);  sub_318 = unsqueeze_460 = None
    mul_1002: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_464);  sub_319 = unsqueeze_464 = None
    mul_1003: "f32[64]" = torch.ops.aten.mul.Tensor(sum_190, squeeze_13);  sum_190 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_325: "f32[64, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_1003, [64, 1, 1, 1]);  mul_1003 = None
    mul_1004: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_325, 0.07450538873672485);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_326: "f32[64, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_1002, [64, 64, 3, 3]);  mul_1002 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:187, code: out = self.conv2(self.act2(out))
    sigmoid_111: "f32[8, 64, 56, 56]" = torch.ops.aten.sigmoid.default(convolution_5)
    sub_320: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(full_default_57, sigmoid_111);  full_default_57 = None
    mul_1005: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_5, sub_320);  convolution_5 = sub_320 = None
    add_139: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Scalar(mul_1005, 1);  mul_1005 = None
    mul_1006: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_111, add_139);  sigmoid_111 = add_139 = None
    mul_1007: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_336, mul_1006);  getitem_336 = mul_1006 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_191: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1007, [0, 2, 3])
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1007, mul_16, view_17, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1007 = view_17 = None
    getitem_339: "f32[8, 128, 56, 56]" = convolution_backward_75[0]
    getitem_340: "f32[64, 128, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_327: "f32[1, 64, 128]" = torch.ops.aten.reshape.default(getitem_340, [1, 64, 128]);  getitem_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_192: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_327, [0, 2])
    sub_321: "f32[1, 64, 128]" = torch.ops.aten.sub.Tensor(view_15, unsqueeze_466);  view_15 = unsqueeze_466 = None
    mul_1008: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(view_327, sub_321)
    sum_193: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1008, [0, 2]);  mul_1008 = None
    mul_1009: "f32[64]" = torch.ops.aten.mul.Tensor(sum_192, 0.0078125);  sum_192 = None
    unsqueeze_467: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1009, 0);  mul_1009 = None
    unsqueeze_468: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    mul_1010: "f32[64]" = torch.ops.aten.mul.Tensor(sum_193, 0.0078125)
    mul_1011: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_11, squeeze_11)
    mul_1012: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1010, mul_1011);  mul_1010 = mul_1011 = None
    unsqueeze_469: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1012, 0);  mul_1012 = None
    unsqueeze_470: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    mul_1013: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_11, view_16);  view_16 = None
    unsqueeze_471: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
    unsqueeze_472: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
    mul_1014: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_470);  sub_321 = unsqueeze_470 = None
    sub_323: "f32[1, 64, 128]" = torch.ops.aten.sub.Tensor(view_327, mul_1014);  view_327 = mul_1014 = None
    sub_324: "f32[1, 64, 128]" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_468);  sub_323 = unsqueeze_468 = None
    mul_1015: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_472);  sub_324 = unsqueeze_472 = None
    mul_1016: "f32[64]" = torch.ops.aten.mul.Tensor(sum_193, squeeze_11);  sum_193 = squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_328: "f32[64, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_1016, [64, 1, 1, 1]);  mul_1016 = None
    mul_1017: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_328, 0.1580497968320339);  view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_329: "f32[64, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_1015, [64, 128, 1, 1]);  mul_1015 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_194: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_962, [0, 2, 3])
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_962, mul_16, view_14, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_962 = mul_16 = view_14 = None
    getitem_342: "f32[8, 128, 56, 56]" = convolution_backward_76[0]
    getitem_343: "f32[256, 128, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    add_140: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(getitem_339, getitem_342);  getitem_339 = getitem_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_330: "f32[1, 256, 128]" = torch.ops.aten.reshape.default(getitem_343, [1, 256, 128]);  getitem_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_195: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_330, [0, 2])
    sub_325: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_12, unsqueeze_474);  view_12 = unsqueeze_474 = None
    mul_1018: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(view_330, sub_325)
    sum_196: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1018, [0, 2]);  mul_1018 = None
    mul_1019: "f32[256]" = torch.ops.aten.mul.Tensor(sum_195, 0.0078125);  sum_195 = None
    unsqueeze_475: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
    unsqueeze_476: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    mul_1020: "f32[256]" = torch.ops.aten.mul.Tensor(sum_196, 0.0078125)
    mul_1021: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_9, squeeze_9)
    mul_1022: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1020, mul_1021);  mul_1020 = mul_1021 = None
    unsqueeze_477: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
    unsqueeze_478: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
    mul_1023: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_9, view_13);  view_13 = None
    unsqueeze_479: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
    unsqueeze_480: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    mul_1024: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_478);  sub_325 = unsqueeze_478 = None
    sub_327: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_330, mul_1024);  view_330 = mul_1024 = None
    sub_328: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(sub_327, unsqueeze_476);  sub_327 = unsqueeze_476 = None
    mul_1025: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_480);  sub_328 = unsqueeze_480 = None
    mul_1026: "f32[256]" = torch.ops.aten.mul.Tensor(sum_196, squeeze_9);  sum_196 = squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_331: "f32[256, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_1026, [256, 1, 1, 1]);  mul_1026 = None
    mul_1027: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_331, 0.1580497968320339);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_332: "f32[256, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_1025, [256, 128, 1, 1]);  mul_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:178, code: out = self.act1(x) * self.beta
    mul_1028: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(add_140, 1.0);  add_140 = None
    sub_329: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(full_default_54, sigmoid_3);  full_default_54 = None
    mul_1029: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_3, sub_329);  convolution_3 = sub_329 = None
    add_141: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Scalar(mul_1029, 1);  mul_1029 = None
    mul_1030: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_3, add_141);  sigmoid_3 = add_141 = None
    mul_1031: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_1028, mul_1030);  mul_1028 = mul_1030 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_197: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1031, [0, 2, 3])
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1031, mul_11, view_11, [128], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1031 = mul_11 = view_11 = None
    getitem_345: "f32[8, 64, 112, 112]" = convolution_backward_77[0]
    getitem_346: "f32[128, 64, 3, 3]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_333: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(getitem_346, [1, 128, 576]);  getitem_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_198: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_333, [0, 2])
    sub_330: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_9, unsqueeze_482);  view_9 = unsqueeze_482 = None
    mul_1032: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(view_333, sub_330)
    sum_199: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1032, [0, 2]);  mul_1032 = None
    mul_1033: "f32[128]" = torch.ops.aten.mul.Tensor(sum_198, 0.001736111111111111);  sum_198 = None
    unsqueeze_483: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1033, 0);  mul_1033 = None
    unsqueeze_484: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
    mul_1034: "f32[128]" = torch.ops.aten.mul.Tensor(sum_199, 0.001736111111111111)
    mul_1035: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1036: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1034, mul_1035);  mul_1034 = mul_1035 = None
    unsqueeze_485: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1036, 0);  mul_1036 = None
    unsqueeze_486: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    mul_1037: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, view_10);  view_10 = None
    unsqueeze_487: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1037, 0);  mul_1037 = None
    unsqueeze_488: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    mul_1038: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_486);  sub_330 = unsqueeze_486 = None
    sub_332: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_333, mul_1038);  view_333 = mul_1038 = None
    sub_333: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(sub_332, unsqueeze_484);  sub_332 = unsqueeze_484 = None
    mul_1039: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_488);  sub_333 = unsqueeze_488 = None
    mul_1040: "f32[128]" = torch.ops.aten.mul.Tensor(sum_199, squeeze_7);  sum_199 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_334: "f32[128, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_1040, [128, 1, 1, 1]);  mul_1040 = None
    mul_1041: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_334, 0.07450538873672485);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_335: "f32[128, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_1039, [128, 64, 3, 3]);  mul_1039 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    sigmoid_113: "f32[8, 64, 112, 112]" = torch.ops.aten.sigmoid.default(convolution_2)
    full_default_61: "f32[8, 64, 112, 112]" = torch.ops.aten.full.default([8, 64, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_334: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(full_default_61, sigmoid_113);  full_default_61 = None
    mul_1042: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(convolution_2, sub_334);  convolution_2 = sub_334 = None
    add_142: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Scalar(mul_1042, 1);  mul_1042 = None
    mul_1043: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_113, add_142);  sigmoid_113 = add_142 = None
    mul_1044: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_345, mul_1043);  getitem_345 = mul_1043 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_200: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1044, [0, 2, 3])
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1044, mul_7, view_8, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1044 = mul_7 = view_8 = None
    getitem_348: "f32[8, 32, 112, 112]" = convolution_backward_78[0]
    getitem_349: "f32[64, 32, 3, 3]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_336: "f32[1, 64, 288]" = torch.ops.aten.reshape.default(getitem_349, [1, 64, 288]);  getitem_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_201: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_336, [0, 2])
    sub_335: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_6, unsqueeze_490);  view_6 = unsqueeze_490 = None
    mul_1045: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(view_336, sub_335)
    sum_202: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1045, [0, 2]);  mul_1045 = None
    mul_1046: "f32[64]" = torch.ops.aten.mul.Tensor(sum_201, 0.003472222222222222);  sum_201 = None
    unsqueeze_491: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
    unsqueeze_492: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    mul_1047: "f32[64]" = torch.ops.aten.mul.Tensor(sum_202, 0.003472222222222222)
    mul_1048: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, squeeze_5)
    mul_1049: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1047, mul_1048);  mul_1047 = mul_1048 = None
    unsqueeze_493: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1049, 0);  mul_1049 = None
    unsqueeze_494: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    mul_1050: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, view_7);  view_7 = None
    unsqueeze_495: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1050, 0);  mul_1050 = None
    unsqueeze_496: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
    mul_1051: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_494);  sub_335 = unsqueeze_494 = None
    sub_337: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_336, mul_1051);  view_336 = mul_1051 = None
    sub_338: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(sub_337, unsqueeze_492);  sub_337 = unsqueeze_492 = None
    mul_1052: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_496);  sub_338 = unsqueeze_496 = None
    mul_1053: "f32[64]" = torch.ops.aten.mul.Tensor(sum_202, squeeze_5);  sum_202 = squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_337: "f32[64, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_1053, [64, 1, 1, 1]);  mul_1053 = None
    mul_1054: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_337, 0.10536653122135592);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_338: "f32[64, 32, 3, 3]" = torch.ops.aten.reshape.default(mul_1052, [64, 32, 3, 3]);  mul_1052 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    sigmoid_114: "f32[8, 32, 112, 112]" = torch.ops.aten.sigmoid.default(convolution_1)
    full_default_62: "f32[8, 32, 112, 112]" = torch.ops.aten.full.default([8, 32, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_339: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(full_default_62, sigmoid_114);  full_default_62 = None
    mul_1055: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(convolution_1, sub_339);  convolution_1 = sub_339 = None
    add_143: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Scalar(mul_1055, 1);  mul_1055 = None
    mul_1056: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_114, add_143);  sigmoid_114 = add_143 = None
    mul_1057: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_348, mul_1056);  getitem_348 = mul_1056 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_203: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1057, [0, 2, 3])
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1057, mul_3, view_5, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1057 = mul_3 = view_5 = None
    getitem_351: "f32[8, 16, 112, 112]" = convolution_backward_79[0]
    getitem_352: "f32[32, 16, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_339: "f32[1, 32, 144]" = torch.ops.aten.reshape.default(getitem_352, [1, 32, 144]);  getitem_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_204: "f32[32]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 2])
    sub_340: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_3, unsqueeze_498);  view_3 = unsqueeze_498 = None
    mul_1058: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(view_339, sub_340)
    sum_205: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1058, [0, 2]);  mul_1058 = None
    mul_1059: "f32[32]" = torch.ops.aten.mul.Tensor(sum_204, 0.006944444444444444);  sum_204 = None
    unsqueeze_499: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1059, 0);  mul_1059 = None
    unsqueeze_500: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    mul_1060: "f32[32]" = torch.ops.aten.mul.Tensor(sum_205, 0.006944444444444444)
    mul_1061: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, squeeze_3)
    mul_1062: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1060, mul_1061);  mul_1060 = mul_1061 = None
    unsqueeze_501: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1062, 0);  mul_1062 = None
    unsqueeze_502: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
    mul_1063: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, view_4);  view_4 = None
    unsqueeze_503: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1063, 0);  mul_1063 = None
    unsqueeze_504: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    mul_1064: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_502);  sub_340 = unsqueeze_502 = None
    sub_342: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_339, mul_1064);  view_339 = mul_1064 = None
    sub_343: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(sub_342, unsqueeze_500);  sub_342 = unsqueeze_500 = None
    mul_1065: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_504);  sub_343 = unsqueeze_504 = None
    mul_1066: "f32[32]" = torch.ops.aten.mul.Tensor(sum_205, squeeze_3);  sum_205 = squeeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_340: "f32[32, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_1066, [32, 1, 1, 1]);  mul_1066 = None
    mul_1067: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_340, 0.1490107774734497);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_341: "f32[32, 16, 3, 3]" = torch.ops.aten.reshape.default(mul_1065, [32, 16, 3, 3]);  mul_1065 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nfnet.py:439, code: x = self.stem(x)
    sigmoid_115: "f32[8, 16, 112, 112]" = torch.ops.aten.sigmoid.default(convolution)
    full_default_63: "f32[8, 16, 112, 112]" = torch.ops.aten.full.default([8, 16, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_344: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(full_default_63, sigmoid_115);  full_default_63 = None
    mul_1068: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(convolution, sub_344);  convolution = sub_344 = None
    add_144: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Scalar(mul_1068, 1);  mul_1068 = None
    mul_1069: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_115, add_144);  sigmoid_115 = add_144 = None
    mul_1070: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_351, mul_1069);  getitem_351 = mul_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:102, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    sum_206: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1070, [0, 2, 3])
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1070, primals_222, view_2, [16], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1070 = primals_222 = view_2 = None
    getitem_355: "f32[16, 3, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:101, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
    view_342: "f32[1, 16, 27]" = torch.ops.aten.reshape.default(getitem_355, [1, 16, 27]);  getitem_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:98, code: weight = F.batch_norm(
    sum_207: "f32[16]" = torch.ops.aten.sum.dim_IntList(view_342, [0, 2])
    sub_345: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view, unsqueeze_506);  view = unsqueeze_506 = None
    mul_1071: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(view_342, sub_345)
    sum_208: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1071, [0, 2]);  mul_1071 = None
    mul_1072: "f32[16]" = torch.ops.aten.mul.Tensor(sum_207, 0.037037037037037035);  sum_207 = None
    unsqueeze_507: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1072, 0);  mul_1072 = None
    unsqueeze_508: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
    mul_1073: "f32[16]" = torch.ops.aten.mul.Tensor(sum_208, 0.037037037037037035)
    mul_1074: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1075: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1073, mul_1074);  mul_1073 = mul_1074 = None
    unsqueeze_509: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1075, 0);  mul_1075 = None
    unsqueeze_510: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    mul_1076: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, view_1);  view_1 = None
    unsqueeze_511: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1076, 0);  mul_1076 = None
    unsqueeze_512: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    mul_1077: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_510);  sub_345 = unsqueeze_510 = None
    sub_347: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view_342, mul_1077);  view_342 = mul_1077 = None
    sub_348: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(sub_347, unsqueeze_508);  sub_347 = unsqueeze_508 = None
    mul_1078: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_512);  sub_348 = unsqueeze_512 = None
    mul_1079: "f32[16]" = torch.ops.aten.mul.Tensor(sum_208, squeeze_1);  sum_208 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:100, code: weight=(self.gain * self.scale).view(-1),
    view_343: "f32[16, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_1079, [16, 1, 1, 1]);  mul_1079 = None
    mul_1080: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_343, 0.34412564994580647);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/std_conv.py:99, code: self.weight.reshape(1, self.out_channels, -1), None, None,
    view_344: "f32[16, 3, 3, 3]" = torch.ops.aten.reshape.default(mul_1078, [16, 3, 3, 3]);  mul_1078 = None
    return [view_344, mul_1080, sum_206, view_341, mul_1067, sum_203, view_338, mul_1054, sum_200, view_335, mul_1041, sum_197, view_332, mul_1027, sum_194, view_329, mul_1017, sum_191, view_326, mul_1004, sum_188, view_323, mul_991, sum_185, view_320, mul_978, sum_182, view_317, mul_958, sum_176, view_314, mul_948, sum_173, view_311, mul_935, sum_170, view_308, mul_922, sum_167, view_305, mul_909, sum_164, view_302, mul_889, sum_158, view_299, mul_876, sum_155, view_296, mul_863, sum_152, view_293, mul_850, sum_149, view_290, mul_830, sum_143, view_287, mul_820, sum_140, view_284, mul_807, sum_137, view_281, mul_794, sum_134, view_278, mul_781, sum_131, view_275, mul_761, sum_125, view_272, mul_748, sum_122, view_269, mul_735, sum_119, view_266, mul_722, sum_116, view_263, mul_702, sum_110, view_260, mul_689, sum_107, view_257, mul_676, sum_104, view_254, mul_663, sum_101, view_251, mul_643, sum_95, view_248, mul_630, sum_92, view_245, mul_617, sum_89, view_242, mul_604, sum_86, view_239, mul_584, sum_80, view_236, mul_571, sum_77, view_233, mul_558, sum_74, view_230, mul_545, sum_71, view_227, mul_525, sum_65, view_224, mul_512, sum_62, view_221, mul_499, sum_59, view_218, mul_486, sum_56, view_215, mul_466, sum_50, view_212, mul_456, sum_47, view_209, mul_443, sum_44, view_206, mul_430, sum_41, view_203, mul_417, sum_38, view_200, mul_397, sum_32, view_197, mul_384, sum_29, view_194, mul_371, sum_26, view_191, mul_358, sum_23, view_188, mul_338, sum_17, view_185, mul_325, sum_14, view_182, mul_312, sum_11, view_179, mul_299, sum_8, view_176, mul_283, sum_2, getitem_328, sum_181, getitem_325, sum_180, getitem_307, sum_163, getitem_304, sum_162, getitem_289, sum_148, getitem_286, sum_147, getitem_268, sum_130, getitem_265, sum_129, getitem_250, sum_115, getitem_247, sum_114, getitem_232, sum_100, getitem_229, sum_99, getitem_214, sum_85, getitem_211, sum_84, getitem_196, sum_70, getitem_193, sum_69, getitem_178, sum_55, getitem_175, sum_54, getitem_157, sum_37, getitem_154, sum_36, getitem_139, sum_22, getitem_136, sum_21, getitem_121, sum_7, getitem_118, sum_6, permute_4, view_172, None]
    