from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_3: "f32[32]", primals_5: "f32[16]", primals_7: "f32[96]", primals_9: "f32[96]", primals_11: "f32[24]", primals_13: "f32[144]", primals_15: "f32[144]", primals_17: "f32[24]", primals_19: "f32[144]", primals_21: "f32[144]", primals_23: "f32[40]", primals_25: "f32[240]", primals_27: "f32[240]", primals_29: "f32[40]", primals_31: "f32[240]", primals_33: "f32[240]", primals_35: "f32[80]", primals_37: "f32[480]", primals_39: "f32[480]", primals_41: "f32[80]", primals_43: "f32[480]", primals_45: "f32[480]", primals_47: "f32[80]", primals_49: "f32[480]", primals_51: "f32[480]", primals_53: "f32[80]", primals_55: "f32[480]", primals_57: "f32[480]", primals_59: "f32[112]", primals_61: "f32[672]", primals_63: "f32[672]", primals_65: "f32[112]", primals_67: "f32[672]", primals_69: "f32[672]", primals_71: "f32[112]", primals_73: "f32[672]", primals_75: "f32[672]", primals_77: "f32[112]", primals_79: "f32[672]", primals_81: "f32[672]", primals_83: "f32[192]", primals_85: "f32[1152]", primals_87: "f32[1152]", primals_89: "f32[192]", primals_91: "f32[1152]", primals_93: "f32[1152]", primals_95: "f32[192]", primals_97: "f32[1152]", primals_99: "f32[1152]", primals_101: "f32[192]", primals_103: "f32[1152]", primals_105: "f32[1152]", primals_107: "f32[192]", primals_109: "f32[1152]", primals_111: "f32[1152]", primals_113: "f32[320]", primals_115: "f32[1280]", primals_117: "f32[32, 3, 3, 3]", primals_118: "f32[32, 1, 3, 3]", primals_119: "f32[8, 32, 1, 1]", primals_121: "f32[32, 8, 1, 1]", primals_123: "f32[16, 32, 1, 1]", primals_124: "f32[96, 16, 1, 1]", primals_125: "f32[96, 1, 3, 3]", primals_126: "f32[4, 96, 1, 1]", primals_128: "f32[96, 4, 1, 1]", primals_130: "f32[24, 96, 1, 1]", primals_131: "f32[144, 24, 1, 1]", primals_132: "f32[144, 1, 3, 3]", primals_133: "f32[6, 144, 1, 1]", primals_135: "f32[144, 6, 1, 1]", primals_137: "f32[24, 144, 1, 1]", primals_138: "f32[144, 24, 1, 1]", primals_139: "f32[144, 1, 5, 5]", primals_140: "f32[6, 144, 1, 1]", primals_142: "f32[144, 6, 1, 1]", primals_144: "f32[40, 144, 1, 1]", primals_145: "f32[240, 40, 1, 1]", primals_146: "f32[240, 1, 5, 5]", primals_147: "f32[10, 240, 1, 1]", primals_149: "f32[240, 10, 1, 1]", primals_151: "f32[40, 240, 1, 1]", primals_152: "f32[240, 40, 1, 1]", primals_153: "f32[240, 1, 3, 3]", primals_154: "f32[10, 240, 1, 1]", primals_156: "f32[240, 10, 1, 1]", primals_158: "f32[80, 240, 1, 1]", primals_159: "f32[480, 80, 1, 1]", primals_160: "f32[480, 1, 3, 3]", primals_161: "f32[20, 480, 1, 1]", primals_163: "f32[480, 20, 1, 1]", primals_165: "f32[80, 480, 1, 1]", primals_166: "f32[480, 80, 1, 1]", primals_167: "f32[480, 1, 3, 3]", primals_168: "f32[20, 480, 1, 1]", primals_170: "f32[480, 20, 1, 1]", primals_172: "f32[80, 480, 1, 1]", primals_173: "f32[480, 80, 1, 1]", primals_174: "f32[480, 1, 3, 3]", primals_175: "f32[20, 480, 1, 1]", primals_177: "f32[480, 20, 1, 1]", primals_179: "f32[80, 480, 1, 1]", primals_180: "f32[480, 80, 1, 1]", primals_181: "f32[480, 1, 5, 5]", primals_182: "f32[20, 480, 1, 1]", primals_184: "f32[480, 20, 1, 1]", primals_186: "f32[112, 480, 1, 1]", primals_187: "f32[672, 112, 1, 1]", primals_188: "f32[672, 1, 5, 5]", primals_189: "f32[28, 672, 1, 1]", primals_191: "f32[672, 28, 1, 1]", primals_193: "f32[112, 672, 1, 1]", primals_194: "f32[672, 112, 1, 1]", primals_195: "f32[672, 1, 5, 5]", primals_196: "f32[28, 672, 1, 1]", primals_198: "f32[672, 28, 1, 1]", primals_200: "f32[112, 672, 1, 1]", primals_201: "f32[672, 112, 1, 1]", primals_202: "f32[672, 1, 5, 5]", primals_203: "f32[28, 672, 1, 1]", primals_205: "f32[672, 28, 1, 1]", primals_207: "f32[112, 672, 1, 1]", primals_208: "f32[672, 112, 1, 1]", primals_209: "f32[672, 1, 5, 5]", primals_210: "f32[28, 672, 1, 1]", primals_212: "f32[672, 28, 1, 1]", primals_214: "f32[192, 672, 1, 1]", primals_215: "f32[1152, 192, 1, 1]", primals_216: "f32[1152, 1, 5, 5]", primals_217: "f32[48, 1152, 1, 1]", primals_219: "f32[1152, 48, 1, 1]", primals_221: "f32[192, 1152, 1, 1]", primals_222: "f32[1152, 192, 1, 1]", primals_223: "f32[1152, 1, 5, 5]", primals_224: "f32[48, 1152, 1, 1]", primals_226: "f32[1152, 48, 1, 1]", primals_228: "f32[192, 1152, 1, 1]", primals_229: "f32[1152, 192, 1, 1]", primals_230: "f32[1152, 1, 5, 5]", primals_231: "f32[48, 1152, 1, 1]", primals_233: "f32[1152, 48, 1, 1]", primals_235: "f32[192, 1152, 1, 1]", primals_236: "f32[1152, 192, 1, 1]", primals_237: "f32[1152, 1, 5, 5]", primals_238: "f32[48, 1152, 1, 1]", primals_240: "f32[1152, 48, 1, 1]", primals_242: "f32[192, 1152, 1, 1]", primals_243: "f32[1152, 192, 1, 1]", primals_244: "f32[1152, 1, 3, 3]", primals_245: "f32[48, 1152, 1, 1]", primals_247: "f32[1152, 48, 1, 1]", primals_249: "f32[320, 1152, 1, 1]", primals_250: "f32[1280, 320, 1, 1]", primals_427: "f32[8, 3, 192, 192]", convolution: "f32[8, 32, 96, 96]", squeeze_1: "f32[32]", mul_7: "f32[8, 32, 96, 96]", convolution_1: "f32[8, 32, 96, 96]", squeeze_4: "f32[32]", add_9: "f32[8, 32, 96, 96]", mean: "f32[8, 32, 1, 1]", convolution_2: "f32[8, 8, 1, 1]", mul_16: "f32[8, 8, 1, 1]", convolution_3: "f32[8, 32, 1, 1]", mul_17: "f32[8, 32, 96, 96]", convolution_4: "f32[8, 16, 96, 96]", squeeze_7: "f32[16]", add_14: "f32[8, 16, 96, 96]", convolution_5: "f32[8, 96, 96, 96]", squeeze_10: "f32[96]", mul_32: "f32[8, 96, 96, 96]", convolution_6: "f32[8, 96, 48, 48]", squeeze_13: "f32[96]", add_24: "f32[8, 96, 48, 48]", mean_1: "f32[8, 96, 1, 1]", convolution_7: "f32[8, 4, 1, 1]", mul_41: "f32[8, 4, 1, 1]", convolution_8: "f32[8, 96, 1, 1]", mul_42: "f32[8, 96, 48, 48]", convolution_9: "f32[8, 24, 48, 48]", squeeze_16: "f32[24]", add_29: "f32[8, 24, 48, 48]", convolution_10: "f32[8, 144, 48, 48]", squeeze_19: "f32[144]", mul_57: "f32[8, 144, 48, 48]", convolution_11: "f32[8, 144, 48, 48]", squeeze_22: "f32[144]", add_39: "f32[8, 144, 48, 48]", mean_2: "f32[8, 144, 1, 1]", convolution_12: "f32[8, 6, 1, 1]", mul_66: "f32[8, 6, 1, 1]", convolution_13: "f32[8, 144, 1, 1]", mul_67: "f32[8, 144, 48, 48]", convolution_14: "f32[8, 24, 48, 48]", squeeze_25: "f32[24]", add_45: "f32[8, 24, 48, 48]", convolution_15: "f32[8, 144, 48, 48]", squeeze_28: "f32[144]", mul_82: "f32[8, 144, 48, 48]", convolution_16: "f32[8, 144, 24, 24]", squeeze_31: "f32[144]", add_55: "f32[8, 144, 24, 24]", mean_3: "f32[8, 144, 1, 1]", convolution_17: "f32[8, 6, 1, 1]", mul_91: "f32[8, 6, 1, 1]", convolution_18: "f32[8, 144, 1, 1]", mul_92: "f32[8, 144, 24, 24]", convolution_19: "f32[8, 40, 24, 24]", squeeze_34: "f32[40]", add_60: "f32[8, 40, 24, 24]", convolution_20: "f32[8, 240, 24, 24]", squeeze_37: "f32[240]", mul_107: "f32[8, 240, 24, 24]", convolution_21: "f32[8, 240, 24, 24]", squeeze_40: "f32[240]", add_70: "f32[8, 240, 24, 24]", mean_4: "f32[8, 240, 1, 1]", convolution_22: "f32[8, 10, 1, 1]", mul_116: "f32[8, 10, 1, 1]", convolution_23: "f32[8, 240, 1, 1]", mul_117: "f32[8, 240, 24, 24]", convolution_24: "f32[8, 40, 24, 24]", squeeze_43: "f32[40]", add_76: "f32[8, 40, 24, 24]", convolution_25: "f32[8, 240, 24, 24]", squeeze_46: "f32[240]", mul_132: "f32[8, 240, 24, 24]", convolution_26: "f32[8, 240, 12, 12]", squeeze_49: "f32[240]", add_86: "f32[8, 240, 12, 12]", mean_5: "f32[8, 240, 1, 1]", convolution_27: "f32[8, 10, 1, 1]", mul_141: "f32[8, 10, 1, 1]", convolution_28: "f32[8, 240, 1, 1]", mul_142: "f32[8, 240, 12, 12]", convolution_29: "f32[8, 80, 12, 12]", squeeze_52: "f32[80]", add_91: "f32[8, 80, 12, 12]", convolution_30: "f32[8, 480, 12, 12]", squeeze_55: "f32[480]", mul_157: "f32[8, 480, 12, 12]", convolution_31: "f32[8, 480, 12, 12]", squeeze_58: "f32[480]", add_101: "f32[8, 480, 12, 12]", mean_6: "f32[8, 480, 1, 1]", convolution_32: "f32[8, 20, 1, 1]", mul_166: "f32[8, 20, 1, 1]", convolution_33: "f32[8, 480, 1, 1]", mul_167: "f32[8, 480, 12, 12]", convolution_34: "f32[8, 80, 12, 12]", squeeze_61: "f32[80]", add_107: "f32[8, 80, 12, 12]", convolution_35: "f32[8, 480, 12, 12]", squeeze_64: "f32[480]", mul_182: "f32[8, 480, 12, 12]", convolution_36: "f32[8, 480, 12, 12]", squeeze_67: "f32[480]", add_117: "f32[8, 480, 12, 12]", mean_7: "f32[8, 480, 1, 1]", convolution_37: "f32[8, 20, 1, 1]", mul_191: "f32[8, 20, 1, 1]", convolution_38: "f32[8, 480, 1, 1]", mul_192: "f32[8, 480, 12, 12]", convolution_39: "f32[8, 80, 12, 12]", squeeze_70: "f32[80]", add_123: "f32[8, 80, 12, 12]", convolution_40: "f32[8, 480, 12, 12]", squeeze_73: "f32[480]", mul_207: "f32[8, 480, 12, 12]", convolution_41: "f32[8, 480, 12, 12]", squeeze_76: "f32[480]", add_133: "f32[8, 480, 12, 12]", mean_8: "f32[8, 480, 1, 1]", convolution_42: "f32[8, 20, 1, 1]", mul_216: "f32[8, 20, 1, 1]", convolution_43: "f32[8, 480, 1, 1]", mul_217: "f32[8, 480, 12, 12]", convolution_44: "f32[8, 80, 12, 12]", squeeze_79: "f32[80]", add_139: "f32[8, 80, 12, 12]", convolution_45: "f32[8, 480, 12, 12]", squeeze_82: "f32[480]", mul_232: "f32[8, 480, 12, 12]", convolution_46: "f32[8, 480, 12, 12]", squeeze_85: "f32[480]", add_149: "f32[8, 480, 12, 12]", mean_9: "f32[8, 480, 1, 1]", convolution_47: "f32[8, 20, 1, 1]", mul_241: "f32[8, 20, 1, 1]", convolution_48: "f32[8, 480, 1, 1]", mul_242: "f32[8, 480, 12, 12]", convolution_49: "f32[8, 112, 12, 12]", squeeze_88: "f32[112]", add_154: "f32[8, 112, 12, 12]", convolution_50: "f32[8, 672, 12, 12]", squeeze_91: "f32[672]", mul_257: "f32[8, 672, 12, 12]", convolution_51: "f32[8, 672, 12, 12]", squeeze_94: "f32[672]", add_164: "f32[8, 672, 12, 12]", mean_10: "f32[8, 672, 1, 1]", convolution_52: "f32[8, 28, 1, 1]", mul_266: "f32[8, 28, 1, 1]", convolution_53: "f32[8, 672, 1, 1]", mul_267: "f32[8, 672, 12, 12]", convolution_54: "f32[8, 112, 12, 12]", squeeze_97: "f32[112]", add_170: "f32[8, 112, 12, 12]", convolution_55: "f32[8, 672, 12, 12]", squeeze_100: "f32[672]", mul_282: "f32[8, 672, 12, 12]", convolution_56: "f32[8, 672, 12, 12]", squeeze_103: "f32[672]", add_180: "f32[8, 672, 12, 12]", mean_11: "f32[8, 672, 1, 1]", convolution_57: "f32[8, 28, 1, 1]", mul_291: "f32[8, 28, 1, 1]", convolution_58: "f32[8, 672, 1, 1]", mul_292: "f32[8, 672, 12, 12]", convolution_59: "f32[8, 112, 12, 12]", squeeze_106: "f32[112]", add_186: "f32[8, 112, 12, 12]", convolution_60: "f32[8, 672, 12, 12]", squeeze_109: "f32[672]", mul_307: "f32[8, 672, 12, 12]", convolution_61: "f32[8, 672, 12, 12]", squeeze_112: "f32[672]", add_196: "f32[8, 672, 12, 12]", mean_12: "f32[8, 672, 1, 1]", convolution_62: "f32[8, 28, 1, 1]", mul_316: "f32[8, 28, 1, 1]", convolution_63: "f32[8, 672, 1, 1]", mul_317: "f32[8, 672, 12, 12]", convolution_64: "f32[8, 112, 12, 12]", squeeze_115: "f32[112]", add_202: "f32[8, 112, 12, 12]", convolution_65: "f32[8, 672, 12, 12]", squeeze_118: "f32[672]", mul_332: "f32[8, 672, 12, 12]", convolution_66: "f32[8, 672, 6, 6]", squeeze_121: "f32[672]", add_212: "f32[8, 672, 6, 6]", mean_13: "f32[8, 672, 1, 1]", convolution_67: "f32[8, 28, 1, 1]", mul_341: "f32[8, 28, 1, 1]", convolution_68: "f32[8, 672, 1, 1]", mul_342: "f32[8, 672, 6, 6]", convolution_69: "f32[8, 192, 6, 6]", squeeze_124: "f32[192]", add_217: "f32[8, 192, 6, 6]", convolution_70: "f32[8, 1152, 6, 6]", squeeze_127: "f32[1152]", mul_357: "f32[8, 1152, 6, 6]", convolution_71: "f32[8, 1152, 6, 6]", squeeze_130: "f32[1152]", add_227: "f32[8, 1152, 6, 6]", mean_14: "f32[8, 1152, 1, 1]", convolution_72: "f32[8, 48, 1, 1]", mul_366: "f32[8, 48, 1, 1]", convolution_73: "f32[8, 1152, 1, 1]", mul_367: "f32[8, 1152, 6, 6]", convolution_74: "f32[8, 192, 6, 6]", squeeze_133: "f32[192]", add_233: "f32[8, 192, 6, 6]", convolution_75: "f32[8, 1152, 6, 6]", squeeze_136: "f32[1152]", mul_382: "f32[8, 1152, 6, 6]", convolution_76: "f32[8, 1152, 6, 6]", squeeze_139: "f32[1152]", add_243: "f32[8, 1152, 6, 6]", mean_15: "f32[8, 1152, 1, 1]", convolution_77: "f32[8, 48, 1, 1]", mul_391: "f32[8, 48, 1, 1]", convolution_78: "f32[8, 1152, 1, 1]", mul_392: "f32[8, 1152, 6, 6]", convolution_79: "f32[8, 192, 6, 6]", squeeze_142: "f32[192]", add_249: "f32[8, 192, 6, 6]", convolution_80: "f32[8, 1152, 6, 6]", squeeze_145: "f32[1152]", mul_407: "f32[8, 1152, 6, 6]", convolution_81: "f32[8, 1152, 6, 6]", squeeze_148: "f32[1152]", add_259: "f32[8, 1152, 6, 6]", mean_16: "f32[8, 1152, 1, 1]", convolution_82: "f32[8, 48, 1, 1]", mul_416: "f32[8, 48, 1, 1]", convolution_83: "f32[8, 1152, 1, 1]", mul_417: "f32[8, 1152, 6, 6]", convolution_84: "f32[8, 192, 6, 6]", squeeze_151: "f32[192]", add_265: "f32[8, 192, 6, 6]", convolution_85: "f32[8, 1152, 6, 6]", squeeze_154: "f32[1152]", mul_432: "f32[8, 1152, 6, 6]", convolution_86: "f32[8, 1152, 6, 6]", squeeze_157: "f32[1152]", add_275: "f32[8, 1152, 6, 6]", mean_17: "f32[8, 1152, 1, 1]", convolution_87: "f32[8, 48, 1, 1]", mul_441: "f32[8, 48, 1, 1]", convolution_88: "f32[8, 1152, 1, 1]", mul_442: "f32[8, 1152, 6, 6]", convolution_89: "f32[8, 192, 6, 6]", squeeze_160: "f32[192]", add_281: "f32[8, 192, 6, 6]", convolution_90: "f32[8, 1152, 6, 6]", squeeze_163: "f32[1152]", mul_457: "f32[8, 1152, 6, 6]", convolution_91: "f32[8, 1152, 6, 6]", squeeze_166: "f32[1152]", add_291: "f32[8, 1152, 6, 6]", mean_18: "f32[8, 1152, 1, 1]", convolution_92: "f32[8, 48, 1, 1]", mul_466: "f32[8, 48, 1, 1]", convolution_93: "f32[8, 1152, 1, 1]", mul_467: "f32[8, 1152, 6, 6]", convolution_94: "f32[8, 320, 6, 6]", squeeze_169: "f32[320]", add_296: "f32[8, 320, 6, 6]", convolution_95: "f32[8, 1280, 6, 6]", squeeze_172: "f32[1280]", view: "f32[8, 1280]", permute_1: "f32[1000, 1280]", mul_484: "f32[8, 1280, 6, 6]", unsqueeze_234: "f32[1, 1280, 1, 1]", unsqueeze_246: "f32[1, 320, 1, 1]", unsqueeze_258: "f32[1, 1152, 1, 1]", mul_524: "f32[8, 1152, 6, 6]", unsqueeze_270: "f32[1, 1152, 1, 1]", unsqueeze_282: "f32[1, 192, 1, 1]", unsqueeze_294: "f32[1, 1152, 1, 1]", mul_564: "f32[8, 1152, 6, 6]", unsqueeze_306: "f32[1, 1152, 1, 1]", unsqueeze_318: "f32[1, 192, 1, 1]", unsqueeze_330: "f32[1, 1152, 1, 1]", mul_604: "f32[8, 1152, 6, 6]", unsqueeze_342: "f32[1, 1152, 1, 1]", unsqueeze_354: "f32[1, 192, 1, 1]", unsqueeze_366: "f32[1, 1152, 1, 1]", mul_644: "f32[8, 1152, 6, 6]", unsqueeze_378: "f32[1, 1152, 1, 1]", unsqueeze_390: "f32[1, 192, 1, 1]", unsqueeze_402: "f32[1, 1152, 1, 1]", mul_684: "f32[8, 1152, 6, 6]", unsqueeze_414: "f32[1, 1152, 1, 1]", unsqueeze_426: "f32[1, 192, 1, 1]", unsqueeze_438: "f32[1, 672, 1, 1]", mul_724: "f32[8, 672, 12, 12]", unsqueeze_450: "f32[1, 672, 1, 1]", unsqueeze_462: "f32[1, 112, 1, 1]", unsqueeze_474: "f32[1, 672, 1, 1]", mul_764: "f32[8, 672, 12, 12]", unsqueeze_486: "f32[1, 672, 1, 1]", unsqueeze_498: "f32[1, 112, 1, 1]", unsqueeze_510: "f32[1, 672, 1, 1]", mul_804: "f32[8, 672, 12, 12]", unsqueeze_522: "f32[1, 672, 1, 1]", unsqueeze_534: "f32[1, 112, 1, 1]", unsqueeze_546: "f32[1, 672, 1, 1]", mul_844: "f32[8, 672, 12, 12]", unsqueeze_558: "f32[1, 672, 1, 1]", unsqueeze_570: "f32[1, 112, 1, 1]", unsqueeze_582: "f32[1, 480, 1, 1]", mul_884: "f32[8, 480, 12, 12]", unsqueeze_594: "f32[1, 480, 1, 1]", unsqueeze_606: "f32[1, 80, 1, 1]", unsqueeze_618: "f32[1, 480, 1, 1]", mul_924: "f32[8, 480, 12, 12]", unsqueeze_630: "f32[1, 480, 1, 1]", unsqueeze_642: "f32[1, 80, 1, 1]", unsqueeze_654: "f32[1, 480, 1, 1]", mul_964: "f32[8, 480, 12, 12]", unsqueeze_666: "f32[1, 480, 1, 1]", unsqueeze_678: "f32[1, 80, 1, 1]", unsqueeze_690: "f32[1, 480, 1, 1]", mul_1004: "f32[8, 480, 12, 12]", unsqueeze_702: "f32[1, 480, 1, 1]", unsqueeze_714: "f32[1, 80, 1, 1]", unsqueeze_726: "f32[1, 240, 1, 1]", mul_1044: "f32[8, 240, 24, 24]", unsqueeze_738: "f32[1, 240, 1, 1]", unsqueeze_750: "f32[1, 40, 1, 1]", unsqueeze_762: "f32[1, 240, 1, 1]", mul_1084: "f32[8, 240, 24, 24]", unsqueeze_774: "f32[1, 240, 1, 1]", unsqueeze_786: "f32[1, 40, 1, 1]", unsqueeze_798: "f32[1, 144, 1, 1]", mul_1124: "f32[8, 144, 48, 48]", unsqueeze_810: "f32[1, 144, 1, 1]", unsqueeze_822: "f32[1, 24, 1, 1]", unsqueeze_834: "f32[1, 144, 1, 1]", mul_1164: "f32[8, 144, 48, 48]", unsqueeze_846: "f32[1, 144, 1, 1]", unsqueeze_858: "f32[1, 24, 1, 1]", unsqueeze_870: "f32[1, 96, 1, 1]", mul_1204: "f32[8, 96, 96, 96]", unsqueeze_882: "f32[1, 96, 1, 1]", unsqueeze_894: "f32[1, 16, 1, 1]", unsqueeze_906: "f32[1, 32, 1, 1]", mul_1244: "f32[8, 32, 96, 96]", unsqueeze_918: "f32[1, 32, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_1: "f32[8, 32, 96, 96]" = torch.ops.aten.clone.default(add_9)
    sigmoid_1: "f32[8, 32, 96, 96]" = torch.ops.aten.sigmoid.default(add_9)
    mul_15: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_1);  add_9 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_2: "f32[8, 8, 1, 1]" = torch.ops.aten.clone.default(convolution_2);  convolution_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_3: "f32[8, 32, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_3);  convolution_3 = None
    alias: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(sigmoid_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_4: "f32[8, 96, 48, 48]" = torch.ops.aten.clone.default(add_24)
    sigmoid_5: "f32[8, 96, 48, 48]" = torch.ops.aten.sigmoid.default(add_24)
    mul_40: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(add_24, sigmoid_5);  add_24 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_5: "f32[8, 4, 1, 1]" = torch.ops.aten.clone.default(convolution_7);  convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_7: "f32[8, 96, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_8);  convolution_8 = None
    alias_1: "f32[8, 96, 1, 1]" = torch.ops.aten.alias.default(sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_7: "f32[8, 144, 48, 48]" = torch.ops.aten.clone.default(add_39)
    sigmoid_9: "f32[8, 144, 48, 48]" = torch.ops.aten.sigmoid.default(add_39)
    mul_65: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(add_39, sigmoid_9);  add_39 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_8: "f32[8, 6, 1, 1]" = torch.ops.aten.clone.default(convolution_12);  convolution_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_11: "f32[8, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_13);  convolution_13 = None
    alias_2: "f32[8, 144, 1, 1]" = torch.ops.aten.alias.default(sigmoid_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_10: "f32[8, 144, 24, 24]" = torch.ops.aten.clone.default(add_55)
    sigmoid_13: "f32[8, 144, 24, 24]" = torch.ops.aten.sigmoid.default(add_55)
    mul_90: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(add_55, sigmoid_13);  add_55 = sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_11: "f32[8, 6, 1, 1]" = torch.ops.aten.clone.default(convolution_17);  convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_15: "f32[8, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_18);  convolution_18 = None
    alias_3: "f32[8, 144, 1, 1]" = torch.ops.aten.alias.default(sigmoid_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_13: "f32[8, 240, 24, 24]" = torch.ops.aten.clone.default(add_70)
    sigmoid_17: "f32[8, 240, 24, 24]" = torch.ops.aten.sigmoid.default(add_70)
    mul_115: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(add_70, sigmoid_17);  add_70 = sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_14: "f32[8, 10, 1, 1]" = torch.ops.aten.clone.default(convolution_22);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_19: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    alias_4: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(sigmoid_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_16: "f32[8, 240, 12, 12]" = torch.ops.aten.clone.default(add_86)
    sigmoid_21: "f32[8, 240, 12, 12]" = torch.ops.aten.sigmoid.default(add_86)
    mul_140: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(add_86, sigmoid_21);  add_86 = sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_17: "f32[8, 10, 1, 1]" = torch.ops.aten.clone.default(convolution_27);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_23: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_28);  convolution_28 = None
    alias_5: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(sigmoid_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_19: "f32[8, 480, 12, 12]" = torch.ops.aten.clone.default(add_101)
    sigmoid_25: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_101)
    mul_165: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_101, sigmoid_25);  add_101 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_20: "f32[8, 20, 1, 1]" = torch.ops.aten.clone.default(convolution_32);  convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_27: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_33);  convolution_33 = None
    alias_6: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_22: "f32[8, 480, 12, 12]" = torch.ops.aten.clone.default(add_117)
    sigmoid_29: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_117)
    mul_190: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_117, sigmoid_29);  add_117 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_23: "f32[8, 20, 1, 1]" = torch.ops.aten.clone.default(convolution_37);  convolution_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_31: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_38);  convolution_38 = None
    alias_7: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_25: "f32[8, 480, 12, 12]" = torch.ops.aten.clone.default(add_133)
    sigmoid_33: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_133)
    mul_215: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_133, sigmoid_33);  add_133 = sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_26: "f32[8, 20, 1, 1]" = torch.ops.aten.clone.default(convolution_42);  convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_35: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43);  convolution_43 = None
    alias_8: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_28: "f32[8, 480, 12, 12]" = torch.ops.aten.clone.default(add_149)
    sigmoid_37: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_149)
    mul_240: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_149, sigmoid_37);  add_149 = sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_29: "f32[8, 20, 1, 1]" = torch.ops.aten.clone.default(convolution_47);  convolution_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_39: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
    alias_9: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_31: "f32[8, 672, 12, 12]" = torch.ops.aten.clone.default(add_164)
    sigmoid_41: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_164)
    mul_265: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_164, sigmoid_41);  add_164 = sigmoid_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_32: "f32[8, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_52);  convolution_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_43: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_53);  convolution_53 = None
    alias_10: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(sigmoid_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_34: "f32[8, 672, 12, 12]" = torch.ops.aten.clone.default(add_180)
    sigmoid_45: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_180)
    mul_290: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_180, sigmoid_45);  add_180 = sigmoid_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_35: "f32[8, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_57);  convolution_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_47: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_58);  convolution_58 = None
    alias_11: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(sigmoid_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_37: "f32[8, 672, 12, 12]" = torch.ops.aten.clone.default(add_196)
    sigmoid_49: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_196)
    mul_315: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_196, sigmoid_49);  add_196 = sigmoid_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_38: "f32[8, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_62);  convolution_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_51: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_63);  convolution_63 = None
    alias_12: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(sigmoid_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_40: "f32[8, 672, 6, 6]" = torch.ops.aten.clone.default(add_212)
    sigmoid_53: "f32[8, 672, 6, 6]" = torch.ops.aten.sigmoid.default(add_212)
    mul_340: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(add_212, sigmoid_53);  add_212 = sigmoid_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_41: "f32[8, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_67);  convolution_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_55: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_68);  convolution_68 = None
    alias_13: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(sigmoid_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_43: "f32[8, 1152, 6, 6]" = torch.ops.aten.clone.default(add_227)
    sigmoid_57: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_227)
    mul_365: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_227, sigmoid_57);  add_227 = sigmoid_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_44: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_72);  convolution_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_59: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
    alias_14: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_46: "f32[8, 1152, 6, 6]" = torch.ops.aten.clone.default(add_243)
    sigmoid_61: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_243)
    mul_390: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_243, sigmoid_61);  add_243 = sigmoid_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_47: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_77);  convolution_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_63: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_78);  convolution_78 = None
    alias_15: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_49: "f32[8, 1152, 6, 6]" = torch.ops.aten.clone.default(add_259)
    sigmoid_65: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_259)
    mul_415: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_259, sigmoid_65);  add_259 = sigmoid_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_50: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_82);  convolution_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_67: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_83);  convolution_83 = None
    alias_16: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_52: "f32[8, 1152, 6, 6]" = torch.ops.aten.clone.default(add_275)
    sigmoid_69: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_275)
    mul_440: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_275, sigmoid_69);  add_275 = sigmoid_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_53: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_87);  convolution_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_71: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_88);  convolution_88 = None
    alias_17: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_55: "f32[8, 1152, 6, 6]" = torch.ops.aten.clone.default(add_291)
    sigmoid_73: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_291)
    mul_465: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_291, sigmoid_73);  add_291 = sigmoid_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_56: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_92);  convolution_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_75: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_93);  convolution_93 = None
    alias_18: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_75)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    mm: "f32[8, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 1280, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1280, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1280, 6, 6]" = torch.ops.aten.expand.default(view_2, [8, 1280, 6, 6]);  view_2 = None
    div: "f32[8, 1280, 6, 6]" = torch.ops.aten.div.Scalar(expand, 36);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_485: "f32[8, 1280, 6, 6]" = torch.ops.aten.mul.Tensor(div, mul_484);  div = mul_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_485, [0, 2, 3])
    sub_59: "f32[8, 1280, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_234);  convolution_95 = unsqueeze_234 = None
    mul_486: "f32[8, 1280, 6, 6]" = torch.ops.aten.mul.Tensor(mul_485, sub_59)
    sum_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_486, [0, 2, 3]);  mul_486 = None
    mul_487: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_2, 0.003472222222222222)
    unsqueeze_235: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
    unsqueeze_236: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
    mul_488: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, 0.003472222222222222)
    mul_489: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_490: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_488, mul_489);  mul_488 = mul_489 = None
    unsqueeze_238: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_490, 0);  mul_490 = None
    unsqueeze_239: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
    unsqueeze_240: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
    mul_491: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_115);  primals_115 = None
    unsqueeze_241: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_491, 0);  mul_491 = None
    unsqueeze_242: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    mul_492: "f32[8, 1280, 6, 6]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_240);  sub_59 = unsqueeze_240 = None
    sub_61: "f32[8, 1280, 6, 6]" = torch.ops.aten.sub.Tensor(mul_485, mul_492);  mul_485 = mul_492 = None
    sub_62: "f32[8, 1280, 6, 6]" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_237);  sub_61 = unsqueeze_237 = None
    mul_493: "f32[8, 1280, 6, 6]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_243);  sub_62 = unsqueeze_243 = None
    mul_494: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_172);  sum_3 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_493, add_296, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_493 = add_296 = primals_250 = None
    getitem_116: "f32[8, 320, 6, 6]" = convolution_backward[0]
    getitem_117: "f32[1280, 320, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_4: "f32[320]" = torch.ops.aten.sum.dim_IntList(getitem_116, [0, 2, 3])
    sub_63: "f32[8, 320, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_246);  convolution_94 = unsqueeze_246 = None
    mul_495: "f32[8, 320, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_116, sub_63)
    sum_5: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_495, [0, 2, 3]);  mul_495 = None
    mul_496: "f32[320]" = torch.ops.aten.mul.Tensor(sum_4, 0.003472222222222222)
    unsqueeze_247: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_248: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
    mul_497: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, 0.003472222222222222)
    mul_498: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_499: "f32[320]" = torch.ops.aten.mul.Tensor(mul_497, mul_498);  mul_497 = mul_498 = None
    unsqueeze_250: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
    unsqueeze_251: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
    unsqueeze_252: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
    mul_500: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_113);  primals_113 = None
    unsqueeze_253: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
    unsqueeze_254: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    mul_501: "f32[8, 320, 6, 6]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_252);  sub_63 = unsqueeze_252 = None
    sub_65: "f32[8, 320, 6, 6]" = torch.ops.aten.sub.Tensor(getitem_116, mul_501);  getitem_116 = mul_501 = None
    sub_66: "f32[8, 320, 6, 6]" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_249);  sub_65 = unsqueeze_249 = None
    mul_502: "f32[8, 320, 6, 6]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_255);  sub_66 = unsqueeze_255 = None
    mul_503: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_169);  sum_5 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_502, mul_467, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_502 = mul_467 = primals_249 = None
    getitem_119: "f32[8, 1152, 6, 6]" = convolution_backward_1[0]
    getitem_120: "f32[320, 1152, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_504: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_119, mul_465);  mul_465 = None
    mul_505: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_119, sigmoid_75);  getitem_119 = sigmoid_75 = None
    sum_6: "f32[8, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_504, [2, 3], True);  mul_504 = None
    alias_19: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    sub_67: "f32[8, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_19)
    mul_506: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_19, sub_67);  alias_19 = sub_67 = None
    mul_507: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, mul_506);  sum_6 = mul_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_7: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_507, mul_466, primals_247, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_507 = mul_466 = primals_247 = None
    getitem_122: "f32[8, 48, 1, 1]" = convolution_backward_2[0]
    getitem_123: "f32[1152, 48, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_78: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(clone_56)
    full_default_1: "f32[8, 48, 1, 1]" = torch.ops.aten.full.default([8, 48, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_68: "f32[8, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_78)
    mul_508: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(clone_56, sub_68);  clone_56 = sub_68 = None
    add_303: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_508, 1);  mul_508 = None
    mul_509: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_78, add_303);  sigmoid_78 = add_303 = None
    mul_510: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_122, mul_509);  getitem_122 = mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_8: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_510, mean_18, primals_245, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_510 = mean_18 = primals_245 = None
    getitem_125: "f32[8, 1152, 1, 1]" = convolution_backward_3[0]
    getitem_126: "f32[48, 1152, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 1152, 6, 6]" = torch.ops.aten.expand.default(getitem_125, [8, 1152, 6, 6]);  getitem_125 = None
    div_1: "f32[8, 1152, 6, 6]" = torch.ops.aten.div.Scalar(expand_1, 36);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_304: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_505, div_1);  mul_505 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_79: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(clone_55)
    full_default_2: "f32[8, 1152, 6, 6]" = torch.ops.aten.full.default([8, 1152, 6, 6], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_69: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_79)
    mul_511: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(clone_55, sub_69);  clone_55 = sub_69 = None
    add_305: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Scalar(mul_511, 1);  mul_511 = None
    mul_512: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sigmoid_79, add_305);  sigmoid_79 = add_305 = None
    mul_513: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_304, mul_512);  add_304 = mul_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_9: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_513, [0, 2, 3])
    sub_70: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_258);  convolution_91 = unsqueeze_258 = None
    mul_514: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_513, sub_70)
    sum_10: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_514, [0, 2, 3]);  mul_514 = None
    mul_515: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_9, 0.003472222222222222)
    unsqueeze_259: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_515, 0);  mul_515 = None
    unsqueeze_260: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    unsqueeze_261: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
    mul_516: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_10, 0.003472222222222222)
    mul_517: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_518: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_516, mul_517);  mul_516 = mul_517 = None
    unsqueeze_262: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_518, 0);  mul_518 = None
    unsqueeze_263: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_519: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_111);  primals_111 = None
    unsqueeze_265: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_519, 0);  mul_519 = None
    unsqueeze_266: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    mul_520: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_264);  sub_70 = unsqueeze_264 = None
    sub_72: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(mul_513, mul_520);  mul_513 = mul_520 = None
    sub_73: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(sub_72, unsqueeze_261);  sub_72 = unsqueeze_261 = None
    mul_521: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_267);  sub_73 = unsqueeze_267 = None
    mul_522: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_166);  sum_10 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_521, mul_457, primals_244, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_521 = mul_457 = primals_244 = None
    getitem_128: "f32[8, 1152, 6, 6]" = convolution_backward_4[0]
    getitem_129: "f32[1152, 1, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_525: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_128, mul_524);  getitem_128 = mul_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_11: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_525, [0, 2, 3])
    sub_75: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_270);  convolution_90 = unsqueeze_270 = None
    mul_526: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_525, sub_75)
    sum_12: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_526, [0, 2, 3]);  mul_526 = None
    mul_527: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_11, 0.003472222222222222)
    unsqueeze_271: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_527, 0);  mul_527 = None
    unsqueeze_272: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_528: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_12, 0.003472222222222222)
    mul_529: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_530: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_528, mul_529);  mul_528 = mul_529 = None
    unsqueeze_274: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_530, 0);  mul_530 = None
    unsqueeze_275: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_531: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_109);  primals_109 = None
    unsqueeze_277: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
    unsqueeze_278: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    mul_532: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_276);  sub_75 = unsqueeze_276 = None
    sub_77: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(mul_525, mul_532);  mul_525 = mul_532 = None
    sub_78: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_273);  sub_77 = unsqueeze_273 = None
    mul_533: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_279);  sub_78 = unsqueeze_279 = None
    mul_534: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_163);  sum_12 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_533, add_281, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_533 = add_281 = primals_243 = None
    getitem_131: "f32[8, 192, 6, 6]" = convolution_backward_5[0]
    getitem_132: "f32[1152, 192, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_13: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_131, [0, 2, 3])
    sub_79: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_282);  convolution_89 = unsqueeze_282 = None
    mul_535: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_131, sub_79)
    sum_14: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
    mul_536: "f32[192]" = torch.ops.aten.mul.Tensor(sum_13, 0.003472222222222222)
    unsqueeze_283: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
    unsqueeze_284: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
    mul_537: "f32[192]" = torch.ops.aten.mul.Tensor(sum_14, 0.003472222222222222)
    mul_538: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_539: "f32[192]" = torch.ops.aten.mul.Tensor(mul_537, mul_538);  mul_537 = mul_538 = None
    unsqueeze_286: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
    unsqueeze_287: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_540: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_107);  primals_107 = None
    unsqueeze_289: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_290: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    mul_541: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_288);  sub_79 = unsqueeze_288 = None
    sub_81: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(getitem_131, mul_541);  mul_541 = None
    sub_82: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_285);  sub_81 = unsqueeze_285 = None
    mul_542: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_291);  sub_82 = unsqueeze_291 = None
    mul_543: "f32[192]" = torch.ops.aten.mul.Tensor(sum_14, squeeze_160);  sum_14 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_542, mul_442, primals_242, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_542 = mul_442 = primals_242 = None
    getitem_134: "f32[8, 1152, 6, 6]" = convolution_backward_6[0]
    getitem_135: "f32[192, 1152, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_544: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_134, mul_440);  mul_440 = None
    mul_545: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_134, sigmoid_71);  getitem_134 = sigmoid_71 = None
    sum_15: "f32[8, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_544, [2, 3], True);  mul_544 = None
    alias_20: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    sub_83: "f32[8, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_20)
    mul_546: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_20, sub_83);  alias_20 = sub_83 = None
    mul_547: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_15, mul_546);  sum_15 = mul_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_16: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_547, mul_441, primals_240, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_547 = mul_441 = primals_240 = None
    getitem_137: "f32[8, 48, 1, 1]" = convolution_backward_7[0]
    getitem_138: "f32[1152, 48, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_81: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(clone_53)
    sub_84: "f32[8, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_81)
    mul_548: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(clone_53, sub_84);  clone_53 = sub_84 = None
    add_307: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_548, 1);  mul_548 = None
    mul_549: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_81, add_307);  sigmoid_81 = add_307 = None
    mul_550: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_137, mul_549);  getitem_137 = mul_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_17: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_550, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_550, mean_17, primals_238, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_550 = mean_17 = primals_238 = None
    getitem_140: "f32[8, 1152, 1, 1]" = convolution_backward_8[0]
    getitem_141: "f32[48, 1152, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 1152, 6, 6]" = torch.ops.aten.expand.default(getitem_140, [8, 1152, 6, 6]);  getitem_140 = None
    div_2: "f32[8, 1152, 6, 6]" = torch.ops.aten.div.Scalar(expand_2, 36);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_308: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_545, div_2);  mul_545 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_82: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(clone_52)
    sub_85: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_82)
    mul_551: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(clone_52, sub_85);  clone_52 = sub_85 = None
    add_309: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Scalar(mul_551, 1);  mul_551 = None
    mul_552: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sigmoid_82, add_309);  sigmoid_82 = add_309 = None
    mul_553: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_308, mul_552);  add_308 = mul_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_553, [0, 2, 3])
    sub_86: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_294);  convolution_86 = unsqueeze_294 = None
    mul_554: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_553, sub_86)
    sum_19: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_554, [0, 2, 3]);  mul_554 = None
    mul_555: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_18, 0.003472222222222222)
    unsqueeze_295: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
    unsqueeze_296: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
    mul_556: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, 0.003472222222222222)
    mul_557: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_558: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_556, mul_557);  mul_556 = mul_557 = None
    unsqueeze_298: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
    unsqueeze_299: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_559: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_105);  primals_105 = None
    unsqueeze_301: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_559, 0);  mul_559 = None
    unsqueeze_302: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    mul_560: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_300);  sub_86 = unsqueeze_300 = None
    sub_88: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(mul_553, mul_560);  mul_553 = mul_560 = None
    sub_89: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(sub_88, unsqueeze_297);  sub_88 = unsqueeze_297 = None
    mul_561: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_303);  sub_89 = unsqueeze_303 = None
    mul_562: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_157);  sum_19 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_561, mul_432, primals_237, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_561 = mul_432 = primals_237 = None
    getitem_143: "f32[8, 1152, 6, 6]" = convolution_backward_9[0]
    getitem_144: "f32[1152, 1, 5, 5]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_565: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_143, mul_564);  getitem_143 = mul_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_20: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_565, [0, 2, 3])
    sub_91: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_306);  convolution_85 = unsqueeze_306 = None
    mul_566: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_565, sub_91)
    sum_21: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_566, [0, 2, 3]);  mul_566 = None
    mul_567: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_20, 0.003472222222222222)
    unsqueeze_307: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_308: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_568: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, 0.003472222222222222)
    mul_569: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_570: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
    unsqueeze_310: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_570, 0);  mul_570 = None
    unsqueeze_311: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_571: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_313: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_571, 0);  mul_571 = None
    unsqueeze_314: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    mul_572: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_312);  sub_91 = unsqueeze_312 = None
    sub_93: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(mul_565, mul_572);  mul_565 = mul_572 = None
    sub_94: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_309);  sub_93 = unsqueeze_309 = None
    mul_573: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_315);  sub_94 = unsqueeze_315 = None
    mul_574: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_154);  sum_21 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_573, add_265, primals_236, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_573 = add_265 = primals_236 = None
    getitem_146: "f32[8, 192, 6, 6]" = convolution_backward_10[0]
    getitem_147: "f32[1152, 192, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_311: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(getitem_131, getitem_146);  getitem_131 = getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_22: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_311, [0, 2, 3])
    sub_95: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_318);  convolution_84 = unsqueeze_318 = None
    mul_575: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(add_311, sub_95)
    sum_23: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 2, 3]);  mul_575 = None
    mul_576: "f32[192]" = torch.ops.aten.mul.Tensor(sum_22, 0.003472222222222222)
    unsqueeze_319: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_320: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
    mul_577: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, 0.003472222222222222)
    mul_578: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_579: "f32[192]" = torch.ops.aten.mul.Tensor(mul_577, mul_578);  mul_577 = mul_578 = None
    unsqueeze_322: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_323: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_580: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_325: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
    unsqueeze_326: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    mul_581: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_324);  sub_95 = unsqueeze_324 = None
    sub_97: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(add_311, mul_581);  mul_581 = None
    sub_98: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_321);  sub_97 = unsqueeze_321 = None
    mul_582: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_327);  sub_98 = unsqueeze_327 = None
    mul_583: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_151);  sum_23 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_582, mul_417, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_582 = mul_417 = primals_235 = None
    getitem_149: "f32[8, 1152, 6, 6]" = convolution_backward_11[0]
    getitem_150: "f32[192, 1152, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_584: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_149, mul_415);  mul_415 = None
    mul_585: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_149, sigmoid_67);  getitem_149 = sigmoid_67 = None
    sum_24: "f32[8, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_584, [2, 3], True);  mul_584 = None
    alias_21: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    sub_99: "f32[8, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_21)
    mul_586: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_21, sub_99);  alias_21 = sub_99 = None
    mul_587: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_24, mul_586);  sum_24 = mul_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_25: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_587, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_587, mul_416, primals_233, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_587 = mul_416 = primals_233 = None
    getitem_152: "f32[8, 48, 1, 1]" = convolution_backward_12[0]
    getitem_153: "f32[1152, 48, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_84: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(clone_50)
    sub_100: "f32[8, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_84)
    mul_588: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(clone_50, sub_100);  clone_50 = sub_100 = None
    add_312: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_588, 1);  mul_588 = None
    mul_589: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_84, add_312);  sigmoid_84 = add_312 = None
    mul_590: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_152, mul_589);  getitem_152 = mul_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_26: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_590, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_590, mean_16, primals_231, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_590 = mean_16 = primals_231 = None
    getitem_155: "f32[8, 1152, 1, 1]" = convolution_backward_13[0]
    getitem_156: "f32[48, 1152, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 1152, 6, 6]" = torch.ops.aten.expand.default(getitem_155, [8, 1152, 6, 6]);  getitem_155 = None
    div_3: "f32[8, 1152, 6, 6]" = torch.ops.aten.div.Scalar(expand_3, 36);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_313: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_585, div_3);  mul_585 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_85: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(clone_49)
    sub_101: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_85)
    mul_591: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(clone_49, sub_101);  clone_49 = sub_101 = None
    add_314: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Scalar(mul_591, 1);  mul_591 = None
    mul_592: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sigmoid_85, add_314);  sigmoid_85 = add_314 = None
    mul_593: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_313, mul_592);  add_313 = mul_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_27: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_593, [0, 2, 3])
    sub_102: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_330);  convolution_81 = unsqueeze_330 = None
    mul_594: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_593, sub_102)
    sum_28: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_594, [0, 2, 3]);  mul_594 = None
    mul_595: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_27, 0.003472222222222222)
    unsqueeze_331: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
    unsqueeze_332: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
    mul_596: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_28, 0.003472222222222222)
    mul_597: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_598: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_596, mul_597);  mul_596 = mul_597 = None
    unsqueeze_334: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
    unsqueeze_335: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_599: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_337: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_599, 0);  mul_599 = None
    unsqueeze_338: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    mul_600: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_336);  sub_102 = unsqueeze_336 = None
    sub_104: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(mul_593, mul_600);  mul_593 = mul_600 = None
    sub_105: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_333);  sub_104 = unsqueeze_333 = None
    mul_601: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_339);  sub_105 = unsqueeze_339 = None
    mul_602: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_148);  sum_28 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_601, mul_407, primals_230, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_601 = mul_407 = primals_230 = None
    getitem_158: "f32[8, 1152, 6, 6]" = convolution_backward_14[0]
    getitem_159: "f32[1152, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_605: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_158, mul_604);  getitem_158 = mul_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_29: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_605, [0, 2, 3])
    sub_107: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_342);  convolution_80 = unsqueeze_342 = None
    mul_606: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_605, sub_107)
    sum_30: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_606, [0, 2, 3]);  mul_606 = None
    mul_607: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_29, 0.003472222222222222)
    unsqueeze_343: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
    unsqueeze_344: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_608: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_30, 0.003472222222222222)
    mul_609: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_610: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_608, mul_609);  mul_608 = mul_609 = None
    unsqueeze_346: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
    unsqueeze_347: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_611: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_349: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
    unsqueeze_350: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    mul_612: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_348);  sub_107 = unsqueeze_348 = None
    sub_109: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(mul_605, mul_612);  mul_605 = mul_612 = None
    sub_110: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_345);  sub_109 = unsqueeze_345 = None
    mul_613: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_351);  sub_110 = unsqueeze_351 = None
    mul_614: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_145);  sum_30 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_613, add_249, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_613 = add_249 = primals_229 = None
    getitem_161: "f32[8, 192, 6, 6]" = convolution_backward_15[0]
    getitem_162: "f32[1152, 192, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_316: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(add_311, getitem_161);  add_311 = getitem_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_31: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_316, [0, 2, 3])
    sub_111: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_354);  convolution_79 = unsqueeze_354 = None
    mul_615: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(add_316, sub_111)
    sum_32: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_615, [0, 2, 3]);  mul_615 = None
    mul_616: "f32[192]" = torch.ops.aten.mul.Tensor(sum_31, 0.003472222222222222)
    unsqueeze_355: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
    unsqueeze_356: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_617: "f32[192]" = torch.ops.aten.mul.Tensor(sum_32, 0.003472222222222222)
    mul_618: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_619: "f32[192]" = torch.ops.aten.mul.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
    unsqueeze_358: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_619, 0);  mul_619 = None
    unsqueeze_359: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_620: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_361: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
    unsqueeze_362: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    mul_621: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_360);  sub_111 = unsqueeze_360 = None
    sub_113: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(add_316, mul_621);  mul_621 = None
    sub_114: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_357);  sub_113 = unsqueeze_357 = None
    mul_622: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_363);  sub_114 = unsqueeze_363 = None
    mul_623: "f32[192]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_142);  sum_32 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_622, mul_392, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_622 = mul_392 = primals_228 = None
    getitem_164: "f32[8, 1152, 6, 6]" = convolution_backward_16[0]
    getitem_165: "f32[192, 1152, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_624: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_164, mul_390);  mul_390 = None
    mul_625: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_164, sigmoid_63);  getitem_164 = sigmoid_63 = None
    sum_33: "f32[8, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_624, [2, 3], True);  mul_624 = None
    alias_22: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    sub_115: "f32[8, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_22)
    mul_626: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_22, sub_115);  alias_22 = sub_115 = None
    mul_627: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_33, mul_626);  sum_33 = mul_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_34: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_627, mul_391, primals_226, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_627 = mul_391 = primals_226 = None
    getitem_167: "f32[8, 48, 1, 1]" = convolution_backward_17[0]
    getitem_168: "f32[1152, 48, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_87: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(clone_47)
    sub_116: "f32[8, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_87)
    mul_628: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(clone_47, sub_116);  clone_47 = sub_116 = None
    add_317: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_628, 1);  mul_628 = None
    mul_629: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_87, add_317);  sigmoid_87 = add_317 = None
    mul_630: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_167, mul_629);  getitem_167 = mul_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_35: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_630, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_630, mean_15, primals_224, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_630 = mean_15 = primals_224 = None
    getitem_170: "f32[8, 1152, 1, 1]" = convolution_backward_18[0]
    getitem_171: "f32[48, 1152, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 1152, 6, 6]" = torch.ops.aten.expand.default(getitem_170, [8, 1152, 6, 6]);  getitem_170 = None
    div_4: "f32[8, 1152, 6, 6]" = torch.ops.aten.div.Scalar(expand_4, 36);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_318: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_625, div_4);  mul_625 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_88: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(clone_46)
    sub_117: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_88)
    mul_631: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(clone_46, sub_117);  clone_46 = sub_117 = None
    add_319: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Scalar(mul_631, 1);  mul_631 = None
    mul_632: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sigmoid_88, add_319);  sigmoid_88 = add_319 = None
    mul_633: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_318, mul_632);  add_318 = mul_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_36: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_633, [0, 2, 3])
    sub_118: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_366);  convolution_76 = unsqueeze_366 = None
    mul_634: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_633, sub_118)
    sum_37: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_634, [0, 2, 3]);  mul_634 = None
    mul_635: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_36, 0.003472222222222222)
    unsqueeze_367: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    unsqueeze_368: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_636: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_37, 0.003472222222222222)
    mul_637: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_638: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_636, mul_637);  mul_636 = mul_637 = None
    unsqueeze_370: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    unsqueeze_371: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_639: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_373: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    unsqueeze_374: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    mul_640: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_372);  sub_118 = unsqueeze_372 = None
    sub_120: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(mul_633, mul_640);  mul_633 = mul_640 = None
    sub_121: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_369);  sub_120 = unsqueeze_369 = None
    mul_641: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_375);  sub_121 = unsqueeze_375 = None
    mul_642: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_139);  sum_37 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_641, mul_382, primals_223, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_641 = mul_382 = primals_223 = None
    getitem_173: "f32[8, 1152, 6, 6]" = convolution_backward_19[0]
    getitem_174: "f32[1152, 1, 5, 5]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_645: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_173, mul_644);  getitem_173 = mul_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_38: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_645, [0, 2, 3])
    sub_123: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_378);  convolution_75 = unsqueeze_378 = None
    mul_646: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_645, sub_123)
    sum_39: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_646, [0, 2, 3]);  mul_646 = None
    mul_647: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_38, 0.003472222222222222)
    unsqueeze_379: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_647, 0);  mul_647 = None
    unsqueeze_380: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_648: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_39, 0.003472222222222222)
    mul_649: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_650: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
    unsqueeze_382: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_650, 0);  mul_650 = None
    unsqueeze_383: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_651: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_385: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
    unsqueeze_386: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    mul_652: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_384);  sub_123 = unsqueeze_384 = None
    sub_125: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(mul_645, mul_652);  mul_645 = mul_652 = None
    sub_126: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_381);  sub_125 = unsqueeze_381 = None
    mul_653: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_387);  sub_126 = unsqueeze_387 = None
    mul_654: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_136);  sum_39 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_653, add_233, primals_222, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_653 = add_233 = primals_222 = None
    getitem_176: "f32[8, 192, 6, 6]" = convolution_backward_20[0]
    getitem_177: "f32[1152, 192, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_321: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(add_316, getitem_176);  add_316 = getitem_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_40: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_321, [0, 2, 3])
    sub_127: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_390);  convolution_74 = unsqueeze_390 = None
    mul_655: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(add_321, sub_127)
    sum_41: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_655, [0, 2, 3]);  mul_655 = None
    mul_656: "f32[192]" = torch.ops.aten.mul.Tensor(sum_40, 0.003472222222222222)
    unsqueeze_391: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
    unsqueeze_392: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_657: "f32[192]" = torch.ops.aten.mul.Tensor(sum_41, 0.003472222222222222)
    mul_658: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_659: "f32[192]" = torch.ops.aten.mul.Tensor(mul_657, mul_658);  mul_657 = mul_658 = None
    unsqueeze_394: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_659, 0);  mul_659 = None
    unsqueeze_395: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_660: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_397: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_660, 0);  mul_660 = None
    unsqueeze_398: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    mul_661: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_396);  sub_127 = unsqueeze_396 = None
    sub_129: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(add_321, mul_661);  mul_661 = None
    sub_130: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_393);  sub_129 = unsqueeze_393 = None
    mul_662: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_399);  sub_130 = unsqueeze_399 = None
    mul_663: "f32[192]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_133);  sum_41 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_662, mul_367, primals_221, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_662 = mul_367 = primals_221 = None
    getitem_179: "f32[8, 1152, 6, 6]" = convolution_backward_21[0]
    getitem_180: "f32[192, 1152, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_664: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_179, mul_365);  mul_365 = None
    mul_665: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_179, sigmoid_59);  getitem_179 = sigmoid_59 = None
    sum_42: "f32[8, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_664, [2, 3], True);  mul_664 = None
    alias_23: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    sub_131: "f32[8, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_23)
    mul_666: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_23, sub_131);  alias_23 = sub_131 = None
    mul_667: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_42, mul_666);  sum_42 = mul_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_43: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_667, mul_366, primals_219, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_667 = mul_366 = primals_219 = None
    getitem_182: "f32[8, 48, 1, 1]" = convolution_backward_22[0]
    getitem_183: "f32[1152, 48, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_90: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(clone_44)
    sub_132: "f32[8, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_90);  full_default_1 = None
    mul_668: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(clone_44, sub_132);  clone_44 = sub_132 = None
    add_322: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_668, 1);  mul_668 = None
    mul_669: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_90, add_322);  sigmoid_90 = add_322 = None
    mul_670: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_182, mul_669);  getitem_182 = mul_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_44: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_670, mean_14, primals_217, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_670 = mean_14 = primals_217 = None
    getitem_185: "f32[8, 1152, 1, 1]" = convolution_backward_23[0]
    getitem_186: "f32[48, 1152, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[8, 1152, 6, 6]" = torch.ops.aten.expand.default(getitem_185, [8, 1152, 6, 6]);  getitem_185 = None
    div_5: "f32[8, 1152, 6, 6]" = torch.ops.aten.div.Scalar(expand_5, 36);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_323: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_665, div_5);  mul_665 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_91: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(clone_43)
    sub_133: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_91);  full_default_2 = None
    mul_671: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(clone_43, sub_133);  clone_43 = sub_133 = None
    add_324: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Scalar(mul_671, 1);  mul_671 = None
    mul_672: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sigmoid_91, add_324);  sigmoid_91 = add_324 = None
    mul_673: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_323, mul_672);  add_323 = mul_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_45: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_673, [0, 2, 3])
    sub_134: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_402);  convolution_71 = unsqueeze_402 = None
    mul_674: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_673, sub_134)
    sum_46: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_674, [0, 2, 3]);  mul_674 = None
    mul_675: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_45, 0.003472222222222222)
    unsqueeze_403: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
    unsqueeze_404: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_676: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_46, 0.003472222222222222)
    mul_677: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_678: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_676, mul_677);  mul_676 = mul_677 = None
    unsqueeze_406: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_678, 0);  mul_678 = None
    unsqueeze_407: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_679: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_409: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_679, 0);  mul_679 = None
    unsqueeze_410: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    mul_680: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_408);  sub_134 = unsqueeze_408 = None
    sub_136: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(mul_673, mul_680);  mul_673 = mul_680 = None
    sub_137: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(sub_136, unsqueeze_405);  sub_136 = unsqueeze_405 = None
    mul_681: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_411);  sub_137 = unsqueeze_411 = None
    mul_682: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_130);  sum_46 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_681, mul_357, primals_216, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_681 = mul_357 = primals_216 = None
    getitem_188: "f32[8, 1152, 6, 6]" = convolution_backward_24[0]
    getitem_189: "f32[1152, 1, 5, 5]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_685: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_188, mul_684);  getitem_188 = mul_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_47: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_685, [0, 2, 3])
    sub_139: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_414);  convolution_70 = unsqueeze_414 = None
    mul_686: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_685, sub_139)
    sum_48: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_686, [0, 2, 3]);  mul_686 = None
    mul_687: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_47, 0.003472222222222222)
    unsqueeze_415: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_687, 0);  mul_687 = None
    unsqueeze_416: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_688: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_48, 0.003472222222222222)
    mul_689: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_690: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_688, mul_689);  mul_688 = mul_689 = None
    unsqueeze_418: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
    unsqueeze_419: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_691: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_421: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_691, 0);  mul_691 = None
    unsqueeze_422: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    mul_692: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_420);  sub_139 = unsqueeze_420 = None
    sub_141: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(mul_685, mul_692);  mul_685 = mul_692 = None
    sub_142: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(sub_141, unsqueeze_417);  sub_141 = unsqueeze_417 = None
    mul_693: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_423);  sub_142 = unsqueeze_423 = None
    mul_694: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_127);  sum_48 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_693, add_217, primals_215, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_693 = add_217 = primals_215 = None
    getitem_191: "f32[8, 192, 6, 6]" = convolution_backward_25[0]
    getitem_192: "f32[1152, 192, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_326: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(add_321, getitem_191);  add_321 = getitem_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_49: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_326, [0, 2, 3])
    sub_143: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_426);  convolution_69 = unsqueeze_426 = None
    mul_695: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(add_326, sub_143)
    sum_50: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_695, [0, 2, 3]);  mul_695 = None
    mul_696: "f32[192]" = torch.ops.aten.mul.Tensor(sum_49, 0.003472222222222222)
    unsqueeze_427: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
    unsqueeze_428: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_697: "f32[192]" = torch.ops.aten.mul.Tensor(sum_50, 0.003472222222222222)
    mul_698: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_699: "f32[192]" = torch.ops.aten.mul.Tensor(mul_697, mul_698);  mul_697 = mul_698 = None
    unsqueeze_430: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
    unsqueeze_431: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_700: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_433: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_700, 0);  mul_700 = None
    unsqueeze_434: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    mul_701: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_432);  sub_143 = unsqueeze_432 = None
    sub_145: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(add_326, mul_701);  add_326 = mul_701 = None
    sub_146: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(sub_145, unsqueeze_429);  sub_145 = unsqueeze_429 = None
    mul_702: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_435);  sub_146 = unsqueeze_435 = None
    mul_703: "f32[192]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_124);  sum_50 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_702, mul_342, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_702 = mul_342 = primals_214 = None
    getitem_194: "f32[8, 672, 6, 6]" = convolution_backward_26[0]
    getitem_195: "f32[192, 672, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_704: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_194, mul_340);  mul_340 = None
    mul_705: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_194, sigmoid_55);  getitem_194 = sigmoid_55 = None
    sum_51: "f32[8, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_704, [2, 3], True);  mul_704 = None
    alias_24: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    sub_147: "f32[8, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_24)
    mul_706: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(alias_24, sub_147);  alias_24 = sub_147 = None
    mul_707: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_51, mul_706);  sum_51 = mul_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_52: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_707, mul_341, primals_212, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_707 = mul_341 = primals_212 = None
    getitem_197: "f32[8, 28, 1, 1]" = convolution_backward_27[0]
    getitem_198: "f32[672, 28, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_93: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_41)
    full_default_16: "f32[8, 28, 1, 1]" = torch.ops.aten.full.default([8, 28, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_148: "f32[8, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_16, sigmoid_93)
    mul_708: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_41, sub_148);  clone_41 = sub_148 = None
    add_327: "f32[8, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_708, 1);  mul_708 = None
    mul_709: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_93, add_327);  sigmoid_93 = add_327 = None
    mul_710: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_197, mul_709);  getitem_197 = mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_53: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_710, [0, 2, 3])
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_710, mean_13, primals_210, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_710 = mean_13 = primals_210 = None
    getitem_200: "f32[8, 672, 1, 1]" = convolution_backward_28[0]
    getitem_201: "f32[28, 672, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[8, 672, 6, 6]" = torch.ops.aten.expand.default(getitem_200, [8, 672, 6, 6]);  getitem_200 = None
    div_6: "f32[8, 672, 6, 6]" = torch.ops.aten.div.Scalar(expand_6, 36);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_328: "f32[8, 672, 6, 6]" = torch.ops.aten.add.Tensor(mul_705, div_6);  mul_705 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_94: "f32[8, 672, 6, 6]" = torch.ops.aten.sigmoid.default(clone_40)
    full_default_17: "f32[8, 672, 6, 6]" = torch.ops.aten.full.default([8, 672, 6, 6], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_149: "f32[8, 672, 6, 6]" = torch.ops.aten.sub.Tensor(full_default_17, sigmoid_94);  full_default_17 = None
    mul_711: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(clone_40, sub_149);  clone_40 = sub_149 = None
    add_329: "f32[8, 672, 6, 6]" = torch.ops.aten.add.Scalar(mul_711, 1);  mul_711 = None
    mul_712: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(sigmoid_94, add_329);  sigmoid_94 = add_329 = None
    mul_713: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(add_328, mul_712);  add_328 = mul_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_54: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_713, [0, 2, 3])
    sub_150: "f32[8, 672, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_438);  convolution_66 = unsqueeze_438 = None
    mul_714: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(mul_713, sub_150)
    sum_55: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_714, [0, 2, 3]);  mul_714 = None
    mul_715: "f32[672]" = torch.ops.aten.mul.Tensor(sum_54, 0.003472222222222222)
    unsqueeze_439: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_715, 0);  mul_715 = None
    unsqueeze_440: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_716: "f32[672]" = torch.ops.aten.mul.Tensor(sum_55, 0.003472222222222222)
    mul_717: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_718: "f32[672]" = torch.ops.aten.mul.Tensor(mul_716, mul_717);  mul_716 = mul_717 = None
    unsqueeze_442: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
    unsqueeze_443: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_719: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_445: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
    unsqueeze_446: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    mul_720: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_444);  sub_150 = unsqueeze_444 = None
    sub_152: "f32[8, 672, 6, 6]" = torch.ops.aten.sub.Tensor(mul_713, mul_720);  mul_713 = mul_720 = None
    sub_153: "f32[8, 672, 6, 6]" = torch.ops.aten.sub.Tensor(sub_152, unsqueeze_441);  sub_152 = unsqueeze_441 = None
    mul_721: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_447);  sub_153 = unsqueeze_447 = None
    mul_722: "f32[672]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_121);  sum_55 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_721, mul_332, primals_209, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_721 = mul_332 = primals_209 = None
    getitem_203: "f32[8, 672, 12, 12]" = convolution_backward_29[0]
    getitem_204: "f32[672, 1, 5, 5]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_18: "f32[8, 672, 12, 12]" = torch.ops.aten.full.default([8, 672, 12, 12], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    mul_725: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_203, mul_724);  getitem_203 = mul_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_725, [0, 2, 3])
    sub_155: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_450);  convolution_65 = unsqueeze_450 = None
    mul_726: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_725, sub_155)
    sum_57: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_726, [0, 2, 3]);  mul_726 = None
    mul_727: "f32[672]" = torch.ops.aten.mul.Tensor(sum_56, 0.0008680555555555555)
    unsqueeze_451: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
    unsqueeze_452: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_728: "f32[672]" = torch.ops.aten.mul.Tensor(sum_57, 0.0008680555555555555)
    mul_729: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_730: "f32[672]" = torch.ops.aten.mul.Tensor(mul_728, mul_729);  mul_728 = mul_729 = None
    unsqueeze_454: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_455: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_731: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_457: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
    unsqueeze_458: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    mul_732: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_456);  sub_155 = unsqueeze_456 = None
    sub_157: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(mul_725, mul_732);  mul_725 = mul_732 = None
    sub_158: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(sub_157, unsqueeze_453);  sub_157 = unsqueeze_453 = None
    mul_733: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_459);  sub_158 = unsqueeze_459 = None
    mul_734: "f32[672]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_118);  sum_57 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_733, add_202, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_733 = add_202 = primals_208 = None
    getitem_206: "f32[8, 112, 12, 12]" = convolution_backward_30[0]
    getitem_207: "f32[672, 112, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_58: "f32[112]" = torch.ops.aten.sum.dim_IntList(getitem_206, [0, 2, 3])
    sub_159: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_462);  convolution_64 = unsqueeze_462 = None
    mul_735: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_206, sub_159)
    sum_59: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_735, [0, 2, 3]);  mul_735 = None
    mul_736: "f32[112]" = torch.ops.aten.mul.Tensor(sum_58, 0.0008680555555555555)
    unsqueeze_463: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
    unsqueeze_464: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_737: "f32[112]" = torch.ops.aten.mul.Tensor(sum_59, 0.0008680555555555555)
    mul_738: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_739: "f32[112]" = torch.ops.aten.mul.Tensor(mul_737, mul_738);  mul_737 = mul_738 = None
    unsqueeze_466: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    unsqueeze_467: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_740: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_469: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    unsqueeze_470: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    mul_741: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_468);  sub_159 = unsqueeze_468 = None
    sub_161: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(getitem_206, mul_741);  mul_741 = None
    sub_162: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_465);  sub_161 = unsqueeze_465 = None
    mul_742: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_471);  sub_162 = unsqueeze_471 = None
    mul_743: "f32[112]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_115);  sum_59 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_742, mul_317, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_742 = mul_317 = primals_207 = None
    getitem_209: "f32[8, 672, 12, 12]" = convolution_backward_31[0]
    getitem_210: "f32[112, 672, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_744: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_209, mul_315);  mul_315 = None
    mul_745: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_209, sigmoid_51);  getitem_209 = sigmoid_51 = None
    sum_60: "f32[8, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_744, [2, 3], True);  mul_744 = None
    alias_25: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    sub_163: "f32[8, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_25)
    mul_746: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(alias_25, sub_163);  alias_25 = sub_163 = None
    mul_747: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_60, mul_746);  sum_60 = mul_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_61: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_747, [0, 2, 3])
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_747, mul_316, primals_205, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_747 = mul_316 = primals_205 = None
    getitem_212: "f32[8, 28, 1, 1]" = convolution_backward_32[0]
    getitem_213: "f32[672, 28, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_96: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_38)
    sub_164: "f32[8, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_16, sigmoid_96)
    mul_748: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_38, sub_164);  clone_38 = sub_164 = None
    add_331: "f32[8, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_748, 1);  mul_748 = None
    mul_749: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_96, add_331);  sigmoid_96 = add_331 = None
    mul_750: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_212, mul_749);  getitem_212 = mul_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_62: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_750, [0, 2, 3])
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_750, mean_12, primals_203, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_750 = mean_12 = primals_203 = None
    getitem_215: "f32[8, 672, 1, 1]" = convolution_backward_33[0]
    getitem_216: "f32[28, 672, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[8, 672, 12, 12]" = torch.ops.aten.expand.default(getitem_215, [8, 672, 12, 12]);  getitem_215 = None
    div_7: "f32[8, 672, 12, 12]" = torch.ops.aten.div.Scalar(expand_7, 144);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_332: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_745, div_7);  mul_745 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_97: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(clone_37)
    sub_165: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_18, sigmoid_97)
    mul_751: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(clone_37, sub_165);  clone_37 = sub_165 = None
    add_333: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Scalar(mul_751, 1);  mul_751 = None
    mul_752: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_97, add_333);  sigmoid_97 = add_333 = None
    mul_753: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_332, mul_752);  add_332 = mul_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_63: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_753, [0, 2, 3])
    sub_166: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_474);  convolution_61 = unsqueeze_474 = None
    mul_754: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_753, sub_166)
    sum_64: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_754, [0, 2, 3]);  mul_754 = None
    mul_755: "f32[672]" = torch.ops.aten.mul.Tensor(sum_63, 0.0008680555555555555)
    unsqueeze_475: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_476: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_756: "f32[672]" = torch.ops.aten.mul.Tensor(sum_64, 0.0008680555555555555)
    mul_757: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_758: "f32[672]" = torch.ops.aten.mul.Tensor(mul_756, mul_757);  mul_756 = mul_757 = None
    unsqueeze_478: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_479: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_759: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_481: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_759, 0);  mul_759 = None
    unsqueeze_482: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    mul_760: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_480);  sub_166 = unsqueeze_480 = None
    sub_168: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(mul_753, mul_760);  mul_753 = mul_760 = None
    sub_169: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_477);  sub_168 = unsqueeze_477 = None
    mul_761: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_483);  sub_169 = unsqueeze_483 = None
    mul_762: "f32[672]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_112);  sum_64 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_761, mul_307, primals_202, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_761 = mul_307 = primals_202 = None
    getitem_218: "f32[8, 672, 12, 12]" = convolution_backward_34[0]
    getitem_219: "f32[672, 1, 5, 5]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_765: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_218, mul_764);  getitem_218 = mul_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_65: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_765, [0, 2, 3])
    sub_171: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_486);  convolution_60 = unsqueeze_486 = None
    mul_766: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_765, sub_171)
    sum_66: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_766, [0, 2, 3]);  mul_766 = None
    mul_767: "f32[672]" = torch.ops.aten.mul.Tensor(sum_65, 0.0008680555555555555)
    unsqueeze_487: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
    unsqueeze_488: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_768: "f32[672]" = torch.ops.aten.mul.Tensor(sum_66, 0.0008680555555555555)
    mul_769: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_770: "f32[672]" = torch.ops.aten.mul.Tensor(mul_768, mul_769);  mul_768 = mul_769 = None
    unsqueeze_490: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
    unsqueeze_491: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_771: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_493: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
    unsqueeze_494: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    mul_772: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_492);  sub_171 = unsqueeze_492 = None
    sub_173: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(mul_765, mul_772);  mul_765 = mul_772 = None
    sub_174: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(sub_173, unsqueeze_489);  sub_173 = unsqueeze_489 = None
    mul_773: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_495);  sub_174 = unsqueeze_495 = None
    mul_774: "f32[672]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_109);  sum_66 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_773, add_186, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_773 = add_186 = primals_201 = None
    getitem_221: "f32[8, 112, 12, 12]" = convolution_backward_35[0]
    getitem_222: "f32[672, 112, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_335: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(getitem_206, getitem_221);  getitem_206 = getitem_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_67: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_335, [0, 2, 3])
    sub_175: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_498);  convolution_59 = unsqueeze_498 = None
    mul_775: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(add_335, sub_175)
    sum_68: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_775, [0, 2, 3]);  mul_775 = None
    mul_776: "f32[112]" = torch.ops.aten.mul.Tensor(sum_67, 0.0008680555555555555)
    unsqueeze_499: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_500: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    unsqueeze_501: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
    mul_777: "f32[112]" = torch.ops.aten.mul.Tensor(sum_68, 0.0008680555555555555)
    mul_778: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_779: "f32[112]" = torch.ops.aten.mul.Tensor(mul_777, mul_778);  mul_777 = mul_778 = None
    unsqueeze_502: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
    unsqueeze_503: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
    unsqueeze_504: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
    mul_780: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_505: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
    unsqueeze_506: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    mul_781: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_504);  sub_175 = unsqueeze_504 = None
    sub_177: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(add_335, mul_781);  mul_781 = None
    sub_178: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(sub_177, unsqueeze_501);  sub_177 = unsqueeze_501 = None
    mul_782: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_507);  sub_178 = unsqueeze_507 = None
    mul_783: "f32[112]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_106);  sum_68 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_782, mul_292, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_782 = mul_292 = primals_200 = None
    getitem_224: "f32[8, 672, 12, 12]" = convolution_backward_36[0]
    getitem_225: "f32[112, 672, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_784: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_224, mul_290);  mul_290 = None
    mul_785: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_224, sigmoid_47);  getitem_224 = sigmoid_47 = None
    sum_69: "f32[8, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_784, [2, 3], True);  mul_784 = None
    alias_26: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    sub_179: "f32[8, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_26)
    mul_786: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(alias_26, sub_179);  alias_26 = sub_179 = None
    mul_787: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, mul_786);  sum_69 = mul_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_70: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 2, 3])
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_787, mul_291, primals_198, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_787 = mul_291 = primals_198 = None
    getitem_227: "f32[8, 28, 1, 1]" = convolution_backward_37[0]
    getitem_228: "f32[672, 28, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_99: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_35)
    sub_180: "f32[8, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_16, sigmoid_99)
    mul_788: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_35, sub_180);  clone_35 = sub_180 = None
    add_336: "f32[8, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_788, 1);  mul_788 = None
    mul_789: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_99, add_336);  sigmoid_99 = add_336 = None
    mul_790: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_227, mul_789);  getitem_227 = mul_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_71: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_790, [0, 2, 3])
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_790, mean_11, primals_196, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_790 = mean_11 = primals_196 = None
    getitem_230: "f32[8, 672, 1, 1]" = convolution_backward_38[0]
    getitem_231: "f32[28, 672, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[8, 672, 12, 12]" = torch.ops.aten.expand.default(getitem_230, [8, 672, 12, 12]);  getitem_230 = None
    div_8: "f32[8, 672, 12, 12]" = torch.ops.aten.div.Scalar(expand_8, 144);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_337: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_785, div_8);  mul_785 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_100: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(clone_34)
    sub_181: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_18, sigmoid_100)
    mul_791: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(clone_34, sub_181);  clone_34 = sub_181 = None
    add_338: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Scalar(mul_791, 1);  mul_791 = None
    mul_792: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_100, add_338);  sigmoid_100 = add_338 = None
    mul_793: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_337, mul_792);  add_337 = mul_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_793, [0, 2, 3])
    sub_182: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_510);  convolution_56 = unsqueeze_510 = None
    mul_794: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_793, sub_182)
    sum_73: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_794, [0, 2, 3]);  mul_794 = None
    mul_795: "f32[672]" = torch.ops.aten.mul.Tensor(sum_72, 0.0008680555555555555)
    unsqueeze_511: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
    unsqueeze_512: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    unsqueeze_513: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
    mul_796: "f32[672]" = torch.ops.aten.mul.Tensor(sum_73, 0.0008680555555555555)
    mul_797: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_798: "f32[672]" = torch.ops.aten.mul.Tensor(mul_796, mul_797);  mul_796 = mul_797 = None
    unsqueeze_514: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
    unsqueeze_515: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
    unsqueeze_516: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
    mul_799: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_517: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
    unsqueeze_518: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    mul_800: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_516);  sub_182 = unsqueeze_516 = None
    sub_184: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(mul_793, mul_800);  mul_793 = mul_800 = None
    sub_185: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_513);  sub_184 = unsqueeze_513 = None
    mul_801: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_519);  sub_185 = unsqueeze_519 = None
    mul_802: "f32[672]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_103);  sum_73 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_801, mul_282, primals_195, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_801 = mul_282 = primals_195 = None
    getitem_233: "f32[8, 672, 12, 12]" = convolution_backward_39[0]
    getitem_234: "f32[672, 1, 5, 5]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_805: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_233, mul_804);  getitem_233 = mul_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_805, [0, 2, 3])
    sub_187: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_522);  convolution_55 = unsqueeze_522 = None
    mul_806: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_805, sub_187)
    sum_75: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_806, [0, 2, 3]);  mul_806 = None
    mul_807: "f32[672]" = torch.ops.aten.mul.Tensor(sum_74, 0.0008680555555555555)
    unsqueeze_523: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
    unsqueeze_524: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
    unsqueeze_525: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
    mul_808: "f32[672]" = torch.ops.aten.mul.Tensor(sum_75, 0.0008680555555555555)
    mul_809: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_810: "f32[672]" = torch.ops.aten.mul.Tensor(mul_808, mul_809);  mul_808 = mul_809 = None
    unsqueeze_526: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
    unsqueeze_527: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
    unsqueeze_528: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
    mul_811: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_529: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_811, 0);  mul_811 = None
    unsqueeze_530: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    mul_812: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_528);  sub_187 = unsqueeze_528 = None
    sub_189: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(mul_805, mul_812);  mul_805 = mul_812 = None
    sub_190: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_525);  sub_189 = unsqueeze_525 = None
    mul_813: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_531);  sub_190 = unsqueeze_531 = None
    mul_814: "f32[672]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_100);  sum_75 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_813, add_170, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_813 = add_170 = primals_194 = None
    getitem_236: "f32[8, 112, 12, 12]" = convolution_backward_40[0]
    getitem_237: "f32[672, 112, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_340: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(add_335, getitem_236);  add_335 = getitem_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_76: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_340, [0, 2, 3])
    sub_191: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_534);  convolution_54 = unsqueeze_534 = None
    mul_815: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(add_340, sub_191)
    sum_77: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_815, [0, 2, 3]);  mul_815 = None
    mul_816: "f32[112]" = torch.ops.aten.mul.Tensor(sum_76, 0.0008680555555555555)
    unsqueeze_535: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
    unsqueeze_536: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
    unsqueeze_537: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
    mul_817: "f32[112]" = torch.ops.aten.mul.Tensor(sum_77, 0.0008680555555555555)
    mul_818: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_819: "f32[112]" = torch.ops.aten.mul.Tensor(mul_817, mul_818);  mul_817 = mul_818 = None
    unsqueeze_538: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
    unsqueeze_539: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
    unsqueeze_540: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
    mul_820: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_541: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_820, 0);  mul_820 = None
    unsqueeze_542: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    mul_821: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_540);  sub_191 = unsqueeze_540 = None
    sub_193: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(add_340, mul_821);  mul_821 = None
    sub_194: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(sub_193, unsqueeze_537);  sub_193 = unsqueeze_537 = None
    mul_822: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_543);  sub_194 = unsqueeze_543 = None
    mul_823: "f32[112]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_97);  sum_77 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_822, mul_267, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_822 = mul_267 = primals_193 = None
    getitem_239: "f32[8, 672, 12, 12]" = convolution_backward_41[0]
    getitem_240: "f32[112, 672, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_824: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_239, mul_265);  mul_265 = None
    mul_825: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_239, sigmoid_43);  getitem_239 = sigmoid_43 = None
    sum_78: "f32[8, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_824, [2, 3], True);  mul_824 = None
    alias_27: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    sub_195: "f32[8, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_27)
    mul_826: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(alias_27, sub_195);  alias_27 = sub_195 = None
    mul_827: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_78, mul_826);  sum_78 = mul_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_79: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_827, [0, 2, 3])
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_827, mul_266, primals_191, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_827 = mul_266 = primals_191 = None
    getitem_242: "f32[8, 28, 1, 1]" = convolution_backward_42[0]
    getitem_243: "f32[672, 28, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_102: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_32)
    sub_196: "f32[8, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_16, sigmoid_102);  full_default_16 = None
    mul_828: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_32, sub_196);  clone_32 = sub_196 = None
    add_341: "f32[8, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_828, 1);  mul_828 = None
    mul_829: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_102, add_341);  sigmoid_102 = add_341 = None
    mul_830: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_242, mul_829);  getitem_242 = mul_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_80: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_830, [0, 2, 3])
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_830, mean_10, primals_189, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_830 = mean_10 = primals_189 = None
    getitem_245: "f32[8, 672, 1, 1]" = convolution_backward_43[0]
    getitem_246: "f32[28, 672, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[8, 672, 12, 12]" = torch.ops.aten.expand.default(getitem_245, [8, 672, 12, 12]);  getitem_245 = None
    div_9: "f32[8, 672, 12, 12]" = torch.ops.aten.div.Scalar(expand_9, 144);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_342: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_825, div_9);  mul_825 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_103: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(clone_31)
    sub_197: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_18, sigmoid_103);  full_default_18 = None
    mul_831: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(clone_31, sub_197);  clone_31 = sub_197 = None
    add_343: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Scalar(mul_831, 1);  mul_831 = None
    mul_832: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_103, add_343);  sigmoid_103 = add_343 = None
    mul_833: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_342, mul_832);  add_342 = mul_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_81: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_833, [0, 2, 3])
    sub_198: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_546);  convolution_51 = unsqueeze_546 = None
    mul_834: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_833, sub_198)
    sum_82: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_834, [0, 2, 3]);  mul_834 = None
    mul_835: "f32[672]" = torch.ops.aten.mul.Tensor(sum_81, 0.0008680555555555555)
    unsqueeze_547: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_835, 0);  mul_835 = None
    unsqueeze_548: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
    unsqueeze_549: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
    mul_836: "f32[672]" = torch.ops.aten.mul.Tensor(sum_82, 0.0008680555555555555)
    mul_837: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_838: "f32[672]" = torch.ops.aten.mul.Tensor(mul_836, mul_837);  mul_836 = mul_837 = None
    unsqueeze_550: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_838, 0);  mul_838 = None
    unsqueeze_551: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
    unsqueeze_552: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
    mul_839: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_553: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_839, 0);  mul_839 = None
    unsqueeze_554: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    mul_840: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_552);  sub_198 = unsqueeze_552 = None
    sub_200: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(mul_833, mul_840);  mul_833 = mul_840 = None
    sub_201: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_549);  sub_200 = unsqueeze_549 = None
    mul_841: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_555);  sub_201 = unsqueeze_555 = None
    mul_842: "f32[672]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_94);  sum_82 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_841, mul_257, primals_188, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_841 = mul_257 = primals_188 = None
    getitem_248: "f32[8, 672, 12, 12]" = convolution_backward_44[0]
    getitem_249: "f32[672, 1, 5, 5]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_845: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_248, mul_844);  getitem_248 = mul_844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_83: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_845, [0, 2, 3])
    sub_203: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_558);  convolution_50 = unsqueeze_558 = None
    mul_846: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_845, sub_203)
    sum_84: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_846, [0, 2, 3]);  mul_846 = None
    mul_847: "f32[672]" = torch.ops.aten.mul.Tensor(sum_83, 0.0008680555555555555)
    unsqueeze_559: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_847, 0);  mul_847 = None
    unsqueeze_560: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_848: "f32[672]" = torch.ops.aten.mul.Tensor(sum_84, 0.0008680555555555555)
    mul_849: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_850: "f32[672]" = torch.ops.aten.mul.Tensor(mul_848, mul_849);  mul_848 = mul_849 = None
    unsqueeze_562: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_850, 0);  mul_850 = None
    unsqueeze_563: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    mul_851: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_565: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_851, 0);  mul_851 = None
    unsqueeze_566: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    mul_852: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_564);  sub_203 = unsqueeze_564 = None
    sub_205: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(mul_845, mul_852);  mul_845 = mul_852 = None
    sub_206: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_561);  sub_205 = unsqueeze_561 = None
    mul_853: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_567);  sub_206 = unsqueeze_567 = None
    mul_854: "f32[672]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_91);  sum_84 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_853, add_154, primals_187, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_853 = add_154 = primals_187 = None
    getitem_251: "f32[8, 112, 12, 12]" = convolution_backward_45[0]
    getitem_252: "f32[672, 112, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_345: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(add_340, getitem_251);  add_340 = getitem_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_85: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_345, [0, 2, 3])
    sub_207: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_570);  convolution_49 = unsqueeze_570 = None
    mul_855: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(add_345, sub_207)
    sum_86: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_855, [0, 2, 3]);  mul_855 = None
    mul_856: "f32[112]" = torch.ops.aten.mul.Tensor(sum_85, 0.0008680555555555555)
    unsqueeze_571: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_856, 0);  mul_856 = None
    unsqueeze_572: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
    unsqueeze_573: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
    mul_857: "f32[112]" = torch.ops.aten.mul.Tensor(sum_86, 0.0008680555555555555)
    mul_858: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_859: "f32[112]" = torch.ops.aten.mul.Tensor(mul_857, mul_858);  mul_857 = mul_858 = None
    unsqueeze_574: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_859, 0);  mul_859 = None
    unsqueeze_575: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
    unsqueeze_576: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
    mul_860: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_577: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_860, 0);  mul_860 = None
    unsqueeze_578: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    mul_861: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_576);  sub_207 = unsqueeze_576 = None
    sub_209: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(add_345, mul_861);  add_345 = mul_861 = None
    sub_210: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(sub_209, unsqueeze_573);  sub_209 = unsqueeze_573 = None
    mul_862: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_579);  sub_210 = unsqueeze_579 = None
    mul_863: "f32[112]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_88);  sum_86 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_862, mul_242, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_862 = mul_242 = primals_186 = None
    getitem_254: "f32[8, 480, 12, 12]" = convolution_backward_46[0]
    getitem_255: "f32[112, 480, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_864: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_254, mul_240);  mul_240 = None
    mul_865: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_254, sigmoid_39);  getitem_254 = sigmoid_39 = None
    sum_87: "f32[8, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_864, [2, 3], True);  mul_864 = None
    alias_28: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    sub_211: "f32[8, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_28)
    mul_866: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(alias_28, sub_211);  alias_28 = sub_211 = None
    mul_867: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_87, mul_866);  sum_87 = mul_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_88: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_867, [0, 2, 3])
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_867, mul_241, primals_184, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_867 = mul_241 = primals_184 = None
    getitem_257: "f32[8, 20, 1, 1]" = convolution_backward_47[0]
    getitem_258: "f32[480, 20, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_105: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(clone_29)
    full_default_28: "f32[8, 20, 1, 1]" = torch.ops.aten.full.default([8, 20, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_212: "f32[8, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_28, sigmoid_105)
    mul_868: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(clone_29, sub_212);  clone_29 = sub_212 = None
    add_346: "f32[8, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_868, 1);  mul_868 = None
    mul_869: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_105, add_346);  sigmoid_105 = add_346 = None
    mul_870: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_257, mul_869);  getitem_257 = mul_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_89: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_870, [0, 2, 3])
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_870, mean_9, primals_182, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_870 = mean_9 = primals_182 = None
    getitem_260: "f32[8, 480, 1, 1]" = convolution_backward_48[0]
    getitem_261: "f32[20, 480, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[8, 480, 12, 12]" = torch.ops.aten.expand.default(getitem_260, [8, 480, 12, 12]);  getitem_260 = None
    div_10: "f32[8, 480, 12, 12]" = torch.ops.aten.div.Scalar(expand_10, 144);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_347: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_865, div_10);  mul_865 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_106: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(clone_28)
    full_default_29: "f32[8, 480, 12, 12]" = torch.ops.aten.full.default([8, 480, 12, 12], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_213: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_29, sigmoid_106)
    mul_871: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(clone_28, sub_213);  clone_28 = sub_213 = None
    add_348: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Scalar(mul_871, 1);  mul_871 = None
    mul_872: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_106, add_348);  sigmoid_106 = add_348 = None
    mul_873: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_347, mul_872);  add_347 = mul_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_90: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_873, [0, 2, 3])
    sub_214: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_582);  convolution_46 = unsqueeze_582 = None
    mul_874: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_873, sub_214)
    sum_91: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_874, [0, 2, 3]);  mul_874 = None
    mul_875: "f32[480]" = torch.ops.aten.mul.Tensor(sum_90, 0.0008680555555555555)
    unsqueeze_583: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
    unsqueeze_584: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
    unsqueeze_585: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
    mul_876: "f32[480]" = torch.ops.aten.mul.Tensor(sum_91, 0.0008680555555555555)
    mul_877: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_878: "f32[480]" = torch.ops.aten.mul.Tensor(mul_876, mul_877);  mul_876 = mul_877 = None
    unsqueeze_586: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_878, 0);  mul_878 = None
    unsqueeze_587: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
    unsqueeze_588: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
    mul_879: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_589: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_879, 0);  mul_879 = None
    unsqueeze_590: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    mul_880: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_588);  sub_214 = unsqueeze_588 = None
    sub_216: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(mul_873, mul_880);  mul_873 = mul_880 = None
    sub_217: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_585);  sub_216 = unsqueeze_585 = None
    mul_881: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_591);  sub_217 = unsqueeze_591 = None
    mul_882: "f32[480]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_85);  sum_91 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_881, mul_232, primals_181, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_881 = mul_232 = primals_181 = None
    getitem_263: "f32[8, 480, 12, 12]" = convolution_backward_49[0]
    getitem_264: "f32[480, 1, 5, 5]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_885: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_263, mul_884);  getitem_263 = mul_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_92: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_885, [0, 2, 3])
    sub_219: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_594);  convolution_45 = unsqueeze_594 = None
    mul_886: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_885, sub_219)
    sum_93: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_886, [0, 2, 3]);  mul_886 = None
    mul_887: "f32[480]" = torch.ops.aten.mul.Tensor(sum_92, 0.0008680555555555555)
    unsqueeze_595: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_887, 0);  mul_887 = None
    unsqueeze_596: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
    unsqueeze_597: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
    mul_888: "f32[480]" = torch.ops.aten.mul.Tensor(sum_93, 0.0008680555555555555)
    mul_889: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_890: "f32[480]" = torch.ops.aten.mul.Tensor(mul_888, mul_889);  mul_888 = mul_889 = None
    unsqueeze_598: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_890, 0);  mul_890 = None
    unsqueeze_599: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
    unsqueeze_600: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
    mul_891: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_601: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
    unsqueeze_602: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    mul_892: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_600);  sub_219 = unsqueeze_600 = None
    sub_221: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(mul_885, mul_892);  mul_885 = mul_892 = None
    sub_222: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_597);  sub_221 = unsqueeze_597 = None
    mul_893: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_603);  sub_222 = unsqueeze_603 = None
    mul_894: "f32[480]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_82);  sum_93 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_893, add_139, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_893 = add_139 = primals_180 = None
    getitem_266: "f32[8, 80, 12, 12]" = convolution_backward_50[0]
    getitem_267: "f32[480, 80, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_94: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_266, [0, 2, 3])
    sub_223: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_606);  convolution_44 = unsqueeze_606 = None
    mul_895: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_266, sub_223)
    sum_95: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_895, [0, 2, 3]);  mul_895 = None
    mul_896: "f32[80]" = torch.ops.aten.mul.Tensor(sum_94, 0.0008680555555555555)
    unsqueeze_607: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_896, 0);  mul_896 = None
    unsqueeze_608: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
    unsqueeze_609: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
    mul_897: "f32[80]" = torch.ops.aten.mul.Tensor(sum_95, 0.0008680555555555555)
    mul_898: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_899: "f32[80]" = torch.ops.aten.mul.Tensor(mul_897, mul_898);  mul_897 = mul_898 = None
    unsqueeze_610: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
    unsqueeze_611: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
    unsqueeze_612: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
    mul_900: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_613: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_900, 0);  mul_900 = None
    unsqueeze_614: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    mul_901: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_612);  sub_223 = unsqueeze_612 = None
    sub_225: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(getitem_266, mul_901);  mul_901 = None
    sub_226: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(sub_225, unsqueeze_609);  sub_225 = unsqueeze_609 = None
    mul_902: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_615);  sub_226 = unsqueeze_615 = None
    mul_903: "f32[80]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_79);  sum_95 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_902, mul_217, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_902 = mul_217 = primals_179 = None
    getitem_269: "f32[8, 480, 12, 12]" = convolution_backward_51[0]
    getitem_270: "f32[80, 480, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_904: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_269, mul_215);  mul_215 = None
    mul_905: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_269, sigmoid_35);  getitem_269 = sigmoid_35 = None
    sum_96: "f32[8, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_904, [2, 3], True);  mul_904 = None
    alias_29: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    sub_227: "f32[8, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_29)
    mul_906: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(alias_29, sub_227);  alias_29 = sub_227 = None
    mul_907: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_96, mul_906);  sum_96 = mul_906 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_97: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_907, [0, 2, 3])
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_907, mul_216, primals_177, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_907 = mul_216 = primals_177 = None
    getitem_272: "f32[8, 20, 1, 1]" = convolution_backward_52[0]
    getitem_273: "f32[480, 20, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_108: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(clone_26)
    sub_228: "f32[8, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_28, sigmoid_108)
    mul_908: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(clone_26, sub_228);  clone_26 = sub_228 = None
    add_350: "f32[8, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_908, 1);  mul_908 = None
    mul_909: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_108, add_350);  sigmoid_108 = add_350 = None
    mul_910: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_272, mul_909);  getitem_272 = mul_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_98: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_910, [0, 2, 3])
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_910, mean_8, primals_175, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_910 = mean_8 = primals_175 = None
    getitem_275: "f32[8, 480, 1, 1]" = convolution_backward_53[0]
    getitem_276: "f32[20, 480, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[8, 480, 12, 12]" = torch.ops.aten.expand.default(getitem_275, [8, 480, 12, 12]);  getitem_275 = None
    div_11: "f32[8, 480, 12, 12]" = torch.ops.aten.div.Scalar(expand_11, 144);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_351: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_905, div_11);  mul_905 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_109: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(clone_25)
    sub_229: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_29, sigmoid_109)
    mul_911: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(clone_25, sub_229);  clone_25 = sub_229 = None
    add_352: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Scalar(mul_911, 1);  mul_911 = None
    mul_912: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_109, add_352);  sigmoid_109 = add_352 = None
    mul_913: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_351, mul_912);  add_351 = mul_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_99: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_913, [0, 2, 3])
    sub_230: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_618);  convolution_41 = unsqueeze_618 = None
    mul_914: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_913, sub_230)
    sum_100: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_914, [0, 2, 3]);  mul_914 = None
    mul_915: "f32[480]" = torch.ops.aten.mul.Tensor(sum_99, 0.0008680555555555555)
    unsqueeze_619: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_915, 0);  mul_915 = None
    unsqueeze_620: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
    unsqueeze_621: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
    mul_916: "f32[480]" = torch.ops.aten.mul.Tensor(sum_100, 0.0008680555555555555)
    mul_917: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_918: "f32[480]" = torch.ops.aten.mul.Tensor(mul_916, mul_917);  mul_916 = mul_917 = None
    unsqueeze_622: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
    unsqueeze_623: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    mul_919: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_625: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_919, 0);  mul_919 = None
    unsqueeze_626: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    mul_920: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_624);  sub_230 = unsqueeze_624 = None
    sub_232: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(mul_913, mul_920);  mul_913 = mul_920 = None
    sub_233: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(sub_232, unsqueeze_621);  sub_232 = unsqueeze_621 = None
    mul_921: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_627);  sub_233 = unsqueeze_627 = None
    mul_922: "f32[480]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_76);  sum_100 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_921, mul_207, primals_174, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_921 = mul_207 = primals_174 = None
    getitem_278: "f32[8, 480, 12, 12]" = convolution_backward_54[0]
    getitem_279: "f32[480, 1, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_925: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_278, mul_924);  getitem_278 = mul_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_101: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_925, [0, 2, 3])
    sub_235: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_630);  convolution_40 = unsqueeze_630 = None
    mul_926: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_925, sub_235)
    sum_102: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_926, [0, 2, 3]);  mul_926 = None
    mul_927: "f32[480]" = torch.ops.aten.mul.Tensor(sum_101, 0.0008680555555555555)
    unsqueeze_631: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
    unsqueeze_632: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
    unsqueeze_633: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
    mul_928: "f32[480]" = torch.ops.aten.mul.Tensor(sum_102, 0.0008680555555555555)
    mul_929: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_930: "f32[480]" = torch.ops.aten.mul.Tensor(mul_928, mul_929);  mul_928 = mul_929 = None
    unsqueeze_634: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_930, 0);  mul_930 = None
    unsqueeze_635: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
    unsqueeze_636: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
    mul_931: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_637: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
    unsqueeze_638: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    mul_932: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_636);  sub_235 = unsqueeze_636 = None
    sub_237: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(mul_925, mul_932);  mul_925 = mul_932 = None
    sub_238: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_633);  sub_237 = unsqueeze_633 = None
    mul_933: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_639);  sub_238 = unsqueeze_639 = None
    mul_934: "f32[480]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_73);  sum_102 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_933, add_123, primals_173, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_933 = add_123 = primals_173 = None
    getitem_281: "f32[8, 80, 12, 12]" = convolution_backward_55[0]
    getitem_282: "f32[480, 80, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_354: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(getitem_266, getitem_281);  getitem_266 = getitem_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_103: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_354, [0, 2, 3])
    sub_239: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_642);  convolution_39 = unsqueeze_642 = None
    mul_935: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(add_354, sub_239)
    sum_104: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_935, [0, 2, 3]);  mul_935 = None
    mul_936: "f32[80]" = torch.ops.aten.mul.Tensor(sum_103, 0.0008680555555555555)
    unsqueeze_643: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
    unsqueeze_644: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
    unsqueeze_645: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
    mul_937: "f32[80]" = torch.ops.aten.mul.Tensor(sum_104, 0.0008680555555555555)
    mul_938: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_939: "f32[80]" = torch.ops.aten.mul.Tensor(mul_937, mul_938);  mul_937 = mul_938 = None
    unsqueeze_646: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_939, 0);  mul_939 = None
    unsqueeze_647: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
    unsqueeze_648: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
    mul_940: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_649: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_940, 0);  mul_940 = None
    unsqueeze_650: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    mul_941: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_648);  sub_239 = unsqueeze_648 = None
    sub_241: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(add_354, mul_941);  mul_941 = None
    sub_242: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(sub_241, unsqueeze_645);  sub_241 = unsqueeze_645 = None
    mul_942: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_651);  sub_242 = unsqueeze_651 = None
    mul_943: "f32[80]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_70);  sum_104 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_942, mul_192, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_942 = mul_192 = primals_172 = None
    getitem_284: "f32[8, 480, 12, 12]" = convolution_backward_56[0]
    getitem_285: "f32[80, 480, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_944: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_284, mul_190);  mul_190 = None
    mul_945: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_284, sigmoid_31);  getitem_284 = sigmoid_31 = None
    sum_105: "f32[8, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_944, [2, 3], True);  mul_944 = None
    alias_30: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    sub_243: "f32[8, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_30)
    mul_946: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(alias_30, sub_243);  alias_30 = sub_243 = None
    mul_947: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_105, mul_946);  sum_105 = mul_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_106: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_947, [0, 2, 3])
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_947, mul_191, primals_170, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_947 = mul_191 = primals_170 = None
    getitem_287: "f32[8, 20, 1, 1]" = convolution_backward_57[0]
    getitem_288: "f32[480, 20, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_111: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(clone_23)
    sub_244: "f32[8, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_28, sigmoid_111)
    mul_948: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(clone_23, sub_244);  clone_23 = sub_244 = None
    add_355: "f32[8, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_948, 1);  mul_948 = None
    mul_949: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_111, add_355);  sigmoid_111 = add_355 = None
    mul_950: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_287, mul_949);  getitem_287 = mul_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_107: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_950, [0, 2, 3])
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_950, mean_7, primals_168, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_950 = mean_7 = primals_168 = None
    getitem_290: "f32[8, 480, 1, 1]" = convolution_backward_58[0]
    getitem_291: "f32[20, 480, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[8, 480, 12, 12]" = torch.ops.aten.expand.default(getitem_290, [8, 480, 12, 12]);  getitem_290 = None
    div_12: "f32[8, 480, 12, 12]" = torch.ops.aten.div.Scalar(expand_12, 144);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_356: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_945, div_12);  mul_945 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_112: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(clone_22)
    sub_245: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_29, sigmoid_112)
    mul_951: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(clone_22, sub_245);  clone_22 = sub_245 = None
    add_357: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Scalar(mul_951, 1);  mul_951 = None
    mul_952: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_112, add_357);  sigmoid_112 = add_357 = None
    mul_953: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_356, mul_952);  add_356 = mul_952 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_108: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_953, [0, 2, 3])
    sub_246: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_654);  convolution_36 = unsqueeze_654 = None
    mul_954: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_953, sub_246)
    sum_109: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_954, [0, 2, 3]);  mul_954 = None
    mul_955: "f32[480]" = torch.ops.aten.mul.Tensor(sum_108, 0.0008680555555555555)
    unsqueeze_655: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_955, 0);  mul_955 = None
    unsqueeze_656: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
    unsqueeze_657: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
    mul_956: "f32[480]" = torch.ops.aten.mul.Tensor(sum_109, 0.0008680555555555555)
    mul_957: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_958: "f32[480]" = torch.ops.aten.mul.Tensor(mul_956, mul_957);  mul_956 = mul_957 = None
    unsqueeze_658: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_958, 0);  mul_958 = None
    unsqueeze_659: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
    unsqueeze_660: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
    mul_959: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_661: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_959, 0);  mul_959 = None
    unsqueeze_662: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    mul_960: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_660);  sub_246 = unsqueeze_660 = None
    sub_248: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(mul_953, mul_960);  mul_953 = mul_960 = None
    sub_249: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(sub_248, unsqueeze_657);  sub_248 = unsqueeze_657 = None
    mul_961: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_663);  sub_249 = unsqueeze_663 = None
    mul_962: "f32[480]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_67);  sum_109 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_961, mul_182, primals_167, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_961 = mul_182 = primals_167 = None
    getitem_293: "f32[8, 480, 12, 12]" = convolution_backward_59[0]
    getitem_294: "f32[480, 1, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_965: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_293, mul_964);  getitem_293 = mul_964 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_110: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_965, [0, 2, 3])
    sub_251: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_666);  convolution_35 = unsqueeze_666 = None
    mul_966: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_965, sub_251)
    sum_111: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_966, [0, 2, 3]);  mul_966 = None
    mul_967: "f32[480]" = torch.ops.aten.mul.Tensor(sum_110, 0.0008680555555555555)
    unsqueeze_667: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_967, 0);  mul_967 = None
    unsqueeze_668: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
    unsqueeze_669: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
    mul_968: "f32[480]" = torch.ops.aten.mul.Tensor(sum_111, 0.0008680555555555555)
    mul_969: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_970: "f32[480]" = torch.ops.aten.mul.Tensor(mul_968, mul_969);  mul_968 = mul_969 = None
    unsqueeze_670: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_970, 0);  mul_970 = None
    unsqueeze_671: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
    unsqueeze_672: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
    mul_971: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_673: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
    unsqueeze_674: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    mul_972: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_672);  sub_251 = unsqueeze_672 = None
    sub_253: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(mul_965, mul_972);  mul_965 = mul_972 = None
    sub_254: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(sub_253, unsqueeze_669);  sub_253 = unsqueeze_669 = None
    mul_973: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_675);  sub_254 = unsqueeze_675 = None
    mul_974: "f32[480]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_64);  sum_111 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_973, add_107, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_973 = add_107 = primals_166 = None
    getitem_296: "f32[8, 80, 12, 12]" = convolution_backward_60[0]
    getitem_297: "f32[480, 80, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_359: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(add_354, getitem_296);  add_354 = getitem_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_112: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_359, [0, 2, 3])
    sub_255: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_678);  convolution_34 = unsqueeze_678 = None
    mul_975: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(add_359, sub_255)
    sum_113: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_975, [0, 2, 3]);  mul_975 = None
    mul_976: "f32[80]" = torch.ops.aten.mul.Tensor(sum_112, 0.0008680555555555555)
    unsqueeze_679: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_976, 0);  mul_976 = None
    unsqueeze_680: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
    unsqueeze_681: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
    mul_977: "f32[80]" = torch.ops.aten.mul.Tensor(sum_113, 0.0008680555555555555)
    mul_978: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_979: "f32[80]" = torch.ops.aten.mul.Tensor(mul_977, mul_978);  mul_977 = mul_978 = None
    unsqueeze_682: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_979, 0);  mul_979 = None
    unsqueeze_683: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
    unsqueeze_684: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
    mul_980: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_685: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_980, 0);  mul_980 = None
    unsqueeze_686: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    mul_981: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_684);  sub_255 = unsqueeze_684 = None
    sub_257: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(add_359, mul_981);  mul_981 = None
    sub_258: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(sub_257, unsqueeze_681);  sub_257 = unsqueeze_681 = None
    mul_982: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_687);  sub_258 = unsqueeze_687 = None
    mul_983: "f32[80]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_61);  sum_113 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_982, mul_167, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_982 = mul_167 = primals_165 = None
    getitem_299: "f32[8, 480, 12, 12]" = convolution_backward_61[0]
    getitem_300: "f32[80, 480, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_984: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_299, mul_165);  mul_165 = None
    mul_985: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_299, sigmoid_27);  getitem_299 = sigmoid_27 = None
    sum_114: "f32[8, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_984, [2, 3], True);  mul_984 = None
    alias_31: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    sub_259: "f32[8, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_31)
    mul_986: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(alias_31, sub_259);  alias_31 = sub_259 = None
    mul_987: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_114, mul_986);  sum_114 = mul_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_115: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_987, [0, 2, 3])
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_987, mul_166, primals_163, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_987 = mul_166 = primals_163 = None
    getitem_302: "f32[8, 20, 1, 1]" = convolution_backward_62[0]
    getitem_303: "f32[480, 20, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_114: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(clone_20)
    sub_260: "f32[8, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_28, sigmoid_114);  full_default_28 = None
    mul_988: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(clone_20, sub_260);  clone_20 = sub_260 = None
    add_360: "f32[8, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_988, 1);  mul_988 = None
    mul_989: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_114, add_360);  sigmoid_114 = add_360 = None
    mul_990: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_302, mul_989);  getitem_302 = mul_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_116: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_990, [0, 2, 3])
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_990, mean_6, primals_161, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_990 = mean_6 = primals_161 = None
    getitem_305: "f32[8, 480, 1, 1]" = convolution_backward_63[0]
    getitem_306: "f32[20, 480, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_13: "f32[8, 480, 12, 12]" = torch.ops.aten.expand.default(getitem_305, [8, 480, 12, 12]);  getitem_305 = None
    div_13: "f32[8, 480, 12, 12]" = torch.ops.aten.div.Scalar(expand_13, 144);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_361: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_985, div_13);  mul_985 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_115: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(clone_19)
    sub_261: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_29, sigmoid_115);  full_default_29 = None
    mul_991: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(clone_19, sub_261);  clone_19 = sub_261 = None
    add_362: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Scalar(mul_991, 1);  mul_991 = None
    mul_992: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_115, add_362);  sigmoid_115 = add_362 = None
    mul_993: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_361, mul_992);  add_361 = mul_992 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_117: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_993, [0, 2, 3])
    sub_262: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_690);  convolution_31 = unsqueeze_690 = None
    mul_994: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_993, sub_262)
    sum_118: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_994, [0, 2, 3]);  mul_994 = None
    mul_995: "f32[480]" = torch.ops.aten.mul.Tensor(sum_117, 0.0008680555555555555)
    unsqueeze_691: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
    unsqueeze_692: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
    unsqueeze_693: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
    mul_996: "f32[480]" = torch.ops.aten.mul.Tensor(sum_118, 0.0008680555555555555)
    mul_997: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_998: "f32[480]" = torch.ops.aten.mul.Tensor(mul_996, mul_997);  mul_996 = mul_997 = None
    unsqueeze_694: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_998, 0);  mul_998 = None
    unsqueeze_695: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    mul_999: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_697: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_999, 0);  mul_999 = None
    unsqueeze_698: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    mul_1000: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_696);  sub_262 = unsqueeze_696 = None
    sub_264: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(mul_993, mul_1000);  mul_993 = mul_1000 = None
    sub_265: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(sub_264, unsqueeze_693);  sub_264 = unsqueeze_693 = None
    mul_1001: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_699);  sub_265 = unsqueeze_699 = None
    mul_1002: "f32[480]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_58);  sum_118 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1001, mul_157, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_1001 = mul_157 = primals_160 = None
    getitem_308: "f32[8, 480, 12, 12]" = convolution_backward_64[0]
    getitem_309: "f32[480, 1, 3, 3]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1005: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_308, mul_1004);  getitem_308 = mul_1004 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_119: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_1005, [0, 2, 3])
    sub_267: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_702);  convolution_30 = unsqueeze_702 = None
    mul_1006: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1005, sub_267)
    sum_120: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_1006, [0, 2, 3]);  mul_1006 = None
    mul_1007: "f32[480]" = torch.ops.aten.mul.Tensor(sum_119, 0.0008680555555555555)
    unsqueeze_703: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_1007, 0);  mul_1007 = None
    unsqueeze_704: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_1008: "f32[480]" = torch.ops.aten.mul.Tensor(sum_120, 0.0008680555555555555)
    mul_1009: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1010: "f32[480]" = torch.ops.aten.mul.Tensor(mul_1008, mul_1009);  mul_1008 = mul_1009 = None
    unsqueeze_706: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_1010, 0);  mul_1010 = None
    unsqueeze_707: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    mul_1011: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_709: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_1011, 0);  mul_1011 = None
    unsqueeze_710: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    mul_1012: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_708);  sub_267 = unsqueeze_708 = None
    sub_269: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(mul_1005, mul_1012);  mul_1005 = mul_1012 = None
    sub_270: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(sub_269, unsqueeze_705);  sub_269 = unsqueeze_705 = None
    mul_1013: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_711);  sub_270 = unsqueeze_711 = None
    mul_1014: "f32[480]" = torch.ops.aten.mul.Tensor(sum_120, squeeze_55);  sum_120 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1013, add_91, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1013 = add_91 = primals_159 = None
    getitem_311: "f32[8, 80, 12, 12]" = convolution_backward_65[0]
    getitem_312: "f32[480, 80, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_364: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(add_359, getitem_311);  add_359 = getitem_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_121: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_364, [0, 2, 3])
    sub_271: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_714);  convolution_29 = unsqueeze_714 = None
    mul_1015: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(add_364, sub_271)
    sum_122: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_1015, [0, 2, 3]);  mul_1015 = None
    mul_1016: "f32[80]" = torch.ops.aten.mul.Tensor(sum_121, 0.0008680555555555555)
    unsqueeze_715: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_1016, 0);  mul_1016 = None
    unsqueeze_716: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
    unsqueeze_717: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
    mul_1017: "f32[80]" = torch.ops.aten.mul.Tensor(sum_122, 0.0008680555555555555)
    mul_1018: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1019: "f32[80]" = torch.ops.aten.mul.Tensor(mul_1017, mul_1018);  mul_1017 = mul_1018 = None
    unsqueeze_718: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
    unsqueeze_719: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    mul_1020: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_721: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_1020, 0);  mul_1020 = None
    unsqueeze_722: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    mul_1021: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_720);  sub_271 = unsqueeze_720 = None
    sub_273: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(add_364, mul_1021);  add_364 = mul_1021 = None
    sub_274: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(sub_273, unsqueeze_717);  sub_273 = unsqueeze_717 = None
    mul_1022: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_723);  sub_274 = unsqueeze_723 = None
    mul_1023: "f32[80]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_52);  sum_122 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1022, mul_142, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1022 = mul_142 = primals_158 = None
    getitem_314: "f32[8, 240, 12, 12]" = convolution_backward_66[0]
    getitem_315: "f32[80, 240, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1024: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_314, mul_140);  mul_140 = None
    mul_1025: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(getitem_314, sigmoid_23);  getitem_314 = sigmoid_23 = None
    sum_123: "f32[8, 240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1024, [2, 3], True);  mul_1024 = None
    alias_32: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    sub_275: "f32[8, 240, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_32)
    mul_1026: "f32[8, 240, 1, 1]" = torch.ops.aten.mul.Tensor(alias_32, sub_275);  alias_32 = sub_275 = None
    mul_1027: "f32[8, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_123, mul_1026);  sum_123 = mul_1026 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_124: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1027, [0, 2, 3])
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1027, mul_141, primals_156, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1027 = mul_141 = primals_156 = None
    getitem_317: "f32[8, 10, 1, 1]" = convolution_backward_67[0]
    getitem_318: "f32[240, 10, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_117: "f32[8, 10, 1, 1]" = torch.ops.aten.sigmoid.default(clone_17)
    full_default_40: "f32[8, 10, 1, 1]" = torch.ops.aten.full.default([8, 10, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_276: "f32[8, 10, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_40, sigmoid_117)
    mul_1028: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(clone_17, sub_276);  clone_17 = sub_276 = None
    add_365: "f32[8, 10, 1, 1]" = torch.ops.aten.add.Scalar(mul_1028, 1);  mul_1028 = None
    mul_1029: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_117, add_365);  sigmoid_117 = add_365 = None
    mul_1030: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_317, mul_1029);  getitem_317 = mul_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_125: "f32[10]" = torch.ops.aten.sum.dim_IntList(mul_1030, [0, 2, 3])
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1030, mean_5, primals_154, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1030 = mean_5 = primals_154 = None
    getitem_320: "f32[8, 240, 1, 1]" = convolution_backward_68[0]
    getitem_321: "f32[10, 240, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_14: "f32[8, 240, 12, 12]" = torch.ops.aten.expand.default(getitem_320, [8, 240, 12, 12]);  getitem_320 = None
    div_14: "f32[8, 240, 12, 12]" = torch.ops.aten.div.Scalar(expand_14, 144);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_366: "f32[8, 240, 12, 12]" = torch.ops.aten.add.Tensor(mul_1025, div_14);  mul_1025 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_118: "f32[8, 240, 12, 12]" = torch.ops.aten.sigmoid.default(clone_16)
    full_default_41: "f32[8, 240, 12, 12]" = torch.ops.aten.full.default([8, 240, 12, 12], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_277: "f32[8, 240, 12, 12]" = torch.ops.aten.sub.Tensor(full_default_41, sigmoid_118);  full_default_41 = None
    mul_1031: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(clone_16, sub_277);  clone_16 = sub_277 = None
    add_367: "f32[8, 240, 12, 12]" = torch.ops.aten.add.Scalar(mul_1031, 1);  mul_1031 = None
    mul_1032: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(sigmoid_118, add_367);  sigmoid_118 = add_367 = None
    mul_1033: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(add_366, mul_1032);  add_366 = mul_1032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_126: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1033, [0, 2, 3])
    sub_278: "f32[8, 240, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_726);  convolution_26 = unsqueeze_726 = None
    mul_1034: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1033, sub_278)
    sum_127: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1034, [0, 2, 3]);  mul_1034 = None
    mul_1035: "f32[240]" = torch.ops.aten.mul.Tensor(sum_126, 0.0008680555555555555)
    unsqueeze_727: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1035, 0);  mul_1035 = None
    unsqueeze_728: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
    unsqueeze_729: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
    mul_1036: "f32[240]" = torch.ops.aten.mul.Tensor(sum_127, 0.0008680555555555555)
    mul_1037: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1038: "f32[240]" = torch.ops.aten.mul.Tensor(mul_1036, mul_1037);  mul_1036 = mul_1037 = None
    unsqueeze_730: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1038, 0);  mul_1038 = None
    unsqueeze_731: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
    unsqueeze_732: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
    mul_1039: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_733: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1039, 0);  mul_1039 = None
    unsqueeze_734: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    mul_1040: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_732);  sub_278 = unsqueeze_732 = None
    sub_280: "f32[8, 240, 12, 12]" = torch.ops.aten.sub.Tensor(mul_1033, mul_1040);  mul_1033 = mul_1040 = None
    sub_281: "f32[8, 240, 12, 12]" = torch.ops.aten.sub.Tensor(sub_280, unsqueeze_729);  sub_280 = unsqueeze_729 = None
    mul_1041: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_735);  sub_281 = unsqueeze_735 = None
    mul_1042: "f32[240]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_49);  sum_127 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1041, mul_132, primals_153, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_1041 = mul_132 = primals_153 = None
    getitem_323: "f32[8, 240, 24, 24]" = convolution_backward_69[0]
    getitem_324: "f32[240, 1, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_42: "f32[8, 240, 24, 24]" = torch.ops.aten.full.default([8, 240, 24, 24], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    mul_1045: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_323, mul_1044);  getitem_323 = mul_1044 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_128: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1045, [0, 2, 3])
    sub_283: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_738);  convolution_25 = unsqueeze_738 = None
    mul_1046: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1045, sub_283)
    sum_129: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1046, [0, 2, 3]);  mul_1046 = None
    mul_1047: "f32[240]" = torch.ops.aten.mul.Tensor(sum_128, 0.00021701388888888888)
    unsqueeze_739: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1047, 0);  mul_1047 = None
    unsqueeze_740: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
    unsqueeze_741: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
    mul_1048: "f32[240]" = torch.ops.aten.mul.Tensor(sum_129, 0.00021701388888888888)
    mul_1049: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1050: "f32[240]" = torch.ops.aten.mul.Tensor(mul_1048, mul_1049);  mul_1048 = mul_1049 = None
    unsqueeze_742: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1050, 0);  mul_1050 = None
    unsqueeze_743: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
    unsqueeze_744: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
    mul_1051: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_745: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1051, 0);  mul_1051 = None
    unsqueeze_746: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    mul_1052: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_744);  sub_283 = unsqueeze_744 = None
    sub_285: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(mul_1045, mul_1052);  mul_1045 = mul_1052 = None
    sub_286: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(sub_285, unsqueeze_741);  sub_285 = unsqueeze_741 = None
    mul_1053: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_747);  sub_286 = unsqueeze_747 = None
    mul_1054: "f32[240]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_46);  sum_129 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1053, add_76, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1053 = add_76 = primals_152 = None
    getitem_326: "f32[8, 40, 24, 24]" = convolution_backward_70[0]
    getitem_327: "f32[240, 40, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_130: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_326, [0, 2, 3])
    sub_287: "f32[8, 40, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_750);  convolution_24 = unsqueeze_750 = None
    mul_1055: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_326, sub_287)
    sum_131: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1055, [0, 2, 3]);  mul_1055 = None
    mul_1056: "f32[40]" = torch.ops.aten.mul.Tensor(sum_130, 0.00021701388888888888)
    unsqueeze_751: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1056, 0);  mul_1056 = None
    unsqueeze_752: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
    unsqueeze_753: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
    mul_1057: "f32[40]" = torch.ops.aten.mul.Tensor(sum_131, 0.00021701388888888888)
    mul_1058: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1059: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1057, mul_1058);  mul_1057 = mul_1058 = None
    unsqueeze_754: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1059, 0);  mul_1059 = None
    unsqueeze_755: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
    unsqueeze_756: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
    mul_1060: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_757: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1060, 0);  mul_1060 = None
    unsqueeze_758: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    mul_1061: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_756);  sub_287 = unsqueeze_756 = None
    sub_289: "f32[8, 40, 24, 24]" = torch.ops.aten.sub.Tensor(getitem_326, mul_1061);  mul_1061 = None
    sub_290: "f32[8, 40, 24, 24]" = torch.ops.aten.sub.Tensor(sub_289, unsqueeze_753);  sub_289 = unsqueeze_753 = None
    mul_1062: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_759);  sub_290 = unsqueeze_759 = None
    mul_1063: "f32[40]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_43);  sum_131 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1062, mul_117, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1062 = mul_117 = primals_151 = None
    getitem_329: "f32[8, 240, 24, 24]" = convolution_backward_71[0]
    getitem_330: "f32[40, 240, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1064: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_329, mul_115);  mul_115 = None
    mul_1065: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_329, sigmoid_19);  getitem_329 = sigmoid_19 = None
    sum_132: "f32[8, 240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1064, [2, 3], True);  mul_1064 = None
    alias_33: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    sub_291: "f32[8, 240, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_33)
    mul_1066: "f32[8, 240, 1, 1]" = torch.ops.aten.mul.Tensor(alias_33, sub_291);  alias_33 = sub_291 = None
    mul_1067: "f32[8, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_132, mul_1066);  sum_132 = mul_1066 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_133: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1067, [0, 2, 3])
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1067, mul_116, primals_149, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1067 = mul_116 = primals_149 = None
    getitem_332: "f32[8, 10, 1, 1]" = convolution_backward_72[0]
    getitem_333: "f32[240, 10, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_120: "f32[8, 10, 1, 1]" = torch.ops.aten.sigmoid.default(clone_14)
    sub_292: "f32[8, 10, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_40, sigmoid_120);  full_default_40 = None
    mul_1068: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(clone_14, sub_292);  clone_14 = sub_292 = None
    add_369: "f32[8, 10, 1, 1]" = torch.ops.aten.add.Scalar(mul_1068, 1);  mul_1068 = None
    mul_1069: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_120, add_369);  sigmoid_120 = add_369 = None
    mul_1070: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_332, mul_1069);  getitem_332 = mul_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_134: "f32[10]" = torch.ops.aten.sum.dim_IntList(mul_1070, [0, 2, 3])
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1070, mean_4, primals_147, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1070 = mean_4 = primals_147 = None
    getitem_335: "f32[8, 240, 1, 1]" = convolution_backward_73[0]
    getitem_336: "f32[10, 240, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_15: "f32[8, 240, 24, 24]" = torch.ops.aten.expand.default(getitem_335, [8, 240, 24, 24]);  getitem_335 = None
    div_15: "f32[8, 240, 24, 24]" = torch.ops.aten.div.Scalar(expand_15, 576);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_370: "f32[8, 240, 24, 24]" = torch.ops.aten.add.Tensor(mul_1065, div_15);  mul_1065 = div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_121: "f32[8, 240, 24, 24]" = torch.ops.aten.sigmoid.default(clone_13)
    sub_293: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(full_default_42, sigmoid_121);  full_default_42 = None
    mul_1071: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(clone_13, sub_293);  clone_13 = sub_293 = None
    add_371: "f32[8, 240, 24, 24]" = torch.ops.aten.add.Scalar(mul_1071, 1);  mul_1071 = None
    mul_1072: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sigmoid_121, add_371);  sigmoid_121 = add_371 = None
    mul_1073: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(add_370, mul_1072);  add_370 = mul_1072 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_135: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1073, [0, 2, 3])
    sub_294: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_762);  convolution_21 = unsqueeze_762 = None
    mul_1074: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1073, sub_294)
    sum_136: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1074, [0, 2, 3]);  mul_1074 = None
    mul_1075: "f32[240]" = torch.ops.aten.mul.Tensor(sum_135, 0.00021701388888888888)
    unsqueeze_763: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1075, 0);  mul_1075 = None
    unsqueeze_764: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
    unsqueeze_765: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
    mul_1076: "f32[240]" = torch.ops.aten.mul.Tensor(sum_136, 0.00021701388888888888)
    mul_1077: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1078: "f32[240]" = torch.ops.aten.mul.Tensor(mul_1076, mul_1077);  mul_1076 = mul_1077 = None
    unsqueeze_766: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1078, 0);  mul_1078 = None
    unsqueeze_767: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
    unsqueeze_768: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
    mul_1079: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_769: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1079, 0);  mul_1079 = None
    unsqueeze_770: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    mul_1080: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_768);  sub_294 = unsqueeze_768 = None
    sub_296: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(mul_1073, mul_1080);  mul_1073 = mul_1080 = None
    sub_297: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(sub_296, unsqueeze_765);  sub_296 = unsqueeze_765 = None
    mul_1081: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_771);  sub_297 = unsqueeze_771 = None
    mul_1082: "f32[240]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_40);  sum_136 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1081, mul_107, primals_146, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_1081 = mul_107 = primals_146 = None
    getitem_338: "f32[8, 240, 24, 24]" = convolution_backward_74[0]
    getitem_339: "f32[240, 1, 5, 5]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1085: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_338, mul_1084);  getitem_338 = mul_1084 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_137: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1085, [0, 2, 3])
    sub_299: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_774);  convolution_20 = unsqueeze_774 = None
    mul_1086: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1085, sub_299)
    sum_138: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1086, [0, 2, 3]);  mul_1086 = None
    mul_1087: "f32[240]" = torch.ops.aten.mul.Tensor(sum_137, 0.00021701388888888888)
    unsqueeze_775: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1087, 0);  mul_1087 = None
    unsqueeze_776: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
    unsqueeze_777: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
    mul_1088: "f32[240]" = torch.ops.aten.mul.Tensor(sum_138, 0.00021701388888888888)
    mul_1089: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1090: "f32[240]" = torch.ops.aten.mul.Tensor(mul_1088, mul_1089);  mul_1088 = mul_1089 = None
    unsqueeze_778: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1090, 0);  mul_1090 = None
    unsqueeze_779: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
    unsqueeze_780: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
    mul_1091: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_781: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1091, 0);  mul_1091 = None
    unsqueeze_782: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    mul_1092: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_780);  sub_299 = unsqueeze_780 = None
    sub_301: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(mul_1085, mul_1092);  mul_1085 = mul_1092 = None
    sub_302: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(sub_301, unsqueeze_777);  sub_301 = unsqueeze_777 = None
    mul_1093: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_783);  sub_302 = unsqueeze_783 = None
    mul_1094: "f32[240]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_37);  sum_138 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1093, add_60, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1093 = add_60 = primals_145 = None
    getitem_341: "f32[8, 40, 24, 24]" = convolution_backward_75[0]
    getitem_342: "f32[240, 40, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_373: "f32[8, 40, 24, 24]" = torch.ops.aten.add.Tensor(getitem_326, getitem_341);  getitem_326 = getitem_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_139: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_373, [0, 2, 3])
    sub_303: "f32[8, 40, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_786);  convolution_19 = unsqueeze_786 = None
    mul_1095: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(add_373, sub_303)
    sum_140: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1095, [0, 2, 3]);  mul_1095 = None
    mul_1096: "f32[40]" = torch.ops.aten.mul.Tensor(sum_139, 0.00021701388888888888)
    unsqueeze_787: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1096, 0);  mul_1096 = None
    unsqueeze_788: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
    unsqueeze_789: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
    mul_1097: "f32[40]" = torch.ops.aten.mul.Tensor(sum_140, 0.00021701388888888888)
    mul_1098: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1099: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1097, mul_1098);  mul_1097 = mul_1098 = None
    unsqueeze_790: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1099, 0);  mul_1099 = None
    unsqueeze_791: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
    unsqueeze_792: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
    mul_1100: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_793: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1100, 0);  mul_1100 = None
    unsqueeze_794: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    mul_1101: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_792);  sub_303 = unsqueeze_792 = None
    sub_305: "f32[8, 40, 24, 24]" = torch.ops.aten.sub.Tensor(add_373, mul_1101);  add_373 = mul_1101 = None
    sub_306: "f32[8, 40, 24, 24]" = torch.ops.aten.sub.Tensor(sub_305, unsqueeze_789);  sub_305 = unsqueeze_789 = None
    mul_1102: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_795);  sub_306 = unsqueeze_795 = None
    mul_1103: "f32[40]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_34);  sum_140 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1102, mul_92, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1102 = mul_92 = primals_144 = None
    getitem_344: "f32[8, 144, 24, 24]" = convolution_backward_76[0]
    getitem_345: "f32[40, 144, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1104: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_344, mul_90);  mul_90 = None
    mul_1105: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_344, sigmoid_15);  getitem_344 = sigmoid_15 = None
    sum_141: "f32[8, 144, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1104, [2, 3], True);  mul_1104 = None
    alias_34: "f32[8, 144, 1, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    sub_307: "f32[8, 144, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_34)
    mul_1106: "f32[8, 144, 1, 1]" = torch.ops.aten.mul.Tensor(alias_34, sub_307);  alias_34 = sub_307 = None
    mul_1107: "f32[8, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sum_141, mul_1106);  sum_141 = mul_1106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_142: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_1107, [0, 2, 3])
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1107, mul_91, primals_142, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1107 = mul_91 = primals_142 = None
    getitem_347: "f32[8, 6, 1, 1]" = convolution_backward_77[0]
    getitem_348: "f32[144, 6, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_123: "f32[8, 6, 1, 1]" = torch.ops.aten.sigmoid.default(clone_11)
    full_default_46: "f32[8, 6, 1, 1]" = torch.ops.aten.full.default([8, 6, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_308: "f32[8, 6, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_46, sigmoid_123)
    mul_1108: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(clone_11, sub_308);  clone_11 = sub_308 = None
    add_374: "f32[8, 6, 1, 1]" = torch.ops.aten.add.Scalar(mul_1108, 1);  mul_1108 = None
    mul_1109: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_123, add_374);  sigmoid_123 = add_374 = None
    mul_1110: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_347, mul_1109);  getitem_347 = mul_1109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_143: "f32[6]" = torch.ops.aten.sum.dim_IntList(mul_1110, [0, 2, 3])
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1110, mean_3, primals_140, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1110 = mean_3 = primals_140 = None
    getitem_350: "f32[8, 144, 1, 1]" = convolution_backward_78[0]
    getitem_351: "f32[6, 144, 1, 1]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_16: "f32[8, 144, 24, 24]" = torch.ops.aten.expand.default(getitem_350, [8, 144, 24, 24]);  getitem_350 = None
    div_16: "f32[8, 144, 24, 24]" = torch.ops.aten.div.Scalar(expand_16, 576);  expand_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_375: "f32[8, 144, 24, 24]" = torch.ops.aten.add.Tensor(mul_1105, div_16);  mul_1105 = div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_124: "f32[8, 144, 24, 24]" = torch.ops.aten.sigmoid.default(clone_10)
    full_default_47: "f32[8, 144, 24, 24]" = torch.ops.aten.full.default([8, 144, 24, 24], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_309: "f32[8, 144, 24, 24]" = torch.ops.aten.sub.Tensor(full_default_47, sigmoid_124);  full_default_47 = None
    mul_1111: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(clone_10, sub_309);  clone_10 = sub_309 = None
    add_376: "f32[8, 144, 24, 24]" = torch.ops.aten.add.Scalar(mul_1111, 1);  mul_1111 = None
    mul_1112: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(sigmoid_124, add_376);  sigmoid_124 = add_376 = None
    mul_1113: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(add_375, mul_1112);  add_375 = mul_1112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_144: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_1113, [0, 2, 3])
    sub_310: "f32[8, 144, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_798);  convolution_16 = unsqueeze_798 = None
    mul_1114: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1113, sub_310)
    sum_145: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_1114, [0, 2, 3]);  mul_1114 = None
    mul_1115: "f32[144]" = torch.ops.aten.mul.Tensor(sum_144, 0.00021701388888888888)
    unsqueeze_799: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_1115, 0);  mul_1115 = None
    unsqueeze_800: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
    unsqueeze_801: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
    mul_1116: "f32[144]" = torch.ops.aten.mul.Tensor(sum_145, 0.00021701388888888888)
    mul_1117: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1118: "f32[144]" = torch.ops.aten.mul.Tensor(mul_1116, mul_1117);  mul_1116 = mul_1117 = None
    unsqueeze_802: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_1118, 0);  mul_1118 = None
    unsqueeze_803: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
    unsqueeze_804: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
    mul_1119: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_805: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_1119, 0);  mul_1119 = None
    unsqueeze_806: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    mul_1120: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_804);  sub_310 = unsqueeze_804 = None
    sub_312: "f32[8, 144, 24, 24]" = torch.ops.aten.sub.Tensor(mul_1113, mul_1120);  mul_1113 = mul_1120 = None
    sub_313: "f32[8, 144, 24, 24]" = torch.ops.aten.sub.Tensor(sub_312, unsqueeze_801);  sub_312 = unsqueeze_801 = None
    mul_1121: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_807);  sub_313 = unsqueeze_807 = None
    mul_1122: "f32[144]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_31);  sum_145 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1121, mul_82, primals_139, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_1121 = mul_82 = primals_139 = None
    getitem_353: "f32[8, 144, 48, 48]" = convolution_backward_79[0]
    getitem_354: "f32[144, 1, 5, 5]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_48: "f32[8, 144, 48, 48]" = torch.ops.aten.full.default([8, 144, 48, 48], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    mul_1125: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(getitem_353, mul_1124);  getitem_353 = mul_1124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_146: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_1125, [0, 2, 3])
    sub_315: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_810);  convolution_15 = unsqueeze_810 = None
    mul_1126: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1125, sub_315)
    sum_147: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_1126, [0, 2, 3]);  mul_1126 = None
    mul_1127: "f32[144]" = torch.ops.aten.mul.Tensor(sum_146, 5.425347222222222e-05)
    unsqueeze_811: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_1127, 0);  mul_1127 = None
    unsqueeze_812: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
    unsqueeze_813: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
    mul_1128: "f32[144]" = torch.ops.aten.mul.Tensor(sum_147, 5.425347222222222e-05)
    mul_1129: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1130: "f32[144]" = torch.ops.aten.mul.Tensor(mul_1128, mul_1129);  mul_1128 = mul_1129 = None
    unsqueeze_814: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_1130, 0);  mul_1130 = None
    unsqueeze_815: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
    unsqueeze_816: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
    mul_1131: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_817: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_1131, 0);  mul_1131 = None
    unsqueeze_818: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    mul_1132: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_816);  sub_315 = unsqueeze_816 = None
    sub_317: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1125, mul_1132);  mul_1125 = mul_1132 = None
    sub_318: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(sub_317, unsqueeze_813);  sub_317 = unsqueeze_813 = None
    mul_1133: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_819);  sub_318 = unsqueeze_819 = None
    mul_1134: "f32[144]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_28);  sum_147 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1133, add_45, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1133 = add_45 = primals_138 = None
    getitem_356: "f32[8, 24, 48, 48]" = convolution_backward_80[0]
    getitem_357: "f32[144, 24, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_148: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_356, [0, 2, 3])
    sub_319: "f32[8, 24, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_822);  convolution_14 = unsqueeze_822 = None
    mul_1135: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(getitem_356, sub_319)
    sum_149: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1135, [0, 2, 3]);  mul_1135 = None
    mul_1136: "f32[24]" = torch.ops.aten.mul.Tensor(sum_148, 5.425347222222222e-05)
    unsqueeze_823: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1136, 0);  mul_1136 = None
    unsqueeze_824: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
    unsqueeze_825: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
    mul_1137: "f32[24]" = torch.ops.aten.mul.Tensor(sum_149, 5.425347222222222e-05)
    mul_1138: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1139: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1137, mul_1138);  mul_1137 = mul_1138 = None
    unsqueeze_826: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1139, 0);  mul_1139 = None
    unsqueeze_827: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
    unsqueeze_828: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
    mul_1140: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_829: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1140, 0);  mul_1140 = None
    unsqueeze_830: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    mul_1141: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_828);  sub_319 = unsqueeze_828 = None
    sub_321: "f32[8, 24, 48, 48]" = torch.ops.aten.sub.Tensor(getitem_356, mul_1141);  mul_1141 = None
    sub_322: "f32[8, 24, 48, 48]" = torch.ops.aten.sub.Tensor(sub_321, unsqueeze_825);  sub_321 = unsqueeze_825 = None
    mul_1142: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_831);  sub_322 = unsqueeze_831 = None
    mul_1143: "f32[24]" = torch.ops.aten.mul.Tensor(sum_149, squeeze_25);  sum_149 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1142, mul_67, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1142 = mul_67 = primals_137 = None
    getitem_359: "f32[8, 144, 48, 48]" = convolution_backward_81[0]
    getitem_360: "f32[24, 144, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1144: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(getitem_359, mul_65);  mul_65 = None
    mul_1145: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(getitem_359, sigmoid_11);  getitem_359 = sigmoid_11 = None
    sum_150: "f32[8, 144, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1144, [2, 3], True);  mul_1144 = None
    alias_35: "f32[8, 144, 1, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    sub_323: "f32[8, 144, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_35)
    mul_1146: "f32[8, 144, 1, 1]" = torch.ops.aten.mul.Tensor(alias_35, sub_323);  alias_35 = sub_323 = None
    mul_1147: "f32[8, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sum_150, mul_1146);  sum_150 = mul_1146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_151: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_1147, [0, 2, 3])
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1147, mul_66, primals_135, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1147 = mul_66 = primals_135 = None
    getitem_362: "f32[8, 6, 1, 1]" = convolution_backward_82[0]
    getitem_363: "f32[144, 6, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_126: "f32[8, 6, 1, 1]" = torch.ops.aten.sigmoid.default(clone_8)
    sub_324: "f32[8, 6, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_46, sigmoid_126);  full_default_46 = None
    mul_1148: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(clone_8, sub_324);  clone_8 = sub_324 = None
    add_378: "f32[8, 6, 1, 1]" = torch.ops.aten.add.Scalar(mul_1148, 1);  mul_1148 = None
    mul_1149: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_126, add_378);  sigmoid_126 = add_378 = None
    mul_1150: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_362, mul_1149);  getitem_362 = mul_1149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_152: "f32[6]" = torch.ops.aten.sum.dim_IntList(mul_1150, [0, 2, 3])
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1150, mean_2, primals_133, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1150 = mean_2 = primals_133 = None
    getitem_365: "f32[8, 144, 1, 1]" = convolution_backward_83[0]
    getitem_366: "f32[6, 144, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_17: "f32[8, 144, 48, 48]" = torch.ops.aten.expand.default(getitem_365, [8, 144, 48, 48]);  getitem_365 = None
    div_17: "f32[8, 144, 48, 48]" = torch.ops.aten.div.Scalar(expand_17, 2304);  expand_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_379: "f32[8, 144, 48, 48]" = torch.ops.aten.add.Tensor(mul_1145, div_17);  mul_1145 = div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_127: "f32[8, 144, 48, 48]" = torch.ops.aten.sigmoid.default(clone_7)
    sub_325: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(full_default_48, sigmoid_127);  full_default_48 = None
    mul_1151: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(clone_7, sub_325);  clone_7 = sub_325 = None
    add_380: "f32[8, 144, 48, 48]" = torch.ops.aten.add.Scalar(mul_1151, 1);  mul_1151 = None
    mul_1152: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sigmoid_127, add_380);  sigmoid_127 = add_380 = None
    mul_1153: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(add_379, mul_1152);  add_379 = mul_1152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_153: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_1153, [0, 2, 3])
    sub_326: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_834);  convolution_11 = unsqueeze_834 = None
    mul_1154: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1153, sub_326)
    sum_154: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_1154, [0, 2, 3]);  mul_1154 = None
    mul_1155: "f32[144]" = torch.ops.aten.mul.Tensor(sum_153, 5.425347222222222e-05)
    unsqueeze_835: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_1155, 0);  mul_1155 = None
    unsqueeze_836: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
    unsqueeze_837: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
    mul_1156: "f32[144]" = torch.ops.aten.mul.Tensor(sum_154, 5.425347222222222e-05)
    mul_1157: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1158: "f32[144]" = torch.ops.aten.mul.Tensor(mul_1156, mul_1157);  mul_1156 = mul_1157 = None
    unsqueeze_838: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_1158, 0);  mul_1158 = None
    unsqueeze_839: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
    unsqueeze_840: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
    mul_1159: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_841: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_1159, 0);  mul_1159 = None
    unsqueeze_842: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
    unsqueeze_843: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
    mul_1160: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_840);  sub_326 = unsqueeze_840 = None
    sub_328: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1153, mul_1160);  mul_1153 = mul_1160 = None
    sub_329: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(sub_328, unsqueeze_837);  sub_328 = unsqueeze_837 = None
    mul_1161: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_843);  sub_329 = unsqueeze_843 = None
    mul_1162: "f32[144]" = torch.ops.aten.mul.Tensor(sum_154, squeeze_22);  sum_154 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1161, mul_57, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_1161 = mul_57 = primals_132 = None
    getitem_368: "f32[8, 144, 48, 48]" = convolution_backward_84[0]
    getitem_369: "f32[144, 1, 3, 3]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1165: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(getitem_368, mul_1164);  getitem_368 = mul_1164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_155: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_1165, [0, 2, 3])
    sub_331: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_846);  convolution_10 = unsqueeze_846 = None
    mul_1166: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1165, sub_331)
    sum_156: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_1166, [0, 2, 3]);  mul_1166 = None
    mul_1167: "f32[144]" = torch.ops.aten.mul.Tensor(sum_155, 5.425347222222222e-05)
    unsqueeze_847: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_1167, 0);  mul_1167 = None
    unsqueeze_848: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
    unsqueeze_849: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
    mul_1168: "f32[144]" = torch.ops.aten.mul.Tensor(sum_156, 5.425347222222222e-05)
    mul_1169: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1170: "f32[144]" = torch.ops.aten.mul.Tensor(mul_1168, mul_1169);  mul_1168 = mul_1169 = None
    unsqueeze_850: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_1170, 0);  mul_1170 = None
    unsqueeze_851: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
    unsqueeze_852: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
    mul_1171: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_853: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_1171, 0);  mul_1171 = None
    unsqueeze_854: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    mul_1172: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_852);  sub_331 = unsqueeze_852 = None
    sub_333: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1165, mul_1172);  mul_1165 = mul_1172 = None
    sub_334: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(sub_333, unsqueeze_849);  sub_333 = unsqueeze_849 = None
    mul_1173: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_855);  sub_334 = unsqueeze_855 = None
    mul_1174: "f32[144]" = torch.ops.aten.mul.Tensor(sum_156, squeeze_19);  sum_156 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1173, add_29, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1173 = add_29 = primals_131 = None
    getitem_371: "f32[8, 24, 48, 48]" = convolution_backward_85[0]
    getitem_372: "f32[144, 24, 1, 1]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_382: "f32[8, 24, 48, 48]" = torch.ops.aten.add.Tensor(getitem_356, getitem_371);  getitem_356 = getitem_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_157: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_382, [0, 2, 3])
    sub_335: "f32[8, 24, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_858);  convolution_9 = unsqueeze_858 = None
    mul_1175: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(add_382, sub_335)
    sum_158: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1175, [0, 2, 3]);  mul_1175 = None
    mul_1176: "f32[24]" = torch.ops.aten.mul.Tensor(sum_157, 5.425347222222222e-05)
    unsqueeze_859: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1176, 0);  mul_1176 = None
    unsqueeze_860: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
    unsqueeze_861: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
    mul_1177: "f32[24]" = torch.ops.aten.mul.Tensor(sum_158, 5.425347222222222e-05)
    mul_1178: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1179: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1177, mul_1178);  mul_1177 = mul_1178 = None
    unsqueeze_862: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1179, 0);  mul_1179 = None
    unsqueeze_863: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
    unsqueeze_864: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
    mul_1180: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_865: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1180, 0);  mul_1180 = None
    unsqueeze_866: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    mul_1181: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_864);  sub_335 = unsqueeze_864 = None
    sub_337: "f32[8, 24, 48, 48]" = torch.ops.aten.sub.Tensor(add_382, mul_1181);  add_382 = mul_1181 = None
    sub_338: "f32[8, 24, 48, 48]" = torch.ops.aten.sub.Tensor(sub_337, unsqueeze_861);  sub_337 = unsqueeze_861 = None
    mul_1182: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_867);  sub_338 = unsqueeze_867 = None
    mul_1183: "f32[24]" = torch.ops.aten.mul.Tensor(sum_158, squeeze_16);  sum_158 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1182, mul_42, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1182 = mul_42 = primals_130 = None
    getitem_374: "f32[8, 96, 48, 48]" = convolution_backward_86[0]
    getitem_375: "f32[24, 96, 1, 1]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1184: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(getitem_374, mul_40);  mul_40 = None
    mul_1185: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(getitem_374, sigmoid_7);  getitem_374 = sigmoid_7 = None
    sum_159: "f32[8, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1184, [2, 3], True);  mul_1184 = None
    alias_36: "f32[8, 96, 1, 1]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    sub_339: "f32[8, 96, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_36)
    mul_1186: "f32[8, 96, 1, 1]" = torch.ops.aten.mul.Tensor(alias_36, sub_339);  alias_36 = sub_339 = None
    mul_1187: "f32[8, 96, 1, 1]" = torch.ops.aten.mul.Tensor(sum_159, mul_1186);  sum_159 = mul_1186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_160: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1187, [0, 2, 3])
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1187, mul_41, primals_128, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1187 = mul_41 = primals_128 = None
    getitem_377: "f32[8, 4, 1, 1]" = convolution_backward_87[0]
    getitem_378: "f32[96, 4, 1, 1]" = convolution_backward_87[1];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_129: "f32[8, 4, 1, 1]" = torch.ops.aten.sigmoid.default(clone_5)
    full_default_52: "f32[8, 4, 1, 1]" = torch.ops.aten.full.default([8, 4, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_340: "f32[8, 4, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_52, sigmoid_129);  full_default_52 = None
    mul_1188: "f32[8, 4, 1, 1]" = torch.ops.aten.mul.Tensor(clone_5, sub_340);  clone_5 = sub_340 = None
    add_383: "f32[8, 4, 1, 1]" = torch.ops.aten.add.Scalar(mul_1188, 1);  mul_1188 = None
    mul_1189: "f32[8, 4, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_129, add_383);  sigmoid_129 = add_383 = None
    mul_1190: "f32[8, 4, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_377, mul_1189);  getitem_377 = mul_1189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_161: "f32[4]" = torch.ops.aten.sum.dim_IntList(mul_1190, [0, 2, 3])
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1190, mean_1, primals_126, [4], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1190 = mean_1 = primals_126 = None
    getitem_380: "f32[8, 96, 1, 1]" = convolution_backward_88[0]
    getitem_381: "f32[4, 96, 1, 1]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_18: "f32[8, 96, 48, 48]" = torch.ops.aten.expand.default(getitem_380, [8, 96, 48, 48]);  getitem_380 = None
    div_18: "f32[8, 96, 48, 48]" = torch.ops.aten.div.Scalar(expand_18, 2304);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_384: "f32[8, 96, 48, 48]" = torch.ops.aten.add.Tensor(mul_1185, div_18);  mul_1185 = div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_130: "f32[8, 96, 48, 48]" = torch.ops.aten.sigmoid.default(clone_4)
    full_default_53: "f32[8, 96, 48, 48]" = torch.ops.aten.full.default([8, 96, 48, 48], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_341: "f32[8, 96, 48, 48]" = torch.ops.aten.sub.Tensor(full_default_53, sigmoid_130);  full_default_53 = None
    mul_1191: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(clone_4, sub_341);  clone_4 = sub_341 = None
    add_385: "f32[8, 96, 48, 48]" = torch.ops.aten.add.Scalar(mul_1191, 1);  mul_1191 = None
    mul_1192: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(sigmoid_130, add_385);  sigmoid_130 = add_385 = None
    mul_1193: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(add_384, mul_1192);  add_384 = mul_1192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_162: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1193, [0, 2, 3])
    sub_342: "f32[8, 96, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_870);  convolution_6 = unsqueeze_870 = None
    mul_1194: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1193, sub_342)
    sum_163: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1194, [0, 2, 3]);  mul_1194 = None
    mul_1195: "f32[96]" = torch.ops.aten.mul.Tensor(sum_162, 5.425347222222222e-05)
    unsqueeze_871: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1195, 0);  mul_1195 = None
    unsqueeze_872: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
    unsqueeze_873: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
    mul_1196: "f32[96]" = torch.ops.aten.mul.Tensor(sum_163, 5.425347222222222e-05)
    mul_1197: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1198: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1196, mul_1197);  mul_1196 = mul_1197 = None
    unsqueeze_874: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1198, 0);  mul_1198 = None
    unsqueeze_875: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
    unsqueeze_876: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
    mul_1199: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_877: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1199, 0);  mul_1199 = None
    unsqueeze_878: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    mul_1200: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_876);  sub_342 = unsqueeze_876 = None
    sub_344: "f32[8, 96, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1193, mul_1200);  mul_1193 = mul_1200 = None
    sub_345: "f32[8, 96, 48, 48]" = torch.ops.aten.sub.Tensor(sub_344, unsqueeze_873);  sub_344 = unsqueeze_873 = None
    mul_1201: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_879);  sub_345 = unsqueeze_879 = None
    mul_1202: "f32[96]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_13);  sum_163 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1201, mul_32, primals_125, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False]);  mul_1201 = mul_32 = primals_125 = None
    getitem_383: "f32[8, 96, 96, 96]" = convolution_backward_89[0]
    getitem_384: "f32[96, 1, 3, 3]" = convolution_backward_89[1];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1205: "f32[8, 96, 96, 96]" = torch.ops.aten.mul.Tensor(getitem_383, mul_1204);  getitem_383 = mul_1204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_164: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1205, [0, 2, 3])
    sub_347: "f32[8, 96, 96, 96]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_882);  convolution_5 = unsqueeze_882 = None
    mul_1206: "f32[8, 96, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1205, sub_347)
    sum_165: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1206, [0, 2, 3]);  mul_1206 = None
    mul_1207: "f32[96]" = torch.ops.aten.mul.Tensor(sum_164, 1.3563368055555555e-05)
    unsqueeze_883: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1207, 0);  mul_1207 = None
    unsqueeze_884: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
    unsqueeze_885: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
    mul_1208: "f32[96]" = torch.ops.aten.mul.Tensor(sum_165, 1.3563368055555555e-05)
    mul_1209: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1210: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1208, mul_1209);  mul_1208 = mul_1209 = None
    unsqueeze_886: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1210, 0);  mul_1210 = None
    unsqueeze_887: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
    unsqueeze_888: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
    mul_1211: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_889: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1211, 0);  mul_1211 = None
    unsqueeze_890: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
    unsqueeze_891: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
    mul_1212: "f32[8, 96, 96, 96]" = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_888);  sub_347 = unsqueeze_888 = None
    sub_349: "f32[8, 96, 96, 96]" = torch.ops.aten.sub.Tensor(mul_1205, mul_1212);  mul_1205 = mul_1212 = None
    sub_350: "f32[8, 96, 96, 96]" = torch.ops.aten.sub.Tensor(sub_349, unsqueeze_885);  sub_349 = unsqueeze_885 = None
    mul_1213: "f32[8, 96, 96, 96]" = torch.ops.aten.mul.Tensor(sub_350, unsqueeze_891);  sub_350 = unsqueeze_891 = None
    mul_1214: "f32[96]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_10);  sum_165 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1213, add_14, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1213 = add_14 = primals_124 = None
    getitem_386: "f32[8, 16, 96, 96]" = convolution_backward_90[0]
    getitem_387: "f32[96, 16, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_166: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_386, [0, 2, 3])
    sub_351: "f32[8, 16, 96, 96]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_894);  convolution_4 = unsqueeze_894 = None
    mul_1215: "f32[8, 16, 96, 96]" = torch.ops.aten.mul.Tensor(getitem_386, sub_351)
    sum_167: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1215, [0, 2, 3]);  mul_1215 = None
    mul_1216: "f32[16]" = torch.ops.aten.mul.Tensor(sum_166, 1.3563368055555555e-05)
    unsqueeze_895: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1216, 0);  mul_1216 = None
    unsqueeze_896: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
    unsqueeze_897: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
    mul_1217: "f32[16]" = torch.ops.aten.mul.Tensor(sum_167, 1.3563368055555555e-05)
    mul_1218: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1219: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1217, mul_1218);  mul_1217 = mul_1218 = None
    unsqueeze_898: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1219, 0);  mul_1219 = None
    unsqueeze_899: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 2);  unsqueeze_898 = None
    unsqueeze_900: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 3);  unsqueeze_899 = None
    mul_1220: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_901: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1220, 0);  mul_1220 = None
    unsqueeze_902: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
    unsqueeze_903: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
    mul_1221: "f32[8, 16, 96, 96]" = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_900);  sub_351 = unsqueeze_900 = None
    sub_353: "f32[8, 16, 96, 96]" = torch.ops.aten.sub.Tensor(getitem_386, mul_1221);  getitem_386 = mul_1221 = None
    sub_354: "f32[8, 16, 96, 96]" = torch.ops.aten.sub.Tensor(sub_353, unsqueeze_897);  sub_353 = unsqueeze_897 = None
    mul_1222: "f32[8, 16, 96, 96]" = torch.ops.aten.mul.Tensor(sub_354, unsqueeze_903);  sub_354 = unsqueeze_903 = None
    mul_1223: "f32[16]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_7);  sum_167 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1222, mul_17, primals_123, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1222 = mul_17 = primals_123 = None
    getitem_389: "f32[8, 32, 96, 96]" = convolution_backward_91[0]
    getitem_390: "f32[16, 32, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1224: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(getitem_389, mul_15);  mul_15 = None
    mul_1225: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(getitem_389, sigmoid_3);  getitem_389 = sigmoid_3 = None
    sum_168: "f32[8, 32, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1224, [2, 3], True);  mul_1224 = None
    alias_37: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(alias);  alias = None
    sub_355: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_37)
    mul_1226: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(alias_37, sub_355);  alias_37 = sub_355 = None
    mul_1227: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sum_168, mul_1226);  sum_168 = mul_1226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_169: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1227, [0, 2, 3])
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1227, mul_16, primals_121, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1227 = mul_16 = primals_121 = None
    getitem_392: "f32[8, 8, 1, 1]" = convolution_backward_92[0]
    getitem_393: "f32[32, 8, 1, 1]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_132: "f32[8, 8, 1, 1]" = torch.ops.aten.sigmoid.default(clone_2)
    full_default_55: "f32[8, 8, 1, 1]" = torch.ops.aten.full.default([8, 8, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_356: "f32[8, 8, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_55, sigmoid_132);  full_default_55 = None
    mul_1228: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(clone_2, sub_356);  clone_2 = sub_356 = None
    add_387: "f32[8, 8, 1, 1]" = torch.ops.aten.add.Scalar(mul_1228, 1);  mul_1228 = None
    mul_1229: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_132, add_387);  sigmoid_132 = add_387 = None
    mul_1230: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_392, mul_1229);  getitem_392 = mul_1229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_170: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_1230, [0, 2, 3])
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1230, mean, primals_119, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1230 = mean = primals_119 = None
    getitem_395: "f32[8, 32, 1, 1]" = convolution_backward_93[0]
    getitem_396: "f32[8, 32, 1, 1]" = convolution_backward_93[1];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_19: "f32[8, 32, 96, 96]" = torch.ops.aten.expand.default(getitem_395, [8, 32, 96, 96]);  getitem_395 = None
    div_19: "f32[8, 32, 96, 96]" = torch.ops.aten.div.Scalar(expand_19, 9216);  expand_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_388: "f32[8, 32, 96, 96]" = torch.ops.aten.add.Tensor(mul_1225, div_19);  mul_1225 = div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_133: "f32[8, 32, 96, 96]" = torch.ops.aten.sigmoid.default(clone_1)
    full_default_56: "f32[8, 32, 96, 96]" = torch.ops.aten.full.default([8, 32, 96, 96], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_357: "f32[8, 32, 96, 96]" = torch.ops.aten.sub.Tensor(full_default_56, sigmoid_133);  full_default_56 = None
    mul_1231: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(clone_1, sub_357);  clone_1 = sub_357 = None
    add_389: "f32[8, 32, 96, 96]" = torch.ops.aten.add.Scalar(mul_1231, 1);  mul_1231 = None
    mul_1232: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(sigmoid_133, add_389);  sigmoid_133 = add_389 = None
    mul_1233: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(add_388, mul_1232);  add_388 = mul_1232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_171: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1233, [0, 2, 3])
    sub_358: "f32[8, 32, 96, 96]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_906);  convolution_1 = unsqueeze_906 = None
    mul_1234: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1233, sub_358)
    sum_172: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1234, [0, 2, 3]);  mul_1234 = None
    mul_1235: "f32[32]" = torch.ops.aten.mul.Tensor(sum_171, 1.3563368055555555e-05)
    unsqueeze_907: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1235, 0);  mul_1235 = None
    unsqueeze_908: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 2);  unsqueeze_907 = None
    unsqueeze_909: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 3);  unsqueeze_908 = None
    mul_1236: "f32[32]" = torch.ops.aten.mul.Tensor(sum_172, 1.3563368055555555e-05)
    mul_1237: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1238: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1236, mul_1237);  mul_1236 = mul_1237 = None
    unsqueeze_910: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1238, 0);  mul_1238 = None
    unsqueeze_911: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 2);  unsqueeze_910 = None
    unsqueeze_912: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 3);  unsqueeze_911 = None
    mul_1239: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_913: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1239, 0);  mul_1239 = None
    unsqueeze_914: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
    unsqueeze_915: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
    mul_1240: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(sub_358, unsqueeze_912);  sub_358 = unsqueeze_912 = None
    sub_360: "f32[8, 32, 96, 96]" = torch.ops.aten.sub.Tensor(mul_1233, mul_1240);  mul_1233 = mul_1240 = None
    sub_361: "f32[8, 32, 96, 96]" = torch.ops.aten.sub.Tensor(sub_360, unsqueeze_909);  sub_360 = unsqueeze_909 = None
    mul_1241: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(sub_361, unsqueeze_915);  sub_361 = unsqueeze_915 = None
    mul_1242: "f32[32]" = torch.ops.aten.mul.Tensor(sum_172, squeeze_4);  sum_172 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1241, mul_7, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1241 = mul_7 = primals_118 = None
    getitem_398: "f32[8, 32, 96, 96]" = convolution_backward_94[0]
    getitem_399: "f32[32, 1, 3, 3]" = convolution_backward_94[1];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1245: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(getitem_398, mul_1244);  getitem_398 = mul_1244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_173: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1245, [0, 2, 3])
    sub_363: "f32[8, 32, 96, 96]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_918);  convolution = unsqueeze_918 = None
    mul_1246: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_1245, sub_363)
    sum_174: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1246, [0, 2, 3]);  mul_1246 = None
    mul_1247: "f32[32]" = torch.ops.aten.mul.Tensor(sum_173, 1.3563368055555555e-05)
    unsqueeze_919: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1247, 0);  mul_1247 = None
    unsqueeze_920: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 2);  unsqueeze_919 = None
    unsqueeze_921: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 3);  unsqueeze_920 = None
    mul_1248: "f32[32]" = torch.ops.aten.mul.Tensor(sum_174, 1.3563368055555555e-05)
    mul_1249: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1250: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1248, mul_1249);  mul_1248 = mul_1249 = None
    unsqueeze_922: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1250, 0);  mul_1250 = None
    unsqueeze_923: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 2);  unsqueeze_922 = None
    unsqueeze_924: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 3);  unsqueeze_923 = None
    mul_1251: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_925: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1251, 0);  mul_1251 = None
    unsqueeze_926: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
    unsqueeze_927: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
    mul_1252: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(sub_363, unsqueeze_924);  sub_363 = unsqueeze_924 = None
    sub_365: "f32[8, 32, 96, 96]" = torch.ops.aten.sub.Tensor(mul_1245, mul_1252);  mul_1245 = mul_1252 = None
    sub_366: "f32[8, 32, 96, 96]" = torch.ops.aten.sub.Tensor(sub_365, unsqueeze_921);  sub_365 = unsqueeze_921 = None
    mul_1253: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(sub_366, unsqueeze_927);  sub_366 = unsqueeze_927 = None
    mul_1254: "f32[32]" = torch.ops.aten.mul.Tensor(sum_174, squeeze_1);  sum_174 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1253, primals_427, primals_117, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1253 = primals_427 = primals_117 = None
    getitem_402: "f32[32, 3, 3, 3]" = convolution_backward_95[1];  convolution_backward_95 = None
    return [mul_1254, sum_173, mul_1242, sum_171, mul_1223, sum_166, mul_1214, sum_164, mul_1202, sum_162, mul_1183, sum_157, mul_1174, sum_155, mul_1162, sum_153, mul_1143, sum_148, mul_1134, sum_146, mul_1122, sum_144, mul_1103, sum_139, mul_1094, sum_137, mul_1082, sum_135, mul_1063, sum_130, mul_1054, sum_128, mul_1042, sum_126, mul_1023, sum_121, mul_1014, sum_119, mul_1002, sum_117, mul_983, sum_112, mul_974, sum_110, mul_962, sum_108, mul_943, sum_103, mul_934, sum_101, mul_922, sum_99, mul_903, sum_94, mul_894, sum_92, mul_882, sum_90, mul_863, sum_85, mul_854, sum_83, mul_842, sum_81, mul_823, sum_76, mul_814, sum_74, mul_802, sum_72, mul_783, sum_67, mul_774, sum_65, mul_762, sum_63, mul_743, sum_58, mul_734, sum_56, mul_722, sum_54, mul_703, sum_49, mul_694, sum_47, mul_682, sum_45, mul_663, sum_40, mul_654, sum_38, mul_642, sum_36, mul_623, sum_31, mul_614, sum_29, mul_602, sum_27, mul_583, sum_22, mul_574, sum_20, mul_562, sum_18, mul_543, sum_13, mul_534, sum_11, mul_522, sum_9, mul_503, sum_4, mul_494, sum_2, getitem_402, getitem_399, getitem_396, sum_170, getitem_393, sum_169, getitem_390, getitem_387, getitem_384, getitem_381, sum_161, getitem_378, sum_160, getitem_375, getitem_372, getitem_369, getitem_366, sum_152, getitem_363, sum_151, getitem_360, getitem_357, getitem_354, getitem_351, sum_143, getitem_348, sum_142, getitem_345, getitem_342, getitem_339, getitem_336, sum_134, getitem_333, sum_133, getitem_330, getitem_327, getitem_324, getitem_321, sum_125, getitem_318, sum_124, getitem_315, getitem_312, getitem_309, getitem_306, sum_116, getitem_303, sum_115, getitem_300, getitem_297, getitem_294, getitem_291, sum_107, getitem_288, sum_106, getitem_285, getitem_282, getitem_279, getitem_276, sum_98, getitem_273, sum_97, getitem_270, getitem_267, getitem_264, getitem_261, sum_89, getitem_258, sum_88, getitem_255, getitem_252, getitem_249, getitem_246, sum_80, getitem_243, sum_79, getitem_240, getitem_237, getitem_234, getitem_231, sum_71, getitem_228, sum_70, getitem_225, getitem_222, getitem_219, getitem_216, sum_62, getitem_213, sum_61, getitem_210, getitem_207, getitem_204, getitem_201, sum_53, getitem_198, sum_52, getitem_195, getitem_192, getitem_189, getitem_186, sum_44, getitem_183, sum_43, getitem_180, getitem_177, getitem_174, getitem_171, sum_35, getitem_168, sum_34, getitem_165, getitem_162, getitem_159, getitem_156, sum_26, getitem_153, sum_25, getitem_150, getitem_147, getitem_144, getitem_141, sum_17, getitem_138, sum_16, getitem_135, getitem_132, getitem_129, getitem_126, sum_8, getitem_123, sum_7, getitem_120, getitem_117, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    