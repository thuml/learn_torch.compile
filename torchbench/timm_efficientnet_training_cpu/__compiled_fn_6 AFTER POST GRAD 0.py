from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_3: "f32[32]", primals_5: "f32[16]", primals_7: "f32[96]", primals_9: "f32[96]", primals_11: "f32[24]", primals_13: "f32[144]", primals_15: "f32[144]", primals_17: "f32[24]", primals_19: "f32[144]", primals_21: "f32[144]", primals_23: "f32[40]", primals_25: "f32[240]", primals_27: "f32[240]", primals_29: "f32[40]", primals_31: "f32[240]", primals_33: "f32[240]", primals_35: "f32[80]", primals_37: "f32[480]", primals_39: "f32[480]", primals_41: "f32[80]", primals_43: "f32[480]", primals_45: "f32[480]", primals_47: "f32[80]", primals_49: "f32[480]", primals_51: "f32[480]", primals_53: "f32[112]", primals_55: "f32[672]", primals_57: "f32[672]", primals_59: "f32[112]", primals_61: "f32[672]", primals_63: "f32[672]", primals_65: "f32[112]", primals_67: "f32[672]", primals_69: "f32[672]", primals_71: "f32[192]", primals_73: "f32[1152]", primals_75: "f32[1152]", primals_77: "f32[192]", primals_79: "f32[1152]", primals_81: "f32[1152]", primals_83: "f32[192]", primals_85: "f32[1152]", primals_87: "f32[1152]", primals_89: "f32[192]", primals_91: "f32[1152]", primals_93: "f32[1152]", primals_95: "f32[320]", primals_97: "f32[1280]", primals_99: "f32[32, 3, 3, 3]", primals_100: "f32[32, 1, 3, 3]", primals_101: "f32[8, 32, 1, 1]", primals_103: "f32[32, 8, 1, 1]", primals_105: "f32[16, 32, 1, 1]", primals_106: "f32[96, 16, 1, 1]", primals_107: "f32[96, 1, 3, 3]", primals_108: "f32[4, 96, 1, 1]", primals_110: "f32[96, 4, 1, 1]", primals_112: "f32[24, 96, 1, 1]", primals_113: "f32[144, 24, 1, 1]", primals_114: "f32[144, 1, 3, 3]", primals_115: "f32[6, 144, 1, 1]", primals_117: "f32[144, 6, 1, 1]", primals_119: "f32[24, 144, 1, 1]", primals_120: "f32[144, 24, 1, 1]", primals_121: "f32[144, 1, 5, 5]", primals_122: "f32[6, 144, 1, 1]", primals_124: "f32[144, 6, 1, 1]", primals_126: "f32[40, 144, 1, 1]", primals_127: "f32[240, 40, 1, 1]", primals_128: "f32[240, 1, 5, 5]", primals_129: "f32[10, 240, 1, 1]", primals_131: "f32[240, 10, 1, 1]", primals_133: "f32[40, 240, 1, 1]", primals_134: "f32[240, 40, 1, 1]", primals_135: "f32[240, 1, 3, 3]", primals_136: "f32[10, 240, 1, 1]", primals_138: "f32[240, 10, 1, 1]", primals_140: "f32[80, 240, 1, 1]", primals_141: "f32[480, 80, 1, 1]", primals_142: "f32[480, 1, 3, 3]", primals_143: "f32[20, 480, 1, 1]", primals_145: "f32[480, 20, 1, 1]", primals_147: "f32[80, 480, 1, 1]", primals_148: "f32[480, 80, 1, 1]", primals_149: "f32[480, 1, 3, 3]", primals_150: "f32[20, 480, 1, 1]", primals_152: "f32[480, 20, 1, 1]", primals_154: "f32[80, 480, 1, 1]", primals_155: "f32[480, 80, 1, 1]", primals_156: "f32[480, 1, 5, 5]", primals_157: "f32[20, 480, 1, 1]", primals_159: "f32[480, 20, 1, 1]", primals_161: "f32[112, 480, 1, 1]", primals_162: "f32[672, 112, 1, 1]", primals_163: "f32[672, 1, 5, 5]", primals_164: "f32[28, 672, 1, 1]", primals_166: "f32[672, 28, 1, 1]", primals_168: "f32[112, 672, 1, 1]", primals_169: "f32[672, 112, 1, 1]", primals_170: "f32[672, 1, 5, 5]", primals_171: "f32[28, 672, 1, 1]", primals_173: "f32[672, 28, 1, 1]", primals_175: "f32[112, 672, 1, 1]", primals_176: "f32[672, 112, 1, 1]", primals_177: "f32[672, 1, 5, 5]", primals_178: "f32[28, 672, 1, 1]", primals_180: "f32[672, 28, 1, 1]", primals_182: "f32[192, 672, 1, 1]", primals_183: "f32[1152, 192, 1, 1]", primals_184: "f32[1152, 1, 5, 5]", primals_185: "f32[48, 1152, 1, 1]", primals_187: "f32[1152, 48, 1, 1]", primals_189: "f32[192, 1152, 1, 1]", primals_190: "f32[1152, 192, 1, 1]", primals_191: "f32[1152, 1, 5, 5]", primals_192: "f32[48, 1152, 1, 1]", primals_194: "f32[1152, 48, 1, 1]", primals_196: "f32[192, 1152, 1, 1]", primals_197: "f32[1152, 192, 1, 1]", primals_198: "f32[1152, 1, 5, 5]", primals_199: "f32[48, 1152, 1, 1]", primals_201: "f32[1152, 48, 1, 1]", primals_203: "f32[192, 1152, 1, 1]", primals_204: "f32[1152, 192, 1, 1]", primals_205: "f32[1152, 1, 3, 3]", primals_206: "f32[48, 1152, 1, 1]", primals_208: "f32[1152, 48, 1, 1]", primals_210: "f32[320, 1152, 1, 1]", primals_211: "f32[1280, 320, 1, 1]", primals_214: "f32[32]", primals_215: "f32[32]", primals_216: "f32[32]", primals_217: "f32[32]", primals_218: "f32[16]", primals_219: "f32[16]", primals_220: "f32[96]", primals_221: "f32[96]", primals_222: "f32[96]", primals_223: "f32[96]", primals_224: "f32[24]", primals_225: "f32[24]", primals_226: "f32[144]", primals_227: "f32[144]", primals_228: "f32[144]", primals_229: "f32[144]", primals_230: "f32[24]", primals_231: "f32[24]", primals_232: "f32[144]", primals_233: "f32[144]", primals_234: "f32[144]", primals_235: "f32[144]", primals_236: "f32[40]", primals_237: "f32[40]", primals_238: "f32[240]", primals_239: "f32[240]", primals_240: "f32[240]", primals_241: "f32[240]", primals_242: "f32[40]", primals_243: "f32[40]", primals_244: "f32[240]", primals_245: "f32[240]", primals_246: "f32[240]", primals_247: "f32[240]", primals_248: "f32[80]", primals_249: "f32[80]", primals_250: "f32[480]", primals_251: "f32[480]", primals_252: "f32[480]", primals_253: "f32[480]", primals_254: "f32[80]", primals_255: "f32[80]", primals_256: "f32[480]", primals_257: "f32[480]", primals_258: "f32[480]", primals_259: "f32[480]", primals_260: "f32[80]", primals_261: "f32[80]", primals_262: "f32[480]", primals_263: "f32[480]", primals_264: "f32[480]", primals_265: "f32[480]", primals_266: "f32[112]", primals_267: "f32[112]", primals_268: "f32[672]", primals_269: "f32[672]", primals_270: "f32[672]", primals_271: "f32[672]", primals_272: "f32[112]", primals_273: "f32[112]", primals_274: "f32[672]", primals_275: "f32[672]", primals_276: "f32[672]", primals_277: "f32[672]", primals_278: "f32[112]", primals_279: "f32[112]", primals_280: "f32[672]", primals_281: "f32[672]", primals_282: "f32[672]", primals_283: "f32[672]", primals_284: "f32[192]", primals_285: "f32[192]", primals_286: "f32[1152]", primals_287: "f32[1152]", primals_288: "f32[1152]", primals_289: "f32[1152]", primals_290: "f32[192]", primals_291: "f32[192]", primals_292: "f32[1152]", primals_293: "f32[1152]", primals_294: "f32[1152]", primals_295: "f32[1152]", primals_296: "f32[192]", primals_297: "f32[192]", primals_298: "f32[1152]", primals_299: "f32[1152]", primals_300: "f32[1152]", primals_301: "f32[1152]", primals_302: "f32[192]", primals_303: "f32[192]", primals_304: "f32[1152]", primals_305: "f32[1152]", primals_306: "f32[1152]", primals_307: "f32[1152]", primals_308: "f32[320]", primals_309: "f32[320]", primals_310: "f32[1280]", primals_311: "f32[1280]", primals_312: "f32[4, 3, 224, 224]", convolution: "f32[4, 32, 112, 112]", mul_3: "f32[4, 32, 112, 112]", convolution_1: "f32[4, 32, 112, 112]", add_3: "f32[4, 32, 112, 112]", mean: "f32[4, 32, 1, 1]", convolution_2: "f32[4, 8, 1, 1]", mul_8: "f32[4, 8, 1, 1]", convolution_3: "f32[4, 32, 1, 1]", mul_9: "f32[4, 32, 112, 112]", convolution_4: "f32[4, 16, 112, 112]", add_5: "f32[4, 16, 112, 112]", convolution_5: "f32[4, 96, 112, 112]", mul_16: "f32[4, 96, 112, 112]", convolution_6: "f32[4, 96, 56, 56]", add_9: "f32[4, 96, 56, 56]", mean_1: "f32[4, 96, 1, 1]", convolution_7: "f32[4, 4, 1, 1]", mul_21: "f32[4, 4, 1, 1]", convolution_8: "f32[4, 96, 1, 1]", mul_22: "f32[4, 96, 56, 56]", convolution_9: "f32[4, 24, 56, 56]", add_11: "f32[4, 24, 56, 56]", convolution_10: "f32[4, 144, 56, 56]", mul_29: "f32[4, 144, 56, 56]", convolution_11: "f32[4, 144, 56, 56]", add_15: "f32[4, 144, 56, 56]", mean_2: "f32[4, 144, 1, 1]", convolution_12: "f32[4, 6, 1, 1]", mul_34: "f32[4, 6, 1, 1]", convolution_13: "f32[4, 144, 1, 1]", mul_35: "f32[4, 144, 56, 56]", convolution_14: "f32[4, 24, 56, 56]", add_18: "f32[4, 24, 56, 56]", convolution_15: "f32[4, 144, 56, 56]", mul_42: "f32[4, 144, 56, 56]", convolution_16: "f32[4, 144, 28, 28]", add_22: "f32[4, 144, 28, 28]", mean_3: "f32[4, 144, 1, 1]", convolution_17: "f32[4, 6, 1, 1]", mul_47: "f32[4, 6, 1, 1]", convolution_18: "f32[4, 144, 1, 1]", mul_48: "f32[4, 144, 28, 28]", convolution_19: "f32[4, 40, 28, 28]", add_24: "f32[4, 40, 28, 28]", convolution_20: "f32[4, 240, 28, 28]", mul_55: "f32[4, 240, 28, 28]", convolution_21: "f32[4, 240, 28, 28]", add_28: "f32[4, 240, 28, 28]", mean_4: "f32[4, 240, 1, 1]", convolution_22: "f32[4, 10, 1, 1]", mul_60: "f32[4, 10, 1, 1]", convolution_23: "f32[4, 240, 1, 1]", mul_61: "f32[4, 240, 28, 28]", convolution_24: "f32[4, 40, 28, 28]", add_31: "f32[4, 40, 28, 28]", convolution_25: "f32[4, 240, 28, 28]", mul_68: "f32[4, 240, 28, 28]", convolution_26: "f32[4, 240, 14, 14]", add_35: "f32[4, 240, 14, 14]", mean_5: "f32[4, 240, 1, 1]", convolution_27: "f32[4, 10, 1, 1]", mul_73: "f32[4, 10, 1, 1]", convolution_28: "f32[4, 240, 1, 1]", mul_74: "f32[4, 240, 14, 14]", convolution_29: "f32[4, 80, 14, 14]", add_37: "f32[4, 80, 14, 14]", convolution_30: "f32[4, 480, 14, 14]", mul_81: "f32[4, 480, 14, 14]", convolution_31: "f32[4, 480, 14, 14]", add_41: "f32[4, 480, 14, 14]", mean_6: "f32[4, 480, 1, 1]", convolution_32: "f32[4, 20, 1, 1]", mul_86: "f32[4, 20, 1, 1]", convolution_33: "f32[4, 480, 1, 1]", mul_87: "f32[4, 480, 14, 14]", convolution_34: "f32[4, 80, 14, 14]", add_44: "f32[4, 80, 14, 14]", convolution_35: "f32[4, 480, 14, 14]", mul_94: "f32[4, 480, 14, 14]", convolution_36: "f32[4, 480, 14, 14]", add_48: "f32[4, 480, 14, 14]", mean_7: "f32[4, 480, 1, 1]", convolution_37: "f32[4, 20, 1, 1]", mul_99: "f32[4, 20, 1, 1]", convolution_38: "f32[4, 480, 1, 1]", mul_100: "f32[4, 480, 14, 14]", convolution_39: "f32[4, 80, 14, 14]", add_51: "f32[4, 80, 14, 14]", convolution_40: "f32[4, 480, 14, 14]", mul_107: "f32[4, 480, 14, 14]", convolution_41: "f32[4, 480, 14, 14]", add_55: "f32[4, 480, 14, 14]", mean_8: "f32[4, 480, 1, 1]", convolution_42: "f32[4, 20, 1, 1]", mul_112: "f32[4, 20, 1, 1]", convolution_43: "f32[4, 480, 1, 1]", mul_113: "f32[4, 480, 14, 14]", convolution_44: "f32[4, 112, 14, 14]", add_57: "f32[4, 112, 14, 14]", convolution_45: "f32[4, 672, 14, 14]", mul_120: "f32[4, 672, 14, 14]", convolution_46: "f32[4, 672, 14, 14]", add_61: "f32[4, 672, 14, 14]", mean_9: "f32[4, 672, 1, 1]", convolution_47: "f32[4, 28, 1, 1]", mul_125: "f32[4, 28, 1, 1]", convolution_48: "f32[4, 672, 1, 1]", mul_126: "f32[4, 672, 14, 14]", convolution_49: "f32[4, 112, 14, 14]", add_64: "f32[4, 112, 14, 14]", convolution_50: "f32[4, 672, 14, 14]", mul_133: "f32[4, 672, 14, 14]", convolution_51: "f32[4, 672, 14, 14]", add_68: "f32[4, 672, 14, 14]", mean_10: "f32[4, 672, 1, 1]", convolution_52: "f32[4, 28, 1, 1]", mul_138: "f32[4, 28, 1, 1]", convolution_53: "f32[4, 672, 1, 1]", mul_139: "f32[4, 672, 14, 14]", convolution_54: "f32[4, 112, 14, 14]", add_71: "f32[4, 112, 14, 14]", convolution_55: "f32[4, 672, 14, 14]", mul_146: "f32[4, 672, 14, 14]", convolution_56: "f32[4, 672, 7, 7]", add_75: "f32[4, 672, 7, 7]", mean_11: "f32[4, 672, 1, 1]", convolution_57: "f32[4, 28, 1, 1]", mul_151: "f32[4, 28, 1, 1]", convolution_58: "f32[4, 672, 1, 1]", mul_152: "f32[4, 672, 7, 7]", convolution_59: "f32[4, 192, 7, 7]", add_77: "f32[4, 192, 7, 7]", convolution_60: "f32[4, 1152, 7, 7]", mul_159: "f32[4, 1152, 7, 7]", convolution_61: "f32[4, 1152, 7, 7]", add_81: "f32[4, 1152, 7, 7]", mean_12: "f32[4, 1152, 1, 1]", convolution_62: "f32[4, 48, 1, 1]", mul_164: "f32[4, 48, 1, 1]", convolution_63: "f32[4, 1152, 1, 1]", mul_165: "f32[4, 1152, 7, 7]", convolution_64: "f32[4, 192, 7, 7]", add_84: "f32[4, 192, 7, 7]", convolution_65: "f32[4, 1152, 7, 7]", mul_172: "f32[4, 1152, 7, 7]", convolution_66: "f32[4, 1152, 7, 7]", add_88: "f32[4, 1152, 7, 7]", mean_13: "f32[4, 1152, 1, 1]", convolution_67: "f32[4, 48, 1, 1]", mul_177: "f32[4, 48, 1, 1]", convolution_68: "f32[4, 1152, 1, 1]", mul_178: "f32[4, 1152, 7, 7]", convolution_69: "f32[4, 192, 7, 7]", add_91: "f32[4, 192, 7, 7]", convolution_70: "f32[4, 1152, 7, 7]", mul_185: "f32[4, 1152, 7, 7]", convolution_71: "f32[4, 1152, 7, 7]", add_95: "f32[4, 1152, 7, 7]", mean_14: "f32[4, 1152, 1, 1]", convolution_72: "f32[4, 48, 1, 1]", mul_190: "f32[4, 48, 1, 1]", convolution_73: "f32[4, 1152, 1, 1]", mul_191: "f32[4, 1152, 7, 7]", convolution_74: "f32[4, 192, 7, 7]", add_98: "f32[4, 192, 7, 7]", convolution_75: "f32[4, 1152, 7, 7]", mul_198: "f32[4, 1152, 7, 7]", convolution_76: "f32[4, 1152, 7, 7]", add_102: "f32[4, 1152, 7, 7]", mean_15: "f32[4, 1152, 1, 1]", convolution_77: "f32[4, 48, 1, 1]", mul_203: "f32[4, 48, 1, 1]", convolution_78: "f32[4, 1152, 1, 1]", mul_204: "f32[4, 1152, 7, 7]", convolution_79: "f32[4, 320, 7, 7]", add_104: "f32[4, 320, 7, 7]", convolution_80: "f32[4, 1280, 7, 7]", view: "f32[4, 1280]", permute_1: "f32[1000, 1280]", mul_213: "f32[4, 1280, 7, 7]", mul_250: "f32[4, 1152, 7, 7]", mul_287: "f32[4, 1152, 7, 7]", mul_324: "f32[4, 1152, 7, 7]", mul_361: "f32[4, 1152, 7, 7]", mul_398: "f32[4, 672, 14, 14]", mul_435: "f32[4, 672, 14, 14]", mul_472: "f32[4, 672, 14, 14]", mul_509: "f32[4, 480, 14, 14]", mul_546: "f32[4, 480, 14, 14]", mul_583: "f32[4, 480, 14, 14]", mul_620: "f32[4, 240, 28, 28]", mul_657: "f32[4, 240, 28, 28]", mul_694: "f32[4, 144, 56, 56]", mul_731: "f32[4, 144, 56, 56]", mul_768: "f32[4, 96, 112, 112]", mul_805: "f32[4, 32, 112, 112]", tangents_1: "f32[4, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_1: "f32[4, 32, 112, 112]" = torch.ops.aten.sigmoid.default(add_3)
    mul_7: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_3, sigmoid_1);  sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_3: "f32[4, 32, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_3);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_5: "f32[4, 96, 56, 56]" = torch.ops.aten.sigmoid.default(add_9)
    mul_20: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_5);  sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_7: "f32[4, 96, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_8);  convolution_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_9: "f32[4, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_15)
    mul_33: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_15, sigmoid_9);  sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_11: "f32[4, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_13);  convolution_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_13: "f32[4, 144, 28, 28]" = torch.ops.aten.sigmoid.default(add_22)
    mul_46: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(add_22, sigmoid_13);  sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_15: "f32[4, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_18);  convolution_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_17: "f32[4, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_28)
    mul_59: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_28, sigmoid_17);  sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_19: "f32[4, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_21: "f32[4, 240, 14, 14]" = torch.ops.aten.sigmoid.default(add_35)
    mul_72: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_35, sigmoid_21);  sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_23: "f32[4, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_28);  convolution_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_25: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_41)
    mul_85: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_41, sigmoid_25);  sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_27: "f32[4, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_33);  convolution_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_29: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_48)
    mul_98: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_48, sigmoid_29);  sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_31: "f32[4, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_38);  convolution_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_33: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_55)
    mul_111: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_55, sigmoid_33);  sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_35: "f32[4, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43);  convolution_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_37: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_61)
    mul_124: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_61, sigmoid_37);  sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_39: "f32[4, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_41: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_68)
    mul_137: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_68, sigmoid_41);  sigmoid_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_43: "f32[4, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_53);  convolution_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_45: "f32[4, 672, 7, 7]" = torch.ops.aten.sigmoid.default(add_75)
    mul_150: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_75, sigmoid_45);  sigmoid_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_47: "f32[4, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_58);  convolution_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_49: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_81)
    mul_163: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_81, sigmoid_49);  sigmoid_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_51: "f32[4, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_63);  convolution_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_53: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_88)
    mul_176: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_88, sigmoid_53);  sigmoid_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_55: "f32[4, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_68);  convolution_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_57: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_95)
    mul_189: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_95, sigmoid_57);  sigmoid_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_59: "f32[4, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_61: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_102)
    mul_202: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_102, sigmoid_61);  sigmoid_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_63: "f32[4, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_78);  convolution_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    mm: "f32[4, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[4, 1280, 1, 1]" = torch.ops.aten.reshape.default(mm, [4, 1280, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[4, 1280, 7, 7]" = torch.ops.aten.expand.default(view_2, [4, 1280, 7, 7]);  view_2 = None
    div: "f32[4, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_214: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(div, mul_213);  div = mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_108: "f32[1280]" = torch.ops.aten.add.Tensor(primals_311, 1e-05);  primals_311 = None
    rsqrt: "f32[1280]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    unsqueeze_392: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_310, 0);  primals_310 = None
    unsqueeze_393: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 2, 3])
    sub_50: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_394);  convolution_80 = unsqueeze_394 = None
    mul_215: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_214, sub_50);  sub_50 = None
    sum_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_215, [0, 2, 3]);  mul_215 = None
    mul_220: "f32[1280]" = torch.ops.aten.mul.Tensor(rsqrt, primals_97);  primals_97 = None
    unsqueeze_401: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_220, 0);  mul_220 = None
    unsqueeze_402: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    mul_221: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_403);  mul_214 = unsqueeze_403 = None
    mul_222: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, rsqrt);  sum_3 = rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_221, add_104, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_221 = add_104 = primals_211 = None
    getitem: "f32[4, 320, 7, 7]" = convolution_backward[0]
    getitem_1: "f32[1280, 320, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_109: "f32[320]" = torch.ops.aten.add.Tensor(primals_309, 1e-05);  primals_309 = None
    rsqrt_1: "f32[320]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    unsqueeze_404: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_308, 0);  primals_308 = None
    unsqueeze_405: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    sum_4: "f32[320]" = torch.ops.aten.sum.dim_IntList(getitem, [0, 2, 3])
    sub_51: "f32[4, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_406);  convolution_79 = unsqueeze_406 = None
    mul_223: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem, sub_51);  sub_51 = None
    sum_5: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_223, [0, 2, 3]);  mul_223 = None
    mul_228: "f32[320]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_95);  primals_95 = None
    unsqueeze_413: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_228, 0);  mul_228 = None
    unsqueeze_414: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    mul_229: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_415);  getitem = unsqueeze_415 = None
    mul_230: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, rsqrt_1);  sum_5 = rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_229, mul_204, primals_210, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_229 = mul_204 = primals_210 = None
    getitem_3: "f32[4, 1152, 7, 7]" = convolution_backward_1[0]
    getitem_4: "f32[320, 1152, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_231: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_3, mul_202);  mul_202 = None
    mul_232: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_3, sigmoid_63);  getitem_3 = None
    sum_6: "f32[4, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2, 3], True);  mul_231 = None
    sub_52: "f32[4, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_63)
    mul_233: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_63, sub_52);  sigmoid_63 = sub_52 = None
    mul_234: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, mul_233);  sum_6 = mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_234, mul_203, primals_208, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_234 = mul_203 = primals_208 = None
    getitem_6: "f32[4, 48, 1, 1]" = convolution_backward_2[0]
    getitem_7: "f32[1152, 48, 1, 1]" = convolution_backward_2[1]
    getitem_8: "f32[1152]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_66: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_77)
    full_default_1: "f32[4, 48, 1, 1]" = torch.ops.aten.full.default([4, 48, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_53: "f32[4, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_66)
    mul_235: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_77, sub_53);  convolution_77 = sub_53 = None
    add_110: "f32[4, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_235, 1);  mul_235 = None
    mul_236: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_66, add_110);  sigmoid_66 = add_110 = None
    mul_237: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_6, mul_236);  getitem_6 = mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_237, mean_15, primals_206, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_237 = mean_15 = primals_206 = None
    getitem_9: "f32[4, 1152, 1, 1]" = convolution_backward_3[0]
    getitem_10: "f32[48, 1152, 1, 1]" = convolution_backward_3[1]
    getitem_11: "f32[48]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[4, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_9, [4, 1152, 7, 7]);  getitem_9 = None
    div_1: "f32[4, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_111: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_232, div_1);  mul_232 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_67: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_102)
    full_default_2: "f32[4, 1152, 7, 7]" = torch.ops.aten.full.default([4, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_54: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_67)
    mul_238: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_102, sub_54);  add_102 = sub_54 = None
    add_112: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_238, 1);  mul_238 = None
    mul_239: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_67, add_112);  sigmoid_67 = add_112 = None
    mul_240: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_111, mul_239);  add_111 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_113: "f32[1152]" = torch.ops.aten.add.Tensor(primals_307, 1e-05);  primals_307 = None
    rsqrt_2: "f32[1152]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    unsqueeze_416: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_306, 0);  primals_306 = None
    unsqueeze_417: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    sum_7: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_240, [0, 2, 3])
    sub_55: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_418);  convolution_76 = unsqueeze_418 = None
    mul_241: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_240, sub_55);  sub_55 = None
    sum_8: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 2, 3]);  mul_241 = None
    mul_246: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_93);  primals_93 = None
    unsqueeze_425: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_246, 0);  mul_246 = None
    unsqueeze_426: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_247: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_240, unsqueeze_427);  mul_240 = unsqueeze_427 = None
    mul_248: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_8, rsqrt_2);  sum_8 = rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_247, mul_198, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_247 = mul_198 = primals_205 = None
    getitem_12: "f32[4, 1152, 7, 7]" = convolution_backward_4[0]
    getitem_13: "f32[1152, 1, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_251: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_12, mul_250);  getitem_12 = mul_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_115: "f32[1152]" = torch.ops.aten.add.Tensor(primals_305, 1e-05);  primals_305 = None
    rsqrt_3: "f32[1152]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    unsqueeze_428: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_304, 0);  primals_304 = None
    unsqueeze_429: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    sum_9: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_251, [0, 2, 3])
    sub_57: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_430);  convolution_75 = unsqueeze_430 = None
    mul_252: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_251, sub_57);  sub_57 = None
    sum_10: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 2, 3]);  mul_252 = None
    mul_257: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_91);  primals_91 = None
    unsqueeze_437: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_257, 0);  mul_257 = None
    unsqueeze_438: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_258: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_439);  mul_251 = unsqueeze_439 = None
    mul_259: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_10, rsqrt_3);  sum_10 = rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_258, add_98, primals_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_258 = add_98 = primals_204 = None
    getitem_15: "f32[4, 192, 7, 7]" = convolution_backward_5[0]
    getitem_16: "f32[1152, 192, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_116: "f32[192]" = torch.ops.aten.add.Tensor(primals_303, 1e-05);  primals_303 = None
    rsqrt_4: "f32[192]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    unsqueeze_440: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_302, 0);  primals_302 = None
    unsqueeze_441: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    sum_11: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_15, [0, 2, 3])
    sub_58: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_442);  convolution_74 = unsqueeze_442 = None
    mul_260: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_15, sub_58);  sub_58 = None
    sum_12: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 2, 3]);  mul_260 = None
    mul_265: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_89);  primals_89 = None
    unsqueeze_449: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_265, 0);  mul_265 = None
    unsqueeze_450: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_266: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_15, unsqueeze_451);  unsqueeze_451 = None
    mul_267: "f32[192]" = torch.ops.aten.mul.Tensor(sum_12, rsqrt_4);  sum_12 = rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_266, mul_191, primals_203, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_266 = mul_191 = primals_203 = None
    getitem_18: "f32[4, 1152, 7, 7]" = convolution_backward_6[0]
    getitem_19: "f32[192, 1152, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_268: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_18, mul_189);  mul_189 = None
    mul_269: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_18, sigmoid_59);  getitem_18 = None
    sum_13: "f32[4, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2, 3], True);  mul_268 = None
    sub_59: "f32[4, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_59)
    mul_270: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_59, sub_59);  sigmoid_59 = sub_59 = None
    mul_271: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_13, mul_270);  sum_13 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_271, mul_190, primals_201, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_271 = mul_190 = primals_201 = None
    getitem_21: "f32[4, 48, 1, 1]" = convolution_backward_7[0]
    getitem_22: "f32[1152, 48, 1, 1]" = convolution_backward_7[1]
    getitem_23: "f32[1152]" = convolution_backward_7[2];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_69: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_72)
    sub_60: "f32[4, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_69)
    mul_272: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_72, sub_60);  convolution_72 = sub_60 = None
    add_117: "f32[4, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_272, 1);  mul_272 = None
    mul_273: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_69, add_117);  sigmoid_69 = add_117 = None
    mul_274: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_21, mul_273);  getitem_21 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_274, mean_14, primals_199, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_274 = mean_14 = primals_199 = None
    getitem_24: "f32[4, 1152, 1, 1]" = convolution_backward_8[0]
    getitem_25: "f32[48, 1152, 1, 1]" = convolution_backward_8[1]
    getitem_26: "f32[48]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[4, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_24, [4, 1152, 7, 7]);  getitem_24 = None
    div_2: "f32[4, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_118: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_269, div_2);  mul_269 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_70: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_95)
    sub_61: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_70)
    mul_275: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_95, sub_61);  add_95 = sub_61 = None
    add_119: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_275, 1);  mul_275 = None
    mul_276: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_70, add_119);  sigmoid_70 = add_119 = None
    mul_277: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_118, mul_276);  add_118 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_120: "f32[1152]" = torch.ops.aten.add.Tensor(primals_301, 1e-05);  primals_301 = None
    rsqrt_5: "f32[1152]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    unsqueeze_452: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_300, 0);  primals_300 = None
    unsqueeze_453: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    sum_14: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 2, 3])
    sub_62: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_454);  convolution_71 = unsqueeze_454 = None
    mul_278: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_277, sub_62);  sub_62 = None
    sum_15: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_278, [0, 2, 3]);  mul_278 = None
    mul_283: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_87);  primals_87 = None
    unsqueeze_461: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_283, 0);  mul_283 = None
    unsqueeze_462: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_284: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_463);  mul_277 = unsqueeze_463 = None
    mul_285: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_5);  sum_15 = rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_284, mul_185, primals_198, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_284 = mul_185 = primals_198 = None
    getitem_27: "f32[4, 1152, 7, 7]" = convolution_backward_9[0]
    getitem_28: "f32[1152, 1, 5, 5]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_288: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_27, mul_287);  getitem_27 = mul_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_122: "f32[1152]" = torch.ops.aten.add.Tensor(primals_299, 1e-05);  primals_299 = None
    rsqrt_6: "f32[1152]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    unsqueeze_464: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_298, 0);  primals_298 = None
    unsqueeze_465: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    sum_16: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 2, 3])
    sub_64: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_466);  convolution_70 = unsqueeze_466 = None
    mul_289: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_288, sub_64);  sub_64 = None
    sum_17: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 2, 3]);  mul_289 = None
    mul_294: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_85);  primals_85 = None
    unsqueeze_473: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_294, 0);  mul_294 = None
    unsqueeze_474: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_295: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_288, unsqueeze_475);  mul_288 = unsqueeze_475 = None
    mul_296: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_17, rsqrt_6);  sum_17 = rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_295, add_91, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_295 = add_91 = primals_197 = None
    getitem_30: "f32[4, 192, 7, 7]" = convolution_backward_10[0]
    getitem_31: "f32[1152, 192, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_123: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(getitem_15, getitem_30);  getitem_15 = getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_124: "f32[192]" = torch.ops.aten.add.Tensor(primals_297, 1e-05);  primals_297 = None
    rsqrt_7: "f32[192]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    unsqueeze_476: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_296, 0);  primals_296 = None
    unsqueeze_477: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    sum_18: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 2, 3])
    sub_65: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_478);  convolution_69 = unsqueeze_478 = None
    mul_297: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_123, sub_65);  sub_65 = None
    sum_19: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 2, 3]);  mul_297 = None
    mul_302: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_83);  primals_83 = None
    unsqueeze_485: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_302, 0);  mul_302 = None
    unsqueeze_486: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_303: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_123, unsqueeze_487);  unsqueeze_487 = None
    mul_304: "f32[192]" = torch.ops.aten.mul.Tensor(sum_19, rsqrt_7);  sum_19 = rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_303, mul_178, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_303 = mul_178 = primals_196 = None
    getitem_33: "f32[4, 1152, 7, 7]" = convolution_backward_11[0]
    getitem_34: "f32[192, 1152, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_305: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_33, mul_176);  mul_176 = None
    mul_306: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_33, sigmoid_55);  getitem_33 = None
    sum_20: "f32[4, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2, 3], True);  mul_305 = None
    sub_66: "f32[4, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_55)
    mul_307: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_55, sub_66);  sigmoid_55 = sub_66 = None
    mul_308: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_20, mul_307);  sum_20 = mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_308, mul_177, primals_194, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_308 = mul_177 = primals_194 = None
    getitem_36: "f32[4, 48, 1, 1]" = convolution_backward_12[0]
    getitem_37: "f32[1152, 48, 1, 1]" = convolution_backward_12[1]
    getitem_38: "f32[1152]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_72: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67)
    sub_67: "f32[4, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_72)
    mul_309: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_67, sub_67);  convolution_67 = sub_67 = None
    add_125: "f32[4, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_309, 1);  mul_309 = None
    mul_310: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_72, add_125);  sigmoid_72 = add_125 = None
    mul_311: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_36, mul_310);  getitem_36 = mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_311, mean_13, primals_192, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_311 = mean_13 = primals_192 = None
    getitem_39: "f32[4, 1152, 1, 1]" = convolution_backward_13[0]
    getitem_40: "f32[48, 1152, 1, 1]" = convolution_backward_13[1]
    getitem_41: "f32[48]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[4, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_39, [4, 1152, 7, 7]);  getitem_39 = None
    div_3: "f32[4, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_126: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_306, div_3);  mul_306 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_73: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_88)
    sub_68: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_73)
    mul_312: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_88, sub_68);  add_88 = sub_68 = None
    add_127: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_312, 1);  mul_312 = None
    mul_313: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_73, add_127);  sigmoid_73 = add_127 = None
    mul_314: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_126, mul_313);  add_126 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_128: "f32[1152]" = torch.ops.aten.add.Tensor(primals_295, 1e-05);  primals_295 = None
    rsqrt_8: "f32[1152]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    unsqueeze_488: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_294, 0);  primals_294 = None
    unsqueeze_489: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    sum_21: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3])
    sub_69: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_490);  convolution_66 = unsqueeze_490 = None
    mul_315: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_314, sub_69);  sub_69 = None
    sum_22: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 2, 3]);  mul_315 = None
    mul_320: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_81);  primals_81 = None
    unsqueeze_497: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_320, 0);  mul_320 = None
    unsqueeze_498: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_321: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_314, unsqueeze_499);  mul_314 = unsqueeze_499 = None
    mul_322: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_22, rsqrt_8);  sum_22 = rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_321, mul_172, primals_191, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_321 = mul_172 = primals_191 = None
    getitem_42: "f32[4, 1152, 7, 7]" = convolution_backward_14[0]
    getitem_43: "f32[1152, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_325: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_42, mul_324);  getitem_42 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_130: "f32[1152]" = torch.ops.aten.add.Tensor(primals_293, 1e-05);  primals_293 = None
    rsqrt_9: "f32[1152]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    unsqueeze_500: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_292, 0);  primals_292 = None
    unsqueeze_501: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    sum_23: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 2, 3])
    sub_71: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_502);  convolution_65 = unsqueeze_502 = None
    mul_326: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_325, sub_71);  sub_71 = None
    sum_24: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 2, 3]);  mul_326 = None
    mul_331: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_79);  primals_79 = None
    unsqueeze_509: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
    unsqueeze_510: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_332: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_511);  mul_325 = unsqueeze_511 = None
    mul_333: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_24, rsqrt_9);  sum_24 = rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_332, add_84, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_332 = add_84 = primals_190 = None
    getitem_45: "f32[4, 192, 7, 7]" = convolution_backward_15[0]
    getitem_46: "f32[1152, 192, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_131: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_123, getitem_45);  add_123 = getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_132: "f32[192]" = torch.ops.aten.add.Tensor(primals_291, 1e-05);  primals_291 = None
    rsqrt_10: "f32[192]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    unsqueeze_512: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_290, 0);  primals_290 = None
    unsqueeze_513: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    sum_25: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_131, [0, 2, 3])
    sub_72: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_514);  convolution_64 = unsqueeze_514 = None
    mul_334: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_131, sub_72);  sub_72 = None
    sum_26: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 2, 3]);  mul_334 = None
    mul_339: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_77);  primals_77 = None
    unsqueeze_521: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_339, 0);  mul_339 = None
    unsqueeze_522: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    mul_340: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_131, unsqueeze_523);  unsqueeze_523 = None
    mul_341: "f32[192]" = torch.ops.aten.mul.Tensor(sum_26, rsqrt_10);  sum_26 = rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_340, mul_165, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_340 = mul_165 = primals_189 = None
    getitem_48: "f32[4, 1152, 7, 7]" = convolution_backward_16[0]
    getitem_49: "f32[192, 1152, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_342: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_48, mul_163);  mul_163 = None
    mul_343: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_48, sigmoid_51);  getitem_48 = None
    sum_27: "f32[4, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2, 3], True);  mul_342 = None
    sub_73: "f32[4, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_51)
    mul_344: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_51, sub_73);  sigmoid_51 = sub_73 = None
    mul_345: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_27, mul_344);  sum_27 = mul_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_345, mul_164, primals_187, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_345 = mul_164 = primals_187 = None
    getitem_51: "f32[4, 48, 1, 1]" = convolution_backward_17[0]
    getitem_52: "f32[1152, 48, 1, 1]" = convolution_backward_17[1]
    getitem_53: "f32[1152]" = convolution_backward_17[2];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_75: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_62)
    sub_74: "f32[4, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_75);  full_default_1 = None
    mul_346: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_62, sub_74);  convolution_62 = sub_74 = None
    add_133: "f32[4, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_346, 1);  mul_346 = None
    mul_347: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_75, add_133);  sigmoid_75 = add_133 = None
    mul_348: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_51, mul_347);  getitem_51 = mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_348, mean_12, primals_185, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_348 = mean_12 = primals_185 = None
    getitem_54: "f32[4, 1152, 1, 1]" = convolution_backward_18[0]
    getitem_55: "f32[48, 1152, 1, 1]" = convolution_backward_18[1]
    getitem_56: "f32[48]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[4, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_54, [4, 1152, 7, 7]);  getitem_54 = None
    div_4: "f32[4, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_134: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_343, div_4);  mul_343 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_76: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_81)
    sub_75: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_76);  full_default_2 = None
    mul_349: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_81, sub_75);  add_81 = sub_75 = None
    add_135: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_349, 1);  mul_349 = None
    mul_350: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_76, add_135);  sigmoid_76 = add_135 = None
    mul_351: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_134, mul_350);  add_134 = mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_136: "f32[1152]" = torch.ops.aten.add.Tensor(primals_289, 1e-05);  primals_289 = None
    rsqrt_11: "f32[1152]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    unsqueeze_524: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_288, 0);  primals_288 = None
    unsqueeze_525: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    sum_28: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 2, 3])
    sub_76: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_526);  convolution_61 = unsqueeze_526 = None
    mul_352: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_351, sub_76);  sub_76 = None
    sum_29: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 2, 3]);  mul_352 = None
    mul_357: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_75);  primals_75 = None
    unsqueeze_533: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_357, 0);  mul_357 = None
    unsqueeze_534: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    mul_358: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_351, unsqueeze_535);  mul_351 = unsqueeze_535 = None
    mul_359: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_11);  sum_29 = rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_358, mul_159, primals_184, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_358 = mul_159 = primals_184 = None
    getitem_57: "f32[4, 1152, 7, 7]" = convolution_backward_19[0]
    getitem_58: "f32[1152, 1, 5, 5]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_362: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_57, mul_361);  getitem_57 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_138: "f32[1152]" = torch.ops.aten.add.Tensor(primals_287, 1e-05);  primals_287 = None
    rsqrt_12: "f32[1152]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    unsqueeze_536: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_286, 0);  primals_286 = None
    unsqueeze_537: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    sum_30: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 2, 3])
    sub_78: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_538);  convolution_60 = unsqueeze_538 = None
    mul_363: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_362, sub_78);  sub_78 = None
    sum_31: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 2, 3]);  mul_363 = None
    mul_368: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_73);  primals_73 = None
    unsqueeze_545: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
    unsqueeze_546: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    mul_369: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_362, unsqueeze_547);  mul_362 = unsqueeze_547 = None
    mul_370: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_12);  sum_31 = rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_369, add_77, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_369 = add_77 = primals_183 = None
    getitem_60: "f32[4, 192, 7, 7]" = convolution_backward_20[0]
    getitem_61: "f32[1152, 192, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_139: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_131, getitem_60);  add_131 = getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_140: "f32[192]" = torch.ops.aten.add.Tensor(primals_285, 1e-05);  primals_285 = None
    rsqrt_13: "f32[192]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    unsqueeze_548: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_284, 0);  primals_284 = None
    unsqueeze_549: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    sum_32: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_139, [0, 2, 3])
    sub_79: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_550);  convolution_59 = unsqueeze_550 = None
    mul_371: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_139, sub_79);  sub_79 = None
    sum_33: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_371, [0, 2, 3]);  mul_371 = None
    mul_376: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_71);  primals_71 = None
    unsqueeze_557: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
    unsqueeze_558: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    mul_377: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_139, unsqueeze_559);  add_139 = unsqueeze_559 = None
    mul_378: "f32[192]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_13);  sum_33 = rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_377, mul_152, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_377 = mul_152 = primals_182 = None
    getitem_63: "f32[4, 672, 7, 7]" = convolution_backward_21[0]
    getitem_64: "f32[192, 672, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_379: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_63, mul_150);  mul_150 = None
    mul_380: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_63, sigmoid_47);  getitem_63 = None
    sum_34: "f32[4, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [2, 3], True);  mul_379 = None
    sub_80: "f32[4, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_47)
    mul_381: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_47, sub_80);  sigmoid_47 = sub_80 = None
    mul_382: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_34, mul_381);  sum_34 = mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_382, mul_151, primals_180, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_382 = mul_151 = primals_180 = None
    getitem_66: "f32[4, 28, 1, 1]" = convolution_backward_22[0]
    getitem_67: "f32[672, 28, 1, 1]" = convolution_backward_22[1]
    getitem_68: "f32[672]" = convolution_backward_22[2];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_78: "f32[4, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_57)
    full_default_13: "f32[4, 28, 1, 1]" = torch.ops.aten.full.default([4, 28, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_81: "f32[4, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_13, sigmoid_78)
    mul_383: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_57, sub_81);  convolution_57 = sub_81 = None
    add_141: "f32[4, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_383, 1);  mul_383 = None
    mul_384: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_78, add_141);  sigmoid_78 = add_141 = None
    mul_385: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_66, mul_384);  getitem_66 = mul_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_385, mean_11, primals_178, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_385 = mean_11 = primals_178 = None
    getitem_69: "f32[4, 672, 1, 1]" = convolution_backward_23[0]
    getitem_70: "f32[28, 672, 1, 1]" = convolution_backward_23[1]
    getitem_71: "f32[28]" = convolution_backward_23[2];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[4, 672, 7, 7]" = torch.ops.aten.expand.default(getitem_69, [4, 672, 7, 7]);  getitem_69 = None
    div_5: "f32[4, 672, 7, 7]" = torch.ops.aten.div.Scalar(expand_5, 49);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_142: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_380, div_5);  mul_380 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_79: "f32[4, 672, 7, 7]" = torch.ops.aten.sigmoid.default(add_75)
    full_default_14: "f32[4, 672, 7, 7]" = torch.ops.aten.full.default([4, 672, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_82: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_79);  full_default_14 = None
    mul_386: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_75, sub_82);  add_75 = sub_82 = None
    add_143: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Scalar(mul_386, 1);  mul_386 = None
    mul_387: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_79, add_143);  sigmoid_79 = add_143 = None
    mul_388: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_142, mul_387);  add_142 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_144: "f32[672]" = torch.ops.aten.add.Tensor(primals_283, 1e-05);  primals_283 = None
    rsqrt_14: "f32[672]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    unsqueeze_560: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_282, 0);  primals_282 = None
    unsqueeze_561: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    sum_35: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 2, 3])
    sub_83: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_562);  convolution_56 = unsqueeze_562 = None
    mul_389: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_388, sub_83);  sub_83 = None
    sum_36: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_389, [0, 2, 3]);  mul_389 = None
    mul_394: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_69);  primals_69 = None
    unsqueeze_569: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
    unsqueeze_570: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    mul_395: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_571);  mul_388 = unsqueeze_571 = None
    mul_396: "f32[672]" = torch.ops.aten.mul.Tensor(sum_36, rsqrt_14);  sum_36 = rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_395, mul_146, primals_177, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_395 = mul_146 = primals_177 = None
    getitem_72: "f32[4, 672, 14, 14]" = convolution_backward_24[0]
    getitem_73: "f32[672, 1, 5, 5]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_15: "f32[4, 672, 14, 14]" = torch.ops.aten.full.default([4, 672, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    mul_399: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_72, mul_398);  getitem_72 = mul_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_146: "f32[672]" = torch.ops.aten.add.Tensor(primals_281, 1e-05);  primals_281 = None
    rsqrt_15: "f32[672]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    unsqueeze_572: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_280, 0);  primals_280 = None
    unsqueeze_573: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    sum_37: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_399, [0, 2, 3])
    sub_85: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_574);  convolution_55 = unsqueeze_574 = None
    mul_400: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_399, sub_85);  sub_85 = None
    sum_38: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None
    mul_405: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_67);  primals_67 = None
    unsqueeze_581: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
    unsqueeze_582: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    mul_406: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_583);  mul_399 = unsqueeze_583 = None
    mul_407: "f32[672]" = torch.ops.aten.mul.Tensor(sum_38, rsqrt_15);  sum_38 = rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_406, add_71, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_406 = add_71 = primals_176 = None
    getitem_75: "f32[4, 112, 14, 14]" = convolution_backward_25[0]
    getitem_76: "f32[672, 112, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_147: "f32[112]" = torch.ops.aten.add.Tensor(primals_279, 1e-05);  primals_279 = None
    rsqrt_16: "f32[112]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    unsqueeze_584: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(primals_278, 0);  primals_278 = None
    unsqueeze_585: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    sum_39: "f32[112]" = torch.ops.aten.sum.dim_IntList(getitem_75, [0, 2, 3])
    sub_86: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_586);  convolution_54 = unsqueeze_586 = None
    mul_408: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_75, sub_86);  sub_86 = None
    sum_40: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 2, 3]);  mul_408 = None
    mul_413: "f32[112]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_65);  primals_65 = None
    unsqueeze_593: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
    unsqueeze_594: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    mul_414: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_75, unsqueeze_595);  unsqueeze_595 = None
    mul_415: "f32[112]" = torch.ops.aten.mul.Tensor(sum_40, rsqrt_16);  sum_40 = rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_414, mul_139, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_414 = mul_139 = primals_175 = None
    getitem_78: "f32[4, 672, 14, 14]" = convolution_backward_26[0]
    getitem_79: "f32[112, 672, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_416: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_78, mul_137);  mul_137 = None
    mul_417: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_78, sigmoid_43);  getitem_78 = None
    sum_41: "f32[4, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2, 3], True);  mul_416 = None
    sub_87: "f32[4, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_43)
    mul_418: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_43, sub_87);  sigmoid_43 = sub_87 = None
    mul_419: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_41, mul_418);  sum_41 = mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_419, mul_138, primals_173, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_419 = mul_138 = primals_173 = None
    getitem_81: "f32[4, 28, 1, 1]" = convolution_backward_27[0]
    getitem_82: "f32[672, 28, 1, 1]" = convolution_backward_27[1]
    getitem_83: "f32[672]" = convolution_backward_27[2];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_81: "f32[4, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52)
    sub_88: "f32[4, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_13, sigmoid_81)
    mul_420: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_52, sub_88);  convolution_52 = sub_88 = None
    add_148: "f32[4, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_420, 1);  mul_420 = None
    mul_421: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_81, add_148);  sigmoid_81 = add_148 = None
    mul_422: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_81, mul_421);  getitem_81 = mul_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_422, mean_10, primals_171, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_422 = mean_10 = primals_171 = None
    getitem_84: "f32[4, 672, 1, 1]" = convolution_backward_28[0]
    getitem_85: "f32[28, 672, 1, 1]" = convolution_backward_28[1]
    getitem_86: "f32[28]" = convolution_backward_28[2];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[4, 672, 14, 14]" = torch.ops.aten.expand.default(getitem_84, [4, 672, 14, 14]);  getitem_84 = None
    div_6: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_149: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_417, div_6);  mul_417 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_82: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_68)
    sub_89: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_15, sigmoid_82)
    mul_423: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_68, sub_89);  add_68 = sub_89 = None
    add_150: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_423, 1);  mul_423 = None
    mul_424: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_82, add_150);  sigmoid_82 = add_150 = None
    mul_425: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_149, mul_424);  add_149 = mul_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_151: "f32[672]" = torch.ops.aten.add.Tensor(primals_277, 1e-05);  primals_277 = None
    rsqrt_17: "f32[672]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    unsqueeze_596: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_276, 0);  primals_276 = None
    unsqueeze_597: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    sum_42: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_425, [0, 2, 3])
    sub_90: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_598);  convolution_51 = unsqueeze_598 = None
    mul_426: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_425, sub_90);  sub_90 = None
    sum_43: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 2, 3]);  mul_426 = None
    mul_431: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_63);  primals_63 = None
    unsqueeze_605: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_606: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_432: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_425, unsqueeze_607);  mul_425 = unsqueeze_607 = None
    mul_433: "f32[672]" = torch.ops.aten.mul.Tensor(sum_43, rsqrt_17);  sum_43 = rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_432, mul_133, primals_170, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_432 = mul_133 = primals_170 = None
    getitem_87: "f32[4, 672, 14, 14]" = convolution_backward_29[0]
    getitem_88: "f32[672, 1, 5, 5]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_436: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_87, mul_435);  getitem_87 = mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_153: "f32[672]" = torch.ops.aten.add.Tensor(primals_275, 1e-05);  primals_275 = None
    rsqrt_18: "f32[672]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    unsqueeze_608: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_274, 0);  primals_274 = None
    unsqueeze_609: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    sum_44: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3])
    sub_92: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_610);  convolution_50 = unsqueeze_610 = None
    mul_437: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_436, sub_92);  sub_92 = None
    sum_45: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 2, 3]);  mul_437 = None
    mul_442: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_61);  primals_61 = None
    unsqueeze_617: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
    unsqueeze_618: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    mul_443: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_436, unsqueeze_619);  mul_436 = unsqueeze_619 = None
    mul_444: "f32[672]" = torch.ops.aten.mul.Tensor(sum_45, rsqrt_18);  sum_45 = rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_443, add_64, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_443 = add_64 = primals_169 = None
    getitem_90: "f32[4, 112, 14, 14]" = convolution_backward_30[0]
    getitem_91: "f32[672, 112, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_154: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(getitem_75, getitem_90);  getitem_75 = getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_155: "f32[112]" = torch.ops.aten.add.Tensor(primals_273, 1e-05);  primals_273 = None
    rsqrt_19: "f32[112]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    unsqueeze_620: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(primals_272, 0);  primals_272 = None
    unsqueeze_621: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    sum_46: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_154, [0, 2, 3])
    sub_93: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_622);  convolution_49 = unsqueeze_622 = None
    mul_445: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_154, sub_93);  sub_93 = None
    sum_47: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3]);  mul_445 = None
    mul_450: "f32[112]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_59);  primals_59 = None
    unsqueeze_629: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_630: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    mul_451: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_154, unsqueeze_631);  unsqueeze_631 = None
    mul_452: "f32[112]" = torch.ops.aten.mul.Tensor(sum_47, rsqrt_19);  sum_47 = rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_451, mul_126, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_451 = mul_126 = primals_168 = None
    getitem_93: "f32[4, 672, 14, 14]" = convolution_backward_31[0]
    getitem_94: "f32[112, 672, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_453: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_93, mul_124);  mul_124 = None
    mul_454: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_93, sigmoid_39);  getitem_93 = None
    sum_48: "f32[4, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2, 3], True);  mul_453 = None
    sub_94: "f32[4, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_39)
    mul_455: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_39, sub_94);  sigmoid_39 = sub_94 = None
    mul_456: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_48, mul_455);  sum_48 = mul_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_456, mul_125, primals_166, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_456 = mul_125 = primals_166 = None
    getitem_96: "f32[4, 28, 1, 1]" = convolution_backward_32[0]
    getitem_97: "f32[672, 28, 1, 1]" = convolution_backward_32[1]
    getitem_98: "f32[672]" = convolution_backward_32[2];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_84: "f32[4, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_47)
    sub_95: "f32[4, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_13, sigmoid_84);  full_default_13 = None
    mul_457: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_47, sub_95);  convolution_47 = sub_95 = None
    add_156: "f32[4, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_457, 1);  mul_457 = None
    mul_458: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_84, add_156);  sigmoid_84 = add_156 = None
    mul_459: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_96, mul_458);  getitem_96 = mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_459, mean_9, primals_164, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_459 = mean_9 = primals_164 = None
    getitem_99: "f32[4, 672, 1, 1]" = convolution_backward_33[0]
    getitem_100: "f32[28, 672, 1, 1]" = convolution_backward_33[1]
    getitem_101: "f32[28]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[4, 672, 14, 14]" = torch.ops.aten.expand.default(getitem_99, [4, 672, 14, 14]);  getitem_99 = None
    div_7: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_157: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_454, div_7);  mul_454 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_85: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_61)
    sub_96: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_15, sigmoid_85);  full_default_15 = None
    mul_460: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_61, sub_96);  add_61 = sub_96 = None
    add_158: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_460, 1);  mul_460 = None
    mul_461: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_85, add_158);  sigmoid_85 = add_158 = None
    mul_462: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_157, mul_461);  add_157 = mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_159: "f32[672]" = torch.ops.aten.add.Tensor(primals_271, 1e-05);  primals_271 = None
    rsqrt_20: "f32[672]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    unsqueeze_632: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_270, 0);  primals_270 = None
    unsqueeze_633: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    sum_49: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_462, [0, 2, 3])
    sub_97: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_634);  convolution_46 = unsqueeze_634 = None
    mul_463: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_462, sub_97);  sub_97 = None
    sum_50: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3]);  mul_463 = None
    mul_468: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_57);  primals_57 = None
    unsqueeze_641: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_642: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_469: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_643);  mul_462 = unsqueeze_643 = None
    mul_470: "f32[672]" = torch.ops.aten.mul.Tensor(sum_50, rsqrt_20);  sum_50 = rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_469, mul_120, primals_163, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_469 = mul_120 = primals_163 = None
    getitem_102: "f32[4, 672, 14, 14]" = convolution_backward_34[0]
    getitem_103: "f32[672, 1, 5, 5]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_473: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_102, mul_472);  getitem_102 = mul_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_161: "f32[672]" = torch.ops.aten.add.Tensor(primals_269, 1e-05);  primals_269 = None
    rsqrt_21: "f32[672]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    unsqueeze_644: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_268, 0);  primals_268 = None
    unsqueeze_645: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    sum_51: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_473, [0, 2, 3])
    sub_99: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_646);  convolution_45 = unsqueeze_646 = None
    mul_474: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_473, sub_99);  sub_99 = None
    sum_52: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_474, [0, 2, 3]);  mul_474 = None
    mul_479: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_55);  primals_55 = None
    unsqueeze_653: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_479, 0);  mul_479 = None
    unsqueeze_654: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_480: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_473, unsqueeze_655);  mul_473 = unsqueeze_655 = None
    mul_481: "f32[672]" = torch.ops.aten.mul.Tensor(sum_52, rsqrt_21);  sum_52 = rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_480, add_57, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_480 = add_57 = primals_162 = None
    getitem_105: "f32[4, 112, 14, 14]" = convolution_backward_35[0]
    getitem_106: "f32[672, 112, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_162: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_154, getitem_105);  add_154 = getitem_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_163: "f32[112]" = torch.ops.aten.add.Tensor(primals_267, 1e-05);  primals_267 = None
    rsqrt_22: "f32[112]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    unsqueeze_656: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(primals_266, 0);  primals_266 = None
    unsqueeze_657: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    sum_53: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 2, 3])
    sub_100: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_658);  convolution_44 = unsqueeze_658 = None
    mul_482: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_162, sub_100);  sub_100 = None
    sum_54: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 2, 3]);  mul_482 = None
    mul_487: "f32[112]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_53);  primals_53 = None
    unsqueeze_665: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
    unsqueeze_666: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    mul_488: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_162, unsqueeze_667);  add_162 = unsqueeze_667 = None
    mul_489: "f32[112]" = torch.ops.aten.mul.Tensor(sum_54, rsqrt_22);  sum_54 = rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_488, mul_113, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_488 = mul_113 = primals_161 = None
    getitem_108: "f32[4, 480, 14, 14]" = convolution_backward_36[0]
    getitem_109: "f32[112, 480, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_490: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_108, mul_111);  mul_111 = None
    mul_491: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_108, sigmoid_35);  getitem_108 = None
    sum_55: "f32[4, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_490, [2, 3], True);  mul_490 = None
    sub_101: "f32[4, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_35)
    mul_492: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_35, sub_101);  sigmoid_35 = sub_101 = None
    mul_493: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_55, mul_492);  sum_55 = mul_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_493, mul_112, primals_159, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_493 = mul_112 = primals_159 = None
    getitem_111: "f32[4, 20, 1, 1]" = convolution_backward_37[0]
    getitem_112: "f32[480, 20, 1, 1]" = convolution_backward_37[1]
    getitem_113: "f32[480]" = convolution_backward_37[2];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_87: "f32[4, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42)
    full_default_22: "f32[4, 20, 1, 1]" = torch.ops.aten.full.default([4, 20, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_102: "f32[4, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_22, sigmoid_87)
    mul_494: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_42, sub_102);  convolution_42 = sub_102 = None
    add_164: "f32[4, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_494, 1);  mul_494 = None
    mul_495: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_87, add_164);  sigmoid_87 = add_164 = None
    mul_496: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_111, mul_495);  getitem_111 = mul_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_496, mean_8, primals_157, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_496 = mean_8 = primals_157 = None
    getitem_114: "f32[4, 480, 1, 1]" = convolution_backward_38[0]
    getitem_115: "f32[20, 480, 1, 1]" = convolution_backward_38[1]
    getitem_116: "f32[20]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[4, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_114, [4, 480, 14, 14]);  getitem_114 = None
    div_8: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_165: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_491, div_8);  mul_491 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_88: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_55)
    full_default_23: "f32[4, 480, 14, 14]" = torch.ops.aten.full.default([4, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_103: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_88)
    mul_497: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_55, sub_103);  add_55 = sub_103 = None
    add_166: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_497, 1);  mul_497 = None
    mul_498: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_88, add_166);  sigmoid_88 = add_166 = None
    mul_499: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_165, mul_498);  add_165 = mul_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_167: "f32[480]" = torch.ops.aten.add.Tensor(primals_265, 1e-05);  primals_265 = None
    rsqrt_23: "f32[480]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    unsqueeze_668: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_264, 0);  primals_264 = None
    unsqueeze_669: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    sum_56: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 2, 3])
    sub_104: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_670);  convolution_41 = unsqueeze_670 = None
    mul_500: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_499, sub_104);  sub_104 = None
    sum_57: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3]);  mul_500 = None
    mul_505: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_23, primals_51);  primals_51 = None
    unsqueeze_677: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
    unsqueeze_678: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    mul_506: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_679);  mul_499 = unsqueeze_679 = None
    mul_507: "f32[480]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_23);  sum_57 = rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_506, mul_107, primals_156, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_506 = mul_107 = primals_156 = None
    getitem_117: "f32[4, 480, 14, 14]" = convolution_backward_39[0]
    getitem_118: "f32[480, 1, 5, 5]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_510: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_117, mul_509);  getitem_117 = mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_169: "f32[480]" = torch.ops.aten.add.Tensor(primals_263, 1e-05);  primals_263 = None
    rsqrt_24: "f32[480]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    unsqueeze_680: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_262, 0);  primals_262 = None
    unsqueeze_681: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    sum_58: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 2, 3])
    sub_106: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_682);  convolution_40 = unsqueeze_682 = None
    mul_511: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_510, sub_106);  sub_106 = None
    sum_59: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 2, 3]);  mul_511 = None
    mul_516: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_24, primals_49);  primals_49 = None
    unsqueeze_689: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
    unsqueeze_690: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    mul_517: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_510, unsqueeze_691);  mul_510 = unsqueeze_691 = None
    mul_518: "f32[480]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_24);  sum_59 = rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_517, add_51, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_517 = add_51 = primals_155 = None
    getitem_120: "f32[4, 80, 14, 14]" = convolution_backward_40[0]
    getitem_121: "f32[480, 80, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_170: "f32[80]" = torch.ops.aten.add.Tensor(primals_261, 1e-05);  primals_261 = None
    rsqrt_25: "f32[80]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    unsqueeze_692: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_260, 0);  primals_260 = None
    unsqueeze_693: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    sum_60: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_120, [0, 2, 3])
    sub_107: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_694);  convolution_39 = unsqueeze_694 = None
    mul_519: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_120, sub_107);  sub_107 = None
    sum_61: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2, 3]);  mul_519 = None
    mul_524: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_25, primals_47);  primals_47 = None
    unsqueeze_701: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
    unsqueeze_702: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    mul_525: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_120, unsqueeze_703);  unsqueeze_703 = None
    mul_526: "f32[80]" = torch.ops.aten.mul.Tensor(sum_61, rsqrt_25);  sum_61 = rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_525, mul_100, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_525 = mul_100 = primals_154 = None
    getitem_123: "f32[4, 480, 14, 14]" = convolution_backward_41[0]
    getitem_124: "f32[80, 480, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_527: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_123, mul_98);  mul_98 = None
    mul_528: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_123, sigmoid_31);  getitem_123 = None
    sum_62: "f32[4, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [2, 3], True);  mul_527 = None
    sub_108: "f32[4, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_31)
    mul_529: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_31, sub_108);  sigmoid_31 = sub_108 = None
    mul_530: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_62, mul_529);  sum_62 = mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_530, mul_99, primals_152, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_530 = mul_99 = primals_152 = None
    getitem_126: "f32[4, 20, 1, 1]" = convolution_backward_42[0]
    getitem_127: "f32[480, 20, 1, 1]" = convolution_backward_42[1]
    getitem_128: "f32[480]" = convolution_backward_42[2];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_90: "f32[4, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_37)
    sub_109: "f32[4, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_22, sigmoid_90)
    mul_531: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_37, sub_109);  convolution_37 = sub_109 = None
    add_171: "f32[4, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_531, 1);  mul_531 = None
    mul_532: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_90, add_171);  sigmoid_90 = add_171 = None
    mul_533: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_126, mul_532);  getitem_126 = mul_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_533, mean_7, primals_150, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_533 = mean_7 = primals_150 = None
    getitem_129: "f32[4, 480, 1, 1]" = convolution_backward_43[0]
    getitem_130: "f32[20, 480, 1, 1]" = convolution_backward_43[1]
    getitem_131: "f32[20]" = convolution_backward_43[2];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[4, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_129, [4, 480, 14, 14]);  getitem_129 = None
    div_9: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_172: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_528, div_9);  mul_528 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_91: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_48)
    sub_110: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_91)
    mul_534: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_48, sub_110);  add_48 = sub_110 = None
    add_173: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_534, 1);  mul_534 = None
    mul_535: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_91, add_173);  sigmoid_91 = add_173 = None
    mul_536: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_172, mul_535);  add_172 = mul_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_174: "f32[480]" = torch.ops.aten.add.Tensor(primals_259, 1e-05);  primals_259 = None
    rsqrt_26: "f32[480]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    unsqueeze_704: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_258, 0);  primals_258 = None
    unsqueeze_705: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
    unsqueeze_706: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
    sum_63: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 2, 3])
    sub_111: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_706);  convolution_36 = unsqueeze_706 = None
    mul_537: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_536, sub_111);  sub_111 = None
    sum_64: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_537, [0, 2, 3]);  mul_537 = None
    mul_542: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_26, primals_45);  primals_45 = None
    unsqueeze_713: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
    unsqueeze_714: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    mul_543: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_536, unsqueeze_715);  mul_536 = unsqueeze_715 = None
    mul_544: "f32[480]" = torch.ops.aten.mul.Tensor(sum_64, rsqrt_26);  sum_64 = rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_543, mul_94, primals_149, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_543 = mul_94 = primals_149 = None
    getitem_132: "f32[4, 480, 14, 14]" = convolution_backward_44[0]
    getitem_133: "f32[480, 1, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_547: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_132, mul_546);  getitem_132 = mul_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_176: "f32[480]" = torch.ops.aten.add.Tensor(primals_257, 1e-05);  primals_257 = None
    rsqrt_27: "f32[480]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    unsqueeze_716: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_256, 0);  primals_256 = None
    unsqueeze_717: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
    unsqueeze_718: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
    sum_65: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3])
    sub_113: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_718);  convolution_35 = unsqueeze_718 = None
    mul_548: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_547, sub_113);  sub_113 = None
    sum_66: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_548, [0, 2, 3]);  mul_548 = None
    mul_553: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_27, primals_43);  primals_43 = None
    unsqueeze_725: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
    unsqueeze_726: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    mul_554: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_547, unsqueeze_727);  mul_547 = unsqueeze_727 = None
    mul_555: "f32[480]" = torch.ops.aten.mul.Tensor(sum_66, rsqrt_27);  sum_66 = rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_554, add_44, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_554 = add_44 = primals_148 = None
    getitem_135: "f32[4, 80, 14, 14]" = convolution_backward_45[0]
    getitem_136: "f32[480, 80, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_177: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_120, getitem_135);  getitem_120 = getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_178: "f32[80]" = torch.ops.aten.add.Tensor(primals_255, 1e-05);  primals_255 = None
    rsqrt_28: "f32[80]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    unsqueeze_728: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_254, 0);  primals_254 = None
    unsqueeze_729: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
    unsqueeze_730: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
    sum_67: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_177, [0, 2, 3])
    sub_114: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_730);  convolution_34 = unsqueeze_730 = None
    mul_556: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_177, sub_114);  sub_114 = None
    sum_68: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 2, 3]);  mul_556 = None
    mul_561: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_28, primals_41);  primals_41 = None
    unsqueeze_737: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
    unsqueeze_738: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    mul_562: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_177, unsqueeze_739);  unsqueeze_739 = None
    mul_563: "f32[80]" = torch.ops.aten.mul.Tensor(sum_68, rsqrt_28);  sum_68 = rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_562, mul_87, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_562 = mul_87 = primals_147 = None
    getitem_138: "f32[4, 480, 14, 14]" = convolution_backward_46[0]
    getitem_139: "f32[80, 480, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_564: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_138, mul_85);  mul_85 = None
    mul_565: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_138, sigmoid_27);  getitem_138 = None
    sum_69: "f32[4, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_564, [2, 3], True);  mul_564 = None
    sub_115: "f32[4, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_27)
    mul_566: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_27, sub_115);  sigmoid_27 = sub_115 = None
    mul_567: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, mul_566);  sum_69 = mul_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_567, mul_86, primals_145, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_567 = mul_86 = primals_145 = None
    getitem_141: "f32[4, 20, 1, 1]" = convolution_backward_47[0]
    getitem_142: "f32[480, 20, 1, 1]" = convolution_backward_47[1]
    getitem_143: "f32[480]" = convolution_backward_47[2];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_93: "f32[4, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_32)
    sub_116: "f32[4, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_22, sigmoid_93);  full_default_22 = None
    mul_568: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_32, sub_116);  convolution_32 = sub_116 = None
    add_179: "f32[4, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_568, 1);  mul_568 = None
    mul_569: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_93, add_179);  sigmoid_93 = add_179 = None
    mul_570: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_141, mul_569);  getitem_141 = mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_570, mean_6, primals_143, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_570 = mean_6 = primals_143 = None
    getitem_144: "f32[4, 480, 1, 1]" = convolution_backward_48[0]
    getitem_145: "f32[20, 480, 1, 1]" = convolution_backward_48[1]
    getitem_146: "f32[20]" = convolution_backward_48[2];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[4, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_144, [4, 480, 14, 14]);  getitem_144 = None
    div_10: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_180: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_565, div_10);  mul_565 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_94: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_41)
    sub_117: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_94);  full_default_23 = None
    mul_571: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_41, sub_117);  add_41 = sub_117 = None
    add_181: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_571, 1);  mul_571 = None
    mul_572: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_94, add_181);  sigmoid_94 = add_181 = None
    mul_573: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_180, mul_572);  add_180 = mul_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_182: "f32[480]" = torch.ops.aten.add.Tensor(primals_253, 1e-05);  primals_253 = None
    rsqrt_29: "f32[480]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    unsqueeze_740: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_252, 0);  primals_252 = None
    unsqueeze_741: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
    unsqueeze_742: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
    sum_70: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 2, 3])
    sub_118: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_742);  convolution_31 = unsqueeze_742 = None
    mul_574: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_573, sub_118);  sub_118 = None
    sum_71: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_574, [0, 2, 3]);  mul_574 = None
    mul_579: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_29, primals_39);  primals_39 = None
    unsqueeze_749: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_750: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    mul_580: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_573, unsqueeze_751);  mul_573 = unsqueeze_751 = None
    mul_581: "f32[480]" = torch.ops.aten.mul.Tensor(sum_71, rsqrt_29);  sum_71 = rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_580, mul_81, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_580 = mul_81 = primals_142 = None
    getitem_147: "f32[4, 480, 14, 14]" = convolution_backward_49[0]
    getitem_148: "f32[480, 1, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_584: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_147, mul_583);  getitem_147 = mul_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_184: "f32[480]" = torch.ops.aten.add.Tensor(primals_251, 1e-05);  primals_251 = None
    rsqrt_30: "f32[480]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    unsqueeze_752: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_250, 0);  primals_250 = None
    unsqueeze_753: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
    unsqueeze_754: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
    sum_72: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 2, 3])
    sub_120: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_754);  convolution_30 = unsqueeze_754 = None
    mul_585: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_584, sub_120);  sub_120 = None
    sum_73: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_585, [0, 2, 3]);  mul_585 = None
    mul_590: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_30, primals_37);  primals_37 = None
    unsqueeze_761: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_590, 0);  mul_590 = None
    unsqueeze_762: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    mul_591: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_584, unsqueeze_763);  mul_584 = unsqueeze_763 = None
    mul_592: "f32[480]" = torch.ops.aten.mul.Tensor(sum_73, rsqrt_30);  sum_73 = rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_591, add_37, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_591 = add_37 = primals_141 = None
    getitem_150: "f32[4, 80, 14, 14]" = convolution_backward_50[0]
    getitem_151: "f32[480, 80, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_185: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_177, getitem_150);  add_177 = getitem_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_186: "f32[80]" = torch.ops.aten.add.Tensor(primals_249, 1e-05);  primals_249 = None
    rsqrt_31: "f32[80]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    unsqueeze_764: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_248, 0);  primals_248 = None
    unsqueeze_765: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
    unsqueeze_766: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
    sum_74: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_185, [0, 2, 3])
    sub_121: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_766);  convolution_29 = unsqueeze_766 = None
    mul_593: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_185, sub_121);  sub_121 = None
    sum_75: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_593, [0, 2, 3]);  mul_593 = None
    mul_598: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_31, primals_35);  primals_35 = None
    unsqueeze_773: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
    unsqueeze_774: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    mul_599: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_185, unsqueeze_775);  add_185 = unsqueeze_775 = None
    mul_600: "f32[80]" = torch.ops.aten.mul.Tensor(sum_75, rsqrt_31);  sum_75 = rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_599, mul_74, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_599 = mul_74 = primals_140 = None
    getitem_153: "f32[4, 240, 14, 14]" = convolution_backward_51[0]
    getitem_154: "f32[80, 240, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_601: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_153, mul_72);  mul_72 = None
    mul_602: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_153, sigmoid_23);  getitem_153 = None
    sum_76: "f32[4, 240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_601, [2, 3], True);  mul_601 = None
    sub_122: "f32[4, 240, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_23)
    mul_603: "f32[4, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_23, sub_122);  sigmoid_23 = sub_122 = None
    mul_604: "f32[4, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_76, mul_603);  sum_76 = mul_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_604, mul_73, primals_138, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_604 = mul_73 = primals_138 = None
    getitem_156: "f32[4, 10, 1, 1]" = convolution_backward_52[0]
    getitem_157: "f32[240, 10, 1, 1]" = convolution_backward_52[1]
    getitem_158: "f32[240]" = convolution_backward_52[2];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_96: "f32[4, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_27)
    full_default_31: "f32[4, 10, 1, 1]" = torch.ops.aten.full.default([4, 10, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_123: "f32[4, 10, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_31, sigmoid_96)
    mul_605: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_27, sub_123);  convolution_27 = sub_123 = None
    add_187: "f32[4, 10, 1, 1]" = torch.ops.aten.add.Scalar(mul_605, 1);  mul_605 = None
    mul_606: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_96, add_187);  sigmoid_96 = add_187 = None
    mul_607: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_156, mul_606);  getitem_156 = mul_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_607, mean_5, primals_136, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_607 = mean_5 = primals_136 = None
    getitem_159: "f32[4, 240, 1, 1]" = convolution_backward_53[0]
    getitem_160: "f32[10, 240, 1, 1]" = convolution_backward_53[1]
    getitem_161: "f32[10]" = convolution_backward_53[2];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[4, 240, 14, 14]" = torch.ops.aten.expand.default(getitem_159, [4, 240, 14, 14]);  getitem_159 = None
    div_11: "f32[4, 240, 14, 14]" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_188: "f32[4, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_602, div_11);  mul_602 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_97: "f32[4, 240, 14, 14]" = torch.ops.aten.sigmoid.default(add_35)
    full_default_32: "f32[4, 240, 14, 14]" = torch.ops.aten.full.default([4, 240, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_124: "f32[4, 240, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_32, sigmoid_97);  full_default_32 = None
    mul_608: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_35, sub_124);  add_35 = sub_124 = None
    add_189: "f32[4, 240, 14, 14]" = torch.ops.aten.add.Scalar(mul_608, 1);  mul_608 = None
    mul_609: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_97, add_189);  sigmoid_97 = add_189 = None
    mul_610: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_188, mul_609);  add_188 = mul_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_190: "f32[240]" = torch.ops.aten.add.Tensor(primals_247, 1e-05);  primals_247 = None
    rsqrt_32: "f32[240]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
    unsqueeze_776: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_246, 0);  primals_246 = None
    unsqueeze_777: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
    unsqueeze_778: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
    sum_77: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_610, [0, 2, 3])
    sub_125: "f32[4, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_778);  convolution_26 = unsqueeze_778 = None
    mul_611: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_610, sub_125);  sub_125 = None
    sum_78: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_611, [0, 2, 3]);  mul_611 = None
    mul_616: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_32, primals_33);  primals_33 = None
    unsqueeze_785: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
    unsqueeze_786: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    mul_617: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_610, unsqueeze_787);  mul_610 = unsqueeze_787 = None
    mul_618: "f32[240]" = torch.ops.aten.mul.Tensor(sum_78, rsqrt_32);  sum_78 = rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_617, mul_68, primals_135, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_617 = mul_68 = primals_135 = None
    getitem_162: "f32[4, 240, 28, 28]" = convolution_backward_54[0]
    getitem_163: "f32[240, 1, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_33: "f32[4, 240, 28, 28]" = torch.ops.aten.full.default([4, 240, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    mul_621: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_162, mul_620);  getitem_162 = mul_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_192: "f32[240]" = torch.ops.aten.add.Tensor(primals_245, 1e-05);  primals_245 = None
    rsqrt_33: "f32[240]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    unsqueeze_788: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_244, 0);  primals_244 = None
    unsqueeze_789: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
    unsqueeze_790: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
    sum_79: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_621, [0, 2, 3])
    sub_127: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_790);  convolution_25 = unsqueeze_790 = None
    mul_622: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_621, sub_127);  sub_127 = None
    sum_80: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_622, [0, 2, 3]);  mul_622 = None
    mul_627: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_33, primals_31);  primals_31 = None
    unsqueeze_797: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_627, 0);  mul_627 = None
    unsqueeze_798: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    mul_628: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_621, unsqueeze_799);  mul_621 = unsqueeze_799 = None
    mul_629: "f32[240]" = torch.ops.aten.mul.Tensor(sum_80, rsqrt_33);  sum_80 = rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_628, add_31, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_628 = add_31 = primals_134 = None
    getitem_165: "f32[4, 40, 28, 28]" = convolution_backward_55[0]
    getitem_166: "f32[240, 40, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_193: "f32[40]" = torch.ops.aten.add.Tensor(primals_243, 1e-05);  primals_243 = None
    rsqrt_34: "f32[40]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    unsqueeze_800: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(primals_242, 0);  primals_242 = None
    unsqueeze_801: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
    unsqueeze_802: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
    sum_81: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_165, [0, 2, 3])
    sub_128: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_802);  convolution_24 = unsqueeze_802 = None
    mul_630: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_165, sub_128);  sub_128 = None
    sum_82: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_630, [0, 2, 3]);  mul_630 = None
    mul_635: "f32[40]" = torch.ops.aten.mul.Tensor(rsqrt_34, primals_29);  primals_29 = None
    unsqueeze_809: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    unsqueeze_810: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    mul_636: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_165, unsqueeze_811);  unsqueeze_811 = None
    mul_637: "f32[40]" = torch.ops.aten.mul.Tensor(sum_82, rsqrt_34);  sum_82 = rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_636, mul_61, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_636 = mul_61 = primals_133 = None
    getitem_168: "f32[4, 240, 28, 28]" = convolution_backward_56[0]
    getitem_169: "f32[40, 240, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_638: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_168, mul_59);  mul_59 = None
    mul_639: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_168, sigmoid_19);  getitem_168 = None
    sum_83: "f32[4, 240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_638, [2, 3], True);  mul_638 = None
    sub_129: "f32[4, 240, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_19)
    mul_640: "f32[4, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_19, sub_129);  sigmoid_19 = sub_129 = None
    mul_641: "f32[4, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_83, mul_640);  sum_83 = mul_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_641, mul_60, primals_131, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_641 = mul_60 = primals_131 = None
    getitem_171: "f32[4, 10, 1, 1]" = convolution_backward_57[0]
    getitem_172: "f32[240, 10, 1, 1]" = convolution_backward_57[1]
    getitem_173: "f32[240]" = convolution_backward_57[2];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_99: "f32[4, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_22)
    sub_130: "f32[4, 10, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_31, sigmoid_99);  full_default_31 = None
    mul_642: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_22, sub_130);  convolution_22 = sub_130 = None
    add_194: "f32[4, 10, 1, 1]" = torch.ops.aten.add.Scalar(mul_642, 1);  mul_642 = None
    mul_643: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_99, add_194);  sigmoid_99 = add_194 = None
    mul_644: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_171, mul_643);  getitem_171 = mul_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_644, mean_4, primals_129, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_644 = mean_4 = primals_129 = None
    getitem_174: "f32[4, 240, 1, 1]" = convolution_backward_58[0]
    getitem_175: "f32[10, 240, 1, 1]" = convolution_backward_58[1]
    getitem_176: "f32[10]" = convolution_backward_58[2];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[4, 240, 28, 28]" = torch.ops.aten.expand.default(getitem_174, [4, 240, 28, 28]);  getitem_174 = None
    div_12: "f32[4, 240, 28, 28]" = torch.ops.aten.div.Scalar(expand_12, 784);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_195: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_639, div_12);  mul_639 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_100: "f32[4, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_28)
    sub_131: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_33, sigmoid_100);  full_default_33 = None
    mul_645: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_28, sub_131);  add_28 = sub_131 = None
    add_196: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Scalar(mul_645, 1);  mul_645 = None
    mul_646: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_100, add_196);  sigmoid_100 = add_196 = None
    mul_647: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_195, mul_646);  add_195 = mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_197: "f32[240]" = torch.ops.aten.add.Tensor(primals_241, 1e-05);  primals_241 = None
    rsqrt_35: "f32[240]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    unsqueeze_812: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_240, 0);  primals_240 = None
    unsqueeze_813: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
    unsqueeze_814: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
    sum_84: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 2, 3])
    sub_132: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_814);  convolution_21 = unsqueeze_814 = None
    mul_648: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_647, sub_132);  sub_132 = None
    sum_85: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_648, [0, 2, 3]);  mul_648 = None
    mul_653: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_35, primals_27);  primals_27 = None
    unsqueeze_821: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_653, 0);  mul_653 = None
    unsqueeze_822: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    mul_654: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_647, unsqueeze_823);  mul_647 = unsqueeze_823 = None
    mul_655: "f32[240]" = torch.ops.aten.mul.Tensor(sum_85, rsqrt_35);  sum_85 = rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_654, mul_55, primals_128, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_654 = mul_55 = primals_128 = None
    getitem_177: "f32[4, 240, 28, 28]" = convolution_backward_59[0]
    getitem_178: "f32[240, 1, 5, 5]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_658: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_177, mul_657);  getitem_177 = mul_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_199: "f32[240]" = torch.ops.aten.add.Tensor(primals_239, 1e-05);  primals_239 = None
    rsqrt_36: "f32[240]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    unsqueeze_824: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_238, 0);  primals_238 = None
    unsqueeze_825: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
    unsqueeze_826: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
    sum_86: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_658, [0, 2, 3])
    sub_134: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_826);  convolution_20 = unsqueeze_826 = None
    mul_659: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_658, sub_134);  sub_134 = None
    sum_87: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_659, [0, 2, 3]);  mul_659 = None
    mul_664: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_36, primals_25);  primals_25 = None
    unsqueeze_833: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
    unsqueeze_834: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    mul_665: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_835);  mul_658 = unsqueeze_835 = None
    mul_666: "f32[240]" = torch.ops.aten.mul.Tensor(sum_87, rsqrt_36);  sum_87 = rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_665, add_24, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_665 = add_24 = primals_127 = None
    getitem_180: "f32[4, 40, 28, 28]" = convolution_backward_60[0]
    getitem_181: "f32[240, 40, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_200: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_165, getitem_180);  getitem_165 = getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_201: "f32[40]" = torch.ops.aten.add.Tensor(primals_237, 1e-05);  primals_237 = None
    rsqrt_37: "f32[40]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
    unsqueeze_836: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(primals_236, 0);  primals_236 = None
    unsqueeze_837: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
    unsqueeze_838: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
    sum_88: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_200, [0, 2, 3])
    sub_135: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_838);  convolution_19 = unsqueeze_838 = None
    mul_667: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_200, sub_135);  sub_135 = None
    sum_89: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 2, 3]);  mul_667 = None
    mul_672: "f32[40]" = torch.ops.aten.mul.Tensor(rsqrt_37, primals_23);  primals_23 = None
    unsqueeze_845: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_846: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    mul_673: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_200, unsqueeze_847);  add_200 = unsqueeze_847 = None
    mul_674: "f32[40]" = torch.ops.aten.mul.Tensor(sum_89, rsqrt_37);  sum_89 = rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_673, mul_48, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_673 = mul_48 = primals_126 = None
    getitem_183: "f32[4, 144, 28, 28]" = convolution_backward_61[0]
    getitem_184: "f32[40, 144, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_675: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_183, mul_46);  mul_46 = None
    mul_676: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_183, sigmoid_15);  getitem_183 = None
    sum_90: "f32[4, 144, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_675, [2, 3], True);  mul_675 = None
    sub_136: "f32[4, 144, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_15)
    mul_677: "f32[4, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_15, sub_136);  sigmoid_15 = sub_136 = None
    mul_678: "f32[4, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sum_90, mul_677);  sum_90 = mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_678, mul_47, primals_124, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_678 = mul_47 = primals_124 = None
    getitem_186: "f32[4, 6, 1, 1]" = convolution_backward_62[0]
    getitem_187: "f32[144, 6, 1, 1]" = convolution_backward_62[1]
    getitem_188: "f32[144]" = convolution_backward_62[2];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_102: "f32[4, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17)
    full_default_37: "f32[4, 6, 1, 1]" = torch.ops.aten.full.default([4, 6, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_137: "f32[4, 6, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_102)
    mul_679: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_17, sub_137);  convolution_17 = sub_137 = None
    add_202: "f32[4, 6, 1, 1]" = torch.ops.aten.add.Scalar(mul_679, 1);  mul_679 = None
    mul_680: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_102, add_202);  sigmoid_102 = add_202 = None
    mul_681: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_186, mul_680);  getitem_186 = mul_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_681, mean_3, primals_122, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_681 = mean_3 = primals_122 = None
    getitem_189: "f32[4, 144, 1, 1]" = convolution_backward_63[0]
    getitem_190: "f32[6, 144, 1, 1]" = convolution_backward_63[1]
    getitem_191: "f32[6]" = convolution_backward_63[2];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_13: "f32[4, 144, 28, 28]" = torch.ops.aten.expand.default(getitem_189, [4, 144, 28, 28]);  getitem_189 = None
    div_13: "f32[4, 144, 28, 28]" = torch.ops.aten.div.Scalar(expand_13, 784);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_203: "f32[4, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_676, div_13);  mul_676 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_103: "f32[4, 144, 28, 28]" = torch.ops.aten.sigmoid.default(add_22)
    full_default_38: "f32[4, 144, 28, 28]" = torch.ops.aten.full.default([4, 144, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_138: "f32[4, 144, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_38, sigmoid_103);  full_default_38 = None
    mul_682: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(add_22, sub_138);  add_22 = sub_138 = None
    add_204: "f32[4, 144, 28, 28]" = torch.ops.aten.add.Scalar(mul_682, 1);  mul_682 = None
    mul_683: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_103, add_204);  sigmoid_103 = add_204 = None
    mul_684: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(add_203, mul_683);  add_203 = mul_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_205: "f32[144]" = torch.ops.aten.add.Tensor(primals_235, 1e-05);  primals_235 = None
    rsqrt_38: "f32[144]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    unsqueeze_848: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(primals_234, 0);  primals_234 = None
    unsqueeze_849: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
    unsqueeze_850: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
    sum_91: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_684, [0, 2, 3])
    sub_139: "f32[4, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_850);  convolution_16 = unsqueeze_850 = None
    mul_685: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_684, sub_139);  sub_139 = None
    sum_92: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_685, [0, 2, 3]);  mul_685 = None
    mul_690: "f32[144]" = torch.ops.aten.mul.Tensor(rsqrt_38, primals_21);  primals_21 = None
    unsqueeze_857: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
    unsqueeze_858: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    mul_691: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_684, unsqueeze_859);  mul_684 = unsqueeze_859 = None
    mul_692: "f32[144]" = torch.ops.aten.mul.Tensor(sum_92, rsqrt_38);  sum_92 = rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_691, mul_42, primals_121, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_691 = mul_42 = primals_121 = None
    getitem_192: "f32[4, 144, 56, 56]" = convolution_backward_64[0]
    getitem_193: "f32[144, 1, 5, 5]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_39: "f32[4, 144, 56, 56]" = torch.ops.aten.full.default([4, 144, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    mul_695: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_192, mul_694);  getitem_192 = mul_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_207: "f32[144]" = torch.ops.aten.add.Tensor(primals_233, 1e-05);  primals_233 = None
    rsqrt_39: "f32[144]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
    unsqueeze_860: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(primals_232, 0);  primals_232 = None
    unsqueeze_861: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
    unsqueeze_862: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None
    sum_93: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_695, [0, 2, 3])
    sub_141: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_862);  convolution_15 = unsqueeze_862 = None
    mul_696: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_695, sub_141);  sub_141 = None
    sum_94: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_696, [0, 2, 3]);  mul_696 = None
    mul_701: "f32[144]" = torch.ops.aten.mul.Tensor(rsqrt_39, primals_19);  primals_19 = None
    unsqueeze_869: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
    unsqueeze_870: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    mul_702: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_695, unsqueeze_871);  mul_695 = unsqueeze_871 = None
    mul_703: "f32[144]" = torch.ops.aten.mul.Tensor(sum_94, rsqrt_39);  sum_94 = rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_702, add_18, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_702 = add_18 = primals_120 = None
    getitem_195: "f32[4, 24, 56, 56]" = convolution_backward_65[0]
    getitem_196: "f32[144, 24, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_208: "f32[24]" = torch.ops.aten.add.Tensor(primals_231, 1e-05);  primals_231 = None
    rsqrt_40: "f32[24]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    unsqueeze_872: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_230, 0);  primals_230 = None
    unsqueeze_873: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
    unsqueeze_874: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None
    sum_95: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_195, [0, 2, 3])
    sub_142: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_874);  convolution_14 = unsqueeze_874 = None
    mul_704: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_195, sub_142);  sub_142 = None
    sum_96: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 2, 3]);  mul_704 = None
    mul_709: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_40, primals_17);  primals_17 = None
    unsqueeze_881: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
    unsqueeze_882: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    mul_710: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_195, unsqueeze_883);  unsqueeze_883 = None
    mul_711: "f32[24]" = torch.ops.aten.mul.Tensor(sum_96, rsqrt_40);  sum_96 = rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_710, mul_35, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_710 = mul_35 = primals_119 = None
    getitem_198: "f32[4, 144, 56, 56]" = convolution_backward_66[0]
    getitem_199: "f32[24, 144, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_712: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_198, mul_33);  mul_33 = None
    mul_713: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_198, sigmoid_11);  getitem_198 = None
    sum_97: "f32[4, 144, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_712, [2, 3], True);  mul_712 = None
    sub_143: "f32[4, 144, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_11)
    mul_714: "f32[4, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_11, sub_143);  sigmoid_11 = sub_143 = None
    mul_715: "f32[4, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sum_97, mul_714);  sum_97 = mul_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_715, mul_34, primals_117, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_715 = mul_34 = primals_117 = None
    getitem_201: "f32[4, 6, 1, 1]" = convolution_backward_67[0]
    getitem_202: "f32[144, 6, 1, 1]" = convolution_backward_67[1]
    getitem_203: "f32[144]" = convolution_backward_67[2];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_105: "f32[4, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_12)
    sub_144: "f32[4, 6, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_105);  full_default_37 = None
    mul_716: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_12, sub_144);  convolution_12 = sub_144 = None
    add_209: "f32[4, 6, 1, 1]" = torch.ops.aten.add.Scalar(mul_716, 1);  mul_716 = None
    mul_717: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_105, add_209);  sigmoid_105 = add_209 = None
    mul_718: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_201, mul_717);  getitem_201 = mul_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_718, mean_2, primals_115, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_718 = mean_2 = primals_115 = None
    getitem_204: "f32[4, 144, 1, 1]" = convolution_backward_68[0]
    getitem_205: "f32[6, 144, 1, 1]" = convolution_backward_68[1]
    getitem_206: "f32[6]" = convolution_backward_68[2];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_14: "f32[4, 144, 56, 56]" = torch.ops.aten.expand.default(getitem_204, [4, 144, 56, 56]);  getitem_204 = None
    div_14: "f32[4, 144, 56, 56]" = torch.ops.aten.div.Scalar(expand_14, 3136);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_210: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_713, div_14);  mul_713 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_106: "f32[4, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_15)
    sub_145: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(full_default_39, sigmoid_106);  full_default_39 = None
    mul_719: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_15, sub_145);  add_15 = sub_145 = None
    add_211: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Scalar(mul_719, 1);  mul_719 = None
    mul_720: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_106, add_211);  sigmoid_106 = add_211 = None
    mul_721: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_210, mul_720);  add_210 = mul_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_212: "f32[144]" = torch.ops.aten.add.Tensor(primals_229, 1e-05);  primals_229 = None
    rsqrt_41: "f32[144]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
    unsqueeze_884: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(primals_228, 0);  primals_228 = None
    unsqueeze_885: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
    unsqueeze_886: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
    sum_98: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 2, 3])
    sub_146: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_886);  convolution_11 = unsqueeze_886 = None
    mul_722: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_721, sub_146);  sub_146 = None
    sum_99: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_722, [0, 2, 3]);  mul_722 = None
    mul_727: "f32[144]" = torch.ops.aten.mul.Tensor(rsqrt_41, primals_15);  primals_15 = None
    unsqueeze_893: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
    unsqueeze_894: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    mul_728: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_721, unsqueeze_895);  mul_721 = unsqueeze_895 = None
    mul_729: "f32[144]" = torch.ops.aten.mul.Tensor(sum_99, rsqrt_41);  sum_99 = rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_728, mul_29, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_728 = mul_29 = primals_114 = None
    getitem_207: "f32[4, 144, 56, 56]" = convolution_backward_69[0]
    getitem_208: "f32[144, 1, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_732: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_207, mul_731);  getitem_207 = mul_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_214: "f32[144]" = torch.ops.aten.add.Tensor(primals_227, 1e-05);  primals_227 = None
    rsqrt_42: "f32[144]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
    unsqueeze_896: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(primals_226, 0);  primals_226 = None
    unsqueeze_897: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
    unsqueeze_898: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None
    sum_100: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 2, 3])
    sub_148: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_898);  convolution_10 = unsqueeze_898 = None
    mul_733: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_732, sub_148);  sub_148 = None
    sum_101: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2, 3]);  mul_733 = None
    mul_738: "f32[144]" = torch.ops.aten.mul.Tensor(rsqrt_42, primals_13);  primals_13 = None
    unsqueeze_905: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_906: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    mul_739: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_732, unsqueeze_907);  mul_732 = unsqueeze_907 = None
    mul_740: "f32[144]" = torch.ops.aten.mul.Tensor(sum_101, rsqrt_42);  sum_101 = rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_739, add_11, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_739 = add_11 = primals_113 = None
    getitem_210: "f32[4, 24, 56, 56]" = convolution_backward_70[0]
    getitem_211: "f32[144, 24, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_215: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_195, getitem_210);  getitem_195 = getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_216: "f32[24]" = torch.ops.aten.add.Tensor(primals_225, 1e-05);  primals_225 = None
    rsqrt_43: "f32[24]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
    unsqueeze_908: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_224, 0);  primals_224 = None
    unsqueeze_909: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
    unsqueeze_910: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None
    sum_102: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_215, [0, 2, 3])
    sub_149: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_910);  convolution_9 = unsqueeze_910 = None
    mul_741: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_215, sub_149);  sub_149 = None
    sum_103: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_741, [0, 2, 3]);  mul_741 = None
    mul_746: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_43, primals_11);  primals_11 = None
    unsqueeze_917: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
    unsqueeze_918: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    mul_747: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_215, unsqueeze_919);  add_215 = unsqueeze_919 = None
    mul_748: "f32[24]" = torch.ops.aten.mul.Tensor(sum_103, rsqrt_43);  sum_103 = rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_747, mul_22, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_747 = mul_22 = primals_112 = None
    getitem_213: "f32[4, 96, 56, 56]" = convolution_backward_71[0]
    getitem_214: "f32[24, 96, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_749: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_213, mul_20);  mul_20 = None
    mul_750: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_213, sigmoid_7);  getitem_213 = None
    sum_104: "f32[4, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2, 3], True);  mul_749 = None
    sub_150: "f32[4, 96, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_7)
    mul_751: "f32[4, 96, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_7, sub_150);  sigmoid_7 = sub_150 = None
    mul_752: "f32[4, 96, 1, 1]" = torch.ops.aten.mul.Tensor(sum_104, mul_751);  sum_104 = mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_752, mul_21, primals_110, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_752 = mul_21 = primals_110 = None
    getitem_216: "f32[4, 4, 1, 1]" = convolution_backward_72[0]
    getitem_217: "f32[96, 4, 1, 1]" = convolution_backward_72[1]
    getitem_218: "f32[96]" = convolution_backward_72[2];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_108: "f32[4, 4, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_7)
    full_default_43: "f32[4, 4, 1, 1]" = torch.ops.aten.full.default([4, 4, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_151: "f32[4, 4, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_43, sigmoid_108);  full_default_43 = None
    mul_753: "f32[4, 4, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_7, sub_151);  convolution_7 = sub_151 = None
    add_217: "f32[4, 4, 1, 1]" = torch.ops.aten.add.Scalar(mul_753, 1);  mul_753 = None
    mul_754: "f32[4, 4, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_108, add_217);  sigmoid_108 = add_217 = None
    mul_755: "f32[4, 4, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_216, mul_754);  getitem_216 = mul_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_755, mean_1, primals_108, [4], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_755 = mean_1 = primals_108 = None
    getitem_219: "f32[4, 96, 1, 1]" = convolution_backward_73[0]
    getitem_220: "f32[4, 96, 1, 1]" = convolution_backward_73[1]
    getitem_221: "f32[4]" = convolution_backward_73[2];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_15: "f32[4, 96, 56, 56]" = torch.ops.aten.expand.default(getitem_219, [4, 96, 56, 56]);  getitem_219 = None
    div_15: "f32[4, 96, 56, 56]" = torch.ops.aten.div.Scalar(expand_15, 3136);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_218: "f32[4, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_750, div_15);  mul_750 = div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_109: "f32[4, 96, 56, 56]" = torch.ops.aten.sigmoid.default(add_9)
    full_default_44: "f32[4, 96, 56, 56]" = torch.ops.aten.full.default([4, 96, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_152: "f32[4, 96, 56, 56]" = torch.ops.aten.sub.Tensor(full_default_44, sigmoid_109);  full_default_44 = None
    mul_756: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_9, sub_152);  add_9 = sub_152 = None
    add_219: "f32[4, 96, 56, 56]" = torch.ops.aten.add.Scalar(mul_756, 1);  mul_756 = None
    mul_757: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_109, add_219);  sigmoid_109 = add_219 = None
    mul_758: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_218, mul_757);  add_218 = mul_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_220: "f32[96]" = torch.ops.aten.add.Tensor(primals_223, 1e-05);  primals_223 = None
    rsqrt_44: "f32[96]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
    unsqueeze_920: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_222, 0);  primals_222 = None
    unsqueeze_921: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 2);  unsqueeze_920 = None
    unsqueeze_922: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 3);  unsqueeze_921 = None
    sum_105: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_758, [0, 2, 3])
    sub_153: "f32[4, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_922);  convolution_6 = unsqueeze_922 = None
    mul_759: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_758, sub_153);  sub_153 = None
    sum_106: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_759, [0, 2, 3]);  mul_759 = None
    mul_764: "f32[96]" = torch.ops.aten.mul.Tensor(rsqrt_44, primals_9);  primals_9 = None
    unsqueeze_929: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_930: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
    unsqueeze_931: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
    mul_765: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_758, unsqueeze_931);  mul_758 = unsqueeze_931 = None
    mul_766: "f32[96]" = torch.ops.aten.mul.Tensor(sum_106, rsqrt_44);  sum_106 = rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_765, mul_16, primals_107, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False]);  mul_765 = mul_16 = primals_107 = None
    getitem_222: "f32[4, 96, 112, 112]" = convolution_backward_74[0]
    getitem_223: "f32[96, 1, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_769: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_222, mul_768);  getitem_222 = mul_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_222: "f32[96]" = torch.ops.aten.add.Tensor(primals_221, 1e-05);  primals_221 = None
    rsqrt_45: "f32[96]" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
    unsqueeze_932: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_220, 0);  primals_220 = None
    unsqueeze_933: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 2);  unsqueeze_932 = None
    unsqueeze_934: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 3);  unsqueeze_933 = None
    sum_107: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3])
    sub_155: "f32[4, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_934);  convolution_5 = unsqueeze_934 = None
    mul_770: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_769, sub_155);  sub_155 = None
    sum_108: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 2, 3]);  mul_770 = None
    mul_775: "f32[96]" = torch.ops.aten.mul.Tensor(rsqrt_45, primals_7);  primals_7 = None
    unsqueeze_941: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
    unsqueeze_942: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
    unsqueeze_943: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
    mul_776: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_769, unsqueeze_943);  mul_769 = unsqueeze_943 = None
    mul_777: "f32[96]" = torch.ops.aten.mul.Tensor(sum_108, rsqrt_45);  sum_108 = rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_776, add_5, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_776 = add_5 = primals_106 = None
    getitem_225: "f32[4, 16, 112, 112]" = convolution_backward_75[0]
    getitem_226: "f32[96, 16, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_223: "f32[16]" = torch.ops.aten.add.Tensor(primals_219, 1e-05);  primals_219 = None
    rsqrt_46: "f32[16]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
    unsqueeze_944: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(primals_218, 0);  primals_218 = None
    unsqueeze_945: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 2);  unsqueeze_944 = None
    unsqueeze_946: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 3);  unsqueeze_945 = None
    sum_109: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_225, [0, 2, 3])
    sub_156: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_946);  convolution_4 = unsqueeze_946 = None
    mul_778: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_225, sub_156);  sub_156 = None
    sum_110: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 2, 3]);  mul_778 = None
    mul_783: "f32[16]" = torch.ops.aten.mul.Tensor(rsqrt_46, primals_5);  primals_5 = None
    unsqueeze_953: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
    unsqueeze_954: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
    unsqueeze_955: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
    mul_784: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_225, unsqueeze_955);  getitem_225 = unsqueeze_955 = None
    mul_785: "f32[16]" = torch.ops.aten.mul.Tensor(sum_110, rsqrt_46);  sum_110 = rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_784, mul_9, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_784 = mul_9 = primals_105 = None
    getitem_228: "f32[4, 32, 112, 112]" = convolution_backward_76[0]
    getitem_229: "f32[16, 32, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_786: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_228, mul_7);  mul_7 = None
    mul_787: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_228, sigmoid_3);  getitem_228 = None
    sum_111: "f32[4, 32, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_786, [2, 3], True);  mul_786 = None
    sub_157: "f32[4, 32, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_3)
    mul_788: "f32[4, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_3, sub_157);  sigmoid_3 = sub_157 = None
    mul_789: "f32[4, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sum_111, mul_788);  sum_111 = mul_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_789, mul_8, primals_103, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_789 = mul_8 = primals_103 = None
    getitem_231: "f32[4, 8, 1, 1]" = convolution_backward_77[0]
    getitem_232: "f32[32, 8, 1, 1]" = convolution_backward_77[1]
    getitem_233: "f32[32]" = convolution_backward_77[2];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_111: "f32[4, 8, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_2)
    full_default_46: "f32[4, 8, 1, 1]" = torch.ops.aten.full.default([4, 8, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_158: "f32[4, 8, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_46, sigmoid_111);  full_default_46 = None
    mul_790: "f32[4, 8, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_2, sub_158);  convolution_2 = sub_158 = None
    add_224: "f32[4, 8, 1, 1]" = torch.ops.aten.add.Scalar(mul_790, 1);  mul_790 = None
    mul_791: "f32[4, 8, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_111, add_224);  sigmoid_111 = add_224 = None
    mul_792: "f32[4, 8, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_231, mul_791);  getitem_231 = mul_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_792, mean, primals_101, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_792 = mean = primals_101 = None
    getitem_234: "f32[4, 32, 1, 1]" = convolution_backward_78[0]
    getitem_235: "f32[8, 32, 1, 1]" = convolution_backward_78[1]
    getitem_236: "f32[8]" = convolution_backward_78[2];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_16: "f32[4, 32, 112, 112]" = torch.ops.aten.expand.default(getitem_234, [4, 32, 112, 112]);  getitem_234 = None
    div_16: "f32[4, 32, 112, 112]" = torch.ops.aten.div.Scalar(expand_16, 12544);  expand_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_225: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_787, div_16);  mul_787 = div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_112: "f32[4, 32, 112, 112]" = torch.ops.aten.sigmoid.default(add_3)
    full_default_47: "f32[4, 32, 112, 112]" = torch.ops.aten.full.default([4, 32, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_159: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(full_default_47, sigmoid_112);  full_default_47 = None
    mul_793: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_3, sub_159);  add_3 = sub_159 = None
    add_226: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Scalar(mul_793, 1);  mul_793 = None
    mul_794: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_112, add_226);  sigmoid_112 = add_226 = None
    mul_795: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_225, mul_794);  add_225 = mul_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_227: "f32[32]" = torch.ops.aten.add.Tensor(primals_217, 1e-05);  primals_217 = None
    rsqrt_47: "f32[32]" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
    unsqueeze_956: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_216, 0);  primals_216 = None
    unsqueeze_957: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 2);  unsqueeze_956 = None
    unsqueeze_958: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 3);  unsqueeze_957 = None
    sum_112: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_795, [0, 2, 3])
    sub_160: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_958);  convolution_1 = unsqueeze_958 = None
    mul_796: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_795, sub_160);  sub_160 = None
    sum_113: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_796, [0, 2, 3]);  mul_796 = None
    mul_801: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_47, primals_3);  primals_3 = None
    unsqueeze_965: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
    unsqueeze_966: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
    unsqueeze_967: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
    mul_802: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_795, unsqueeze_967);  mul_795 = unsqueeze_967 = None
    mul_803: "f32[32]" = torch.ops.aten.mul.Tensor(sum_113, rsqrt_47);  sum_113 = rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_802, mul_3, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_802 = mul_3 = primals_100 = None
    getitem_237: "f32[4, 32, 112, 112]" = convolution_backward_79[0]
    getitem_238: "f32[32, 1, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_806: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_237, mul_805);  getitem_237 = mul_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_229: "f32[32]" = torch.ops.aten.add.Tensor(primals_215, 1e-05);  primals_215 = None
    rsqrt_48: "f32[32]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
    unsqueeze_968: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_214, 0);  primals_214 = None
    unsqueeze_969: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
    unsqueeze_970: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None
    sum_114: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_806, [0, 2, 3])
    sub_162: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_970);  convolution = unsqueeze_970 = None
    mul_807: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_806, sub_162);  sub_162 = None
    sum_115: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_807, [0, 2, 3]);  mul_807 = None
    mul_812: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_48, primals_1);  primals_1 = None
    unsqueeze_977: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    unsqueeze_978: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    mul_813: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_806, unsqueeze_979);  mul_806 = unsqueeze_979 = None
    mul_814: "f32[32]" = torch.ops.aten.mul.Tensor(sum_115, rsqrt_48);  sum_115 = rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_813, primals_312, primals_99, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_813 = primals_312 = primals_99 = None
    getitem_241: "f32[32, 3, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None
    return [mul_814, sum_114, mul_803, sum_112, mul_785, sum_109, mul_777, sum_107, mul_766, sum_105, mul_748, sum_102, mul_740, sum_100, mul_729, sum_98, mul_711, sum_95, mul_703, sum_93, mul_692, sum_91, mul_674, sum_88, mul_666, sum_86, mul_655, sum_84, mul_637, sum_81, mul_629, sum_79, mul_618, sum_77, mul_600, sum_74, mul_592, sum_72, mul_581, sum_70, mul_563, sum_67, mul_555, sum_65, mul_544, sum_63, mul_526, sum_60, mul_518, sum_58, mul_507, sum_56, mul_489, sum_53, mul_481, sum_51, mul_470, sum_49, mul_452, sum_46, mul_444, sum_44, mul_433, sum_42, mul_415, sum_39, mul_407, sum_37, mul_396, sum_35, mul_378, sum_32, mul_370, sum_30, mul_359, sum_28, mul_341, sum_25, mul_333, sum_23, mul_322, sum_21, mul_304, sum_18, mul_296, sum_16, mul_285, sum_14, mul_267, sum_11, mul_259, sum_9, mul_248, sum_7, mul_230, sum_4, mul_222, sum_2, getitem_241, getitem_238, getitem_235, getitem_236, getitem_232, getitem_233, getitem_229, getitem_226, getitem_223, getitem_220, getitem_221, getitem_217, getitem_218, getitem_214, getitem_211, getitem_208, getitem_205, getitem_206, getitem_202, getitem_203, getitem_199, getitem_196, getitem_193, getitem_190, getitem_191, getitem_187, getitem_188, getitem_184, getitem_181, getitem_178, getitem_175, getitem_176, getitem_172, getitem_173, getitem_169, getitem_166, getitem_163, getitem_160, getitem_161, getitem_157, getitem_158, getitem_154, getitem_151, getitem_148, getitem_145, getitem_146, getitem_142, getitem_143, getitem_139, getitem_136, getitem_133, getitem_130, getitem_131, getitem_127, getitem_128, getitem_124, getitem_121, getitem_118, getitem_115, getitem_116, getitem_112, getitem_113, getitem_109, getitem_106, getitem_103, getitem_100, getitem_101, getitem_97, getitem_98, getitem_94, getitem_91, getitem_88, getitem_85, getitem_86, getitem_82, getitem_83, getitem_79, getitem_76, getitem_73, getitem_70, getitem_71, getitem_67, getitem_68, getitem_64, getitem_61, getitem_58, getitem_55, getitem_56, getitem_52, getitem_53, getitem_49, getitem_46, getitem_43, getitem_40, getitem_41, getitem_37, getitem_38, getitem_34, getitem_31, getitem_28, getitem_25, getitem_26, getitem_22, getitem_23, getitem_19, getitem_16, getitem_13, getitem_10, getitem_11, getitem_7, getitem_8, getitem_4, getitem_1, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    