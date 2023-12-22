from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_2: "f32[32]", primals_3: "f32[32]", primals_4: "f32[32]", primals_5: "f32[16]", primals_6: "f32[16]", primals_7: "f32[96]", primals_8: "f32[96]", primals_9: "f32[96]", primals_10: "f32[96]", primals_11: "f32[24]", primals_12: "f32[24]", primals_13: "f32[144]", primals_14: "f32[144]", primals_15: "f32[144]", primals_16: "f32[144]", primals_17: "f32[24]", primals_18: "f32[24]", primals_19: "f32[144]", primals_20: "f32[144]", primals_21: "f32[144]", primals_22: "f32[144]", primals_23: "f32[40]", primals_24: "f32[40]", primals_25: "f32[240]", primals_26: "f32[240]", primals_27: "f32[240]", primals_28: "f32[240]", primals_29: "f32[40]", primals_30: "f32[40]", primals_31: "f32[240]", primals_32: "f32[240]", primals_33: "f32[240]", primals_34: "f32[240]", primals_35: "f32[80]", primals_36: "f32[80]", primals_37: "f32[480]", primals_38: "f32[480]", primals_39: "f32[480]", primals_40: "f32[480]", primals_41: "f32[80]", primals_42: "f32[80]", primals_43: "f32[480]", primals_44: "f32[480]", primals_45: "f32[480]", primals_46: "f32[480]", primals_47: "f32[80]", primals_48: "f32[80]", primals_49: "f32[480]", primals_50: "f32[480]", primals_51: "f32[480]", primals_52: "f32[480]", primals_53: "f32[112]", primals_54: "f32[112]", primals_55: "f32[672]", primals_56: "f32[672]", primals_57: "f32[672]", primals_58: "f32[672]", primals_59: "f32[112]", primals_60: "f32[112]", primals_61: "f32[672]", primals_62: "f32[672]", primals_63: "f32[672]", primals_64: "f32[672]", primals_65: "f32[112]", primals_66: "f32[112]", primals_67: "f32[672]", primals_68: "f32[672]", primals_69: "f32[672]", primals_70: "f32[672]", primals_71: "f32[192]", primals_72: "f32[192]", primals_73: "f32[1152]", primals_74: "f32[1152]", primals_75: "f32[1152]", primals_76: "f32[1152]", primals_77: "f32[192]", primals_78: "f32[192]", primals_79: "f32[1152]", primals_80: "f32[1152]", primals_81: "f32[1152]", primals_82: "f32[1152]", primals_83: "f32[192]", primals_84: "f32[192]", primals_85: "f32[1152]", primals_86: "f32[1152]", primals_87: "f32[1152]", primals_88: "f32[1152]", primals_89: "f32[192]", primals_90: "f32[192]", primals_91: "f32[1152]", primals_92: "f32[1152]", primals_93: "f32[1152]", primals_94: "f32[1152]", primals_95: "f32[320]", primals_96: "f32[320]", primals_97: "f32[1280]", primals_98: "f32[1280]", primals_99: "f32[32, 3, 3, 3]", primals_100: "f32[32, 1, 3, 3]", primals_101: "f32[8, 32, 1, 1]", primals_102: "f32[8]", primals_103: "f32[32, 8, 1, 1]", primals_104: "f32[32]", primals_105: "f32[16, 32, 1, 1]", primals_106: "f32[96, 16, 1, 1]", primals_107: "f32[96, 1, 3, 3]", primals_108: "f32[4, 96, 1, 1]", primals_109: "f32[4]", primals_110: "f32[96, 4, 1, 1]", primals_111: "f32[96]", primals_112: "f32[24, 96, 1, 1]", primals_113: "f32[144, 24, 1, 1]", primals_114: "f32[144, 1, 3, 3]", primals_115: "f32[6, 144, 1, 1]", primals_116: "f32[6]", primals_117: "f32[144, 6, 1, 1]", primals_118: "f32[144]", primals_119: "f32[24, 144, 1, 1]", primals_120: "f32[144, 24, 1, 1]", primals_121: "f32[144, 1, 5, 5]", primals_122: "f32[6, 144, 1, 1]", primals_123: "f32[6]", primals_124: "f32[144, 6, 1, 1]", primals_125: "f32[144]", primals_126: "f32[40, 144, 1, 1]", primals_127: "f32[240, 40, 1, 1]", primals_128: "f32[240, 1, 5, 5]", primals_129: "f32[10, 240, 1, 1]", primals_130: "f32[10]", primals_131: "f32[240, 10, 1, 1]", primals_132: "f32[240]", primals_133: "f32[40, 240, 1, 1]", primals_134: "f32[240, 40, 1, 1]", primals_135: "f32[240, 1, 3, 3]", primals_136: "f32[10, 240, 1, 1]", primals_137: "f32[10]", primals_138: "f32[240, 10, 1, 1]", primals_139: "f32[240]", primals_140: "f32[80, 240, 1, 1]", primals_141: "f32[480, 80, 1, 1]", primals_142: "f32[480, 1, 3, 3]", primals_143: "f32[20, 480, 1, 1]", primals_144: "f32[20]", primals_145: "f32[480, 20, 1, 1]", primals_146: "f32[480]", primals_147: "f32[80, 480, 1, 1]", primals_148: "f32[480, 80, 1, 1]", primals_149: "f32[480, 1, 3, 3]", primals_150: "f32[20, 480, 1, 1]", primals_151: "f32[20]", primals_152: "f32[480, 20, 1, 1]", primals_153: "f32[480]", primals_154: "f32[80, 480, 1, 1]", primals_155: "f32[480, 80, 1, 1]", primals_156: "f32[480, 1, 5, 5]", primals_157: "f32[20, 480, 1, 1]", primals_158: "f32[20]", primals_159: "f32[480, 20, 1, 1]", primals_160: "f32[480]", primals_161: "f32[112, 480, 1, 1]", primals_162: "f32[672, 112, 1, 1]", primals_163: "f32[672, 1, 5, 5]", primals_164: "f32[28, 672, 1, 1]", primals_165: "f32[28]", primals_166: "f32[672, 28, 1, 1]", primals_167: "f32[672]", primals_168: "f32[112, 672, 1, 1]", primals_169: "f32[672, 112, 1, 1]", primals_170: "f32[672, 1, 5, 5]", primals_171: "f32[28, 672, 1, 1]", primals_172: "f32[28]", primals_173: "f32[672, 28, 1, 1]", primals_174: "f32[672]", primals_175: "f32[112, 672, 1, 1]", primals_176: "f32[672, 112, 1, 1]", primals_177: "f32[672, 1, 5, 5]", primals_178: "f32[28, 672, 1, 1]", primals_179: "f32[28]", primals_180: "f32[672, 28, 1, 1]", primals_181: "f32[672]", primals_182: "f32[192, 672, 1, 1]", primals_183: "f32[1152, 192, 1, 1]", primals_184: "f32[1152, 1, 5, 5]", primals_185: "f32[48, 1152, 1, 1]", primals_186: "f32[48]", primals_187: "f32[1152, 48, 1, 1]", primals_188: "f32[1152]", primals_189: "f32[192, 1152, 1, 1]", primals_190: "f32[1152, 192, 1, 1]", primals_191: "f32[1152, 1, 5, 5]", primals_192: "f32[48, 1152, 1, 1]", primals_193: "f32[48]", primals_194: "f32[1152, 48, 1, 1]", primals_195: "f32[1152]", primals_196: "f32[192, 1152, 1, 1]", primals_197: "f32[1152, 192, 1, 1]", primals_198: "f32[1152, 1, 5, 5]", primals_199: "f32[48, 1152, 1, 1]", primals_200: "f32[48]", primals_201: "f32[1152, 48, 1, 1]", primals_202: "f32[1152]", primals_203: "f32[192, 1152, 1, 1]", primals_204: "f32[1152, 192, 1, 1]", primals_205: "f32[1152, 1, 3, 3]", primals_206: "f32[48, 1152, 1, 1]", primals_207: "f32[48]", primals_208: "f32[1152, 48, 1, 1]", primals_209: "f32[1152]", primals_210: "f32[320, 1152, 1, 1]", primals_211: "f32[1280, 320, 1, 1]", primals_212: "f32[1000, 1280]", primals_213: "f32[1000]", primals_214: "f32[32]", primals_215: "f32[32]", primals_216: "f32[32]", primals_217: "f32[32]", primals_218: "f32[16]", primals_219: "f32[16]", primals_220: "f32[96]", primals_221: "f32[96]", primals_222: "f32[96]", primals_223: "f32[96]", primals_224: "f32[24]", primals_225: "f32[24]", primals_226: "f32[144]", primals_227: "f32[144]", primals_228: "f32[144]", primals_229: "f32[144]", primals_230: "f32[24]", primals_231: "f32[24]", primals_232: "f32[144]", primals_233: "f32[144]", primals_234: "f32[144]", primals_235: "f32[144]", primals_236: "f32[40]", primals_237: "f32[40]", primals_238: "f32[240]", primals_239: "f32[240]", primals_240: "f32[240]", primals_241: "f32[240]", primals_242: "f32[40]", primals_243: "f32[40]", primals_244: "f32[240]", primals_245: "f32[240]", primals_246: "f32[240]", primals_247: "f32[240]", primals_248: "f32[80]", primals_249: "f32[80]", primals_250: "f32[480]", primals_251: "f32[480]", primals_252: "f32[480]", primals_253: "f32[480]", primals_254: "f32[80]", primals_255: "f32[80]", primals_256: "f32[480]", primals_257: "f32[480]", primals_258: "f32[480]", primals_259: "f32[480]", primals_260: "f32[80]", primals_261: "f32[80]", primals_262: "f32[480]", primals_263: "f32[480]", primals_264: "f32[480]", primals_265: "f32[480]", primals_266: "f32[112]", primals_267: "f32[112]", primals_268: "f32[672]", primals_269: "f32[672]", primals_270: "f32[672]", primals_271: "f32[672]", primals_272: "f32[112]", primals_273: "f32[112]", primals_274: "f32[672]", primals_275: "f32[672]", primals_276: "f32[672]", primals_277: "f32[672]", primals_278: "f32[112]", primals_279: "f32[112]", primals_280: "f32[672]", primals_281: "f32[672]", primals_282: "f32[672]", primals_283: "f32[672]", primals_284: "f32[192]", primals_285: "f32[192]", primals_286: "f32[1152]", primals_287: "f32[1152]", primals_288: "f32[1152]", primals_289: "f32[1152]", primals_290: "f32[192]", primals_291: "f32[192]", primals_292: "f32[1152]", primals_293: "f32[1152]", primals_294: "f32[1152]", primals_295: "f32[1152]", primals_296: "f32[192]", primals_297: "f32[192]", primals_298: "f32[1152]", primals_299: "f32[1152]", primals_300: "f32[1152]", primals_301: "f32[1152]", primals_302: "f32[192]", primals_303: "f32[192]", primals_304: "f32[1152]", primals_305: "f32[1152]", primals_306: "f32[1152]", primals_307: "f32[1152]", primals_308: "f32[320]", primals_309: "f32[320]", primals_310: "f32[1280]", primals_311: "f32[1280]", primals_312: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution: "f32[4, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_312, primals_99, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add: "f32[32]" = torch.ops.aten.add.Tensor(primals_215, 1e-05)
    sqrt: "f32[32]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_214, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  unsqueeze_1 = None
    mul_1: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid: "f32[4, 32, 112, 112]" = torch.ops.aten.sigmoid.default(add_1)
    mul_3: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_1, sigmoid);  sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_1: "f32[4, 32, 112, 112]" = torch.ops.aten.convolution.default(mul_3, primals_100, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(primals_217, 1e-05)
    sqrt_1: "f32[32]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_216, -1)
    unsqueeze_9: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_4, -1);  mul_4 = None
    unsqueeze_11: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
    mul_5: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_6: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_13);  mul_5 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_15);  mul_6 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_1: "f32[4, 32, 112, 112]" = torch.ops.aten.sigmoid.default(add_3)
    mul_7: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_3, sigmoid_1);  sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[4, 32, 1, 1]" = torch.ops.aten.mean.dim(mul_7, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_2: "f32[4, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_101, primals_102, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_2: "f32[4, 8, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_2)
    mul_8: "f32[4, 8, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_2, sigmoid_2);  sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_3: "f32[4, 32, 1, 1]" = torch.ops.aten.convolution.default(mul_8, primals_103, primals_104, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_3: "f32[4, 32, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_3)
    mul_9: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, sigmoid_3);  mul_7 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_4: "f32[4, 16, 112, 112]" = torch.ops.aten.convolution.default(mul_9, primals_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_4: "f32[16]" = torch.ops.aten.add.Tensor(primals_219, 1e-05)
    sqrt_2: "f32[16]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_10: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_218, -1)
    unsqueeze_17: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_10, -1);  mul_10 = None
    unsqueeze_19: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_17);  unsqueeze_17 = None
    mul_11: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_21: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_12: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_11, unsqueeze_21);  mul_11 = unsqueeze_21 = None
    unsqueeze_22: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_23: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_12, unsqueeze_23);  mul_12 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_5: "f32[4, 96, 112, 112]" = torch.ops.aten.convolution.default(add_5, primals_106, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_6: "f32[96]" = torch.ops.aten.add.Tensor(primals_221, 1e-05)
    sqrt_3: "f32[96]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_13: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_220, -1)
    unsqueeze_25: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_13, -1);  mul_13 = None
    unsqueeze_27: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_25);  unsqueeze_25 = None
    mul_14: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_29: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_15: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_29);  mul_14 = unsqueeze_29 = None
    unsqueeze_30: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_31: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[4, 96, 112, 112]" = torch.ops.aten.add.Tensor(mul_15, unsqueeze_31);  mul_15 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_4: "f32[4, 96, 112, 112]" = torch.ops.aten.sigmoid.default(add_7)
    mul_16: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(add_7, sigmoid_4);  sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_6: "f32[4, 96, 56, 56]" = torch.ops.aten.convolution.default(mul_16, primals_107, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_8: "f32[96]" = torch.ops.aten.add.Tensor(primals_223, 1e-05)
    sqrt_4: "f32[96]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_17: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_222, -1)
    unsqueeze_33: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_17, -1);  mul_17 = None
    unsqueeze_35: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_33);  unsqueeze_33 = None
    mul_18: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_37: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_19: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_18, unsqueeze_37);  mul_18 = unsqueeze_37 = None
    unsqueeze_38: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_39: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[4, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_19, unsqueeze_39);  mul_19 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_5: "f32[4, 96, 56, 56]" = torch.ops.aten.sigmoid.default(add_9)
    mul_20: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_5);  sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[4, 96, 1, 1]" = torch.ops.aten.mean.dim(mul_20, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_7: "f32[4, 4, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_108, primals_109, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_6: "f32[4, 4, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_7)
    mul_21: "f32[4, 4, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_7, sigmoid_6);  sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_8: "f32[4, 96, 1, 1]" = torch.ops.aten.convolution.default(mul_21, primals_110, primals_111, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_7: "f32[4, 96, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_8)
    mul_22: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_20, sigmoid_7);  mul_20 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_9: "f32[4, 24, 56, 56]" = torch.ops.aten.convolution.default(mul_22, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_10: "f32[24]" = torch.ops.aten.add.Tensor(primals_225, 1e-05)
    sqrt_5: "f32[24]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_23: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_224, -1)
    unsqueeze_41: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_23, -1);  mul_23 = None
    unsqueeze_43: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_41);  unsqueeze_41 = None
    mul_24: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_45: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_25: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_45);  mul_24 = unsqueeze_45 = None
    unsqueeze_46: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_47: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_25, unsqueeze_47);  mul_25 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_10: "f32[4, 144, 56, 56]" = torch.ops.aten.convolution.default(add_11, primals_113, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_12: "f32[144]" = torch.ops.aten.add.Tensor(primals_227, 1e-05)
    sqrt_6: "f32[144]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_26: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_226, -1)
    unsqueeze_49: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_26, -1);  mul_26 = None
    unsqueeze_51: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_49);  unsqueeze_49 = None
    mul_27: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_53: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_28: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_27, unsqueeze_53);  mul_27 = unsqueeze_53 = None
    unsqueeze_54: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_55: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_55);  mul_28 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_8: "f32[4, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_13)
    mul_29: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_13, sigmoid_8);  sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_11: "f32[4, 144, 56, 56]" = torch.ops.aten.convolution.default(mul_29, primals_114, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_14: "f32[144]" = torch.ops.aten.add.Tensor(primals_229, 1e-05)
    sqrt_7: "f32[144]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_7: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_30: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1)
    unsqueeze_57: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_59: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_57);  unsqueeze_57 = None
    mul_31: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_61: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_32: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_61);  mul_31 = unsqueeze_61 = None
    unsqueeze_62: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_63: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_15: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_63);  mul_32 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_9: "f32[4, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_15)
    mul_33: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_15, sigmoid_9);  sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[4, 144, 1, 1]" = torch.ops.aten.mean.dim(mul_33, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_12: "f32[4, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_115, primals_116, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_10: "f32[4, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_12)
    mul_34: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_12, sigmoid_10);  sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_13: "f32[4, 144, 1, 1]" = torch.ops.aten.convolution.default(mul_34, primals_117, primals_118, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_11: "f32[4, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_13)
    mul_35: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_33, sigmoid_11);  mul_33 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_14: "f32[4, 24, 56, 56]" = torch.ops.aten.convolution.default(mul_35, primals_119, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_16: "f32[24]" = torch.ops.aten.add.Tensor(primals_231, 1e-05)
    sqrt_8: "f32[24]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_8: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_36: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_230, -1)
    unsqueeze_65: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_67: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_65);  unsqueeze_65 = None
    mul_37: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_69: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_38: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_69);  mul_37 = unsqueeze_69 = None
    unsqueeze_70: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_71: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_17: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_71);  mul_38 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_18: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_17, add_11);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_15: "f32[4, 144, 56, 56]" = torch.ops.aten.convolution.default(add_18, primals_120, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_19: "f32[144]" = torch.ops.aten.add.Tensor(primals_233, 1e-05)
    sqrt_9: "f32[144]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    reciprocal_9: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_39: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_232, -1)
    unsqueeze_73: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_75: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_73);  unsqueeze_73 = None
    mul_40: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_77: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_41: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_77);  mul_40 = unsqueeze_77 = None
    unsqueeze_78: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_79: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_20: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_79);  mul_41 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_12: "f32[4, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_20)
    mul_42: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_20, sigmoid_12);  sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_16: "f32[4, 144, 28, 28]" = torch.ops.aten.convolution.default(mul_42, primals_121, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_21: "f32[144]" = torch.ops.aten.add.Tensor(primals_235, 1e-05)
    sqrt_10: "f32[144]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_10: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_43: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_234, -1)
    unsqueeze_81: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_43, -1);  mul_43 = None
    unsqueeze_83: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_81);  unsqueeze_81 = None
    mul_44: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_85: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_45: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_44, unsqueeze_85);  mul_44 = unsqueeze_85 = None
    unsqueeze_86: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_87: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_22: "f32[4, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_45, unsqueeze_87);  mul_45 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_13: "f32[4, 144, 28, 28]" = torch.ops.aten.sigmoid.default(add_22)
    mul_46: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(add_22, sigmoid_13);  sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[4, 144, 1, 1]" = torch.ops.aten.mean.dim(mul_46, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_17: "f32[4, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_122, primals_123, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_14: "f32[4, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17)
    mul_47: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_17, sigmoid_14);  sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_18: "f32[4, 144, 1, 1]" = torch.ops.aten.convolution.default(mul_47, primals_124, primals_125, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_15: "f32[4, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_18)
    mul_48: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_46, sigmoid_15);  mul_46 = sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_19: "f32[4, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_48, primals_126, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_23: "f32[40]" = torch.ops.aten.add.Tensor(primals_237, 1e-05)
    sqrt_11: "f32[40]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_11: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_49: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_236, -1)
    unsqueeze_89: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_49, -1);  mul_49 = None
    unsqueeze_91: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_89);  unsqueeze_89 = None
    mul_50: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_93: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_51: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_50, unsqueeze_93);  mul_50 = unsqueeze_93 = None
    unsqueeze_94: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_95: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_24: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_51, unsqueeze_95);  mul_51 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_20: "f32[4, 240, 28, 28]" = torch.ops.aten.convolution.default(add_24, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_25: "f32[240]" = torch.ops.aten.add.Tensor(primals_239, 1e-05)
    sqrt_12: "f32[240]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
    reciprocal_12: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_52: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_238, -1)
    unsqueeze_97: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_52, -1);  mul_52 = None
    unsqueeze_99: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_97);  unsqueeze_97 = None
    mul_53: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_101: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_54: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_53, unsqueeze_101);  mul_53 = unsqueeze_101 = None
    unsqueeze_102: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_103: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_26: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_54, unsqueeze_103);  mul_54 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_16: "f32[4, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_26)
    mul_55: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_26, sigmoid_16);  sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_21: "f32[4, 240, 28, 28]" = torch.ops.aten.convolution.default(mul_55, primals_128, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_27: "f32[240]" = torch.ops.aten.add.Tensor(primals_241, 1e-05)
    sqrt_13: "f32[240]" = torch.ops.aten.sqrt.default(add_27);  add_27 = None
    reciprocal_13: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_56: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_240, -1)
    unsqueeze_105: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_56, -1);  mul_56 = None
    unsqueeze_107: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_105);  unsqueeze_105 = None
    mul_57: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_109: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_58: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_57, unsqueeze_109);  mul_57 = unsqueeze_109 = None
    unsqueeze_110: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_111: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_28: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_58, unsqueeze_111);  mul_58 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_17: "f32[4, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_28)
    mul_59: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_28, sigmoid_17);  sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[4, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_59, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_22: "f32[4, 10, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_129, primals_130, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_18: "f32[4, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_22)
    mul_60: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_22, sigmoid_18);  sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_23: "f32[4, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_60, primals_131, primals_132, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_19: "f32[4, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23)
    mul_61: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_59, sigmoid_19);  mul_59 = sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_24: "f32[4, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_61, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_29: "f32[40]" = torch.ops.aten.add.Tensor(primals_243, 1e-05)
    sqrt_14: "f32[40]" = torch.ops.aten.sqrt.default(add_29);  add_29 = None
    reciprocal_14: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_62: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_242, -1)
    unsqueeze_113: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_62, -1);  mul_62 = None
    unsqueeze_115: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_113);  unsqueeze_113 = None
    mul_63: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_117: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_64: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_117);  mul_63 = unsqueeze_117 = None
    unsqueeze_118: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_119: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_30: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_119);  mul_64 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_31: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_30, add_24);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_25: "f32[4, 240, 28, 28]" = torch.ops.aten.convolution.default(add_31, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_32: "f32[240]" = torch.ops.aten.add.Tensor(primals_245, 1e-05)
    sqrt_15: "f32[240]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_15: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_65: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_244, -1)
    unsqueeze_121: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_65, -1);  mul_65 = None
    unsqueeze_123: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_121);  unsqueeze_121 = None
    mul_66: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_125: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_67: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_66, unsqueeze_125);  mul_66 = unsqueeze_125 = None
    unsqueeze_126: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_127: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_33: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_67, unsqueeze_127);  mul_67 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_20: "f32[4, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_33)
    mul_68: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_33, sigmoid_20);  sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_26: "f32[4, 240, 14, 14]" = torch.ops.aten.convolution.default(mul_68, primals_135, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_34: "f32[240]" = torch.ops.aten.add.Tensor(primals_247, 1e-05)
    sqrt_16: "f32[240]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_16: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_69: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_246, -1)
    unsqueeze_129: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_131: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_129);  unsqueeze_129 = None
    mul_70: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_133: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_71: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_133);  mul_70 = unsqueeze_133 = None
    unsqueeze_134: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_135: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_35: "f32[4, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_135);  mul_71 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_21: "f32[4, 240, 14, 14]" = torch.ops.aten.sigmoid.default(add_35)
    mul_72: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_35, sigmoid_21);  sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[4, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_72, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_27: "f32[4, 10, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_136, primals_137, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_22: "f32[4, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_27)
    mul_73: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_27, sigmoid_22);  sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_28: "f32[4, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_73, primals_138, primals_139, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_23: "f32[4, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_28)
    mul_74: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_72, sigmoid_23);  mul_72 = sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_29: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(mul_74, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_36: "f32[80]" = torch.ops.aten.add.Tensor(primals_249, 1e-05)
    sqrt_17: "f32[80]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_17: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_75: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_248, -1)
    unsqueeze_137: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_139: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_137);  unsqueeze_137 = None
    mul_76: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_141: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_77: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_141);  mul_76 = unsqueeze_141 = None
    unsqueeze_142: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_143: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_37: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_143);  mul_77 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_30: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(add_37, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_38: "f32[480]" = torch.ops.aten.add.Tensor(primals_251, 1e-05)
    sqrt_18: "f32[480]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_18: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_78: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_250, -1)
    unsqueeze_145: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
    unsqueeze_147: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_145);  unsqueeze_145 = None
    mul_79: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_149: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_80: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_149);  mul_79 = unsqueeze_149 = None
    unsqueeze_150: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_151: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_39: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_151);  mul_80 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_24: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_39)
    mul_81: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_39, sigmoid_24);  sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_31: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(mul_81, primals_142, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_40: "f32[480]" = torch.ops.aten.add.Tensor(primals_253, 1e-05)
    sqrt_19: "f32[480]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_19: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_82: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_252, -1)
    unsqueeze_153: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_82, -1);  mul_82 = None
    unsqueeze_155: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_153);  unsqueeze_153 = None
    mul_83: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_157: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_84: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_83, unsqueeze_157);  mul_83 = unsqueeze_157 = None
    unsqueeze_158: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_159: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_41: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_84, unsqueeze_159);  mul_84 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_25: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_41)
    mul_85: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_41, sigmoid_25);  sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[4, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_85, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_32: "f32[4, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_143, primals_144, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_26: "f32[4, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_32)
    mul_86: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_32, sigmoid_26);  sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_33: "f32[4, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_86, primals_145, primals_146, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_27: "f32[4, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_33)
    mul_87: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_85, sigmoid_27);  mul_85 = sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_34: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(mul_87, primals_147, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_42: "f32[80]" = torch.ops.aten.add.Tensor(primals_255, 1e-05)
    sqrt_20: "f32[80]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    reciprocal_20: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_88: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_254, -1)
    unsqueeze_161: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_88, -1);  mul_88 = None
    unsqueeze_163: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_161);  unsqueeze_161 = None
    mul_89: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_165: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_90: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_165);  mul_89 = unsqueeze_165 = None
    unsqueeze_166: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_167: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_43: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_167);  mul_90 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_44: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_43, add_37);  add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_35: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(add_44, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_45: "f32[480]" = torch.ops.aten.add.Tensor(primals_257, 1e-05)
    sqrt_21: "f32[480]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_21: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_91: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_256, -1)
    unsqueeze_169: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_91, -1);  mul_91 = None
    unsqueeze_171: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_169);  unsqueeze_169 = None
    mul_92: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_173: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_93: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_92, unsqueeze_173);  mul_92 = unsqueeze_173 = None
    unsqueeze_174: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_175: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_46: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_93, unsqueeze_175);  mul_93 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_28: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_46)
    mul_94: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_46, sigmoid_28);  sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_36: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(mul_94, primals_149, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_47: "f32[480]" = torch.ops.aten.add.Tensor(primals_259, 1e-05)
    sqrt_22: "f32[480]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_22: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_95: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_258, -1)
    unsqueeze_177: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_95, -1);  mul_95 = None
    unsqueeze_179: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_177);  unsqueeze_177 = None
    mul_96: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_181: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_97: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_181);  mul_96 = unsqueeze_181 = None
    unsqueeze_182: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_183: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_48: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_183);  mul_97 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_29: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_48)
    mul_98: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_48, sigmoid_29);  sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[4, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_98, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_37: "f32[4, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_150, primals_151, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_30: "f32[4, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_37)
    mul_99: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_37, sigmoid_30);  sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_38: "f32[4, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_99, primals_152, primals_153, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_31: "f32[4, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_38)
    mul_100: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_98, sigmoid_31);  mul_98 = sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_39: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(mul_100, primals_154, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_49: "f32[80]" = torch.ops.aten.add.Tensor(primals_261, 1e-05)
    sqrt_23: "f32[80]" = torch.ops.aten.sqrt.default(add_49);  add_49 = None
    reciprocal_23: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_101: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_260, -1)
    unsqueeze_185: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_101, -1);  mul_101 = None
    unsqueeze_187: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_185);  unsqueeze_185 = None
    mul_102: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_189: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_103: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_102, unsqueeze_189);  mul_102 = unsqueeze_189 = None
    unsqueeze_190: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_191: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_50: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_103, unsqueeze_191);  mul_103 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_51: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_50, add_44);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_40: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(add_51, primals_155, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_52: "f32[480]" = torch.ops.aten.add.Tensor(primals_263, 1e-05)
    sqrt_24: "f32[480]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_24: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_104: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_262, -1)
    unsqueeze_193: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_104, -1);  mul_104 = None
    unsqueeze_195: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_193);  unsqueeze_193 = None
    mul_105: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_197: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_106: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_197);  mul_105 = unsqueeze_197 = None
    unsqueeze_198: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_199: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_53: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_199);  mul_106 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_32: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_53)
    mul_107: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_53, sigmoid_32);  sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_41: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(mul_107, primals_156, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_54: "f32[480]" = torch.ops.aten.add.Tensor(primals_265, 1e-05)
    sqrt_25: "f32[480]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
    reciprocal_25: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_108: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_264, -1)
    unsqueeze_201: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_203: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_201);  unsqueeze_201 = None
    mul_109: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_205: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_110: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_205);  mul_109 = unsqueeze_205 = None
    unsqueeze_206: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_207: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_55: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_207);  mul_110 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_33: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_55)
    mul_111: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_55, sigmoid_33);  sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[4, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_111, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_42: "f32[4, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_157, primals_158, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_34: "f32[4, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42)
    mul_112: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_42, sigmoid_34);  sigmoid_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_43: "f32[4, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_112, primals_159, primals_160, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_35: "f32[4, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43)
    mul_113: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_111, sigmoid_35);  mul_111 = sigmoid_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_44: "f32[4, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_113, primals_161, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_56: "f32[112]" = torch.ops.aten.add.Tensor(primals_267, 1e-05)
    sqrt_26: "f32[112]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
    reciprocal_26: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_114: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_266, -1)
    unsqueeze_209: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_211: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_209);  unsqueeze_209 = None
    mul_115: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_213: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_116: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_213);  mul_115 = unsqueeze_213 = None
    unsqueeze_214: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_215: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_57: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_215);  mul_116 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_45: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(add_57, primals_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_58: "f32[672]" = torch.ops.aten.add.Tensor(primals_269, 1e-05)
    sqrt_27: "f32[672]" = torch.ops.aten.sqrt.default(add_58);  add_58 = None
    reciprocal_27: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_117: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_268, -1)
    unsqueeze_217: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
    unsqueeze_219: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_217);  unsqueeze_217 = None
    mul_118: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_221: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_119: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_221);  mul_118 = unsqueeze_221 = None
    unsqueeze_222: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_223: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_59: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_223);  mul_119 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_36: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_59)
    mul_120: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_59, sigmoid_36);  sigmoid_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_46: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(mul_120, primals_163, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_60: "f32[672]" = torch.ops.aten.add.Tensor(primals_271, 1e-05)
    sqrt_28: "f32[672]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_28: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_121: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_270, -1)
    unsqueeze_225: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_121, -1);  mul_121 = None
    unsqueeze_227: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_225);  unsqueeze_225 = None
    mul_122: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_229: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_123: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_122, unsqueeze_229);  mul_122 = unsqueeze_229 = None
    unsqueeze_230: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_231: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_61: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_123, unsqueeze_231);  mul_123 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_37: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_61)
    mul_124: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_61, sigmoid_37);  sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[4, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_124, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_47: "f32[4, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_164, primals_165, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_38: "f32[4, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_47)
    mul_125: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_47, sigmoid_38);  sigmoid_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_48: "f32[4, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_125, primals_166, primals_167, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_39: "f32[4, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48)
    mul_126: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_124, sigmoid_39);  mul_124 = sigmoid_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_49: "f32[4, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_126, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_62: "f32[112]" = torch.ops.aten.add.Tensor(primals_273, 1e-05)
    sqrt_29: "f32[112]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
    reciprocal_29: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_127: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_272, -1)
    unsqueeze_233: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_127, -1);  mul_127 = None
    unsqueeze_235: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_233);  unsqueeze_233 = None
    mul_128: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_237: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_129: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_128, unsqueeze_237);  mul_128 = unsqueeze_237 = None
    unsqueeze_238: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_239: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_63: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_129, unsqueeze_239);  mul_129 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_64: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_63, add_57);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_50: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(add_64, primals_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_65: "f32[672]" = torch.ops.aten.add.Tensor(primals_275, 1e-05)
    sqrt_30: "f32[672]" = torch.ops.aten.sqrt.default(add_65);  add_65 = None
    reciprocal_30: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_130: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_274, -1)
    unsqueeze_241: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_130, -1);  mul_130 = None
    unsqueeze_243: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_241);  unsqueeze_241 = None
    mul_131: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_245: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_132: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_245);  mul_131 = unsqueeze_245 = None
    unsqueeze_246: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_247: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_66: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_247);  mul_132 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_40: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_66)
    mul_133: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_66, sigmoid_40);  sigmoid_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_51: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(mul_133, primals_170, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_67: "f32[672]" = torch.ops.aten.add.Tensor(primals_277, 1e-05)
    sqrt_31: "f32[672]" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
    reciprocal_31: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_134: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_276, -1)
    unsqueeze_249: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_134, -1);  mul_134 = None
    unsqueeze_251: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_249);  unsqueeze_249 = None
    mul_135: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_253: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_136: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_135, unsqueeze_253);  mul_135 = unsqueeze_253 = None
    unsqueeze_254: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_255: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_68: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_255);  mul_136 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_41: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_68)
    mul_137: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_68, sigmoid_41);  sigmoid_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[4, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_137, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_52: "f32[4, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_171, primals_172, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_42: "f32[4, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52)
    mul_138: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_52, sigmoid_42);  sigmoid_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_53: "f32[4, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_138, primals_173, primals_174, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_43: "f32[4, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_53)
    mul_139: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_137, sigmoid_43);  mul_137 = sigmoid_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_54: "f32[4, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_139, primals_175, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_69: "f32[112]" = torch.ops.aten.add.Tensor(primals_279, 1e-05)
    sqrt_32: "f32[112]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
    reciprocal_32: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_140: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_278, -1)
    unsqueeze_257: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_140, -1);  mul_140 = None
    unsqueeze_259: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_257);  unsqueeze_257 = None
    mul_141: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_261: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_142: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_141, unsqueeze_261);  mul_141 = unsqueeze_261 = None
    unsqueeze_262: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_263: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_70: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_263);  mul_142 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_71: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_70, add_64);  add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_55: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(add_71, primals_176, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_72: "f32[672]" = torch.ops.aten.add.Tensor(primals_281, 1e-05)
    sqrt_33: "f32[672]" = torch.ops.aten.sqrt.default(add_72);  add_72 = None
    reciprocal_33: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_143: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_280, -1)
    unsqueeze_265: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_143, -1);  mul_143 = None
    unsqueeze_267: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_265);  unsqueeze_265 = None
    mul_144: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_269: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_145: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_144, unsqueeze_269);  mul_144 = unsqueeze_269 = None
    unsqueeze_270: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_271: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_73: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_145, unsqueeze_271);  mul_145 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_44: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_73)
    mul_146: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_73, sigmoid_44);  sigmoid_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_56: "f32[4, 672, 7, 7]" = torch.ops.aten.convolution.default(mul_146, primals_177, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_74: "f32[672]" = torch.ops.aten.add.Tensor(primals_283, 1e-05)
    sqrt_34: "f32[672]" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
    reciprocal_34: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_147: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_282, -1)
    unsqueeze_273: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_275: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_273);  unsqueeze_273 = None
    mul_148: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_277: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_149: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_277);  mul_148 = unsqueeze_277 = None
    unsqueeze_278: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_279: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_75: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_279);  mul_149 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_45: "f32[4, 672, 7, 7]" = torch.ops.aten.sigmoid.default(add_75)
    mul_150: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_75, sigmoid_45);  sigmoid_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[4, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_150, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_57: "f32[4, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_178, primals_179, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_46: "f32[4, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_57)
    mul_151: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_57, sigmoid_46);  sigmoid_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_58: "f32[4, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_151, primals_180, primals_181, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_47: "f32[4, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_58)
    mul_152: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_150, sigmoid_47);  mul_150 = sigmoid_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_59: "f32[4, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_152, primals_182, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_76: "f32[192]" = torch.ops.aten.add.Tensor(primals_285, 1e-05)
    sqrt_35: "f32[192]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_35: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_153: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_284, -1)
    unsqueeze_281: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_283: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_281);  unsqueeze_281 = None
    mul_154: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_285: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_155: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_285);  mul_154 = unsqueeze_285 = None
    unsqueeze_286: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_287: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_77: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_287);  mul_155 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_60: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_77, primals_183, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_78: "f32[1152]" = torch.ops.aten.add.Tensor(primals_287, 1e-05)
    sqrt_36: "f32[1152]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    reciprocal_36: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_156: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_286, -1)
    unsqueeze_289: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
    unsqueeze_291: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_289);  unsqueeze_289 = None
    mul_157: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_293: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_158: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_293);  mul_157 = unsqueeze_293 = None
    unsqueeze_294: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_295: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_79: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_295);  mul_158 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_48: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_79)
    mul_159: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_79, sigmoid_48);  sigmoid_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_61: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_159, primals_184, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_80: "f32[1152]" = torch.ops.aten.add.Tensor(primals_289, 1e-05)
    sqrt_37: "f32[1152]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_37: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_160: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_288, -1)
    unsqueeze_297: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_160, -1);  mul_160 = None
    unsqueeze_299: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_297);  unsqueeze_297 = None
    mul_161: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_301: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_162: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_301);  mul_161 = unsqueeze_301 = None
    unsqueeze_302: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_303: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_81: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_162, unsqueeze_303);  mul_162 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_49: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_81)
    mul_163: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_81, sigmoid_49);  sigmoid_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[4, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_163, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_62: "f32[4, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_185, primals_186, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_50: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_62)
    mul_164: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_62, sigmoid_50);  sigmoid_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_63: "f32[4, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_164, primals_187, primals_188, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_51: "f32[4, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_63)
    mul_165: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_163, sigmoid_51);  mul_163 = sigmoid_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_64: "f32[4, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_165, primals_189, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_82: "f32[192]" = torch.ops.aten.add.Tensor(primals_291, 1e-05)
    sqrt_38: "f32[192]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_38: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_166: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_290, -1)
    unsqueeze_305: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_166, -1);  mul_166 = None
    unsqueeze_307: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_305);  unsqueeze_305 = None
    mul_167: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_309: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_168: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_309);  mul_167 = unsqueeze_309 = None
    unsqueeze_310: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_311: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_83: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_168, unsqueeze_311);  mul_168 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_84: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_83, add_77);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_65: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_84, primals_190, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_85: "f32[1152]" = torch.ops.aten.add.Tensor(primals_293, 1e-05)
    sqrt_39: "f32[1152]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
    reciprocal_39: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_169: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_292, -1)
    unsqueeze_313: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_169, -1);  mul_169 = None
    unsqueeze_315: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_313);  unsqueeze_313 = None
    mul_170: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_317: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_171: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_170, unsqueeze_317);  mul_170 = unsqueeze_317 = None
    unsqueeze_318: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_319: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_86: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_171, unsqueeze_319);  mul_171 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_52: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_86)
    mul_172: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_86, sigmoid_52);  sigmoid_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_66: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_172, primals_191, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_87: "f32[1152]" = torch.ops.aten.add.Tensor(primals_295, 1e-05)
    sqrt_40: "f32[1152]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    reciprocal_40: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_173: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_294, -1)
    unsqueeze_321: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_173, -1);  mul_173 = None
    unsqueeze_323: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_321);  unsqueeze_321 = None
    mul_174: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_325: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_175: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_174, unsqueeze_325);  mul_174 = unsqueeze_325 = None
    unsqueeze_326: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_327: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_88: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_327);  mul_175 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_53: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_88)
    mul_176: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_88, sigmoid_53);  sigmoid_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_13: "f32[4, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_176, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_67: "f32[4, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_13, primals_192, primals_193, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_54: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67)
    mul_177: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_67, sigmoid_54);  sigmoid_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_68: "f32[4, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_177, primals_194, primals_195, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_55: "f32[4, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_68)
    mul_178: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_176, sigmoid_55);  mul_176 = sigmoid_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_69: "f32[4, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_178, primals_196, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_89: "f32[192]" = torch.ops.aten.add.Tensor(primals_297, 1e-05)
    sqrt_41: "f32[192]" = torch.ops.aten.sqrt.default(add_89);  add_89 = None
    reciprocal_41: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_179: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_296, -1)
    unsqueeze_329: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_179, -1);  mul_179 = None
    unsqueeze_331: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_329);  unsqueeze_329 = None
    mul_180: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_333: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_181: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_333);  mul_180 = unsqueeze_333 = None
    unsqueeze_334: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_335: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_90: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_335);  mul_181 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_91: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_90, add_84);  add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_70: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_91, primals_197, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_92: "f32[1152]" = torch.ops.aten.add.Tensor(primals_299, 1e-05)
    sqrt_42: "f32[1152]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_42: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_182: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_298, -1)
    unsqueeze_337: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_182, -1);  mul_182 = None
    unsqueeze_339: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_337);  unsqueeze_337 = None
    mul_183: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_341: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_184: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_341);  mul_183 = unsqueeze_341 = None
    unsqueeze_342: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_343: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_93: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_343);  mul_184 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_56: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_93)
    mul_185: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_93, sigmoid_56);  sigmoid_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_71: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_185, primals_198, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_94: "f32[1152]" = torch.ops.aten.add.Tensor(primals_301, 1e-05)
    sqrt_43: "f32[1152]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_43: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_186: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_300, -1)
    unsqueeze_345: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
    unsqueeze_347: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_345);  unsqueeze_345 = None
    mul_187: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_349: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_188: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_349);  mul_187 = unsqueeze_349 = None
    unsqueeze_350: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_351: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_95: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_351);  mul_188 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_57: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_95)
    mul_189: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_95, sigmoid_57);  sigmoid_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_14: "f32[4, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_189, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_72: "f32[4, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_14, primals_199, primals_200, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_58: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_72)
    mul_190: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_72, sigmoid_58);  sigmoid_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_73: "f32[4, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_190, primals_201, primals_202, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_59: "f32[4, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73)
    mul_191: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_189, sigmoid_59);  mul_189 = sigmoid_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_74: "f32[4, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_191, primals_203, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_96: "f32[192]" = torch.ops.aten.add.Tensor(primals_303, 1e-05)
    sqrt_44: "f32[192]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_44: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_192: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_302, -1)
    unsqueeze_353: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
    unsqueeze_355: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_353);  unsqueeze_353 = None
    mul_193: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_357: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_194: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_357);  mul_193 = unsqueeze_357 = None
    unsqueeze_358: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_359: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_97: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_359);  mul_194 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_98: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_97, add_91);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_75: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_98, primals_204, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_99: "f32[1152]" = torch.ops.aten.add.Tensor(primals_305, 1e-05)
    sqrt_45: "f32[1152]" = torch.ops.aten.sqrt.default(add_99);  add_99 = None
    reciprocal_45: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_195: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_304, -1)
    unsqueeze_361: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
    unsqueeze_363: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_361);  unsqueeze_361 = None
    mul_196: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_365: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_197: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_365);  mul_196 = unsqueeze_365 = None
    unsqueeze_366: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_367: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_100: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_367);  mul_197 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_60: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_100)
    mul_198: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_100, sigmoid_60);  sigmoid_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_76: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_198, primals_205, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_101: "f32[1152]" = torch.ops.aten.add.Tensor(primals_307, 1e-05)
    sqrt_46: "f32[1152]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
    reciprocal_46: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_199: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_306, -1)
    unsqueeze_369: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_199, -1);  mul_199 = None
    unsqueeze_371: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_369);  unsqueeze_369 = None
    mul_200: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1)
    unsqueeze_373: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_201: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_200, unsqueeze_373);  mul_200 = unsqueeze_373 = None
    unsqueeze_374: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1);  primals_94 = None
    unsqueeze_375: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_102: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_201, unsqueeze_375);  mul_201 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_61: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_102)
    mul_202: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_102, sigmoid_61);  sigmoid_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_15: "f32[4, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_202, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_77: "f32[4, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_15, primals_206, primals_207, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_62: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_77)
    mul_203: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_62);  sigmoid_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_78: "f32[4, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_203, primals_208, primals_209, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_63: "f32[4, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_78)
    mul_204: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_202, sigmoid_63);  mul_202 = sigmoid_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_79: "f32[4, 320, 7, 7]" = torch.ops.aten.convolution.default(mul_204, primals_210, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_103: "f32[320]" = torch.ops.aten.add.Tensor(primals_309, 1e-05)
    sqrt_47: "f32[320]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
    reciprocal_47: "f32[320]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_205: "f32[320]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_308, -1)
    unsqueeze_377: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(mul_205, -1);  mul_205 = None
    unsqueeze_379: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[4, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_377);  unsqueeze_377 = None
    mul_206: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_381: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_207: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(mul_206, unsqueeze_381);  mul_206 = unsqueeze_381 = None
    unsqueeze_382: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_383: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_104: "f32[4, 320, 7, 7]" = torch.ops.aten.add.Tensor(mul_207, unsqueeze_383);  mul_207 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_80: "f32[4, 1280, 7, 7]" = torch.ops.aten.convolution.default(add_104, primals_211, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_105: "f32[1280]" = torch.ops.aten.add.Tensor(primals_311, 1e-05)
    sqrt_48: "f32[1280]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    reciprocal_48: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_208: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_310, -1)
    unsqueeze_385: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_208, -1);  mul_208 = None
    unsqueeze_387: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_385);  unsqueeze_385 = None
    mul_209: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_389: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_210: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_389);  mul_209 = unsqueeze_389 = None
    unsqueeze_390: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_391: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_106: "f32[4, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_210, unsqueeze_391);  mul_210 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_64: "f32[4, 1280, 7, 7]" = torch.ops.aten.sigmoid.default(add_106)
    mul_211: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(add_106, sigmoid_64);  sigmoid_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_16: "f32[4, 1280, 1, 1]" = torch.ops.aten.mean.dim(mul_211, [-1, -2], True);  mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[4, 1280]" = torch.ops.aten.reshape.default(mean_16, [4, 1280]);  mean_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_212, [1, 0]);  primals_212 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_213, view, permute);  primals_213 = None
    permute_1: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_65: "f32[4, 1280, 7, 7]" = torch.ops.aten.sigmoid.default(add_106)
    full_default: "f32[4, 1280, 7, 7]" = torch.ops.aten.full.default([4, 1280, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_49: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(full_default, sigmoid_65);  full_default = None
    mul_212: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(add_106, sub_49);  add_106 = sub_49 = None
    add_107: "f32[4, 1280, 7, 7]" = torch.ops.aten.add.Scalar(mul_212, 1);  mul_212 = None
    mul_213: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_65, add_107);  sigmoid_65 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_2: "f32[4, 1152, 7, 7]" = torch.ops.aten.full.default([4, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_68: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_100)
    sub_56: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_68)
    mul_249: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_100, sub_56);  add_100 = sub_56 = None
    add_114: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_249, 1);  mul_249 = None
    mul_250: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_68, add_114);  sigmoid_68 = add_114 = None
    sigmoid_71: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_93)
    sub_63: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_71)
    mul_286: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_93, sub_63);  add_93 = sub_63 = None
    add_121: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_286, 1);  mul_286 = None
    mul_287: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_71, add_121);  sigmoid_71 = add_121 = None
    sigmoid_74: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_86)
    sub_70: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_74)
    mul_323: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_86, sub_70);  add_86 = sub_70 = None
    add_129: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_323, 1);  mul_323 = None
    mul_324: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_74, add_129);  sigmoid_74 = add_129 = None
    sigmoid_77: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_79)
    sub_77: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_77);  full_default_2 = None
    mul_360: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_79, sub_77);  add_79 = sub_77 = None
    add_137: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_360, 1);  mul_360 = None
    mul_361: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_77, add_137);  sigmoid_77 = add_137 = None
    sigmoid_80: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_73)
    full_default_15: "f32[4, 672, 14, 14]" = torch.ops.aten.full.default([4, 672, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_84: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_15, sigmoid_80)
    mul_397: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_73, sub_84);  add_73 = sub_84 = None
    add_145: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_397, 1);  mul_397 = None
    mul_398: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_80, add_145);  sigmoid_80 = add_145 = None
    sigmoid_83: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_66)
    sub_91: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_15, sigmoid_83)
    mul_434: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_66, sub_91);  add_66 = sub_91 = None
    add_152: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_434, 1);  mul_434 = None
    mul_435: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_83, add_152);  sigmoid_83 = add_152 = None
    sigmoid_86: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_59)
    sub_98: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_15, sigmoid_86);  full_default_15 = None
    mul_471: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_59, sub_98);  add_59 = sub_98 = None
    add_160: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_471, 1);  mul_471 = None
    mul_472: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_86, add_160);  sigmoid_86 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_23: "f32[4, 480, 14, 14]" = torch.ops.aten.full.default([4, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_89: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_53)
    sub_105: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_89)
    mul_508: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_53, sub_105);  add_53 = sub_105 = None
    add_168: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_508, 1);  mul_508 = None
    mul_509: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_89, add_168);  sigmoid_89 = add_168 = None
    sigmoid_92: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_46)
    sub_112: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_92)
    mul_545: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_46, sub_112);  add_46 = sub_112 = None
    add_175: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_545, 1);  mul_545 = None
    mul_546: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_92, add_175);  sigmoid_92 = add_175 = None
    sigmoid_95: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_39)
    sub_119: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_95);  full_default_23 = None
    mul_582: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_39, sub_119);  add_39 = sub_119 = None
    add_183: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_582, 1);  mul_582 = None
    mul_583: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_95, add_183);  sigmoid_95 = add_183 = None
    sigmoid_98: "f32[4, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_33)
    full_default_33: "f32[4, 240, 28, 28]" = torch.ops.aten.full.default([4, 240, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_126: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_33, sigmoid_98)
    mul_619: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_33, sub_126);  add_33 = sub_126 = None
    add_191: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Scalar(mul_619, 1);  mul_619 = None
    mul_620: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_98, add_191);  sigmoid_98 = add_191 = None
    sigmoid_101: "f32[4, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_26)
    sub_133: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_33, sigmoid_101);  full_default_33 = None
    mul_656: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_26, sub_133);  add_26 = sub_133 = None
    add_198: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Scalar(mul_656, 1);  mul_656 = None
    mul_657: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_101, add_198);  sigmoid_101 = add_198 = None
    sigmoid_104: "f32[4, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_20)
    full_default_39: "f32[4, 144, 56, 56]" = torch.ops.aten.full.default([4, 144, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_140: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(full_default_39, sigmoid_104)
    mul_693: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_20, sub_140);  add_20 = sub_140 = None
    add_206: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Scalar(mul_693, 1);  mul_693 = None
    mul_694: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_104, add_206);  sigmoid_104 = add_206 = None
    sigmoid_107: "f32[4, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_13)
    sub_147: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(full_default_39, sigmoid_107);  full_default_39 = None
    mul_730: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_13, sub_147);  add_13 = sub_147 = None
    add_213: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Scalar(mul_730, 1);  mul_730 = None
    mul_731: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_107, add_213);  sigmoid_107 = add_213 = None
    sigmoid_110: "f32[4, 96, 112, 112]" = torch.ops.aten.sigmoid.default(add_7)
    full_default_45: "f32[4, 96, 112, 112]" = torch.ops.aten.full.default([4, 96, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_154: "f32[4, 96, 112, 112]" = torch.ops.aten.sub.Tensor(full_default_45, sigmoid_110);  full_default_45 = None
    mul_767: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(add_7, sub_154);  add_7 = sub_154 = None
    add_221: "f32[4, 96, 112, 112]" = torch.ops.aten.add.Scalar(mul_767, 1);  mul_767 = None
    mul_768: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_110, add_221);  sigmoid_110 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_47: "f32[4, 32, 112, 112]" = torch.ops.aten.full.default([4, 32, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_113: "f32[4, 32, 112, 112]" = torch.ops.aten.sigmoid.default(add_1)
    sub_161: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(full_default_47, sigmoid_113);  full_default_47 = None
    mul_804: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_1, sub_161);  add_1 = sub_161 = None
    add_228: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Scalar(mul_804, 1);  mul_804 = None
    mul_805: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_113, add_228);  sigmoid_113 = add_228 = None
    return [addmm, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_100, primals_101, primals_103, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_119, primals_120, primals_121, primals_122, primals_124, primals_126, primals_127, primals_128, primals_129, primals_131, primals_133, primals_134, primals_135, primals_136, primals_138, primals_140, primals_141, primals_142, primals_143, primals_145, primals_147, primals_148, primals_149, primals_150, primals_152, primals_154, primals_155, primals_156, primals_157, primals_159, primals_161, primals_162, primals_163, primals_164, primals_166, primals_168, primals_169, primals_170, primals_171, primals_173, primals_175, primals_176, primals_177, primals_178, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_194, primals_196, primals_197, primals_198, primals_199, primals_201, primals_203, primals_204, primals_205, primals_206, primals_208, primals_210, primals_211, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, convolution, mul_3, convolution_1, add_3, mean, convolution_2, mul_8, convolution_3, mul_9, convolution_4, add_5, convolution_5, mul_16, convolution_6, add_9, mean_1, convolution_7, mul_21, convolution_8, mul_22, convolution_9, add_11, convolution_10, mul_29, convolution_11, add_15, mean_2, convolution_12, mul_34, convolution_13, mul_35, convolution_14, add_18, convolution_15, mul_42, convolution_16, add_22, mean_3, convolution_17, mul_47, convolution_18, mul_48, convolution_19, add_24, convolution_20, mul_55, convolution_21, add_28, mean_4, convolution_22, mul_60, convolution_23, mul_61, convolution_24, add_31, convolution_25, mul_68, convolution_26, add_35, mean_5, convolution_27, mul_73, convolution_28, mul_74, convolution_29, add_37, convolution_30, mul_81, convolution_31, add_41, mean_6, convolution_32, mul_86, convolution_33, mul_87, convolution_34, add_44, convolution_35, mul_94, convolution_36, add_48, mean_7, convolution_37, mul_99, convolution_38, mul_100, convolution_39, add_51, convolution_40, mul_107, convolution_41, add_55, mean_8, convolution_42, mul_112, convolution_43, mul_113, convolution_44, add_57, convolution_45, mul_120, convolution_46, add_61, mean_9, convolution_47, mul_125, convolution_48, mul_126, convolution_49, add_64, convolution_50, mul_133, convolution_51, add_68, mean_10, convolution_52, mul_138, convolution_53, mul_139, convolution_54, add_71, convolution_55, mul_146, convolution_56, add_75, mean_11, convolution_57, mul_151, convolution_58, mul_152, convolution_59, add_77, convolution_60, mul_159, convolution_61, add_81, mean_12, convolution_62, mul_164, convolution_63, mul_165, convolution_64, add_84, convolution_65, mul_172, convolution_66, add_88, mean_13, convolution_67, mul_177, convolution_68, mul_178, convolution_69, add_91, convolution_70, mul_185, convolution_71, add_95, mean_14, convolution_72, mul_190, convolution_73, mul_191, convolution_74, add_98, convolution_75, mul_198, convolution_76, add_102, mean_15, convolution_77, mul_203, convolution_78, mul_204, convolution_79, add_104, convolution_80, view, permute_1, mul_213, mul_250, mul_287, mul_324, mul_361, mul_398, mul_435, mul_472, mul_509, mul_546, mul_583, mul_620, mul_657, mul_694, mul_731, mul_768, mul_805]
    