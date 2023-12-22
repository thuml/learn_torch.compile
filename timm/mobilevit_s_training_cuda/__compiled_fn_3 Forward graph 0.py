from __future__ import annotations



def forward(self, primals_1: "f32[16]", primals_2: "f32[16]", primals_3: "f32[64]", primals_4: "f32[64]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[32]", primals_8: "f32[32]", primals_9: "f32[128]", primals_10: "f32[128]", primals_11: "f32[128]", primals_12: "f32[128]", primals_13: "f32[64]", primals_14: "f32[64]", primals_15: "f32[256]", primals_16: "f32[256]", primals_17: "f32[256]", primals_18: "f32[256]", primals_19: "f32[64]", primals_20: "f32[64]", primals_21: "f32[256]", primals_22: "f32[256]", primals_23: "f32[256]", primals_24: "f32[256]", primals_25: "f32[64]", primals_26: "f32[64]", primals_27: "f32[256]", primals_28: "f32[256]", primals_29: "f32[256]", primals_30: "f32[256]", primals_31: "f32[96]", primals_32: "f32[96]", primals_33: "f32[96]", primals_34: "f32[96]", primals_35: "f32[96]", primals_36: "f32[96]", primals_37: "f32[96]", primals_38: "f32[96]", primals_39: "f32[384]", primals_40: "f32[384]", primals_41: "f32[384]", primals_42: "f32[384]", primals_43: "f32[128]", primals_44: "f32[128]", primals_45: "f32[128]", primals_46: "f32[128]", primals_47: "f32[128]", primals_48: "f32[128]", primals_49: "f32[128]", primals_50: "f32[128]", primals_51: "f32[512]", primals_52: "f32[512]", primals_53: "f32[512]", primals_54: "f32[512]", primals_55: "f32[160]", primals_56: "f32[160]", primals_57: "f32[160]", primals_58: "f32[160]", primals_59: "f32[160]", primals_60: "f32[160]", primals_61: "f32[160]", primals_62: "f32[160]", primals_63: "f32[640]", primals_64: "f32[640]", primals_65: "f32[16, 3, 3, 3]", primals_66: "f32[64, 16, 1, 1]", primals_67: "f32[64, 1, 3, 3]", primals_68: "f32[32, 64, 1, 1]", primals_69: "f32[128, 32, 1, 1]", primals_70: "f32[128, 1, 3, 3]", primals_71: "f32[64, 128, 1, 1]", primals_72: "f32[256, 64, 1, 1]", primals_73: "f32[256, 1, 3, 3]", primals_74: "f32[64, 256, 1, 1]", primals_75: "f32[256, 64, 1, 1]", primals_76: "f32[256, 1, 3, 3]", primals_77: "f32[64, 256, 1, 1]", primals_78: "f32[256, 64, 1, 1]", primals_79: "f32[256, 1, 3, 3]", primals_80: "f32[96, 256, 1, 1]", primals_81: "f32[96, 96, 3, 3]", primals_82: "f32[144, 96, 1, 1]", primals_83: "f32[144]", primals_84: "f32[144]", primals_85: "f32[432, 144]", primals_86: "f32[432]", primals_87: "f32[144, 144]", primals_88: "f32[144]", primals_89: "f32[144]", primals_90: "f32[144]", primals_91: "f32[288, 144]", primals_92: "f32[288]", primals_93: "f32[144, 288]", primals_94: "f32[144]", primals_95: "f32[144]", primals_96: "f32[144]", primals_97: "f32[432, 144]", primals_98: "f32[432]", primals_99: "f32[144, 144]", primals_100: "f32[144]", primals_101: "f32[144]", primals_102: "f32[144]", primals_103: "f32[288, 144]", primals_104: "f32[288]", primals_105: "f32[144, 288]", primals_106: "f32[144]", primals_107: "f32[144]", primals_108: "f32[144]", primals_109: "f32[96, 144, 1, 1]", primals_110: "f32[96, 192, 3, 3]", primals_111: "f32[384, 96, 1, 1]", primals_112: "f32[384, 1, 3, 3]", primals_113: "f32[128, 384, 1, 1]", primals_114: "f32[128, 128, 3, 3]", primals_115: "f32[192, 128, 1, 1]", primals_116: "f32[192]", primals_117: "f32[192]", primals_118: "f32[576, 192]", primals_119: "f32[576]", primals_120: "f32[192, 192]", primals_121: "f32[192]", primals_122: "f32[192]", primals_123: "f32[192]", primals_124: "f32[384, 192]", primals_125: "f32[384]", primals_126: "f32[192, 384]", primals_127: "f32[192]", primals_128: "f32[192]", primals_129: "f32[192]", primals_130: "f32[576, 192]", primals_131: "f32[576]", primals_132: "f32[192, 192]", primals_133: "f32[192]", primals_134: "f32[192]", primals_135: "f32[192]", primals_136: "f32[384, 192]", primals_137: "f32[384]", primals_138: "f32[192, 384]", primals_139: "f32[192]", primals_140: "f32[192]", primals_141: "f32[192]", primals_142: "f32[576, 192]", primals_143: "f32[576]", primals_144: "f32[192, 192]", primals_145: "f32[192]", primals_146: "f32[192]", primals_147: "f32[192]", primals_148: "f32[384, 192]", primals_149: "f32[384]", primals_150: "f32[192, 384]", primals_151: "f32[192]", primals_152: "f32[192]", primals_153: "f32[192]", primals_154: "f32[576, 192]", primals_155: "f32[576]", primals_156: "f32[192, 192]", primals_157: "f32[192]", primals_158: "f32[192]", primals_159: "f32[192]", primals_160: "f32[384, 192]", primals_161: "f32[384]", primals_162: "f32[192, 384]", primals_163: "f32[192]", primals_164: "f32[192]", primals_165: "f32[192]", primals_166: "f32[128, 192, 1, 1]", primals_167: "f32[128, 256, 3, 3]", primals_168: "f32[512, 128, 1, 1]", primals_169: "f32[512, 1, 3, 3]", primals_170: "f32[160, 512, 1, 1]", primals_171: "f32[160, 160, 3, 3]", primals_172: "f32[240, 160, 1, 1]", primals_173: "f32[240]", primals_174: "f32[240]", primals_175: "f32[720, 240]", primals_176: "f32[720]", primals_177: "f32[240, 240]", primals_178: "f32[240]", primals_179: "f32[240]", primals_180: "f32[240]", primals_181: "f32[480, 240]", primals_182: "f32[480]", primals_183: "f32[240, 480]", primals_184: "f32[240]", primals_185: "f32[240]", primals_186: "f32[240]", primals_187: "f32[720, 240]", primals_188: "f32[720]", primals_189: "f32[240, 240]", primals_190: "f32[240]", primals_191: "f32[240]", primals_192: "f32[240]", primals_193: "f32[480, 240]", primals_194: "f32[480]", primals_195: "f32[240, 480]", primals_196: "f32[240]", primals_197: "f32[240]", primals_198: "f32[240]", primals_199: "f32[720, 240]", primals_200: "f32[720]", primals_201: "f32[240, 240]", primals_202: "f32[240]", primals_203: "f32[240]", primals_204: "f32[240]", primals_205: "f32[480, 240]", primals_206: "f32[480]", primals_207: "f32[240, 480]", primals_208: "f32[240]", primals_209: "f32[240]", primals_210: "f32[240]", primals_211: "f32[160, 240, 1, 1]", primals_212: "f32[160, 320, 3, 3]", primals_213: "f32[640, 160, 1, 1]", primals_214: "f32[1000, 640]", primals_215: "f32[1000]", primals_216: "i64[]", primals_217: "f32[16]", primals_218: "f32[16]", primals_219: "i64[]", primals_220: "f32[64]", primals_221: "f32[64]", primals_222: "i64[]", primals_223: "f32[64]", primals_224: "f32[64]", primals_225: "i64[]", primals_226: "f32[32]", primals_227: "f32[32]", primals_228: "i64[]", primals_229: "f32[128]", primals_230: "f32[128]", primals_231: "i64[]", primals_232: "f32[128]", primals_233: "f32[128]", primals_234: "i64[]", primals_235: "f32[64]", primals_236: "f32[64]", primals_237: "i64[]", primals_238: "f32[256]", primals_239: "f32[256]", primals_240: "i64[]", primals_241: "f32[256]", primals_242: "f32[256]", primals_243: "i64[]", primals_244: "f32[64]", primals_245: "f32[64]", primals_246: "i64[]", primals_247: "f32[256]", primals_248: "f32[256]", primals_249: "i64[]", primals_250: "f32[256]", primals_251: "f32[256]", primals_252: "i64[]", primals_253: "f32[64]", primals_254: "f32[64]", primals_255: "i64[]", primals_256: "f32[256]", primals_257: "f32[256]", primals_258: "i64[]", primals_259: "f32[256]", primals_260: "f32[256]", primals_261: "i64[]", primals_262: "f32[96]", primals_263: "f32[96]", primals_264: "i64[]", primals_265: "f32[96]", primals_266: "f32[96]", primals_267: "i64[]", primals_268: "f32[96]", primals_269: "f32[96]", primals_270: "i64[]", primals_271: "f32[96]", primals_272: "f32[96]", primals_273: "i64[]", primals_274: "f32[384]", primals_275: "f32[384]", primals_276: "i64[]", primals_277: "f32[384]", primals_278: "f32[384]", primals_279: "i64[]", primals_280: "f32[128]", primals_281: "f32[128]", primals_282: "i64[]", primals_283: "f32[128]", primals_284: "f32[128]", primals_285: "i64[]", primals_286: "f32[128]", primals_287: "f32[128]", primals_288: "i64[]", primals_289: "f32[128]", primals_290: "f32[128]", primals_291: "i64[]", primals_292: "f32[512]", primals_293: "f32[512]", primals_294: "i64[]", primals_295: "f32[512]", primals_296: "f32[512]", primals_297: "i64[]", primals_298: "f32[160]", primals_299: "f32[160]", primals_300: "i64[]", primals_301: "f32[160]", primals_302: "f32[160]", primals_303: "i64[]", primals_304: "f32[160]", primals_305: "f32[160]", primals_306: "i64[]", primals_307: "f32[160]", primals_308: "f32[160]", primals_309: "i64[]", primals_310: "f32[640]", primals_311: "f32[640]", primals_312: "f32[8, 3, 256, 256]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(primals_312, primals_65, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_216, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 16, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 16, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
    add_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000076294527394);  squeeze_2 = None
    mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[16]" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
    add_3: "f32[16]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone: "f32[8, 16, 128, 128]" = torch.ops.aten.clone.default(add_4)
    sigmoid: "f32[8, 16, 128, 128]" = torch.ops.aten.sigmoid.default(add_4)
    mul_7: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(add_4, sigmoid);  add_4 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(mul_7, primals_66, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 64, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_8: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
    add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000076294527394);  squeeze_5 = None
    mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[64]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
    add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_1: "f32[8, 64, 128, 128]" = torch.ops.aten.clone.default(add_9)
    sigmoid_1: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(add_9)
    mul_15: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_1);  add_9 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(mul_15, primals_67, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_222, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_16: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
    add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
    squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000076294527394);  squeeze_8 = None
    mul_20: "f32[64]" = torch.ops.aten.mul.Tensor(mul_19, 0.1);  mul_19 = None
    mul_21: "f32[64]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
    add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_22: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_9);  mul_16 = unsqueeze_9 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_11);  mul_22 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_2: "f32[8, 64, 128, 128]" = torch.ops.aten.clone.default(add_14)
    sigmoid_2: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(add_14)
    mul_23: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_14, sigmoid_2);  add_14 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_3: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(mul_23, primals_68, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 32, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 32, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_24: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_25: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_26: "f32[32]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
    add_17: "f32[32]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    squeeze_11: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_27: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000076294527394);  squeeze_11 = None
    mul_28: "f32[32]" = torch.ops.aten.mul.Tensor(mul_27, 0.1);  mul_27 = None
    mul_29: "f32[32]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
    add_18: "f32[32]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_30: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_13);  mul_24 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_15);  mul_30 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[8, 128, 128, 128]" = torch.ops.aten.convolution.default(add_19, primals_69, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_228, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
    mul_31: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_32: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_33: "f32[128]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
    add_22: "f32[128]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    squeeze_14: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_34: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000076294527394);  squeeze_14 = None
    mul_35: "f32[128]" = torch.ops.aten.mul.Tensor(mul_34, 0.1);  mul_34 = None
    mul_36: "f32[128]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_23: "f32[128]" = torch.ops.aten.add.Tensor(mul_35, mul_36);  mul_35 = mul_36 = None
    unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_37: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_17);  mul_31 = unsqueeze_17 = None
    unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 128, 128, 128]" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_19);  mul_37 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_3: "f32[8, 128, 128, 128]" = torch.ops.aten.clone.default(add_24)
    sigmoid_3: "f32[8, 128, 128, 128]" = torch.ops.aten.sigmoid.default(add_24)
    mul_38: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(add_24, sigmoid_3);  add_24 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(mul_38, primals_70, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
    mul_39: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_40: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_41: "f32[128]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
    add_27: "f32[128]" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
    squeeze_17: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_42: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.000030518509476);  squeeze_17 = None
    mul_43: "f32[128]" = torch.ops.aten.mul.Tensor(mul_42, 0.1);  mul_42 = None
    mul_44: "f32[128]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_28: "f32[128]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    unsqueeze_20: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_45: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_39, unsqueeze_21);  mul_39 = unsqueeze_21 = None
    unsqueeze_22: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_45, unsqueeze_23);  mul_45 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_4: "f32[8, 128, 64, 64]" = torch.ops.aten.clone.default(add_29)
    sigmoid_4: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(add_29)
    mul_46: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_29, sigmoid_4);  add_29 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_46, primals_71, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_234, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 64, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 64, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
    mul_47: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_48: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_49: "f32[64]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
    add_32: "f32[64]" = torch.ops.aten.add.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None
    squeeze_20: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_50: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.000030518509476);  squeeze_20 = None
    mul_51: "f32[64]" = torch.ops.aten.mul.Tensor(mul_50, 0.1);  mul_50 = None
    mul_52: "f32[64]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_33: "f32[64]" = torch.ops.aten.add.Tensor(mul_51, mul_52);  mul_51 = mul_52 = None
    unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_53: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_47, unsqueeze_25);  mul_47 = unsqueeze_25 = None
    unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_27);  mul_53 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(add_34, primals_72, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 256, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 256, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_36: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_7: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
    mul_54: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_55: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_56: "f32[256]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
    add_37: "f32[256]" = torch.ops.aten.add.Tensor(mul_55, mul_56);  mul_55 = mul_56 = None
    squeeze_23: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_57: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.000030518509476);  squeeze_23 = None
    mul_58: "f32[256]" = torch.ops.aten.mul.Tensor(mul_57, 0.1);  mul_57 = None
    mul_59: "f32[256]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_38: "f32[256]" = torch.ops.aten.add.Tensor(mul_58, mul_59);  mul_58 = mul_59 = None
    unsqueeze_28: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_60: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_54, unsqueeze_29);  mul_54 = unsqueeze_29 = None
    unsqueeze_30: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_39: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_60, unsqueeze_31);  mul_60 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_5: "f32[8, 256, 64, 64]" = torch.ops.aten.clone.default(add_39)
    sigmoid_5: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_39)
    mul_61: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_39, sigmoid_5);  add_39 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_61, primals_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_40: "i64[]" = torch.ops.aten.add.Tensor(primals_240, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 256, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 256, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_41: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_8: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
    mul_62: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_63: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_64: "f32[256]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
    add_42: "f32[256]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
    squeeze_26: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_65: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.000030518509476);  squeeze_26 = None
    mul_66: "f32[256]" = torch.ops.aten.mul.Tensor(mul_65, 0.1);  mul_65 = None
    mul_67: "f32[256]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_43: "f32[256]" = torch.ops.aten.add.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
    unsqueeze_32: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_68: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_62, unsqueeze_33);  mul_62 = unsqueeze_33 = None
    unsqueeze_34: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_44: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_35);  mul_68 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_6: "f32[8, 256, 64, 64]" = torch.ops.aten.clone.default(add_44)
    sigmoid_6: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_44)
    mul_69: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_44, sigmoid_6);  add_44 = sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_9: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_69, primals_74, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_45: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 64, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 64, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_46: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_9: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
    mul_70: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_71: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_72: "f32[64]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
    add_47: "f32[64]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_29: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_73: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.000030518509476);  squeeze_29 = None
    mul_74: "f32[64]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[64]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_48: "f32[64]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_76: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_37);  mul_70 = unsqueeze_37 = None
    unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_49: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_39);  mul_76 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_50: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_49, add_34);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_10: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(add_50, primals_75, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_246, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 256, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 256, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_52: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_10: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
    mul_77: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_78: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_79: "f32[256]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
    add_53: "f32[256]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_32: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_80: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.000030518509476);  squeeze_32 = None
    mul_81: "f32[256]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[256]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_54: "f32[256]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_40: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_83: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_41);  mul_77 = unsqueeze_41 = None
    unsqueeze_42: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_55: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_43);  mul_83 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_7: "f32[8, 256, 64, 64]" = torch.ops.aten.clone.default(add_55)
    sigmoid_7: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_55)
    mul_84: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_55, sigmoid_7);  add_55 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_84, primals_76, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_56: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 256, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 256, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_11: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
    mul_85: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_86: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_87: "f32[256]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
    add_58: "f32[256]" = torch.ops.aten.add.Tensor(mul_86, mul_87);  mul_86 = mul_87 = None
    squeeze_35: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_88: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.000030518509476);  squeeze_35 = None
    mul_89: "f32[256]" = torch.ops.aten.mul.Tensor(mul_88, 0.1);  mul_88 = None
    mul_90: "f32[256]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_59: "f32[256]" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
    unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_91: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_45);  mul_85 = unsqueeze_45 = None
    unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_60: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_91, unsqueeze_47);  mul_91 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_8: "f32[8, 256, 64, 64]" = torch.ops.aten.clone.default(add_60)
    sigmoid_8: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_60)
    mul_92: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_60, sigmoid_8);  add_60 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_92, primals_77, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_61: "i64[]" = torch.ops.aten.add.Tensor(primals_252, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 64, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 64, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_62: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_12: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
    mul_93: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_94: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_95: "f32[64]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_63: "f32[64]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
    squeeze_38: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_96: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.000030518509476);  squeeze_38 = None
    mul_97: "f32[64]" = torch.ops.aten.mul.Tensor(mul_96, 0.1);  mul_96 = None
    mul_98: "f32[64]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_64: "f32[64]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_99: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_49);  mul_93 = unsqueeze_49 = None
    unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_65: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_99, unsqueeze_51);  mul_99 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_66: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_65, add_50);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(add_66, primals_78, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_255, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 256, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 256, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_68: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_13: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
    mul_100: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_101: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_102: "f32[256]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_69: "f32[256]" = torch.ops.aten.add.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
    squeeze_41: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_103: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.000030518509476);  squeeze_41 = None
    mul_104: "f32[256]" = torch.ops.aten.mul.Tensor(mul_103, 0.1);  mul_103 = None
    mul_105: "f32[256]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_70: "f32[256]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
    unsqueeze_52: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_106: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_53);  mul_100 = unsqueeze_53 = None
    unsqueeze_54: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_71: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_55);  mul_106 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_9: "f32[8, 256, 64, 64]" = torch.ops.aten.clone.default(add_71)
    sigmoid_9: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_71)
    mul_107: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_71, sigmoid_9);  add_71 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_107, primals_79, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_72: "i64[]" = torch.ops.aten.add.Tensor(primals_258, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 256, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 256, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_73: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_14: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
    mul_108: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_109: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_110: "f32[256]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_74: "f32[256]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    squeeze_44: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_111: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001220852154804);  squeeze_44 = None
    mul_112: "f32[256]" = torch.ops.aten.mul.Tensor(mul_111, 0.1);  mul_111 = None
    mul_113: "f32[256]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_75: "f32[256]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    unsqueeze_56: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_114: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_108, unsqueeze_57);  mul_108 = unsqueeze_57 = None
    unsqueeze_58: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_76: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_114, unsqueeze_59);  mul_114 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_10: "f32[8, 256, 32, 32]" = torch.ops.aten.clone.default(add_76)
    sigmoid_10: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(add_76)
    mul_115: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_76, sigmoid_10);  add_76 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_15: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(mul_115, primals_80, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_77: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 96, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 96, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_78: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_15: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
    mul_116: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_117: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_118: "f32[96]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
    add_79: "f32[96]" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    squeeze_47: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_119: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001220852154804);  squeeze_47 = None
    mul_120: "f32[96]" = torch.ops.aten.mul.Tensor(mul_119, 0.1);  mul_119 = None
    mul_121: "f32[96]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_80: "f32[96]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    unsqueeze_60: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_122: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_61);  mul_116 = unsqueeze_61 = None
    unsqueeze_62: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_81: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_63);  mul_122 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(add_81, primals_81, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_82: "i64[]" = torch.ops.aten.add.Tensor(primals_264, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 96, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 96, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_83: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_16: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
    mul_123: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_124: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_125: "f32[96]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
    add_84: "f32[96]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
    squeeze_50: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_126: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001220852154804);  squeeze_50 = None
    mul_127: "f32[96]" = torch.ops.aten.mul.Tensor(mul_126, 0.1);  mul_126 = None
    mul_128: "f32[96]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_85: "f32[96]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    unsqueeze_64: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_129: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_123, unsqueeze_65);  mul_123 = unsqueeze_65 = None
    unsqueeze_66: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_86: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_129, unsqueeze_67);  mul_129 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_11: "f32[8, 96, 32, 32]" = torch.ops.aten.clone.default(add_86)
    sigmoid_11: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(add_86)
    mul_130: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_86, sigmoid_11);  add_86 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_17: "f32[8, 144, 32, 32]" = torch.ops.aten.convolution.default(mul_130, primals_82, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    view: "f32[18432, 2, 16, 2]" = torch.ops.aten.view.default(convolution_17, [18432, 2, 16, 2]);  convolution_17 = None
    permute: "f32[18432, 16, 2, 2]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    clone_12: "f32[18432, 16, 2, 2]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    view_1: "f32[8, 144, 256, 4]" = torch.ops.aten.view.default(clone_12, [8, 144, 256, 4]);  clone_12 = None
    permute_1: "f32[8, 4, 256, 144]" = torch.ops.aten.permute.default(view_1, [0, 3, 2, 1]);  view_1 = None
    clone_13: "f32[8, 4, 256, 144]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_2: "f32[32, 256, 144]" = torch.ops.aten.view.default(clone_13, [32, 256, 144]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_17 = torch.ops.aten.var_mean.correction(view_2, [2], correction = 0, keepdim = True)
    getitem_34: "f32[32, 256, 1]" = var_mean_17[0]
    getitem_35: "f32[32, 256, 1]" = var_mean_17[1];  var_mean_17 = None
    add_87: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_17: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(view_2, getitem_35);  getitem_35 = None
    mul_131: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_132: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_131, primals_83)
    add_88: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_132, primals_84);  mul_132 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_3: "f32[8192, 144]" = torch.ops.aten.view.default(add_88, [8192, 144]);  add_88 = None
    permute_2: "f32[144, 432]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm: "f32[8192, 432]" = torch.ops.aten.addmm.default(primals_86, view_3, permute_2);  primals_86 = None
    view_4: "f32[32, 256, 432]" = torch.ops.aten.view.default(addmm, [32, 256, 432]);  addmm = None
    view_5: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.view.default(view_4, [32, 256, 3, 4, 36]);  view_4 = None
    permute_3: "f32[3, 32, 4, 256, 36]" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
    getitem_36: "f32[32, 4, 256, 36]" = unbind[0]
    getitem_37: "f32[32, 4, 256, 36]" = unbind[1]
    getitem_38: "f32[32, 4, 256, 36]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_36, getitem_37, getitem_38, None, True)
    getitem_39: "f32[32, 4, 256, 36]" = _scaled_dot_product_efficient_attention[0]
    getitem_40: "f32[32, 4, 256]" = _scaled_dot_product_efficient_attention[1]
    getitem_41: "i64[]" = _scaled_dot_product_efficient_attention[2]
    getitem_42: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
    alias: "f32[32, 4, 256, 36]" = torch.ops.aten.alias.default(getitem_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_4: "f32[32, 256, 4, 36]" = torch.ops.aten.permute.default(getitem_39, [0, 2, 1, 3]);  getitem_39 = None
    view_6: "f32[32, 256, 144]" = torch.ops.aten.view.default(permute_4, [32, 256, 144]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_7: "f32[8192, 144]" = torch.ops.aten.view.default(view_6, [8192, 144]);  view_6 = None
    permute_5: "f32[144, 144]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_1: "f32[8192, 144]" = torch.ops.aten.addmm.default(primals_88, view_7, permute_5);  primals_88 = None
    view_8: "f32[32, 256, 144]" = torch.ops.aten.view.default(addmm_1, [32, 256, 144]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_14: "f32[32, 256, 144]" = torch.ops.aten.clone.default(view_8);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_89: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(view_2, clone_14);  view_2 = clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_18 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_43: "f32[32, 256, 1]" = var_mean_18[0]
    getitem_44: "f32[32, 256, 1]" = var_mean_18[1];  var_mean_18 = None
    add_90: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_43, 1e-05);  getitem_43 = None
    rsqrt_18: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_18: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_89, getitem_44);  getitem_44 = None
    mul_133: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_134: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_133, primals_89)
    add_91: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_134, primals_90);  mul_134 = primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_9: "f32[8192, 144]" = torch.ops.aten.view.default(add_91, [8192, 144]);  add_91 = None
    permute_6: "f32[144, 288]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_2: "f32[8192, 288]" = torch.ops.aten.addmm.default(primals_92, view_9, permute_6);  primals_92 = None
    view_10: "f32[32, 256, 288]" = torch.ops.aten.view.default(addmm_2, [32, 256, 288])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_12: "f32[32, 256, 288]" = torch.ops.aten.sigmoid.default(view_10)
    mul_135: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_10, sigmoid_12);  view_10 = sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_15: "f32[32, 256, 288]" = torch.ops.aten.clone.default(mul_135);  mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_11: "f32[8192, 288]" = torch.ops.aten.view.default(clone_15, [8192, 288]);  clone_15 = None
    permute_7: "f32[288, 144]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    addmm_3: "f32[8192, 144]" = torch.ops.aten.addmm.default(primals_94, view_11, permute_7);  primals_94 = None
    view_12: "f32[32, 256, 144]" = torch.ops.aten.view.default(addmm_3, [32, 256, 144]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_16: "f32[32, 256, 144]" = torch.ops.aten.clone.default(view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_92: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_89, clone_16);  add_89 = clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_19 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_45: "f32[32, 256, 1]" = var_mean_19[0]
    getitem_46: "f32[32, 256, 1]" = var_mean_19[1];  var_mean_19 = None
    add_93: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_45, 1e-05);  getitem_45 = None
    rsqrt_19: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_19: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_92, getitem_46);  getitem_46 = None
    mul_136: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_137: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_136, primals_95)
    add_94: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_137, primals_96);  mul_137 = primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_13: "f32[8192, 144]" = torch.ops.aten.view.default(add_94, [8192, 144]);  add_94 = None
    permute_8: "f32[144, 432]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_4: "f32[8192, 432]" = torch.ops.aten.addmm.default(primals_98, view_13, permute_8);  primals_98 = None
    view_14: "f32[32, 256, 432]" = torch.ops.aten.view.default(addmm_4, [32, 256, 432]);  addmm_4 = None
    view_15: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.view.default(view_14, [32, 256, 3, 4, 36]);  view_14 = None
    permute_9: "f32[3, 32, 4, 256, 36]" = torch.ops.aten.permute.default(view_15, [2, 0, 3, 1, 4]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
    getitem_47: "f32[32, 4, 256, 36]" = unbind_1[0]
    getitem_48: "f32[32, 4, 256, 36]" = unbind_1[1]
    getitem_49: "f32[32, 4, 256, 36]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_47, getitem_48, getitem_49, None, True)
    getitem_50: "f32[32, 4, 256, 36]" = _scaled_dot_product_efficient_attention_1[0]
    getitem_51: "f32[32, 4, 256]" = _scaled_dot_product_efficient_attention_1[1]
    getitem_52: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
    getitem_53: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
    alias_1: "f32[32, 4, 256, 36]" = torch.ops.aten.alias.default(getitem_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_10: "f32[32, 256, 4, 36]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
    view_16: "f32[32, 256, 144]" = torch.ops.aten.view.default(permute_10, [32, 256, 144]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_17: "f32[8192, 144]" = torch.ops.aten.view.default(view_16, [8192, 144]);  view_16 = None
    permute_11: "f32[144, 144]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_5: "f32[8192, 144]" = torch.ops.aten.addmm.default(primals_100, view_17, permute_11);  primals_100 = None
    view_18: "f32[32, 256, 144]" = torch.ops.aten.view.default(addmm_5, [32, 256, 144]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_17: "f32[32, 256, 144]" = torch.ops.aten.clone.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_95: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_92, clone_17);  add_92 = clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_20 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
    getitem_54: "f32[32, 256, 1]" = var_mean_20[0]
    getitem_55: "f32[32, 256, 1]" = var_mean_20[1];  var_mean_20 = None
    add_96: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_20: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_20: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_95, getitem_55);  getitem_55 = None
    mul_138: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_139: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_138, primals_101)
    add_97: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_139, primals_102);  mul_139 = primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_19: "f32[8192, 144]" = torch.ops.aten.view.default(add_97, [8192, 144]);  add_97 = None
    permute_12: "f32[144, 288]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_6: "f32[8192, 288]" = torch.ops.aten.addmm.default(primals_104, view_19, permute_12);  primals_104 = None
    view_20: "f32[32, 256, 288]" = torch.ops.aten.view.default(addmm_6, [32, 256, 288])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_13: "f32[32, 256, 288]" = torch.ops.aten.sigmoid.default(view_20)
    mul_140: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_20, sigmoid_13);  view_20 = sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_18: "f32[32, 256, 288]" = torch.ops.aten.clone.default(mul_140);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_21: "f32[8192, 288]" = torch.ops.aten.view.default(clone_18, [8192, 288]);  clone_18 = None
    permute_13: "f32[288, 144]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_7: "f32[8192, 144]" = torch.ops.aten.addmm.default(primals_106, view_21, permute_13);  primals_106 = None
    view_22: "f32[32, 256, 144]" = torch.ops.aten.view.default(addmm_7, [32, 256, 144]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_19: "f32[32, 256, 144]" = torch.ops.aten.clone.default(view_22);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_98: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_95, clone_19);  add_95 = clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
    getitem_56: "f32[32, 256, 1]" = var_mean_21[0]
    getitem_57: "f32[32, 256, 1]" = var_mean_21[1];  var_mean_21 = None
    add_99: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_21: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_21: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_98, getitem_57);  add_98 = getitem_57 = None
    mul_141: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_142: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_141, primals_107)
    add_100: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_142, primals_108);  mul_142 = primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    view_23: "f32[8, 4, 256, 144]" = torch.ops.aten.view.default(add_100, [8, 4, 256, -1]);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    permute_14: "f32[8, 144, 256, 4]" = torch.ops.aten.permute.default(view_23, [0, 3, 2, 1]);  view_23 = None
    clone_20: "f32[8, 144, 256, 4]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    view_24: "f32[18432, 16, 2, 2]" = torch.ops.aten.view.default(clone_20, [18432, 16, 2, 2]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    permute_15: "f32[18432, 2, 16, 2]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
    clone_21: "f32[18432, 2, 16, 2]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    view_25: "f32[8, 144, 32, 32]" = torch.ops.aten.view.default(clone_21, [8, 144, 32, 32]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(view_25, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_101: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 96, 1, 1]" = var_mean_22[0]
    getitem_59: "f32[1, 96, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_102: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_22: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_22: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_59)
    mul_143: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_51: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_52: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_144: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_145: "f32[96]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
    add_103: "f32[96]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    squeeze_53: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_146: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001220852154804);  squeeze_53 = None
    mul_147: "f32[96]" = torch.ops.aten.mul.Tensor(mul_146, 0.1);  mul_146 = None
    mul_148: "f32[96]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_104: "f32[96]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
    unsqueeze_68: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_149: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_69);  mul_143 = unsqueeze_69 = None
    unsqueeze_70: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_105: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_71);  mul_149 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_22: "f32[8, 96, 32, 32]" = torch.ops.aten.clone.default(add_105)
    sigmoid_14: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(add_105)
    mul_150: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_105, sigmoid_14);  add_105 = sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    cat: "f32[8, 192, 32, 32]" = torch.ops.aten.cat.default([add_81, mul_150], 1);  mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(cat, primals_110, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_106: "i64[]" = torch.ops.aten.add.Tensor(primals_270, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 96, 1, 1]" = var_mean_23[0]
    getitem_61: "f32[1, 96, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_107: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_23: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
    sub_23: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_61)
    mul_151: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_54: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_55: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_152: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_153: "f32[96]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
    add_108: "f32[96]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
    squeeze_56: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_154: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001220852154804);  squeeze_56 = None
    mul_155: "f32[96]" = torch.ops.aten.mul.Tensor(mul_154, 0.1);  mul_154 = None
    mul_156: "f32[96]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_109: "f32[96]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    unsqueeze_72: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_157: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_73);  mul_151 = unsqueeze_73 = None
    unsqueeze_74: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_110: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_157, unsqueeze_75);  mul_157 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_23: "f32[8, 96, 32, 32]" = torch.ops.aten.clone.default(add_110)
    sigmoid_15: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(add_110)
    mul_158: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_110, sigmoid_15);  add_110 = sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 384, 32, 32]" = torch.ops.aten.convolution.default(mul_158, primals_111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_111: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 384, 1, 1]" = var_mean_24[0]
    getitem_63: "f32[1, 384, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_112: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_24: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_24: "f32[8, 384, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_63)
    mul_159: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_57: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_58: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_160: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_161: "f32[384]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
    add_113: "f32[384]" = torch.ops.aten.add.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None
    squeeze_59: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_162: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001220852154804);  squeeze_59 = None
    mul_163: "f32[384]" = torch.ops.aten.mul.Tensor(mul_162, 0.1);  mul_162 = None
    mul_164: "f32[384]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_114: "f32[384]" = torch.ops.aten.add.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None
    unsqueeze_76: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_165: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(mul_159, unsqueeze_77);  mul_159 = unsqueeze_77 = None
    unsqueeze_78: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_115: "f32[8, 384, 32, 32]" = torch.ops.aten.add.Tensor(mul_165, unsqueeze_79);  mul_165 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_24: "f32[8, 384, 32, 32]" = torch.ops.aten.clone.default(add_115)
    sigmoid_16: "f32[8, 384, 32, 32]" = torch.ops.aten.sigmoid.default(add_115)
    mul_166: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(add_115, sigmoid_16);  add_115 = sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 384, 16, 16]" = torch.ops.aten.convolution.default(mul_166, primals_112, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 384)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_116: "i64[]" = torch.ops.aten.add.Tensor(primals_276, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 384, 1, 1]" = var_mean_25[0]
    getitem_65: "f32[1, 384, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_117: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_25: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_25: "f32[8, 384, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_65)
    mul_167: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_60: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_61: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_168: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_169: "f32[384]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
    add_118: "f32[384]" = torch.ops.aten.add.Tensor(mul_168, mul_169);  mul_168 = mul_169 = None
    squeeze_62: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_170: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0004885197850513);  squeeze_62 = None
    mul_171: "f32[384]" = torch.ops.aten.mul.Tensor(mul_170, 0.1);  mul_170 = None
    mul_172: "f32[384]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_119: "f32[384]" = torch.ops.aten.add.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
    unsqueeze_80: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_173: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_81);  mul_167 = unsqueeze_81 = None
    unsqueeze_82: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_120: "f32[8, 384, 16, 16]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_83);  mul_173 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_25: "f32[8, 384, 16, 16]" = torch.ops.aten.clone.default(add_120)
    sigmoid_17: "f32[8, 384, 16, 16]" = torch.ops.aten.sigmoid.default(add_120)
    mul_174: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(add_120, sigmoid_17);  add_120 = sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_22: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(mul_174, primals_113, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_121: "i64[]" = torch.ops.aten.add.Tensor(primals_279, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1, 1]" = var_mean_26[0]
    getitem_67: "f32[1, 128, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_122: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_26: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_26: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_67)
    mul_175: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_63: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_64: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_176: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_177: "f32[128]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
    add_123: "f32[128]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_65: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_178: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0004885197850513);  squeeze_65 = None
    mul_179: "f32[128]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[128]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_124: "f32[128]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_84: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_181: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_85);  mul_175 = unsqueeze_85 = None
    unsqueeze_86: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_125: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_87);  mul_181 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(add_125, primals_114, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_126: "i64[]" = torch.ops.aten.add.Tensor(primals_282, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1, 1]" = var_mean_27[0]
    getitem_69: "f32[1, 128, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_127: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_27: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_27: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_69)
    mul_182: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_66: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_67: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_183: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_184: "f32[128]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
    add_128: "f32[128]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_68: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_185: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0004885197850513);  squeeze_68 = None
    mul_186: "f32[128]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[128]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_129: "f32[128]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_88: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_188: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_89);  mul_182 = unsqueeze_89 = None
    unsqueeze_90: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_130: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_91);  mul_188 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_26: "f32[8, 128, 16, 16]" = torch.ops.aten.clone.default(add_130)
    sigmoid_18: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(add_130)
    mul_189: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_130, sigmoid_18);  add_130 = sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_24: "f32[8, 192, 16, 16]" = torch.ops.aten.convolution.default(mul_189, primals_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    view_26: "f32[12288, 2, 8, 2]" = torch.ops.aten.view.default(convolution_24, [12288, 2, 8, 2]);  convolution_24 = None
    permute_16: "f32[12288, 8, 2, 2]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    clone_27: "f32[12288, 8, 2, 2]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    view_27: "f32[8, 192, 64, 4]" = torch.ops.aten.view.default(clone_27, [8, 192, 64, 4]);  clone_27 = None
    permute_17: "f32[8, 4, 64, 192]" = torch.ops.aten.permute.default(view_27, [0, 3, 2, 1]);  view_27 = None
    clone_28: "f32[8, 4, 64, 192]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_28: "f32[32, 64, 192]" = torch.ops.aten.view.default(clone_28, [32, 64, 192]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_28 = torch.ops.aten.var_mean.correction(view_28, [2], correction = 0, keepdim = True)
    getitem_70: "f32[32, 64, 1]" = var_mean_28[0]
    getitem_71: "f32[32, 64, 1]" = var_mean_28[1];  var_mean_28 = None
    add_131: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_28: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    sub_28: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(view_28, getitem_71);  getitem_71 = None
    mul_190: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    mul_191: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_190, primals_116)
    add_132: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_191, primals_117);  mul_191 = primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_29: "f32[2048, 192]" = torch.ops.aten.view.default(add_132, [2048, 192]);  add_132 = None
    permute_18: "f32[192, 576]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_8: "f32[2048, 576]" = torch.ops.aten.addmm.default(primals_119, view_29, permute_18);  primals_119 = None
    view_30: "f32[32, 64, 576]" = torch.ops.aten.view.default(addmm_8, [32, 64, 576]);  addmm_8 = None
    view_31: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_30, [32, 64, 3, 4, 48]);  view_30 = None
    permute_19: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_31, [2, 0, 3, 1, 4]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_19);  permute_19 = None
    getitem_72: "f32[32, 4, 64, 48]" = unbind_2[0]
    getitem_73: "f32[32, 4, 64, 48]" = unbind_2[1]
    getitem_74: "f32[32, 4, 64, 48]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_72, getitem_73, getitem_74, None, True)
    getitem_75: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_2[0]
    getitem_76: "f32[32, 4, 64]" = _scaled_dot_product_efficient_attention_2[1]
    getitem_77: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
    getitem_78: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
    alias_2: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(getitem_75)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_20: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_75, [0, 2, 1, 3]);  getitem_75 = None
    view_32: "f32[32, 64, 192]" = torch.ops.aten.view.default(permute_20, [32, 64, 192]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_33: "f32[2048, 192]" = torch.ops.aten.view.default(view_32, [2048, 192]);  view_32 = None
    permute_21: "f32[192, 192]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_9: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_121, view_33, permute_21);  primals_121 = None
    view_34: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_9, [32, 64, 192]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_29: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_34);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_133: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(view_28, clone_29);  view_28 = clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_29 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
    getitem_79: "f32[32, 64, 1]" = var_mean_29[0]
    getitem_80: "f32[32, 64, 1]" = var_mean_29[1];  var_mean_29 = None
    add_134: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_79, 1e-05);  getitem_79 = None
    rsqrt_29: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_29: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_133, getitem_80);  getitem_80 = None
    mul_192: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    mul_193: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_192, primals_122)
    add_135: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_193, primals_123);  mul_193 = primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_35: "f32[2048, 192]" = torch.ops.aten.view.default(add_135, [2048, 192]);  add_135 = None
    permute_22: "f32[192, 384]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm_10: "f32[2048, 384]" = torch.ops.aten.addmm.default(primals_125, view_35, permute_22);  primals_125 = None
    view_36: "f32[32, 64, 384]" = torch.ops.aten.view.default(addmm_10, [32, 64, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_19: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_36)
    mul_194: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_36, sigmoid_19);  view_36 = sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_30: "f32[32, 64, 384]" = torch.ops.aten.clone.default(mul_194);  mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_37: "f32[2048, 384]" = torch.ops.aten.view.default(clone_30, [2048, 384]);  clone_30 = None
    permute_23: "f32[384, 192]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    addmm_11: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_127, view_37, permute_23);  primals_127 = None
    view_38: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_11, [32, 64, 192]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_31: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_38);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_136: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_133, clone_31);  add_133 = clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_30 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
    getitem_81: "f32[32, 64, 1]" = var_mean_30[0]
    getitem_82: "f32[32, 64, 1]" = var_mean_30[1];  var_mean_30 = None
    add_137: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_81, 1e-05);  getitem_81 = None
    rsqrt_30: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_30: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_136, getitem_82);  getitem_82 = None
    mul_195: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    mul_196: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_195, primals_128)
    add_138: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_196, primals_129);  mul_196 = primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_39: "f32[2048, 192]" = torch.ops.aten.view.default(add_138, [2048, 192]);  add_138 = None
    permute_24: "f32[192, 576]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_12: "f32[2048, 576]" = torch.ops.aten.addmm.default(primals_131, view_39, permute_24);  primals_131 = None
    view_40: "f32[32, 64, 576]" = torch.ops.aten.view.default(addmm_12, [32, 64, 576]);  addmm_12 = None
    view_41: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_40, [32, 64, 3, 4, 48]);  view_40 = None
    permute_25: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_41, [2, 0, 3, 1, 4]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_25);  permute_25 = None
    getitem_83: "f32[32, 4, 64, 48]" = unbind_3[0]
    getitem_84: "f32[32, 4, 64, 48]" = unbind_3[1]
    getitem_85: "f32[32, 4, 64, 48]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_83, getitem_84, getitem_85, None, True)
    getitem_86: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_3[0]
    getitem_87: "f32[32, 4, 64]" = _scaled_dot_product_efficient_attention_3[1]
    getitem_88: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
    getitem_89: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
    alias_3: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(getitem_86)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_26: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_86, [0, 2, 1, 3]);  getitem_86 = None
    view_42: "f32[32, 64, 192]" = torch.ops.aten.view.default(permute_26, [32, 64, 192]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_43: "f32[2048, 192]" = torch.ops.aten.view.default(view_42, [2048, 192]);  view_42 = None
    permute_27: "f32[192, 192]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    addmm_13: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_133, view_43, permute_27);  primals_133 = None
    view_44: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_13, [32, 64, 192]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_32: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_139: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_136, clone_32);  add_136 = clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_31 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
    getitem_90: "f32[32, 64, 1]" = var_mean_31[0]
    getitem_91: "f32[32, 64, 1]" = var_mean_31[1];  var_mean_31 = None
    add_140: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_31: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_31: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_139, getitem_91);  getitem_91 = None
    mul_197: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    mul_198: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_197, primals_134)
    add_141: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_198, primals_135);  mul_198 = primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_45: "f32[2048, 192]" = torch.ops.aten.view.default(add_141, [2048, 192]);  add_141 = None
    permute_28: "f32[192, 384]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_14: "f32[2048, 384]" = torch.ops.aten.addmm.default(primals_137, view_45, permute_28);  primals_137 = None
    view_46: "f32[32, 64, 384]" = torch.ops.aten.view.default(addmm_14, [32, 64, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_20: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_46)
    mul_199: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_46, sigmoid_20);  view_46 = sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_33: "f32[32, 64, 384]" = torch.ops.aten.clone.default(mul_199);  mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[2048, 384]" = torch.ops.aten.view.default(clone_33, [2048, 384]);  clone_33 = None
    permute_29: "f32[384, 192]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_15: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_139, view_47, permute_29);  primals_139 = None
    view_48: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_15, [32, 64, 192]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_34: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_142: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_139, clone_34);  add_139 = clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_32 = torch.ops.aten.var_mean.correction(add_142, [2], correction = 0, keepdim = True)
    getitem_92: "f32[32, 64, 1]" = var_mean_32[0]
    getitem_93: "f32[32, 64, 1]" = var_mean_32[1];  var_mean_32 = None
    add_143: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_32: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_32: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_142, getitem_93);  getitem_93 = None
    mul_200: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    mul_201: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_200, primals_140)
    add_144: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_201, primals_141);  mul_201 = primals_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_49: "f32[2048, 192]" = torch.ops.aten.view.default(add_144, [2048, 192]);  add_144 = None
    permute_30: "f32[192, 576]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
    addmm_16: "f32[2048, 576]" = torch.ops.aten.addmm.default(primals_143, view_49, permute_30);  primals_143 = None
    view_50: "f32[32, 64, 576]" = torch.ops.aten.view.default(addmm_16, [32, 64, 576]);  addmm_16 = None
    view_51: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_50, [32, 64, 3, 4, 48]);  view_50 = None
    permute_31: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_51, [2, 0, 3, 1, 4]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_31);  permute_31 = None
    getitem_94: "f32[32, 4, 64, 48]" = unbind_4[0]
    getitem_95: "f32[32, 4, 64, 48]" = unbind_4[1]
    getitem_96: "f32[32, 4, 64, 48]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_94, getitem_95, getitem_96, None, True)
    getitem_97: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_4[0]
    getitem_98: "f32[32, 4, 64]" = _scaled_dot_product_efficient_attention_4[1]
    getitem_99: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
    getitem_100: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None
    alias_4: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(getitem_97)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_32: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_97, [0, 2, 1, 3]);  getitem_97 = None
    view_52: "f32[32, 64, 192]" = torch.ops.aten.view.default(permute_32, [32, 64, 192]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_53: "f32[2048, 192]" = torch.ops.aten.view.default(view_52, [2048, 192]);  view_52 = None
    permute_33: "f32[192, 192]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_17: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_145, view_53, permute_33);  primals_145 = None
    view_54: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_17, [32, 64, 192]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_35: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_54);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_145: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_142, clone_35);  add_142 = clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_33 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
    getitem_101: "f32[32, 64, 1]" = var_mean_33[0]
    getitem_102: "f32[32, 64, 1]" = var_mean_33[1];  var_mean_33 = None
    add_146: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_101, 1e-05);  getitem_101 = None
    rsqrt_33: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_33: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_145, getitem_102);  getitem_102 = None
    mul_202: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    mul_203: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_202, primals_146)
    add_147: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_203, primals_147);  mul_203 = primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_55: "f32[2048, 192]" = torch.ops.aten.view.default(add_147, [2048, 192]);  add_147 = None
    permute_34: "f32[192, 384]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_18: "f32[2048, 384]" = torch.ops.aten.addmm.default(primals_149, view_55, permute_34);  primals_149 = None
    view_56: "f32[32, 64, 384]" = torch.ops.aten.view.default(addmm_18, [32, 64, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_21: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_56)
    mul_204: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_56, sigmoid_21);  view_56 = sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_36: "f32[32, 64, 384]" = torch.ops.aten.clone.default(mul_204);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_57: "f32[2048, 384]" = torch.ops.aten.view.default(clone_36, [2048, 384]);  clone_36 = None
    permute_35: "f32[384, 192]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_19: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_151, view_57, permute_35);  primals_151 = None
    view_58: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_19, [32, 64, 192]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_37: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_58);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_148: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_145, clone_37);  add_145 = clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_34 = torch.ops.aten.var_mean.correction(add_148, [2], correction = 0, keepdim = True)
    getitem_103: "f32[32, 64, 1]" = var_mean_34[0]
    getitem_104: "f32[32, 64, 1]" = var_mean_34[1];  var_mean_34 = None
    add_149: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_103, 1e-05);  getitem_103 = None
    rsqrt_34: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    sub_34: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_148, getitem_104);  getitem_104 = None
    mul_205: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    mul_206: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_205, primals_152)
    add_150: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_206, primals_153);  mul_206 = primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_59: "f32[2048, 192]" = torch.ops.aten.view.default(add_150, [2048, 192]);  add_150 = None
    permute_36: "f32[192, 576]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_20: "f32[2048, 576]" = torch.ops.aten.addmm.default(primals_155, view_59, permute_36);  primals_155 = None
    view_60: "f32[32, 64, 576]" = torch.ops.aten.view.default(addmm_20, [32, 64, 576]);  addmm_20 = None
    view_61: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_60, [32, 64, 3, 4, 48]);  view_60 = None
    permute_37: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_61, [2, 0, 3, 1, 4]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_37);  permute_37 = None
    getitem_105: "f32[32, 4, 64, 48]" = unbind_5[0]
    getitem_106: "f32[32, 4, 64, 48]" = unbind_5[1]
    getitem_107: "f32[32, 4, 64, 48]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_105, getitem_106, getitem_107, None, True)
    getitem_108: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_5[0]
    getitem_109: "f32[32, 4, 64]" = _scaled_dot_product_efficient_attention_5[1]
    getitem_110: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
    getitem_111: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None
    alias_5: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(getitem_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_38: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3]);  getitem_108 = None
    view_62: "f32[32, 64, 192]" = torch.ops.aten.view.default(permute_38, [32, 64, 192]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_63: "f32[2048, 192]" = torch.ops.aten.view.default(view_62, [2048, 192]);  view_62 = None
    permute_39: "f32[192, 192]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_21: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_157, view_63, permute_39);  primals_157 = None
    view_64: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_21, [32, 64, 192]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_38: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_64);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_151: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_148, clone_38);  add_148 = clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_35 = torch.ops.aten.var_mean.correction(add_151, [2], correction = 0, keepdim = True)
    getitem_112: "f32[32, 64, 1]" = var_mean_35[0]
    getitem_113: "f32[32, 64, 1]" = var_mean_35[1];  var_mean_35 = None
    add_152: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_35: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_35: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_151, getitem_113);  getitem_113 = None
    mul_207: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    mul_208: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_207, primals_158)
    add_153: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_208, primals_159);  mul_208 = primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_65: "f32[2048, 192]" = torch.ops.aten.view.default(add_153, [2048, 192]);  add_153 = None
    permute_40: "f32[192, 384]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_22: "f32[2048, 384]" = torch.ops.aten.addmm.default(primals_161, view_65, permute_40);  primals_161 = None
    view_66: "f32[32, 64, 384]" = torch.ops.aten.view.default(addmm_22, [32, 64, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_22: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_66)
    mul_209: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_66, sigmoid_22);  view_66 = sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_39: "f32[32, 64, 384]" = torch.ops.aten.clone.default(mul_209);  mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_67: "f32[2048, 384]" = torch.ops.aten.view.default(clone_39, [2048, 384]);  clone_39 = None
    permute_41: "f32[384, 192]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_23: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_163, view_67, permute_41);  primals_163 = None
    view_68: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_23, [32, 64, 192]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_40: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_154: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_151, clone_40);  add_151 = clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
    getitem_114: "f32[32, 64, 1]" = var_mean_36[0]
    getitem_115: "f32[32, 64, 1]" = var_mean_36[1];  var_mean_36 = None
    add_155: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_36: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_36: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_154, getitem_115);  add_154 = getitem_115 = None
    mul_210: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    mul_211: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_210, primals_164)
    add_156: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_211, primals_165);  mul_211 = primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    view_69: "f32[8, 4, 64, 192]" = torch.ops.aten.view.default(add_156, [8, 4, 64, -1]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    permute_42: "f32[8, 192, 64, 4]" = torch.ops.aten.permute.default(view_69, [0, 3, 2, 1]);  view_69 = None
    clone_41: "f32[8, 192, 64, 4]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
    view_70: "f32[12288, 8, 2, 2]" = torch.ops.aten.view.default(clone_41, [12288, 8, 2, 2]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    permute_43: "f32[12288, 2, 8, 2]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    clone_42: "f32[12288, 2, 8, 2]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    view_71: "f32[8, 192, 16, 16]" = torch.ops.aten.view.default(clone_42, [8, 192, 16, 16]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(view_71, primals_166, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_157: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[1, 128, 1, 1]" = var_mean_37[0]
    getitem_117: "f32[1, 128, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_158: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
    rsqrt_37: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_37: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_117)
    mul_212: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_69: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
    squeeze_70: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_213: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_214: "f32[128]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
    add_159: "f32[128]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
    squeeze_71: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
    mul_215: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0004885197850513);  squeeze_71 = None
    mul_216: "f32[128]" = torch.ops.aten.mul.Tensor(mul_215, 0.1);  mul_215 = None
    mul_217: "f32[128]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_160: "f32[128]" = torch.ops.aten.add.Tensor(mul_216, mul_217);  mul_216 = mul_217 = None
    unsqueeze_92: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_218: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_212, unsqueeze_93);  mul_212 = unsqueeze_93 = None
    unsqueeze_94: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_161: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_95);  mul_218 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_43: "f32[8, 128, 16, 16]" = torch.ops.aten.clone.default(add_161)
    sigmoid_23: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(add_161)
    mul_219: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_161, sigmoid_23);  add_161 = sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    cat_1: "f32[8, 256, 16, 16]" = torch.ops.aten.cat.default([add_125, mul_219], 1);  mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_26: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(cat_1, primals_167, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_162: "i64[]" = torch.ops.aten.add.Tensor(primals_288, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 128, 1, 1]" = var_mean_38[0]
    getitem_119: "f32[1, 128, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_163: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_38: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_38: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_119)
    mul_220: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_72: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_73: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_221: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_222: "f32[128]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
    add_164: "f32[128]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    squeeze_74: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_223: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0004885197850513);  squeeze_74 = None
    mul_224: "f32[128]" = torch.ops.aten.mul.Tensor(mul_223, 0.1);  mul_223 = None
    mul_225: "f32[128]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_165: "f32[128]" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
    unsqueeze_96: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_226: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_97);  mul_220 = unsqueeze_97 = None
    unsqueeze_98: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_166: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_99);  mul_226 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_44: "f32[8, 128, 16, 16]" = torch.ops.aten.clone.default(add_166)
    sigmoid_24: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(add_166)
    mul_227: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_166, sigmoid_24);  add_166 = sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(mul_227, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_167: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 512, 1, 1]" = var_mean_39[0]
    getitem_121: "f32[1, 512, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_168: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
    rsqrt_39: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_39: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_121)
    mul_228: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_75: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_76: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_229: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_230: "f32[512]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
    add_169: "f32[512]" = torch.ops.aten.add.Tensor(mul_229, mul_230);  mul_229 = mul_230 = None
    squeeze_77: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_231: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0004885197850513);  squeeze_77 = None
    mul_232: "f32[512]" = torch.ops.aten.mul.Tensor(mul_231, 0.1);  mul_231 = None
    mul_233: "f32[512]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_170: "f32[512]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    unsqueeze_100: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_234: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_228, unsqueeze_101);  mul_228 = unsqueeze_101 = None
    unsqueeze_102: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_171: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_234, unsqueeze_103);  mul_234 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_45: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(add_171)
    sigmoid_25: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(add_171)
    mul_235: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_171, sigmoid_25);  add_171 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(mul_235, primals_169, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_172: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[1, 512, 1, 1]" = var_mean_40[0]
    getitem_123: "f32[1, 512, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_173: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
    rsqrt_40: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    sub_40: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_123)
    mul_236: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_78: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    squeeze_79: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_237: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_238: "f32[512]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
    add_174: "f32[512]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
    squeeze_80: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
    mul_239: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0019569471624266);  squeeze_80 = None
    mul_240: "f32[512]" = torch.ops.aten.mul.Tensor(mul_239, 0.1);  mul_239 = None
    mul_241: "f32[512]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_175: "f32[512]" = torch.ops.aten.add.Tensor(mul_240, mul_241);  mul_240 = mul_241 = None
    unsqueeze_104: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_242: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_236, unsqueeze_105);  mul_236 = unsqueeze_105 = None
    unsqueeze_106: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_176: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_107);  mul_242 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_46: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(add_176)
    sigmoid_26: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_176)
    mul_243: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_176, sigmoid_26);  add_176 = sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(mul_243, primals_170, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_177: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 160, 1, 1]" = var_mean_41[0]
    getitem_125: "f32[1, 160, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_178: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
    rsqrt_41: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_41: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_125)
    mul_244: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_81: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_82: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_245: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_246: "f32[160]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
    add_179: "f32[160]" = torch.ops.aten.add.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None
    squeeze_83: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_247: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0019569471624266);  squeeze_83 = None
    mul_248: "f32[160]" = torch.ops.aten.mul.Tensor(mul_247, 0.1);  mul_247 = None
    mul_249: "f32[160]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_180: "f32[160]" = torch.ops.aten.add.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None
    unsqueeze_108: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_109: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_250: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_109);  mul_244 = unsqueeze_109 = None
    unsqueeze_110: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_111: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_181: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_250, unsqueeze_111);  mul_250 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(add_181, primals_171, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_182: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 160, 1, 1]" = var_mean_42[0]
    getitem_127: "f32[1, 160, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_183: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
    rsqrt_42: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    sub_42: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_127)
    mul_251: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_84: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_85: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_252: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_253: "f32[160]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
    add_184: "f32[160]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    squeeze_86: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_254: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0019569471624266);  squeeze_86 = None
    mul_255: "f32[160]" = torch.ops.aten.mul.Tensor(mul_254, 0.1);  mul_254 = None
    mul_256: "f32[160]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_185: "f32[160]" = torch.ops.aten.add.Tensor(mul_255, mul_256);  mul_255 = mul_256 = None
    unsqueeze_112: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_113: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_257: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_113);  mul_251 = unsqueeze_113 = None
    unsqueeze_114: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_115: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_186: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_257, unsqueeze_115);  mul_257 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_47: "f32[8, 160, 8, 8]" = torch.ops.aten.clone.default(add_186)
    sigmoid_27: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(add_186)
    mul_258: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_186, sigmoid_27);  add_186 = sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_31: "f32[8, 240, 8, 8]" = torch.ops.aten.convolution.default(mul_258, primals_172, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    view_72: "f32[7680, 2, 4, 2]" = torch.ops.aten.view.default(convolution_31, [7680, 2, 4, 2]);  convolution_31 = None
    permute_44: "f32[7680, 4, 2, 2]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    clone_48: "f32[7680, 4, 2, 2]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
    view_73: "f32[8, 240, 16, 4]" = torch.ops.aten.view.default(clone_48, [8, 240, 16, 4]);  clone_48 = None
    permute_45: "f32[8, 4, 16, 240]" = torch.ops.aten.permute.default(view_73, [0, 3, 2, 1]);  view_73 = None
    clone_49: "f32[8, 4, 16, 240]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_74: "f32[32, 16, 240]" = torch.ops.aten.view.default(clone_49, [32, 16, 240]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_43 = torch.ops.aten.var_mean.correction(view_74, [2], correction = 0, keepdim = True)
    getitem_128: "f32[32, 16, 1]" = var_mean_43[0]
    getitem_129: "f32[32, 16, 1]" = var_mean_43[1];  var_mean_43 = None
    add_187: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
    rsqrt_43: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    sub_43: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(view_74, getitem_129);  getitem_129 = None
    mul_259: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    mul_260: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_259, primals_173)
    add_188: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_260, primals_174);  mul_260 = primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_75: "f32[512, 240]" = torch.ops.aten.view.default(add_188, [512, 240]);  add_188 = None
    permute_46: "f32[240, 720]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_24: "f32[512, 720]" = torch.ops.aten.addmm.default(primals_176, view_75, permute_46);  primals_176 = None
    view_76: "f32[32, 16, 720]" = torch.ops.aten.view.default(addmm_24, [32, 16, 720]);  addmm_24 = None
    view_77: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.view.default(view_76, [32, 16, 3, 4, 60]);  view_76 = None
    permute_47: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.permute.default(view_77, [2, 0, 3, 1, 4]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_47);  permute_47 = None
    getitem_130: "f32[32, 4, 16, 60]" = unbind_6[0]
    getitem_131: "f32[32, 4, 16, 60]" = unbind_6[1]
    getitem_132: "f32[32, 4, 16, 60]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_130, getitem_131, getitem_132, None, True)
    getitem_133: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_6[0]
    getitem_134: "f32[32, 4, 32]" = _scaled_dot_product_efficient_attention_6[1]
    getitem_135: "i64[]" = _scaled_dot_product_efficient_attention_6[2]
    getitem_136: "i64[]" = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None
    alias_6: "f32[32, 4, 16, 60]" = torch.ops.aten.alias.default(getitem_133)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_48: "f32[32, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
    view_78: "f32[32, 16, 240]" = torch.ops.aten.view.default(permute_48, [32, 16, 240]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_79: "f32[512, 240]" = torch.ops.aten.view.default(view_78, [512, 240]);  view_78 = None
    permute_49: "f32[240, 240]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    addmm_25: "f32[512, 240]" = torch.ops.aten.addmm.default(primals_178, view_79, permute_49);  primals_178 = None
    view_80: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_25, [32, 16, 240]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_50: "f32[32, 16, 240]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_189: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(view_74, clone_50);  view_74 = clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_44 = torch.ops.aten.var_mean.correction(add_189, [2], correction = 0, keepdim = True)
    getitem_137: "f32[32, 16, 1]" = var_mean_44[0]
    getitem_138: "f32[32, 16, 1]" = var_mean_44[1];  var_mean_44 = None
    add_190: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_137, 1e-05);  getitem_137 = None
    rsqrt_44: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
    sub_44: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_189, getitem_138);  getitem_138 = None
    mul_261: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    mul_262: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_261, primals_179)
    add_191: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_262, primals_180);  mul_262 = primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_81: "f32[512, 240]" = torch.ops.aten.view.default(add_191, [512, 240]);  add_191 = None
    permute_50: "f32[240, 480]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_26: "f32[512, 480]" = torch.ops.aten.addmm.default(primals_182, view_81, permute_50);  primals_182 = None
    view_82: "f32[32, 16, 480]" = torch.ops.aten.view.default(addmm_26, [32, 16, 480])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_28: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_82)
    mul_263: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_82, sigmoid_28);  view_82 = sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_51: "f32[32, 16, 480]" = torch.ops.aten.clone.default(mul_263);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_83: "f32[512, 480]" = torch.ops.aten.view.default(clone_51, [512, 480]);  clone_51 = None
    permute_51: "f32[480, 240]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    addmm_27: "f32[512, 240]" = torch.ops.aten.addmm.default(primals_184, view_83, permute_51);  primals_184 = None
    view_84: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_27, [32, 16, 240]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_52: "f32[32, 16, 240]" = torch.ops.aten.clone.default(view_84);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_192: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_189, clone_52);  add_189 = clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_45 = torch.ops.aten.var_mean.correction(add_192, [2], correction = 0, keepdim = True)
    getitem_139: "f32[32, 16, 1]" = var_mean_45[0]
    getitem_140: "f32[32, 16, 1]" = var_mean_45[1];  var_mean_45 = None
    add_193: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_139, 1e-05);  getitem_139 = None
    rsqrt_45: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    sub_45: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_192, getitem_140);  getitem_140 = None
    mul_264: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    mul_265: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_264, primals_185)
    add_194: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_265, primals_186);  mul_265 = primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_85: "f32[512, 240]" = torch.ops.aten.view.default(add_194, [512, 240]);  add_194 = None
    permute_52: "f32[240, 720]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    addmm_28: "f32[512, 720]" = torch.ops.aten.addmm.default(primals_188, view_85, permute_52);  primals_188 = None
    view_86: "f32[32, 16, 720]" = torch.ops.aten.view.default(addmm_28, [32, 16, 720]);  addmm_28 = None
    view_87: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.view.default(view_86, [32, 16, 3, 4, 60]);  view_86 = None
    permute_53: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.permute.default(view_87, [2, 0, 3, 1, 4]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_53);  permute_53 = None
    getitem_141: "f32[32, 4, 16, 60]" = unbind_7[0]
    getitem_142: "f32[32, 4, 16, 60]" = unbind_7[1]
    getitem_143: "f32[32, 4, 16, 60]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_141, getitem_142, getitem_143, None, True)
    getitem_144: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_7[0]
    getitem_145: "f32[32, 4, 32]" = _scaled_dot_product_efficient_attention_7[1]
    getitem_146: "i64[]" = _scaled_dot_product_efficient_attention_7[2]
    getitem_147: "i64[]" = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None
    alias_7: "f32[32, 4, 16, 60]" = torch.ops.aten.alias.default(getitem_144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_54: "f32[32, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None
    view_88: "f32[32, 16, 240]" = torch.ops.aten.view.default(permute_54, [32, 16, 240]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_89: "f32[512, 240]" = torch.ops.aten.view.default(view_88, [512, 240]);  view_88 = None
    permute_55: "f32[240, 240]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    addmm_29: "f32[512, 240]" = torch.ops.aten.addmm.default(primals_190, view_89, permute_55);  primals_190 = None
    view_90: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_29, [32, 16, 240]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_53: "f32[32, 16, 240]" = torch.ops.aten.clone.default(view_90);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_195: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_192, clone_53);  add_192 = clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_46 = torch.ops.aten.var_mean.correction(add_195, [2], correction = 0, keepdim = True)
    getitem_148: "f32[32, 16, 1]" = var_mean_46[0]
    getitem_149: "f32[32, 16, 1]" = var_mean_46[1];  var_mean_46 = None
    add_196: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
    rsqrt_46: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    sub_46: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_195, getitem_149);  getitem_149 = None
    mul_266: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    mul_267: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_266, primals_191)
    add_197: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_267, primals_192);  mul_267 = primals_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_91: "f32[512, 240]" = torch.ops.aten.view.default(add_197, [512, 240]);  add_197 = None
    permute_56: "f32[240, 480]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
    addmm_30: "f32[512, 480]" = torch.ops.aten.addmm.default(primals_194, view_91, permute_56);  primals_194 = None
    view_92: "f32[32, 16, 480]" = torch.ops.aten.view.default(addmm_30, [32, 16, 480])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_29: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_92)
    mul_268: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_92, sigmoid_29);  view_92 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_54: "f32[32, 16, 480]" = torch.ops.aten.clone.default(mul_268);  mul_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_93: "f32[512, 480]" = torch.ops.aten.view.default(clone_54, [512, 480]);  clone_54 = None
    permute_57: "f32[480, 240]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    addmm_31: "f32[512, 240]" = torch.ops.aten.addmm.default(primals_196, view_93, permute_57);  primals_196 = None
    view_94: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_31, [32, 16, 240]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_55: "f32[32, 16, 240]" = torch.ops.aten.clone.default(view_94);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_198: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_195, clone_55);  add_195 = clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_47 = torch.ops.aten.var_mean.correction(add_198, [2], correction = 0, keepdim = True)
    getitem_150: "f32[32, 16, 1]" = var_mean_47[0]
    getitem_151: "f32[32, 16, 1]" = var_mean_47[1];  var_mean_47 = None
    add_199: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05);  getitem_150 = None
    rsqrt_47: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_47: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_198, getitem_151);  getitem_151 = None
    mul_269: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    mul_270: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_269, primals_197)
    add_200: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_270, primals_198);  mul_270 = primals_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_95: "f32[512, 240]" = torch.ops.aten.view.default(add_200, [512, 240]);  add_200 = None
    permute_58: "f32[240, 720]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    addmm_32: "f32[512, 720]" = torch.ops.aten.addmm.default(primals_200, view_95, permute_58);  primals_200 = None
    view_96: "f32[32, 16, 720]" = torch.ops.aten.view.default(addmm_32, [32, 16, 720]);  addmm_32 = None
    view_97: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.view.default(view_96, [32, 16, 3, 4, 60]);  view_96 = None
    permute_59: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.permute.default(view_97, [2, 0, 3, 1, 4]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_59);  permute_59 = None
    getitem_152: "f32[32, 4, 16, 60]" = unbind_8[0]
    getitem_153: "f32[32, 4, 16, 60]" = unbind_8[1]
    getitem_154: "f32[32, 4, 16, 60]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_152, getitem_153, getitem_154, None, True)
    getitem_155: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_8[0]
    getitem_156: "f32[32, 4, 32]" = _scaled_dot_product_efficient_attention_8[1]
    getitem_157: "i64[]" = _scaled_dot_product_efficient_attention_8[2]
    getitem_158: "i64[]" = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None
    alias_8: "f32[32, 4, 16, 60]" = torch.ops.aten.alias.default(getitem_155)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_60: "f32[32, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3]);  getitem_155 = None
    view_98: "f32[32, 16, 240]" = torch.ops.aten.view.default(permute_60, [32, 16, 240]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_99: "f32[512, 240]" = torch.ops.aten.view.default(view_98, [512, 240]);  view_98 = None
    permute_61: "f32[240, 240]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
    addmm_33: "f32[512, 240]" = torch.ops.aten.addmm.default(primals_202, view_99, permute_61);  primals_202 = None
    view_100: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_33, [32, 16, 240]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_56: "f32[32, 16, 240]" = torch.ops.aten.clone.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_201: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_198, clone_56);  add_198 = clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_48 = torch.ops.aten.var_mean.correction(add_201, [2], correction = 0, keepdim = True)
    getitem_159: "f32[32, 16, 1]" = var_mean_48[0]
    getitem_160: "f32[32, 16, 1]" = var_mean_48[1];  var_mean_48 = None
    add_202: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_159, 1e-05);  getitem_159 = None
    rsqrt_48: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    sub_48: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_201, getitem_160);  getitem_160 = None
    mul_271: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    mul_272: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_271, primals_203)
    add_203: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_272, primals_204);  mul_272 = primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[512, 240]" = torch.ops.aten.view.default(add_203, [512, 240]);  add_203 = None
    permute_62: "f32[240, 480]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    addmm_34: "f32[512, 480]" = torch.ops.aten.addmm.default(primals_206, view_101, permute_62);  primals_206 = None
    view_102: "f32[32, 16, 480]" = torch.ops.aten.view.default(addmm_34, [32, 16, 480])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_30: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_102)
    mul_273: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_102, sigmoid_30);  view_102 = sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_57: "f32[32, 16, 480]" = torch.ops.aten.clone.default(mul_273);  mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_103: "f32[512, 480]" = torch.ops.aten.view.default(clone_57, [512, 480]);  clone_57 = None
    permute_63: "f32[480, 240]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    addmm_35: "f32[512, 240]" = torch.ops.aten.addmm.default(primals_208, view_103, permute_63);  primals_208 = None
    view_104: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_35, [32, 16, 240]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_58: "f32[32, 16, 240]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_204: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_201, clone_58);  add_201 = clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    var_mean_49 = torch.ops.aten.var_mean.correction(add_204, [2], correction = 0, keepdim = True)
    getitem_161: "f32[32, 16, 1]" = var_mean_49[0]
    getitem_162: "f32[32, 16, 1]" = var_mean_49[1];  var_mean_49 = None
    add_205: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_161, 1e-05);  getitem_161 = None
    rsqrt_49: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    sub_49: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_204, getitem_162);  add_204 = getitem_162 = None
    mul_274: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    mul_275: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_274, primals_209)
    add_206: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_275, primals_210);  mul_275 = primals_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    view_105: "f32[8, 4, 16, 240]" = torch.ops.aten.view.default(add_206, [8, 4, 16, -1]);  add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    permute_64: "f32[8, 240, 16, 4]" = torch.ops.aten.permute.default(view_105, [0, 3, 2, 1]);  view_105 = None
    clone_59: "f32[8, 240, 16, 4]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
    view_106: "f32[7680, 4, 2, 2]" = torch.ops.aten.view.default(clone_59, [7680, 4, 2, 2]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    permute_65: "f32[7680, 2, 4, 2]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
    clone_60: "f32[7680, 2, 4, 2]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
    view_107: "f32[8, 240, 8, 8]" = torch.ops.aten.view.default(clone_60, [8, 240, 8, 8]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(view_107, primals_211, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_207: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_163: "f32[1, 160, 1, 1]" = var_mean_50[0]
    getitem_164: "f32[1, 160, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_208: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_163, 1e-05)
    rsqrt_50: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_50: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_164)
    mul_276: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_87: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
    squeeze_88: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_277: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_278: "f32[160]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
    add_209: "f32[160]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    squeeze_89: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
    mul_279: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0019569471624266);  squeeze_89 = None
    mul_280: "f32[160]" = torch.ops.aten.mul.Tensor(mul_279, 0.1);  mul_279 = None
    mul_281: "f32[160]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_210: "f32[160]" = torch.ops.aten.add.Tensor(mul_280, mul_281);  mul_280 = mul_281 = None
    unsqueeze_116: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_117: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_282: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_276, unsqueeze_117);  mul_276 = unsqueeze_117 = None
    unsqueeze_118: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_119: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_211: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_282, unsqueeze_119);  mul_282 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_61: "f32[8, 160, 8, 8]" = torch.ops.aten.clone.default(add_211)
    sigmoid_31: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(add_211)
    mul_283: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_211, sigmoid_31);  add_211 = sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    cat_2: "f32[8, 320, 8, 8]" = torch.ops.aten.cat.default([add_181, mul_283], 1);  mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(cat_2, primals_212, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_165: "f32[1, 160, 1, 1]" = var_mean_51[0]
    getitem_166: "f32[1, 160, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_213: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_165, 1e-05)
    rsqrt_51: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_51: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_166)
    mul_284: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_90: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
    squeeze_91: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_285: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_286: "f32[160]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_214: "f32[160]" = torch.ops.aten.add.Tensor(mul_285, mul_286);  mul_285 = mul_286 = None
    squeeze_92: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
    mul_287: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0019569471624266);  squeeze_92 = None
    mul_288: "f32[160]" = torch.ops.aten.mul.Tensor(mul_287, 0.1);  mul_287 = None
    mul_289: "f32[160]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_215: "f32[160]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    unsqueeze_120: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_121: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_290: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_121);  mul_284 = unsqueeze_121 = None
    unsqueeze_122: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_123: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_216: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_123);  mul_290 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_62: "f32[8, 160, 8, 8]" = torch.ops.aten.clone.default(add_216)
    sigmoid_32: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(add_216)
    mul_291: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_216, sigmoid_32);  add_216 = sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(mul_291, primals_213, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_217: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_167: "f32[1, 640, 1, 1]" = var_mean_52[0]
    getitem_168: "f32[1, 640, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_218: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-05)
    rsqrt_52: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    sub_52: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_168)
    mul_292: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_93: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
    squeeze_94: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_293: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_294: "f32[640]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_219: "f32[640]" = torch.ops.aten.add.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
    squeeze_95: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
    mul_295: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0019569471624266);  squeeze_95 = None
    mul_296: "f32[640]" = torch.ops.aten.mul.Tensor(mul_295, 0.1);  mul_295 = None
    mul_297: "f32[640]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_220: "f32[640]" = torch.ops.aten.add.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
    unsqueeze_124: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_125: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_298: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_125);  mul_292 = unsqueeze_125 = None
    unsqueeze_126: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_127: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_221: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_298, unsqueeze_127);  mul_298 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_63: "f32[8, 640, 8, 8]" = torch.ops.aten.clone.default(add_221)
    sigmoid_33: "f32[8, 640, 8, 8]" = torch.ops.aten.sigmoid.default(add_221)
    mul_299: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(add_221, sigmoid_33);  add_221 = sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 640, 1, 1]" = torch.ops.aten.mean.dim(mul_299, [-1, -2], True);  mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_108: "f32[8, 640]" = torch.ops.aten.view.default(mean, [8, 640]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone_64: "f32[8, 640]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute_66: "f32[640, 1000]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
    addmm_36: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_215, clone_64, permute_66);  primals_215 = None
    permute_67: "f32[1000, 640]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_34: "f32[8, 640, 8, 8]" = torch.ops.aten.sigmoid.default(clone_63)
    full_default: "f32[8, 640, 8, 8]" = torch.ops.aten.full.default([8, 640, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_53: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(full_default, sigmoid_34);  full_default = None
    mul_300: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(clone_63, sub_53);  clone_63 = sub_53 = None
    add_222: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Scalar(mul_300, 1);  mul_300 = None
    mul_301: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_34, add_222);  sigmoid_34 = add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_128: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_129: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 2);  unsqueeze_128 = None
    unsqueeze_130: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 3);  unsqueeze_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_35: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(clone_62)
    full_default_1: "f32[8, 160, 8, 8]" = torch.ops.aten.full.default([8, 160, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_58: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_35)
    mul_312: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(clone_62, sub_58);  clone_62 = sub_58 = None
    add_223: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Scalar(mul_312, 1);  mul_312 = None
    mul_313: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_35, add_223);  sigmoid_35 = add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_140: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_141: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 2);  unsqueeze_140 = None
    unsqueeze_142: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 3);  unsqueeze_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_36: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(clone_61)
    sub_63: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_36)
    mul_324: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(clone_61, sub_63);  clone_61 = sub_63 = None
    add_224: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Scalar(mul_324, 1);  mul_324 = None
    mul_325: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_36, add_224);  sigmoid_36 = add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_152: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_153: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 2);  unsqueeze_152 = None
    unsqueeze_154: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 3);  unsqueeze_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    div_1: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 240);  rsqrt_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_76: "f32[240, 480]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_81: "f32[480, 240]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_2: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 240);  rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_85: "f32[240, 240]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_9: "f32[32, 4, 16, 60]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_91: "f32[720, 240]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_3: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 240);  rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_95: "f32[240, 480]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_100: "f32[480, 240]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_4: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 240);  rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_104: "f32[240, 240]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_10: "f32[32, 4, 16, 60]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_110: "f32[720, 240]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_5: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 240);  rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_114: "f32[240, 480]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_119: "f32[480, 240]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_6: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 240);  rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_123: "f32[240, 240]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_11: "f32[32, 4, 16, 60]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_129: "f32[720, 240]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_7: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 240);  rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_40: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(clone_47)
    sub_92: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_40);  full_default_1 = None
    mul_394: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(clone_47, sub_92);  clone_47 = sub_92 = None
    add_234: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Scalar(mul_394, 1);  mul_394 = None
    mul_395: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_40, add_234);  sigmoid_40 = add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_164: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_165: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 2);  unsqueeze_164 = None
    unsqueeze_166: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 3);  unsqueeze_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_176: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_177: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
    unsqueeze_178: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_41: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(clone_46)
    full_default_7: "f32[8, 512, 8, 8]" = torch.ops.aten.full.default([8, 512, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_101: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_default_7, sigmoid_41);  full_default_7 = None
    mul_415: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(clone_46, sub_101);  clone_46 = sub_101 = None
    add_236: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_415, 1);  mul_415 = None
    mul_416: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_41, add_236);  sigmoid_41 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_188: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_189: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
    unsqueeze_190: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_42: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(clone_45)
    full_default_8: "f32[8, 512, 16, 16]" = torch.ops.aten.full.default([8, 512, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_106: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_8, sigmoid_42);  full_default_8 = None
    mul_427: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(clone_45, sub_106);  clone_45 = sub_106 = None
    add_237: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Scalar(mul_427, 1);  mul_427 = None
    mul_428: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_42, add_237);  sigmoid_42 = add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_200: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_201: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
    unsqueeze_202: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_43: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(clone_44)
    full_default_9: "f32[8, 128, 16, 16]" = torch.ops.aten.full.default([8, 128, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_111: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_9, sigmoid_43)
    mul_439: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(clone_44, sub_111);  clone_44 = sub_111 = None
    add_238: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Scalar(mul_439, 1);  mul_439 = None
    mul_440: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_43, add_238);  sigmoid_43 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_212: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_213: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
    unsqueeze_214: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_44: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(clone_43)
    sub_116: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_9, sigmoid_44)
    mul_451: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(clone_43, sub_116);  clone_43 = sub_116 = None
    add_239: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Scalar(mul_451, 1);  mul_451 = None
    mul_452: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_44, add_239);  sigmoid_44 = add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_224: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_225: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
    unsqueeze_226: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    div_8: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 192);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_142: "f32[192, 384]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_147: "f32[384, 192]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_9: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 192);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_151: "f32[192, 192]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_12: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_157: "f32[576, 192]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_10: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 192);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_161: "f32[192, 384]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_166: "f32[384, 192]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_11: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 192);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_170: "f32[192, 192]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_13: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_176: "f32[576, 192]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_12: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 192);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_180: "f32[192, 384]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_185: "f32[384, 192]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_13: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 192);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_189: "f32[192, 192]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_14: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_195: "f32[576, 192]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_14: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 192);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_199: "f32[192, 384]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_204: "f32[384, 192]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_15: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 192);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_208: "f32[192, 192]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_15: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_214: "f32[576, 192]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_16: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 192);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_49: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(clone_26)
    sub_152: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_9, sigmoid_49);  full_default_9 = None
    mul_538: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(clone_26, sub_152);  clone_26 = sub_152 = None
    add_252: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Scalar(mul_538, 1);  mul_538 = None
    mul_539: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_49, add_252);  sigmoid_49 = add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_236: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_237: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
    unsqueeze_238: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_248: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_249: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
    unsqueeze_250: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_50: "f32[8, 384, 16, 16]" = torch.ops.aten.sigmoid.default(clone_25)
    full_default_16: "f32[8, 384, 16, 16]" = torch.ops.aten.full.default([8, 384, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_161: "f32[8, 384, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_16, sigmoid_50);  full_default_16 = None
    mul_559: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(clone_25, sub_161);  clone_25 = sub_161 = None
    add_254: "f32[8, 384, 16, 16]" = torch.ops.aten.add.Scalar(mul_559, 1);  mul_559 = None
    mul_560: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_50, add_254);  sigmoid_50 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_260: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_261: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_51: "f32[8, 384, 32, 32]" = torch.ops.aten.sigmoid.default(clone_24)
    full_default_17: "f32[8, 384, 32, 32]" = torch.ops.aten.full.default([8, 384, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_166: "f32[8, 384, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_17, sigmoid_51);  full_default_17 = None
    mul_571: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(clone_24, sub_166);  clone_24 = sub_166 = None
    add_255: "f32[8, 384, 32, 32]" = torch.ops.aten.add.Scalar(mul_571, 1);  mul_571 = None
    mul_572: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_51, add_255);  sigmoid_51 = add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_272: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_273: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_52: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(clone_23)
    full_default_18: "f32[8, 96, 32, 32]" = torch.ops.aten.full.default([8, 96, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_171: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_18, sigmoid_52)
    mul_583: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(clone_23, sub_171);  clone_23 = sub_171 = None
    add_256: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Scalar(mul_583, 1);  mul_583 = None
    mul_584: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_52, add_256);  sigmoid_52 = add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_284: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_285: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_53: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(clone_22)
    sub_176: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_18, sigmoid_53)
    mul_595: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(clone_22, sub_176);  clone_22 = sub_176 = None
    add_257: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Scalar(mul_595, 1);  mul_595 = None
    mul_596: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_53, add_257);  sigmoid_53 = add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_296: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_297: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    div_17: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 144);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_227: "f32[144, 288]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_232: "f32[288, 144]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_18: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 144);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_236: "f32[144, 144]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_16: "f32[32, 4, 256, 36]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_242: "f32[432, 144]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_19: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 144);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_246: "f32[144, 288]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_251: "f32[288, 144]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_20: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 144);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_255: "f32[144, 144]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_17: "f32[32, 4, 256, 36]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_261: "f32[432, 144]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_21: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 144);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_56: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(clone_11)
    sub_198: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_18, sigmoid_56);  full_default_18 = None
    mul_648: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(clone_11, sub_198);  clone_11 = sub_198 = None
    add_264: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Scalar(mul_648, 1);  mul_648 = None
    mul_649: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_56, add_264);  sigmoid_56 = add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_308: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_309: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_320: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_321: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_57: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(clone_10)
    full_default_23: "f32[8, 256, 32, 32]" = torch.ops.aten.full.default([8, 256, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_207: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_57);  full_default_23 = None
    mul_669: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(clone_10, sub_207);  clone_10 = sub_207 = None
    add_266: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Scalar(mul_669, 1);  mul_669 = None
    mul_670: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_57, add_266);  sigmoid_57 = add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_332: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_333: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_58: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_9)
    full_default_24: "f32[8, 256, 64, 64]" = torch.ops.aten.full.default([8, 256, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_212: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_24, sigmoid_58)
    mul_681: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_9, sub_212);  clone_9 = sub_212 = None
    add_267: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_681, 1);  mul_681 = None
    mul_682: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_58, add_267);  sigmoid_58 = add_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_344: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_345: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_356: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_357: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
    unsqueeze_358: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_59: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_8)
    sub_221: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_24, sigmoid_59)
    mul_702: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_8, sub_221);  clone_8 = sub_221 = None
    add_268: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_702, 1);  mul_702 = None
    mul_703: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_59, add_268);  sigmoid_59 = add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_368: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_369: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
    unsqueeze_370: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_60: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_7)
    sub_226: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_24, sigmoid_60)
    mul_714: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_7, sub_226);  clone_7 = sub_226 = None
    add_269: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_714, 1);  mul_714 = None
    mul_715: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_60, add_269);  sigmoid_60 = add_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_380: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_381: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
    unsqueeze_382: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_392: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_393: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_61: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_6)
    sub_235: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_24, sigmoid_61)
    mul_735: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_6, sub_235);  clone_6 = sub_235 = None
    add_271: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_735, 1);  mul_735 = None
    mul_736: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_61, add_271);  sigmoid_61 = add_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_404: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_405: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_62: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_5)
    sub_240: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_24, sigmoid_62);  full_default_24 = None
    mul_747: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_5, sub_240);  clone_5 = sub_240 = None
    add_272: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_747, 1);  mul_747 = None
    mul_748: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_62, add_272);  sigmoid_62 = add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_416: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_417: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_428: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_429: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_63: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(clone_4)
    full_default_29: "f32[8, 128, 64, 64]" = torch.ops.aten.full.default([8, 128, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_249: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_29, sigmoid_63);  full_default_29 = None
    mul_768: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(clone_4, sub_249);  clone_4 = sub_249 = None
    add_274: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Scalar(mul_768, 1);  mul_768 = None
    mul_769: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_63, add_274);  sigmoid_63 = add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_440: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_441: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_64: "f32[8, 128, 128, 128]" = torch.ops.aten.sigmoid.default(clone_3)
    full_default_30: "f32[8, 128, 128, 128]" = torch.ops.aten.full.default([8, 128, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_254: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(full_default_30, sigmoid_64);  full_default_30 = None
    mul_780: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(clone_3, sub_254);  clone_3 = sub_254 = None
    add_275: "f32[8, 128, 128, 128]" = torch.ops.aten.add.Scalar(mul_780, 1);  mul_780 = None
    mul_781: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_64, add_275);  sigmoid_64 = add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_452: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_453: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_464: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_465: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_65: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(clone_2)
    full_default_31: "f32[8, 64, 128, 128]" = torch.ops.aten.full.default([8, 64, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_263: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(full_default_31, sigmoid_65)
    mul_801: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(clone_2, sub_263);  clone_2 = sub_263 = None
    add_276: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Scalar(mul_801, 1);  mul_801 = None
    mul_802: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_65, add_276);  sigmoid_65 = add_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_476: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_477: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_66: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(clone_1)
    sub_268: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(full_default_31, sigmoid_66);  full_default_31 = None
    mul_813: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(clone_1, sub_268);  clone_1 = sub_268 = None
    add_277: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Scalar(mul_813, 1);  mul_813 = None
    mul_814: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_66, add_277);  sigmoid_66 = add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_488: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_489: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_67: "f32[8, 16, 128, 128]" = torch.ops.aten.sigmoid.default(clone)
    full_default_33: "f32[8, 16, 128, 128]" = torch.ops.aten.full.default([8, 16, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_273: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(full_default_33, sigmoid_67);  full_default_33 = None
    mul_825: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(clone, sub_273);  clone = sub_273 = None
    add_278: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Scalar(mul_825, 1);  mul_825 = None
    mul_826: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_67, add_278);  sigmoid_67 = add_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_500: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_501: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_216, add);  primals_216 = add = None
    copy__1: "f32[16]" = torch.ops.aten.copy_.default(primals_217, add_2);  primals_217 = add_2 = None
    copy__2: "f32[16]" = torch.ops.aten.copy_.default(primals_218, add_3);  primals_218 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_219, add_5);  primals_219 = add_5 = None
    copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_220, add_7);  primals_220 = add_7 = None
    copy__5: "f32[64]" = torch.ops.aten.copy_.default(primals_221, add_8);  primals_221 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_222, add_10);  primals_222 = add_10 = None
    copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_223, add_12);  primals_223 = add_12 = None
    copy__8: "f32[64]" = torch.ops.aten.copy_.default(primals_224, add_13);  primals_224 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_225, add_15);  primals_225 = add_15 = None
    copy__10: "f32[32]" = torch.ops.aten.copy_.default(primals_226, add_17);  primals_226 = add_17 = None
    copy__11: "f32[32]" = torch.ops.aten.copy_.default(primals_227, add_18);  primals_227 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_228, add_20);  primals_228 = add_20 = None
    copy__13: "f32[128]" = torch.ops.aten.copy_.default(primals_229, add_22);  primals_229 = add_22 = None
    copy__14: "f32[128]" = torch.ops.aten.copy_.default(primals_230, add_23);  primals_230 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_25);  primals_231 = add_25 = None
    copy__16: "f32[128]" = torch.ops.aten.copy_.default(primals_232, add_27);  primals_232 = add_27 = None
    copy__17: "f32[128]" = torch.ops.aten.copy_.default(primals_233, add_28);  primals_233 = add_28 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_234, add_30);  primals_234 = add_30 = None
    copy__19: "f32[64]" = torch.ops.aten.copy_.default(primals_235, add_32);  primals_235 = add_32 = None
    copy__20: "f32[64]" = torch.ops.aten.copy_.default(primals_236, add_33);  primals_236 = add_33 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_35);  primals_237 = add_35 = None
    copy__22: "f32[256]" = torch.ops.aten.copy_.default(primals_238, add_37);  primals_238 = add_37 = None
    copy__23: "f32[256]" = torch.ops.aten.copy_.default(primals_239, add_38);  primals_239 = add_38 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_240, add_40);  primals_240 = add_40 = None
    copy__25: "f32[256]" = torch.ops.aten.copy_.default(primals_241, add_42);  primals_241 = add_42 = None
    copy__26: "f32[256]" = torch.ops.aten.copy_.default(primals_242, add_43);  primals_242 = add_43 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_45);  primals_243 = add_45 = None
    copy__28: "f32[64]" = torch.ops.aten.copy_.default(primals_244, add_47);  primals_244 = add_47 = None
    copy__29: "f32[64]" = torch.ops.aten.copy_.default(primals_245, add_48);  primals_245 = add_48 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_246, add_51);  primals_246 = add_51 = None
    copy__31: "f32[256]" = torch.ops.aten.copy_.default(primals_247, add_53);  primals_247 = add_53 = None
    copy__32: "f32[256]" = torch.ops.aten.copy_.default(primals_248, add_54);  primals_248 = add_54 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_56);  primals_249 = add_56 = None
    copy__34: "f32[256]" = torch.ops.aten.copy_.default(primals_250, add_58);  primals_250 = add_58 = None
    copy__35: "f32[256]" = torch.ops.aten.copy_.default(primals_251, add_59);  primals_251 = add_59 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_252, add_61);  primals_252 = add_61 = None
    copy__37: "f32[64]" = torch.ops.aten.copy_.default(primals_253, add_63);  primals_253 = add_63 = None
    copy__38: "f32[64]" = torch.ops.aten.copy_.default(primals_254, add_64);  primals_254 = add_64 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_255, add_67);  primals_255 = add_67 = None
    copy__40: "f32[256]" = torch.ops.aten.copy_.default(primals_256, add_69);  primals_256 = add_69 = None
    copy__41: "f32[256]" = torch.ops.aten.copy_.default(primals_257, add_70);  primals_257 = add_70 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_258, add_72);  primals_258 = add_72 = None
    copy__43: "f32[256]" = torch.ops.aten.copy_.default(primals_259, add_74);  primals_259 = add_74 = None
    copy__44: "f32[256]" = torch.ops.aten.copy_.default(primals_260, add_75);  primals_260 = add_75 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_261, add_77);  primals_261 = add_77 = None
    copy__46: "f32[96]" = torch.ops.aten.copy_.default(primals_262, add_79);  primals_262 = add_79 = None
    copy__47: "f32[96]" = torch.ops.aten.copy_.default(primals_263, add_80);  primals_263 = add_80 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_264, add_82);  primals_264 = add_82 = None
    copy__49: "f32[96]" = torch.ops.aten.copy_.default(primals_265, add_84);  primals_265 = add_84 = None
    copy__50: "f32[96]" = torch.ops.aten.copy_.default(primals_266, add_85);  primals_266 = add_85 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_267, add_101);  primals_267 = add_101 = None
    copy__52: "f32[96]" = torch.ops.aten.copy_.default(primals_268, add_103);  primals_268 = add_103 = None
    copy__53: "f32[96]" = torch.ops.aten.copy_.default(primals_269, add_104);  primals_269 = add_104 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_270, add_106);  primals_270 = add_106 = None
    copy__55: "f32[96]" = torch.ops.aten.copy_.default(primals_271, add_108);  primals_271 = add_108 = None
    copy__56: "f32[96]" = torch.ops.aten.copy_.default(primals_272, add_109);  primals_272 = add_109 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_111);  primals_273 = add_111 = None
    copy__58: "f32[384]" = torch.ops.aten.copy_.default(primals_274, add_113);  primals_274 = add_113 = None
    copy__59: "f32[384]" = torch.ops.aten.copy_.default(primals_275, add_114);  primals_275 = add_114 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_276, add_116);  primals_276 = add_116 = None
    copy__61: "f32[384]" = torch.ops.aten.copy_.default(primals_277, add_118);  primals_277 = add_118 = None
    copy__62: "f32[384]" = torch.ops.aten.copy_.default(primals_278, add_119);  primals_278 = add_119 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_279, add_121);  primals_279 = add_121 = None
    copy__64: "f32[128]" = torch.ops.aten.copy_.default(primals_280, add_123);  primals_280 = add_123 = None
    copy__65: "f32[128]" = torch.ops.aten.copy_.default(primals_281, add_124);  primals_281 = add_124 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_282, add_126);  primals_282 = add_126 = None
    copy__67: "f32[128]" = torch.ops.aten.copy_.default(primals_283, add_128);  primals_283 = add_128 = None
    copy__68: "f32[128]" = torch.ops.aten.copy_.default(primals_284, add_129);  primals_284 = add_129 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_285, add_157);  primals_285 = add_157 = None
    copy__70: "f32[128]" = torch.ops.aten.copy_.default(primals_286, add_159);  primals_286 = add_159 = None
    copy__71: "f32[128]" = torch.ops.aten.copy_.default(primals_287, add_160);  primals_287 = add_160 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_288, add_162);  primals_288 = add_162 = None
    copy__73: "f32[128]" = torch.ops.aten.copy_.default(primals_289, add_164);  primals_289 = add_164 = None
    copy__74: "f32[128]" = torch.ops.aten.copy_.default(primals_290, add_165);  primals_290 = add_165 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_167);  primals_291 = add_167 = None
    copy__76: "f32[512]" = torch.ops.aten.copy_.default(primals_292, add_169);  primals_292 = add_169 = None
    copy__77: "f32[512]" = torch.ops.aten.copy_.default(primals_293, add_170);  primals_293 = add_170 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_172);  primals_294 = add_172 = None
    copy__79: "f32[512]" = torch.ops.aten.copy_.default(primals_295, add_174);  primals_295 = add_174 = None
    copy__80: "f32[512]" = torch.ops.aten.copy_.default(primals_296, add_175);  primals_296 = add_175 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_177);  primals_297 = add_177 = None
    copy__82: "f32[160]" = torch.ops.aten.copy_.default(primals_298, add_179);  primals_298 = add_179 = None
    copy__83: "f32[160]" = torch.ops.aten.copy_.default(primals_299, add_180);  primals_299 = add_180 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_182);  primals_300 = add_182 = None
    copy__85: "f32[160]" = torch.ops.aten.copy_.default(primals_301, add_184);  primals_301 = add_184 = None
    copy__86: "f32[160]" = torch.ops.aten.copy_.default(primals_302, add_185);  primals_302 = add_185 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_207);  primals_303 = add_207 = None
    copy__88: "f32[160]" = torch.ops.aten.copy_.default(primals_304, add_209);  primals_304 = add_209 = None
    copy__89: "f32[160]" = torch.ops.aten.copy_.default(primals_305, add_210);  primals_305 = add_210 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_306, add_212);  primals_306 = add_212 = None
    copy__91: "f32[160]" = torch.ops.aten.copy_.default(primals_307, add_214);  primals_307 = add_214 = None
    copy__92: "f32[160]" = torch.ops.aten.copy_.default(primals_308, add_215);  primals_308 = add_215 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_217);  primals_309 = add_217 = None
    copy__94: "f32[640]" = torch.ops.aten.copy_.default(primals_310, add_219);  primals_310 = add_219 = None
    copy__95: "f32[640]" = torch.ops.aten.copy_.default(primals_311, add_220);  primals_311 = add_220 = None
    return [addmm_36, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_89, primals_95, primals_101, primals_107, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, primals_152, primals_158, primals_164, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_211, primals_212, primals_213, primals_312, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, convolution_3, squeeze_10, add_19, convolution_4, squeeze_13, mul_38, convolution_5, squeeze_16, mul_46, convolution_6, squeeze_19, add_34, convolution_7, squeeze_22, mul_61, convolution_8, squeeze_25, mul_69, convolution_9, squeeze_28, add_50, convolution_10, squeeze_31, mul_84, convolution_11, squeeze_34, mul_92, convolution_12, squeeze_37, add_66, convolution_13, squeeze_40, mul_107, convolution_14, squeeze_43, mul_115, convolution_15, squeeze_46, add_81, convolution_16, squeeze_49, mul_130, mul_131, view_3, getitem_36, getitem_37, getitem_38, getitem_40, getitem_41, getitem_42, view_7, mul_133, view_9, addmm_2, view_11, mul_136, view_13, getitem_47, getitem_48, getitem_49, getitem_51, getitem_52, getitem_53, view_17, mul_138, view_19, addmm_6, view_21, mul_141, view_25, convolution_18, squeeze_52, cat, convolution_19, squeeze_55, mul_158, convolution_20, squeeze_58, mul_166, convolution_21, squeeze_61, mul_174, convolution_22, squeeze_64, add_125, convolution_23, squeeze_67, mul_189, mul_190, view_29, getitem_72, getitem_73, getitem_74, getitem_76, getitem_77, getitem_78, view_33, mul_192, view_35, addmm_10, view_37, mul_195, view_39, getitem_83, getitem_84, getitem_85, getitem_87, getitem_88, getitem_89, view_43, mul_197, view_45, addmm_14, view_47, mul_200, view_49, getitem_94, getitem_95, getitem_96, getitem_98, getitem_99, getitem_100, view_53, mul_202, view_55, addmm_18, view_57, mul_205, view_59, getitem_105, getitem_106, getitem_107, getitem_109, getitem_110, getitem_111, view_63, mul_207, view_65, addmm_22, view_67, mul_210, view_71, convolution_25, squeeze_70, cat_1, convolution_26, squeeze_73, mul_227, convolution_27, squeeze_76, mul_235, convolution_28, squeeze_79, mul_243, convolution_29, squeeze_82, add_181, convolution_30, squeeze_85, mul_258, mul_259, view_75, getitem_130, getitem_131, getitem_132, getitem_134, getitem_135, getitem_136, view_79, mul_261, view_81, addmm_26, view_83, mul_264, view_85, getitem_141, getitem_142, getitem_143, getitem_145, getitem_146, getitem_147, view_89, mul_266, view_91, addmm_30, view_93, mul_269, view_95, getitem_152, getitem_153, getitem_154, getitem_156, getitem_157, getitem_158, view_99, mul_271, view_101, addmm_34, view_103, mul_274, view_107, convolution_32, squeeze_88, cat_2, convolution_33, squeeze_91, mul_291, convolution_34, squeeze_94, clone_64, permute_67, mul_301, unsqueeze_130, mul_313, unsqueeze_142, mul_325, unsqueeze_154, div_1, permute_76, permute_81, div_2, permute_85, alias_9, permute_91, div_3, permute_95, permute_100, div_4, permute_104, alias_10, permute_110, div_5, permute_114, permute_119, div_6, permute_123, alias_11, permute_129, div_7, mul_395, unsqueeze_166, unsqueeze_178, mul_416, unsqueeze_190, mul_428, unsqueeze_202, mul_440, unsqueeze_214, mul_452, unsqueeze_226, div_8, permute_142, permute_147, div_9, permute_151, alias_12, permute_157, div_10, permute_161, permute_166, div_11, permute_170, alias_13, permute_176, div_12, permute_180, permute_185, div_13, permute_189, alias_14, permute_195, div_14, permute_199, permute_204, div_15, permute_208, alias_15, permute_214, div_16, mul_539, unsqueeze_238, unsqueeze_250, mul_560, unsqueeze_262, mul_572, unsqueeze_274, mul_584, unsqueeze_286, mul_596, unsqueeze_298, div_17, permute_227, permute_232, div_18, permute_236, alias_16, permute_242, div_19, permute_246, permute_251, div_20, permute_255, alias_17, permute_261, div_21, mul_649, unsqueeze_310, unsqueeze_322, mul_670, unsqueeze_334, mul_682, unsqueeze_346, unsqueeze_358, mul_703, unsqueeze_370, mul_715, unsqueeze_382, unsqueeze_394, mul_736, unsqueeze_406, mul_748, unsqueeze_418, unsqueeze_430, mul_769, unsqueeze_442, mul_781, unsqueeze_454, unsqueeze_466, mul_802, unsqueeze_478, mul_814, unsqueeze_490, mul_826, unsqueeze_502]
    