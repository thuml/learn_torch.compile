from __future__ import annotations



def forward(self, primals_1: "f32[384, 3, 16, 16]", primals_2: "f32[384]", primals_3: "f32[384]", primals_4: "f32[384]", primals_5: "f32[384, 196]", primals_6: "f32[384]", primals_7: "f32[196, 192]", primals_8: "f32[196]", primals_9: "f32[384]", primals_10: "f32[384]", primals_11: "f32[1536, 384]", primals_12: "f32[1536]", primals_13: "f32[384, 768]", primals_14: "f32[384]", primals_15: "f32[384]", primals_16: "f32[384]", primals_17: "f32[384, 196]", primals_18: "f32[384]", primals_19: "f32[196, 192]", primals_20: "f32[196]", primals_21: "f32[384]", primals_22: "f32[384]", primals_23: "f32[1536, 384]", primals_24: "f32[1536]", primals_25: "f32[384, 768]", primals_26: "f32[384]", primals_27: "f32[384]", primals_28: "f32[384]", primals_29: "f32[384, 196]", primals_30: "f32[384]", primals_31: "f32[196, 192]", primals_32: "f32[196]", primals_33: "f32[384]", primals_34: "f32[384]", primals_35: "f32[1536, 384]", primals_36: "f32[1536]", primals_37: "f32[384, 768]", primals_38: "f32[384]", primals_39: "f32[384]", primals_40: "f32[384]", primals_41: "f32[384, 196]", primals_42: "f32[384]", primals_43: "f32[196, 192]", primals_44: "f32[196]", primals_45: "f32[384]", primals_46: "f32[384]", primals_47: "f32[1536, 384]", primals_48: "f32[1536]", primals_49: "f32[384, 768]", primals_50: "f32[384]", primals_51: "f32[384]", primals_52: "f32[384]", primals_53: "f32[384, 196]", primals_54: "f32[384]", primals_55: "f32[196, 192]", primals_56: "f32[196]", primals_57: "f32[384]", primals_58: "f32[384]", primals_59: "f32[1536, 384]", primals_60: "f32[1536]", primals_61: "f32[384, 768]", primals_62: "f32[384]", primals_63: "f32[384]", primals_64: "f32[384]", primals_65: "f32[384, 196]", primals_66: "f32[384]", primals_67: "f32[196, 192]", primals_68: "f32[196]", primals_69: "f32[384]", primals_70: "f32[384]", primals_71: "f32[1536, 384]", primals_72: "f32[1536]", primals_73: "f32[384, 768]", primals_74: "f32[384]", primals_75: "f32[384]", primals_76: "f32[384]", primals_77: "f32[384, 196]", primals_78: "f32[384]", primals_79: "f32[196, 192]", primals_80: "f32[196]", primals_81: "f32[384]", primals_82: "f32[384]", primals_83: "f32[1536, 384]", primals_84: "f32[1536]", primals_85: "f32[384, 768]", primals_86: "f32[384]", primals_87: "f32[384]", primals_88: "f32[384]", primals_89: "f32[384, 196]", primals_90: "f32[384]", primals_91: "f32[196, 192]", primals_92: "f32[196]", primals_93: "f32[384]", primals_94: "f32[384]", primals_95: "f32[1536, 384]", primals_96: "f32[1536]", primals_97: "f32[384, 768]", primals_98: "f32[384]", primals_99: "f32[384]", primals_100: "f32[384]", primals_101: "f32[384, 196]", primals_102: "f32[384]", primals_103: "f32[196, 192]", primals_104: "f32[196]", primals_105: "f32[384]", primals_106: "f32[384]", primals_107: "f32[1536, 384]", primals_108: "f32[1536]", primals_109: "f32[384, 768]", primals_110: "f32[384]", primals_111: "f32[384]", primals_112: "f32[384]", primals_113: "f32[384, 196]", primals_114: "f32[384]", primals_115: "f32[196, 192]", primals_116: "f32[196]", primals_117: "f32[384]", primals_118: "f32[384]", primals_119: "f32[1536, 384]", primals_120: "f32[1536]", primals_121: "f32[384, 768]", primals_122: "f32[384]", primals_123: "f32[384]", primals_124: "f32[384]", primals_125: "f32[384, 196]", primals_126: "f32[384]", primals_127: "f32[196, 192]", primals_128: "f32[196]", primals_129: "f32[384]", primals_130: "f32[384]", primals_131: "f32[1536, 384]", primals_132: "f32[1536]", primals_133: "f32[384, 768]", primals_134: "f32[384]", primals_135: "f32[384]", primals_136: "f32[384]", primals_137: "f32[384, 196]", primals_138: "f32[384]", primals_139: "f32[196, 192]", primals_140: "f32[196]", primals_141: "f32[384]", primals_142: "f32[384]", primals_143: "f32[1536, 384]", primals_144: "f32[1536]", primals_145: "f32[384, 768]", primals_146: "f32[384]", primals_147: "f32[384]", primals_148: "f32[384]", primals_149: "f32[384, 196]", primals_150: "f32[384]", primals_151: "f32[196, 192]", primals_152: "f32[196]", primals_153: "f32[384]", primals_154: "f32[384]", primals_155: "f32[1536, 384]", primals_156: "f32[1536]", primals_157: "f32[384, 768]", primals_158: "f32[384]", primals_159: "f32[384]", primals_160: "f32[384]", primals_161: "f32[384, 196]", primals_162: "f32[384]", primals_163: "f32[196, 192]", primals_164: "f32[196]", primals_165: "f32[384]", primals_166: "f32[384]", primals_167: "f32[1536, 384]", primals_168: "f32[1536]", primals_169: "f32[384, 768]", primals_170: "f32[384]", primals_171: "f32[384]", primals_172: "f32[384]", primals_173: "f32[384, 196]", primals_174: "f32[384]", primals_175: "f32[196, 192]", primals_176: "f32[196]", primals_177: "f32[384]", primals_178: "f32[384]", primals_179: "f32[1536, 384]", primals_180: "f32[1536]", primals_181: "f32[384, 768]", primals_182: "f32[384]", primals_183: "f32[384]", primals_184: "f32[384]", primals_185: "f32[384, 196]", primals_186: "f32[384]", primals_187: "f32[196, 192]", primals_188: "f32[196]", primals_189: "f32[384]", primals_190: "f32[384]", primals_191: "f32[1536, 384]", primals_192: "f32[1536]", primals_193: "f32[384, 768]", primals_194: "f32[384]", primals_195: "f32[384]", primals_196: "f32[384]", primals_197: "f32[384, 196]", primals_198: "f32[384]", primals_199: "f32[196, 192]", primals_200: "f32[196]", primals_201: "f32[384]", primals_202: "f32[384]", primals_203: "f32[1536, 384]", primals_204: "f32[1536]", primals_205: "f32[384, 768]", primals_206: "f32[384]", primals_207: "f32[384]", primals_208: "f32[384]", primals_209: "f32[384, 196]", primals_210: "f32[384]", primals_211: "f32[196, 192]", primals_212: "f32[196]", primals_213: "f32[384]", primals_214: "f32[384]", primals_215: "f32[1536, 384]", primals_216: "f32[1536]", primals_217: "f32[384, 768]", primals_218: "f32[384]", primals_219: "f32[384]", primals_220: "f32[384]", primals_221: "f32[384, 196]", primals_222: "f32[384]", primals_223: "f32[196, 192]", primals_224: "f32[196]", primals_225: "f32[384]", primals_226: "f32[384]", primals_227: "f32[1536, 384]", primals_228: "f32[1536]", primals_229: "f32[384, 768]", primals_230: "f32[384]", primals_231: "f32[384]", primals_232: "f32[384]", primals_233: "f32[384, 196]", primals_234: "f32[384]", primals_235: "f32[196, 192]", primals_236: "f32[196]", primals_237: "f32[384]", primals_238: "f32[384]", primals_239: "f32[1536, 384]", primals_240: "f32[1536]", primals_241: "f32[384, 768]", primals_242: "f32[384]", primals_243: "f32[384]", primals_244: "f32[384]", primals_245: "f32[384, 196]", primals_246: "f32[384]", primals_247: "f32[196, 192]", primals_248: "f32[196]", primals_249: "f32[384]", primals_250: "f32[384]", primals_251: "f32[1536, 384]", primals_252: "f32[1536]", primals_253: "f32[384, 768]", primals_254: "f32[384]", primals_255: "f32[384]", primals_256: "f32[384]", primals_257: "f32[384, 196]", primals_258: "f32[384]", primals_259: "f32[196, 192]", primals_260: "f32[196]", primals_261: "f32[384]", primals_262: "f32[384]", primals_263: "f32[1536, 384]", primals_264: "f32[1536]", primals_265: "f32[384, 768]", primals_266: "f32[384]", primals_267: "f32[384]", primals_268: "f32[384]", primals_269: "f32[384, 196]", primals_270: "f32[384]", primals_271: "f32[196, 192]", primals_272: "f32[196]", primals_273: "f32[384]", primals_274: "f32[384]", primals_275: "f32[1536, 384]", primals_276: "f32[1536]", primals_277: "f32[384, 768]", primals_278: "f32[384]", primals_279: "f32[384]", primals_280: "f32[384]", primals_281: "f32[384, 196]", primals_282: "f32[384]", primals_283: "f32[196, 192]", primals_284: "f32[196]", primals_285: "f32[384]", primals_286: "f32[384]", primals_287: "f32[1536, 384]", primals_288: "f32[1536]", primals_289: "f32[384, 768]", primals_290: "f32[384]", primals_291: "f32[384]", primals_292: "f32[384]", primals_293: "f32[1000, 384]", primals_294: "f32[1000]", primals_295: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(primals_295, primals_1, primals_2, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 384, 196]" = torch.ops.aten.view.default(convolution, [8, 384, 196]);  convolution = None
    permute: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 196, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul, primals_3)
    add_1: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    permute_1: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_1, [0, 2, 1]);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_2: "f32[196, 384]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    clone_1: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_1: "f32[3072, 196]" = torch.ops.aten.view.default(clone_1, [3072, 196]);  clone_1 = None
    mm: "f32[3072, 384]" = torch.ops.aten.mm.default(view_1, permute_2)
    view_2: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm, [8, 384, 384]);  mm = None
    add_2: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_2, primals_6);  view_2 = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split = torch.ops.aten.split.Tensor(add_2, 192, -1);  add_2 = None
    getitem_2: "f32[8, 384, 192]" = split[0]
    getitem_3: "f32[8, 384, 192]" = split[1];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_3)
    mul_2: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_3, sigmoid);  sigmoid = None
    mul_3: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_2, mul_2);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_2: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_3);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_3: "f32[3072, 192]" = torch.ops.aten.view.default(clone_2, [3072, 192]);  clone_2 = None
    permute_3: "f32[192, 196]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_8, view_3, permute_3);  primals_8 = None
    view_4: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm, [8, 384, 196]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_3: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_4);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_4: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_3, [0, 2, 1]);  clone_3 = None
    add_3: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(permute, permute_4);  permute = permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_4: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_3, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_4, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_5: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_1: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_4, getitem_5);  clone_4 = getitem_5 = None
    mul_4: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_5: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_4, primals_9)
    add_5: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_5, primals_10);  mul_5 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_5: "f32[1568, 384]" = torch.ops.aten.view.default(add_5, [1568, 384]);  add_5 = None
    permute_5: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm_1: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_12, view_5, permute_5);  primals_12 = None
    view_6: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_1, [8, 196, 1536]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_1 = torch.ops.aten.split.Tensor(view_6, 768, -1);  view_6 = None
    getitem_6: "f32[8, 196, 768]" = split_1[0]
    getitem_7: "f32[8, 196, 768]" = split_1[1];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_1: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_7)
    mul_6: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_7, sigmoid_1);  sigmoid_1 = None
    mul_7: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_6, mul_6);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_5: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_7);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_7: "f32[1568, 768]" = torch.ops.aten.view.default(clone_5, [1568, 768]);  clone_5 = None
    permute_6: "f32[768, 384]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_2: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_14, view_7, permute_6);  primals_14 = None
    view_8: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_2, [8, 196, 384]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_6: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_8);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_6: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_3, clone_6);  add_3 = clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_7: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_6, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_7, [2], correction = 0, keepdim = True)
    getitem_8: "f32[8, 196, 1]" = var_mean_2[0]
    getitem_9: "f32[8, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    add_7: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_2: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_2: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_7, getitem_9);  clone_7 = getitem_9 = None
    mul_8: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_9: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_8, primals_15)
    add_8: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_9, primals_16);  mul_9 = primals_16 = None
    permute_7: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_8, [0, 2, 1]);  add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_8: "f32[196, 384]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    clone_8: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_9: "f32[3072, 196]" = torch.ops.aten.view.default(clone_8, [3072, 196]);  clone_8 = None
    mm_1: "f32[3072, 384]" = torch.ops.aten.mm.default(view_9, permute_8)
    view_10: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_1, [8, 384, 384]);  mm_1 = None
    add_9: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_10, primals_18);  view_10 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_2 = torch.ops.aten.split.Tensor(add_9, 192, -1);  add_9 = None
    getitem_10: "f32[8, 384, 192]" = split_2[0]
    getitem_11: "f32[8, 384, 192]" = split_2[1];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_2: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_11)
    mul_10: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_11, sigmoid_2);  sigmoid_2 = None
    mul_11: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_10, mul_10);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_9: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_11);  mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_11: "f32[3072, 192]" = torch.ops.aten.view.default(clone_9, [3072, 192]);  clone_9 = None
    permute_9: "f32[192, 196]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    addmm_3: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_20, view_11, permute_9);  primals_20 = None
    view_12: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_3, [8, 384, 196]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_10: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_10: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_10, [0, 2, 1]);  clone_10 = None
    add_10: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_6, permute_10);  add_6 = permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_11: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_10, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_11, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 196, 1]" = var_mean_3[0]
    getitem_13: "f32[8, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_3: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_3: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_11, getitem_13);  clone_11 = getitem_13 = None
    mul_12: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_13: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_12, primals_21)
    add_12: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_13, primals_22);  mul_13 = primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_13: "f32[1568, 384]" = torch.ops.aten.view.default(add_12, [1568, 384]);  add_12 = None
    permute_11: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_4: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_24, view_13, permute_11);  primals_24 = None
    view_14: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_4, [8, 196, 1536]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_3 = torch.ops.aten.split.Tensor(view_14, 768, -1);  view_14 = None
    getitem_14: "f32[8, 196, 768]" = split_3[0]
    getitem_15: "f32[8, 196, 768]" = split_3[1];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_3: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_15)
    mul_14: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_15, sigmoid_3);  sigmoid_3 = None
    mul_15: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_14, mul_14);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_12: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_15);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_15: "f32[1568, 768]" = torch.ops.aten.view.default(clone_12, [1568, 768]);  clone_12 = None
    permute_12: "f32[768, 384]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_5: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_26, view_15, permute_12);  primals_26 = None
    view_16: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_5, [8, 196, 384]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_13: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_16);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_13: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_10, clone_13);  add_10 = clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_14: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_13, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_14, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 196, 1]" = var_mean_4[0]
    getitem_17: "f32[8, 196, 1]" = var_mean_4[1];  var_mean_4 = None
    add_14: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_4: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_4: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_14, getitem_17);  clone_14 = getitem_17 = None
    mul_16: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_17: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_16, primals_27)
    add_15: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_17, primals_28);  mul_17 = primals_28 = None
    permute_13: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_15, [0, 2, 1]);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_14: "f32[196, 384]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
    clone_15: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    view_17: "f32[3072, 196]" = torch.ops.aten.view.default(clone_15, [3072, 196]);  clone_15 = None
    mm_2: "f32[3072, 384]" = torch.ops.aten.mm.default(view_17, permute_14)
    view_18: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_2, [8, 384, 384]);  mm_2 = None
    add_16: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_18, primals_30);  view_18 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_4 = torch.ops.aten.split.Tensor(add_16, 192, -1);  add_16 = None
    getitem_18: "f32[8, 384, 192]" = split_4[0]
    getitem_19: "f32[8, 384, 192]" = split_4[1];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_4: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_19)
    mul_18: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_19, sigmoid_4);  sigmoid_4 = None
    mul_19: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_18, mul_18);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_16: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_19);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_19: "f32[3072, 192]" = torch.ops.aten.view.default(clone_16, [3072, 192]);  clone_16 = None
    permute_15: "f32[192, 196]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    addmm_6: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_32, view_19, permute_15);  primals_32 = None
    view_20: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_6, [8, 384, 196]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_17: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_16: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_17, [0, 2, 1]);  clone_17 = None
    add_17: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_13, permute_16);  add_13 = permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_18: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_17, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_18, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 196, 1]" = var_mean_5[0]
    getitem_21: "f32[8, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_5: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_5: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_18, getitem_21);  clone_18 = getitem_21 = None
    mul_20: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_21: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_20, primals_33)
    add_19: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_21, primals_34);  mul_21 = primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_21: "f32[1568, 384]" = torch.ops.aten.view.default(add_19, [1568, 384]);  add_19 = None
    permute_17: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
    addmm_7: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_36, view_21, permute_17);  primals_36 = None
    view_22: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_7, [8, 196, 1536]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_5 = torch.ops.aten.split.Tensor(view_22, 768, -1);  view_22 = None
    getitem_22: "f32[8, 196, 768]" = split_5[0]
    getitem_23: "f32[8, 196, 768]" = split_5[1];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_5: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_23)
    mul_22: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_23, sigmoid_5);  sigmoid_5 = None
    mul_23: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_22, mul_22);  mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_19: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_23);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_23: "f32[1568, 768]" = torch.ops.aten.view.default(clone_19, [1568, 768]);  clone_19 = None
    permute_18: "f32[768, 384]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    addmm_8: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_38, view_23, permute_18);  primals_38 = None
    view_24: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_8, [8, 196, 384]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_20: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_24);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_20: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_17, clone_20);  add_17 = clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_21: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_21, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 196, 1]" = var_mean_6[0]
    getitem_25: "f32[8, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_6: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_6: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_21, getitem_25);  clone_21 = getitem_25 = None
    mul_24: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_25: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_24, primals_39)
    add_22: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_25, primals_40);  mul_25 = primals_40 = None
    permute_19: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_22, [0, 2, 1]);  add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_20: "f32[196, 384]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
    clone_22: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_25: "f32[3072, 196]" = torch.ops.aten.view.default(clone_22, [3072, 196]);  clone_22 = None
    mm_3: "f32[3072, 384]" = torch.ops.aten.mm.default(view_25, permute_20)
    view_26: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_3, [8, 384, 384]);  mm_3 = None
    add_23: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_26, primals_42);  view_26 = primals_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_6 = torch.ops.aten.split.Tensor(add_23, 192, -1);  add_23 = None
    getitem_26: "f32[8, 384, 192]" = split_6[0]
    getitem_27: "f32[8, 384, 192]" = split_6[1];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_6: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_27)
    mul_26: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_27, sigmoid_6);  sigmoid_6 = None
    mul_27: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_26, mul_26);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_23: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_27);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_27: "f32[3072, 192]" = torch.ops.aten.view.default(clone_23, [3072, 192]);  clone_23 = None
    permute_21: "f32[192, 196]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    addmm_9: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_44, view_27, permute_21);  primals_44 = None
    view_28: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_9, [8, 384, 196]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_24: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_28);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_22: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_24, [0, 2, 1]);  clone_24 = None
    add_24: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_20, permute_22);  add_20 = permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_25: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_24, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_25, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 196, 1]" = var_mean_7[0]
    getitem_29: "f32[8, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_7: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_7: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_25, getitem_29);  clone_25 = getitem_29 = None
    mul_28: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_29: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_28, primals_45)
    add_26: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_29, primals_46);  mul_29 = primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_29: "f32[1568, 384]" = torch.ops.aten.view.default(add_26, [1568, 384]);  add_26 = None
    permute_23: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    addmm_10: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_48, view_29, permute_23);  primals_48 = None
    view_30: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_10, [8, 196, 1536]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_7 = torch.ops.aten.split.Tensor(view_30, 768, -1);  view_30 = None
    getitem_30: "f32[8, 196, 768]" = split_7[0]
    getitem_31: "f32[8, 196, 768]" = split_7[1];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_7: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_31)
    mul_30: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_31, sigmoid_7);  sigmoid_7 = None
    mul_31: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_30, mul_30);  mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_26: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_31);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_31: "f32[1568, 768]" = torch.ops.aten.view.default(clone_26, [1568, 768]);  clone_26 = None
    permute_24: "f32[768, 384]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    addmm_11: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_50, view_31, permute_24);  primals_50 = None
    view_32: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_11, [8, 196, 384]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_27: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_32);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_27: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_24, clone_27);  add_24 = clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_28: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_27, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_28, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 196, 1]" = var_mean_8[0]
    getitem_33: "f32[8, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_8: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_8: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_28, getitem_33);  clone_28 = getitem_33 = None
    mul_32: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_33: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_32, primals_51)
    add_29: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_33, primals_52);  mul_33 = primals_52 = None
    permute_25: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_29, [0, 2, 1]);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_26: "f32[196, 384]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    clone_29: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    view_33: "f32[3072, 196]" = torch.ops.aten.view.default(clone_29, [3072, 196]);  clone_29 = None
    mm_4: "f32[3072, 384]" = torch.ops.aten.mm.default(view_33, permute_26)
    view_34: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_4, [8, 384, 384]);  mm_4 = None
    add_30: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_34, primals_54);  view_34 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_8 = torch.ops.aten.split.Tensor(add_30, 192, -1);  add_30 = None
    getitem_34: "f32[8, 384, 192]" = split_8[0]
    getitem_35: "f32[8, 384, 192]" = split_8[1];  split_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_8: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_35)
    mul_34: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_35, sigmoid_8);  sigmoid_8 = None
    mul_35: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_34, mul_34);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_30: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_35);  mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_35: "f32[3072, 192]" = torch.ops.aten.view.default(clone_30, [3072, 192]);  clone_30 = None
    permute_27: "f32[192, 196]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_12: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_56, view_35, permute_27);  primals_56 = None
    view_36: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_12, [8, 384, 196]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_31: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_28: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_31, [0, 2, 1]);  clone_31 = None
    add_31: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_27, permute_28);  add_27 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_32: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_31, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_32, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 196, 1]" = var_mean_9[0]
    getitem_37: "f32[8, 196, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_9: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_9: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_32, getitem_37);  clone_32 = getitem_37 = None
    mul_36: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_37: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_36, primals_57)
    add_33: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_37, primals_58);  mul_37 = primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_37: "f32[1568, 384]" = torch.ops.aten.view.default(add_33, [1568, 384]);  add_33 = None
    permute_29: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    addmm_13: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_60, view_37, permute_29);  primals_60 = None
    view_38: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_13, [8, 196, 1536]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_9 = torch.ops.aten.split.Tensor(view_38, 768, -1);  view_38 = None
    getitem_38: "f32[8, 196, 768]" = split_9[0]
    getitem_39: "f32[8, 196, 768]" = split_9[1];  split_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_9: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_39)
    mul_38: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_39, sigmoid_9);  sigmoid_9 = None
    mul_39: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_38, mul_38);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_33: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_39);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_39: "f32[1568, 768]" = torch.ops.aten.view.default(clone_33, [1568, 768]);  clone_33 = None
    permute_30: "f32[768, 384]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    addmm_14: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_62, view_39, permute_30);  primals_62 = None
    view_40: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_14, [8, 196, 384]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_34: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_34: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_31, clone_34);  add_31 = clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_35: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_34, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_35, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_41: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_10: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_35, getitem_41);  clone_35 = getitem_41 = None
    mul_40: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_41: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_40, primals_63)
    add_36: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_41, primals_64);  mul_41 = primals_64 = None
    permute_31: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_36, [0, 2, 1]);  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_32: "f32[196, 384]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    clone_36: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    view_41: "f32[3072, 196]" = torch.ops.aten.view.default(clone_36, [3072, 196]);  clone_36 = None
    mm_5: "f32[3072, 384]" = torch.ops.aten.mm.default(view_41, permute_32)
    view_42: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_5, [8, 384, 384]);  mm_5 = None
    add_37: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_42, primals_66);  view_42 = primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_10 = torch.ops.aten.split.Tensor(add_37, 192, -1);  add_37 = None
    getitem_42: "f32[8, 384, 192]" = split_10[0]
    getitem_43: "f32[8, 384, 192]" = split_10[1];  split_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_10: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_43)
    mul_42: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_43, sigmoid_10);  sigmoid_10 = None
    mul_43: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_42, mul_42);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_37: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_43);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_43: "f32[3072, 192]" = torch.ops.aten.view.default(clone_37, [3072, 192]);  clone_37 = None
    permute_33: "f32[192, 196]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    addmm_15: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_68, view_43, permute_33);  primals_68 = None
    view_44: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_15, [8, 384, 196]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_38: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_34: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_38, [0, 2, 1]);  clone_38 = None
    add_38: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_34, permute_34);  add_34 = permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_39: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_38, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_39, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 196, 1]" = var_mean_11[0]
    getitem_45: "f32[8, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    add_39: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_11: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_11: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_39, getitem_45);  clone_39 = getitem_45 = None
    mul_44: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_45: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_44, primals_69)
    add_40: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_45, primals_70);  mul_45 = primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_45: "f32[1568, 384]" = torch.ops.aten.view.default(add_40, [1568, 384]);  add_40 = None
    permute_35: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_16: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_72, view_45, permute_35);  primals_72 = None
    view_46: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_16, [8, 196, 1536]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_11 = torch.ops.aten.split.Tensor(view_46, 768, -1);  view_46 = None
    getitem_46: "f32[8, 196, 768]" = split_11[0]
    getitem_47: "f32[8, 196, 768]" = split_11[1];  split_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_11: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_47)
    mul_46: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_47, sigmoid_11);  sigmoid_11 = None
    mul_47: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_46, mul_46);  mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_40: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_47);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_47: "f32[1568, 768]" = torch.ops.aten.view.default(clone_40, [1568, 768]);  clone_40 = None
    permute_36: "f32[768, 384]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_17: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_74, view_47, permute_36);  primals_74 = None
    view_48: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_17, [8, 196, 384]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_41: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_41: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_38, clone_41);  add_38 = clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_42: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 196, 1]" = var_mean_12[0]
    getitem_49: "f32[8, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_12: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_12: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_42, getitem_49);  clone_42 = getitem_49 = None
    mul_48: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_49: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_48, primals_75)
    add_43: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_49, primals_76);  mul_49 = primals_76 = None
    permute_37: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_43, [0, 2, 1]);  add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_38: "f32[196, 384]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    clone_43: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    view_49: "f32[3072, 196]" = torch.ops.aten.view.default(clone_43, [3072, 196]);  clone_43 = None
    mm_6: "f32[3072, 384]" = torch.ops.aten.mm.default(view_49, permute_38)
    view_50: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_6, [8, 384, 384]);  mm_6 = None
    add_44: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_50, primals_78);  view_50 = primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_12 = torch.ops.aten.split.Tensor(add_44, 192, -1);  add_44 = None
    getitem_50: "f32[8, 384, 192]" = split_12[0]
    getitem_51: "f32[8, 384, 192]" = split_12[1];  split_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_12: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_51)
    mul_50: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_51, sigmoid_12);  sigmoid_12 = None
    mul_51: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_50, mul_50);  mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_44: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_51);  mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_51: "f32[3072, 192]" = torch.ops.aten.view.default(clone_44, [3072, 192]);  clone_44 = None
    permute_39: "f32[192, 196]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    addmm_18: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_80, view_51, permute_39);  primals_80 = None
    view_52: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_18, [8, 384, 196]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_45: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_52);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_40: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_45, [0, 2, 1]);  clone_45 = None
    add_45: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_41, permute_40);  add_41 = permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_46: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_45, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_46, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 196, 1]" = var_mean_13[0]
    getitem_53: "f32[8, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    add_46: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_13: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_13: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_46, getitem_53);  clone_46 = getitem_53 = None
    mul_52: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_53: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_52, primals_81)
    add_47: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_53, primals_82);  mul_53 = primals_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_53: "f32[1568, 384]" = torch.ops.aten.view.default(add_47, [1568, 384]);  add_47 = None
    permute_41: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    addmm_19: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_84, view_53, permute_41);  primals_84 = None
    view_54: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_19, [8, 196, 1536]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_13 = torch.ops.aten.split.Tensor(view_54, 768, -1);  view_54 = None
    getitem_54: "f32[8, 196, 768]" = split_13[0]
    getitem_55: "f32[8, 196, 768]" = split_13[1];  split_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_13: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_55)
    mul_54: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_55, sigmoid_13);  sigmoid_13 = None
    mul_55: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_54, mul_54);  mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_47: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_55);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_55: "f32[1568, 768]" = torch.ops.aten.view.default(clone_47, [1568, 768]);  clone_47 = None
    permute_42: "f32[768, 384]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_20: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_86, view_55, permute_42);  primals_86 = None
    view_56: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_20, [8, 196, 384]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_48: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_56);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_48: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_45, clone_48);  add_45 = clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_49: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_48, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_49, [2], correction = 0, keepdim = True)
    getitem_56: "f32[8, 196, 1]" = var_mean_14[0]
    getitem_57: "f32[8, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_14: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_14: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_49, getitem_57);  clone_49 = getitem_57 = None
    mul_56: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_57: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_56, primals_87)
    add_50: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_57, primals_88);  mul_57 = primals_88 = None
    permute_43: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_50, [0, 2, 1]);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_44: "f32[196, 384]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    clone_50: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    view_57: "f32[3072, 196]" = torch.ops.aten.view.default(clone_50, [3072, 196]);  clone_50 = None
    mm_7: "f32[3072, 384]" = torch.ops.aten.mm.default(view_57, permute_44)
    view_58: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_7, [8, 384, 384]);  mm_7 = None
    add_51: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_58, primals_90);  view_58 = primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_14 = torch.ops.aten.split.Tensor(add_51, 192, -1);  add_51 = None
    getitem_58: "f32[8, 384, 192]" = split_14[0]
    getitem_59: "f32[8, 384, 192]" = split_14[1];  split_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_14: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_59)
    mul_58: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_59, sigmoid_14);  sigmoid_14 = None
    mul_59: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_58, mul_58);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_51: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_59);  mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_59: "f32[3072, 192]" = torch.ops.aten.view.default(clone_51, [3072, 192]);  clone_51 = None
    permute_45: "f32[192, 196]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_21: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_92, view_59, permute_45);  primals_92 = None
    view_60: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_21, [8, 384, 196]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_52: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_46: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_52, [0, 2, 1]);  clone_52 = None
    add_52: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_48, permute_46);  add_48 = permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_53: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_52, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_53, [2], correction = 0, keepdim = True)
    getitem_60: "f32[8, 196, 1]" = var_mean_15[0]
    getitem_61: "f32[8, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    add_53: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
    rsqrt_15: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_15: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_53, getitem_61);  clone_53 = getitem_61 = None
    mul_60: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_61: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_60, primals_93)
    add_54: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_61, primals_94);  mul_61 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_61: "f32[1568, 384]" = torch.ops.aten.view.default(add_54, [1568, 384]);  add_54 = None
    permute_47: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_22: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_96, view_61, permute_47);  primals_96 = None
    view_62: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_22, [8, 196, 1536]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_15 = torch.ops.aten.split.Tensor(view_62, 768, -1);  view_62 = None
    getitem_62: "f32[8, 196, 768]" = split_15[0]
    getitem_63: "f32[8, 196, 768]" = split_15[1];  split_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_15: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_63)
    mul_62: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_63, sigmoid_15);  sigmoid_15 = None
    mul_63: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_62, mul_62);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_54: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_63);  mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_63: "f32[1568, 768]" = torch.ops.aten.view.default(clone_54, [1568, 768]);  clone_54 = None
    permute_48: "f32[768, 384]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_23: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_98, view_63, permute_48);  primals_98 = None
    view_64: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_23, [8, 196, 384]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_55: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_64);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_55: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_52, clone_55);  add_52 = clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_56: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_55, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_56, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 196, 1]" = var_mean_16[0]
    getitem_65: "f32[8, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    add_56: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_16: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_16: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_56, getitem_65);  clone_56 = getitem_65 = None
    mul_64: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_65: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_64, primals_99)
    add_57: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_65, primals_100);  mul_65 = primals_100 = None
    permute_49: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_57, [0, 2, 1]);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_50: "f32[196, 384]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    clone_57: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    view_65: "f32[3072, 196]" = torch.ops.aten.view.default(clone_57, [3072, 196]);  clone_57 = None
    mm_8: "f32[3072, 384]" = torch.ops.aten.mm.default(view_65, permute_50)
    view_66: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_8, [8, 384, 384]);  mm_8 = None
    add_58: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_66, primals_102);  view_66 = primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_16 = torch.ops.aten.split.Tensor(add_58, 192, -1);  add_58 = None
    getitem_66: "f32[8, 384, 192]" = split_16[0]
    getitem_67: "f32[8, 384, 192]" = split_16[1];  split_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_16: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_67)
    mul_66: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_67, sigmoid_16);  sigmoid_16 = None
    mul_67: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_66, mul_66);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_58: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_67);  mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_67: "f32[3072, 192]" = torch.ops.aten.view.default(clone_58, [3072, 192]);  clone_58 = None
    permute_51: "f32[192, 196]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_24: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_104, view_67, permute_51);  primals_104 = None
    view_68: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_24, [8, 384, 196]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_59: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_52: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_59, [0, 2, 1]);  clone_59 = None
    add_59: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_55, permute_52);  add_55 = permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_60: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_59, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_60, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 196, 1]" = var_mean_17[0]
    getitem_69: "f32[8, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    add_60: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_17: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_17: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_60, getitem_69);  clone_60 = getitem_69 = None
    mul_68: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_69: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_68, primals_105)
    add_61: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_69, primals_106);  mul_69 = primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_69: "f32[1568, 384]" = torch.ops.aten.view.default(add_61, [1568, 384]);  add_61 = None
    permute_53: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_25: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_108, view_69, permute_53);  primals_108 = None
    view_70: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_25, [8, 196, 1536]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_17 = torch.ops.aten.split.Tensor(view_70, 768, -1);  view_70 = None
    getitem_70: "f32[8, 196, 768]" = split_17[0]
    getitem_71: "f32[8, 196, 768]" = split_17[1];  split_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_17: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_71)
    mul_70: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_71, sigmoid_17);  sigmoid_17 = None
    mul_71: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_70, mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_61: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_71);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_71: "f32[1568, 768]" = torch.ops.aten.view.default(clone_61, [1568, 768]);  clone_61 = None
    permute_54: "f32[768, 384]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    addmm_26: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_110, view_71, permute_54);  primals_110 = None
    view_72: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_26, [8, 196, 384]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_62: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_62: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_59, clone_62);  add_59 = clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_63: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_63, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 196, 1]" = var_mean_18[0]
    getitem_73: "f32[8, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    add_63: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_18: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_18: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_63, getitem_73);  clone_63 = getitem_73 = None
    mul_72: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_73: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_72, primals_111)
    add_64: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_73, primals_112);  mul_73 = primals_112 = None
    permute_55: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_64, [0, 2, 1]);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_56: "f32[196, 384]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    clone_64: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
    view_73: "f32[3072, 196]" = torch.ops.aten.view.default(clone_64, [3072, 196]);  clone_64 = None
    mm_9: "f32[3072, 384]" = torch.ops.aten.mm.default(view_73, permute_56)
    view_74: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_9, [8, 384, 384]);  mm_9 = None
    add_65: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_74, primals_114);  view_74 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_18 = torch.ops.aten.split.Tensor(add_65, 192, -1);  add_65 = None
    getitem_74: "f32[8, 384, 192]" = split_18[0]
    getitem_75: "f32[8, 384, 192]" = split_18[1];  split_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_18: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_75)
    mul_74: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_75, sigmoid_18);  sigmoid_18 = None
    mul_75: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_74, mul_74);  mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_65: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_75);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_75: "f32[3072, 192]" = torch.ops.aten.view.default(clone_65, [3072, 192]);  clone_65 = None
    permute_57: "f32[192, 196]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_27: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_116, view_75, permute_57);  primals_116 = None
    view_76: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_27, [8, 384, 196]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_66: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_76);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_58: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_66, [0, 2, 1]);  clone_66 = None
    add_66: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_62, permute_58);  add_62 = permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_67: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_66, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_67, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 196, 1]" = var_mean_19[0]
    getitem_77: "f32[8, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    add_67: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_19: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_19: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_67, getitem_77);  clone_67 = getitem_77 = None
    mul_76: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_77: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_76, primals_117)
    add_68: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_77, primals_118);  mul_77 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_77: "f32[1568, 384]" = torch.ops.aten.view.default(add_68, [1568, 384]);  add_68 = None
    permute_59: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_28: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_120, view_77, permute_59);  primals_120 = None
    view_78: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_28, [8, 196, 1536]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_19 = torch.ops.aten.split.Tensor(view_78, 768, -1);  view_78 = None
    getitem_78: "f32[8, 196, 768]" = split_19[0]
    getitem_79: "f32[8, 196, 768]" = split_19[1];  split_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_19: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_79)
    mul_78: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_79, sigmoid_19);  sigmoid_19 = None
    mul_79: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_78, mul_78);  mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_68: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_79);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_79: "f32[1568, 768]" = torch.ops.aten.view.default(clone_68, [1568, 768]);  clone_68 = None
    permute_60: "f32[768, 384]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_29: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_122, view_79, permute_60);  primals_122 = None
    view_80: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_29, [8, 196, 384]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_69: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_69: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_66, clone_69);  add_66 = clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_70: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_69, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_70, [2], correction = 0, keepdim = True)
    getitem_80: "f32[8, 196, 1]" = var_mean_20[0]
    getitem_81: "f32[8, 196, 1]" = var_mean_20[1];  var_mean_20 = None
    add_70: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
    rsqrt_20: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_20: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_70, getitem_81);  clone_70 = getitem_81 = None
    mul_80: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_81: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_80, primals_123)
    add_71: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_81, primals_124);  mul_81 = primals_124 = None
    permute_61: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_71, [0, 2, 1]);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_62: "f32[196, 384]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    clone_71: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
    view_81: "f32[3072, 196]" = torch.ops.aten.view.default(clone_71, [3072, 196]);  clone_71 = None
    mm_10: "f32[3072, 384]" = torch.ops.aten.mm.default(view_81, permute_62)
    view_82: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_10, [8, 384, 384]);  mm_10 = None
    add_72: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_82, primals_126);  view_82 = primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_20 = torch.ops.aten.split.Tensor(add_72, 192, -1);  add_72 = None
    getitem_82: "f32[8, 384, 192]" = split_20[0]
    getitem_83: "f32[8, 384, 192]" = split_20[1];  split_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_20: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_83)
    mul_82: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_83, sigmoid_20);  sigmoid_20 = None
    mul_83: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_82, mul_82);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_72: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_83);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_83: "f32[3072, 192]" = torch.ops.aten.view.default(clone_72, [3072, 192]);  clone_72 = None
    permute_63: "f32[192, 196]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    addmm_30: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_128, view_83, permute_63);  primals_128 = None
    view_84: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_30, [8, 384, 196]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_73: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_84);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_64: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_73, [0, 2, 1]);  clone_73 = None
    add_73: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_69, permute_64);  add_69 = permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_74: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_73, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_74, [2], correction = 0, keepdim = True)
    getitem_84: "f32[8, 196, 1]" = var_mean_21[0]
    getitem_85: "f32[8, 196, 1]" = var_mean_21[1];  var_mean_21 = None
    add_74: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
    rsqrt_21: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_21: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_74, getitem_85);  clone_74 = getitem_85 = None
    mul_84: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_85: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_84, primals_129)
    add_75: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_85, primals_130);  mul_85 = primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_85: "f32[1568, 384]" = torch.ops.aten.view.default(add_75, [1568, 384]);  add_75 = None
    permute_65: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_31: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_132, view_85, permute_65);  primals_132 = None
    view_86: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_31, [8, 196, 1536]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_21 = torch.ops.aten.split.Tensor(view_86, 768, -1);  view_86 = None
    getitem_86: "f32[8, 196, 768]" = split_21[0]
    getitem_87: "f32[8, 196, 768]" = split_21[1];  split_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_21: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_87)
    mul_86: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_87, sigmoid_21);  sigmoid_21 = None
    mul_87: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_86, mul_86);  mul_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_75: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_87);  mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_87: "f32[1568, 768]" = torch.ops.aten.view.default(clone_75, [1568, 768]);  clone_75 = None
    permute_66: "f32[768, 384]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    addmm_32: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_134, view_87, permute_66);  primals_134 = None
    view_88: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_32, [8, 196, 384]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_76: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_88);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_76: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_73, clone_76);  add_73 = clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_77: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_77, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 196, 1]" = var_mean_22[0]
    getitem_89: "f32[8, 196, 1]" = var_mean_22[1];  var_mean_22 = None
    add_77: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_22: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_22: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_77, getitem_89);  clone_77 = getitem_89 = None
    mul_88: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_89: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_88, primals_135)
    add_78: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_89, primals_136);  mul_89 = primals_136 = None
    permute_67: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_78, [0, 2, 1]);  add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_68: "f32[196, 384]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    clone_78: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    view_89: "f32[3072, 196]" = torch.ops.aten.view.default(clone_78, [3072, 196]);  clone_78 = None
    mm_11: "f32[3072, 384]" = torch.ops.aten.mm.default(view_89, permute_68)
    view_90: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_11, [8, 384, 384]);  mm_11 = None
    add_79: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_90, primals_138);  view_90 = primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_22 = torch.ops.aten.split.Tensor(add_79, 192, -1);  add_79 = None
    getitem_90: "f32[8, 384, 192]" = split_22[0]
    getitem_91: "f32[8, 384, 192]" = split_22[1];  split_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_22: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_91)
    mul_90: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_91, sigmoid_22);  sigmoid_22 = None
    mul_91: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_90, mul_90);  mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_79: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_91);  mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_91: "f32[3072, 192]" = torch.ops.aten.view.default(clone_79, [3072, 192]);  clone_79 = None
    permute_69: "f32[192, 196]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    addmm_33: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_140, view_91, permute_69);  primals_140 = None
    view_92: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_33, [8, 384, 196]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_80: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_92);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_70: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_80, [0, 2, 1]);  clone_80 = None
    add_80: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_76, permute_70);  add_76 = permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_81: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_81, [2], correction = 0, keepdim = True)
    getitem_92: "f32[8, 196, 1]" = var_mean_23[0]
    getitem_93: "f32[8, 196, 1]" = var_mean_23[1];  var_mean_23 = None
    add_81: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
    rsqrt_23: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_23: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_81, getitem_93);  clone_81 = getitem_93 = None
    mul_92: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_93: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_92, primals_141)
    add_82: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_93, primals_142);  mul_93 = primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_93: "f32[1568, 384]" = torch.ops.aten.view.default(add_82, [1568, 384]);  add_82 = None
    permute_71: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_34: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_144, view_93, permute_71);  primals_144 = None
    view_94: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_34, [8, 196, 1536]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_23 = torch.ops.aten.split.Tensor(view_94, 768, -1);  view_94 = None
    getitem_94: "f32[8, 196, 768]" = split_23[0]
    getitem_95: "f32[8, 196, 768]" = split_23[1];  split_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_23: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_95)
    mul_94: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_95, sigmoid_23);  sigmoid_23 = None
    mul_95: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_94, mul_94);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_82: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_95);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_95: "f32[1568, 768]" = torch.ops.aten.view.default(clone_82, [1568, 768]);  clone_82 = None
    permute_72: "f32[768, 384]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_35: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_146, view_95, permute_72);  primals_146 = None
    view_96: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_35, [8, 196, 384]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_83: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_83: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_80, clone_83);  add_80 = clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_84: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_83, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_84, [2], correction = 0, keepdim = True)
    getitem_96: "f32[8, 196, 1]" = var_mean_24[0]
    getitem_97: "f32[8, 196, 1]" = var_mean_24[1];  var_mean_24 = None
    add_84: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
    rsqrt_24: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_24: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_84, getitem_97);  clone_84 = getitem_97 = None
    mul_96: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_97: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_96, primals_147)
    add_85: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_97, primals_148);  mul_97 = primals_148 = None
    permute_73: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_85, [0, 2, 1]);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_74: "f32[196, 384]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    clone_85: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_97: "f32[3072, 196]" = torch.ops.aten.view.default(clone_85, [3072, 196]);  clone_85 = None
    mm_12: "f32[3072, 384]" = torch.ops.aten.mm.default(view_97, permute_74)
    view_98: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_12, [8, 384, 384]);  mm_12 = None
    add_86: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_98, primals_150);  view_98 = primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_24 = torch.ops.aten.split.Tensor(add_86, 192, -1);  add_86 = None
    getitem_98: "f32[8, 384, 192]" = split_24[0]
    getitem_99: "f32[8, 384, 192]" = split_24[1];  split_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_24: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_99)
    mul_98: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_99, sigmoid_24);  sigmoid_24 = None
    mul_99: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_98, mul_98);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_86: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_99);  mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_99: "f32[3072, 192]" = torch.ops.aten.view.default(clone_86, [3072, 192]);  clone_86 = None
    permute_75: "f32[192, 196]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_36: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_152, view_99, permute_75);  primals_152 = None
    view_100: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_36, [8, 384, 196]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_87: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_76: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_87, [0, 2, 1]);  clone_87 = None
    add_87: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_83, permute_76);  add_83 = permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_88: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_87, memory_format = torch.contiguous_format)
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_88, [2], correction = 0, keepdim = True)
    getitem_100: "f32[8, 196, 1]" = var_mean_25[0]
    getitem_101: "f32[8, 196, 1]" = var_mean_25[1];  var_mean_25 = None
    add_88: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-06);  getitem_100 = None
    rsqrt_25: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_25: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_88, getitem_101);  clone_88 = getitem_101 = None
    mul_100: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    mul_101: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_100, primals_153)
    add_89: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_101, primals_154);  mul_101 = primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_101: "f32[1568, 384]" = torch.ops.aten.view.default(add_89, [1568, 384]);  add_89 = None
    permute_77: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_37: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_156, view_101, permute_77);  primals_156 = None
    view_102: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_37, [8, 196, 1536]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_25 = torch.ops.aten.split.Tensor(view_102, 768, -1);  view_102 = None
    getitem_102: "f32[8, 196, 768]" = split_25[0]
    getitem_103: "f32[8, 196, 768]" = split_25[1];  split_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_25: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_103)
    mul_102: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_103, sigmoid_25);  sigmoid_25 = None
    mul_103: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_102, mul_102);  mul_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_89: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_103);  mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_103: "f32[1568, 768]" = torch.ops.aten.view.default(clone_89, [1568, 768]);  clone_89 = None
    permute_78: "f32[768, 384]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm_38: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_158, view_103, permute_78);  primals_158 = None
    view_104: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_38, [8, 196, 384]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_90: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_90: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_87, clone_90);  add_87 = clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_91: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_91, [2], correction = 0, keepdim = True)
    getitem_104: "f32[8, 196, 1]" = var_mean_26[0]
    getitem_105: "f32[8, 196, 1]" = var_mean_26[1];  var_mean_26 = None
    add_91: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
    rsqrt_26: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_26: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_91, getitem_105);  clone_91 = getitem_105 = None
    mul_104: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    mul_105: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_104, primals_159)
    add_92: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_105, primals_160);  mul_105 = primals_160 = None
    permute_79: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_92, [0, 2, 1]);  add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_80: "f32[196, 384]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    clone_92: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    view_105: "f32[3072, 196]" = torch.ops.aten.view.default(clone_92, [3072, 196]);  clone_92 = None
    mm_13: "f32[3072, 384]" = torch.ops.aten.mm.default(view_105, permute_80)
    view_106: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_13, [8, 384, 384]);  mm_13 = None
    add_93: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_106, primals_162);  view_106 = primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_26 = torch.ops.aten.split.Tensor(add_93, 192, -1);  add_93 = None
    getitem_106: "f32[8, 384, 192]" = split_26[0]
    getitem_107: "f32[8, 384, 192]" = split_26[1];  split_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_26: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_107)
    mul_106: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_107, sigmoid_26);  sigmoid_26 = None
    mul_107: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_106, mul_106);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_93: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_107);  mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_107: "f32[3072, 192]" = torch.ops.aten.view.default(clone_93, [3072, 192]);  clone_93 = None
    permute_81: "f32[192, 196]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    addmm_39: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_164, view_107, permute_81);  primals_164 = None
    view_108: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_39, [8, 384, 196]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_94: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_82: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_94, [0, 2, 1]);  clone_94 = None
    add_94: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_90, permute_82);  add_90 = permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_95: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_95, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 196, 1]" = var_mean_27[0]
    getitem_109: "f32[8, 196, 1]" = var_mean_27[1];  var_mean_27 = None
    add_95: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_27: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_27: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_95, getitem_109);  clone_95 = getitem_109 = None
    mul_108: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    mul_109: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_108, primals_165)
    add_96: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_109, primals_166);  mul_109 = primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_109: "f32[1568, 384]" = torch.ops.aten.view.default(add_96, [1568, 384]);  add_96 = None
    permute_83: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
    addmm_40: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_168, view_109, permute_83);  primals_168 = None
    view_110: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_40, [8, 196, 1536]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_27 = torch.ops.aten.split.Tensor(view_110, 768, -1);  view_110 = None
    getitem_110: "f32[8, 196, 768]" = split_27[0]
    getitem_111: "f32[8, 196, 768]" = split_27[1];  split_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_27: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_111)
    mul_110: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_111, sigmoid_27);  sigmoid_27 = None
    mul_111: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_110, mul_110);  mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_96: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_111);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_111: "f32[1568, 768]" = torch.ops.aten.view.default(clone_96, [1568, 768]);  clone_96 = None
    permute_84: "f32[768, 384]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    addmm_41: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_170, view_111, permute_84);  primals_170 = None
    view_112: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_41, [8, 196, 384]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_97: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_112);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_97: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_94, clone_97);  add_94 = clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_98: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_97, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_98, [2], correction = 0, keepdim = True)
    getitem_112: "f32[8, 196, 1]" = var_mean_28[0]
    getitem_113: "f32[8, 196, 1]" = var_mean_28[1];  var_mean_28 = None
    add_98: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
    rsqrt_28: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_28: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_98, getitem_113);  clone_98 = getitem_113 = None
    mul_112: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    mul_113: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_112, primals_171)
    add_99: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_113, primals_172);  mul_113 = primals_172 = None
    permute_85: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_99, [0, 2, 1]);  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_86: "f32[196, 384]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
    clone_99: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_113: "f32[3072, 196]" = torch.ops.aten.view.default(clone_99, [3072, 196]);  clone_99 = None
    mm_14: "f32[3072, 384]" = torch.ops.aten.mm.default(view_113, permute_86)
    view_114: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_14, [8, 384, 384]);  mm_14 = None
    add_100: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_114, primals_174);  view_114 = primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_28 = torch.ops.aten.split.Tensor(add_100, 192, -1);  add_100 = None
    getitem_114: "f32[8, 384, 192]" = split_28[0]
    getitem_115: "f32[8, 384, 192]" = split_28[1];  split_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_28: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_115)
    mul_114: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_115, sigmoid_28);  sigmoid_28 = None
    mul_115: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_114, mul_114);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_100: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_115);  mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_115: "f32[3072, 192]" = torch.ops.aten.view.default(clone_100, [3072, 192]);  clone_100 = None
    permute_87: "f32[192, 196]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_42: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_176, view_115, permute_87);  primals_176 = None
    view_116: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_42, [8, 384, 196]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_101: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_116);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_88: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_101, [0, 2, 1]);  clone_101 = None
    add_101: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_97, permute_88);  add_97 = permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_102: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_101, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_102, [2], correction = 0, keepdim = True)
    getitem_116: "f32[8, 196, 1]" = var_mean_29[0]
    getitem_117: "f32[8, 196, 1]" = var_mean_29[1];  var_mean_29 = None
    add_102: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-06);  getitem_116 = None
    rsqrt_29: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_29: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_102, getitem_117);  clone_102 = getitem_117 = None
    mul_116: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    mul_117: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_116, primals_177)
    add_103: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_117, primals_178);  mul_117 = primals_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_117: "f32[1568, 384]" = torch.ops.aten.view.default(add_103, [1568, 384]);  add_103 = None
    permute_89: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    addmm_43: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_180, view_117, permute_89);  primals_180 = None
    view_118: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_43, [8, 196, 1536]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_29 = torch.ops.aten.split.Tensor(view_118, 768, -1);  view_118 = None
    getitem_118: "f32[8, 196, 768]" = split_29[0]
    getitem_119: "f32[8, 196, 768]" = split_29[1];  split_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_29: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_119)
    mul_118: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_119, sigmoid_29);  sigmoid_29 = None
    mul_119: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_118, mul_118);  mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_103: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_119);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_119: "f32[1568, 768]" = torch.ops.aten.view.default(clone_103, [1568, 768]);  clone_103 = None
    permute_90: "f32[768, 384]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_44: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_182, view_119, permute_90);  primals_182 = None
    view_120: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_44, [8, 196, 384]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_104: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_104: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_101, clone_104);  add_101 = clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_105: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_104, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_105, [2], correction = 0, keepdim = True)
    getitem_120: "f32[8, 196, 1]" = var_mean_30[0]
    getitem_121: "f32[8, 196, 1]" = var_mean_30[1];  var_mean_30 = None
    add_105: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
    rsqrt_30: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_30: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_105, getitem_121);  clone_105 = getitem_121 = None
    mul_120: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    mul_121: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_120, primals_183)
    add_106: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_121, primals_184);  mul_121 = primals_184 = None
    permute_91: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_106, [0, 2, 1]);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_92: "f32[196, 384]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    clone_106: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    view_121: "f32[3072, 196]" = torch.ops.aten.view.default(clone_106, [3072, 196]);  clone_106 = None
    mm_15: "f32[3072, 384]" = torch.ops.aten.mm.default(view_121, permute_92)
    view_122: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_15, [8, 384, 384]);  mm_15 = None
    add_107: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_122, primals_186);  view_122 = primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_30 = torch.ops.aten.split.Tensor(add_107, 192, -1);  add_107 = None
    getitem_122: "f32[8, 384, 192]" = split_30[0]
    getitem_123: "f32[8, 384, 192]" = split_30[1];  split_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_30: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_123)
    mul_122: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_123, sigmoid_30);  sigmoid_30 = None
    mul_123: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_122, mul_122);  mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_107: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_123);  mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_123: "f32[3072, 192]" = torch.ops.aten.view.default(clone_107, [3072, 192]);  clone_107 = None
    permute_93: "f32[192, 196]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    addmm_45: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_188, view_123, permute_93);  primals_188 = None
    view_124: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_45, [8, 384, 196]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_108: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_124);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_94: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_108, [0, 2, 1]);  clone_108 = None
    add_108: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_104, permute_94);  add_104 = permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_109: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_108, memory_format = torch.contiguous_format)
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_109, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 196, 1]" = var_mean_31[0]
    getitem_125: "f32[8, 196, 1]" = var_mean_31[1];  var_mean_31 = None
    add_109: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-06);  getitem_124 = None
    rsqrt_31: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    sub_31: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_109, getitem_125);  clone_109 = getitem_125 = None
    mul_124: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    mul_125: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_124, primals_189)
    add_110: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_125, primals_190);  mul_125 = primals_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_125: "f32[1568, 384]" = torch.ops.aten.view.default(add_110, [1568, 384]);  add_110 = None
    permute_95: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
    addmm_46: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_192, view_125, permute_95);  primals_192 = None
    view_126: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_46, [8, 196, 1536]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_31 = torch.ops.aten.split.Tensor(view_126, 768, -1);  view_126 = None
    getitem_126: "f32[8, 196, 768]" = split_31[0]
    getitem_127: "f32[8, 196, 768]" = split_31[1];  split_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_31: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_127)
    mul_126: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_127, sigmoid_31);  sigmoid_31 = None
    mul_127: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_126, mul_126);  mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_110: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_127);  mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_127: "f32[1568, 768]" = torch.ops.aten.view.default(clone_110, [1568, 768]);  clone_110 = None
    permute_96: "f32[768, 384]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
    addmm_47: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_194, view_127, permute_96);  primals_194 = None
    view_128: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_47, [8, 196, 384]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_111: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_128);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_111: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_108, clone_111);  add_108 = clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_112: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_112, [2], correction = 0, keepdim = True)
    getitem_128: "f32[8, 196, 1]" = var_mean_32[0]
    getitem_129: "f32[8, 196, 1]" = var_mean_32[1];  var_mean_32 = None
    add_112: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
    rsqrt_32: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_32: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_112, getitem_129);  clone_112 = getitem_129 = None
    mul_128: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    mul_129: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_128, primals_195)
    add_113: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_129, primals_196);  mul_129 = primals_196 = None
    permute_97: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_113, [0, 2, 1]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_98: "f32[196, 384]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
    clone_113: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    view_129: "f32[3072, 196]" = torch.ops.aten.view.default(clone_113, [3072, 196]);  clone_113 = None
    mm_16: "f32[3072, 384]" = torch.ops.aten.mm.default(view_129, permute_98)
    view_130: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_16, [8, 384, 384]);  mm_16 = None
    add_114: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_130, primals_198);  view_130 = primals_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_32 = torch.ops.aten.split.Tensor(add_114, 192, -1);  add_114 = None
    getitem_130: "f32[8, 384, 192]" = split_32[0]
    getitem_131: "f32[8, 384, 192]" = split_32[1];  split_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_32: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_131)
    mul_130: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_131, sigmoid_32);  sigmoid_32 = None
    mul_131: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_130, mul_130);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_114: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_131);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_131: "f32[3072, 192]" = torch.ops.aten.view.default(clone_114, [3072, 192]);  clone_114 = None
    permute_99: "f32[192, 196]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    addmm_48: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_200, view_131, permute_99);  primals_200 = None
    view_132: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_48, [8, 384, 196]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_115: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_132);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_100: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_115, [0, 2, 1]);  clone_115 = None
    add_115: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_111, permute_100);  add_111 = permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_116: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_115, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_116, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 196, 1]" = var_mean_33[0]
    getitem_133: "f32[8, 196, 1]" = var_mean_33[1];  var_mean_33 = None
    add_116: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_33: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_33: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_116, getitem_133);  clone_116 = getitem_133 = None
    mul_132: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    mul_133: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_132, primals_201)
    add_117: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_133, primals_202);  mul_133 = primals_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_133: "f32[1568, 384]" = torch.ops.aten.view.default(add_117, [1568, 384]);  add_117 = None
    permute_101: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
    addmm_49: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_204, view_133, permute_101);  primals_204 = None
    view_134: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_49, [8, 196, 1536]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_33 = torch.ops.aten.split.Tensor(view_134, 768, -1);  view_134 = None
    getitem_134: "f32[8, 196, 768]" = split_33[0]
    getitem_135: "f32[8, 196, 768]" = split_33[1];  split_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_33: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_135)
    mul_134: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_135, sigmoid_33);  sigmoid_33 = None
    mul_135: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_134, mul_134);  mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_117: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_135);  mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_135: "f32[1568, 768]" = torch.ops.aten.view.default(clone_117, [1568, 768]);  clone_117 = None
    permute_102: "f32[768, 384]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    addmm_50: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_206, view_135, permute_102);  primals_206 = None
    view_136: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_50, [8, 196, 384]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_118: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_136);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_118: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_115, clone_118);  add_115 = clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_119: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_118, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_119, [2], correction = 0, keepdim = True)
    getitem_136: "f32[8, 196, 1]" = var_mean_34[0]
    getitem_137: "f32[8, 196, 1]" = var_mean_34[1];  var_mean_34 = None
    add_119: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-06);  getitem_136 = None
    rsqrt_34: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_34: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_119, getitem_137);  clone_119 = getitem_137 = None
    mul_136: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    mul_137: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_136, primals_207)
    add_120: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_137, primals_208);  mul_137 = primals_208 = None
    permute_103: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_120, [0, 2, 1]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_104: "f32[196, 384]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    clone_120: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    view_137: "f32[3072, 196]" = torch.ops.aten.view.default(clone_120, [3072, 196]);  clone_120 = None
    mm_17: "f32[3072, 384]" = torch.ops.aten.mm.default(view_137, permute_104)
    view_138: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_17, [8, 384, 384]);  mm_17 = None
    add_121: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_138, primals_210);  view_138 = primals_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_34 = torch.ops.aten.split.Tensor(add_121, 192, -1);  add_121 = None
    getitem_138: "f32[8, 384, 192]" = split_34[0]
    getitem_139: "f32[8, 384, 192]" = split_34[1];  split_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_34: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_139)
    mul_138: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_139, sigmoid_34);  sigmoid_34 = None
    mul_139: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_138, mul_138);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_121: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_139);  mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_139: "f32[3072, 192]" = torch.ops.aten.view.default(clone_121, [3072, 192]);  clone_121 = None
    permute_105: "f32[192, 196]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
    addmm_51: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_212, view_139, permute_105);  primals_212 = None
    view_140: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_51, [8, 384, 196]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_122: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_106: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_122, [0, 2, 1]);  clone_122 = None
    add_122: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_118, permute_106);  add_118 = permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_123: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_122, memory_format = torch.contiguous_format)
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_123, [2], correction = 0, keepdim = True)
    getitem_140: "f32[8, 196, 1]" = var_mean_35[0]
    getitem_141: "f32[8, 196, 1]" = var_mean_35[1];  var_mean_35 = None
    add_123: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-06);  getitem_140 = None
    rsqrt_35: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_35: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_123, getitem_141);  clone_123 = getitem_141 = None
    mul_140: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    mul_141: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_140, primals_213)
    add_124: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_141, primals_214);  mul_141 = primals_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_141: "f32[1568, 384]" = torch.ops.aten.view.default(add_124, [1568, 384]);  add_124 = None
    permute_107: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    addmm_52: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_216, view_141, permute_107);  primals_216 = None
    view_142: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_52, [8, 196, 1536]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_35 = torch.ops.aten.split.Tensor(view_142, 768, -1);  view_142 = None
    getitem_142: "f32[8, 196, 768]" = split_35[0]
    getitem_143: "f32[8, 196, 768]" = split_35[1];  split_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_35: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_143)
    mul_142: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_143, sigmoid_35);  sigmoid_35 = None
    mul_143: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_142, mul_142);  mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_124: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_143);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_143: "f32[1568, 768]" = torch.ops.aten.view.default(clone_124, [1568, 768]);  clone_124 = None
    permute_108: "f32[768, 384]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
    addmm_53: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_218, view_143, permute_108);  primals_218 = None
    view_144: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_53, [8, 196, 384]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_125: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_125: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_122, clone_125);  add_122 = clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_126: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_126, [2], correction = 0, keepdim = True)
    getitem_144: "f32[8, 196, 1]" = var_mean_36[0]
    getitem_145: "f32[8, 196, 1]" = var_mean_36[1];  var_mean_36 = None
    add_126: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_36: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_36: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_126, getitem_145);  clone_126 = getitem_145 = None
    mul_144: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    mul_145: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_144, primals_219)
    add_127: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_145, primals_220);  mul_145 = primals_220 = None
    permute_109: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_127, [0, 2, 1]);  add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_110: "f32[196, 384]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
    clone_127: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
    view_145: "f32[3072, 196]" = torch.ops.aten.view.default(clone_127, [3072, 196]);  clone_127 = None
    mm_18: "f32[3072, 384]" = torch.ops.aten.mm.default(view_145, permute_110)
    view_146: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_18, [8, 384, 384]);  mm_18 = None
    add_128: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_146, primals_222);  view_146 = primals_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_36 = torch.ops.aten.split.Tensor(add_128, 192, -1);  add_128 = None
    getitem_146: "f32[8, 384, 192]" = split_36[0]
    getitem_147: "f32[8, 384, 192]" = split_36[1];  split_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_36: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_147)
    mul_146: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_147, sigmoid_36);  sigmoid_36 = None
    mul_147: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_146, mul_146);  mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_128: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_147);  mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_147: "f32[3072, 192]" = torch.ops.aten.view.default(clone_128, [3072, 192]);  clone_128 = None
    permute_111: "f32[192, 196]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
    addmm_54: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_224, view_147, permute_111);  primals_224 = None
    view_148: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_54, [8, 384, 196]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_129: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_148);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_112: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_129, [0, 2, 1]);  clone_129 = None
    add_129: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_125, permute_112);  add_125 = permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_130: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_129, memory_format = torch.contiguous_format)
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_130, [2], correction = 0, keepdim = True)
    getitem_148: "f32[8, 196, 1]" = var_mean_37[0]
    getitem_149: "f32[8, 196, 1]" = var_mean_37[1];  var_mean_37 = None
    add_130: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-06);  getitem_148 = None
    rsqrt_37: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_37: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_130, getitem_149);  clone_130 = getitem_149 = None
    mul_148: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    mul_149: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_148, primals_225)
    add_131: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_149, primals_226);  mul_149 = primals_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_149: "f32[1568, 384]" = torch.ops.aten.view.default(add_131, [1568, 384]);  add_131 = None
    permute_113: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
    addmm_55: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_228, view_149, permute_113);  primals_228 = None
    view_150: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_55, [8, 196, 1536]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_37 = torch.ops.aten.split.Tensor(view_150, 768, -1);  view_150 = None
    getitem_150: "f32[8, 196, 768]" = split_37[0]
    getitem_151: "f32[8, 196, 768]" = split_37[1];  split_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_37: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_151)
    mul_150: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_151, sigmoid_37);  sigmoid_37 = None
    mul_151: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_150, mul_150);  mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_131: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_151);  mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_151: "f32[1568, 768]" = torch.ops.aten.view.default(clone_131, [1568, 768]);  clone_131 = None
    permute_114: "f32[768, 384]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
    addmm_56: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_230, view_151, permute_114);  primals_230 = None
    view_152: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_56, [8, 196, 384]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_132: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_152);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_132: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_129, clone_132);  add_129 = clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_133: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_132, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_133, [2], correction = 0, keepdim = True)
    getitem_152: "f32[8, 196, 1]" = var_mean_38[0]
    getitem_153: "f32[8, 196, 1]" = var_mean_38[1];  var_mean_38 = None
    add_133: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-06);  getitem_152 = None
    rsqrt_38: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_38: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_133, getitem_153);  clone_133 = getitem_153 = None
    mul_152: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    mul_153: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_152, primals_231)
    add_134: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_153, primals_232);  mul_153 = primals_232 = None
    permute_115: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_134, [0, 2, 1]);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_116: "f32[196, 384]" = torch.ops.aten.permute.default(primals_233, [1, 0]);  primals_233 = None
    clone_134: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    view_153: "f32[3072, 196]" = torch.ops.aten.view.default(clone_134, [3072, 196]);  clone_134 = None
    mm_19: "f32[3072, 384]" = torch.ops.aten.mm.default(view_153, permute_116)
    view_154: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_19, [8, 384, 384]);  mm_19 = None
    add_135: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_154, primals_234);  view_154 = primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_38 = torch.ops.aten.split.Tensor(add_135, 192, -1);  add_135 = None
    getitem_154: "f32[8, 384, 192]" = split_38[0]
    getitem_155: "f32[8, 384, 192]" = split_38[1];  split_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_38: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_155)
    mul_154: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_155, sigmoid_38);  sigmoid_38 = None
    mul_155: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_154, mul_154);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_135: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_155);  mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_155: "f32[3072, 192]" = torch.ops.aten.view.default(clone_135, [3072, 192]);  clone_135 = None
    permute_117: "f32[192, 196]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    addmm_57: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_236, view_155, permute_117);  primals_236 = None
    view_156: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_57, [8, 384, 196]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_136: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_156);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_118: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_136, [0, 2, 1]);  clone_136 = None
    add_136: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_132, permute_118);  add_132 = permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_137: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_136, memory_format = torch.contiguous_format)
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_137, [2], correction = 0, keepdim = True)
    getitem_156: "f32[8, 196, 1]" = var_mean_39[0]
    getitem_157: "f32[8, 196, 1]" = var_mean_39[1];  var_mean_39 = None
    add_137: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
    rsqrt_39: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_39: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_137, getitem_157);  clone_137 = getitem_157 = None
    mul_156: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    mul_157: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_156, primals_237)
    add_138: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_157, primals_238);  mul_157 = primals_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_157: "f32[1568, 384]" = torch.ops.aten.view.default(add_138, [1568, 384]);  add_138 = None
    permute_119: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
    addmm_58: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_240, view_157, permute_119);  primals_240 = None
    view_158: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_58, [8, 196, 1536]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_39 = torch.ops.aten.split.Tensor(view_158, 768, -1);  view_158 = None
    getitem_158: "f32[8, 196, 768]" = split_39[0]
    getitem_159: "f32[8, 196, 768]" = split_39[1];  split_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_39: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_159)
    mul_158: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_159, sigmoid_39);  sigmoid_39 = None
    mul_159: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_158, mul_158);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_138: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_159);  mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_159: "f32[1568, 768]" = torch.ops.aten.view.default(clone_138, [1568, 768]);  clone_138 = None
    permute_120: "f32[768, 384]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
    addmm_59: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_242, view_159, permute_120);  primals_242 = None
    view_160: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_59, [8, 196, 384]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_139: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_160);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_139: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_136, clone_139);  add_136 = clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_140: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_139, memory_format = torch.contiguous_format)
    var_mean_40 = torch.ops.aten.var_mean.correction(clone_140, [2], correction = 0, keepdim = True)
    getitem_160: "f32[8, 196, 1]" = var_mean_40[0]
    getitem_161: "f32[8, 196, 1]" = var_mean_40[1];  var_mean_40 = None
    add_140: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
    rsqrt_40: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_40: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_140, getitem_161);  clone_140 = getitem_161 = None
    mul_160: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    mul_161: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_160, primals_243)
    add_141: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_161, primals_244);  mul_161 = primals_244 = None
    permute_121: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_141, [0, 2, 1]);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_122: "f32[196, 384]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
    clone_141: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_161: "f32[3072, 196]" = torch.ops.aten.view.default(clone_141, [3072, 196]);  clone_141 = None
    mm_20: "f32[3072, 384]" = torch.ops.aten.mm.default(view_161, permute_122)
    view_162: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_20, [8, 384, 384]);  mm_20 = None
    add_142: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_162, primals_246);  view_162 = primals_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_40 = torch.ops.aten.split.Tensor(add_142, 192, -1);  add_142 = None
    getitem_162: "f32[8, 384, 192]" = split_40[0]
    getitem_163: "f32[8, 384, 192]" = split_40[1];  split_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_40: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_163)
    mul_162: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_163, sigmoid_40);  sigmoid_40 = None
    mul_163: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_162, mul_162);  mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_142: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_163);  mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_163: "f32[3072, 192]" = torch.ops.aten.view.default(clone_142, [3072, 192]);  clone_142 = None
    permute_123: "f32[192, 196]" = torch.ops.aten.permute.default(primals_247, [1, 0]);  primals_247 = None
    addmm_60: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_248, view_163, permute_123);  primals_248 = None
    view_164: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_60, [8, 384, 196]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_143: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_164);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_124: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_143, [0, 2, 1]);  clone_143 = None
    add_143: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_139, permute_124);  add_139 = permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_144: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_143, memory_format = torch.contiguous_format)
    var_mean_41 = torch.ops.aten.var_mean.correction(clone_144, [2], correction = 0, keepdim = True)
    getitem_164: "f32[8, 196, 1]" = var_mean_41[0]
    getitem_165: "f32[8, 196, 1]" = var_mean_41[1];  var_mean_41 = None
    add_144: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-06);  getitem_164 = None
    rsqrt_41: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_41: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_144, getitem_165);  clone_144 = getitem_165 = None
    mul_164: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    mul_165: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_164, primals_249)
    add_145: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_165, primals_250);  mul_165 = primals_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_165: "f32[1568, 384]" = torch.ops.aten.view.default(add_145, [1568, 384]);  add_145 = None
    permute_125: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
    addmm_61: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_252, view_165, permute_125);  primals_252 = None
    view_166: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_61, [8, 196, 1536]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_41 = torch.ops.aten.split.Tensor(view_166, 768, -1);  view_166 = None
    getitem_166: "f32[8, 196, 768]" = split_41[0]
    getitem_167: "f32[8, 196, 768]" = split_41[1];  split_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_41: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_167)
    mul_166: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_167, sigmoid_41);  sigmoid_41 = None
    mul_167: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_166, mul_166);  mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_145: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_167);  mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_167: "f32[1568, 768]" = torch.ops.aten.view.default(clone_145, [1568, 768]);  clone_145 = None
    permute_126: "f32[768, 384]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
    addmm_62: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_254, view_167, permute_126);  primals_254 = None
    view_168: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_62, [8, 196, 384]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_146: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_168);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_146: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_143, clone_146);  add_143 = clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_147: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_146, memory_format = torch.contiguous_format)
    var_mean_42 = torch.ops.aten.var_mean.correction(clone_147, [2], correction = 0, keepdim = True)
    getitem_168: "f32[8, 196, 1]" = var_mean_42[0]
    getitem_169: "f32[8, 196, 1]" = var_mean_42[1];  var_mean_42 = None
    add_147: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
    rsqrt_42: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_42: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_147, getitem_169);  clone_147 = getitem_169 = None
    mul_168: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    mul_169: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_168, primals_255)
    add_148: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_169, primals_256);  mul_169 = primals_256 = None
    permute_127: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_148, [0, 2, 1]);  add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_128: "f32[196, 384]" = torch.ops.aten.permute.default(primals_257, [1, 0]);  primals_257 = None
    clone_148: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    view_169: "f32[3072, 196]" = torch.ops.aten.view.default(clone_148, [3072, 196]);  clone_148 = None
    mm_21: "f32[3072, 384]" = torch.ops.aten.mm.default(view_169, permute_128)
    view_170: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_21, [8, 384, 384]);  mm_21 = None
    add_149: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_170, primals_258);  view_170 = primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_42 = torch.ops.aten.split.Tensor(add_149, 192, -1);  add_149 = None
    getitem_170: "f32[8, 384, 192]" = split_42[0]
    getitem_171: "f32[8, 384, 192]" = split_42[1];  split_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_42: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_171)
    mul_170: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_171, sigmoid_42);  sigmoid_42 = None
    mul_171: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_170, mul_170);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_149: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_171);  mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_171: "f32[3072, 192]" = torch.ops.aten.view.default(clone_149, [3072, 192]);  clone_149 = None
    permute_129: "f32[192, 196]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
    addmm_63: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_260, view_171, permute_129);  primals_260 = None
    view_172: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_63, [8, 384, 196]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_150: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_172);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_130: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_150, [0, 2, 1]);  clone_150 = None
    add_150: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_146, permute_130);  add_146 = permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_151: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_150, memory_format = torch.contiguous_format)
    var_mean_43 = torch.ops.aten.var_mean.correction(clone_151, [2], correction = 0, keepdim = True)
    getitem_172: "f32[8, 196, 1]" = var_mean_43[0]
    getitem_173: "f32[8, 196, 1]" = var_mean_43[1];  var_mean_43 = None
    add_151: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-06);  getitem_172 = None
    rsqrt_43: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_43: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_151, getitem_173);  clone_151 = getitem_173 = None
    mul_172: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    mul_173: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_172, primals_261)
    add_152: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_173, primals_262);  mul_173 = primals_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_173: "f32[1568, 384]" = torch.ops.aten.view.default(add_152, [1568, 384]);  add_152 = None
    permute_131: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_263, [1, 0]);  primals_263 = None
    addmm_64: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_264, view_173, permute_131);  primals_264 = None
    view_174: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_64, [8, 196, 1536]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_43 = torch.ops.aten.split.Tensor(view_174, 768, -1);  view_174 = None
    getitem_174: "f32[8, 196, 768]" = split_43[0]
    getitem_175: "f32[8, 196, 768]" = split_43[1];  split_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_43: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_175)
    mul_174: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_175, sigmoid_43);  sigmoid_43 = None
    mul_175: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_174, mul_174);  mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_152: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_175);  mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_175: "f32[1568, 768]" = torch.ops.aten.view.default(clone_152, [1568, 768]);  clone_152 = None
    permute_132: "f32[768, 384]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
    addmm_65: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_266, view_175, permute_132);  primals_266 = None
    view_176: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_65, [8, 196, 384]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_153: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_176);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_153: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_150, clone_153);  add_150 = clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_154: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_153, memory_format = torch.contiguous_format)
    var_mean_44 = torch.ops.aten.var_mean.correction(clone_154, [2], correction = 0, keepdim = True)
    getitem_176: "f32[8, 196, 1]" = var_mean_44[0]
    getitem_177: "f32[8, 196, 1]" = var_mean_44[1];  var_mean_44 = None
    add_154: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
    rsqrt_44: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_44: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_154, getitem_177);  clone_154 = getitem_177 = None
    mul_176: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    mul_177: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_176, primals_267)
    add_155: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_177, primals_268);  mul_177 = primals_268 = None
    permute_133: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_155, [0, 2, 1]);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_134: "f32[196, 384]" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
    clone_155: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    view_177: "f32[3072, 196]" = torch.ops.aten.view.default(clone_155, [3072, 196]);  clone_155 = None
    mm_22: "f32[3072, 384]" = torch.ops.aten.mm.default(view_177, permute_134)
    view_178: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_22, [8, 384, 384]);  mm_22 = None
    add_156: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_178, primals_270);  view_178 = primals_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_44 = torch.ops.aten.split.Tensor(add_156, 192, -1);  add_156 = None
    getitem_178: "f32[8, 384, 192]" = split_44[0]
    getitem_179: "f32[8, 384, 192]" = split_44[1];  split_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_44: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_179)
    mul_178: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_179, sigmoid_44);  sigmoid_44 = None
    mul_179: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_178, mul_178);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_156: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_179);  mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_179: "f32[3072, 192]" = torch.ops.aten.view.default(clone_156, [3072, 192]);  clone_156 = None
    permute_135: "f32[192, 196]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
    addmm_66: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_272, view_179, permute_135);  primals_272 = None
    view_180: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_66, [8, 384, 196]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_157: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_136: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_157, [0, 2, 1]);  clone_157 = None
    add_157: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_153, permute_136);  add_153 = permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_158: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_157, memory_format = torch.contiguous_format)
    var_mean_45 = torch.ops.aten.var_mean.correction(clone_158, [2], correction = 0, keepdim = True)
    getitem_180: "f32[8, 196, 1]" = var_mean_45[0]
    getitem_181: "f32[8, 196, 1]" = var_mean_45[1];  var_mean_45 = None
    add_158: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-06);  getitem_180 = None
    rsqrt_45: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_45: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_158, getitem_181);  clone_158 = getitem_181 = None
    mul_180: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    mul_181: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_180, primals_273)
    add_159: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_181, primals_274);  mul_181 = primals_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_181: "f32[1568, 384]" = torch.ops.aten.view.default(add_159, [1568, 384]);  add_159 = None
    permute_137: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_275, [1, 0]);  primals_275 = None
    addmm_67: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_276, view_181, permute_137);  primals_276 = None
    view_182: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_67, [8, 196, 1536]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_45 = torch.ops.aten.split.Tensor(view_182, 768, -1);  view_182 = None
    getitem_182: "f32[8, 196, 768]" = split_45[0]
    getitem_183: "f32[8, 196, 768]" = split_45[1];  split_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_45: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_183)
    mul_182: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_183, sigmoid_45);  sigmoid_45 = None
    mul_183: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_182, mul_182);  mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_159: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_183);  mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_183: "f32[1568, 768]" = torch.ops.aten.view.default(clone_159, [1568, 768]);  clone_159 = None
    permute_138: "f32[768, 384]" = torch.ops.aten.permute.default(primals_277, [1, 0]);  primals_277 = None
    addmm_68: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_278, view_183, permute_138);  primals_278 = None
    view_184: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_68, [8, 196, 384]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_160: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_184);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_160: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_157, clone_160);  add_157 = clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_161: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_160, memory_format = torch.contiguous_format)
    var_mean_46 = torch.ops.aten.var_mean.correction(clone_161, [2], correction = 0, keepdim = True)
    getitem_184: "f32[8, 196, 1]" = var_mean_46[0]
    getitem_185: "f32[8, 196, 1]" = var_mean_46[1];  var_mean_46 = None
    add_161: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-06);  getitem_184 = None
    rsqrt_46: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_46: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_161, getitem_185);  clone_161 = getitem_185 = None
    mul_184: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    mul_185: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_184, primals_279)
    add_162: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_185, primals_280);  mul_185 = primals_280 = None
    permute_139: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_162, [0, 2, 1]);  add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_140: "f32[196, 384]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
    clone_162: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    view_185: "f32[3072, 196]" = torch.ops.aten.view.default(clone_162, [3072, 196]);  clone_162 = None
    mm_23: "f32[3072, 384]" = torch.ops.aten.mm.default(view_185, permute_140)
    view_186: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_23, [8, 384, 384]);  mm_23 = None
    add_163: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_186, primals_282);  view_186 = primals_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_46 = torch.ops.aten.split.Tensor(add_163, 192, -1);  add_163 = None
    getitem_186: "f32[8, 384, 192]" = split_46[0]
    getitem_187: "f32[8, 384, 192]" = split_46[1];  split_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_46: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_187)
    mul_186: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_187, sigmoid_46);  sigmoid_46 = None
    mul_187: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_186, mul_186);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_163: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_187);  mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_187: "f32[3072, 192]" = torch.ops.aten.view.default(clone_163, [3072, 192]);  clone_163 = None
    permute_141: "f32[192, 196]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
    addmm_69: "f32[3072, 196]" = torch.ops.aten.addmm.default(primals_284, view_187, permute_141);  primals_284 = None
    view_188: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_69, [8, 384, 196]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_164: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_188);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_142: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_164, [0, 2, 1]);  clone_164 = None
    add_164: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_160, permute_142);  add_160 = permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_165: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_164, memory_format = torch.contiguous_format)
    var_mean_47 = torch.ops.aten.var_mean.correction(clone_165, [2], correction = 0, keepdim = True)
    getitem_188: "f32[8, 196, 1]" = var_mean_47[0]
    getitem_189: "f32[8, 196, 1]" = var_mean_47[1];  var_mean_47 = None
    add_165: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-06);  getitem_188 = None
    rsqrt_47: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_47: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_165, getitem_189);  clone_165 = getitem_189 = None
    mul_188: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    mul_189: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_188, primals_285)
    add_166: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_189, primals_286);  mul_189 = primals_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_189: "f32[1568, 384]" = torch.ops.aten.view.default(add_166, [1568, 384]);  add_166 = None
    permute_143: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_287, [1, 0]);  primals_287 = None
    addmm_70: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_288, view_189, permute_143);  primals_288 = None
    view_190: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_70, [8, 196, 1536]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_47 = torch.ops.aten.split.Tensor(view_190, 768, -1);  view_190 = None
    getitem_190: "f32[8, 196, 768]" = split_47[0]
    getitem_191: "f32[8, 196, 768]" = split_47[1];  split_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_47: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_191)
    mul_190: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_191, sigmoid_47);  sigmoid_47 = None
    mul_191: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_190, mul_190);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_166: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_191);  mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_191: "f32[1568, 768]" = torch.ops.aten.view.default(clone_166, [1568, 768]);  clone_166 = None
    permute_144: "f32[768, 384]" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
    addmm_71: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_290, view_191, permute_144);  primals_290 = None
    view_192: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_71, [8, 196, 384]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_167: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_192);  view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_167: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_164, clone_167);  add_164 = clone_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    clone_168: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format);  add_167 = None
    var_mean_48 = torch.ops.aten.var_mean.correction(clone_168, [2], correction = 0, keepdim = True)
    getitem_192: "f32[8, 196, 1]" = var_mean_48[0]
    getitem_193: "f32[8, 196, 1]" = var_mean_48[1];  var_mean_48 = None
    add_168: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
    rsqrt_48: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_48: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_168, getitem_193);  clone_168 = getitem_193 = None
    mul_192: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    mul_193: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_192, primals_291)
    add_169: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_193, primals_292);  mul_193 = primals_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    mean: "f32[8, 384]" = torch.ops.aten.mean.dim(add_169, [1]);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:272, code: x = self.head_drop(x)
    clone_169: "f32[8, 384]" = torch.ops.aten.clone.default(mean);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    permute_145: "f32[384, 1000]" = torch.ops.aten.permute.default(primals_293, [1, 0]);  primals_293 = None
    addmm_72: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_294, clone_169, permute_145);  primals_294 = None
    permute_146: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    div_1: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 384);  rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_150: "f32[384, 768]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_155: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_2: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 384);  rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_160: "f32[196, 192]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_167: "f32[384, 196]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_3: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 384);  rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_170: "f32[384, 768]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_175: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_4: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 384);  rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_180: "f32[196, 192]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_187: "f32[384, 196]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_5: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 384);  rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_190: "f32[384, 768]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_195: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_6: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 384);  rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_200: "f32[196, 192]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_207: "f32[384, 196]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_7: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 384);  rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_210: "f32[384, 768]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_215: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_8: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 384);  rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_220: "f32[196, 192]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_227: "f32[384, 196]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_9: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 384);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_230: "f32[384, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_235: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_10: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 384);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_240: "f32[196, 192]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_247: "f32[384, 196]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_11: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 384);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_250: "f32[384, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_255: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_12: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 384);  rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_260: "f32[196, 192]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_267: "f32[384, 196]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_13: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 384);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_270: "f32[384, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_275: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_14: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 384);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_280: "f32[196, 192]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_287: "f32[384, 196]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_15: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 384);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_290: "f32[384, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_295: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_16: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 384);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_300: "f32[196, 192]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_307: "f32[384, 196]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_17: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 384);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_310: "f32[384, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_315: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_18: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 384);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_320: "f32[196, 192]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_327: "f32[384, 196]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_19: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 384);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_330: "f32[384, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_335: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_20: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 384);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_340: "f32[196, 192]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_347: "f32[384, 196]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_21: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 384);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_350: "f32[384, 768]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_355: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_22: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 384);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_360: "f32[196, 192]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_367: "f32[384, 196]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_23: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 384);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_370: "f32[384, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_375: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_24: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 384);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_380: "f32[196, 192]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_387: "f32[384, 196]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_25: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 384);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_390: "f32[384, 768]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_395: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_26: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 384);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_400: "f32[196, 192]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_407: "f32[384, 196]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_27: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 384);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_410: "f32[384, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_415: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_28: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 384);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_420: "f32[196, 192]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_427: "f32[384, 196]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_29: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 384);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_430: "f32[384, 768]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_435: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_30: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 384);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_440: "f32[196, 192]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_447: "f32[384, 196]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_31: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 384);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_450: "f32[384, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_455: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_32: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 384);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_460: "f32[196, 192]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_467: "f32[384, 196]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_33: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 384);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_470: "f32[384, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_475: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_34: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 384);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_480: "f32[196, 192]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_487: "f32[384, 196]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_35: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 384);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_490: "f32[384, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_495: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_36: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 384);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_500: "f32[196, 192]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_507: "f32[384, 196]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_37: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 384);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_510: "f32[384, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_515: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_38: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 384);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_520: "f32[196, 192]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_527: "f32[384, 196]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_39: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 384);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_530: "f32[384, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_535: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_40: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 384);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_540: "f32[196, 192]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_547: "f32[384, 196]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_41: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 384);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_550: "f32[384, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_555: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_42: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 384);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_560: "f32[196, 192]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_567: "f32[384, 196]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_43: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 384);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_570: "f32[384, 768]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_575: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_44: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 384);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_580: "f32[196, 192]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_587: "f32[384, 196]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_45: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 384);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_590: "f32[384, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_595: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_46: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 384);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_600: "f32[196, 192]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_607: "f32[384, 196]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_47: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 384);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_610: "f32[384, 768]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_615: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    div_48: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 384);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    permute_620: "f32[196, 192]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_627: "f32[384, 196]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    div_49: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt, 384);  rsqrt = None
    return [addmm_72, primals_1, primals_3, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_147, primals_153, primals_159, primals_165, primals_171, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_231, primals_237, primals_243, primals_249, primals_255, primals_261, primals_267, primals_273, primals_279, primals_285, primals_291, primals_295, mul, view_1, getitem_2, getitem_3, view_3, mul_4, view_5, getitem_6, getitem_7, view_7, mul_8, view_9, getitem_10, getitem_11, view_11, mul_12, view_13, getitem_14, getitem_15, view_15, mul_16, view_17, getitem_18, getitem_19, view_19, mul_20, view_21, getitem_22, getitem_23, view_23, mul_24, view_25, getitem_26, getitem_27, view_27, mul_28, view_29, getitem_30, getitem_31, view_31, mul_32, view_33, getitem_34, getitem_35, view_35, mul_36, view_37, getitem_38, getitem_39, view_39, mul_40, view_41, getitem_42, getitem_43, view_43, mul_44, view_45, getitem_46, getitem_47, view_47, mul_48, view_49, getitem_50, getitem_51, view_51, mul_52, view_53, getitem_54, getitem_55, view_55, mul_56, view_57, getitem_58, getitem_59, view_59, mul_60, view_61, getitem_62, getitem_63, view_63, mul_64, view_65, getitem_66, getitem_67, view_67, mul_68, view_69, getitem_70, getitem_71, view_71, mul_72, view_73, getitem_74, getitem_75, view_75, mul_76, view_77, getitem_78, getitem_79, view_79, mul_80, view_81, getitem_82, getitem_83, view_83, mul_84, view_85, getitem_86, getitem_87, view_87, mul_88, view_89, getitem_90, getitem_91, view_91, mul_92, view_93, getitem_94, getitem_95, view_95, mul_96, view_97, getitem_98, getitem_99, view_99, mul_100, view_101, getitem_102, getitem_103, view_103, mul_104, view_105, getitem_106, getitem_107, view_107, mul_108, view_109, getitem_110, getitem_111, view_111, mul_112, view_113, getitem_114, getitem_115, view_115, mul_116, view_117, getitem_118, getitem_119, view_119, mul_120, view_121, getitem_122, getitem_123, view_123, mul_124, view_125, getitem_126, getitem_127, view_127, mul_128, view_129, getitem_130, getitem_131, view_131, mul_132, view_133, getitem_134, getitem_135, view_135, mul_136, view_137, getitem_138, getitem_139, view_139, mul_140, view_141, getitem_142, getitem_143, view_143, mul_144, view_145, getitem_146, getitem_147, view_147, mul_148, view_149, getitem_150, getitem_151, view_151, mul_152, view_153, getitem_154, getitem_155, view_155, mul_156, view_157, getitem_158, getitem_159, view_159, mul_160, view_161, getitem_162, getitem_163, view_163, mul_164, view_165, getitem_166, getitem_167, view_167, mul_168, view_169, getitem_170, getitem_171, view_171, mul_172, view_173, getitem_174, getitem_175, view_175, mul_176, view_177, getitem_178, getitem_179, view_179, mul_180, view_181, getitem_182, getitem_183, view_183, mul_184, view_185, getitem_186, getitem_187, view_187, mul_188, view_189, getitem_190, getitem_191, view_191, mul_192, clone_169, permute_146, div_1, permute_150, permute_155, div_2, permute_160, permute_167, div_3, permute_170, permute_175, div_4, permute_180, permute_187, div_5, permute_190, permute_195, div_6, permute_200, permute_207, div_7, permute_210, permute_215, div_8, permute_220, permute_227, div_9, permute_230, permute_235, div_10, permute_240, permute_247, div_11, permute_250, permute_255, div_12, permute_260, permute_267, div_13, permute_270, permute_275, div_14, permute_280, permute_287, div_15, permute_290, permute_295, div_16, permute_300, permute_307, div_17, permute_310, permute_315, div_18, permute_320, permute_327, div_19, permute_330, permute_335, div_20, permute_340, permute_347, div_21, permute_350, permute_355, div_22, permute_360, permute_367, div_23, permute_370, permute_375, div_24, permute_380, permute_387, div_25, permute_390, permute_395, div_26, permute_400, permute_407, div_27, permute_410, permute_415, div_28, permute_420, permute_427, div_29, permute_430, permute_435, div_30, permute_440, permute_447, div_31, permute_450, permute_455, div_32, permute_460, permute_467, div_33, permute_470, permute_475, div_34, permute_480, permute_487, div_35, permute_490, permute_495, div_36, permute_500, permute_507, div_37, permute_510, permute_515, div_38, permute_520, permute_527, div_39, permute_530, permute_535, div_40, permute_540, permute_547, div_41, permute_550, permute_555, div_42, permute_560, permute_567, div_43, permute_570, permute_575, div_44, permute_580, permute_587, div_45, permute_590, permute_595, div_46, permute_600, permute_607, div_47, permute_610, permute_615, div_48, permute_620, permute_627, div_49]
    