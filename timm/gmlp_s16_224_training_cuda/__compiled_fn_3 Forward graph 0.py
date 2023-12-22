from __future__ import annotations



def forward(self, primals_1: "f32[256, 3, 16, 16]", primals_2: "f32[256]", primals_3: "f32[256]", primals_4: "f32[256]", primals_5: "f32[1536, 256]", primals_6: "f32[1536]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[196, 196]", primals_10: "f32[196]", primals_11: "f32[256, 768]", primals_12: "f32[256]", primals_13: "f32[256]", primals_14: "f32[256]", primals_15: "f32[1536, 256]", primals_16: "f32[1536]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[196, 196]", primals_20: "f32[196]", primals_21: "f32[256, 768]", primals_22: "f32[256]", primals_23: "f32[256]", primals_24: "f32[256]", primals_25: "f32[1536, 256]", primals_26: "f32[1536]", primals_27: "f32[768]", primals_28: "f32[768]", primals_29: "f32[196, 196]", primals_30: "f32[196]", primals_31: "f32[256, 768]", primals_32: "f32[256]", primals_33: "f32[256]", primals_34: "f32[256]", primals_35: "f32[1536, 256]", primals_36: "f32[1536]", primals_37: "f32[768]", primals_38: "f32[768]", primals_39: "f32[196, 196]", primals_40: "f32[196]", primals_41: "f32[256, 768]", primals_42: "f32[256]", primals_43: "f32[256]", primals_44: "f32[256]", primals_45: "f32[1536, 256]", primals_46: "f32[1536]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[196, 196]", primals_50: "f32[196]", primals_51: "f32[256, 768]", primals_52: "f32[256]", primals_53: "f32[256]", primals_54: "f32[256]", primals_55: "f32[1536, 256]", primals_56: "f32[1536]", primals_57: "f32[768]", primals_58: "f32[768]", primals_59: "f32[196, 196]", primals_60: "f32[196]", primals_61: "f32[256, 768]", primals_62: "f32[256]", primals_63: "f32[256]", primals_64: "f32[256]", primals_65: "f32[1536, 256]", primals_66: "f32[1536]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[196, 196]", primals_70: "f32[196]", primals_71: "f32[256, 768]", primals_72: "f32[256]", primals_73: "f32[256]", primals_74: "f32[256]", primals_75: "f32[1536, 256]", primals_76: "f32[1536]", primals_77: "f32[768]", primals_78: "f32[768]", primals_79: "f32[196, 196]", primals_80: "f32[196]", primals_81: "f32[256, 768]", primals_82: "f32[256]", primals_83: "f32[256]", primals_84: "f32[256]", primals_85: "f32[1536, 256]", primals_86: "f32[1536]", primals_87: "f32[768]", primals_88: "f32[768]", primals_89: "f32[196, 196]", primals_90: "f32[196]", primals_91: "f32[256, 768]", primals_92: "f32[256]", primals_93: "f32[256]", primals_94: "f32[256]", primals_95: "f32[1536, 256]", primals_96: "f32[1536]", primals_97: "f32[768]", primals_98: "f32[768]", primals_99: "f32[196, 196]", primals_100: "f32[196]", primals_101: "f32[256, 768]", primals_102: "f32[256]", primals_103: "f32[256]", primals_104: "f32[256]", primals_105: "f32[1536, 256]", primals_106: "f32[1536]", primals_107: "f32[768]", primals_108: "f32[768]", primals_109: "f32[196, 196]", primals_110: "f32[196]", primals_111: "f32[256, 768]", primals_112: "f32[256]", primals_113: "f32[256]", primals_114: "f32[256]", primals_115: "f32[1536, 256]", primals_116: "f32[1536]", primals_117: "f32[768]", primals_118: "f32[768]", primals_119: "f32[196, 196]", primals_120: "f32[196]", primals_121: "f32[256, 768]", primals_122: "f32[256]", primals_123: "f32[256]", primals_124: "f32[256]", primals_125: "f32[1536, 256]", primals_126: "f32[1536]", primals_127: "f32[768]", primals_128: "f32[768]", primals_129: "f32[196, 196]", primals_130: "f32[196]", primals_131: "f32[256, 768]", primals_132: "f32[256]", primals_133: "f32[256]", primals_134: "f32[256]", primals_135: "f32[1536, 256]", primals_136: "f32[1536]", primals_137: "f32[768]", primals_138: "f32[768]", primals_139: "f32[196, 196]", primals_140: "f32[196]", primals_141: "f32[256, 768]", primals_142: "f32[256]", primals_143: "f32[256]", primals_144: "f32[256]", primals_145: "f32[1536, 256]", primals_146: "f32[1536]", primals_147: "f32[768]", primals_148: "f32[768]", primals_149: "f32[196, 196]", primals_150: "f32[196]", primals_151: "f32[256, 768]", primals_152: "f32[256]", primals_153: "f32[256]", primals_154: "f32[256]", primals_155: "f32[1536, 256]", primals_156: "f32[1536]", primals_157: "f32[768]", primals_158: "f32[768]", primals_159: "f32[196, 196]", primals_160: "f32[196]", primals_161: "f32[256, 768]", primals_162: "f32[256]", primals_163: "f32[256]", primals_164: "f32[256]", primals_165: "f32[1536, 256]", primals_166: "f32[1536]", primals_167: "f32[768]", primals_168: "f32[768]", primals_169: "f32[196, 196]", primals_170: "f32[196]", primals_171: "f32[256, 768]", primals_172: "f32[256]", primals_173: "f32[256]", primals_174: "f32[256]", primals_175: "f32[1536, 256]", primals_176: "f32[1536]", primals_177: "f32[768]", primals_178: "f32[768]", primals_179: "f32[196, 196]", primals_180: "f32[196]", primals_181: "f32[256, 768]", primals_182: "f32[256]", primals_183: "f32[256]", primals_184: "f32[256]", primals_185: "f32[1536, 256]", primals_186: "f32[1536]", primals_187: "f32[768]", primals_188: "f32[768]", primals_189: "f32[196, 196]", primals_190: "f32[196]", primals_191: "f32[256, 768]", primals_192: "f32[256]", primals_193: "f32[256]", primals_194: "f32[256]", primals_195: "f32[1536, 256]", primals_196: "f32[1536]", primals_197: "f32[768]", primals_198: "f32[768]", primals_199: "f32[196, 196]", primals_200: "f32[196]", primals_201: "f32[256, 768]", primals_202: "f32[256]", primals_203: "f32[256]", primals_204: "f32[256]", primals_205: "f32[1536, 256]", primals_206: "f32[1536]", primals_207: "f32[768]", primals_208: "f32[768]", primals_209: "f32[196, 196]", primals_210: "f32[196]", primals_211: "f32[256, 768]", primals_212: "f32[256]", primals_213: "f32[256]", primals_214: "f32[256]", primals_215: "f32[1536, 256]", primals_216: "f32[1536]", primals_217: "f32[768]", primals_218: "f32[768]", primals_219: "f32[196, 196]", primals_220: "f32[196]", primals_221: "f32[256, 768]", primals_222: "f32[256]", primals_223: "f32[256]", primals_224: "f32[256]", primals_225: "f32[1536, 256]", primals_226: "f32[1536]", primals_227: "f32[768]", primals_228: "f32[768]", primals_229: "f32[196, 196]", primals_230: "f32[196]", primals_231: "f32[256, 768]", primals_232: "f32[256]", primals_233: "f32[256]", primals_234: "f32[256]", primals_235: "f32[1536, 256]", primals_236: "f32[1536]", primals_237: "f32[768]", primals_238: "f32[768]", primals_239: "f32[196, 196]", primals_240: "f32[196]", primals_241: "f32[256, 768]", primals_242: "f32[256]", primals_243: "f32[256]", primals_244: "f32[256]", primals_245: "f32[1536, 256]", primals_246: "f32[1536]", primals_247: "f32[768]", primals_248: "f32[768]", primals_249: "f32[196, 196]", primals_250: "f32[196]", primals_251: "f32[256, 768]", primals_252: "f32[256]", primals_253: "f32[256]", primals_254: "f32[256]", primals_255: "f32[1536, 256]", primals_256: "f32[1536]", primals_257: "f32[768]", primals_258: "f32[768]", primals_259: "f32[196, 196]", primals_260: "f32[196]", primals_261: "f32[256, 768]", primals_262: "f32[256]", primals_263: "f32[256]", primals_264: "f32[256]", primals_265: "f32[1536, 256]", primals_266: "f32[1536]", primals_267: "f32[768]", primals_268: "f32[768]", primals_269: "f32[196, 196]", primals_270: "f32[196]", primals_271: "f32[256, 768]", primals_272: "f32[256]", primals_273: "f32[256]", primals_274: "f32[256]", primals_275: "f32[1536, 256]", primals_276: "f32[1536]", primals_277: "f32[768]", primals_278: "f32[768]", primals_279: "f32[196, 196]", primals_280: "f32[196]", primals_281: "f32[256, 768]", primals_282: "f32[256]", primals_283: "f32[256]", primals_284: "f32[256]", primals_285: "f32[1536, 256]", primals_286: "f32[1536]", primals_287: "f32[768]", primals_288: "f32[768]", primals_289: "f32[196, 196]", primals_290: "f32[196]", primals_291: "f32[256, 768]", primals_292: "f32[256]", primals_293: "f32[256]", primals_294: "f32[256]", primals_295: "f32[1536, 256]", primals_296: "f32[1536]", primals_297: "f32[768]", primals_298: "f32[768]", primals_299: "f32[196, 196]", primals_300: "f32[196]", primals_301: "f32[256, 768]", primals_302: "f32[256]", primals_303: "f32[256]", primals_304: "f32[256]", primals_305: "f32[1000, 256]", primals_306: "f32[1000]", primals_307: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(primals_307, primals_1, primals_2, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 256, 196]" = torch.ops.aten.view.default(convolution, [8, 256, 196]);  convolution = None
    permute: "f32[8, 196, 256]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone: "f32[8, 196, 256]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 196, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul, primals_3)
    add_1: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_1: "f32[1568, 256]" = torch.ops.aten.view.default(add_1, [1568, 256]);  add_1 = None
    permute_1: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_6, view_1, permute_1);  primals_6 = None
    view_2: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_2: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_2, 0.5)
    mul_3: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_2, 0.7071067811865476);  view_2 = None
    erf: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
    add_2: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_4: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_2, add_2);  mul_2 = add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_1: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_4);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split = torch.ops.aten.split.Tensor(clone_1, 768, -1);  clone_1 = None
    getitem_2: "f32[8, 196, 768]" = split[0]
    getitem_3: "f32[8, 196, 768]" = split[1];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_2: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_3, memory_format = torch.contiguous_format);  getitem_3 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_2, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_5: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_3: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_1: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_2, getitem_5);  clone_2 = getitem_5 = None
    mul_5: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_6: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_5, primals_7)
    add_4: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_6, primals_8);  mul_6 = primals_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_2: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_4, [0, 2, 1]);  add_4 = None
    permute_3: "f32[196, 196]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
    clone_3: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    view_3: "f32[6144, 196]" = torch.ops.aten.view.default(clone_3, [6144, 196]);  clone_3 = None
    mm: "f32[6144, 196]" = torch.ops.aten.mm.default(view_3, permute_3)
    view_4: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm, [8, 768, 196])
    add_5: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_4, primals_10);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_4: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_5, [0, 2, 1]);  add_5 = None
    mul_7: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_2, permute_4);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_5: "f32[1568, 768]" = torch.ops.aten.view.default(mul_7, [1568, 768]);  mul_7 = None
    permute_5: "f32[768, 256]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm_1: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_12, view_5, permute_5);  primals_12 = None
    view_6: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_1, [8, 196, 256]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_4: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_6);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_6: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(permute, clone_4);  permute = clone_4 = None
    clone_5: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_6, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_5, [2], correction = 0, keepdim = True)
    getitem_6: "f32[8, 196, 1]" = var_mean_2[0]
    getitem_7: "f32[8, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    add_7: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_2: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_2: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_5, getitem_7);  clone_5 = getitem_7 = None
    mul_8: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_9: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_8, primals_13)
    add_8: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_9, primals_14);  mul_9 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_7: "f32[1568, 256]" = torch.ops.aten.view.default(add_8, [1568, 256]);  add_8 = None
    permute_6: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_2: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_16, view_7, permute_6);  primals_16 = None
    view_8: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_2, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_10: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_8, 0.5)
    mul_11: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476);  view_8 = None
    erf_1: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
    add_9: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_12: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_10, add_9);  mul_10 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_6: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_12);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_1 = torch.ops.aten.split.Tensor(clone_6, 768, -1);  clone_6 = None
    getitem_8: "f32[8, 196, 768]" = split_1[0]
    getitem_9: "f32[8, 196, 768]" = split_1[1];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_7: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_9, memory_format = torch.contiguous_format);  getitem_9 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_7, [2], correction = 0, keepdim = True)
    getitem_10: "f32[8, 196, 1]" = var_mean_3[0]
    getitem_11: "f32[8, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    add_10: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_3: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_3: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_7, getitem_11);  clone_7 = getitem_11 = None
    mul_13: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_14: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_13, primals_17)
    add_11: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_14, primals_18);  mul_14 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_7: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_11, [0, 2, 1]);  add_11 = None
    permute_8: "f32[196, 196]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    clone_8: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_9: "f32[6144, 196]" = torch.ops.aten.view.default(clone_8, [6144, 196]);  clone_8 = None
    mm_1: "f32[6144, 196]" = torch.ops.aten.mm.default(view_9, permute_8)
    view_10: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_1, [8, 768, 196])
    add_12: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_10, primals_20);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_9: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_12, [0, 2, 1]);  add_12 = None
    mul_15: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_8, permute_9);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_11: "f32[1568, 768]" = torch.ops.aten.view.default(mul_15, [1568, 768]);  mul_15 = None
    permute_10: "f32[768, 256]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
    addmm_3: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_22, view_11, permute_10);  primals_22 = None
    view_12: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_3, [8, 196, 256]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_9: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_13: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_6, clone_9);  add_6 = clone_9 = None
    clone_10: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_13, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_10, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 196, 1]" = var_mean_4[0]
    getitem_13: "f32[8, 196, 1]" = var_mean_4[1];  var_mean_4 = None
    add_14: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_4: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_4: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_10, getitem_13);  clone_10 = getitem_13 = None
    mul_16: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_17: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_16, primals_23)
    add_15: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_17, primals_24);  mul_17 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_13: "f32[1568, 256]" = torch.ops.aten.view.default(add_15, [1568, 256]);  add_15 = None
    permute_11: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_4: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_26, view_13, permute_11);  primals_26 = None
    view_14: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_4, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_18: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_14, 0.5)
    mul_19: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476);  view_14 = None
    erf_2: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_16: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_20: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_18, add_16);  mul_18 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_11: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_2 = torch.ops.aten.split.Tensor(clone_11, 768, -1);  clone_11 = None
    getitem_14: "f32[8, 196, 768]" = split_2[0]
    getitem_15: "f32[8, 196, 768]" = split_2[1];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_12: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_15, memory_format = torch.contiguous_format);  getitem_15 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_12, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 196, 1]" = var_mean_5[0]
    getitem_17: "f32[8, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    add_17: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_5: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_5: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_12, getitem_17);  clone_12 = getitem_17 = None
    mul_21: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_22: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_27)
    add_18: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_22, primals_28);  mul_22 = primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_12: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_18, [0, 2, 1]);  add_18 = None
    permute_13: "f32[196, 196]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
    clone_13: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    view_15: "f32[6144, 196]" = torch.ops.aten.view.default(clone_13, [6144, 196]);  clone_13 = None
    mm_2: "f32[6144, 196]" = torch.ops.aten.mm.default(view_15, permute_13)
    view_16: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_2, [8, 768, 196])
    add_19: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_16, primals_30);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_14: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_19, [0, 2, 1]);  add_19 = None
    mul_23: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_14, permute_14);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_17: "f32[1568, 768]" = torch.ops.aten.view.default(mul_23, [1568, 768]);  mul_23 = None
    permute_15: "f32[768, 256]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    addmm_5: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_32, view_17, permute_15);  primals_32 = None
    view_18: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_5, [8, 196, 256]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_14: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_20: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_13, clone_14);  add_13 = clone_14 = None
    clone_15: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_15, [2], correction = 0, keepdim = True)
    getitem_18: "f32[8, 196, 1]" = var_mean_6[0]
    getitem_19: "f32[8, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_6: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_6: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_15, getitem_19);  clone_15 = getitem_19 = None
    mul_24: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_25: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_24, primals_33)
    add_22: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_25, primals_34);  mul_25 = primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_19: "f32[1568, 256]" = torch.ops.aten.view.default(add_22, [1568, 256]);  add_22 = None
    permute_16: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
    addmm_6: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_36, view_19, permute_16);  primals_36 = None
    view_20: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_6, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_26: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_20, 0.5)
    mul_27: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_20, 0.7071067811865476);  view_20 = None
    erf_3: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_23: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_28: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_26, add_23);  mul_26 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_16: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_28);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_3 = torch.ops.aten.split.Tensor(clone_16, 768, -1);  clone_16 = None
    getitem_20: "f32[8, 196, 768]" = split_3[0]
    getitem_21: "f32[8, 196, 768]" = split_3[1];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_17: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_21, memory_format = torch.contiguous_format);  getitem_21 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_17, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 196, 1]" = var_mean_7[0]
    getitem_23: "f32[8, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    add_24: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_7: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_7: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_17, getitem_23);  clone_17 = getitem_23 = None
    mul_29: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_30: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_29, primals_37)
    add_25: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_30, primals_38);  mul_30 = primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_17: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_25, [0, 2, 1]);  add_25 = None
    permute_18: "f32[196, 196]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
    clone_18: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_21: "f32[6144, 196]" = torch.ops.aten.view.default(clone_18, [6144, 196]);  clone_18 = None
    mm_3: "f32[6144, 196]" = torch.ops.aten.mm.default(view_21, permute_18)
    view_22: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_3, [8, 768, 196])
    add_26: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_22, primals_40);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_19: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_26, [0, 2, 1]);  add_26 = None
    mul_31: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_20, permute_19);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_23: "f32[1568, 768]" = torch.ops.aten.view.default(mul_31, [1568, 768]);  mul_31 = None
    permute_20: "f32[768, 256]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
    addmm_7: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_42, view_23, permute_20);  primals_42 = None
    view_24: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_7, [8, 196, 256]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_19: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_24);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_27: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_20, clone_19);  add_20 = clone_19 = None
    clone_20: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_27, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_20, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 196, 1]" = var_mean_8[0]
    getitem_25: "f32[8, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_8: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_8: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_20, getitem_25);  clone_20 = getitem_25 = None
    mul_32: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_33: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_32, primals_43)
    add_29: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_33, primals_44);  mul_33 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_25: "f32[1568, 256]" = torch.ops.aten.view.default(add_29, [1568, 256]);  add_29 = None
    permute_21: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    addmm_8: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_46, view_25, permute_21);  primals_46 = None
    view_26: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_8, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_34: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_26, 0.5)
    mul_35: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_26, 0.7071067811865476);  view_26 = None
    erf_4: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
    add_30: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_36: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_34, add_30);  mul_34 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_21: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_36);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_4 = torch.ops.aten.split.Tensor(clone_21, 768, -1);  clone_21 = None
    getitem_26: "f32[8, 196, 768]" = split_4[0]
    getitem_27: "f32[8, 196, 768]" = split_4[1];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_22: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_27, memory_format = torch.contiguous_format);  getitem_27 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_22, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 196, 1]" = var_mean_9[0]
    getitem_29: "f32[8, 196, 1]" = var_mean_9[1];  var_mean_9 = None
    add_31: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_9: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_9: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_22, getitem_29);  clone_22 = getitem_29 = None
    mul_37: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_38: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_47)
    add_32: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_48);  mul_38 = primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_22: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_32, [0, 2, 1]);  add_32 = None
    permute_23: "f32[196, 196]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    clone_23: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
    view_27: "f32[6144, 196]" = torch.ops.aten.view.default(clone_23, [6144, 196]);  clone_23 = None
    mm_4: "f32[6144, 196]" = torch.ops.aten.mm.default(view_27, permute_23)
    view_28: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_4, [8, 768, 196])
    add_33: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_28, primals_50);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_24: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_33, [0, 2, 1]);  add_33 = None
    mul_39: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_26, permute_24);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_29: "f32[1568, 768]" = torch.ops.aten.view.default(mul_39, [1568, 768]);  mul_39 = None
    permute_25: "f32[768, 256]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    addmm_9: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_52, view_29, permute_25);  primals_52 = None
    view_30: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_9, [8, 196, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_24: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_30);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_34: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_27, clone_24);  add_27 = clone_24 = None
    clone_25: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_34, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_25, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_31: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_10: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_25, getitem_31);  clone_25 = getitem_31 = None
    mul_40: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_41: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_40, primals_53)
    add_36: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_41, primals_54);  mul_41 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_31: "f32[1568, 256]" = torch.ops.aten.view.default(add_36, [1568, 256]);  add_36 = None
    permute_26: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_10: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_56, view_31, permute_26);  primals_56 = None
    view_32: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_10, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_42: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_32, 0.5)
    mul_43: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_32, 0.7071067811865476);  view_32 = None
    erf_5: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
    add_37: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_44: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_42, add_37);  mul_42 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_26: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_44);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_5 = torch.ops.aten.split.Tensor(clone_26, 768, -1);  clone_26 = None
    getitem_32: "f32[8, 196, 768]" = split_5[0]
    getitem_33: "f32[8, 196, 768]" = split_5[1];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_27: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_33, memory_format = torch.contiguous_format);  getitem_33 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_27, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 196, 1]" = var_mean_11[0]
    getitem_35: "f32[8, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_11: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_11: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_27, getitem_35);  clone_27 = getitem_35 = None
    mul_45: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_46: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_45, primals_57)
    add_39: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_46, primals_58);  mul_46 = primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_27: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_39, [0, 2, 1]);  add_39 = None
    permute_28: "f32[196, 196]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    clone_28: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    view_33: "f32[6144, 196]" = torch.ops.aten.view.default(clone_28, [6144, 196]);  clone_28 = None
    mm_5: "f32[6144, 196]" = torch.ops.aten.mm.default(view_33, permute_28)
    view_34: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_5, [8, 768, 196])
    add_40: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_34, primals_60);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_29: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_40, [0, 2, 1]);  add_40 = None
    mul_47: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_32, permute_29);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_35: "f32[1568, 768]" = torch.ops.aten.view.default(mul_47, [1568, 768]);  mul_47 = None
    permute_30: "f32[768, 256]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    addmm_11: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_62, view_35, permute_30);  primals_62 = None
    view_36: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_11, [8, 196, 256]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_29: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_41: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_34, clone_29);  add_34 = clone_29 = None
    clone_30: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_30, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 196, 1]" = var_mean_12[0]
    getitem_37: "f32[8, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_12: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_12: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_30, getitem_37);  clone_30 = getitem_37 = None
    mul_48: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_49: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_48, primals_63)
    add_43: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_49, primals_64);  mul_49 = primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_37: "f32[1568, 256]" = torch.ops.aten.view.default(add_43, [1568, 256]);  add_43 = None
    permute_31: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    addmm_12: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_66, view_37, permute_31);  primals_66 = None
    view_38: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_12, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_50: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_51: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_6: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_44: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_52: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_50, add_44);  mul_50 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_31: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_6 = torch.ops.aten.split.Tensor(clone_31, 768, -1);  clone_31 = None
    getitem_38: "f32[8, 196, 768]" = split_6[0]
    getitem_39: "f32[8, 196, 768]" = split_6[1];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_32: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_39, memory_format = torch.contiguous_format);  getitem_39 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_32, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 196, 1]" = var_mean_13[0]
    getitem_41: "f32[8, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    add_45: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_13: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_13: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_32, getitem_41);  clone_32 = getitem_41 = None
    mul_53: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_54: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_53, primals_67)
    add_46: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_54, primals_68);  mul_54 = primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_32: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_46, [0, 2, 1]);  add_46 = None
    permute_33: "f32[196, 196]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    clone_33: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    view_39: "f32[6144, 196]" = torch.ops.aten.view.default(clone_33, [6144, 196]);  clone_33 = None
    mm_6: "f32[6144, 196]" = torch.ops.aten.mm.default(view_39, permute_33)
    view_40: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_6, [8, 768, 196])
    add_47: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_40, primals_70);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_34: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_47, [0, 2, 1]);  add_47 = None
    mul_55: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_38, permute_34);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_41: "f32[1568, 768]" = torch.ops.aten.view.default(mul_55, [1568, 768]);  mul_55 = None
    permute_35: "f32[768, 256]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_13: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_72, view_41, permute_35);  primals_72 = None
    view_42: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_13, [8, 196, 256]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_34: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_42);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_48: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_41, clone_34);  add_41 = clone_34 = None
    clone_35: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_48, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_35, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 196, 1]" = var_mean_14[0]
    getitem_43: "f32[8, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_14: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_14: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_35, getitem_43);  clone_35 = getitem_43 = None
    mul_56: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_57: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_56, primals_73)
    add_50: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_57, primals_74);  mul_57 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_43: "f32[1568, 256]" = torch.ops.aten.view.default(add_50, [1568, 256]);  add_50 = None
    permute_36: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    addmm_14: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_76, view_43, permute_36);  primals_76 = None
    view_44: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_14, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_58: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_44, 0.5)
    mul_59: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_44, 0.7071067811865476);  view_44 = None
    erf_7: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_59);  mul_59 = None
    add_51: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_60: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_58, add_51);  mul_58 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_36: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_60);  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_7 = torch.ops.aten.split.Tensor(clone_36, 768, -1);  clone_36 = None
    getitem_44: "f32[8, 196, 768]" = split_7[0]
    getitem_45: "f32[8, 196, 768]" = split_7[1];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_37: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_45, memory_format = torch.contiguous_format);  getitem_45 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_37, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 196, 1]" = var_mean_15[0]
    getitem_47: "f32[8, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    add_52: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_15: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_15: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_37, getitem_47);  clone_37 = getitem_47 = None
    mul_61: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_62: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_61, primals_77)
    add_53: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_62, primals_78);  mul_62 = primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_37: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_53, [0, 2, 1]);  add_53 = None
    permute_38: "f32[196, 196]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    clone_38: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    view_45: "f32[6144, 196]" = torch.ops.aten.view.default(clone_38, [6144, 196]);  clone_38 = None
    mm_7: "f32[6144, 196]" = torch.ops.aten.mm.default(view_45, permute_38)
    view_46: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_7, [8, 768, 196])
    add_54: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_46, primals_80);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_39: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_54, [0, 2, 1]);  add_54 = None
    mul_63: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_44, permute_39);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_47: "f32[1568, 768]" = torch.ops.aten.view.default(mul_63, [1568, 768]);  mul_63 = None
    permute_40: "f32[768, 256]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_15: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_82, view_47, permute_40);  primals_82 = None
    view_48: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_15, [8, 196, 256]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_39: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_55: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_48, clone_39);  add_48 = clone_39 = None
    clone_40: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_55, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_40, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 196, 1]" = var_mean_16[0]
    getitem_49: "f32[8, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    add_56: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_16: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_16: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_40, getitem_49);  clone_40 = getitem_49 = None
    mul_64: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_65: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_64, primals_83)
    add_57: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_65, primals_84);  mul_65 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_49: "f32[1568, 256]" = torch.ops.aten.view.default(add_57, [1568, 256]);  add_57 = None
    permute_41: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_16: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_86, view_49, permute_41);  primals_86 = None
    view_50: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_16, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_66: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_50, 0.5)
    mul_67: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476);  view_50 = None
    erf_8: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_67);  mul_67 = None
    add_58: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_68: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_66, add_58);  mul_66 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_41: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_68);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_8 = torch.ops.aten.split.Tensor(clone_41, 768, -1);  clone_41 = None
    getitem_50: "f32[8, 196, 768]" = split_8[0]
    getitem_51: "f32[8, 196, 768]" = split_8[1];  split_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_42: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_51, memory_format = torch.contiguous_format);  getitem_51 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 196, 1]" = var_mean_17[0]
    getitem_53: "f32[8, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    add_59: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_17: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_17: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_42, getitem_53);  clone_42 = getitem_53 = None
    mul_69: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_70: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_69, primals_87)
    add_60: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_70, primals_88);  mul_70 = primals_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_42: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_60, [0, 2, 1]);  add_60 = None
    permute_43: "f32[196, 196]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    clone_43: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
    view_51: "f32[6144, 196]" = torch.ops.aten.view.default(clone_43, [6144, 196]);  clone_43 = None
    mm_8: "f32[6144, 196]" = torch.ops.aten.mm.default(view_51, permute_43)
    view_52: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_8, [8, 768, 196])
    add_61: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_52, primals_90);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_44: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_61, [0, 2, 1]);  add_61 = None
    mul_71: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_50, permute_44);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_53: "f32[1568, 768]" = torch.ops.aten.view.default(mul_71, [1568, 768]);  mul_71 = None
    permute_45: "f32[768, 256]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_17: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_92, view_53, permute_45);  primals_92 = None
    view_54: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_17, [8, 196, 256]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_44: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_54);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_62: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_55, clone_44);  add_55 = clone_44 = None
    clone_45: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_45, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 196, 1]" = var_mean_18[0]
    getitem_55: "f32[8, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    add_63: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_18: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_18: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_45, getitem_55);  clone_45 = getitem_55 = None
    mul_72: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_73: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_72, primals_93)
    add_64: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_73, primals_94);  mul_73 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_55: "f32[1568, 256]" = torch.ops.aten.view.default(add_64, [1568, 256]);  add_64 = None
    permute_46: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_18: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_96, view_55, permute_46);  primals_96 = None
    view_56: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_18, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_74: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_56, 0.5)
    mul_75: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_56, 0.7071067811865476);  view_56 = None
    erf_9: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_65: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_76: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_74, add_65);  mul_74 = add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_46: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_9 = torch.ops.aten.split.Tensor(clone_46, 768, -1);  clone_46 = None
    getitem_56: "f32[8, 196, 768]" = split_9[0]
    getitem_57: "f32[8, 196, 768]" = split_9[1];  split_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_47: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_57, memory_format = torch.contiguous_format);  getitem_57 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_47, [2], correction = 0, keepdim = True)
    getitem_58: "f32[8, 196, 1]" = var_mean_19[0]
    getitem_59: "f32[8, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    add_66: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_19: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_19: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_47, getitem_59);  clone_47 = getitem_59 = None
    mul_77: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_78: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_77, primals_97)
    add_67: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_78, primals_98);  mul_78 = primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_47: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_67, [0, 2, 1]);  add_67 = None
    permute_48: "f32[196, 196]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    clone_48: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    view_57: "f32[6144, 196]" = torch.ops.aten.view.default(clone_48, [6144, 196]);  clone_48 = None
    mm_9: "f32[6144, 196]" = torch.ops.aten.mm.default(view_57, permute_48)
    view_58: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_9, [8, 768, 196])
    add_68: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_58, primals_100);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_49: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_68, [0, 2, 1]);  add_68 = None
    mul_79: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_56, permute_49);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_59: "f32[1568, 768]" = torch.ops.aten.view.default(mul_79, [1568, 768]);  mul_79 = None
    permute_50: "f32[768, 256]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_19: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_102, view_59, permute_50);  primals_102 = None
    view_60: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_19, [8, 196, 256]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_49: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_69: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_62, clone_49);  add_62 = clone_49 = None
    clone_50: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_69, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_50, [2], correction = 0, keepdim = True)
    getitem_60: "f32[8, 196, 1]" = var_mean_20[0]
    getitem_61: "f32[8, 196, 1]" = var_mean_20[1];  var_mean_20 = None
    add_70: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
    rsqrt_20: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_20: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_50, getitem_61);  clone_50 = getitem_61 = None
    mul_80: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_81: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_80, primals_103)
    add_71: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_81, primals_104);  mul_81 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_61: "f32[1568, 256]" = torch.ops.aten.view.default(add_71, [1568, 256]);  add_71 = None
    permute_51: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_20: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_106, view_61, permute_51);  primals_106 = None
    view_62: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_20, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_82: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_62, 0.5)
    mul_83: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_62, 0.7071067811865476);  view_62 = None
    erf_10: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_72: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_84: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_82, add_72);  mul_82 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_51: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_84);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_10 = torch.ops.aten.split.Tensor(clone_51, 768, -1);  clone_51 = None
    getitem_62: "f32[8, 196, 768]" = split_10[0]
    getitem_63: "f32[8, 196, 768]" = split_10[1];  split_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_52: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_63, memory_format = torch.contiguous_format);  getitem_63 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_52, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 196, 1]" = var_mean_21[0]
    getitem_65: "f32[8, 196, 1]" = var_mean_21[1];  var_mean_21 = None
    add_73: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_21: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_21: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_52, getitem_65);  clone_52 = getitem_65 = None
    mul_85: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_86: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_85, primals_107)
    add_74: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_86, primals_108);  mul_86 = primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_52: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_74, [0, 2, 1]);  add_74 = None
    permute_53: "f32[196, 196]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    clone_53: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_63: "f32[6144, 196]" = torch.ops.aten.view.default(clone_53, [6144, 196]);  clone_53 = None
    mm_10: "f32[6144, 196]" = torch.ops.aten.mm.default(view_63, permute_53)
    view_64: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_10, [8, 768, 196])
    add_75: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_64, primals_110);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_54: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_75, [0, 2, 1]);  add_75 = None
    mul_87: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_62, permute_54);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_65: "f32[1568, 768]" = torch.ops.aten.view.default(mul_87, [1568, 768]);  mul_87 = None
    permute_55: "f32[768, 256]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_21: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_112, view_65, permute_55);  primals_112 = None
    view_66: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_21, [8, 196, 256]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_54: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_66);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_76: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_69, clone_54);  add_69 = clone_54 = None
    clone_55: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_55, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 196, 1]" = var_mean_22[0]
    getitem_67: "f32[8, 196, 1]" = var_mean_22[1];  var_mean_22 = None
    add_77: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_22: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_22: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_55, getitem_67);  clone_55 = getitem_67 = None
    mul_88: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_89: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_88, primals_113)
    add_78: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_89, primals_114);  mul_89 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_67: "f32[1568, 256]" = torch.ops.aten.view.default(add_78, [1568, 256]);  add_78 = None
    permute_56: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_22: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_116, view_67, permute_56);  primals_116 = None
    view_68: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_22, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_90: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_68, 0.5)
    mul_91: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476);  view_68 = None
    erf_11: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
    add_79: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_92: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_90, add_79);  mul_90 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_56: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_92);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_11 = torch.ops.aten.split.Tensor(clone_56, 768, -1);  clone_56 = None
    getitem_68: "f32[8, 196, 768]" = split_11[0]
    getitem_69: "f32[8, 196, 768]" = split_11[1];  split_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_57: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_69, memory_format = torch.contiguous_format);  getitem_69 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_57, [2], correction = 0, keepdim = True)
    getitem_70: "f32[8, 196, 1]" = var_mean_23[0]
    getitem_71: "f32[8, 196, 1]" = var_mean_23[1];  var_mean_23 = None
    add_80: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_23: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_23: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_57, getitem_71);  clone_57 = getitem_71 = None
    mul_93: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_94: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_93, primals_117)
    add_81: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_94, primals_118);  mul_94 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_57: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_81, [0, 2, 1]);  add_81 = None
    permute_58: "f32[196, 196]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    clone_58: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    view_69: "f32[6144, 196]" = torch.ops.aten.view.default(clone_58, [6144, 196]);  clone_58 = None
    mm_11: "f32[6144, 196]" = torch.ops.aten.mm.default(view_69, permute_58)
    view_70: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_11, [8, 768, 196])
    add_82: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_70, primals_120);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_59: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_82, [0, 2, 1]);  add_82 = None
    mul_95: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_68, permute_59);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_71: "f32[1568, 768]" = torch.ops.aten.view.default(mul_95, [1568, 768]);  mul_95 = None
    permute_60: "f32[768, 256]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_23: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_122, view_71, permute_60);  primals_122 = None
    view_72: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_23, [8, 196, 256]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_59: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_83: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_76, clone_59);  add_76 = clone_59 = None
    clone_60: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_83, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_60, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 196, 1]" = var_mean_24[0]
    getitem_73: "f32[8, 196, 1]" = var_mean_24[1];  var_mean_24 = None
    add_84: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_24: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_24: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_60, getitem_73);  clone_60 = getitem_73 = None
    mul_96: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_97: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_96, primals_123)
    add_85: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_97, primals_124);  mul_97 = primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_73: "f32[1568, 256]" = torch.ops.aten.view.default(add_85, [1568, 256]);  add_85 = None
    permute_61: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_24: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_126, view_73, permute_61);  primals_126 = None
    view_74: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_24, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_98: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_74, 0.5)
    mul_99: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_74, 0.7071067811865476);  view_74 = None
    erf_12: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_86: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_100: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_98, add_86);  mul_98 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_61: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_100);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_12 = torch.ops.aten.split.Tensor(clone_61, 768, -1);  clone_61 = None
    getitem_74: "f32[8, 196, 768]" = split_12[0]
    getitem_75: "f32[8, 196, 768]" = split_12[1];  split_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_62: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_75, memory_format = torch.contiguous_format);  getitem_75 = None
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_62, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 196, 1]" = var_mean_25[0]
    getitem_77: "f32[8, 196, 1]" = var_mean_25[1];  var_mean_25 = None
    add_87: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_25: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_25: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_62, getitem_77);  clone_62 = getitem_77 = None
    mul_101: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    mul_102: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_101, primals_127)
    add_88: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_102, primals_128);  mul_102 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_62: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_88, [0, 2, 1]);  add_88 = None
    permute_63: "f32[196, 196]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    clone_63: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_75: "f32[6144, 196]" = torch.ops.aten.view.default(clone_63, [6144, 196]);  clone_63 = None
    mm_12: "f32[6144, 196]" = torch.ops.aten.mm.default(view_75, permute_63)
    view_76: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_12, [8, 768, 196])
    add_89: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_76, primals_130);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_64: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_89, [0, 2, 1]);  add_89 = None
    mul_103: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_74, permute_64);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_77: "f32[1568, 768]" = torch.ops.aten.view.default(mul_103, [1568, 768]);  mul_103 = None
    permute_65: "f32[768, 256]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_25: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_132, view_77, permute_65);  primals_132 = None
    view_78: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_25, [8, 196, 256]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_64: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_78);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_90: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_83, clone_64);  add_83 = clone_64 = None
    clone_65: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_65, [2], correction = 0, keepdim = True)
    getitem_78: "f32[8, 196, 1]" = var_mean_26[0]
    getitem_79: "f32[8, 196, 1]" = var_mean_26[1];  var_mean_26 = None
    add_91: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_26: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_26: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_65, getitem_79);  clone_65 = getitem_79 = None
    mul_104: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    mul_105: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_104, primals_133)
    add_92: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_105, primals_134);  mul_105 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_79: "f32[1568, 256]" = torch.ops.aten.view.default(add_92, [1568, 256]);  add_92 = None
    permute_66: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_26: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_136, view_79, permute_66);  primals_136 = None
    view_80: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_26, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_106: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_80, 0.5)
    mul_107: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476);  view_80 = None
    erf_13: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
    add_93: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_108: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_106, add_93);  mul_106 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_66: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_108);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_13 = torch.ops.aten.split.Tensor(clone_66, 768, -1);  clone_66 = None
    getitem_80: "f32[8, 196, 768]" = split_13[0]
    getitem_81: "f32[8, 196, 768]" = split_13[1];  split_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_67: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_81, memory_format = torch.contiguous_format);  getitem_81 = None
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_67, [2], correction = 0, keepdim = True)
    getitem_82: "f32[8, 196, 1]" = var_mean_27[0]
    getitem_83: "f32[8, 196, 1]" = var_mean_27[1];  var_mean_27 = None
    add_94: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_27: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_27: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_67, getitem_83);  clone_67 = getitem_83 = None
    mul_109: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    mul_110: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_109, primals_137)
    add_95: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_110, primals_138);  mul_110 = primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_67: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_95, [0, 2, 1]);  add_95 = None
    permute_68: "f32[196, 196]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    clone_68: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    view_81: "f32[6144, 196]" = torch.ops.aten.view.default(clone_68, [6144, 196]);  clone_68 = None
    mm_13: "f32[6144, 196]" = torch.ops.aten.mm.default(view_81, permute_68)
    view_82: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_13, [8, 768, 196])
    add_96: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_82, primals_140);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_69: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_96, [0, 2, 1]);  add_96 = None
    mul_111: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_80, permute_69);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_83: "f32[1568, 768]" = torch.ops.aten.view.default(mul_111, [1568, 768]);  mul_111 = None
    permute_70: "f32[768, 256]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_27: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_142, view_83, permute_70);  primals_142 = None
    view_84: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_27, [8, 196, 256]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_69: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_84);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_97: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_90, clone_69);  add_90 = clone_69 = None
    clone_70: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_97, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_70, [2], correction = 0, keepdim = True)
    getitem_84: "f32[8, 196, 1]" = var_mean_28[0]
    getitem_85: "f32[8, 196, 1]" = var_mean_28[1];  var_mean_28 = None
    add_98: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
    rsqrt_28: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_28: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_70, getitem_85);  clone_70 = getitem_85 = None
    mul_112: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    mul_113: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_112, primals_143)
    add_99: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_113, primals_144);  mul_113 = primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_85: "f32[1568, 256]" = torch.ops.aten.view.default(add_99, [1568, 256]);  add_99 = None
    permute_71: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_28: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_146, view_85, permute_71);  primals_146 = None
    view_86: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_28, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_114: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_86, 0.5)
    mul_115: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_86, 0.7071067811865476);  view_86 = None
    erf_14: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_115);  mul_115 = None
    add_100: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_116: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_114, add_100);  mul_114 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_71: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_116);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_14 = torch.ops.aten.split.Tensor(clone_71, 768, -1);  clone_71 = None
    getitem_86: "f32[8, 196, 768]" = split_14[0]
    getitem_87: "f32[8, 196, 768]" = split_14[1];  split_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_72: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_87, memory_format = torch.contiguous_format);  getitem_87 = None
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_72, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 196, 1]" = var_mean_29[0]
    getitem_89: "f32[8, 196, 1]" = var_mean_29[1];  var_mean_29 = None
    add_101: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_29: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_29: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_72, getitem_89);  clone_72 = getitem_89 = None
    mul_117: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    mul_118: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_117, primals_147)
    add_102: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_118, primals_148);  mul_118 = primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_72: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_102, [0, 2, 1]);  add_102 = None
    permute_73: "f32[196, 196]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    clone_73: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    view_87: "f32[6144, 196]" = torch.ops.aten.view.default(clone_73, [6144, 196]);  clone_73 = None
    mm_14: "f32[6144, 196]" = torch.ops.aten.mm.default(view_87, permute_73)
    view_88: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_14, [8, 768, 196])
    add_103: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_88, primals_150);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_74: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_103, [0, 2, 1]);  add_103 = None
    mul_119: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_86, permute_74);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_89: "f32[1568, 768]" = torch.ops.aten.view.default(mul_119, [1568, 768]);  mul_119 = None
    permute_75: "f32[768, 256]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_29: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_152, view_89, permute_75);  primals_152 = None
    view_90: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_29, [8, 196, 256]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_74: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_90);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_104: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_97, clone_74);  add_97 = clone_74 = None
    clone_75: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_104, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_75, [2], correction = 0, keepdim = True)
    getitem_90: "f32[8, 196, 1]" = var_mean_30[0]
    getitem_91: "f32[8, 196, 1]" = var_mean_30[1];  var_mean_30 = None
    add_105: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
    rsqrt_30: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_30: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_75, getitem_91);  clone_75 = getitem_91 = None
    mul_120: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    mul_121: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_120, primals_153)
    add_106: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_121, primals_154);  mul_121 = primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_91: "f32[1568, 256]" = torch.ops.aten.view.default(add_106, [1568, 256]);  add_106 = None
    permute_76: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_30: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_156, view_91, permute_76);  primals_156 = None
    view_92: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_30, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_122: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_92, 0.5)
    mul_123: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_92, 0.7071067811865476);  view_92 = None
    erf_15: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_107: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_124: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_122, add_107);  mul_122 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_76: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_124);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_15 = torch.ops.aten.split.Tensor(clone_76, 768, -1);  clone_76 = None
    getitem_92: "f32[8, 196, 768]" = split_15[0]
    getitem_93: "f32[8, 196, 768]" = split_15[1];  split_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_77: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_93, memory_format = torch.contiguous_format);  getitem_93 = None
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_77, [2], correction = 0, keepdim = True)
    getitem_94: "f32[8, 196, 1]" = var_mean_31[0]
    getitem_95: "f32[8, 196, 1]" = var_mean_31[1];  var_mean_31 = None
    add_108: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_31: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_31: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_77, getitem_95);  clone_77 = getitem_95 = None
    mul_125: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    mul_126: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_125, primals_157)
    add_109: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_126, primals_158);  mul_126 = primals_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_77: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_109, [0, 2, 1]);  add_109 = None
    permute_78: "f32[196, 196]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    clone_78: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_93: "f32[6144, 196]" = torch.ops.aten.view.default(clone_78, [6144, 196]);  clone_78 = None
    mm_15: "f32[6144, 196]" = torch.ops.aten.mm.default(view_93, permute_78)
    view_94: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_15, [8, 768, 196])
    add_110: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_94, primals_160);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_79: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_110, [0, 2, 1]);  add_110 = None
    mul_127: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_92, permute_79);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_95: "f32[1568, 768]" = torch.ops.aten.view.default(mul_127, [1568, 768]);  mul_127 = None
    permute_80: "f32[768, 256]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    addmm_31: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_162, view_95, permute_80);  primals_162 = None
    view_96: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_31, [8, 196, 256]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_79: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_111: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_104, clone_79);  add_104 = clone_79 = None
    clone_80: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_80, [2], correction = 0, keepdim = True)
    getitem_96: "f32[8, 196, 1]" = var_mean_32[0]
    getitem_97: "f32[8, 196, 1]" = var_mean_32[1];  var_mean_32 = None
    add_112: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
    rsqrt_32: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_32: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_80, getitem_97);  clone_80 = getitem_97 = None
    mul_128: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    mul_129: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_128, primals_163)
    add_113: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_129, primals_164);  mul_129 = primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_97: "f32[1568, 256]" = torch.ops.aten.view.default(add_113, [1568, 256]);  add_113 = None
    permute_81: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    addmm_32: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_166, view_97, permute_81);  primals_166 = None
    view_98: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_32, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_130: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_131: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
    erf_16: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
    add_114: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_132: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_130, add_114);  mul_130 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_81: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_132);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_16 = torch.ops.aten.split.Tensor(clone_81, 768, -1);  clone_81 = None
    getitem_98: "f32[8, 196, 768]" = split_16[0]
    getitem_99: "f32[8, 196, 768]" = split_16[1];  split_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_82: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_99, memory_format = torch.contiguous_format);  getitem_99 = None
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_82, [2], correction = 0, keepdim = True)
    getitem_100: "f32[8, 196, 1]" = var_mean_33[0]
    getitem_101: "f32[8, 196, 1]" = var_mean_33[1];  var_mean_33 = None
    add_115: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_33: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_33: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_82, getitem_101);  clone_82 = getitem_101 = None
    mul_133: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    mul_134: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_133, primals_167)
    add_116: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_134, primals_168);  mul_134 = primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_82: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_116, [0, 2, 1]);  add_116 = None
    permute_83: "f32[196, 196]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    clone_83: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_99: "f32[6144, 196]" = torch.ops.aten.view.default(clone_83, [6144, 196]);  clone_83 = None
    mm_16: "f32[6144, 196]" = torch.ops.aten.mm.default(view_99, permute_83)
    view_100: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_16, [8, 768, 196])
    add_117: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_100, primals_170);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_84: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_117, [0, 2, 1]);  add_117 = None
    mul_135: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_98, permute_84);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_101: "f32[1568, 768]" = torch.ops.aten.view.default(mul_135, [1568, 768]);  mul_135 = None
    permute_85: "f32[768, 256]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm_33: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_172, view_101, permute_85);  primals_172 = None
    view_102: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_33, [8, 196, 256]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_84: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_102);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_118: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_111, clone_84);  add_111 = clone_84 = None
    clone_85: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_118, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
    getitem_102: "f32[8, 196, 1]" = var_mean_34[0]
    getitem_103: "f32[8, 196, 1]" = var_mean_34[1];  var_mean_34 = None
    add_119: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-06);  getitem_102 = None
    rsqrt_34: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_34: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_85, getitem_103);  clone_85 = getitem_103 = None
    mul_136: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    mul_137: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_136, primals_173)
    add_120: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_137, primals_174);  mul_137 = primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_103: "f32[1568, 256]" = torch.ops.aten.view.default(add_120, [1568, 256]);  add_120 = None
    permute_86: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_34: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_176, view_103, permute_86);  primals_176 = None
    view_104: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_34, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_138: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_104, 0.5)
    mul_139: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_104, 0.7071067811865476);  view_104 = None
    erf_17: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_121: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_140: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_138, add_121);  mul_138 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_86: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_140);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_17 = torch.ops.aten.split.Tensor(clone_86, 768, -1);  clone_86 = None
    getitem_104: "f32[8, 196, 768]" = split_17[0]
    getitem_105: "f32[8, 196, 768]" = split_17[1];  split_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_87: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_105, memory_format = torch.contiguous_format);  getitem_105 = None
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_87, [2], correction = 0, keepdim = True)
    getitem_106: "f32[8, 196, 1]" = var_mean_35[0]
    getitem_107: "f32[8, 196, 1]" = var_mean_35[1];  var_mean_35 = None
    add_122: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_35: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_35: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_87, getitem_107);  clone_87 = getitem_107 = None
    mul_141: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    mul_142: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_141, primals_177)
    add_123: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_142, primals_178);  mul_142 = primals_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_87: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_123, [0, 2, 1]);  add_123 = None
    permute_88: "f32[196, 196]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    clone_88: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    view_105: "f32[6144, 196]" = torch.ops.aten.view.default(clone_88, [6144, 196]);  clone_88 = None
    mm_17: "f32[6144, 196]" = torch.ops.aten.mm.default(view_105, permute_88)
    view_106: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_17, [8, 768, 196])
    add_124: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_106, primals_180);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_89: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_124, [0, 2, 1]);  add_124 = None
    mul_143: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_104, permute_89);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_107: "f32[1568, 768]" = torch.ops.aten.view.default(mul_143, [1568, 768]);  mul_143 = None
    permute_90: "f32[768, 256]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_35: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_182, view_107, permute_90);  primals_182 = None
    view_108: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_35, [8, 196, 256]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_89: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_125: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_118, clone_89);  add_118 = clone_89 = None
    clone_90: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_90, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 196, 1]" = var_mean_36[0]
    getitem_109: "f32[8, 196, 1]" = var_mean_36[1];  var_mean_36 = None
    add_126: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_36: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_36: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_90, getitem_109);  clone_90 = getitem_109 = None
    mul_144: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    mul_145: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_144, primals_183)
    add_127: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_145, primals_184);  mul_145 = primals_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_109: "f32[1568, 256]" = torch.ops.aten.view.default(add_127, [1568, 256]);  add_127 = None
    permute_91: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    addmm_36: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_186, view_109, permute_91);  primals_186 = None
    view_110: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_36, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_146: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_110, 0.5)
    mul_147: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_110, 0.7071067811865476);  view_110 = None
    erf_18: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_128: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_148: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_146, add_128);  mul_146 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_91: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_148);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_18 = torch.ops.aten.split.Tensor(clone_91, 768, -1);  clone_91 = None
    getitem_110: "f32[8, 196, 768]" = split_18[0]
    getitem_111: "f32[8, 196, 768]" = split_18[1];  split_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_92: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_111, memory_format = torch.contiguous_format);  getitem_111 = None
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_92, [2], correction = 0, keepdim = True)
    getitem_112: "f32[8, 196, 1]" = var_mean_37[0]
    getitem_113: "f32[8, 196, 1]" = var_mean_37[1];  var_mean_37 = None
    add_129: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_37: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_37: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_92, getitem_113);  clone_92 = getitem_113 = None
    mul_149: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    mul_150: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_149, primals_187)
    add_130: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_150, primals_188);  mul_150 = primals_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_92: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_130, [0, 2, 1]);  add_130 = None
    permute_93: "f32[196, 196]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    clone_93: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    view_111: "f32[6144, 196]" = torch.ops.aten.view.default(clone_93, [6144, 196]);  clone_93 = None
    mm_18: "f32[6144, 196]" = torch.ops.aten.mm.default(view_111, permute_93)
    view_112: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_18, [8, 768, 196])
    add_131: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_112, primals_190);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_94: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_131, [0, 2, 1]);  add_131 = None
    mul_151: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_110, permute_94);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_113: "f32[1568, 768]" = torch.ops.aten.view.default(mul_151, [1568, 768]);  mul_151 = None
    permute_95: "f32[768, 256]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
    addmm_37: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_192, view_113, permute_95);  primals_192 = None
    view_114: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_37, [8, 196, 256]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_94: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_114);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_132: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_125, clone_94);  add_125 = clone_94 = None
    clone_95: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_132, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_95, [2], correction = 0, keepdim = True)
    getitem_114: "f32[8, 196, 1]" = var_mean_38[0]
    getitem_115: "f32[8, 196, 1]" = var_mean_38[1];  var_mean_38 = None
    add_133: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-06);  getitem_114 = None
    rsqrt_38: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_38: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_95, getitem_115);  clone_95 = getitem_115 = None
    mul_152: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    mul_153: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_152, primals_193)
    add_134: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_153, primals_194);  mul_153 = primals_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_115: "f32[1568, 256]" = torch.ops.aten.view.default(add_134, [1568, 256]);  add_134 = None
    permute_96: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    addmm_38: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_196, view_115, permute_96);  primals_196 = None
    view_116: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_38, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_154: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_116, 0.5)
    mul_155: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476);  view_116 = None
    erf_19: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_135: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_156: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_154, add_135);  mul_154 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_96: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_156);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_19 = torch.ops.aten.split.Tensor(clone_96, 768, -1);  clone_96 = None
    getitem_116: "f32[8, 196, 768]" = split_19[0]
    getitem_117: "f32[8, 196, 768]" = split_19[1];  split_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_97: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_117, memory_format = torch.contiguous_format);  getitem_117 = None
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_97, [2], correction = 0, keepdim = True)
    getitem_118: "f32[8, 196, 1]" = var_mean_39[0]
    getitem_119: "f32[8, 196, 1]" = var_mean_39[1];  var_mean_39 = None
    add_136: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_39: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_39: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_97, getitem_119);  clone_97 = getitem_119 = None
    mul_157: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    mul_158: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_157, primals_197)
    add_137: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_158, primals_198);  mul_158 = primals_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_97: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_137, [0, 2, 1]);  add_137 = None
    permute_98: "f32[196, 196]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    clone_98: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    view_117: "f32[6144, 196]" = torch.ops.aten.view.default(clone_98, [6144, 196]);  clone_98 = None
    mm_19: "f32[6144, 196]" = torch.ops.aten.mm.default(view_117, permute_98)
    view_118: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_19, [8, 768, 196])
    add_138: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_118, primals_200);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_99: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_138, [0, 2, 1]);  add_138 = None
    mul_159: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_116, permute_99);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_119: "f32[1568, 768]" = torch.ops.aten.view.default(mul_159, [1568, 768]);  mul_159 = None
    permute_100: "f32[768, 256]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
    addmm_39: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_202, view_119, permute_100);  primals_202 = None
    view_120: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_39, [8, 196, 256]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_99: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_139: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_132, clone_99);  add_132 = clone_99 = None
    clone_100: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_139, memory_format = torch.contiguous_format)
    var_mean_40 = torch.ops.aten.var_mean.correction(clone_100, [2], correction = 0, keepdim = True)
    getitem_120: "f32[8, 196, 1]" = var_mean_40[0]
    getitem_121: "f32[8, 196, 1]" = var_mean_40[1];  var_mean_40 = None
    add_140: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
    rsqrt_40: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_40: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_100, getitem_121);  clone_100 = getitem_121 = None
    mul_160: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    mul_161: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_160, primals_203)
    add_141: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_161, primals_204);  mul_161 = primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_121: "f32[1568, 256]" = torch.ops.aten.view.default(add_141, [1568, 256]);  add_141 = None
    permute_101: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    addmm_40: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_206, view_121, permute_101);  primals_206 = None
    view_122: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_40, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_162: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_122, 0.5)
    mul_163: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_122, 0.7071067811865476);  view_122 = None
    erf_20: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
    add_142: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_164: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_162, add_142);  mul_162 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_101: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_164);  mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_20 = torch.ops.aten.split.Tensor(clone_101, 768, -1);  clone_101 = None
    getitem_122: "f32[8, 196, 768]" = split_20[0]
    getitem_123: "f32[8, 196, 768]" = split_20[1];  split_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_102: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_123, memory_format = torch.contiguous_format);  getitem_123 = None
    var_mean_41 = torch.ops.aten.var_mean.correction(clone_102, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 196, 1]" = var_mean_41[0]
    getitem_125: "f32[8, 196, 1]" = var_mean_41[1];  var_mean_41 = None
    add_143: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
    rsqrt_41: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_41: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_102, getitem_125);  clone_102 = getitem_125 = None
    mul_165: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    mul_166: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_165, primals_207)
    add_144: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_166, primals_208);  mul_166 = primals_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_102: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_144, [0, 2, 1]);  add_144 = None
    permute_103: "f32[196, 196]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    clone_103: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    view_123: "f32[6144, 196]" = torch.ops.aten.view.default(clone_103, [6144, 196]);  clone_103 = None
    mm_20: "f32[6144, 196]" = torch.ops.aten.mm.default(view_123, permute_103)
    view_124: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_20, [8, 768, 196])
    add_145: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_124, primals_210);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_104: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_145, [0, 2, 1]);  add_145 = None
    mul_167: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_122, permute_104);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_125: "f32[1568, 768]" = torch.ops.aten.view.default(mul_167, [1568, 768]);  mul_167 = None
    permute_105: "f32[768, 256]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
    addmm_41: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_212, view_125, permute_105);  primals_212 = None
    view_126: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_41, [8, 196, 256]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_104: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_126);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_146: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_139, clone_104);  add_139 = clone_104 = None
    clone_105: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_146, memory_format = torch.contiguous_format)
    var_mean_42 = torch.ops.aten.var_mean.correction(clone_105, [2], correction = 0, keepdim = True)
    getitem_126: "f32[8, 196, 1]" = var_mean_42[0]
    getitem_127: "f32[8, 196, 1]" = var_mean_42[1];  var_mean_42 = None
    add_147: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
    rsqrt_42: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_42: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_105, getitem_127);  clone_105 = getitem_127 = None
    mul_168: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    mul_169: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_168, primals_213)
    add_148: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_169, primals_214);  mul_169 = primals_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_127: "f32[1568, 256]" = torch.ops.aten.view.default(add_148, [1568, 256]);  add_148 = None
    permute_106: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    addmm_42: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_216, view_127, permute_106);  primals_216 = None
    view_128: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_42, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_170: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_128, 0.5)
    mul_171: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476);  view_128 = None
    erf_21: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
    add_149: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_172: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_170, add_149);  mul_170 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_106: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_172);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_21 = torch.ops.aten.split.Tensor(clone_106, 768, -1);  clone_106 = None
    getitem_128: "f32[8, 196, 768]" = split_21[0]
    getitem_129: "f32[8, 196, 768]" = split_21[1];  split_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_107: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_129, memory_format = torch.contiguous_format);  getitem_129 = None
    var_mean_43 = torch.ops.aten.var_mean.correction(clone_107, [2], correction = 0, keepdim = True)
    getitem_130: "f32[8, 196, 1]" = var_mean_43[0]
    getitem_131: "f32[8, 196, 1]" = var_mean_43[1];  var_mean_43 = None
    add_150: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
    rsqrt_43: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_43: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_107, getitem_131);  clone_107 = getitem_131 = None
    mul_173: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    mul_174: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_173, primals_217)
    add_151: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_174, primals_218);  mul_174 = primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_107: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_151, [0, 2, 1]);  add_151 = None
    permute_108: "f32[196, 196]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
    clone_108: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    view_129: "f32[6144, 196]" = torch.ops.aten.view.default(clone_108, [6144, 196]);  clone_108 = None
    mm_21: "f32[6144, 196]" = torch.ops.aten.mm.default(view_129, permute_108)
    view_130: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_21, [8, 768, 196])
    add_152: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_130, primals_220);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_109: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_152, [0, 2, 1]);  add_152 = None
    mul_175: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_128, permute_109);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_131: "f32[1568, 768]" = torch.ops.aten.view.default(mul_175, [1568, 768]);  mul_175 = None
    permute_110: "f32[768, 256]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
    addmm_43: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_222, view_131, permute_110);  primals_222 = None
    view_132: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_43, [8, 196, 256]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_109: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_132);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_153: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_146, clone_109);  add_146 = clone_109 = None
    clone_110: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_153, memory_format = torch.contiguous_format)
    var_mean_44 = torch.ops.aten.var_mean.correction(clone_110, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 196, 1]" = var_mean_44[0]
    getitem_133: "f32[8, 196, 1]" = var_mean_44[1];  var_mean_44 = None
    add_154: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_44: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_44: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_110, getitem_133);  clone_110 = getitem_133 = None
    mul_176: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    mul_177: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_176, primals_223)
    add_155: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_177, primals_224);  mul_177 = primals_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_133: "f32[1568, 256]" = torch.ops.aten.view.default(add_155, [1568, 256]);  add_155 = None
    permute_111: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
    addmm_44: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_226, view_133, permute_111);  primals_226 = None
    view_134: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_44, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_178: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_134, 0.5)
    mul_179: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_134, 0.7071067811865476);  view_134 = None
    erf_22: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_179);  mul_179 = None
    add_156: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_180: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_178, add_156);  mul_178 = add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_111: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_180);  mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_22 = torch.ops.aten.split.Tensor(clone_111, 768, -1);  clone_111 = None
    getitem_134: "f32[8, 196, 768]" = split_22[0]
    getitem_135: "f32[8, 196, 768]" = split_22[1];  split_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_112: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_135, memory_format = torch.contiguous_format);  getitem_135 = None
    var_mean_45 = torch.ops.aten.var_mean.correction(clone_112, [2], correction = 0, keepdim = True)
    getitem_136: "f32[8, 196, 1]" = var_mean_45[0]
    getitem_137: "f32[8, 196, 1]" = var_mean_45[1];  var_mean_45 = None
    add_157: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05);  getitem_136 = None
    rsqrt_45: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_45: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_112, getitem_137);  clone_112 = getitem_137 = None
    mul_181: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    mul_182: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_181, primals_227)
    add_158: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_182, primals_228);  mul_182 = primals_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_112: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_158, [0, 2, 1]);  add_158 = None
    permute_113: "f32[196, 196]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
    clone_113: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    view_135: "f32[6144, 196]" = torch.ops.aten.view.default(clone_113, [6144, 196]);  clone_113 = None
    mm_22: "f32[6144, 196]" = torch.ops.aten.mm.default(view_135, permute_113)
    view_136: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_22, [8, 768, 196])
    add_159: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_136, primals_230);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_114: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_159, [0, 2, 1]);  add_159 = None
    mul_183: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_134, permute_114);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_137: "f32[1568, 768]" = torch.ops.aten.view.default(mul_183, [1568, 768]);  mul_183 = None
    permute_115: "f32[768, 256]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
    addmm_45: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_232, view_137, permute_115);  primals_232 = None
    view_138: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_45, [8, 196, 256]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_114: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_138);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_160: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_153, clone_114);  add_153 = clone_114 = None
    clone_115: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_160, memory_format = torch.contiguous_format)
    var_mean_46 = torch.ops.aten.var_mean.correction(clone_115, [2], correction = 0, keepdim = True)
    getitem_138: "f32[8, 196, 1]" = var_mean_46[0]
    getitem_139: "f32[8, 196, 1]" = var_mean_46[1];  var_mean_46 = None
    add_161: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-06);  getitem_138 = None
    rsqrt_46: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_46: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_115, getitem_139);  clone_115 = getitem_139 = None
    mul_184: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    mul_185: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_184, primals_233)
    add_162: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_185, primals_234);  mul_185 = primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_139: "f32[1568, 256]" = torch.ops.aten.view.default(add_162, [1568, 256]);  add_162 = None
    permute_116: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    addmm_46: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_236, view_139, permute_116);  primals_236 = None
    view_140: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_46, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_186: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_140, 0.5)
    mul_187: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_140, 0.7071067811865476);  view_140 = None
    erf_23: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_187);  mul_187 = None
    add_163: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_188: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_186, add_163);  mul_186 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_116: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_188);  mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_23 = torch.ops.aten.split.Tensor(clone_116, 768, -1);  clone_116 = None
    getitem_140: "f32[8, 196, 768]" = split_23[0]
    getitem_141: "f32[8, 196, 768]" = split_23[1];  split_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_117: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_141, memory_format = torch.contiguous_format);  getitem_141 = None
    var_mean_47 = torch.ops.aten.var_mean.correction(clone_117, [2], correction = 0, keepdim = True)
    getitem_142: "f32[8, 196, 1]" = var_mean_47[0]
    getitem_143: "f32[8, 196, 1]" = var_mean_47[1];  var_mean_47 = None
    add_164: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
    rsqrt_47: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_47: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_117, getitem_143);  clone_117 = getitem_143 = None
    mul_189: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    mul_190: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_189, primals_237)
    add_165: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_190, primals_238);  mul_190 = primals_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_117: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_165, [0, 2, 1]);  add_165 = None
    permute_118: "f32[196, 196]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
    clone_118: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_141: "f32[6144, 196]" = torch.ops.aten.view.default(clone_118, [6144, 196]);  clone_118 = None
    mm_23: "f32[6144, 196]" = torch.ops.aten.mm.default(view_141, permute_118)
    view_142: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_23, [8, 768, 196])
    add_166: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_142, primals_240);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_119: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_166, [0, 2, 1]);  add_166 = None
    mul_191: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_140, permute_119);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_143: "f32[1568, 768]" = torch.ops.aten.view.default(mul_191, [1568, 768]);  mul_191 = None
    permute_120: "f32[768, 256]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
    addmm_47: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_242, view_143, permute_120);  primals_242 = None
    view_144: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_47, [8, 196, 256]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_119: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_167: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_160, clone_119);  add_160 = clone_119 = None
    clone_120: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format)
    var_mean_48 = torch.ops.aten.var_mean.correction(clone_120, [2], correction = 0, keepdim = True)
    getitem_144: "f32[8, 196, 1]" = var_mean_48[0]
    getitem_145: "f32[8, 196, 1]" = var_mean_48[1];  var_mean_48 = None
    add_168: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_48: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_48: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_120, getitem_145);  clone_120 = getitem_145 = None
    mul_192: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    mul_193: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_192, primals_243)
    add_169: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_193, primals_244);  mul_193 = primals_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_145: "f32[1568, 256]" = torch.ops.aten.view.default(add_169, [1568, 256]);  add_169 = None
    permute_121: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
    addmm_48: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_246, view_145, permute_121);  primals_246 = None
    view_146: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_48, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_194: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_146, 0.5)
    mul_195: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476);  view_146 = None
    erf_24: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_170: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_196: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_194, add_170);  mul_194 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_121: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_196);  mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_24 = torch.ops.aten.split.Tensor(clone_121, 768, -1);  clone_121 = None
    getitem_146: "f32[8, 196, 768]" = split_24[0]
    getitem_147: "f32[8, 196, 768]" = split_24[1];  split_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_122: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_147, memory_format = torch.contiguous_format);  getitem_147 = None
    var_mean_49 = torch.ops.aten.var_mean.correction(clone_122, [2], correction = 0, keepdim = True)
    getitem_148: "f32[8, 196, 1]" = var_mean_49[0]
    getitem_149: "f32[8, 196, 1]" = var_mean_49[1];  var_mean_49 = None
    add_171: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
    rsqrt_49: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_49: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_122, getitem_149);  clone_122 = getitem_149 = None
    mul_197: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    mul_198: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_197, primals_247)
    add_172: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_198, primals_248);  mul_198 = primals_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_122: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_172, [0, 2, 1]);  add_172 = None
    permute_123: "f32[196, 196]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
    clone_123: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_147: "f32[6144, 196]" = torch.ops.aten.view.default(clone_123, [6144, 196]);  clone_123 = None
    mm_24: "f32[6144, 196]" = torch.ops.aten.mm.default(view_147, permute_123)
    view_148: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_24, [8, 768, 196])
    add_173: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_148, primals_250);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_124: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_173, [0, 2, 1]);  add_173 = None
    mul_199: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_146, permute_124);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_149: "f32[1568, 768]" = torch.ops.aten.view.default(mul_199, [1568, 768]);  mul_199 = None
    permute_125: "f32[768, 256]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
    addmm_49: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_252, view_149, permute_125);  primals_252 = None
    view_150: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_49, [8, 196, 256]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_124: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_150);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_174: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_167, clone_124);  add_167 = clone_124 = None
    clone_125: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_174, memory_format = torch.contiguous_format)
    var_mean_50 = torch.ops.aten.var_mean.correction(clone_125, [2], correction = 0, keepdim = True)
    getitem_150: "f32[8, 196, 1]" = var_mean_50[0]
    getitem_151: "f32[8, 196, 1]" = var_mean_50[1];  var_mean_50 = None
    add_175: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-06);  getitem_150 = None
    rsqrt_50: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_50: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_125, getitem_151);  clone_125 = getitem_151 = None
    mul_200: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    mul_201: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_200, primals_253)
    add_176: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_201, primals_254);  mul_201 = primals_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_151: "f32[1568, 256]" = torch.ops.aten.view.default(add_176, [1568, 256]);  add_176 = None
    permute_126: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_50: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_256, view_151, permute_126);  primals_256 = None
    view_152: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_50, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_202: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_152, 0.5)
    mul_203: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_152, 0.7071067811865476);  view_152 = None
    erf_25: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_203);  mul_203 = None
    add_177: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_204: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_202, add_177);  mul_202 = add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_126: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_204);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_25 = torch.ops.aten.split.Tensor(clone_126, 768, -1);  clone_126 = None
    getitem_152: "f32[8, 196, 768]" = split_25[0]
    getitem_153: "f32[8, 196, 768]" = split_25[1];  split_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_127: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_153, memory_format = torch.contiguous_format);  getitem_153 = None
    var_mean_51 = torch.ops.aten.var_mean.correction(clone_127, [2], correction = 0, keepdim = True)
    getitem_154: "f32[8, 196, 1]" = var_mean_51[0]
    getitem_155: "f32[8, 196, 1]" = var_mean_51[1];  var_mean_51 = None
    add_178: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
    rsqrt_51: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_51: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_127, getitem_155);  clone_127 = getitem_155 = None
    mul_205: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    mul_206: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_205, primals_257)
    add_179: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_206, primals_258);  mul_206 = primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_127: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_179, [0, 2, 1]);  add_179 = None
    permute_128: "f32[196, 196]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
    clone_128: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    view_153: "f32[6144, 196]" = torch.ops.aten.view.default(clone_128, [6144, 196]);  clone_128 = None
    mm_25: "f32[6144, 196]" = torch.ops.aten.mm.default(view_153, permute_128)
    view_154: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_25, [8, 768, 196])
    add_180: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_154, primals_260);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_129: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_180, [0, 2, 1]);  add_180 = None
    mul_207: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_152, permute_129);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_155: "f32[1568, 768]" = torch.ops.aten.view.default(mul_207, [1568, 768]);  mul_207 = None
    permute_130: "f32[768, 256]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
    addmm_51: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_262, view_155, permute_130);  primals_262 = None
    view_156: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_51, [8, 196, 256]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_129: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_156);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_181: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_174, clone_129);  add_174 = clone_129 = None
    clone_130: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_181, memory_format = torch.contiguous_format)
    var_mean_52 = torch.ops.aten.var_mean.correction(clone_130, [2], correction = 0, keepdim = True)
    getitem_156: "f32[8, 196, 1]" = var_mean_52[0]
    getitem_157: "f32[8, 196, 1]" = var_mean_52[1];  var_mean_52 = None
    add_182: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
    rsqrt_52: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_52: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_130, getitem_157);  clone_130 = getitem_157 = None
    mul_208: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    mul_209: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_208, primals_263)
    add_183: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_209, primals_264);  mul_209 = primals_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_157: "f32[1568, 256]" = torch.ops.aten.view.default(add_183, [1568, 256]);  add_183 = None
    permute_131: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
    addmm_52: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_266, view_157, permute_131);  primals_266 = None
    view_158: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_52, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_210: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
    mul_211: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
    erf_26: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
    add_184: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_212: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_210, add_184);  mul_210 = add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_131: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_212);  mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_26 = torch.ops.aten.split.Tensor(clone_131, 768, -1);  clone_131 = None
    getitem_158: "f32[8, 196, 768]" = split_26[0]
    getitem_159: "f32[8, 196, 768]" = split_26[1];  split_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_132: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_159, memory_format = torch.contiguous_format);  getitem_159 = None
    var_mean_53 = torch.ops.aten.var_mean.correction(clone_132, [2], correction = 0, keepdim = True)
    getitem_160: "f32[8, 196, 1]" = var_mean_53[0]
    getitem_161: "f32[8, 196, 1]" = var_mean_53[1];  var_mean_53 = None
    add_185: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
    rsqrt_53: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_53: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_132, getitem_161);  clone_132 = getitem_161 = None
    mul_213: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    mul_214: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_213, primals_267)
    add_186: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_214, primals_268);  mul_214 = primals_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_132: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_186, [0, 2, 1]);  add_186 = None
    permute_133: "f32[196, 196]" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
    clone_133: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    view_159: "f32[6144, 196]" = torch.ops.aten.view.default(clone_133, [6144, 196]);  clone_133 = None
    mm_26: "f32[6144, 196]" = torch.ops.aten.mm.default(view_159, permute_133)
    view_160: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_26, [8, 768, 196])
    add_187: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_160, primals_270);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_134: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_187, [0, 2, 1]);  add_187 = None
    mul_215: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_158, permute_134);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_161: "f32[1568, 768]" = torch.ops.aten.view.default(mul_215, [1568, 768]);  mul_215 = None
    permute_135: "f32[768, 256]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
    addmm_53: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_272, view_161, permute_135);  primals_272 = None
    view_162: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_53, [8, 196, 256]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_134: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_162);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_188: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_181, clone_134);  add_181 = clone_134 = None
    clone_135: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_188, memory_format = torch.contiguous_format)
    var_mean_54 = torch.ops.aten.var_mean.correction(clone_135, [2], correction = 0, keepdim = True)
    getitem_162: "f32[8, 196, 1]" = var_mean_54[0]
    getitem_163: "f32[8, 196, 1]" = var_mean_54[1];  var_mean_54 = None
    add_189: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
    rsqrt_54: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_54: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_135, getitem_163);  clone_135 = getitem_163 = None
    mul_216: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    mul_217: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_216, primals_273)
    add_190: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_217, primals_274);  mul_217 = primals_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_163: "f32[1568, 256]" = torch.ops.aten.view.default(add_190, [1568, 256]);  add_190 = None
    permute_136: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_275, [1, 0]);  primals_275 = None
    addmm_54: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_276, view_163, permute_136);  primals_276 = None
    view_164: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_54, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_218: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_164, 0.5)
    mul_219: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_164, 0.7071067811865476);  view_164 = None
    erf_27: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_219);  mul_219 = None
    add_191: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_220: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_218, add_191);  mul_218 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_136: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_220);  mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_27 = torch.ops.aten.split.Tensor(clone_136, 768, -1);  clone_136 = None
    getitem_164: "f32[8, 196, 768]" = split_27[0]
    getitem_165: "f32[8, 196, 768]" = split_27[1];  split_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_137: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_165, memory_format = torch.contiguous_format);  getitem_165 = None
    var_mean_55 = torch.ops.aten.var_mean.correction(clone_137, [2], correction = 0, keepdim = True)
    getitem_166: "f32[8, 196, 1]" = var_mean_55[0]
    getitem_167: "f32[8, 196, 1]" = var_mean_55[1];  var_mean_55 = None
    add_192: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05);  getitem_166 = None
    rsqrt_55: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_55: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_137, getitem_167);  clone_137 = getitem_167 = None
    mul_221: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    mul_222: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_221, primals_277)
    add_193: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_222, primals_278);  mul_222 = primals_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_137: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_193, [0, 2, 1]);  add_193 = None
    permute_138: "f32[196, 196]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
    clone_138: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    view_165: "f32[6144, 196]" = torch.ops.aten.view.default(clone_138, [6144, 196]);  clone_138 = None
    mm_27: "f32[6144, 196]" = torch.ops.aten.mm.default(view_165, permute_138)
    view_166: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_27, [8, 768, 196])
    add_194: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_166, primals_280);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_139: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_194, [0, 2, 1]);  add_194 = None
    mul_223: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_164, permute_139);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_167: "f32[1568, 768]" = torch.ops.aten.view.default(mul_223, [1568, 768]);  mul_223 = None
    permute_140: "f32[768, 256]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
    addmm_55: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_282, view_167, permute_140);  primals_282 = None
    view_168: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_55, [8, 196, 256]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_139: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_168);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_195: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_188, clone_139);  add_188 = clone_139 = None
    clone_140: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_195, memory_format = torch.contiguous_format)
    var_mean_56 = torch.ops.aten.var_mean.correction(clone_140, [2], correction = 0, keepdim = True)
    getitem_168: "f32[8, 196, 1]" = var_mean_56[0]
    getitem_169: "f32[8, 196, 1]" = var_mean_56[1];  var_mean_56 = None
    add_196: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
    rsqrt_56: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    sub_56: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_140, getitem_169);  clone_140 = getitem_169 = None
    mul_224: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    mul_225: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_224, primals_283)
    add_197: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_225, primals_284);  mul_225 = primals_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_169: "f32[1568, 256]" = torch.ops.aten.view.default(add_197, [1568, 256]);  add_197 = None
    permute_141: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
    addmm_56: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_286, view_169, permute_141);  primals_286 = None
    view_170: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_56, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_226: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_170, 0.5)
    mul_227: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_170, 0.7071067811865476);  view_170 = None
    erf_28: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_227);  mul_227 = None
    add_198: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_228: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_226, add_198);  mul_226 = add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_141: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_228);  mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_28 = torch.ops.aten.split.Tensor(clone_141, 768, -1);  clone_141 = None
    getitem_170: "f32[8, 196, 768]" = split_28[0]
    getitem_171: "f32[8, 196, 768]" = split_28[1];  split_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_142: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_171, memory_format = torch.contiguous_format);  getitem_171 = None
    var_mean_57 = torch.ops.aten.var_mean.correction(clone_142, [2], correction = 0, keepdim = True)
    getitem_172: "f32[8, 196, 1]" = var_mean_57[0]
    getitem_173: "f32[8, 196, 1]" = var_mean_57[1];  var_mean_57 = None
    add_199: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05);  getitem_172 = None
    rsqrt_57: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_57: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_142, getitem_173);  clone_142 = getitem_173 = None
    mul_229: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    mul_230: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_229, primals_287)
    add_200: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_230, primals_288);  mul_230 = primals_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_142: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_200, [0, 2, 1]);  add_200 = None
    permute_143: "f32[196, 196]" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
    clone_143: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    view_171: "f32[6144, 196]" = torch.ops.aten.view.default(clone_143, [6144, 196]);  clone_143 = None
    mm_28: "f32[6144, 196]" = torch.ops.aten.mm.default(view_171, permute_143)
    view_172: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_28, [8, 768, 196])
    add_201: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_172, primals_290);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_144: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_201, [0, 2, 1]);  add_201 = None
    mul_231: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_170, permute_144);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_173: "f32[1568, 768]" = torch.ops.aten.view.default(mul_231, [1568, 768]);  mul_231 = None
    permute_145: "f32[768, 256]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
    addmm_57: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_292, view_173, permute_145);  primals_292 = None
    view_174: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_57, [8, 196, 256]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_144: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_174);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_202: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_195, clone_144);  add_195 = clone_144 = None
    clone_145: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_202, memory_format = torch.contiguous_format)
    var_mean_58 = torch.ops.aten.var_mean.correction(clone_145, [2], correction = 0, keepdim = True)
    getitem_174: "f32[8, 196, 1]" = var_mean_58[0]
    getitem_175: "f32[8, 196, 1]" = var_mean_58[1];  var_mean_58 = None
    add_203: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
    rsqrt_58: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    sub_58: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_145, getitem_175);  clone_145 = getitem_175 = None
    mul_232: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    mul_233: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_232, primals_293)
    add_204: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_233, primals_294);  mul_233 = primals_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_175: "f32[1568, 256]" = torch.ops.aten.view.default(add_204, [1568, 256]);  add_204 = None
    permute_146: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_295, [1, 0]);  primals_295 = None
    addmm_58: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_296, view_175, permute_146);  primals_296 = None
    view_176: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_58, [8, 196, 1536])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_234: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_176, 0.5)
    mul_235: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476);  view_176 = None
    erf_29: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_235);  mul_235 = None
    add_205: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_236: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_234, add_205);  mul_234 = add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_146: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_236);  mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_29 = torch.ops.aten.split.Tensor(clone_146, 768, -1);  clone_146 = None
    getitem_176: "f32[8, 196, 768]" = split_29[0]
    getitem_177: "f32[8, 196, 768]" = split_29[1];  split_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_147: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_177, memory_format = torch.contiguous_format);  getitem_177 = None
    var_mean_59 = torch.ops.aten.var_mean.correction(clone_147, [2], correction = 0, keepdim = True)
    getitem_178: "f32[8, 196, 1]" = var_mean_59[0]
    getitem_179: "f32[8, 196, 1]" = var_mean_59[1];  var_mean_59 = None
    add_206: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
    rsqrt_59: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
    sub_59: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_147, getitem_179);  clone_147 = getitem_179 = None
    mul_237: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    mul_238: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_237, primals_297)
    add_207: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_238, primals_298);  mul_238 = primals_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_147: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_207, [0, 2, 1]);  add_207 = None
    permute_148: "f32[196, 196]" = torch.ops.aten.permute.default(primals_299, [1, 0]);  primals_299 = None
    clone_148: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    view_177: "f32[6144, 196]" = torch.ops.aten.view.default(clone_148, [6144, 196]);  clone_148 = None
    mm_29: "f32[6144, 196]" = torch.ops.aten.mm.default(view_177, permute_148)
    view_178: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_29, [8, 768, 196])
    add_208: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_178, primals_300);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_149: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_208, [0, 2, 1]);  add_208 = None
    mul_239: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_176, permute_149);  permute_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_179: "f32[1568, 768]" = torch.ops.aten.view.default(mul_239, [1568, 768]);  mul_239 = None
    permute_150: "f32[768, 256]" = torch.ops.aten.permute.default(primals_301, [1, 0]);  primals_301 = None
    addmm_59: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_302, view_179, permute_150);  primals_302 = None
    view_180: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_59, [8, 196, 256]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_149: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_209: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_202, clone_149);  add_202 = clone_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    clone_150: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_209, memory_format = torch.contiguous_format);  add_209 = None
    var_mean_60 = torch.ops.aten.var_mean.correction(clone_150, [2], correction = 0, keepdim = True)
    getitem_180: "f32[8, 196, 1]" = var_mean_60[0]
    getitem_181: "f32[8, 196, 1]" = var_mean_60[1];  var_mean_60 = None
    add_210: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-06);  getitem_180 = None
    rsqrt_60: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_60: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_150, getitem_181);  clone_150 = getitem_181 = None
    mul_240: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    mul_241: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_240, primals_303)
    add_211: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_241, primals_304);  mul_241 = primals_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    mean: "f32[8, 256]" = torch.ops.aten.mean.dim(add_211, [1]);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:272, code: x = self.head_drop(x)
    clone_151: "f32[8, 256]" = torch.ops.aten.clone.default(mean);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    permute_151: "f32[256, 1000]" = torch.ops.aten.permute.default(primals_305, [1, 0]);  primals_305 = None
    addmm_60: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_306, clone_151, permute_151);  primals_306 = None
    permute_152: "f32[1000, 256]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    div_1: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_60, 256);  rsqrt_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_156: "f32[256, 768]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_163: "f32[196, 196]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_2: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_59, 768);  rsqrt_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_166: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_3: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_58, 256);  rsqrt_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_170: "f32[256, 768]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_177: "f32[196, 196]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_4: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_57, 768);  rsqrt_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_180: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_5: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_56, 256);  rsqrt_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_184: "f32[256, 768]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_191: "f32[196, 196]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_6: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_55, 768);  rsqrt_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_194: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_7: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_54, 256);  rsqrt_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_198: "f32[256, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_205: "f32[196, 196]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_8: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_53, 768);  rsqrt_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_208: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_9: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_52, 256);  rsqrt_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_212: "f32[256, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_219: "f32[196, 196]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_10: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_51, 768);  rsqrt_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_222: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_11: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 256);  rsqrt_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_226: "f32[256, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_233: "f32[196, 196]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_12: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 768);  rsqrt_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_236: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_13: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 256);  rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_240: "f32[256, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_247: "f32[196, 196]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_14: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 768);  rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_250: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_15: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 256);  rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_254: "f32[256, 768]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_261: "f32[196, 196]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_16: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 768);  rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_264: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_17: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 256);  rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_268: "f32[256, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_275: "f32[196, 196]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_18: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 768);  rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_278: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_19: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 256);  rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_282: "f32[256, 768]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_289: "f32[196, 196]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_20: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 768);  rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_292: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_21: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 256);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_296: "f32[256, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_303: "f32[196, 196]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_22: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 768);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_306: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_23: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 256);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_310: "f32[256, 768]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_317: "f32[196, 196]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_24: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 768);  rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_320: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_25: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 256);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_324: "f32[256, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_331: "f32[196, 196]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_26: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 768);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_334: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_27: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 256);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_338: "f32[256, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_345: "f32[196, 196]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_28: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 768);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_348: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_29: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 256);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_352: "f32[256, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_359: "f32[196, 196]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_30: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 768);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_362: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_31: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 256);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_366: "f32[256, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_373: "f32[196, 196]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_32: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 768);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_376: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_33: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 256);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_380: "f32[256, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_387: "f32[196, 196]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_34: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 768);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_390: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_35: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 256);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_394: "f32[256, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_401: "f32[196, 196]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_36: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_404: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_37: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 256);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_408: "f32[256, 768]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_415: "f32[196, 196]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_38: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_418: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_39: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 256);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_422: "f32[256, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_429: "f32[196, 196]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_40: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_432: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_41: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 256);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_436: "f32[256, 768]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_443: "f32[196, 196]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_42: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_446: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_43: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 256);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_450: "f32[256, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_457: "f32[196, 196]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_44: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_460: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_45: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 256);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_464: "f32[256, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_471: "f32[196, 196]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_46: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_474: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_47: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 256);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_478: "f32[256, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_485: "f32[196, 196]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_48: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_488: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_49: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 256);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_492: "f32[256, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_499: "f32[196, 196]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_50: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_502: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_51: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 256);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_506: "f32[256, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_513: "f32[196, 196]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_52: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_516: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_53: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_520: "f32[256, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_527: "f32[196, 196]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_54: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_530: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_55: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_534: "f32[256, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_541: "f32[196, 196]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_56: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_544: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_57: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 256);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_548: "f32[256, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_555: "f32[196, 196]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_58: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_558: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_59: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 256);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    permute_562: "f32[256, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_569: "f32[196, 196]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    div_60: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    permute_572: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    div_61: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt, 256);  rsqrt = None
    return [addmm_60, primals_1, primals_3, primals_7, primals_10, primals_13, primals_17, primals_20, primals_23, primals_27, primals_30, primals_33, primals_37, primals_40, primals_43, primals_47, primals_50, primals_53, primals_57, primals_60, primals_63, primals_67, primals_70, primals_73, primals_77, primals_80, primals_83, primals_87, primals_90, primals_93, primals_97, primals_100, primals_103, primals_107, primals_110, primals_113, primals_117, primals_120, primals_123, primals_127, primals_130, primals_133, primals_137, primals_140, primals_143, primals_147, primals_150, primals_153, primals_157, primals_160, primals_163, primals_167, primals_170, primals_173, primals_177, primals_180, primals_183, primals_187, primals_190, primals_193, primals_197, primals_200, primals_203, primals_207, primals_210, primals_213, primals_217, primals_220, primals_223, primals_227, primals_230, primals_233, primals_237, primals_240, primals_243, primals_247, primals_250, primals_253, primals_257, primals_260, primals_263, primals_267, primals_270, primals_273, primals_277, primals_280, primals_283, primals_287, primals_290, primals_293, primals_297, primals_300, primals_303, primals_307, mul, view_1, addmm, getitem_2, mul_5, view_3, mm, view_5, mul_8, view_7, addmm_2, getitem_8, mul_13, view_9, mm_1, view_11, mul_16, view_13, addmm_4, getitem_14, mul_21, view_15, mm_2, view_17, mul_24, view_19, addmm_6, getitem_20, mul_29, view_21, mm_3, view_23, mul_32, view_25, addmm_8, getitem_26, mul_37, view_27, mm_4, view_29, mul_40, view_31, addmm_10, getitem_32, mul_45, view_33, mm_5, view_35, mul_48, view_37, addmm_12, getitem_38, mul_53, view_39, mm_6, view_41, mul_56, view_43, addmm_14, getitem_44, mul_61, view_45, mm_7, view_47, mul_64, view_49, addmm_16, getitem_50, mul_69, view_51, mm_8, view_53, mul_72, view_55, addmm_18, getitem_56, mul_77, view_57, mm_9, view_59, mul_80, view_61, addmm_20, getitem_62, mul_85, view_63, mm_10, view_65, mul_88, view_67, addmm_22, getitem_68, mul_93, view_69, mm_11, view_71, mul_96, view_73, addmm_24, getitem_74, mul_101, view_75, mm_12, view_77, mul_104, view_79, addmm_26, getitem_80, mul_109, view_81, mm_13, view_83, mul_112, view_85, addmm_28, getitem_86, mul_117, view_87, mm_14, view_89, mul_120, view_91, addmm_30, getitem_92, mul_125, view_93, mm_15, view_95, mul_128, view_97, addmm_32, getitem_98, mul_133, view_99, mm_16, view_101, mul_136, view_103, addmm_34, getitem_104, mul_141, view_105, mm_17, view_107, mul_144, view_109, addmm_36, getitem_110, mul_149, view_111, mm_18, view_113, mul_152, view_115, addmm_38, getitem_116, mul_157, view_117, mm_19, view_119, mul_160, view_121, addmm_40, getitem_122, mul_165, view_123, mm_20, view_125, mul_168, view_127, addmm_42, getitem_128, mul_173, view_129, mm_21, view_131, mul_176, view_133, addmm_44, getitem_134, mul_181, view_135, mm_22, view_137, mul_184, view_139, addmm_46, getitem_140, mul_189, view_141, mm_23, view_143, mul_192, view_145, addmm_48, getitem_146, mul_197, view_147, mm_24, view_149, mul_200, view_151, addmm_50, getitem_152, mul_205, view_153, mm_25, view_155, mul_208, view_157, addmm_52, getitem_158, mul_213, view_159, mm_26, view_161, mul_216, view_163, addmm_54, getitem_164, mul_221, view_165, mm_27, view_167, mul_224, view_169, addmm_56, getitem_170, mul_229, view_171, mm_28, view_173, mul_232, view_175, addmm_58, getitem_176, mul_237, view_177, mm_29, view_179, mul_240, clone_151, permute_152, div_1, permute_156, permute_163, div_2, permute_166, div_3, permute_170, permute_177, div_4, permute_180, div_5, permute_184, permute_191, div_6, permute_194, div_7, permute_198, permute_205, div_8, permute_208, div_9, permute_212, permute_219, div_10, permute_222, div_11, permute_226, permute_233, div_12, permute_236, div_13, permute_240, permute_247, div_14, permute_250, div_15, permute_254, permute_261, div_16, permute_264, div_17, permute_268, permute_275, div_18, permute_278, div_19, permute_282, permute_289, div_20, permute_292, div_21, permute_296, permute_303, div_22, permute_306, div_23, permute_310, permute_317, div_24, permute_320, div_25, permute_324, permute_331, div_26, permute_334, div_27, permute_338, permute_345, div_28, permute_348, div_29, permute_352, permute_359, div_30, permute_362, div_31, permute_366, permute_373, div_32, permute_376, div_33, permute_380, permute_387, div_34, permute_390, div_35, permute_394, permute_401, div_36, permute_404, div_37, permute_408, permute_415, div_38, permute_418, div_39, permute_422, permute_429, div_40, permute_432, div_41, permute_436, permute_443, div_42, permute_446, div_43, permute_450, permute_457, div_44, permute_460, div_45, permute_464, permute_471, div_46, permute_474, div_47, permute_478, permute_485, div_48, permute_488, div_49, permute_492, permute_499, div_50, permute_502, div_51, permute_506, permute_513, div_52, permute_516, div_53, permute_520, permute_527, div_54, permute_530, div_55, permute_534, permute_541, div_56, permute_544, div_57, permute_548, permute_555, div_58, permute_558, div_59, permute_562, permute_569, div_60, permute_572, div_61]
    