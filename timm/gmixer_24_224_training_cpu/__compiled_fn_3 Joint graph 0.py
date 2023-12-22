from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[384, 3, 16, 16]"; primals_2: "f32[384]"; primals_3: "f32[384]"; primals_4: "f32[384]"; primals_5: "f32[384, 196]"; primals_6: "f32[384]"; primals_7: "f32[196, 192]"; primals_8: "f32[196]"; primals_9: "f32[384]"; primals_10: "f32[384]"; primals_11: "f32[1536, 384]"; primals_12: "f32[1536]"; primals_13: "f32[384, 768]"; primals_14: "f32[384]"; primals_15: "f32[384]"; primals_16: "f32[384]"; primals_17: "f32[384, 196]"; primals_18: "f32[384]"; primals_19: "f32[196, 192]"; primals_20: "f32[196]"; primals_21: "f32[384]"; primals_22: "f32[384]"; primals_23: "f32[1536, 384]"; primals_24: "f32[1536]"; primals_25: "f32[384, 768]"; primals_26: "f32[384]"; primals_27: "f32[384]"; primals_28: "f32[384]"; primals_29: "f32[384, 196]"; primals_30: "f32[384]"; primals_31: "f32[196, 192]"; primals_32: "f32[196]"; primals_33: "f32[384]"; primals_34: "f32[384]"; primals_35: "f32[1536, 384]"; primals_36: "f32[1536]"; primals_37: "f32[384, 768]"; primals_38: "f32[384]"; primals_39: "f32[384]"; primals_40: "f32[384]"; primals_41: "f32[384, 196]"; primals_42: "f32[384]"; primals_43: "f32[196, 192]"; primals_44: "f32[196]"; primals_45: "f32[384]"; primals_46: "f32[384]"; primals_47: "f32[1536, 384]"; primals_48: "f32[1536]"; primals_49: "f32[384, 768]"; primals_50: "f32[384]"; primals_51: "f32[384]"; primals_52: "f32[384]"; primals_53: "f32[384, 196]"; primals_54: "f32[384]"; primals_55: "f32[196, 192]"; primals_56: "f32[196]"; primals_57: "f32[384]"; primals_58: "f32[384]"; primals_59: "f32[1536, 384]"; primals_60: "f32[1536]"; primals_61: "f32[384, 768]"; primals_62: "f32[384]"; primals_63: "f32[384]"; primals_64: "f32[384]"; primals_65: "f32[384, 196]"; primals_66: "f32[384]"; primals_67: "f32[196, 192]"; primals_68: "f32[196]"; primals_69: "f32[384]"; primals_70: "f32[384]"; primals_71: "f32[1536, 384]"; primals_72: "f32[1536]"; primals_73: "f32[384, 768]"; primals_74: "f32[384]"; primals_75: "f32[384]"; primals_76: "f32[384]"; primals_77: "f32[384, 196]"; primals_78: "f32[384]"; primals_79: "f32[196, 192]"; primals_80: "f32[196]"; primals_81: "f32[384]"; primals_82: "f32[384]"; primals_83: "f32[1536, 384]"; primals_84: "f32[1536]"; primals_85: "f32[384, 768]"; primals_86: "f32[384]"; primals_87: "f32[384]"; primals_88: "f32[384]"; primals_89: "f32[384, 196]"; primals_90: "f32[384]"; primals_91: "f32[196, 192]"; primals_92: "f32[196]"; primals_93: "f32[384]"; primals_94: "f32[384]"; primals_95: "f32[1536, 384]"; primals_96: "f32[1536]"; primals_97: "f32[384, 768]"; primals_98: "f32[384]"; primals_99: "f32[384]"; primals_100: "f32[384]"; primals_101: "f32[384, 196]"; primals_102: "f32[384]"; primals_103: "f32[196, 192]"; primals_104: "f32[196]"; primals_105: "f32[384]"; primals_106: "f32[384]"; primals_107: "f32[1536, 384]"; primals_108: "f32[1536]"; primals_109: "f32[384, 768]"; primals_110: "f32[384]"; primals_111: "f32[384]"; primals_112: "f32[384]"; primals_113: "f32[384, 196]"; primals_114: "f32[384]"; primals_115: "f32[196, 192]"; primals_116: "f32[196]"; primals_117: "f32[384]"; primals_118: "f32[384]"; primals_119: "f32[1536, 384]"; primals_120: "f32[1536]"; primals_121: "f32[384, 768]"; primals_122: "f32[384]"; primals_123: "f32[384]"; primals_124: "f32[384]"; primals_125: "f32[384, 196]"; primals_126: "f32[384]"; primals_127: "f32[196, 192]"; primals_128: "f32[196]"; primals_129: "f32[384]"; primals_130: "f32[384]"; primals_131: "f32[1536, 384]"; primals_132: "f32[1536]"; primals_133: "f32[384, 768]"; primals_134: "f32[384]"; primals_135: "f32[384]"; primals_136: "f32[384]"; primals_137: "f32[384, 196]"; primals_138: "f32[384]"; primals_139: "f32[196, 192]"; primals_140: "f32[196]"; primals_141: "f32[384]"; primals_142: "f32[384]"; primals_143: "f32[1536, 384]"; primals_144: "f32[1536]"; primals_145: "f32[384, 768]"; primals_146: "f32[384]"; primals_147: "f32[384]"; primals_148: "f32[384]"; primals_149: "f32[384, 196]"; primals_150: "f32[384]"; primals_151: "f32[196, 192]"; primals_152: "f32[196]"; primals_153: "f32[384]"; primals_154: "f32[384]"; primals_155: "f32[1536, 384]"; primals_156: "f32[1536]"; primals_157: "f32[384, 768]"; primals_158: "f32[384]"; primals_159: "f32[384]"; primals_160: "f32[384]"; primals_161: "f32[384, 196]"; primals_162: "f32[384]"; primals_163: "f32[196, 192]"; primals_164: "f32[196]"; primals_165: "f32[384]"; primals_166: "f32[384]"; primals_167: "f32[1536, 384]"; primals_168: "f32[1536]"; primals_169: "f32[384, 768]"; primals_170: "f32[384]"; primals_171: "f32[384]"; primals_172: "f32[384]"; primals_173: "f32[384, 196]"; primals_174: "f32[384]"; primals_175: "f32[196, 192]"; primals_176: "f32[196]"; primals_177: "f32[384]"; primals_178: "f32[384]"; primals_179: "f32[1536, 384]"; primals_180: "f32[1536]"; primals_181: "f32[384, 768]"; primals_182: "f32[384]"; primals_183: "f32[384]"; primals_184: "f32[384]"; primals_185: "f32[384, 196]"; primals_186: "f32[384]"; primals_187: "f32[196, 192]"; primals_188: "f32[196]"; primals_189: "f32[384]"; primals_190: "f32[384]"; primals_191: "f32[1536, 384]"; primals_192: "f32[1536]"; primals_193: "f32[384, 768]"; primals_194: "f32[384]"; primals_195: "f32[384]"; primals_196: "f32[384]"; primals_197: "f32[384, 196]"; primals_198: "f32[384]"; primals_199: "f32[196, 192]"; primals_200: "f32[196]"; primals_201: "f32[384]"; primals_202: "f32[384]"; primals_203: "f32[1536, 384]"; primals_204: "f32[1536]"; primals_205: "f32[384, 768]"; primals_206: "f32[384]"; primals_207: "f32[384]"; primals_208: "f32[384]"; primals_209: "f32[384, 196]"; primals_210: "f32[384]"; primals_211: "f32[196, 192]"; primals_212: "f32[196]"; primals_213: "f32[384]"; primals_214: "f32[384]"; primals_215: "f32[1536, 384]"; primals_216: "f32[1536]"; primals_217: "f32[384, 768]"; primals_218: "f32[384]"; primals_219: "f32[384]"; primals_220: "f32[384]"; primals_221: "f32[384, 196]"; primals_222: "f32[384]"; primals_223: "f32[196, 192]"; primals_224: "f32[196]"; primals_225: "f32[384]"; primals_226: "f32[384]"; primals_227: "f32[1536, 384]"; primals_228: "f32[1536]"; primals_229: "f32[384, 768]"; primals_230: "f32[384]"; primals_231: "f32[384]"; primals_232: "f32[384]"; primals_233: "f32[384, 196]"; primals_234: "f32[384]"; primals_235: "f32[196, 192]"; primals_236: "f32[196]"; primals_237: "f32[384]"; primals_238: "f32[384]"; primals_239: "f32[1536, 384]"; primals_240: "f32[1536]"; primals_241: "f32[384, 768]"; primals_242: "f32[384]"; primals_243: "f32[384]"; primals_244: "f32[384]"; primals_245: "f32[384, 196]"; primals_246: "f32[384]"; primals_247: "f32[196, 192]"; primals_248: "f32[196]"; primals_249: "f32[384]"; primals_250: "f32[384]"; primals_251: "f32[1536, 384]"; primals_252: "f32[1536]"; primals_253: "f32[384, 768]"; primals_254: "f32[384]"; primals_255: "f32[384]"; primals_256: "f32[384]"; primals_257: "f32[384, 196]"; primals_258: "f32[384]"; primals_259: "f32[196, 192]"; primals_260: "f32[196]"; primals_261: "f32[384]"; primals_262: "f32[384]"; primals_263: "f32[1536, 384]"; primals_264: "f32[1536]"; primals_265: "f32[384, 768]"; primals_266: "f32[384]"; primals_267: "f32[384]"; primals_268: "f32[384]"; primals_269: "f32[384, 196]"; primals_270: "f32[384]"; primals_271: "f32[196, 192]"; primals_272: "f32[196]"; primals_273: "f32[384]"; primals_274: "f32[384]"; primals_275: "f32[1536, 384]"; primals_276: "f32[1536]"; primals_277: "f32[384, 768]"; primals_278: "f32[384]"; primals_279: "f32[384]"; primals_280: "f32[384]"; primals_281: "f32[384, 196]"; primals_282: "f32[384]"; primals_283: "f32[196, 192]"; primals_284: "f32[196]"; primals_285: "f32[384]"; primals_286: "f32[384]"; primals_287: "f32[1536, 384]"; primals_288: "f32[1536]"; primals_289: "f32[384, 768]"; primals_290: "f32[384]"; primals_291: "f32[384]"; primals_292: "f32[384]"; primals_293: "f32[1000, 384]"; primals_294: "f32[1000]"; primals_295: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
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
    sub: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = None
    mul: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul, primals_3);  mul = None
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
    mul_3: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_2, mul_2)
    
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
    add_3: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(permute, permute_4);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_4: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_3, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_4, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_5: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_1: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_4, getitem_5);  clone_4 = None
    mul_4: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_5: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_4, primals_9);  mul_4 = None
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
    mul_7: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_6, mul_6)
    
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
    add_6: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_3, clone_6);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_7: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_6, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_7, [2], correction = 0, keepdim = True)
    getitem_8: "f32[8, 196, 1]" = var_mean_2[0]
    getitem_9: "f32[8, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    add_7: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_2: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_2: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_7, getitem_9);  clone_7 = None
    mul_8: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_9: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_8, primals_15);  mul_8 = None
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
    mul_11: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_10, mul_10)
    
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
    add_10: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_6, permute_10);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_11: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_10, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_11, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 196, 1]" = var_mean_3[0]
    getitem_13: "f32[8, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_3: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_3: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_11, getitem_13);  clone_11 = None
    mul_12: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_13: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_12, primals_21);  mul_12 = None
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
    mul_15: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_14, mul_14)
    
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
    add_13: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_10, clone_13);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_14: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_13, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_14, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 196, 1]" = var_mean_4[0]
    getitem_17: "f32[8, 196, 1]" = var_mean_4[1];  var_mean_4 = None
    add_14: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_4: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_4: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_14, getitem_17);  clone_14 = None
    mul_16: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_17: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_16, primals_27);  mul_16 = None
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
    mul_19: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_18, mul_18)
    
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
    add_17: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_13, permute_16);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_18: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_17, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_18, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 196, 1]" = var_mean_5[0]
    getitem_21: "f32[8, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_5: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_5: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_18, getitem_21);  clone_18 = None
    mul_20: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_21: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_20, primals_33);  mul_20 = None
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
    mul_23: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_22, mul_22)
    
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
    add_20: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_17, clone_20);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_21: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_21, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 196, 1]" = var_mean_6[0]
    getitem_25: "f32[8, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_6: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_6: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_21, getitem_25);  clone_21 = None
    mul_24: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_25: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_24, primals_39);  mul_24 = None
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
    mul_27: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_26, mul_26)
    
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
    add_24: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_20, permute_22);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_25: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_24, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_25, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 196, 1]" = var_mean_7[0]
    getitem_29: "f32[8, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_7: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_7: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_25, getitem_29);  clone_25 = None
    mul_28: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_29: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_28, primals_45);  mul_28 = None
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
    mul_31: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_30, mul_30)
    
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
    add_27: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_24, clone_27);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_28: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_27, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_28, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 196, 1]" = var_mean_8[0]
    getitem_33: "f32[8, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_8: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_8: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_28, getitem_33);  clone_28 = None
    mul_32: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_33: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_32, primals_51);  mul_32 = None
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
    mul_35: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_34, mul_34)
    
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
    add_31: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_27, permute_28);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_32: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_31, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_32, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 196, 1]" = var_mean_9[0]
    getitem_37: "f32[8, 196, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_9: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_9: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_32, getitem_37);  clone_32 = None
    mul_36: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_37: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_36, primals_57);  mul_36 = None
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
    mul_39: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_38, mul_38)
    
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
    add_34: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_31, clone_34);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_35: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_34, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_35, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_41: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_10: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_35, getitem_41);  clone_35 = None
    mul_40: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_41: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_40, primals_63);  mul_40 = None
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
    mul_43: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_42, mul_42)
    
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
    add_38: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_34, permute_34);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_39: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_38, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_39, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 196, 1]" = var_mean_11[0]
    getitem_45: "f32[8, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    add_39: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_11: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_11: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_39, getitem_45);  clone_39 = None
    mul_44: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_45: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_44, primals_69);  mul_44 = None
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
    mul_47: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_46, mul_46)
    
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
    add_41: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_38, clone_41);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_42: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 196, 1]" = var_mean_12[0]
    getitem_49: "f32[8, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_12: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_12: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_42, getitem_49);  clone_42 = None
    mul_48: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_49: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_48, primals_75);  mul_48 = None
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
    mul_51: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_50, mul_50)
    
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
    add_45: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_41, permute_40);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_46: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_45, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_46, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 196, 1]" = var_mean_13[0]
    getitem_53: "f32[8, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    add_46: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_13: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_13: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_46, getitem_53);  clone_46 = None
    mul_52: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_53: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_52, primals_81);  mul_52 = None
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
    mul_55: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_54, mul_54)
    
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
    add_48: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_45, clone_48);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_49: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_48, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_49, [2], correction = 0, keepdim = True)
    getitem_56: "f32[8, 196, 1]" = var_mean_14[0]
    getitem_57: "f32[8, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_14: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_14: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_49, getitem_57);  clone_49 = None
    mul_56: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_57: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_56, primals_87);  mul_56 = None
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
    mul_59: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_58, mul_58)
    
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
    add_52: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_48, permute_46);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_53: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_52, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_53, [2], correction = 0, keepdim = True)
    getitem_60: "f32[8, 196, 1]" = var_mean_15[0]
    getitem_61: "f32[8, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    add_53: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
    rsqrt_15: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_15: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_53, getitem_61);  clone_53 = None
    mul_60: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_61: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_60, primals_93);  mul_60 = None
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
    mul_63: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_62, mul_62)
    
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
    add_55: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_52, clone_55);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_56: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_55, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_56, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 196, 1]" = var_mean_16[0]
    getitem_65: "f32[8, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    add_56: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_16: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_16: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_56, getitem_65);  clone_56 = None
    mul_64: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_65: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_64, primals_99);  mul_64 = None
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
    mul_67: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_66, mul_66)
    
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
    add_59: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_55, permute_52);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_60: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_59, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_60, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 196, 1]" = var_mean_17[0]
    getitem_69: "f32[8, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    add_60: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_17: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_17: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_60, getitem_69);  clone_60 = None
    mul_68: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_69: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_68, primals_105);  mul_68 = None
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
    mul_71: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_70, mul_70)
    
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
    add_62: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_59, clone_62);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_63: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_63, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 196, 1]" = var_mean_18[0]
    getitem_73: "f32[8, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    add_63: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_18: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_18: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_63, getitem_73);  clone_63 = None
    mul_72: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_73: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_72, primals_111);  mul_72 = None
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
    mul_75: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_74, mul_74)
    
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
    add_66: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_62, permute_58);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_67: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_66, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_67, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 196, 1]" = var_mean_19[0]
    getitem_77: "f32[8, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    add_67: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_19: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_19: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_67, getitem_77);  clone_67 = None
    mul_76: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_77: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_76, primals_117);  mul_76 = None
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
    mul_79: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_78, mul_78)
    
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
    add_69: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_66, clone_69);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_70: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_69, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_70, [2], correction = 0, keepdim = True)
    getitem_80: "f32[8, 196, 1]" = var_mean_20[0]
    getitem_81: "f32[8, 196, 1]" = var_mean_20[1];  var_mean_20 = None
    add_70: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
    rsqrt_20: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_20: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_70, getitem_81);  clone_70 = None
    mul_80: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_81: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_80, primals_123);  mul_80 = None
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
    mul_83: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_82, mul_82)
    
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
    add_73: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_69, permute_64);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_74: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_73, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_74, [2], correction = 0, keepdim = True)
    getitem_84: "f32[8, 196, 1]" = var_mean_21[0]
    getitem_85: "f32[8, 196, 1]" = var_mean_21[1];  var_mean_21 = None
    add_74: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
    rsqrt_21: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_21: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_74, getitem_85);  clone_74 = None
    mul_84: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_85: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_84, primals_129);  mul_84 = None
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
    mul_87: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_86, mul_86)
    
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
    add_76: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_73, clone_76);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_77: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_77, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 196, 1]" = var_mean_22[0]
    getitem_89: "f32[8, 196, 1]" = var_mean_22[1];  var_mean_22 = None
    add_77: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_22: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_22: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_77, getitem_89);  clone_77 = None
    mul_88: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_89: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_88, primals_135);  mul_88 = None
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
    mul_91: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_90, mul_90)
    
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
    add_80: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_76, permute_70);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_81: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_81, [2], correction = 0, keepdim = True)
    getitem_92: "f32[8, 196, 1]" = var_mean_23[0]
    getitem_93: "f32[8, 196, 1]" = var_mean_23[1];  var_mean_23 = None
    add_81: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
    rsqrt_23: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_23: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_81, getitem_93);  clone_81 = None
    mul_92: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_93: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_92, primals_141);  mul_92 = None
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
    mul_95: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_94, mul_94)
    
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
    add_83: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_80, clone_83);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_84: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_83, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_84, [2], correction = 0, keepdim = True)
    getitem_96: "f32[8, 196, 1]" = var_mean_24[0]
    getitem_97: "f32[8, 196, 1]" = var_mean_24[1];  var_mean_24 = None
    add_84: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
    rsqrt_24: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_24: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_84, getitem_97);  clone_84 = None
    mul_96: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_97: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_96, primals_147);  mul_96 = None
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
    mul_99: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_98, mul_98)
    
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
    add_87: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_83, permute_76);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_88: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_87, memory_format = torch.contiguous_format)
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_88, [2], correction = 0, keepdim = True)
    getitem_100: "f32[8, 196, 1]" = var_mean_25[0]
    getitem_101: "f32[8, 196, 1]" = var_mean_25[1];  var_mean_25 = None
    add_88: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-06);  getitem_100 = None
    rsqrt_25: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_25: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_88, getitem_101);  clone_88 = None
    mul_100: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    mul_101: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_100, primals_153);  mul_100 = None
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
    mul_103: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_102, mul_102)
    
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
    add_90: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_87, clone_90);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_91: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_91, [2], correction = 0, keepdim = True)
    getitem_104: "f32[8, 196, 1]" = var_mean_26[0]
    getitem_105: "f32[8, 196, 1]" = var_mean_26[1];  var_mean_26 = None
    add_91: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
    rsqrt_26: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_26: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_91, getitem_105);  clone_91 = None
    mul_104: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    mul_105: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_104, primals_159);  mul_104 = None
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
    mul_107: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_106, mul_106)
    
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
    add_94: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_90, permute_82);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_95: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_95, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 196, 1]" = var_mean_27[0]
    getitem_109: "f32[8, 196, 1]" = var_mean_27[1];  var_mean_27 = None
    add_95: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_27: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_27: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_95, getitem_109);  clone_95 = None
    mul_108: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    mul_109: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_108, primals_165);  mul_108 = None
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
    mul_111: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_110, mul_110)
    
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
    add_97: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_94, clone_97);  clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_98: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_97, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_98, [2], correction = 0, keepdim = True)
    getitem_112: "f32[8, 196, 1]" = var_mean_28[0]
    getitem_113: "f32[8, 196, 1]" = var_mean_28[1];  var_mean_28 = None
    add_98: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
    rsqrt_28: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_28: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_98, getitem_113);  clone_98 = None
    mul_112: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    mul_113: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_112, primals_171);  mul_112 = None
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
    mul_115: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_114, mul_114)
    
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
    add_101: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_97, permute_88);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_102: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_101, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_102, [2], correction = 0, keepdim = True)
    getitem_116: "f32[8, 196, 1]" = var_mean_29[0]
    getitem_117: "f32[8, 196, 1]" = var_mean_29[1];  var_mean_29 = None
    add_102: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-06);  getitem_116 = None
    rsqrt_29: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_29: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_102, getitem_117);  clone_102 = None
    mul_116: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    mul_117: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_116, primals_177);  mul_116 = None
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
    mul_119: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_118, mul_118)
    
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
    add_104: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_101, clone_104);  clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_105: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_104, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_105, [2], correction = 0, keepdim = True)
    getitem_120: "f32[8, 196, 1]" = var_mean_30[0]
    getitem_121: "f32[8, 196, 1]" = var_mean_30[1];  var_mean_30 = None
    add_105: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
    rsqrt_30: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_30: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_105, getitem_121);  clone_105 = None
    mul_120: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    mul_121: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_120, primals_183);  mul_120 = None
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
    mul_123: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_122, mul_122)
    
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
    add_108: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_104, permute_94);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_109: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_108, memory_format = torch.contiguous_format)
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_109, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 196, 1]" = var_mean_31[0]
    getitem_125: "f32[8, 196, 1]" = var_mean_31[1];  var_mean_31 = None
    add_109: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-06);  getitem_124 = None
    rsqrt_31: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    sub_31: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_109, getitem_125);  clone_109 = None
    mul_124: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    mul_125: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_124, primals_189);  mul_124 = None
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
    mul_127: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_126, mul_126)
    
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
    add_111: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_108, clone_111);  clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_112: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_112, [2], correction = 0, keepdim = True)
    getitem_128: "f32[8, 196, 1]" = var_mean_32[0]
    getitem_129: "f32[8, 196, 1]" = var_mean_32[1];  var_mean_32 = None
    add_112: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
    rsqrt_32: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_32: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_112, getitem_129);  clone_112 = None
    mul_128: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    mul_129: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_128, primals_195);  mul_128 = None
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
    mul_131: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_130, mul_130)
    
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
    add_115: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_111, permute_100);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_116: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_115, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_116, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 196, 1]" = var_mean_33[0]
    getitem_133: "f32[8, 196, 1]" = var_mean_33[1];  var_mean_33 = None
    add_116: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_33: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_33: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_116, getitem_133);  clone_116 = None
    mul_132: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    mul_133: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_132, primals_201);  mul_132 = None
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
    mul_135: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_134, mul_134)
    
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
    add_118: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_115, clone_118);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_119: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_118, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_119, [2], correction = 0, keepdim = True)
    getitem_136: "f32[8, 196, 1]" = var_mean_34[0]
    getitem_137: "f32[8, 196, 1]" = var_mean_34[1];  var_mean_34 = None
    add_119: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-06);  getitem_136 = None
    rsqrt_34: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_34: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_119, getitem_137);  clone_119 = None
    mul_136: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    mul_137: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_136, primals_207);  mul_136 = None
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
    mul_139: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_138, mul_138)
    
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
    add_122: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_118, permute_106);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_123: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_122, memory_format = torch.contiguous_format)
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_123, [2], correction = 0, keepdim = True)
    getitem_140: "f32[8, 196, 1]" = var_mean_35[0]
    getitem_141: "f32[8, 196, 1]" = var_mean_35[1];  var_mean_35 = None
    add_123: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-06);  getitem_140 = None
    rsqrt_35: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_35: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_123, getitem_141);  clone_123 = None
    mul_140: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    mul_141: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_140, primals_213);  mul_140 = None
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
    mul_143: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_142, mul_142)
    
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
    add_125: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_122, clone_125);  clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_126: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_126, [2], correction = 0, keepdim = True)
    getitem_144: "f32[8, 196, 1]" = var_mean_36[0]
    getitem_145: "f32[8, 196, 1]" = var_mean_36[1];  var_mean_36 = None
    add_126: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_36: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_36: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_126, getitem_145);  clone_126 = None
    mul_144: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    mul_145: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_144, primals_219);  mul_144 = None
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
    mul_147: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_146, mul_146)
    
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
    add_129: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_125, permute_112);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_130: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_129, memory_format = torch.contiguous_format)
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_130, [2], correction = 0, keepdim = True)
    getitem_148: "f32[8, 196, 1]" = var_mean_37[0]
    getitem_149: "f32[8, 196, 1]" = var_mean_37[1];  var_mean_37 = None
    add_130: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-06);  getitem_148 = None
    rsqrt_37: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_37: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_130, getitem_149);  clone_130 = None
    mul_148: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    mul_149: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_148, primals_225);  mul_148 = None
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
    mul_151: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_150, mul_150)
    
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
    add_132: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_129, clone_132);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_133: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_132, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_133, [2], correction = 0, keepdim = True)
    getitem_152: "f32[8, 196, 1]" = var_mean_38[0]
    getitem_153: "f32[8, 196, 1]" = var_mean_38[1];  var_mean_38 = None
    add_133: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-06);  getitem_152 = None
    rsqrt_38: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_38: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_133, getitem_153);  clone_133 = None
    mul_152: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    mul_153: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_152, primals_231);  mul_152 = None
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
    mul_155: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_154, mul_154)
    
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
    add_136: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_132, permute_118);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_137: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_136, memory_format = torch.contiguous_format)
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_137, [2], correction = 0, keepdim = True)
    getitem_156: "f32[8, 196, 1]" = var_mean_39[0]
    getitem_157: "f32[8, 196, 1]" = var_mean_39[1];  var_mean_39 = None
    add_137: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
    rsqrt_39: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_39: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_137, getitem_157);  clone_137 = None
    mul_156: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    mul_157: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_156, primals_237);  mul_156 = None
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
    mul_159: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_158, mul_158)
    
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
    add_139: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_136, clone_139);  clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_140: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_139, memory_format = torch.contiguous_format)
    var_mean_40 = torch.ops.aten.var_mean.correction(clone_140, [2], correction = 0, keepdim = True)
    getitem_160: "f32[8, 196, 1]" = var_mean_40[0]
    getitem_161: "f32[8, 196, 1]" = var_mean_40[1];  var_mean_40 = None
    add_140: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
    rsqrt_40: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_40: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_140, getitem_161);  clone_140 = None
    mul_160: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    mul_161: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_160, primals_243);  mul_160 = None
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
    mul_163: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_162, mul_162)
    
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
    add_143: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_139, permute_124);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_144: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_143, memory_format = torch.contiguous_format)
    var_mean_41 = torch.ops.aten.var_mean.correction(clone_144, [2], correction = 0, keepdim = True)
    getitem_164: "f32[8, 196, 1]" = var_mean_41[0]
    getitem_165: "f32[8, 196, 1]" = var_mean_41[1];  var_mean_41 = None
    add_144: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-06);  getitem_164 = None
    rsqrt_41: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_41: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_144, getitem_165);  clone_144 = None
    mul_164: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    mul_165: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_164, primals_249);  mul_164 = None
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
    mul_167: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_166, mul_166)
    
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
    add_146: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_143, clone_146);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_147: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_146, memory_format = torch.contiguous_format)
    var_mean_42 = torch.ops.aten.var_mean.correction(clone_147, [2], correction = 0, keepdim = True)
    getitem_168: "f32[8, 196, 1]" = var_mean_42[0]
    getitem_169: "f32[8, 196, 1]" = var_mean_42[1];  var_mean_42 = None
    add_147: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
    rsqrt_42: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_42: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_147, getitem_169);  clone_147 = None
    mul_168: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    mul_169: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_168, primals_255);  mul_168 = None
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
    mul_171: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_170, mul_170)
    
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
    add_150: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_146, permute_130);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_151: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_150, memory_format = torch.contiguous_format)
    var_mean_43 = torch.ops.aten.var_mean.correction(clone_151, [2], correction = 0, keepdim = True)
    getitem_172: "f32[8, 196, 1]" = var_mean_43[0]
    getitem_173: "f32[8, 196, 1]" = var_mean_43[1];  var_mean_43 = None
    add_151: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-06);  getitem_172 = None
    rsqrt_43: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_43: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_151, getitem_173);  clone_151 = None
    mul_172: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    mul_173: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_172, primals_261);  mul_172 = None
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
    mul_175: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_174, mul_174)
    
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
    add_153: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_150, clone_153);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_154: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_153, memory_format = torch.contiguous_format)
    var_mean_44 = torch.ops.aten.var_mean.correction(clone_154, [2], correction = 0, keepdim = True)
    getitem_176: "f32[8, 196, 1]" = var_mean_44[0]
    getitem_177: "f32[8, 196, 1]" = var_mean_44[1];  var_mean_44 = None
    add_154: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
    rsqrt_44: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_44: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_154, getitem_177);  clone_154 = None
    mul_176: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    mul_177: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_176, primals_267);  mul_176 = None
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
    mul_179: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_178, mul_178)
    
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
    add_157: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_153, permute_136);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_158: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_157, memory_format = torch.contiguous_format)
    var_mean_45 = torch.ops.aten.var_mean.correction(clone_158, [2], correction = 0, keepdim = True)
    getitem_180: "f32[8, 196, 1]" = var_mean_45[0]
    getitem_181: "f32[8, 196, 1]" = var_mean_45[1];  var_mean_45 = None
    add_158: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-06);  getitem_180 = None
    rsqrt_45: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_45: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_158, getitem_181);  clone_158 = None
    mul_180: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    mul_181: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_180, primals_273);  mul_180 = None
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
    mul_183: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_182, mul_182)
    
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
    add_160: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_157, clone_160);  clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_161: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_160, memory_format = torch.contiguous_format)
    var_mean_46 = torch.ops.aten.var_mean.correction(clone_161, [2], correction = 0, keepdim = True)
    getitem_184: "f32[8, 196, 1]" = var_mean_46[0]
    getitem_185: "f32[8, 196, 1]" = var_mean_46[1];  var_mean_46 = None
    add_161: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-06);  getitem_184 = None
    rsqrt_46: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_46: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_161, getitem_185);  clone_161 = None
    mul_184: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    mul_185: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_184, primals_279);  mul_184 = None
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
    mul_187: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_186, mul_186)
    
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
    add_164: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_160, permute_142);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_165: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_164, memory_format = torch.contiguous_format)
    var_mean_47 = torch.ops.aten.var_mean.correction(clone_165, [2], correction = 0, keepdim = True)
    getitem_188: "f32[8, 196, 1]" = var_mean_47[0]
    getitem_189: "f32[8, 196, 1]" = var_mean_47[1];  var_mean_47 = None
    add_165: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-06);  getitem_188 = None
    rsqrt_47: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_47: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_165, getitem_189);  clone_165 = None
    mul_188: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    mul_189: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_188, primals_285);  mul_188 = None
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
    mul_191: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_190, mul_190)
    
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
    add_167: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_164, clone_167);  clone_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    clone_168: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format)
    var_mean_48 = torch.ops.aten.var_mean.correction(clone_168, [2], correction = 0, keepdim = True)
    getitem_192: "f32[8, 196, 1]" = var_mean_48[0]
    getitem_193: "f32[8, 196, 1]" = var_mean_48[1];  var_mean_48 = None
    add_168: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
    rsqrt_48: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_48: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_168, getitem_193);  clone_168 = None
    mul_192: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    mul_193: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_192, primals_291);  mul_192 = None
    add_169: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_193, primals_292);  mul_193 = primals_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    mean: "f32[8, 384]" = torch.ops.aten.mean.dim(add_169, [1]);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:272, code: x = self.head_drop(x)
    clone_169: "f32[8, 384]" = torch.ops.aten.clone.default(mean);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    permute_145: "f32[384, 1000]" = torch.ops.aten.permute.default(primals_293, [1, 0]);  primals_293 = None
    addmm_72: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_294, clone_169, permute_145);  primals_294 = None
    permute_146: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    mm_24: "f32[8, 384]" = torch.ops.aten.mm.default(tangents_1, permute_146);  permute_146 = None
    permute_147: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_25: "f32[1000, 384]" = torch.ops.aten.mm.default(permute_147, clone_169);  permute_147 = clone_169 = None
    permute_148: "f32[384, 1000]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_193: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_149: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    unsqueeze: "f32[8, 1, 384]" = torch.ops.aten.unsqueeze.default(mm_24, 1);  mm_24 = None
    expand: "f32[8, 196, 384]" = torch.ops.aten.expand.default(unsqueeze, [8, 196, 384]);  unsqueeze = None
    div: "f32[8, 196, 384]" = torch.ops.aten.div.Scalar(expand, 196);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    clone_170: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format);  add_167 = None
    sub_49: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_170, getitem_193);  clone_170 = getitem_193 = None
    mul_194: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_48);  sub_49 = None
    mul_195: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div, primals_291);  primals_291 = None
    mul_196: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_195, 384)
    sum_2: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
    mul_197: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_195, mul_194);  mul_195 = None
    sum_3: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
    mul_198: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_194, sum_3);  sum_3 = None
    sub_50: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_196, sum_2);  mul_196 = sum_2 = None
    sub_51: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_50, mul_198);  sub_50 = mul_198 = None
    div_1: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 384);  rsqrt_48 = None
    mul_199: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_1, sub_51);  div_1 = sub_51 = None
    mul_200: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div, mul_194);  mul_194 = None
    sum_4: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
    sum_5: "f32[384]" = torch.ops.aten.sum.dim_IntList(div, [0, 1]);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_194: "f32[1568, 384]" = torch.ops.aten.view.default(mul_199, [1568, 384])
    permute_150: "f32[384, 768]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    mm_26: "f32[1568, 768]" = torch.ops.aten.mm.default(view_194, permute_150);  permute_150 = None
    permute_151: "f32[384, 1568]" = torch.ops.aten.permute.default(view_194, [1, 0])
    mm_27: "f32[384, 768]" = torch.ops.aten.mm.default(permute_151, view_191);  permute_151 = view_191 = None
    permute_152: "f32[768, 384]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_6: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_194, [0], True);  view_194 = None
    view_195: "f32[384]" = torch.ops.aten.view.default(sum_6, [384]);  sum_6 = None
    permute_153: "f32[384, 768]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_196: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_26, [8, 196, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_201: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_196, getitem_190);  getitem_190 = None
    mul_202: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_196, mul_190);  view_196 = mul_190 = None
    sigmoid_48: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_191)
    full: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_52: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full, sigmoid_48);  full = None
    mul_203: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_191, sub_52);  getitem_191 = sub_52 = None
    add_170: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_203, 1);  mul_203 = None
    mul_204: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_48, add_170);  sigmoid_48 = add_170 = None
    mul_205: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_201, mul_204);  mul_201 = mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_202, mul_205], 2);  mul_202 = mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_197: "f32[1568, 1536]" = torch.ops.aten.view.default(cat, [1568, 1536]);  cat = None
    permute_155: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    mm_28: "f32[1568, 384]" = torch.ops.aten.mm.default(view_197, permute_155);  permute_155 = None
    permute_156: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_197, [1, 0])
    mm_29: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_156, view_189);  permute_156 = view_189 = None
    permute_157: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_7: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_197, [0], True);  view_197 = None
    view_198: "f32[1536]" = torch.ops.aten.view.default(sum_7, [1536]);  sum_7 = None
    permute_158: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    view_199: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_28, [8, 196, 384]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_171: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_164, memory_format = torch.contiguous_format);  add_164 = None
    sub_53: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_171, getitem_189);  clone_171 = getitem_189 = None
    mul_206: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_47);  sub_53 = None
    mul_207: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_199, primals_285);  primals_285 = None
    mul_208: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_207, 384)
    sum_8: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_207, [2], True)
    mul_209: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_207, mul_206);  mul_207 = None
    sum_9: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [2], True);  mul_209 = None
    mul_210: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_206, sum_9);  sum_9 = None
    sub_54: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_208, sum_8);  mul_208 = sum_8 = None
    sub_55: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_54, mul_210);  sub_54 = mul_210 = None
    div_2: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 384);  rsqrt_47 = None
    mul_211: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_2, sub_55);  div_2 = sub_55 = None
    mul_212: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_199, mul_206);  mul_206 = None
    sum_10: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_212, [0, 1]);  mul_212 = None
    sum_11: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_199, [0, 1]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_171: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_199, mul_211);  mul_199 = mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_159: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_171, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_172: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    view_200: "f32[3072, 196]" = torch.ops.aten.view.default(clone_172, [3072, 196]);  clone_172 = None
    permute_160: "f32[196, 192]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    mm_30: "f32[3072, 192]" = torch.ops.aten.mm.default(view_200, permute_160);  permute_160 = None
    permute_161: "f32[196, 3072]" = torch.ops.aten.permute.default(view_200, [1, 0])
    mm_31: "f32[196, 192]" = torch.ops.aten.mm.default(permute_161, view_187);  permute_161 = view_187 = None
    permute_162: "f32[192, 196]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_12: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_200, [0], True);  view_200 = None
    view_201: "f32[196]" = torch.ops.aten.view.default(sum_12, [196]);  sum_12 = None
    permute_163: "f32[196, 192]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    view_202: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_30, [8, 384, 192]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_213: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_202, getitem_186);  getitem_186 = None
    mul_214: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_202, mul_186);  view_202 = mul_186 = None
    sigmoid_49: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_187)
    full_1: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_56: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_1, sigmoid_49);  full_1 = None
    mul_215: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_187, sub_56);  getitem_187 = sub_56 = None
    add_172: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_215, 1);  mul_215 = None
    mul_216: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_49, add_172);  sigmoid_49 = add_172 = None
    mul_217: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_213, mul_216);  mul_213 = mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_1: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_214, mul_217], 2);  mul_214 = mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_13: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_1, [0, 1], True)
    view_203: "f32[384]" = torch.ops.aten.view.default(sum_13, [384]);  sum_13 = None
    view_204: "f32[3072, 384]" = torch.ops.aten.view.default(cat_1, [3072, 384]);  cat_1 = None
    permute_165: "f32[384, 3072]" = torch.ops.aten.permute.default(view_204, [1, 0])
    mm_32: "f32[384, 196]" = torch.ops.aten.mm.default(permute_165, view_185);  permute_165 = view_185 = None
    permute_166: "f32[196, 384]" = torch.ops.aten.permute.default(mm_32, [1, 0]);  mm_32 = None
    permute_167: "f32[384, 196]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    mm_33: "f32[3072, 196]" = torch.ops.aten.mm.default(view_204, permute_167);  view_204 = permute_167 = None
    view_205: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_33, [8, 384, 196]);  mm_33 = None
    permute_168: "f32[384, 196]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_169: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    clone_173: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
    clone_174: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_160, memory_format = torch.contiguous_format);  add_160 = None
    sub_57: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_174, getitem_185);  clone_174 = getitem_185 = None
    mul_218: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_46);  sub_57 = None
    mul_219: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_173, primals_279);  primals_279 = None
    mul_220: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_219, 384)
    sum_14: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [2], True)
    mul_221: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_219, mul_218);  mul_219 = None
    sum_15: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [2], True);  mul_221 = None
    mul_222: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_218, sum_15);  sum_15 = None
    sub_58: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_220, sum_14);  mul_220 = sum_14 = None
    sub_59: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_58, mul_222);  sub_58 = mul_222 = None
    div_3: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 384);  rsqrt_46 = None
    mul_223: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_3, sub_59);  div_3 = sub_59 = None
    mul_224: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_173, mul_218);  mul_218 = None
    sum_16: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_224, [0, 1]);  mul_224 = None
    sum_17: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_173, [0, 1]);  clone_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_173: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_171, mul_223);  add_171 = mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_206: "f32[1568, 384]" = torch.ops.aten.view.default(add_173, [1568, 384])
    permute_170: "f32[384, 768]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    mm_34: "f32[1568, 768]" = torch.ops.aten.mm.default(view_206, permute_170);  permute_170 = None
    permute_171: "f32[384, 1568]" = torch.ops.aten.permute.default(view_206, [1, 0])
    mm_35: "f32[384, 768]" = torch.ops.aten.mm.default(permute_171, view_183);  permute_171 = view_183 = None
    permute_172: "f32[768, 384]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_18: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_206, [0], True);  view_206 = None
    view_207: "f32[384]" = torch.ops.aten.view.default(sum_18, [384]);  sum_18 = None
    permute_173: "f32[384, 768]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    view_208: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_34, [8, 196, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_225: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_208, getitem_182);  getitem_182 = None
    mul_226: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_208, mul_182);  view_208 = mul_182 = None
    sigmoid_50: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_183)
    full_2: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_60: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_2, sigmoid_50);  full_2 = None
    mul_227: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_183, sub_60);  getitem_183 = sub_60 = None
    add_174: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_227, 1);  mul_227 = None
    mul_228: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_50, add_174);  sigmoid_50 = add_174 = None
    mul_229: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_225, mul_228);  mul_225 = mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_2: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_226, mul_229], 2);  mul_226 = mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_209: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_2, [1568, 1536]);  cat_2 = None
    permute_175: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    mm_36: "f32[1568, 384]" = torch.ops.aten.mm.default(view_209, permute_175);  permute_175 = None
    permute_176: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_209, [1, 0])
    mm_37: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_176, view_181);  permute_176 = view_181 = None
    permute_177: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_19: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_209, [0], True);  view_209 = None
    view_210: "f32[1536]" = torch.ops.aten.view.default(sum_19, [1536]);  sum_19 = None
    permute_178: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    view_211: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_36, [8, 196, 384]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_175: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_157, memory_format = torch.contiguous_format);  add_157 = None
    sub_61: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_175, getitem_181);  clone_175 = getitem_181 = None
    mul_230: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_45);  sub_61 = None
    mul_231: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_211, primals_273);  primals_273 = None
    mul_232: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_231, 384)
    sum_20: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2], True)
    mul_233: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_231, mul_230);  mul_231 = None
    sum_21: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_233, [2], True);  mul_233 = None
    mul_234: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_230, sum_21);  sum_21 = None
    sub_62: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_232, sum_20);  mul_232 = sum_20 = None
    sub_63: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_62, mul_234);  sub_62 = mul_234 = None
    div_4: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 384);  rsqrt_45 = None
    mul_235: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_4, sub_63);  div_4 = sub_63 = None
    mul_236: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_211, mul_230);  mul_230 = None
    sum_22: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_236, [0, 1]);  mul_236 = None
    sum_23: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_211, [0, 1]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_175: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_173, mul_235);  add_173 = mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_179: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_175, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_176: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    view_212: "f32[3072, 196]" = torch.ops.aten.view.default(clone_176, [3072, 196]);  clone_176 = None
    permute_180: "f32[196, 192]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    mm_38: "f32[3072, 192]" = torch.ops.aten.mm.default(view_212, permute_180);  permute_180 = None
    permute_181: "f32[196, 3072]" = torch.ops.aten.permute.default(view_212, [1, 0])
    mm_39: "f32[196, 192]" = torch.ops.aten.mm.default(permute_181, view_179);  permute_181 = view_179 = None
    permute_182: "f32[192, 196]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_24: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_212, [0], True);  view_212 = None
    view_213: "f32[196]" = torch.ops.aten.view.default(sum_24, [196]);  sum_24 = None
    permute_183: "f32[196, 192]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    view_214: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_38, [8, 384, 192]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_237: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_214, getitem_178);  getitem_178 = None
    mul_238: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_214, mul_178);  view_214 = mul_178 = None
    sigmoid_51: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_179)
    full_3: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_64: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_3, sigmoid_51);  full_3 = None
    mul_239: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_179, sub_64);  getitem_179 = sub_64 = None
    add_176: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_239, 1);  mul_239 = None
    mul_240: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_51, add_176);  sigmoid_51 = add_176 = None
    mul_241: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_237, mul_240);  mul_237 = mul_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_3: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_238, mul_241], 2);  mul_238 = mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_25: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_3, [0, 1], True)
    view_215: "f32[384]" = torch.ops.aten.view.default(sum_25, [384]);  sum_25 = None
    view_216: "f32[3072, 384]" = torch.ops.aten.view.default(cat_3, [3072, 384]);  cat_3 = None
    permute_185: "f32[384, 3072]" = torch.ops.aten.permute.default(view_216, [1, 0])
    mm_40: "f32[384, 196]" = torch.ops.aten.mm.default(permute_185, view_177);  permute_185 = view_177 = None
    permute_186: "f32[196, 384]" = torch.ops.aten.permute.default(mm_40, [1, 0]);  mm_40 = None
    permute_187: "f32[384, 196]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    mm_41: "f32[3072, 196]" = torch.ops.aten.mm.default(view_216, permute_187);  view_216 = permute_187 = None
    view_217: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_41, [8, 384, 196]);  mm_41 = None
    permute_188: "f32[384, 196]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_189: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_217, [0, 2, 1]);  view_217 = None
    clone_177: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    clone_178: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_153, memory_format = torch.contiguous_format);  add_153 = None
    sub_65: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_178, getitem_177);  clone_178 = getitem_177 = None
    mul_242: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_44);  sub_65 = None
    mul_243: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_177, primals_267);  primals_267 = None
    mul_244: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_243, 384)
    sum_26: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True)
    mul_245: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_243, mul_242);  mul_243 = None
    sum_27: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True);  mul_245 = None
    mul_246: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_242, sum_27);  sum_27 = None
    sub_66: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_244, sum_26);  mul_244 = sum_26 = None
    sub_67: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_66, mul_246);  sub_66 = mul_246 = None
    div_5: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 384);  rsqrt_44 = None
    mul_247: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_5, sub_67);  div_5 = sub_67 = None
    mul_248: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_177, mul_242);  mul_242 = None
    sum_28: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1]);  mul_248 = None
    sum_29: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_177, [0, 1]);  clone_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_177: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_175, mul_247);  add_175 = mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_218: "f32[1568, 384]" = torch.ops.aten.view.default(add_177, [1568, 384])
    permute_190: "f32[384, 768]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_42: "f32[1568, 768]" = torch.ops.aten.mm.default(view_218, permute_190);  permute_190 = None
    permute_191: "f32[384, 1568]" = torch.ops.aten.permute.default(view_218, [1, 0])
    mm_43: "f32[384, 768]" = torch.ops.aten.mm.default(permute_191, view_175);  permute_191 = view_175 = None
    permute_192: "f32[768, 384]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_30: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
    view_219: "f32[384]" = torch.ops.aten.view.default(sum_30, [384]);  sum_30 = None
    permute_193: "f32[384, 768]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    view_220: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_42, [8, 196, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_249: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_220, getitem_174);  getitem_174 = None
    mul_250: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_220, mul_174);  view_220 = mul_174 = None
    sigmoid_52: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_175)
    full_4: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_68: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_4, sigmoid_52);  full_4 = None
    mul_251: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_175, sub_68);  getitem_175 = sub_68 = None
    add_178: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_251, 1);  mul_251 = None
    mul_252: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_52, add_178);  sigmoid_52 = add_178 = None
    mul_253: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_249, mul_252);  mul_249 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_4: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_250, mul_253], 2);  mul_250 = mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_221: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_4, [1568, 1536]);  cat_4 = None
    permute_195: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_44: "f32[1568, 384]" = torch.ops.aten.mm.default(view_221, permute_195);  permute_195 = None
    permute_196: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_221, [1, 0])
    mm_45: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_196, view_173);  permute_196 = view_173 = None
    permute_197: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_31: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True);  view_221 = None
    view_222: "f32[1536]" = torch.ops.aten.view.default(sum_31, [1536]);  sum_31 = None
    permute_198: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_223: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_44, [8, 196, 384]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_179: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_150, memory_format = torch.contiguous_format);  add_150 = None
    sub_69: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_179, getitem_173);  clone_179 = getitem_173 = None
    mul_254: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_43);  sub_69 = None
    mul_255: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_223, primals_261);  primals_261 = None
    mul_256: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_255, 384)
    sum_32: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True)
    mul_257: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_255, mul_254);  mul_255 = None
    sum_33: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
    mul_258: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_254, sum_33);  sum_33 = None
    sub_70: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_256, sum_32);  mul_256 = sum_32 = None
    sub_71: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_70, mul_258);  sub_70 = mul_258 = None
    div_6: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 384);  rsqrt_43 = None
    mul_259: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_6, sub_71);  div_6 = sub_71 = None
    mul_260: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_223, mul_254);  mul_254 = None
    sum_34: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 1]);  mul_260 = None
    sum_35: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_223, [0, 1]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_179: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_177, mul_259);  add_177 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_199: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_179, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_180: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_224: "f32[3072, 196]" = torch.ops.aten.view.default(clone_180, [3072, 196]);  clone_180 = None
    permute_200: "f32[196, 192]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    mm_46: "f32[3072, 192]" = torch.ops.aten.mm.default(view_224, permute_200);  permute_200 = None
    permute_201: "f32[196, 3072]" = torch.ops.aten.permute.default(view_224, [1, 0])
    mm_47: "f32[196, 192]" = torch.ops.aten.mm.default(permute_201, view_171);  permute_201 = view_171 = None
    permute_202: "f32[192, 196]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_36: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
    view_225: "f32[196]" = torch.ops.aten.view.default(sum_36, [196]);  sum_36 = None
    permute_203: "f32[196, 192]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    view_226: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_46, [8, 384, 192]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_261: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_226, getitem_170);  getitem_170 = None
    mul_262: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_226, mul_170);  view_226 = mul_170 = None
    sigmoid_53: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_171)
    full_5: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_72: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_5, sigmoid_53);  full_5 = None
    mul_263: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_171, sub_72);  getitem_171 = sub_72 = None
    add_180: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_263, 1);  mul_263 = None
    mul_264: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_53, add_180);  sigmoid_53 = add_180 = None
    mul_265: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_261, mul_264);  mul_261 = mul_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_5: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_262, mul_265], 2);  mul_262 = mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_37: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_5, [0, 1], True)
    view_227: "f32[384]" = torch.ops.aten.view.default(sum_37, [384]);  sum_37 = None
    view_228: "f32[3072, 384]" = torch.ops.aten.view.default(cat_5, [3072, 384]);  cat_5 = None
    permute_205: "f32[384, 3072]" = torch.ops.aten.permute.default(view_228, [1, 0])
    mm_48: "f32[384, 196]" = torch.ops.aten.mm.default(permute_205, view_169);  permute_205 = view_169 = None
    permute_206: "f32[196, 384]" = torch.ops.aten.permute.default(mm_48, [1, 0]);  mm_48 = None
    permute_207: "f32[384, 196]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    mm_49: "f32[3072, 196]" = torch.ops.aten.mm.default(view_228, permute_207);  view_228 = permute_207 = None
    view_229: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_49, [8, 384, 196]);  mm_49 = None
    permute_208: "f32[384, 196]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_209: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
    clone_181: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
    clone_182: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_146, memory_format = torch.contiguous_format);  add_146 = None
    sub_73: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_182, getitem_169);  clone_182 = getitem_169 = None
    mul_266: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_42);  sub_73 = None
    mul_267: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_181, primals_255);  primals_255 = None
    mul_268: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_267, 384)
    sum_38: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
    mul_269: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_267, mul_266);  mul_267 = None
    sum_39: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
    mul_270: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_266, sum_39);  sum_39 = None
    sub_74: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_268, sum_38);  mul_268 = sum_38 = None
    sub_75: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_74, mul_270);  sub_74 = mul_270 = None
    div_7: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 384);  rsqrt_42 = None
    mul_271: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_7, sub_75);  div_7 = sub_75 = None
    mul_272: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_181, mul_266);  mul_266 = None
    sum_40: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None
    sum_41: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_181, [0, 1]);  clone_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_181: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_179, mul_271);  add_179 = mul_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_230: "f32[1568, 384]" = torch.ops.aten.view.default(add_181, [1568, 384])
    permute_210: "f32[384, 768]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    mm_50: "f32[1568, 768]" = torch.ops.aten.mm.default(view_230, permute_210);  permute_210 = None
    permute_211: "f32[384, 1568]" = torch.ops.aten.permute.default(view_230, [1, 0])
    mm_51: "f32[384, 768]" = torch.ops.aten.mm.default(permute_211, view_167);  permute_211 = view_167 = None
    permute_212: "f32[768, 384]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_42: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_230, [0], True);  view_230 = None
    view_231: "f32[384]" = torch.ops.aten.view.default(sum_42, [384]);  sum_42 = None
    permute_213: "f32[384, 768]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
    view_232: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_50, [8, 196, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_273: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_232, getitem_166);  getitem_166 = None
    mul_274: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_232, mul_166);  view_232 = mul_166 = None
    sigmoid_54: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_167)
    full_6: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_76: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_6, sigmoid_54);  full_6 = None
    mul_275: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_167, sub_76);  getitem_167 = sub_76 = None
    add_182: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_275, 1);  mul_275 = None
    mul_276: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_54, add_182);  sigmoid_54 = add_182 = None
    mul_277: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_273, mul_276);  mul_273 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_6: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_274, mul_277], 2);  mul_274 = mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_233: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_6, [1568, 1536]);  cat_6 = None
    permute_215: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    mm_52: "f32[1568, 384]" = torch.ops.aten.mm.default(view_233, permute_215);  permute_215 = None
    permute_216: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_233, [1, 0])
    mm_53: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_216, view_165);  permute_216 = view_165 = None
    permute_217: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_43: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True);  view_233 = None
    view_234: "f32[1536]" = torch.ops.aten.view.default(sum_43, [1536]);  sum_43 = None
    permute_218: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
    view_235: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_52, [8, 196, 384]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_183: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_143, memory_format = torch.contiguous_format);  add_143 = None
    sub_77: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_183, getitem_165);  clone_183 = getitem_165 = None
    mul_278: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_41);  sub_77 = None
    mul_279: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_235, primals_249);  primals_249 = None
    mul_280: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_279, 384)
    sum_44: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True)
    mul_281: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_279, mul_278);  mul_279 = None
    sum_45: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_281, [2], True);  mul_281 = None
    mul_282: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_278, sum_45);  sum_45 = None
    sub_78: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_280, sum_44);  mul_280 = sum_44 = None
    sub_79: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_78, mul_282);  sub_78 = mul_282 = None
    div_8: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 384);  rsqrt_41 = None
    mul_283: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_8, sub_79);  div_8 = sub_79 = None
    mul_284: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_235, mul_278);  mul_278 = None
    sum_46: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_284, [0, 1]);  mul_284 = None
    sum_47: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_235, [0, 1]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_183: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_181, mul_283);  add_181 = mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_219: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_183, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_184: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
    view_236: "f32[3072, 196]" = torch.ops.aten.view.default(clone_184, [3072, 196]);  clone_184 = None
    permute_220: "f32[196, 192]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    mm_54: "f32[3072, 192]" = torch.ops.aten.mm.default(view_236, permute_220);  permute_220 = None
    permute_221: "f32[196, 3072]" = torch.ops.aten.permute.default(view_236, [1, 0])
    mm_55: "f32[196, 192]" = torch.ops.aten.mm.default(permute_221, view_163);  permute_221 = view_163 = None
    permute_222: "f32[192, 196]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_48: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_236, [0], True);  view_236 = None
    view_237: "f32[196]" = torch.ops.aten.view.default(sum_48, [196]);  sum_48 = None
    permute_223: "f32[196, 192]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    view_238: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_54, [8, 384, 192]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_285: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_238, getitem_162);  getitem_162 = None
    mul_286: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_238, mul_162);  view_238 = mul_162 = None
    sigmoid_55: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_163)
    full_7: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_80: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_7, sigmoid_55);  full_7 = None
    mul_287: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_163, sub_80);  getitem_163 = sub_80 = None
    add_184: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_287, 1);  mul_287 = None
    mul_288: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_55, add_184);  sigmoid_55 = add_184 = None
    mul_289: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_285, mul_288);  mul_285 = mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_7: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_286, mul_289], 2);  mul_286 = mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_49: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_7, [0, 1], True)
    view_239: "f32[384]" = torch.ops.aten.view.default(sum_49, [384]);  sum_49 = None
    view_240: "f32[3072, 384]" = torch.ops.aten.view.default(cat_7, [3072, 384]);  cat_7 = None
    permute_225: "f32[384, 3072]" = torch.ops.aten.permute.default(view_240, [1, 0])
    mm_56: "f32[384, 196]" = torch.ops.aten.mm.default(permute_225, view_161);  permute_225 = view_161 = None
    permute_226: "f32[196, 384]" = torch.ops.aten.permute.default(mm_56, [1, 0]);  mm_56 = None
    permute_227: "f32[384, 196]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    mm_57: "f32[3072, 196]" = torch.ops.aten.mm.default(view_240, permute_227);  view_240 = permute_227 = None
    view_241: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_57, [8, 384, 196]);  mm_57 = None
    permute_228: "f32[384, 196]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_229: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_241, [0, 2, 1]);  view_241 = None
    clone_185: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
    clone_186: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_139, memory_format = torch.contiguous_format);  add_139 = None
    sub_81: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_186, getitem_161);  clone_186 = getitem_161 = None
    mul_290: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_40);  sub_81 = None
    mul_291: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_185, primals_243);  primals_243 = None
    mul_292: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_291, 384)
    sum_50: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True)
    mul_293: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_291, mul_290);  mul_291 = None
    sum_51: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True);  mul_293 = None
    mul_294: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_290, sum_51);  sum_51 = None
    sub_82: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_292, sum_50);  mul_292 = sum_50 = None
    sub_83: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_82, mul_294);  sub_82 = mul_294 = None
    div_9: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 384);  rsqrt_40 = None
    mul_295: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_9, sub_83);  div_9 = sub_83 = None
    mul_296: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_185, mul_290);  mul_290 = None
    sum_52: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 1]);  mul_296 = None
    sum_53: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_185, [0, 1]);  clone_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_185: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_183, mul_295);  add_183 = mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_242: "f32[1568, 384]" = torch.ops.aten.view.default(add_185, [1568, 384])
    permute_230: "f32[384, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    mm_58: "f32[1568, 768]" = torch.ops.aten.mm.default(view_242, permute_230);  permute_230 = None
    permute_231: "f32[384, 1568]" = torch.ops.aten.permute.default(view_242, [1, 0])
    mm_59: "f32[384, 768]" = torch.ops.aten.mm.default(permute_231, view_159);  permute_231 = view_159 = None
    permute_232: "f32[768, 384]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_54: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_242, [0], True);  view_242 = None
    view_243: "f32[384]" = torch.ops.aten.view.default(sum_54, [384]);  sum_54 = None
    permute_233: "f32[384, 768]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    view_244: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_58, [8, 196, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_297: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_244, getitem_158);  getitem_158 = None
    mul_298: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_244, mul_158);  view_244 = mul_158 = None
    sigmoid_56: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_159)
    full_8: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_84: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_8, sigmoid_56);  full_8 = None
    mul_299: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_159, sub_84);  getitem_159 = sub_84 = None
    add_186: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_299, 1);  mul_299 = None
    mul_300: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_56, add_186);  sigmoid_56 = add_186 = None
    mul_301: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_297, mul_300);  mul_297 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_8: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_298, mul_301], 2);  mul_298 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_245: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_8, [1568, 1536]);  cat_8 = None
    permute_235: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_60: "f32[1568, 384]" = torch.ops.aten.mm.default(view_245, permute_235);  permute_235 = None
    permute_236: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_245, [1, 0])
    mm_61: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_236, view_157);  permute_236 = view_157 = None
    permute_237: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_55: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_245, [0], True);  view_245 = None
    view_246: "f32[1536]" = torch.ops.aten.view.default(sum_55, [1536]);  sum_55 = None
    permute_238: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
    view_247: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_60, [8, 196, 384]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_187: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_136, memory_format = torch.contiguous_format);  add_136 = None
    sub_85: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_187, getitem_157);  clone_187 = getitem_157 = None
    mul_302: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_39);  sub_85 = None
    mul_303: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_247, primals_237);  primals_237 = None
    mul_304: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_303, 384)
    sum_56: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [2], True)
    mul_305: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_303, mul_302);  mul_303 = None
    sum_57: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True);  mul_305 = None
    mul_306: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_302, sum_57);  sum_57 = None
    sub_86: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_304, sum_56);  mul_304 = sum_56 = None
    sub_87: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_86, mul_306);  sub_86 = mul_306 = None
    div_10: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 384);  rsqrt_39 = None
    mul_307: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_10, sub_87);  div_10 = sub_87 = None
    mul_308: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_247, mul_302);  mul_302 = None
    sum_58: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 1]);  mul_308 = None
    sum_59: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_247, [0, 1]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_187: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_185, mul_307);  add_185 = mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_239: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_187, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_188: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
    view_248: "f32[3072, 196]" = torch.ops.aten.view.default(clone_188, [3072, 196]);  clone_188 = None
    permute_240: "f32[196, 192]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    mm_62: "f32[3072, 192]" = torch.ops.aten.mm.default(view_248, permute_240);  permute_240 = None
    permute_241: "f32[196, 3072]" = torch.ops.aten.permute.default(view_248, [1, 0])
    mm_63: "f32[196, 192]" = torch.ops.aten.mm.default(permute_241, view_155);  permute_241 = view_155 = None
    permute_242: "f32[192, 196]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_60: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True);  view_248 = None
    view_249: "f32[196]" = torch.ops.aten.view.default(sum_60, [196]);  sum_60 = None
    permute_243: "f32[196, 192]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    view_250: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_62, [8, 384, 192]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_309: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_250, getitem_154);  getitem_154 = None
    mul_310: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_250, mul_154);  view_250 = mul_154 = None
    sigmoid_57: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_155)
    full_9: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_88: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_9, sigmoid_57);  full_9 = None
    mul_311: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_155, sub_88);  getitem_155 = sub_88 = None
    add_188: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_311, 1);  mul_311 = None
    mul_312: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_57, add_188);  sigmoid_57 = add_188 = None
    mul_313: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_309, mul_312);  mul_309 = mul_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_9: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_310, mul_313], 2);  mul_310 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_61: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_9, [0, 1], True)
    view_251: "f32[384]" = torch.ops.aten.view.default(sum_61, [384]);  sum_61 = None
    view_252: "f32[3072, 384]" = torch.ops.aten.view.default(cat_9, [3072, 384]);  cat_9 = None
    permute_245: "f32[384, 3072]" = torch.ops.aten.permute.default(view_252, [1, 0])
    mm_64: "f32[384, 196]" = torch.ops.aten.mm.default(permute_245, view_153);  permute_245 = view_153 = None
    permute_246: "f32[196, 384]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    permute_247: "f32[384, 196]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    mm_65: "f32[3072, 196]" = torch.ops.aten.mm.default(view_252, permute_247);  view_252 = permute_247 = None
    view_253: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_65, [8, 384, 196]);  mm_65 = None
    permute_248: "f32[384, 196]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_249: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
    clone_189: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    clone_190: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_132, memory_format = torch.contiguous_format);  add_132 = None
    sub_89: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_190, getitem_153);  clone_190 = getitem_153 = None
    mul_314: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_38);  sub_89 = None
    mul_315: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_189, primals_231);  primals_231 = None
    mul_316: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_315, 384)
    sum_62: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True)
    mul_317: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_315, mul_314);  mul_315 = None
    sum_63: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True);  mul_317 = None
    mul_318: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_314, sum_63);  sum_63 = None
    sub_90: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_316, sum_62);  mul_316 = sum_62 = None
    sub_91: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_90, mul_318);  sub_90 = mul_318 = None
    div_11: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 384);  rsqrt_38 = None
    mul_319: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_11, sub_91);  div_11 = sub_91 = None
    mul_320: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_189, mul_314);  mul_314 = None
    sum_64: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_320, [0, 1]);  mul_320 = None
    sum_65: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_189, [0, 1]);  clone_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_189: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_187, mul_319);  add_187 = mul_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_254: "f32[1568, 384]" = torch.ops.aten.view.default(add_189, [1568, 384])
    permute_250: "f32[384, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    mm_66: "f32[1568, 768]" = torch.ops.aten.mm.default(view_254, permute_250);  permute_250 = None
    permute_251: "f32[384, 1568]" = torch.ops.aten.permute.default(view_254, [1, 0])
    mm_67: "f32[384, 768]" = torch.ops.aten.mm.default(permute_251, view_151);  permute_251 = view_151 = None
    permute_252: "f32[768, 384]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_66: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
    view_255: "f32[384]" = torch.ops.aten.view.default(sum_66, [384]);  sum_66 = None
    permute_253: "f32[384, 768]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    view_256: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_66, [8, 196, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_321: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_256, getitem_150);  getitem_150 = None
    mul_322: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_256, mul_150);  view_256 = mul_150 = None
    sigmoid_58: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_151)
    full_10: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_92: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_10, sigmoid_58);  full_10 = None
    mul_323: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_151, sub_92);  getitem_151 = sub_92 = None
    add_190: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_323, 1);  mul_323 = None
    mul_324: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_58, add_190);  sigmoid_58 = add_190 = None
    mul_325: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_321, mul_324);  mul_321 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_10: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_322, mul_325], 2);  mul_322 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_257: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_10, [1568, 1536]);  cat_10 = None
    permute_255: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    mm_68: "f32[1568, 384]" = torch.ops.aten.mm.default(view_257, permute_255);  permute_255 = None
    permute_256: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_257, [1, 0])
    mm_69: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_256, view_149);  permute_256 = view_149 = None
    permute_257: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_67: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_257, [0], True);  view_257 = None
    view_258: "f32[1536]" = torch.ops.aten.view.default(sum_67, [1536]);  sum_67 = None
    permute_258: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    view_259: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_68, [8, 196, 384]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_191: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_129, memory_format = torch.contiguous_format);  add_129 = None
    sub_93: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_191, getitem_149);  clone_191 = getitem_149 = None
    mul_326: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_37);  sub_93 = None
    mul_327: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_259, primals_225);  primals_225 = None
    mul_328: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_327, 384)
    sum_68: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True)
    mul_329: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_327, mul_326);  mul_327 = None
    sum_69: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True);  mul_329 = None
    mul_330: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_326, sum_69);  sum_69 = None
    sub_94: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_328, sum_68);  mul_328 = sum_68 = None
    sub_95: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_94, mul_330);  sub_94 = mul_330 = None
    div_12: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 384);  rsqrt_37 = None
    mul_331: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_12, sub_95);  div_12 = sub_95 = None
    mul_332: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_259, mul_326);  mul_326 = None
    sum_70: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1]);  mul_332 = None
    sum_71: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_259, [0, 1]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_191: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_189, mul_331);  add_189 = mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_259: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_191, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_192: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
    view_260: "f32[3072, 196]" = torch.ops.aten.view.default(clone_192, [3072, 196]);  clone_192 = None
    permute_260: "f32[196, 192]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_70: "f32[3072, 192]" = torch.ops.aten.mm.default(view_260, permute_260);  permute_260 = None
    permute_261: "f32[196, 3072]" = torch.ops.aten.permute.default(view_260, [1, 0])
    mm_71: "f32[196, 192]" = torch.ops.aten.mm.default(permute_261, view_147);  permute_261 = view_147 = None
    permute_262: "f32[192, 196]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_72: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_260, [0], True);  view_260 = None
    view_261: "f32[196]" = torch.ops.aten.view.default(sum_72, [196]);  sum_72 = None
    permute_263: "f32[196, 192]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    view_262: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_70, [8, 384, 192]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_333: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_262, getitem_146);  getitem_146 = None
    mul_334: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_262, mul_146);  view_262 = mul_146 = None
    sigmoid_59: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_147)
    full_11: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_96: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_11, sigmoid_59);  full_11 = None
    mul_335: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_147, sub_96);  getitem_147 = sub_96 = None
    add_192: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_335, 1);  mul_335 = None
    mul_336: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_59, add_192);  sigmoid_59 = add_192 = None
    mul_337: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_333, mul_336);  mul_333 = mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_11: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_334, mul_337], 2);  mul_334 = mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_73: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_11, [0, 1], True)
    view_263: "f32[384]" = torch.ops.aten.view.default(sum_73, [384]);  sum_73 = None
    view_264: "f32[3072, 384]" = torch.ops.aten.view.default(cat_11, [3072, 384]);  cat_11 = None
    permute_265: "f32[384, 3072]" = torch.ops.aten.permute.default(view_264, [1, 0])
    mm_72: "f32[384, 196]" = torch.ops.aten.mm.default(permute_265, view_145);  permute_265 = view_145 = None
    permute_266: "f32[196, 384]" = torch.ops.aten.permute.default(mm_72, [1, 0]);  mm_72 = None
    permute_267: "f32[384, 196]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_73: "f32[3072, 196]" = torch.ops.aten.mm.default(view_264, permute_267);  view_264 = permute_267 = None
    view_265: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_73, [8, 384, 196]);  mm_73 = None
    permute_268: "f32[384, 196]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_269: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_265, [0, 2, 1]);  view_265 = None
    clone_193: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
    clone_194: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format);  add_125 = None
    sub_97: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_194, getitem_145);  clone_194 = getitem_145 = None
    mul_338: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_36);  sub_97 = None
    mul_339: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_193, primals_219);  primals_219 = None
    mul_340: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_339, 384)
    sum_74: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True)
    mul_341: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_339, mul_338);  mul_339 = None
    sum_75: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
    mul_342: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_338, sum_75);  sum_75 = None
    sub_98: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_340, sum_74);  mul_340 = sum_74 = None
    sub_99: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_98, mul_342);  sub_98 = mul_342 = None
    div_13: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 384);  rsqrt_36 = None
    mul_343: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_13, sub_99);  div_13 = sub_99 = None
    mul_344: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_193, mul_338);  mul_338 = None
    sum_76: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 1]);  mul_344 = None
    sum_77: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_193, [0, 1]);  clone_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_193: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_191, mul_343);  add_191 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_266: "f32[1568, 384]" = torch.ops.aten.view.default(add_193, [1568, 384])
    permute_270: "f32[384, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_74: "f32[1568, 768]" = torch.ops.aten.mm.default(view_266, permute_270);  permute_270 = None
    permute_271: "f32[384, 1568]" = torch.ops.aten.permute.default(view_266, [1, 0])
    mm_75: "f32[384, 768]" = torch.ops.aten.mm.default(permute_271, view_143);  permute_271 = view_143 = None
    permute_272: "f32[768, 384]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_78: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
    view_267: "f32[384]" = torch.ops.aten.view.default(sum_78, [384]);  sum_78 = None
    permute_273: "f32[384, 768]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_268: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_74, [8, 196, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_345: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_268, getitem_142);  getitem_142 = None
    mul_346: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_268, mul_142);  view_268 = mul_142 = None
    sigmoid_60: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_143)
    full_12: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_100: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_12, sigmoid_60);  full_12 = None
    mul_347: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_143, sub_100);  getitem_143 = sub_100 = None
    add_194: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_347, 1);  mul_347 = None
    mul_348: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_60, add_194);  sigmoid_60 = add_194 = None
    mul_349: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_345, mul_348);  mul_345 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_12: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_346, mul_349], 2);  mul_346 = mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_269: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_12, [1568, 1536]);  cat_12 = None
    permute_275: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_76: "f32[1568, 384]" = torch.ops.aten.mm.default(view_269, permute_275);  permute_275 = None
    permute_276: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_269, [1, 0])
    mm_77: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_276, view_141);  permute_276 = view_141 = None
    permute_277: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_79: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
    view_270: "f32[1536]" = torch.ops.aten.view.default(sum_79, [1536]);  sum_79 = None
    permute_278: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    view_271: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_76, [8, 196, 384]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_195: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_122, memory_format = torch.contiguous_format);  add_122 = None
    sub_101: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_195, getitem_141);  clone_195 = getitem_141 = None
    mul_350: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_35);  sub_101 = None
    mul_351: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_271, primals_213);  primals_213 = None
    mul_352: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_351, 384)
    sum_80: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [2], True)
    mul_353: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_351, mul_350);  mul_351 = None
    sum_81: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True);  mul_353 = None
    mul_354: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_350, sum_81);  sum_81 = None
    sub_102: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_352, sum_80);  mul_352 = sum_80 = None
    sub_103: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_102, mul_354);  sub_102 = mul_354 = None
    div_14: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 384);  rsqrt_35 = None
    mul_355: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_14, sub_103);  div_14 = sub_103 = None
    mul_356: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_271, mul_350);  mul_350 = None
    sum_82: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_356, [0, 1]);  mul_356 = None
    sum_83: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_271, [0, 1]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_195: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_193, mul_355);  add_193 = mul_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_279: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_195, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_196: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
    view_272: "f32[3072, 196]" = torch.ops.aten.view.default(clone_196, [3072, 196]);  clone_196 = None
    permute_280: "f32[196, 192]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    mm_78: "f32[3072, 192]" = torch.ops.aten.mm.default(view_272, permute_280);  permute_280 = None
    permute_281: "f32[196, 3072]" = torch.ops.aten.permute.default(view_272, [1, 0])
    mm_79: "f32[196, 192]" = torch.ops.aten.mm.default(permute_281, view_139);  permute_281 = view_139 = None
    permute_282: "f32[192, 196]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_84: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
    view_273: "f32[196]" = torch.ops.aten.view.default(sum_84, [196]);  sum_84 = None
    permute_283: "f32[196, 192]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    view_274: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_78, [8, 384, 192]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_357: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_274, getitem_138);  getitem_138 = None
    mul_358: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_274, mul_138);  view_274 = mul_138 = None
    sigmoid_61: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_139)
    full_13: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_104: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_13, sigmoid_61);  full_13 = None
    mul_359: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_139, sub_104);  getitem_139 = sub_104 = None
    add_196: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_359, 1);  mul_359 = None
    mul_360: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_61, add_196);  sigmoid_61 = add_196 = None
    mul_361: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_357, mul_360);  mul_357 = mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_13: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_358, mul_361], 2);  mul_358 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_85: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_13, [0, 1], True)
    view_275: "f32[384]" = torch.ops.aten.view.default(sum_85, [384]);  sum_85 = None
    view_276: "f32[3072, 384]" = torch.ops.aten.view.default(cat_13, [3072, 384]);  cat_13 = None
    permute_285: "f32[384, 3072]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_80: "f32[384, 196]" = torch.ops.aten.mm.default(permute_285, view_137);  permute_285 = view_137 = None
    permute_286: "f32[196, 384]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    permute_287: "f32[384, 196]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    mm_81: "f32[3072, 196]" = torch.ops.aten.mm.default(view_276, permute_287);  view_276 = permute_287 = None
    view_277: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_81, [8, 384, 196]);  mm_81 = None
    permute_288: "f32[384, 196]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_289: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None
    clone_197: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
    clone_198: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_118, memory_format = torch.contiguous_format);  add_118 = None
    sub_105: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_198, getitem_137);  clone_198 = getitem_137 = None
    mul_362: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_34);  sub_105 = None
    mul_363: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_197, primals_207);  primals_207 = None
    mul_364: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_363, 384)
    sum_86: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [2], True)
    mul_365: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_363, mul_362);  mul_363 = None
    sum_87: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True);  mul_365 = None
    mul_366: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_362, sum_87);  sum_87 = None
    sub_106: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_364, sum_86);  mul_364 = sum_86 = None
    sub_107: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_106, mul_366);  sub_106 = mul_366 = None
    div_15: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 384);  rsqrt_34 = None
    mul_367: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_15, sub_107);  div_15 = sub_107 = None
    mul_368: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_197, mul_362);  mul_362 = None
    sum_88: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 1]);  mul_368 = None
    sum_89: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_197, [0, 1]);  clone_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_197: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_195, mul_367);  add_195 = mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_278: "f32[1568, 384]" = torch.ops.aten.view.default(add_197, [1568, 384])
    permute_290: "f32[384, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    mm_82: "f32[1568, 768]" = torch.ops.aten.mm.default(view_278, permute_290);  permute_290 = None
    permute_291: "f32[384, 1568]" = torch.ops.aten.permute.default(view_278, [1, 0])
    mm_83: "f32[384, 768]" = torch.ops.aten.mm.default(permute_291, view_135);  permute_291 = view_135 = None
    permute_292: "f32[768, 384]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_90: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
    view_279: "f32[384]" = torch.ops.aten.view.default(sum_90, [384]);  sum_90 = None
    permute_293: "f32[384, 768]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
    view_280: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_82, [8, 196, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_369: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_280, getitem_134);  getitem_134 = None
    mul_370: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_280, mul_134);  view_280 = mul_134 = None
    sigmoid_62: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_135)
    full_14: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_108: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_14, sigmoid_62);  full_14 = None
    mul_371: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_135, sub_108);  getitem_135 = sub_108 = None
    add_198: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_371, 1);  mul_371 = None
    mul_372: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_62, add_198);  sigmoid_62 = add_198 = None
    mul_373: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_369, mul_372);  mul_369 = mul_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_14: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_370, mul_373], 2);  mul_370 = mul_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_281: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_14, [1568, 1536]);  cat_14 = None
    permute_295: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    mm_84: "f32[1568, 384]" = torch.ops.aten.mm.default(view_281, permute_295);  permute_295 = None
    permute_296: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_281, [1, 0])
    mm_85: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_296, view_133);  permute_296 = view_133 = None
    permute_297: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_91: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_281, [0], True);  view_281 = None
    view_282: "f32[1536]" = torch.ops.aten.view.default(sum_91, [1536]);  sum_91 = None
    permute_298: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    view_283: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_84, [8, 196, 384]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_199: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_115, memory_format = torch.contiguous_format);  add_115 = None
    sub_109: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_199, getitem_133);  clone_199 = getitem_133 = None
    mul_374: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_33);  sub_109 = None
    mul_375: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_283, primals_201);  primals_201 = None
    mul_376: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_375, 384)
    sum_92: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [2], True)
    mul_377: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_375, mul_374);  mul_375 = None
    sum_93: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True);  mul_377 = None
    mul_378: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_374, sum_93);  sum_93 = None
    sub_110: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_376, sum_92);  mul_376 = sum_92 = None
    sub_111: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_110, mul_378);  sub_110 = mul_378 = None
    div_16: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 384);  rsqrt_33 = None
    mul_379: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_16, sub_111);  div_16 = sub_111 = None
    mul_380: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_283, mul_374);  mul_374 = None
    sum_94: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_380, [0, 1]);  mul_380 = None
    sum_95: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_283, [0, 1]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_199: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_197, mul_379);  add_197 = mul_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_299: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_199, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_200: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
    view_284: "f32[3072, 196]" = torch.ops.aten.view.default(clone_200, [3072, 196]);  clone_200 = None
    permute_300: "f32[196, 192]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_86: "f32[3072, 192]" = torch.ops.aten.mm.default(view_284, permute_300);  permute_300 = None
    permute_301: "f32[196, 3072]" = torch.ops.aten.permute.default(view_284, [1, 0])
    mm_87: "f32[196, 192]" = torch.ops.aten.mm.default(permute_301, view_131);  permute_301 = view_131 = None
    permute_302: "f32[192, 196]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_96: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_284, [0], True);  view_284 = None
    view_285: "f32[196]" = torch.ops.aten.view.default(sum_96, [196]);  sum_96 = None
    permute_303: "f32[196, 192]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    view_286: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_86, [8, 384, 192]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_381: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_286, getitem_130);  getitem_130 = None
    mul_382: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_286, mul_130);  view_286 = mul_130 = None
    sigmoid_63: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_131)
    full_15: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_112: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_15, sigmoid_63);  full_15 = None
    mul_383: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_131, sub_112);  getitem_131 = sub_112 = None
    add_200: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_383, 1);  mul_383 = None
    mul_384: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_63, add_200);  sigmoid_63 = add_200 = None
    mul_385: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_381, mul_384);  mul_381 = mul_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_15: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_382, mul_385], 2);  mul_382 = mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_97: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_15, [0, 1], True)
    view_287: "f32[384]" = torch.ops.aten.view.default(sum_97, [384]);  sum_97 = None
    view_288: "f32[3072, 384]" = torch.ops.aten.view.default(cat_15, [3072, 384]);  cat_15 = None
    permute_305: "f32[384, 3072]" = torch.ops.aten.permute.default(view_288, [1, 0])
    mm_88: "f32[384, 196]" = torch.ops.aten.mm.default(permute_305, view_129);  permute_305 = view_129 = None
    permute_306: "f32[196, 384]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    permute_307: "f32[384, 196]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_89: "f32[3072, 196]" = torch.ops.aten.mm.default(view_288, permute_307);  view_288 = permute_307 = None
    view_289: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_89, [8, 384, 196]);  mm_89 = None
    permute_308: "f32[384, 196]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_309: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_289, [0, 2, 1]);  view_289 = None
    clone_201: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
    clone_202: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format);  add_111 = None
    sub_113: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_202, getitem_129);  clone_202 = getitem_129 = None
    mul_386: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_32);  sub_113 = None
    mul_387: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_201, primals_195);  primals_195 = None
    mul_388: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_387, 384)
    sum_98: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_387, [2], True)
    mul_389: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_387, mul_386);  mul_387 = None
    sum_99: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_389, [2], True);  mul_389 = None
    mul_390: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_386, sum_99);  sum_99 = None
    sub_114: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_388, sum_98);  mul_388 = sum_98 = None
    sub_115: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_114, mul_390);  sub_114 = mul_390 = None
    div_17: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 384);  rsqrt_32 = None
    mul_391: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_17, sub_115);  div_17 = sub_115 = None
    mul_392: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_201, mul_386);  mul_386 = None
    sum_100: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 1]);  mul_392 = None
    sum_101: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_201, [0, 1]);  clone_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_201: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_199, mul_391);  add_199 = mul_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_290: "f32[1568, 384]" = torch.ops.aten.view.default(add_201, [1568, 384])
    permute_310: "f32[384, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_90: "f32[1568, 768]" = torch.ops.aten.mm.default(view_290, permute_310);  permute_310 = None
    permute_311: "f32[384, 1568]" = torch.ops.aten.permute.default(view_290, [1, 0])
    mm_91: "f32[384, 768]" = torch.ops.aten.mm.default(permute_311, view_127);  permute_311 = view_127 = None
    permute_312: "f32[768, 384]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_102: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
    view_291: "f32[384]" = torch.ops.aten.view.default(sum_102, [384]);  sum_102 = None
    permute_313: "f32[384, 768]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_292: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_90, [8, 196, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_393: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_292, getitem_126);  getitem_126 = None
    mul_394: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_292, mul_126);  view_292 = mul_126 = None
    sigmoid_64: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_127)
    full_16: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_116: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_16, sigmoid_64);  full_16 = None
    mul_395: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_127, sub_116);  getitem_127 = sub_116 = None
    add_202: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_395, 1);  mul_395 = None
    mul_396: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_64, add_202);  sigmoid_64 = add_202 = None
    mul_397: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_393, mul_396);  mul_393 = mul_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_16: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_394, mul_397], 2);  mul_394 = mul_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_293: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_16, [1568, 1536]);  cat_16 = None
    permute_315: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    mm_92: "f32[1568, 384]" = torch.ops.aten.mm.default(view_293, permute_315);  permute_315 = None
    permute_316: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_293, [1, 0])
    mm_93: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_316, view_125);  permute_316 = view_125 = None
    permute_317: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_103: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
    view_294: "f32[1536]" = torch.ops.aten.view.default(sum_103, [1536]);  sum_103 = None
    permute_318: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    view_295: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_92, [8, 196, 384]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_203: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_108, memory_format = torch.contiguous_format);  add_108 = None
    sub_117: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_203, getitem_125);  clone_203 = getitem_125 = None
    mul_398: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_31);  sub_117 = None
    mul_399: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_295, primals_189);  primals_189 = None
    mul_400: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_399, 384)
    sum_104: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True)
    mul_401: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_399, mul_398);  mul_399 = None
    sum_105: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [2], True);  mul_401 = None
    mul_402: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_398, sum_105);  sum_105 = None
    sub_118: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_400, sum_104);  mul_400 = sum_104 = None
    sub_119: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_118, mul_402);  sub_118 = mul_402 = None
    div_18: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 384);  rsqrt_31 = None
    mul_403: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_18, sub_119);  div_18 = sub_119 = None
    mul_404: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_295, mul_398);  mul_398 = None
    sum_106: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 1]);  mul_404 = None
    sum_107: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_295, [0, 1]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_203: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_201, mul_403);  add_201 = mul_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_319: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_203, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_204: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_296: "f32[3072, 196]" = torch.ops.aten.view.default(clone_204, [3072, 196]);  clone_204 = None
    permute_320: "f32[196, 192]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    mm_94: "f32[3072, 192]" = torch.ops.aten.mm.default(view_296, permute_320);  permute_320 = None
    permute_321: "f32[196, 3072]" = torch.ops.aten.permute.default(view_296, [1, 0])
    mm_95: "f32[196, 192]" = torch.ops.aten.mm.default(permute_321, view_123);  permute_321 = view_123 = None
    permute_322: "f32[192, 196]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_108: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
    view_297: "f32[196]" = torch.ops.aten.view.default(sum_108, [196]);  sum_108 = None
    permute_323: "f32[196, 192]" = torch.ops.aten.permute.default(permute_322, [1, 0]);  permute_322 = None
    view_298: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_94, [8, 384, 192]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_405: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_298, getitem_122);  getitem_122 = None
    mul_406: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_298, mul_122);  view_298 = mul_122 = None
    sigmoid_65: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_123)
    full_17: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_120: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_17, sigmoid_65);  full_17 = None
    mul_407: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_123, sub_120);  getitem_123 = sub_120 = None
    add_204: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_407, 1);  mul_407 = None
    mul_408: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_65, add_204);  sigmoid_65 = add_204 = None
    mul_409: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_405, mul_408);  mul_405 = mul_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_17: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_406, mul_409], 2);  mul_406 = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_109: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_17, [0, 1], True)
    view_299: "f32[384]" = torch.ops.aten.view.default(sum_109, [384]);  sum_109 = None
    view_300: "f32[3072, 384]" = torch.ops.aten.view.default(cat_17, [3072, 384]);  cat_17 = None
    permute_325: "f32[384, 3072]" = torch.ops.aten.permute.default(view_300, [1, 0])
    mm_96: "f32[384, 196]" = torch.ops.aten.mm.default(permute_325, view_121);  permute_325 = view_121 = None
    permute_326: "f32[196, 384]" = torch.ops.aten.permute.default(mm_96, [1, 0]);  mm_96 = None
    permute_327: "f32[384, 196]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    mm_97: "f32[3072, 196]" = torch.ops.aten.mm.default(view_300, permute_327);  view_300 = permute_327 = None
    view_301: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_97, [8, 384, 196]);  mm_97 = None
    permute_328: "f32[384, 196]" = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_329: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_301, [0, 2, 1]);  view_301 = None
    clone_205: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
    clone_206: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_104, memory_format = torch.contiguous_format);  add_104 = None
    sub_121: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_206, getitem_121);  clone_206 = getitem_121 = None
    mul_410: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_30);  sub_121 = None
    mul_411: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_205, primals_183);  primals_183 = None
    mul_412: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_411, 384)
    sum_110: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
    mul_413: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_411, mul_410);  mul_411 = None
    sum_111: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
    mul_414: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_410, sum_111);  sum_111 = None
    sub_122: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_412, sum_110);  mul_412 = sum_110 = None
    sub_123: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_122, mul_414);  sub_122 = mul_414 = None
    div_19: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 384);  rsqrt_30 = None
    mul_415: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_19, sub_123);  div_19 = sub_123 = None
    mul_416: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_205, mul_410);  mul_410 = None
    sum_112: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
    sum_113: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_205, [0, 1]);  clone_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_205: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_203, mul_415);  add_203 = mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_302: "f32[1568, 384]" = torch.ops.aten.view.default(add_205, [1568, 384])
    permute_330: "f32[384, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    mm_98: "f32[1568, 768]" = torch.ops.aten.mm.default(view_302, permute_330);  permute_330 = None
    permute_331: "f32[384, 1568]" = torch.ops.aten.permute.default(view_302, [1, 0])
    mm_99: "f32[384, 768]" = torch.ops.aten.mm.default(permute_331, view_119);  permute_331 = view_119 = None
    permute_332: "f32[768, 384]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_114: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
    view_303: "f32[384]" = torch.ops.aten.view.default(sum_114, [384]);  sum_114 = None
    permute_333: "f32[384, 768]" = torch.ops.aten.permute.default(permute_332, [1, 0]);  permute_332 = None
    view_304: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_98, [8, 196, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_417: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_304, getitem_118);  getitem_118 = None
    mul_418: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_304, mul_118);  view_304 = mul_118 = None
    sigmoid_66: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_119)
    full_18: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_124: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_18, sigmoid_66);  full_18 = None
    mul_419: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_119, sub_124);  getitem_119 = sub_124 = None
    add_206: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_419, 1);  mul_419 = None
    mul_420: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_66, add_206);  sigmoid_66 = add_206 = None
    mul_421: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_417, mul_420);  mul_417 = mul_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_18: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_418, mul_421], 2);  mul_418 = mul_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_305: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_18, [1568, 1536]);  cat_18 = None
    permute_335: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_100: "f32[1568, 384]" = torch.ops.aten.mm.default(view_305, permute_335);  permute_335 = None
    permute_336: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_305, [1, 0])
    mm_101: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_336, view_117);  permute_336 = view_117 = None
    permute_337: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_115: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
    view_306: "f32[1536]" = torch.ops.aten.view.default(sum_115, [1536]);  sum_115 = None
    permute_338: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
    view_307: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_100, [8, 196, 384]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_207: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_101, memory_format = torch.contiguous_format);  add_101 = None
    sub_125: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_207, getitem_117);  clone_207 = getitem_117 = None
    mul_422: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_29);  sub_125 = None
    mul_423: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_307, primals_177);  primals_177 = None
    mul_424: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_423, 384)
    sum_116: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_423, [2], True)
    mul_425: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_423, mul_422);  mul_423 = None
    sum_117: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True);  mul_425 = None
    mul_426: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_422, sum_117);  sum_117 = None
    sub_126: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_424, sum_116);  mul_424 = sum_116 = None
    sub_127: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_126, mul_426);  sub_126 = mul_426 = None
    div_20: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 384);  rsqrt_29 = None
    mul_427: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_20, sub_127);  div_20 = sub_127 = None
    mul_428: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_307, mul_422);  mul_422 = None
    sum_118: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_428, [0, 1]);  mul_428 = None
    sum_119: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_307, [0, 1]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_207: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_205, mul_427);  add_205 = mul_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_339: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_207, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_208: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_339, memory_format = torch.contiguous_format);  permute_339 = None
    view_308: "f32[3072, 196]" = torch.ops.aten.view.default(clone_208, [3072, 196]);  clone_208 = None
    permute_340: "f32[196, 192]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_102: "f32[3072, 192]" = torch.ops.aten.mm.default(view_308, permute_340);  permute_340 = None
    permute_341: "f32[196, 3072]" = torch.ops.aten.permute.default(view_308, [1, 0])
    mm_103: "f32[196, 192]" = torch.ops.aten.mm.default(permute_341, view_115);  permute_341 = view_115 = None
    permute_342: "f32[192, 196]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_120: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
    view_309: "f32[196]" = torch.ops.aten.view.default(sum_120, [196]);  sum_120 = None
    permute_343: "f32[196, 192]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_310: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_102, [8, 384, 192]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_429: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_310, getitem_114);  getitem_114 = None
    mul_430: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_310, mul_114);  view_310 = mul_114 = None
    sigmoid_67: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_115)
    full_19: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_128: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_19, sigmoid_67);  full_19 = None
    mul_431: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_115, sub_128);  getitem_115 = sub_128 = None
    add_208: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_431, 1);  mul_431 = None
    mul_432: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_67, add_208);  sigmoid_67 = add_208 = None
    mul_433: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_429, mul_432);  mul_429 = mul_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_19: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_430, mul_433], 2);  mul_430 = mul_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_121: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_19, [0, 1], True)
    view_311: "f32[384]" = torch.ops.aten.view.default(sum_121, [384]);  sum_121 = None
    view_312: "f32[3072, 384]" = torch.ops.aten.view.default(cat_19, [3072, 384]);  cat_19 = None
    permute_345: "f32[384, 3072]" = torch.ops.aten.permute.default(view_312, [1, 0])
    mm_104: "f32[384, 196]" = torch.ops.aten.mm.default(permute_345, view_113);  permute_345 = view_113 = None
    permute_346: "f32[196, 384]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    permute_347: "f32[384, 196]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_105: "f32[3072, 196]" = torch.ops.aten.mm.default(view_312, permute_347);  view_312 = permute_347 = None
    view_313: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_105, [8, 384, 196]);  mm_105 = None
    permute_348: "f32[384, 196]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_349: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
    clone_209: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_349, memory_format = torch.contiguous_format);  permute_349 = None
    clone_210: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_97, memory_format = torch.contiguous_format);  add_97 = None
    sub_129: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_210, getitem_113);  clone_210 = getitem_113 = None
    mul_434: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_129, rsqrt_28);  sub_129 = None
    mul_435: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_209, primals_171);  primals_171 = None
    mul_436: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_435, 384)
    sum_122: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_435, [2], True)
    mul_437: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_435, mul_434);  mul_435 = None
    sum_123: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_437, [2], True);  mul_437 = None
    mul_438: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_434, sum_123);  sum_123 = None
    sub_130: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_436, sum_122);  mul_436 = sum_122 = None
    sub_131: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_130, mul_438);  sub_130 = mul_438 = None
    div_21: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 384);  rsqrt_28 = None
    mul_439: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_21, sub_131);  div_21 = sub_131 = None
    mul_440: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_209, mul_434);  mul_434 = None
    sum_124: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_440, [0, 1]);  mul_440 = None
    sum_125: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_209, [0, 1]);  clone_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_209: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_207, mul_439);  add_207 = mul_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_314: "f32[1568, 384]" = torch.ops.aten.view.default(add_209, [1568, 384])
    permute_350: "f32[384, 768]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    mm_106: "f32[1568, 768]" = torch.ops.aten.mm.default(view_314, permute_350);  permute_350 = None
    permute_351: "f32[384, 1568]" = torch.ops.aten.permute.default(view_314, [1, 0])
    mm_107: "f32[384, 768]" = torch.ops.aten.mm.default(permute_351, view_111);  permute_351 = view_111 = None
    permute_352: "f32[768, 384]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_126: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
    view_315: "f32[384]" = torch.ops.aten.view.default(sum_126, [384]);  sum_126 = None
    permute_353: "f32[384, 768]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    view_316: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_106, [8, 196, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_441: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_316, getitem_110);  getitem_110 = None
    mul_442: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_316, mul_110);  view_316 = mul_110 = None
    sigmoid_68: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_111)
    full_20: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_132: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_20, sigmoid_68);  full_20 = None
    mul_443: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_111, sub_132);  getitem_111 = sub_132 = None
    add_210: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_443, 1);  mul_443 = None
    mul_444: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_68, add_210);  sigmoid_68 = add_210 = None
    mul_445: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_441, mul_444);  mul_441 = mul_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_20: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_442, mul_445], 2);  mul_442 = mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_317: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_20, [1568, 1536]);  cat_20 = None
    permute_355: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    mm_108: "f32[1568, 384]" = torch.ops.aten.mm.default(view_317, permute_355);  permute_355 = None
    permute_356: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_317, [1, 0])
    mm_109: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_356, view_109);  permute_356 = view_109 = None
    permute_357: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_127: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
    view_318: "f32[1536]" = torch.ops.aten.view.default(sum_127, [1536]);  sum_127 = None
    permute_358: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
    view_319: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_108, [8, 196, 384]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_211: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format);  add_94 = None
    sub_133: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_211, getitem_109);  clone_211 = getitem_109 = None
    mul_446: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_27);  sub_133 = None
    mul_447: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_319, primals_165);  primals_165 = None
    mul_448: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_447, 384)
    sum_128: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True)
    mul_449: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_447, mul_446);  mul_447 = None
    sum_129: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_449, [2], True);  mul_449 = None
    mul_450: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_446, sum_129);  sum_129 = None
    sub_134: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_448, sum_128);  mul_448 = sum_128 = None
    sub_135: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_134, mul_450);  sub_134 = mul_450 = None
    div_22: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 384);  rsqrt_27 = None
    mul_451: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_22, sub_135);  div_22 = sub_135 = None
    mul_452: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_319, mul_446);  mul_446 = None
    sum_130: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 1]);  mul_452 = None
    sum_131: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_319, [0, 1]);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_211: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_209, mul_451);  add_209 = mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_359: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_211, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_212: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_320: "f32[3072, 196]" = torch.ops.aten.view.default(clone_212, [3072, 196]);  clone_212 = None
    permute_360: "f32[196, 192]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    mm_110: "f32[3072, 192]" = torch.ops.aten.mm.default(view_320, permute_360);  permute_360 = None
    permute_361: "f32[196, 3072]" = torch.ops.aten.permute.default(view_320, [1, 0])
    mm_111: "f32[196, 192]" = torch.ops.aten.mm.default(permute_361, view_107);  permute_361 = view_107 = None
    permute_362: "f32[192, 196]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_132: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_320, [0], True);  view_320 = None
    view_321: "f32[196]" = torch.ops.aten.view.default(sum_132, [196]);  sum_132 = None
    permute_363: "f32[196, 192]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_322: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_110, [8, 384, 192]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_453: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_322, getitem_106);  getitem_106 = None
    mul_454: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_322, mul_106);  view_322 = mul_106 = None
    sigmoid_69: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_107)
    full_21: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_136: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_21, sigmoid_69);  full_21 = None
    mul_455: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_107, sub_136);  getitem_107 = sub_136 = None
    add_212: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_455, 1);  mul_455 = None
    mul_456: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_69, add_212);  sigmoid_69 = add_212 = None
    mul_457: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_453, mul_456);  mul_453 = mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_21: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_454, mul_457], 2);  mul_454 = mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_133: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_21, [0, 1], True)
    view_323: "f32[384]" = torch.ops.aten.view.default(sum_133, [384]);  sum_133 = None
    view_324: "f32[3072, 384]" = torch.ops.aten.view.default(cat_21, [3072, 384]);  cat_21 = None
    permute_365: "f32[384, 3072]" = torch.ops.aten.permute.default(view_324, [1, 0])
    mm_112: "f32[384, 196]" = torch.ops.aten.mm.default(permute_365, view_105);  permute_365 = view_105 = None
    permute_366: "f32[196, 384]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    permute_367: "f32[384, 196]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    mm_113: "f32[3072, 196]" = torch.ops.aten.mm.default(view_324, permute_367);  view_324 = permute_367 = None
    view_325: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_113, [8, 384, 196]);  mm_113 = None
    permute_368: "f32[384, 196]" = torch.ops.aten.permute.default(permute_366, [1, 0]);  permute_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_369: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
    clone_213: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_369, memory_format = torch.contiguous_format);  permute_369 = None
    clone_214: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format);  add_90 = None
    sub_137: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_214, getitem_105);  clone_214 = getitem_105 = None
    mul_458: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_26);  sub_137 = None
    mul_459: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_213, primals_159);  primals_159 = None
    mul_460: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_459, 384)
    sum_134: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True)
    mul_461: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_459, mul_458);  mul_459 = None
    sum_135: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True);  mul_461 = None
    mul_462: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_458, sum_135);  sum_135 = None
    sub_138: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_460, sum_134);  mul_460 = sum_134 = None
    sub_139: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_138, mul_462);  sub_138 = mul_462 = None
    div_23: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 384);  rsqrt_26 = None
    mul_463: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_23, sub_139);  div_23 = sub_139 = None
    mul_464: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_213, mul_458);  mul_458 = None
    sum_136: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 1]);  mul_464 = None
    sum_137: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_213, [0, 1]);  clone_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_213: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_211, mul_463);  add_211 = mul_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_326: "f32[1568, 384]" = torch.ops.aten.view.default(add_213, [1568, 384])
    permute_370: "f32[384, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_114: "f32[1568, 768]" = torch.ops.aten.mm.default(view_326, permute_370);  permute_370 = None
    permute_371: "f32[384, 1568]" = torch.ops.aten.permute.default(view_326, [1, 0])
    mm_115: "f32[384, 768]" = torch.ops.aten.mm.default(permute_371, view_103);  permute_371 = view_103 = None
    permute_372: "f32[768, 384]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_138: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_326, [0], True);  view_326 = None
    view_327: "f32[384]" = torch.ops.aten.view.default(sum_138, [384]);  sum_138 = None
    permute_373: "f32[384, 768]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    view_328: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_114, [8, 196, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_465: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_328, getitem_102);  getitem_102 = None
    mul_466: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_328, mul_102);  view_328 = mul_102 = None
    sigmoid_70: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_103)
    full_22: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_140: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_22, sigmoid_70);  full_22 = None
    mul_467: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_103, sub_140);  getitem_103 = sub_140 = None
    add_214: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_467, 1);  mul_467 = None
    mul_468: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_70, add_214);  sigmoid_70 = add_214 = None
    mul_469: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_465, mul_468);  mul_465 = mul_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_22: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_466, mul_469], 2);  mul_466 = mul_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_329: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_22, [1568, 1536]);  cat_22 = None
    permute_375: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_116: "f32[1568, 384]" = torch.ops.aten.mm.default(view_329, permute_375);  permute_375 = None
    permute_376: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_329, [1, 0])
    mm_117: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_376, view_101);  permute_376 = view_101 = None
    permute_377: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_139: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_329, [0], True);  view_329 = None
    view_330: "f32[1536]" = torch.ops.aten.view.default(sum_139, [1536]);  sum_139 = None
    permute_378: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    view_331: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_116, [8, 196, 384]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_215: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_87, memory_format = torch.contiguous_format);  add_87 = None
    sub_141: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_215, getitem_101);  clone_215 = getitem_101 = None
    mul_470: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_141, rsqrt_25);  sub_141 = None
    mul_471: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_331, primals_153);  primals_153 = None
    mul_472: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_471, 384)
    sum_140: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_471, [2], True)
    mul_473: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_471, mul_470);  mul_471 = None
    sum_141: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2], True);  mul_473 = None
    mul_474: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_470, sum_141);  sum_141 = None
    sub_142: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_472, sum_140);  mul_472 = sum_140 = None
    sub_143: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_142, mul_474);  sub_142 = mul_474 = None
    div_24: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 384);  rsqrt_25 = None
    mul_475: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_24, sub_143);  div_24 = sub_143 = None
    mul_476: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_331, mul_470);  mul_470 = None
    sum_142: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 1]);  mul_476 = None
    sum_143: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_331, [0, 1]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_215: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_213, mul_475);  add_213 = mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_379: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_215, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_216: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_379, memory_format = torch.contiguous_format);  permute_379 = None
    view_332: "f32[3072, 196]" = torch.ops.aten.view.default(clone_216, [3072, 196]);  clone_216 = None
    permute_380: "f32[196, 192]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_118: "f32[3072, 192]" = torch.ops.aten.mm.default(view_332, permute_380);  permute_380 = None
    permute_381: "f32[196, 3072]" = torch.ops.aten.permute.default(view_332, [1, 0])
    mm_119: "f32[196, 192]" = torch.ops.aten.mm.default(permute_381, view_99);  permute_381 = view_99 = None
    permute_382: "f32[192, 196]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_144: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
    view_333: "f32[196]" = torch.ops.aten.view.default(sum_144, [196]);  sum_144 = None
    permute_383: "f32[196, 192]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
    view_334: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_118, [8, 384, 192]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_477: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_334, getitem_98);  getitem_98 = None
    mul_478: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_334, mul_98);  view_334 = mul_98 = None
    sigmoid_71: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_99)
    full_23: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_144: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_23, sigmoid_71);  full_23 = None
    mul_479: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_99, sub_144);  getitem_99 = sub_144 = None
    add_216: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_479, 1);  mul_479 = None
    mul_480: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_71, add_216);  sigmoid_71 = add_216 = None
    mul_481: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_477, mul_480);  mul_477 = mul_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_23: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_478, mul_481], 2);  mul_478 = mul_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_145: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_23, [0, 1], True)
    view_335: "f32[384]" = torch.ops.aten.view.default(sum_145, [384]);  sum_145 = None
    view_336: "f32[3072, 384]" = torch.ops.aten.view.default(cat_23, [3072, 384]);  cat_23 = None
    permute_385: "f32[384, 3072]" = torch.ops.aten.permute.default(view_336, [1, 0])
    mm_120: "f32[384, 196]" = torch.ops.aten.mm.default(permute_385, view_97);  permute_385 = view_97 = None
    permute_386: "f32[196, 384]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    permute_387: "f32[384, 196]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    mm_121: "f32[3072, 196]" = torch.ops.aten.mm.default(view_336, permute_387);  view_336 = permute_387 = None
    view_337: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_121, [8, 384, 196]);  mm_121 = None
    permute_388: "f32[384, 196]" = torch.ops.aten.permute.default(permute_386, [1, 0]);  permute_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_389: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
    clone_217: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_389, memory_format = torch.contiguous_format);  permute_389 = None
    clone_218: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_83, memory_format = torch.contiguous_format);  add_83 = None
    sub_145: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_218, getitem_97);  clone_218 = getitem_97 = None
    mul_482: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_24);  sub_145 = None
    mul_483: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_217, primals_147);  primals_147 = None
    mul_484: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_483, 384)
    sum_146: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True)
    mul_485: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_483, mul_482);  mul_483 = None
    sum_147: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True);  mul_485 = None
    mul_486: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_482, sum_147);  sum_147 = None
    sub_146: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_484, sum_146);  mul_484 = sum_146 = None
    sub_147: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_146, mul_486);  sub_146 = mul_486 = None
    div_25: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 384);  rsqrt_24 = None
    mul_487: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_25, sub_147);  div_25 = sub_147 = None
    mul_488: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_217, mul_482);  mul_482 = None
    sum_148: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_488, [0, 1]);  mul_488 = None
    sum_149: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_217, [0, 1]);  clone_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_217: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_215, mul_487);  add_215 = mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_338: "f32[1568, 384]" = torch.ops.aten.view.default(add_217, [1568, 384])
    permute_390: "f32[384, 768]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    mm_122: "f32[1568, 768]" = torch.ops.aten.mm.default(view_338, permute_390);  permute_390 = None
    permute_391: "f32[384, 1568]" = torch.ops.aten.permute.default(view_338, [1, 0])
    mm_123: "f32[384, 768]" = torch.ops.aten.mm.default(permute_391, view_95);  permute_391 = view_95 = None
    permute_392: "f32[768, 384]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_150: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
    view_339: "f32[384]" = torch.ops.aten.view.default(sum_150, [384]);  sum_150 = None
    permute_393: "f32[384, 768]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
    view_340: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_122, [8, 196, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_489: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_340, getitem_94);  getitem_94 = None
    mul_490: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_340, mul_94);  view_340 = mul_94 = None
    sigmoid_72: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_95)
    full_24: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_148: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_24, sigmoid_72);  full_24 = None
    mul_491: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_95, sub_148);  getitem_95 = sub_148 = None
    add_218: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_491, 1);  mul_491 = None
    mul_492: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_72, add_218);  sigmoid_72 = add_218 = None
    mul_493: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_489, mul_492);  mul_489 = mul_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_24: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_490, mul_493], 2);  mul_490 = mul_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_341: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_24, [1568, 1536]);  cat_24 = None
    permute_395: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    mm_124: "f32[1568, 384]" = torch.ops.aten.mm.default(view_341, permute_395);  permute_395 = None
    permute_396: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_341, [1, 0])
    mm_125: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_396, view_93);  permute_396 = view_93 = None
    permute_397: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_151: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
    view_342: "f32[1536]" = torch.ops.aten.view.default(sum_151, [1536]);  sum_151 = None
    permute_398: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
    view_343: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_124, [8, 196, 384]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_219: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format);  add_80 = None
    sub_149: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_219, getitem_93);  clone_219 = getitem_93 = None
    mul_494: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_149, rsqrt_23);  sub_149 = None
    mul_495: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_343, primals_141);  primals_141 = None
    mul_496: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_495, 384)
    sum_152: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True)
    mul_497: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_495, mul_494);  mul_495 = None
    sum_153: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
    mul_498: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_494, sum_153);  sum_153 = None
    sub_150: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_496, sum_152);  mul_496 = sum_152 = None
    sub_151: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_150, mul_498);  sub_150 = mul_498 = None
    div_26: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 384);  rsqrt_23 = None
    mul_499: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_26, sub_151);  div_26 = sub_151 = None
    mul_500: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_343, mul_494);  mul_494 = None
    sum_154: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1]);  mul_500 = None
    sum_155: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_343, [0, 1]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_219: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_217, mul_499);  add_217 = mul_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_399: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_219, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_220: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_399, memory_format = torch.contiguous_format);  permute_399 = None
    view_344: "f32[3072, 196]" = torch.ops.aten.view.default(clone_220, [3072, 196]);  clone_220 = None
    permute_400: "f32[196, 192]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    mm_126: "f32[3072, 192]" = torch.ops.aten.mm.default(view_344, permute_400);  permute_400 = None
    permute_401: "f32[196, 3072]" = torch.ops.aten.permute.default(view_344, [1, 0])
    mm_127: "f32[196, 192]" = torch.ops.aten.mm.default(permute_401, view_91);  permute_401 = view_91 = None
    permute_402: "f32[192, 196]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_156: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True);  view_344 = None
    view_345: "f32[196]" = torch.ops.aten.view.default(sum_156, [196]);  sum_156 = None
    permute_403: "f32[196, 192]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    view_346: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_126, [8, 384, 192]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_501: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_346, getitem_90);  getitem_90 = None
    mul_502: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_346, mul_90);  view_346 = mul_90 = None
    sigmoid_73: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_91)
    full_25: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_152: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_25, sigmoid_73);  full_25 = None
    mul_503: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_91, sub_152);  getitem_91 = sub_152 = None
    add_220: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_503, 1);  mul_503 = None
    mul_504: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_73, add_220);  sigmoid_73 = add_220 = None
    mul_505: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_501, mul_504);  mul_501 = mul_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_25: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_502, mul_505], 2);  mul_502 = mul_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_157: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_25, [0, 1], True)
    view_347: "f32[384]" = torch.ops.aten.view.default(sum_157, [384]);  sum_157 = None
    view_348: "f32[3072, 384]" = torch.ops.aten.view.default(cat_25, [3072, 384]);  cat_25 = None
    permute_405: "f32[384, 3072]" = torch.ops.aten.permute.default(view_348, [1, 0])
    mm_128: "f32[384, 196]" = torch.ops.aten.mm.default(permute_405, view_89);  permute_405 = view_89 = None
    permute_406: "f32[196, 384]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    permute_407: "f32[384, 196]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    mm_129: "f32[3072, 196]" = torch.ops.aten.mm.default(view_348, permute_407);  view_348 = permute_407 = None
    view_349: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_129, [8, 384, 196]);  mm_129 = None
    permute_408: "f32[384, 196]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_409: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
    clone_221: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_409, memory_format = torch.contiguous_format);  permute_409 = None
    clone_222: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format);  add_76 = None
    sub_153: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_222, getitem_89);  clone_222 = getitem_89 = None
    mul_506: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_153, rsqrt_22);  sub_153 = None
    mul_507: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_221, primals_135);  primals_135 = None
    mul_508: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_507, 384)
    sum_158: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_507, [2], True)
    mul_509: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_507, mul_506);  mul_507 = None
    sum_159: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [2], True);  mul_509 = None
    mul_510: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_506, sum_159);  sum_159 = None
    sub_154: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_508, sum_158);  mul_508 = sum_158 = None
    sub_155: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_154, mul_510);  sub_154 = mul_510 = None
    div_27: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 384);  rsqrt_22 = None
    mul_511: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_27, sub_155);  div_27 = sub_155 = None
    mul_512: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_221, mul_506);  mul_506 = None
    sum_160: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 1]);  mul_512 = None
    sum_161: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_221, [0, 1]);  clone_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_221: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_219, mul_511);  add_219 = mul_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_350: "f32[1568, 384]" = torch.ops.aten.view.default(add_221, [1568, 384])
    permute_410: "f32[384, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_130: "f32[1568, 768]" = torch.ops.aten.mm.default(view_350, permute_410);  permute_410 = None
    permute_411: "f32[384, 1568]" = torch.ops.aten.permute.default(view_350, [1, 0])
    mm_131: "f32[384, 768]" = torch.ops.aten.mm.default(permute_411, view_87);  permute_411 = view_87 = None
    permute_412: "f32[768, 384]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_162: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_350, [0], True);  view_350 = None
    view_351: "f32[384]" = torch.ops.aten.view.default(sum_162, [384]);  sum_162 = None
    permute_413: "f32[384, 768]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_352: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_130, [8, 196, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_513: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_352, getitem_86);  getitem_86 = None
    mul_514: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_352, mul_86);  view_352 = mul_86 = None
    sigmoid_74: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_87)
    full_26: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_156: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_26, sigmoid_74);  full_26 = None
    mul_515: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_87, sub_156);  getitem_87 = sub_156 = None
    add_222: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_515, 1);  mul_515 = None
    mul_516: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_74, add_222);  sigmoid_74 = add_222 = None
    mul_517: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_513, mul_516);  mul_513 = mul_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_26: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_514, mul_517], 2);  mul_514 = mul_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_353: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_26, [1568, 1536]);  cat_26 = None
    permute_415: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_132: "f32[1568, 384]" = torch.ops.aten.mm.default(view_353, permute_415);  permute_415 = None
    permute_416: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_133: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_416, view_85);  permute_416 = view_85 = None
    permute_417: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_163: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[1536]" = torch.ops.aten.view.default(sum_163, [1536]);  sum_163 = None
    permute_418: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    view_355: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_132, [8, 196, 384]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_223: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_73, memory_format = torch.contiguous_format);  add_73 = None
    sub_157: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_223, getitem_85);  clone_223 = getitem_85 = None
    mul_518: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_157, rsqrt_21);  sub_157 = None
    mul_519: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_355, primals_129);  primals_129 = None
    mul_520: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_519, 384)
    sum_164: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_519, [2], True)
    mul_521: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_519, mul_518);  mul_519 = None
    sum_165: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_521, [2], True);  mul_521 = None
    mul_522: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_518, sum_165);  sum_165 = None
    sub_158: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_520, sum_164);  mul_520 = sum_164 = None
    sub_159: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_158, mul_522);  sub_158 = mul_522 = None
    div_28: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 384);  rsqrt_21 = None
    mul_523: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_28, sub_159);  div_28 = sub_159 = None
    mul_524: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_355, mul_518);  mul_518 = None
    sum_166: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_524, [0, 1]);  mul_524 = None
    sum_167: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_355, [0, 1]);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_223: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_221, mul_523);  add_221 = mul_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_419: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_223, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_224: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
    view_356: "f32[3072, 196]" = torch.ops.aten.view.default(clone_224, [3072, 196]);  clone_224 = None
    permute_420: "f32[196, 192]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_134: "f32[3072, 192]" = torch.ops.aten.mm.default(view_356, permute_420);  permute_420 = None
    permute_421: "f32[196, 3072]" = torch.ops.aten.permute.default(view_356, [1, 0])
    mm_135: "f32[196, 192]" = torch.ops.aten.mm.default(permute_421, view_83);  permute_421 = view_83 = None
    permute_422: "f32[192, 196]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_168: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
    view_357: "f32[196]" = torch.ops.aten.view.default(sum_168, [196]);  sum_168 = None
    permute_423: "f32[196, 192]" = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None
    view_358: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_134, [8, 384, 192]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_525: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_358, getitem_82);  getitem_82 = None
    mul_526: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_358, mul_82);  view_358 = mul_82 = None
    sigmoid_75: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_83)
    full_27: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_160: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_27, sigmoid_75);  full_27 = None
    mul_527: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_83, sub_160);  getitem_83 = sub_160 = None
    add_224: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_527, 1);  mul_527 = None
    mul_528: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_75, add_224);  sigmoid_75 = add_224 = None
    mul_529: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_525, mul_528);  mul_525 = mul_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_27: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_526, mul_529], 2);  mul_526 = mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_169: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_27, [0, 1], True)
    view_359: "f32[384]" = torch.ops.aten.view.default(sum_169, [384]);  sum_169 = None
    view_360: "f32[3072, 384]" = torch.ops.aten.view.default(cat_27, [3072, 384]);  cat_27 = None
    permute_425: "f32[384, 3072]" = torch.ops.aten.permute.default(view_360, [1, 0])
    mm_136: "f32[384, 196]" = torch.ops.aten.mm.default(permute_425, view_81);  permute_425 = view_81 = None
    permute_426: "f32[196, 384]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    permute_427: "f32[384, 196]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    mm_137: "f32[3072, 196]" = torch.ops.aten.mm.default(view_360, permute_427);  view_360 = permute_427 = None
    view_361: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_137, [8, 384, 196]);  mm_137 = None
    permute_428: "f32[384, 196]" = torch.ops.aten.permute.default(permute_426, [1, 0]);  permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_429: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_361, [0, 2, 1]);  view_361 = None
    clone_225: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_429, memory_format = torch.contiguous_format);  permute_429 = None
    clone_226: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_69, memory_format = torch.contiguous_format);  add_69 = None
    sub_161: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_226, getitem_81);  clone_226 = getitem_81 = None
    mul_530: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_161, rsqrt_20);  sub_161 = None
    mul_531: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_225, primals_123);  primals_123 = None
    mul_532: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_531, 384)
    sum_170: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_531, [2], True)
    mul_533: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_531, mul_530);  mul_531 = None
    sum_171: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_533, [2], True);  mul_533 = None
    mul_534: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_530, sum_171);  sum_171 = None
    sub_162: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_532, sum_170);  mul_532 = sum_170 = None
    sub_163: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_162, mul_534);  sub_162 = mul_534 = None
    div_29: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 384);  rsqrt_20 = None
    mul_535: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_29, sub_163);  div_29 = sub_163 = None
    mul_536: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_225, mul_530);  mul_530 = None
    sum_172: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 1]);  mul_536 = None
    sum_173: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_225, [0, 1]);  clone_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_225: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_223, mul_535);  add_223 = mul_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_362: "f32[1568, 384]" = torch.ops.aten.view.default(add_225, [1568, 384])
    permute_430: "f32[384, 768]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    mm_138: "f32[1568, 768]" = torch.ops.aten.mm.default(view_362, permute_430);  permute_430 = None
    permute_431: "f32[384, 1568]" = torch.ops.aten.permute.default(view_362, [1, 0])
    mm_139: "f32[384, 768]" = torch.ops.aten.mm.default(permute_431, view_79);  permute_431 = view_79 = None
    permute_432: "f32[768, 384]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_174: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True);  view_362 = None
    view_363: "f32[384]" = torch.ops.aten.view.default(sum_174, [384]);  sum_174 = None
    permute_433: "f32[384, 768]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
    view_364: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_138, [8, 196, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_537: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_364, getitem_78);  getitem_78 = None
    mul_538: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_364, mul_78);  view_364 = mul_78 = None
    sigmoid_76: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_79)
    full_28: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_164: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_28, sigmoid_76);  full_28 = None
    mul_539: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_79, sub_164);  getitem_79 = sub_164 = None
    add_226: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_539, 1);  mul_539 = None
    mul_540: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_76, add_226);  sigmoid_76 = add_226 = None
    mul_541: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_537, mul_540);  mul_537 = mul_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_28: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_538, mul_541], 2);  mul_538 = mul_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_365: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_28, [1568, 1536]);  cat_28 = None
    permute_435: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_140: "f32[1568, 384]" = torch.ops.aten.mm.default(view_365, permute_435);  permute_435 = None
    permute_436: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_365, [1, 0])
    mm_141: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_436, view_77);  permute_436 = view_77 = None
    permute_437: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_175: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_365, [0], True);  view_365 = None
    view_366: "f32[1536]" = torch.ops.aten.view.default(sum_175, [1536]);  sum_175 = None
    permute_438: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    view_367: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_140, [8, 196, 384]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_227: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_66, memory_format = torch.contiguous_format);  add_66 = None
    sub_165: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_227, getitem_77);  clone_227 = getitem_77 = None
    mul_542: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_165, rsqrt_19);  sub_165 = None
    mul_543: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_367, primals_117);  primals_117 = None
    mul_544: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_543, 384)
    sum_176: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_543, [2], True)
    mul_545: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_543, mul_542);  mul_543 = None
    sum_177: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_545, [2], True);  mul_545 = None
    mul_546: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_542, sum_177);  sum_177 = None
    sub_166: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_544, sum_176);  mul_544 = sum_176 = None
    sub_167: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_166, mul_546);  sub_166 = mul_546 = None
    div_30: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 384);  rsqrt_19 = None
    mul_547: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_30, sub_167);  div_30 = sub_167 = None
    mul_548: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_367, mul_542);  mul_542 = None
    sum_178: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_548, [0, 1]);  mul_548 = None
    sum_179: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_367, [0, 1]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_227: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_225, mul_547);  add_225 = mul_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_439: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_227, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_228: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_439, memory_format = torch.contiguous_format);  permute_439 = None
    view_368: "f32[3072, 196]" = torch.ops.aten.view.default(clone_228, [3072, 196]);  clone_228 = None
    permute_440: "f32[196, 192]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_142: "f32[3072, 192]" = torch.ops.aten.mm.default(view_368, permute_440);  permute_440 = None
    permute_441: "f32[196, 3072]" = torch.ops.aten.permute.default(view_368, [1, 0])
    mm_143: "f32[196, 192]" = torch.ops.aten.mm.default(permute_441, view_75);  permute_441 = view_75 = None
    permute_442: "f32[192, 196]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_180: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_368, [0], True);  view_368 = None
    view_369: "f32[196]" = torch.ops.aten.view.default(sum_180, [196]);  sum_180 = None
    permute_443: "f32[196, 192]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    view_370: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_142, [8, 384, 192]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_549: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_370, getitem_74);  getitem_74 = None
    mul_550: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_370, mul_74);  view_370 = mul_74 = None
    sigmoid_77: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_75)
    full_29: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_168: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_29, sigmoid_77);  full_29 = None
    mul_551: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_75, sub_168);  getitem_75 = sub_168 = None
    add_228: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_551, 1);  mul_551 = None
    mul_552: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_77, add_228);  sigmoid_77 = add_228 = None
    mul_553: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_549, mul_552);  mul_549 = mul_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_29: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_550, mul_553], 2);  mul_550 = mul_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_181: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_29, [0, 1], True)
    view_371: "f32[384]" = torch.ops.aten.view.default(sum_181, [384]);  sum_181 = None
    view_372: "f32[3072, 384]" = torch.ops.aten.view.default(cat_29, [3072, 384]);  cat_29 = None
    permute_445: "f32[384, 3072]" = torch.ops.aten.permute.default(view_372, [1, 0])
    mm_144: "f32[384, 196]" = torch.ops.aten.mm.default(permute_445, view_73);  permute_445 = view_73 = None
    permute_446: "f32[196, 384]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    permute_447: "f32[384, 196]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_145: "f32[3072, 196]" = torch.ops.aten.mm.default(view_372, permute_447);  view_372 = permute_447 = None
    view_373: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_145, [8, 384, 196]);  mm_145 = None
    permute_448: "f32[384, 196]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_449: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_373, [0, 2, 1]);  view_373 = None
    clone_229: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_449, memory_format = torch.contiguous_format);  permute_449 = None
    clone_230: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format);  add_62 = None
    sub_169: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_230, getitem_73);  clone_230 = getitem_73 = None
    mul_554: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_169, rsqrt_18);  sub_169 = None
    mul_555: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_229, primals_111);  primals_111 = None
    mul_556: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_555, 384)
    sum_182: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_555, [2], True)
    mul_557: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_555, mul_554);  mul_555 = None
    sum_183: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_557, [2], True);  mul_557 = None
    mul_558: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_554, sum_183);  sum_183 = None
    sub_170: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_556, sum_182);  mul_556 = sum_182 = None
    sub_171: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_170, mul_558);  sub_170 = mul_558 = None
    div_31: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 384);  rsqrt_18 = None
    mul_559: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_31, sub_171);  div_31 = sub_171 = None
    mul_560: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_229, mul_554);  mul_554 = None
    sum_184: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_560, [0, 1]);  mul_560 = None
    sum_185: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_229, [0, 1]);  clone_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_229: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_227, mul_559);  add_227 = mul_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_374: "f32[1568, 384]" = torch.ops.aten.view.default(add_229, [1568, 384])
    permute_450: "f32[384, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_146: "f32[1568, 768]" = torch.ops.aten.mm.default(view_374, permute_450);  permute_450 = None
    permute_451: "f32[384, 1568]" = torch.ops.aten.permute.default(view_374, [1, 0])
    mm_147: "f32[384, 768]" = torch.ops.aten.mm.default(permute_451, view_71);  permute_451 = view_71 = None
    permute_452: "f32[768, 384]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_186: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_374, [0], True);  view_374 = None
    view_375: "f32[384]" = torch.ops.aten.view.default(sum_186, [384]);  sum_186 = None
    permute_453: "f32[384, 768]" = torch.ops.aten.permute.default(permute_452, [1, 0]);  permute_452 = None
    view_376: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_146, [8, 196, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_561: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_376, getitem_70);  getitem_70 = None
    mul_562: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_376, mul_70);  view_376 = mul_70 = None
    sigmoid_78: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_71)
    full_30: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_172: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_30, sigmoid_78);  full_30 = None
    mul_563: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_71, sub_172);  getitem_71 = sub_172 = None
    add_230: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_563, 1);  mul_563 = None
    mul_564: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_78, add_230);  sigmoid_78 = add_230 = None
    mul_565: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_561, mul_564);  mul_561 = mul_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_30: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_562, mul_565], 2);  mul_562 = mul_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_377: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_30, [1568, 1536]);  cat_30 = None
    permute_455: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_148: "f32[1568, 384]" = torch.ops.aten.mm.default(view_377, permute_455);  permute_455 = None
    permute_456: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_377, [1, 0])
    mm_149: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_456, view_69);  permute_456 = view_69 = None
    permute_457: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_187: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
    view_378: "f32[1536]" = torch.ops.aten.view.default(sum_187, [1536]);  sum_187 = None
    permute_458: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_457, [1, 0]);  permute_457 = None
    view_379: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_148, [8, 196, 384]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_231: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_59, memory_format = torch.contiguous_format);  add_59 = None
    sub_173: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_231, getitem_69);  clone_231 = getitem_69 = None
    mul_566: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_173, rsqrt_17);  sub_173 = None
    mul_567: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_379, primals_105);  primals_105 = None
    mul_568: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_567, 384)
    sum_188: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_567, [2], True)
    mul_569: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_567, mul_566);  mul_567 = None
    sum_189: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_569, [2], True);  mul_569 = None
    mul_570: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_566, sum_189);  sum_189 = None
    sub_174: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_568, sum_188);  mul_568 = sum_188 = None
    sub_175: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_174, mul_570);  sub_174 = mul_570 = None
    div_32: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 384);  rsqrt_17 = None
    mul_571: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_32, sub_175);  div_32 = sub_175 = None
    mul_572: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_379, mul_566);  mul_566 = None
    sum_190: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_572, [0, 1]);  mul_572 = None
    sum_191: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_379, [0, 1]);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_231: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_229, mul_571);  add_229 = mul_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_459: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_231, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_232: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    view_380: "f32[3072, 196]" = torch.ops.aten.view.default(clone_232, [3072, 196]);  clone_232 = None
    permute_460: "f32[196, 192]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    mm_150: "f32[3072, 192]" = torch.ops.aten.mm.default(view_380, permute_460);  permute_460 = None
    permute_461: "f32[196, 3072]" = torch.ops.aten.permute.default(view_380, [1, 0])
    mm_151: "f32[196, 192]" = torch.ops.aten.mm.default(permute_461, view_67);  permute_461 = view_67 = None
    permute_462: "f32[192, 196]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_192: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
    view_381: "f32[196]" = torch.ops.aten.view.default(sum_192, [196]);  sum_192 = None
    permute_463: "f32[196, 192]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    view_382: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_150, [8, 384, 192]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_573: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_382, getitem_66);  getitem_66 = None
    mul_574: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_382, mul_66);  view_382 = mul_66 = None
    sigmoid_79: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_67)
    full_31: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_176: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_31, sigmoid_79);  full_31 = None
    mul_575: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_67, sub_176);  getitem_67 = sub_176 = None
    add_232: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_575, 1);  mul_575 = None
    mul_576: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_79, add_232);  sigmoid_79 = add_232 = None
    mul_577: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_573, mul_576);  mul_573 = mul_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_31: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_574, mul_577], 2);  mul_574 = mul_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_193: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_31, [0, 1], True)
    view_383: "f32[384]" = torch.ops.aten.view.default(sum_193, [384]);  sum_193 = None
    view_384: "f32[3072, 384]" = torch.ops.aten.view.default(cat_31, [3072, 384]);  cat_31 = None
    permute_465: "f32[384, 3072]" = torch.ops.aten.permute.default(view_384, [1, 0])
    mm_152: "f32[384, 196]" = torch.ops.aten.mm.default(permute_465, view_65);  permute_465 = view_65 = None
    permute_466: "f32[196, 384]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    permute_467: "f32[384, 196]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    mm_153: "f32[3072, 196]" = torch.ops.aten.mm.default(view_384, permute_467);  view_384 = permute_467 = None
    view_385: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_153, [8, 384, 196]);  mm_153 = None
    permute_468: "f32[384, 196]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_469: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    clone_233: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_469, memory_format = torch.contiguous_format);  permute_469 = None
    clone_234: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_55, memory_format = torch.contiguous_format);  add_55 = None
    sub_177: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_234, getitem_65);  clone_234 = getitem_65 = None
    mul_578: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_177, rsqrt_16);  sub_177 = None
    mul_579: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_233, primals_99);  primals_99 = None
    mul_580: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_579, 384)
    sum_194: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_579, [2], True)
    mul_581: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_579, mul_578);  mul_579 = None
    sum_195: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [2], True);  mul_581 = None
    mul_582: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_578, sum_195);  sum_195 = None
    sub_178: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_580, sum_194);  mul_580 = sum_194 = None
    sub_179: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_178, mul_582);  sub_178 = mul_582 = None
    div_33: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 384);  rsqrt_16 = None
    mul_583: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_33, sub_179);  div_33 = sub_179 = None
    mul_584: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_233, mul_578);  mul_578 = None
    sum_196: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 1]);  mul_584 = None
    sum_197: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_233, [0, 1]);  clone_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_233: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_231, mul_583);  add_231 = mul_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_386: "f32[1568, 384]" = torch.ops.aten.view.default(add_233, [1568, 384])
    permute_470: "f32[384, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_154: "f32[1568, 768]" = torch.ops.aten.mm.default(view_386, permute_470);  permute_470 = None
    permute_471: "f32[384, 1568]" = torch.ops.aten.permute.default(view_386, [1, 0])
    mm_155: "f32[384, 768]" = torch.ops.aten.mm.default(permute_471, view_63);  permute_471 = view_63 = None
    permute_472: "f32[768, 384]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_198: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
    view_387: "f32[384]" = torch.ops.aten.view.default(sum_198, [384]);  sum_198 = None
    permute_473: "f32[384, 768]" = torch.ops.aten.permute.default(permute_472, [1, 0]);  permute_472 = None
    view_388: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_154, [8, 196, 768]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_585: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_388, getitem_62);  getitem_62 = None
    mul_586: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_388, mul_62);  view_388 = mul_62 = None
    sigmoid_80: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_63)
    full_32: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_180: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_32, sigmoid_80);  full_32 = None
    mul_587: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_63, sub_180);  getitem_63 = sub_180 = None
    add_234: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_587, 1);  mul_587 = None
    mul_588: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_80, add_234);  sigmoid_80 = add_234 = None
    mul_589: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_585, mul_588);  mul_585 = mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_32: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_586, mul_589], 2);  mul_586 = mul_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_389: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_32, [1568, 1536]);  cat_32 = None
    permute_475: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_156: "f32[1568, 384]" = torch.ops.aten.mm.default(view_389, permute_475);  permute_475 = None
    permute_476: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_389, [1, 0])
    mm_157: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_476, view_61);  permute_476 = view_61 = None
    permute_477: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_199: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
    view_390: "f32[1536]" = torch.ops.aten.view.default(sum_199, [1536]);  sum_199 = None
    permute_478: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_391: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_156, [8, 196, 384]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_235: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_52, memory_format = torch.contiguous_format);  add_52 = None
    sub_181: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_235, getitem_61);  clone_235 = getitem_61 = None
    mul_590: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_181, rsqrt_15);  sub_181 = None
    mul_591: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_391, primals_93);  primals_93 = None
    mul_592: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_591, 384)
    sum_200: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_591, [2], True)
    mul_593: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_591, mul_590);  mul_591 = None
    sum_201: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_593, [2], True);  mul_593 = None
    mul_594: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_590, sum_201);  sum_201 = None
    sub_182: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_592, sum_200);  mul_592 = sum_200 = None
    sub_183: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_182, mul_594);  sub_182 = mul_594 = None
    div_34: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 384);  rsqrt_15 = None
    mul_595: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_34, sub_183);  div_34 = sub_183 = None
    mul_596: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_391, mul_590);  mul_590 = None
    sum_202: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_596, [0, 1]);  mul_596 = None
    sum_203: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_391, [0, 1]);  view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_235: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_233, mul_595);  add_233 = mul_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_479: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_235, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_236: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
    view_392: "f32[3072, 196]" = torch.ops.aten.view.default(clone_236, [3072, 196]);  clone_236 = None
    permute_480: "f32[196, 192]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_158: "f32[3072, 192]" = torch.ops.aten.mm.default(view_392, permute_480);  permute_480 = None
    permute_481: "f32[196, 3072]" = torch.ops.aten.permute.default(view_392, [1, 0])
    mm_159: "f32[196, 192]" = torch.ops.aten.mm.default(permute_481, view_59);  permute_481 = view_59 = None
    permute_482: "f32[192, 196]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_204: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_392, [0], True);  view_392 = None
    view_393: "f32[196]" = torch.ops.aten.view.default(sum_204, [196]);  sum_204 = None
    permute_483: "f32[196, 192]" = torch.ops.aten.permute.default(permute_482, [1, 0]);  permute_482 = None
    view_394: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_158, [8, 384, 192]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_597: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_394, getitem_58);  getitem_58 = None
    mul_598: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_394, mul_58);  view_394 = mul_58 = None
    sigmoid_81: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_59)
    full_33: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_184: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_33, sigmoid_81);  full_33 = None
    mul_599: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_59, sub_184);  getitem_59 = sub_184 = None
    add_236: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_599, 1);  mul_599 = None
    mul_600: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_81, add_236);  sigmoid_81 = add_236 = None
    mul_601: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_597, mul_600);  mul_597 = mul_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_33: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_598, mul_601], 2);  mul_598 = mul_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_205: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_33, [0, 1], True)
    view_395: "f32[384]" = torch.ops.aten.view.default(sum_205, [384]);  sum_205 = None
    view_396: "f32[3072, 384]" = torch.ops.aten.view.default(cat_33, [3072, 384]);  cat_33 = None
    permute_485: "f32[384, 3072]" = torch.ops.aten.permute.default(view_396, [1, 0])
    mm_160: "f32[384, 196]" = torch.ops.aten.mm.default(permute_485, view_57);  permute_485 = view_57 = None
    permute_486: "f32[196, 384]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    permute_487: "f32[384, 196]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_161: "f32[3072, 196]" = torch.ops.aten.mm.default(view_396, permute_487);  view_396 = permute_487 = None
    view_397: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_161, [8, 384, 196]);  mm_161 = None
    permute_488: "f32[384, 196]" = torch.ops.aten.permute.default(permute_486, [1, 0]);  permute_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_489: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_397, [0, 2, 1]);  view_397 = None
    clone_237: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
    clone_238: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_48, memory_format = torch.contiguous_format);  add_48 = None
    sub_185: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_238, getitem_57);  clone_238 = getitem_57 = None
    mul_602: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_185, rsqrt_14);  sub_185 = None
    mul_603: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_237, primals_87);  primals_87 = None
    mul_604: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_603, 384)
    sum_206: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_603, [2], True)
    mul_605: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_603, mul_602);  mul_603 = None
    sum_207: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_605, [2], True);  mul_605 = None
    mul_606: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_602, sum_207);  sum_207 = None
    sub_186: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_604, sum_206);  mul_604 = sum_206 = None
    sub_187: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_186, mul_606);  sub_186 = mul_606 = None
    div_35: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 384);  rsqrt_14 = None
    mul_607: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_35, sub_187);  div_35 = sub_187 = None
    mul_608: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_237, mul_602);  mul_602 = None
    sum_208: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 1]);  mul_608 = None
    sum_209: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_237, [0, 1]);  clone_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_237: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_235, mul_607);  add_235 = mul_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_398: "f32[1568, 384]" = torch.ops.aten.view.default(add_237, [1568, 384])
    permute_490: "f32[384, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_162: "f32[1568, 768]" = torch.ops.aten.mm.default(view_398, permute_490);  permute_490 = None
    permute_491: "f32[384, 1568]" = torch.ops.aten.permute.default(view_398, [1, 0])
    mm_163: "f32[384, 768]" = torch.ops.aten.mm.default(permute_491, view_55);  permute_491 = view_55 = None
    permute_492: "f32[768, 384]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_210: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_398, [0], True);  view_398 = None
    view_399: "f32[384]" = torch.ops.aten.view.default(sum_210, [384]);  sum_210 = None
    permute_493: "f32[384, 768]" = torch.ops.aten.permute.default(permute_492, [1, 0]);  permute_492 = None
    view_400: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_162, [8, 196, 768]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_609: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_400, getitem_54);  getitem_54 = None
    mul_610: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_400, mul_54);  view_400 = mul_54 = None
    sigmoid_82: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_55)
    full_34: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_188: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_34, sigmoid_82);  full_34 = None
    mul_611: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_55, sub_188);  getitem_55 = sub_188 = None
    add_238: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_611, 1);  mul_611 = None
    mul_612: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_82, add_238);  sigmoid_82 = add_238 = None
    mul_613: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_609, mul_612);  mul_609 = mul_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_34: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_610, mul_613], 2);  mul_610 = mul_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_401: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_34, [1568, 1536]);  cat_34 = None
    permute_495: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_164: "f32[1568, 384]" = torch.ops.aten.mm.default(view_401, permute_495);  permute_495 = None
    permute_496: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_401, [1, 0])
    mm_165: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_496, view_53);  permute_496 = view_53 = None
    permute_497: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_211: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_401, [0], True);  view_401 = None
    view_402: "f32[1536]" = torch.ops.aten.view.default(sum_211, [1536]);  sum_211 = None
    permute_498: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_497, [1, 0]);  permute_497 = None
    view_403: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_164, [8, 196, 384]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_239: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_45, memory_format = torch.contiguous_format);  add_45 = None
    sub_189: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_239, getitem_53);  clone_239 = getitem_53 = None
    mul_614: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_189, rsqrt_13);  sub_189 = None
    mul_615: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_403, primals_81);  primals_81 = None
    mul_616: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_615, 384)
    sum_212: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_615, [2], True)
    mul_617: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_615, mul_614);  mul_615 = None
    sum_213: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_617, [2], True);  mul_617 = None
    mul_618: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_614, sum_213);  sum_213 = None
    sub_190: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_616, sum_212);  mul_616 = sum_212 = None
    sub_191: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_190, mul_618);  sub_190 = mul_618 = None
    div_36: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 384);  rsqrt_13 = None
    mul_619: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_36, sub_191);  div_36 = sub_191 = None
    mul_620: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_403, mul_614);  mul_614 = None
    sum_214: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 1]);  mul_620 = None
    sum_215: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_403, [0, 1]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_239: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_237, mul_619);  add_237 = mul_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_499: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_239, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_240: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_499, memory_format = torch.contiguous_format);  permute_499 = None
    view_404: "f32[3072, 196]" = torch.ops.aten.view.default(clone_240, [3072, 196]);  clone_240 = None
    permute_500: "f32[196, 192]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    mm_166: "f32[3072, 192]" = torch.ops.aten.mm.default(view_404, permute_500);  permute_500 = None
    permute_501: "f32[196, 3072]" = torch.ops.aten.permute.default(view_404, [1, 0])
    mm_167: "f32[196, 192]" = torch.ops.aten.mm.default(permute_501, view_51);  permute_501 = view_51 = None
    permute_502: "f32[192, 196]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    sum_216: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_404, [0], True);  view_404 = None
    view_405: "f32[196]" = torch.ops.aten.view.default(sum_216, [196]);  sum_216 = None
    permute_503: "f32[196, 192]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    view_406: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_166, [8, 384, 192]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_621: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_406, getitem_50);  getitem_50 = None
    mul_622: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_406, mul_50);  view_406 = mul_50 = None
    sigmoid_83: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_51)
    full_35: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_192: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_35, sigmoid_83);  full_35 = None
    mul_623: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_51, sub_192);  getitem_51 = sub_192 = None
    add_240: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_623, 1);  mul_623 = None
    mul_624: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_83, add_240);  sigmoid_83 = add_240 = None
    mul_625: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_621, mul_624);  mul_621 = mul_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_35: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_622, mul_625], 2);  mul_622 = mul_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_217: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_35, [0, 1], True)
    view_407: "f32[384]" = torch.ops.aten.view.default(sum_217, [384]);  sum_217 = None
    view_408: "f32[3072, 384]" = torch.ops.aten.view.default(cat_35, [3072, 384]);  cat_35 = None
    permute_505: "f32[384, 3072]" = torch.ops.aten.permute.default(view_408, [1, 0])
    mm_168: "f32[384, 196]" = torch.ops.aten.mm.default(permute_505, view_49);  permute_505 = view_49 = None
    permute_506: "f32[196, 384]" = torch.ops.aten.permute.default(mm_168, [1, 0]);  mm_168 = None
    permute_507: "f32[384, 196]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    mm_169: "f32[3072, 196]" = torch.ops.aten.mm.default(view_408, permute_507);  view_408 = permute_507 = None
    view_409: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_169, [8, 384, 196]);  mm_169 = None
    permute_508: "f32[384, 196]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_509: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None
    clone_241: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_509, memory_format = torch.contiguous_format);  permute_509 = None
    clone_242: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format);  add_41 = None
    sub_193: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_242, getitem_49);  clone_242 = getitem_49 = None
    mul_626: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_193, rsqrt_12);  sub_193 = None
    mul_627: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_241, primals_75);  primals_75 = None
    mul_628: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_627, 384)
    sum_218: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_627, [2], True)
    mul_629: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_627, mul_626);  mul_627 = None
    sum_219: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_629, [2], True);  mul_629 = None
    mul_630: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_626, sum_219);  sum_219 = None
    sub_194: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_628, sum_218);  mul_628 = sum_218 = None
    sub_195: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_194, mul_630);  sub_194 = mul_630 = None
    div_37: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 384);  rsqrt_12 = None
    mul_631: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_37, sub_195);  div_37 = sub_195 = None
    mul_632: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_241, mul_626);  mul_626 = None
    sum_220: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_632, [0, 1]);  mul_632 = None
    sum_221: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_241, [0, 1]);  clone_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_241: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_239, mul_631);  add_239 = mul_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_410: "f32[1568, 384]" = torch.ops.aten.view.default(add_241, [1568, 384])
    permute_510: "f32[384, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_170: "f32[1568, 768]" = torch.ops.aten.mm.default(view_410, permute_510);  permute_510 = None
    permute_511: "f32[384, 1568]" = torch.ops.aten.permute.default(view_410, [1, 0])
    mm_171: "f32[384, 768]" = torch.ops.aten.mm.default(permute_511, view_47);  permute_511 = view_47 = None
    permute_512: "f32[768, 384]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_222: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_410, [0], True);  view_410 = None
    view_411: "f32[384]" = torch.ops.aten.view.default(sum_222, [384]);  sum_222 = None
    permute_513: "f32[384, 768]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    view_412: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_170, [8, 196, 768]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_633: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_412, getitem_46);  getitem_46 = None
    mul_634: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_412, mul_46);  view_412 = mul_46 = None
    sigmoid_84: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_47)
    full_36: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_196: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_36, sigmoid_84);  full_36 = None
    mul_635: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_47, sub_196);  getitem_47 = sub_196 = None
    add_242: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_635, 1);  mul_635 = None
    mul_636: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_84, add_242);  sigmoid_84 = add_242 = None
    mul_637: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_633, mul_636);  mul_633 = mul_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_36: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_634, mul_637], 2);  mul_634 = mul_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_413: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_36, [1568, 1536]);  cat_36 = None
    permute_515: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_172: "f32[1568, 384]" = torch.ops.aten.mm.default(view_413, permute_515);  permute_515 = None
    permute_516: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_413, [1, 0])
    mm_173: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_516, view_45);  permute_516 = view_45 = None
    permute_517: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_223: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_413, [0], True);  view_413 = None
    view_414: "f32[1536]" = torch.ops.aten.view.default(sum_223, [1536]);  sum_223 = None
    permute_518: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_517, [1, 0]);  permute_517 = None
    view_415: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_172, [8, 196, 384]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_243: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_38, memory_format = torch.contiguous_format);  add_38 = None
    sub_197: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_243, getitem_45);  clone_243 = getitem_45 = None
    mul_638: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_197, rsqrt_11);  sub_197 = None
    mul_639: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_415, primals_69);  primals_69 = None
    mul_640: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_639, 384)
    sum_224: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_639, [2], True)
    mul_641: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_639, mul_638);  mul_639 = None
    sum_225: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_641, [2], True);  mul_641 = None
    mul_642: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_638, sum_225);  sum_225 = None
    sub_198: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_640, sum_224);  mul_640 = sum_224 = None
    sub_199: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_198, mul_642);  sub_198 = mul_642 = None
    div_38: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 384);  rsqrt_11 = None
    mul_643: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_38, sub_199);  div_38 = sub_199 = None
    mul_644: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_415, mul_638);  mul_638 = None
    sum_226: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_644, [0, 1]);  mul_644 = None
    sum_227: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_415, [0, 1]);  view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_243: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_241, mul_643);  add_241 = mul_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_519: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_243, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_244: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
    view_416: "f32[3072, 196]" = torch.ops.aten.view.default(clone_244, [3072, 196]);  clone_244 = None
    permute_520: "f32[196, 192]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_174: "f32[3072, 192]" = torch.ops.aten.mm.default(view_416, permute_520);  permute_520 = None
    permute_521: "f32[196, 3072]" = torch.ops.aten.permute.default(view_416, [1, 0])
    mm_175: "f32[196, 192]" = torch.ops.aten.mm.default(permute_521, view_43);  permute_521 = view_43 = None
    permute_522: "f32[192, 196]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_228: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
    view_417: "f32[196]" = torch.ops.aten.view.default(sum_228, [196]);  sum_228 = None
    permute_523: "f32[196, 192]" = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
    view_418: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_174, [8, 384, 192]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_645: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_418, getitem_42);  getitem_42 = None
    mul_646: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_418, mul_42);  view_418 = mul_42 = None
    sigmoid_85: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_43)
    full_37: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_200: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_37, sigmoid_85);  full_37 = None
    mul_647: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_43, sub_200);  getitem_43 = sub_200 = None
    add_244: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_647, 1);  mul_647 = None
    mul_648: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_85, add_244);  sigmoid_85 = add_244 = None
    mul_649: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_645, mul_648);  mul_645 = mul_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_37: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_646, mul_649], 2);  mul_646 = mul_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_229: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_37, [0, 1], True)
    view_419: "f32[384]" = torch.ops.aten.view.default(sum_229, [384]);  sum_229 = None
    view_420: "f32[3072, 384]" = torch.ops.aten.view.default(cat_37, [3072, 384]);  cat_37 = None
    permute_525: "f32[384, 3072]" = torch.ops.aten.permute.default(view_420, [1, 0])
    mm_176: "f32[384, 196]" = torch.ops.aten.mm.default(permute_525, view_41);  permute_525 = view_41 = None
    permute_526: "f32[196, 384]" = torch.ops.aten.permute.default(mm_176, [1, 0]);  mm_176 = None
    permute_527: "f32[384, 196]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_177: "f32[3072, 196]" = torch.ops.aten.mm.default(view_420, permute_527);  view_420 = permute_527 = None
    view_421: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_177, [8, 384, 196]);  mm_177 = None
    permute_528: "f32[384, 196]" = torch.ops.aten.permute.default(permute_526, [1, 0]);  permute_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_529: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_421, [0, 2, 1]);  view_421 = None
    clone_245: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_529, memory_format = torch.contiguous_format);  permute_529 = None
    clone_246: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_34, memory_format = torch.contiguous_format);  add_34 = None
    sub_201: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_246, getitem_41);  clone_246 = getitem_41 = None
    mul_650: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_201, rsqrt_10);  sub_201 = None
    mul_651: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_245, primals_63);  primals_63 = None
    mul_652: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_651, 384)
    sum_230: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_651, [2], True)
    mul_653: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_651, mul_650);  mul_651 = None
    sum_231: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_653, [2], True);  mul_653 = None
    mul_654: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_650, sum_231);  sum_231 = None
    sub_202: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_652, sum_230);  mul_652 = sum_230 = None
    sub_203: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_202, mul_654);  sub_202 = mul_654 = None
    div_39: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 384);  rsqrt_10 = None
    mul_655: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_39, sub_203);  div_39 = sub_203 = None
    mul_656: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_245, mul_650);  mul_650 = None
    sum_232: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_656, [0, 1]);  mul_656 = None
    sum_233: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_245, [0, 1]);  clone_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_245: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_243, mul_655);  add_243 = mul_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_422: "f32[1568, 384]" = torch.ops.aten.view.default(add_245, [1568, 384])
    permute_530: "f32[384, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_178: "f32[1568, 768]" = torch.ops.aten.mm.default(view_422, permute_530);  permute_530 = None
    permute_531: "f32[384, 1568]" = torch.ops.aten.permute.default(view_422, [1, 0])
    mm_179: "f32[384, 768]" = torch.ops.aten.mm.default(permute_531, view_39);  permute_531 = view_39 = None
    permute_532: "f32[768, 384]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    sum_234: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
    view_423: "f32[384]" = torch.ops.aten.view.default(sum_234, [384]);  sum_234 = None
    permute_533: "f32[384, 768]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    view_424: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_178, [8, 196, 768]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_657: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_424, getitem_38);  getitem_38 = None
    mul_658: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_424, mul_38);  view_424 = mul_38 = None
    sigmoid_86: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_39)
    full_38: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_204: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_38, sigmoid_86);  full_38 = None
    mul_659: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_39, sub_204);  getitem_39 = sub_204 = None
    add_246: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_659, 1);  mul_659 = None
    mul_660: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_86, add_246);  sigmoid_86 = add_246 = None
    mul_661: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_657, mul_660);  mul_657 = mul_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_38: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_658, mul_661], 2);  mul_658 = mul_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_425: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_38, [1568, 1536]);  cat_38 = None
    permute_535: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    mm_180: "f32[1568, 384]" = torch.ops.aten.mm.default(view_425, permute_535);  permute_535 = None
    permute_536: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_425, [1, 0])
    mm_181: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_536, view_37);  permute_536 = view_37 = None
    permute_537: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_235: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
    view_426: "f32[1536]" = torch.ops.aten.view.default(sum_235, [1536]);  sum_235 = None
    permute_538: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_537, [1, 0]);  permute_537 = None
    view_427: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_180, [8, 196, 384]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_247: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_31, memory_format = torch.contiguous_format);  add_31 = None
    sub_205: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_247, getitem_37);  clone_247 = getitem_37 = None
    mul_662: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_205, rsqrt_9);  sub_205 = None
    mul_663: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_427, primals_57);  primals_57 = None
    mul_664: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_663, 384)
    sum_236: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_663, [2], True)
    mul_665: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_663, mul_662);  mul_663 = None
    sum_237: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_665, [2], True);  mul_665 = None
    mul_666: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_662, sum_237);  sum_237 = None
    sub_206: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_664, sum_236);  mul_664 = sum_236 = None
    sub_207: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_206, mul_666);  sub_206 = mul_666 = None
    div_40: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 384);  rsqrt_9 = None
    mul_667: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_40, sub_207);  div_40 = sub_207 = None
    mul_668: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_427, mul_662);  mul_662 = None
    sum_238: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 1]);  mul_668 = None
    sum_239: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_427, [0, 1]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_247: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_245, mul_667);  add_245 = mul_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_539: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_247, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_248: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_539, memory_format = torch.contiguous_format);  permute_539 = None
    view_428: "f32[3072, 196]" = torch.ops.aten.view.default(clone_248, [3072, 196]);  clone_248 = None
    permute_540: "f32[196, 192]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    mm_182: "f32[3072, 192]" = torch.ops.aten.mm.default(view_428, permute_540);  permute_540 = None
    permute_541: "f32[196, 3072]" = torch.ops.aten.permute.default(view_428, [1, 0])
    mm_183: "f32[196, 192]" = torch.ops.aten.mm.default(permute_541, view_35);  permute_541 = view_35 = None
    permute_542: "f32[192, 196]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_240: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_428, [0], True);  view_428 = None
    view_429: "f32[196]" = torch.ops.aten.view.default(sum_240, [196]);  sum_240 = None
    permute_543: "f32[196, 192]" = torch.ops.aten.permute.default(permute_542, [1, 0]);  permute_542 = None
    view_430: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_182, [8, 384, 192]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_669: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_430, getitem_34);  getitem_34 = None
    mul_670: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_430, mul_34);  view_430 = mul_34 = None
    sigmoid_87: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_35)
    full_39: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_208: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_39, sigmoid_87);  full_39 = None
    mul_671: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_35, sub_208);  getitem_35 = sub_208 = None
    add_248: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_671, 1);  mul_671 = None
    mul_672: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_87, add_248);  sigmoid_87 = add_248 = None
    mul_673: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_669, mul_672);  mul_669 = mul_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_39: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_670, mul_673], 2);  mul_670 = mul_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_241: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_39, [0, 1], True)
    view_431: "f32[384]" = torch.ops.aten.view.default(sum_241, [384]);  sum_241 = None
    view_432: "f32[3072, 384]" = torch.ops.aten.view.default(cat_39, [3072, 384]);  cat_39 = None
    permute_545: "f32[384, 3072]" = torch.ops.aten.permute.default(view_432, [1, 0])
    mm_184: "f32[384, 196]" = torch.ops.aten.mm.default(permute_545, view_33);  permute_545 = view_33 = None
    permute_546: "f32[196, 384]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    permute_547: "f32[384, 196]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    mm_185: "f32[3072, 196]" = torch.ops.aten.mm.default(view_432, permute_547);  view_432 = permute_547 = None
    view_433: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_185, [8, 384, 196]);  mm_185 = None
    permute_548: "f32[384, 196]" = torch.ops.aten.permute.default(permute_546, [1, 0]);  permute_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_549: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_433, [0, 2, 1]);  view_433 = None
    clone_249: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_549, memory_format = torch.contiguous_format);  permute_549 = None
    clone_250: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_27, memory_format = torch.contiguous_format);  add_27 = None
    sub_209: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_250, getitem_33);  clone_250 = getitem_33 = None
    mul_674: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_209, rsqrt_8);  sub_209 = None
    mul_675: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_249, primals_51);  primals_51 = None
    mul_676: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_675, 384)
    sum_242: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_675, [2], True)
    mul_677: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_675, mul_674);  mul_675 = None
    sum_243: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_677, [2], True);  mul_677 = None
    mul_678: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_674, sum_243);  sum_243 = None
    sub_210: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_676, sum_242);  mul_676 = sum_242 = None
    sub_211: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_210, mul_678);  sub_210 = mul_678 = None
    div_41: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 384);  rsqrt_8 = None
    mul_679: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_41, sub_211);  div_41 = sub_211 = None
    mul_680: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_249, mul_674);  mul_674 = None
    sum_244: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 1]);  mul_680 = None
    sum_245: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_249, [0, 1]);  clone_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_249: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_247, mul_679);  add_247 = mul_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_434: "f32[1568, 384]" = torch.ops.aten.view.default(add_249, [1568, 384])
    permute_550: "f32[384, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_186: "f32[1568, 768]" = torch.ops.aten.mm.default(view_434, permute_550);  permute_550 = None
    permute_551: "f32[384, 1568]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_187: "f32[384, 768]" = torch.ops.aten.mm.default(permute_551, view_31);  permute_551 = view_31 = None
    permute_552: "f32[768, 384]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_246: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[384]" = torch.ops.aten.view.default(sum_246, [384]);  sum_246 = None
    permute_553: "f32[384, 768]" = torch.ops.aten.permute.default(permute_552, [1, 0]);  permute_552 = None
    view_436: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_186, [8, 196, 768]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_681: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_436, getitem_30);  getitem_30 = None
    mul_682: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_436, mul_30);  view_436 = mul_30 = None
    sigmoid_88: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_31)
    full_40: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_212: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_40, sigmoid_88);  full_40 = None
    mul_683: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_31, sub_212);  getitem_31 = sub_212 = None
    add_250: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_683, 1);  mul_683 = None
    mul_684: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_88, add_250);  sigmoid_88 = add_250 = None
    mul_685: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_681, mul_684);  mul_681 = mul_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_40: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_682, mul_685], 2);  mul_682 = mul_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_437: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_40, [1568, 1536]);  cat_40 = None
    permute_555: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_188: "f32[1568, 384]" = torch.ops.aten.mm.default(view_437, permute_555);  permute_555 = None
    permute_556: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_437, [1, 0])
    mm_189: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_556, view_29);  permute_556 = view_29 = None
    permute_557: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_247: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_437, [0], True);  view_437 = None
    view_438: "f32[1536]" = torch.ops.aten.view.default(sum_247, [1536]);  sum_247 = None
    permute_558: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_557, [1, 0]);  permute_557 = None
    view_439: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_188, [8, 196, 384]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_251: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_24, memory_format = torch.contiguous_format);  add_24 = None
    sub_213: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_251, getitem_29);  clone_251 = getitem_29 = None
    mul_686: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_213, rsqrt_7);  sub_213 = None
    mul_687: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_439, primals_45);  primals_45 = None
    mul_688: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_687, 384)
    sum_248: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_687, [2], True)
    mul_689: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_687, mul_686);  mul_687 = None
    sum_249: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_689, [2], True);  mul_689 = None
    mul_690: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_686, sum_249);  sum_249 = None
    sub_214: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_688, sum_248);  mul_688 = sum_248 = None
    sub_215: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_214, mul_690);  sub_214 = mul_690 = None
    div_42: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 384);  rsqrt_7 = None
    mul_691: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_42, sub_215);  div_42 = sub_215 = None
    mul_692: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_439, mul_686);  mul_686 = None
    sum_250: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_692, [0, 1]);  mul_692 = None
    sum_251: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_439, [0, 1]);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_251: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_249, mul_691);  add_249 = mul_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_559: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_251, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_252: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
    view_440: "f32[3072, 196]" = torch.ops.aten.view.default(clone_252, [3072, 196]);  clone_252 = None
    permute_560: "f32[196, 192]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_190: "f32[3072, 192]" = torch.ops.aten.mm.default(view_440, permute_560);  permute_560 = None
    permute_561: "f32[196, 3072]" = torch.ops.aten.permute.default(view_440, [1, 0])
    mm_191: "f32[196, 192]" = torch.ops.aten.mm.default(permute_561, view_27);  permute_561 = view_27 = None
    permute_562: "f32[192, 196]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    sum_252: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
    view_441: "f32[196]" = torch.ops.aten.view.default(sum_252, [196]);  sum_252 = None
    permute_563: "f32[196, 192]" = torch.ops.aten.permute.default(permute_562, [1, 0]);  permute_562 = None
    view_442: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_190, [8, 384, 192]);  mm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_693: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_442, getitem_26);  getitem_26 = None
    mul_694: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_442, mul_26);  view_442 = mul_26 = None
    sigmoid_89: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_27)
    full_41: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_216: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_41, sigmoid_89);  full_41 = None
    mul_695: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_27, sub_216);  getitem_27 = sub_216 = None
    add_252: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_695, 1);  mul_695 = None
    mul_696: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_89, add_252);  sigmoid_89 = add_252 = None
    mul_697: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_693, mul_696);  mul_693 = mul_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_41: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_694, mul_697], 2);  mul_694 = mul_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_253: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_41, [0, 1], True)
    view_443: "f32[384]" = torch.ops.aten.view.default(sum_253, [384]);  sum_253 = None
    view_444: "f32[3072, 384]" = torch.ops.aten.view.default(cat_41, [3072, 384]);  cat_41 = None
    permute_565: "f32[384, 3072]" = torch.ops.aten.permute.default(view_444, [1, 0])
    mm_192: "f32[384, 196]" = torch.ops.aten.mm.default(permute_565, view_25);  permute_565 = view_25 = None
    permute_566: "f32[196, 384]" = torch.ops.aten.permute.default(mm_192, [1, 0]);  mm_192 = None
    permute_567: "f32[384, 196]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_193: "f32[3072, 196]" = torch.ops.aten.mm.default(view_444, permute_567);  view_444 = permute_567 = None
    view_445: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_193, [8, 384, 196]);  mm_193 = None
    permute_568: "f32[384, 196]" = torch.ops.aten.permute.default(permute_566, [1, 0]);  permute_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_569: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_445, [0, 2, 1]);  view_445 = None
    clone_253: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_569, memory_format = torch.contiguous_format);  permute_569 = None
    clone_254: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format);  add_20 = None
    sub_217: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_254, getitem_25);  clone_254 = getitem_25 = None
    mul_698: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_217, rsqrt_6);  sub_217 = None
    mul_699: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_253, primals_39);  primals_39 = None
    mul_700: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_699, 384)
    sum_254: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_699, [2], True)
    mul_701: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_699, mul_698);  mul_699 = None
    sum_255: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_701, [2], True);  mul_701 = None
    mul_702: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_698, sum_255);  sum_255 = None
    sub_218: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_700, sum_254);  mul_700 = sum_254 = None
    sub_219: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_218, mul_702);  sub_218 = mul_702 = None
    div_43: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 384);  rsqrt_6 = None
    mul_703: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_43, sub_219);  div_43 = sub_219 = None
    mul_704: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_253, mul_698);  mul_698 = None
    sum_256: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 1]);  mul_704 = None
    sum_257: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_253, [0, 1]);  clone_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_253: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_251, mul_703);  add_251 = mul_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_446: "f32[1568, 384]" = torch.ops.aten.view.default(add_253, [1568, 384])
    permute_570: "f32[384, 768]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_194: "f32[1568, 768]" = torch.ops.aten.mm.default(view_446, permute_570);  permute_570 = None
    permute_571: "f32[384, 1568]" = torch.ops.aten.permute.default(view_446, [1, 0])
    mm_195: "f32[384, 768]" = torch.ops.aten.mm.default(permute_571, view_23);  permute_571 = view_23 = None
    permute_572: "f32[768, 384]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_258: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_446, [0], True);  view_446 = None
    view_447: "f32[384]" = torch.ops.aten.view.default(sum_258, [384]);  sum_258 = None
    permute_573: "f32[384, 768]" = torch.ops.aten.permute.default(permute_572, [1, 0]);  permute_572 = None
    view_448: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_194, [8, 196, 768]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_705: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_448, getitem_22);  getitem_22 = None
    mul_706: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_448, mul_22);  view_448 = mul_22 = None
    sigmoid_90: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_23)
    full_42: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_220: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_42, sigmoid_90);  full_42 = None
    mul_707: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_23, sub_220);  getitem_23 = sub_220 = None
    add_254: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_707, 1);  mul_707 = None
    mul_708: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_90, add_254);  sigmoid_90 = add_254 = None
    mul_709: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_705, mul_708);  mul_705 = mul_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_42: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_706, mul_709], 2);  mul_706 = mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_449: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_42, [1568, 1536]);  cat_42 = None
    permute_575: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    mm_196: "f32[1568, 384]" = torch.ops.aten.mm.default(view_449, permute_575);  permute_575 = None
    permute_576: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_449, [1, 0])
    mm_197: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_576, view_21);  permute_576 = view_21 = None
    permute_577: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    sum_259: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_449, [0], True);  view_449 = None
    view_450: "f32[1536]" = torch.ops.aten.view.default(sum_259, [1536]);  sum_259 = None
    permute_578: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_577, [1, 0]);  permute_577 = None
    view_451: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_196, [8, 196, 384]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_255: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_17, memory_format = torch.contiguous_format);  add_17 = None
    sub_221: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_255, getitem_21);  clone_255 = getitem_21 = None
    mul_710: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_221, rsqrt_5);  sub_221 = None
    mul_711: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_451, primals_33);  primals_33 = None
    mul_712: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_711, 384)
    sum_260: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_711, [2], True)
    mul_713: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_711, mul_710);  mul_711 = None
    sum_261: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_713, [2], True);  mul_713 = None
    mul_714: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_710, sum_261);  sum_261 = None
    sub_222: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_712, sum_260);  mul_712 = sum_260 = None
    sub_223: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_222, mul_714);  sub_222 = mul_714 = None
    div_44: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 384);  rsqrt_5 = None
    mul_715: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_44, sub_223);  div_44 = sub_223 = None
    mul_716: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_451, mul_710);  mul_710 = None
    sum_262: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_716, [0, 1]);  mul_716 = None
    sum_263: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_451, [0, 1]);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_255: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_253, mul_715);  add_253 = mul_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_579: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_255, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_256: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_579, memory_format = torch.contiguous_format);  permute_579 = None
    view_452: "f32[3072, 196]" = torch.ops.aten.view.default(clone_256, [3072, 196]);  clone_256 = None
    permute_580: "f32[196, 192]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    mm_198: "f32[3072, 192]" = torch.ops.aten.mm.default(view_452, permute_580);  permute_580 = None
    permute_581: "f32[196, 3072]" = torch.ops.aten.permute.default(view_452, [1, 0])
    mm_199: "f32[196, 192]" = torch.ops.aten.mm.default(permute_581, view_19);  permute_581 = view_19 = None
    permute_582: "f32[192, 196]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_264: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_452, [0], True);  view_452 = None
    view_453: "f32[196]" = torch.ops.aten.view.default(sum_264, [196]);  sum_264 = None
    permute_583: "f32[196, 192]" = torch.ops.aten.permute.default(permute_582, [1, 0]);  permute_582 = None
    view_454: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_198, [8, 384, 192]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_717: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_454, getitem_18);  getitem_18 = None
    mul_718: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_454, mul_18);  view_454 = mul_18 = None
    sigmoid_91: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_19)
    full_43: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_224: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_43, sigmoid_91);  full_43 = None
    mul_719: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_19, sub_224);  getitem_19 = sub_224 = None
    add_256: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_719, 1);  mul_719 = None
    mul_720: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_91, add_256);  sigmoid_91 = add_256 = None
    mul_721: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_717, mul_720);  mul_717 = mul_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_43: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_718, mul_721], 2);  mul_718 = mul_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_265: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_43, [0, 1], True)
    view_455: "f32[384]" = torch.ops.aten.view.default(sum_265, [384]);  sum_265 = None
    view_456: "f32[3072, 384]" = torch.ops.aten.view.default(cat_43, [3072, 384]);  cat_43 = None
    permute_585: "f32[384, 3072]" = torch.ops.aten.permute.default(view_456, [1, 0])
    mm_200: "f32[384, 196]" = torch.ops.aten.mm.default(permute_585, view_17);  permute_585 = view_17 = None
    permute_586: "f32[196, 384]" = torch.ops.aten.permute.default(mm_200, [1, 0]);  mm_200 = None
    permute_587: "f32[384, 196]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_201: "f32[3072, 196]" = torch.ops.aten.mm.default(view_456, permute_587);  view_456 = permute_587 = None
    view_457: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_201, [8, 384, 196]);  mm_201 = None
    permute_588: "f32[384, 196]" = torch.ops.aten.permute.default(permute_586, [1, 0]);  permute_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_589: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_457, [0, 2, 1]);  view_457 = None
    clone_257: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_589, memory_format = torch.contiguous_format);  permute_589 = None
    clone_258: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_13, memory_format = torch.contiguous_format);  add_13 = None
    sub_225: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_258, getitem_17);  clone_258 = getitem_17 = None
    mul_722: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_225, rsqrt_4);  sub_225 = None
    mul_723: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_257, primals_27);  primals_27 = None
    mul_724: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_723, 384)
    sum_266: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_723, [2], True)
    mul_725: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_723, mul_722);  mul_723 = None
    sum_267: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_725, [2], True);  mul_725 = None
    mul_726: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_722, sum_267);  sum_267 = None
    sub_226: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_724, sum_266);  mul_724 = sum_266 = None
    sub_227: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_226, mul_726);  sub_226 = mul_726 = None
    div_45: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 384);  rsqrt_4 = None
    mul_727: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_45, sub_227);  div_45 = sub_227 = None
    mul_728: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_257, mul_722);  mul_722 = None
    sum_268: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_728, [0, 1]);  mul_728 = None
    sum_269: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_257, [0, 1]);  clone_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_257: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_255, mul_727);  add_255 = mul_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_458: "f32[1568, 384]" = torch.ops.aten.view.default(add_257, [1568, 384])
    permute_590: "f32[384, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_202: "f32[1568, 768]" = torch.ops.aten.mm.default(view_458, permute_590);  permute_590 = None
    permute_591: "f32[384, 1568]" = torch.ops.aten.permute.default(view_458, [1, 0])
    mm_203: "f32[384, 768]" = torch.ops.aten.mm.default(permute_591, view_15);  permute_591 = view_15 = None
    permute_592: "f32[768, 384]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    sum_270: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_458, [0], True);  view_458 = None
    view_459: "f32[384]" = torch.ops.aten.view.default(sum_270, [384]);  sum_270 = None
    permute_593: "f32[384, 768]" = torch.ops.aten.permute.default(permute_592, [1, 0]);  permute_592 = None
    view_460: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_202, [8, 196, 768]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_729: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_460, getitem_14);  getitem_14 = None
    mul_730: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_460, mul_14);  view_460 = mul_14 = None
    sigmoid_92: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_15)
    full_44: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_228: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_44, sigmoid_92);  full_44 = None
    mul_731: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_15, sub_228);  getitem_15 = sub_228 = None
    add_258: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_731, 1);  mul_731 = None
    mul_732: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_92, add_258);  sigmoid_92 = add_258 = None
    mul_733: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_729, mul_732);  mul_729 = mul_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_44: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_730, mul_733], 2);  mul_730 = mul_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_461: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_44, [1568, 1536]);  cat_44 = None
    permute_595: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_204: "f32[1568, 384]" = torch.ops.aten.mm.default(view_461, permute_595);  permute_595 = None
    permute_596: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_461, [1, 0])
    mm_205: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_596, view_13);  permute_596 = view_13 = None
    permute_597: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_271: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_461, [0], True);  view_461 = None
    view_462: "f32[1536]" = torch.ops.aten.view.default(sum_271, [1536]);  sum_271 = None
    permute_598: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_597, [1, 0]);  permute_597 = None
    view_463: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_204, [8, 196, 384]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_259: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_10, memory_format = torch.contiguous_format);  add_10 = None
    sub_229: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_259, getitem_13);  clone_259 = getitem_13 = None
    mul_734: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_229, rsqrt_3);  sub_229 = None
    mul_735: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_463, primals_21);  primals_21 = None
    mul_736: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_735, 384)
    sum_272: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_735, [2], True)
    mul_737: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_735, mul_734);  mul_735 = None
    sum_273: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_737, [2], True);  mul_737 = None
    mul_738: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_734, sum_273);  sum_273 = None
    sub_230: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_736, sum_272);  mul_736 = sum_272 = None
    sub_231: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_230, mul_738);  sub_230 = mul_738 = None
    div_46: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 384);  rsqrt_3 = None
    mul_739: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_46, sub_231);  div_46 = sub_231 = None
    mul_740: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_463, mul_734);  mul_734 = None
    sum_274: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_740, [0, 1]);  mul_740 = None
    sum_275: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_463, [0, 1]);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_259: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_257, mul_739);  add_257 = mul_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_599: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_259, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_260: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_599, memory_format = torch.contiguous_format);  permute_599 = None
    view_464: "f32[3072, 196]" = torch.ops.aten.view.default(clone_260, [3072, 196]);  clone_260 = None
    permute_600: "f32[196, 192]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_206: "f32[3072, 192]" = torch.ops.aten.mm.default(view_464, permute_600);  permute_600 = None
    permute_601: "f32[196, 3072]" = torch.ops.aten.permute.default(view_464, [1, 0])
    mm_207: "f32[196, 192]" = torch.ops.aten.mm.default(permute_601, view_11);  permute_601 = view_11 = None
    permute_602: "f32[192, 196]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    sum_276: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
    view_465: "f32[196]" = torch.ops.aten.view.default(sum_276, [196]);  sum_276 = None
    permute_603: "f32[196, 192]" = torch.ops.aten.permute.default(permute_602, [1, 0]);  permute_602 = None
    view_466: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_206, [8, 384, 192]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_741: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_466, getitem_10);  getitem_10 = None
    mul_742: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_466, mul_10);  view_466 = mul_10 = None
    sigmoid_93: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_11)
    full_45: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_232: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_45, sigmoid_93);  full_45 = None
    mul_743: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_11, sub_232);  getitem_11 = sub_232 = None
    add_260: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_743, 1);  mul_743 = None
    mul_744: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_93, add_260);  sigmoid_93 = add_260 = None
    mul_745: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_741, mul_744);  mul_741 = mul_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_45: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_742, mul_745], 2);  mul_742 = mul_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_277: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_45, [0, 1], True)
    view_467: "f32[384]" = torch.ops.aten.view.default(sum_277, [384]);  sum_277 = None
    view_468: "f32[3072, 384]" = torch.ops.aten.view.default(cat_45, [3072, 384]);  cat_45 = None
    permute_605: "f32[384, 3072]" = torch.ops.aten.permute.default(view_468, [1, 0])
    mm_208: "f32[384, 196]" = torch.ops.aten.mm.default(permute_605, view_9);  permute_605 = view_9 = None
    permute_606: "f32[196, 384]" = torch.ops.aten.permute.default(mm_208, [1, 0]);  mm_208 = None
    permute_607: "f32[384, 196]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_209: "f32[3072, 196]" = torch.ops.aten.mm.default(view_468, permute_607);  view_468 = permute_607 = None
    view_469: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_209, [8, 384, 196]);  mm_209 = None
    permute_608: "f32[384, 196]" = torch.ops.aten.permute.default(permute_606, [1, 0]);  permute_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_609: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_469, [0, 2, 1]);  view_469 = None
    clone_261: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_609, memory_format = torch.contiguous_format);  permute_609 = None
    clone_262: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_6, memory_format = torch.contiguous_format);  add_6 = None
    sub_233: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_262, getitem_9);  clone_262 = getitem_9 = None
    mul_746: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_233, rsqrt_2);  sub_233 = None
    mul_747: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_261, primals_15);  primals_15 = None
    mul_748: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_747, 384)
    sum_278: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_747, [2], True)
    mul_749: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_747, mul_746);  mul_747 = None
    sum_279: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True);  mul_749 = None
    mul_750: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_746, sum_279);  sum_279 = None
    sub_234: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_748, sum_278);  mul_748 = sum_278 = None
    sub_235: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_234, mul_750);  sub_234 = mul_750 = None
    div_47: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 384);  rsqrt_2 = None
    mul_751: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_47, sub_235);  div_47 = sub_235 = None
    mul_752: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_261, mul_746);  mul_746 = None
    sum_280: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 1]);  mul_752 = None
    sum_281: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_261, [0, 1]);  clone_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_261: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_259, mul_751);  add_259 = mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_470: "f32[1568, 384]" = torch.ops.aten.view.default(add_261, [1568, 384])
    permute_610: "f32[384, 768]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    mm_210: "f32[1568, 768]" = torch.ops.aten.mm.default(view_470, permute_610);  permute_610 = None
    permute_611: "f32[384, 1568]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_211: "f32[384, 768]" = torch.ops.aten.mm.default(permute_611, view_7);  permute_611 = view_7 = None
    permute_612: "f32[768, 384]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    sum_282: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
    view_471: "f32[384]" = torch.ops.aten.view.default(sum_282, [384]);  sum_282 = None
    permute_613: "f32[384, 768]" = torch.ops.aten.permute.default(permute_612, [1, 0]);  permute_612 = None
    view_472: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_210, [8, 196, 768]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_753: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_472, getitem_6);  getitem_6 = None
    mul_754: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_472, mul_6);  view_472 = mul_6 = None
    sigmoid_94: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_7)
    full_46: "f32[8, 196, 768]" = torch.ops.aten.full.default([8, 196, 768], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_236: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(full_46, sigmoid_94);  full_46 = None
    mul_755: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_7, sub_236);  getitem_7 = sub_236 = None
    add_262: "f32[8, 196, 768]" = torch.ops.aten.add.Scalar(mul_755, 1);  mul_755 = None
    mul_756: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sigmoid_94, add_262);  sigmoid_94 = add_262 = None
    mul_757: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_753, mul_756);  mul_753 = mul_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_46: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_754, mul_757], 2);  mul_754 = mul_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_473: "f32[1568, 1536]" = torch.ops.aten.view.default(cat_46, [1568, 1536]);  cat_46 = None
    permute_615: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    mm_212: "f32[1568, 384]" = torch.ops.aten.mm.default(view_473, permute_615);  permute_615 = None
    permute_616: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_213: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_616, view_5);  permute_616 = view_5 = None
    permute_617: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    sum_283: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
    view_474: "f32[1536]" = torch.ops.aten.view.default(sum_283, [1536]);  sum_283 = None
    permute_618: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_617, [1, 0]);  permute_617 = None
    view_475: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_212, [8, 196, 384]);  mm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_263: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_3, memory_format = torch.contiguous_format);  add_3 = None
    sub_237: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_263, getitem_5);  clone_263 = getitem_5 = None
    mul_758: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_237, rsqrt_1);  sub_237 = None
    mul_759: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_475, primals_9);  primals_9 = None
    mul_760: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_759, 384)
    sum_284: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_759, [2], True)
    mul_761: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_759, mul_758);  mul_759 = None
    sum_285: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_761, [2], True);  mul_761 = None
    mul_762: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_758, sum_285);  sum_285 = None
    sub_238: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_760, sum_284);  mul_760 = sum_284 = None
    sub_239: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_238, mul_762);  sub_238 = mul_762 = None
    div_48: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 384);  rsqrt_1 = None
    mul_763: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_48, sub_239);  div_48 = sub_239 = None
    mul_764: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_475, mul_758);  mul_758 = None
    sum_286: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_764, [0, 1]);  mul_764 = None
    sum_287: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_475, [0, 1]);  view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_263: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_261, mul_763);  add_261 = mul_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_619: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_263, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    clone_264: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_619, memory_format = torch.contiguous_format);  permute_619 = None
    view_476: "f32[3072, 196]" = torch.ops.aten.view.default(clone_264, [3072, 196]);  clone_264 = None
    permute_620: "f32[196, 192]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_214: "f32[3072, 192]" = torch.ops.aten.mm.default(view_476, permute_620);  permute_620 = None
    permute_621: "f32[196, 3072]" = torch.ops.aten.permute.default(view_476, [1, 0])
    mm_215: "f32[196, 192]" = torch.ops.aten.mm.default(permute_621, view_3);  permute_621 = view_3 = None
    permute_622: "f32[192, 196]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    sum_288: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_476, [0], True);  view_476 = None
    view_477: "f32[196]" = torch.ops.aten.view.default(sum_288, [196]);  sum_288 = None
    permute_623: "f32[196, 192]" = torch.ops.aten.permute.default(permute_622, [1, 0]);  permute_622 = None
    view_478: "f32[8, 384, 192]" = torch.ops.aten.view.default(mm_214, [8, 384, 192]);  mm_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    mul_765: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_478, getitem_2);  getitem_2 = None
    mul_766: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(view_478, mul_2);  view_478 = mul_2 = None
    sigmoid_95: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_3)
    full_47: "f32[8, 384, 192]" = torch.ops.aten.full.default([8, 384, 192], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_240: "f32[8, 384, 192]" = torch.ops.aten.sub.Tensor(full_47, sigmoid_95);  full_47 = None
    mul_767: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_3, sub_240);  getitem_3 = sub_240 = None
    add_264: "f32[8, 384, 192]" = torch.ops.aten.add.Scalar(mul_767, 1);  mul_767 = None
    mul_768: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(sigmoid_95, add_264);  sigmoid_95 = add_264 = None
    mul_769: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(mul_765, mul_768);  mul_765 = mul_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    cat_47: "f32[8, 384, 384]" = torch.ops.aten.cat.default([mul_766, mul_769], 2);  mul_766 = mul_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    sum_289: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(cat_47, [0, 1], True)
    view_479: "f32[384]" = torch.ops.aten.view.default(sum_289, [384]);  sum_289 = None
    view_480: "f32[3072, 384]" = torch.ops.aten.view.default(cat_47, [3072, 384]);  cat_47 = None
    permute_625: "f32[384, 3072]" = torch.ops.aten.permute.default(view_480, [1, 0])
    mm_216: "f32[384, 196]" = torch.ops.aten.mm.default(permute_625, view_1);  permute_625 = view_1 = None
    permute_626: "f32[196, 384]" = torch.ops.aten.permute.default(mm_216, [1, 0]);  mm_216 = None
    permute_627: "f32[384, 196]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_217: "f32[3072, 196]" = torch.ops.aten.mm.default(view_480, permute_627);  view_480 = permute_627 = None
    view_481: "f32[8, 384, 196]" = torch.ops.aten.view.default(mm_217, [8, 384, 196]);  mm_217 = None
    permute_628: "f32[384, 196]" = torch.ops.aten.permute.default(permute_626, [1, 0]);  permute_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_629: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_481, [0, 2, 1]);  view_481 = None
    clone_265: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_629, memory_format = torch.contiguous_format);  permute_629 = None
    clone_266: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    sub_241: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_266, getitem_1);  clone_266 = getitem_1 = None
    mul_770: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_241, rsqrt);  sub_241 = None
    mul_771: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_265, primals_3);  primals_3 = None
    mul_772: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_771, 384)
    sum_290: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_771, [2], True)
    mul_773: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_771, mul_770);  mul_771 = None
    sum_291: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_773, [2], True);  mul_773 = None
    mul_774: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_770, sum_291);  sum_291 = None
    sub_242: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_772, sum_290);  mul_772 = sum_290 = None
    sub_243: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_242, mul_774);  sub_242 = mul_774 = None
    div_49: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt, 384);  rsqrt = None
    mul_775: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_49, sub_243);  div_49 = sub_243 = None
    mul_776: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_265, mul_770);  mul_770 = None
    sum_292: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_776, [0, 1]);  mul_776 = None
    sum_293: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_265, [0, 1]);  clone_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_265: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_263, mul_775);  add_263 = mul_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_630: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_265, [0, 2, 1]);  add_265 = None
    view_482: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(permute_630, [8, 384, 14, 14]);  permute_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(view_482, primals_295, primals_1, [384], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_482 = primals_295 = primals_1 = None
    getitem_195: "f32[384, 3, 16, 16]" = convolution_backward[1]
    getitem_196: "f32[384]" = convolution_backward[2];  convolution_backward = None
    return pytree.tree_unflatten([addmm_72, getitem_195, getitem_196, sum_292, sum_293, permute_628, view_479, permute_623, view_477, sum_286, sum_287, permute_618, view_474, permute_613, view_471, sum_280, sum_281, permute_608, view_467, permute_603, view_465, sum_274, sum_275, permute_598, view_462, permute_593, view_459, sum_268, sum_269, permute_588, view_455, permute_583, view_453, sum_262, sum_263, permute_578, view_450, permute_573, view_447, sum_256, sum_257, permute_568, view_443, permute_563, view_441, sum_250, sum_251, permute_558, view_438, permute_553, view_435, sum_244, sum_245, permute_548, view_431, permute_543, view_429, sum_238, sum_239, permute_538, view_426, permute_533, view_423, sum_232, sum_233, permute_528, view_419, permute_523, view_417, sum_226, sum_227, permute_518, view_414, permute_513, view_411, sum_220, sum_221, permute_508, view_407, permute_503, view_405, sum_214, sum_215, permute_498, view_402, permute_493, view_399, sum_208, sum_209, permute_488, view_395, permute_483, view_393, sum_202, sum_203, permute_478, view_390, permute_473, view_387, sum_196, sum_197, permute_468, view_383, permute_463, view_381, sum_190, sum_191, permute_458, view_378, permute_453, view_375, sum_184, sum_185, permute_448, view_371, permute_443, view_369, sum_178, sum_179, permute_438, view_366, permute_433, view_363, sum_172, sum_173, permute_428, view_359, permute_423, view_357, sum_166, sum_167, permute_418, view_354, permute_413, view_351, sum_160, sum_161, permute_408, view_347, permute_403, view_345, sum_154, sum_155, permute_398, view_342, permute_393, view_339, sum_148, sum_149, permute_388, view_335, permute_383, view_333, sum_142, sum_143, permute_378, view_330, permute_373, view_327, sum_136, sum_137, permute_368, view_323, permute_363, view_321, sum_130, sum_131, permute_358, view_318, permute_353, view_315, sum_124, sum_125, permute_348, view_311, permute_343, view_309, sum_118, sum_119, permute_338, view_306, permute_333, view_303, sum_112, sum_113, permute_328, view_299, permute_323, view_297, sum_106, sum_107, permute_318, view_294, permute_313, view_291, sum_100, sum_101, permute_308, view_287, permute_303, view_285, sum_94, sum_95, permute_298, view_282, permute_293, view_279, sum_88, sum_89, permute_288, view_275, permute_283, view_273, sum_82, sum_83, permute_278, view_270, permute_273, view_267, sum_76, sum_77, permute_268, view_263, permute_263, view_261, sum_70, sum_71, permute_258, view_258, permute_253, view_255, sum_64, sum_65, permute_248, view_251, permute_243, view_249, sum_58, sum_59, permute_238, view_246, permute_233, view_243, sum_52, sum_53, permute_228, view_239, permute_223, view_237, sum_46, sum_47, permute_218, view_234, permute_213, view_231, sum_40, sum_41, permute_208, view_227, permute_203, view_225, sum_34, sum_35, permute_198, view_222, permute_193, view_219, sum_28, sum_29, permute_188, view_215, permute_183, view_213, sum_22, sum_23, permute_178, view_210, permute_173, view_207, sum_16, sum_17, permute_168, view_203, permute_163, view_201, sum_10, sum_11, permute_158, view_198, permute_153, view_195, sum_4, sum_5, permute_149, view_193, None], self._out_spec)
    