from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[256, 3, 16, 16]"; primals_2: "f32[256]"; primals_3: "f32[256]"; primals_4: "f32[256]"; primals_5: "f32[1536, 256]"; primals_6: "f32[1536]"; primals_7: "f32[768]"; primals_8: "f32[768]"; primals_9: "f32[196, 196]"; primals_10: "f32[196]"; primals_11: "f32[256, 768]"; primals_12: "f32[256]"; primals_13: "f32[256]"; primals_14: "f32[256]"; primals_15: "f32[1536, 256]"; primals_16: "f32[1536]"; primals_17: "f32[768]"; primals_18: "f32[768]"; primals_19: "f32[196, 196]"; primals_20: "f32[196]"; primals_21: "f32[256, 768]"; primals_22: "f32[256]"; primals_23: "f32[256]"; primals_24: "f32[256]"; primals_25: "f32[1536, 256]"; primals_26: "f32[1536]"; primals_27: "f32[768]"; primals_28: "f32[768]"; primals_29: "f32[196, 196]"; primals_30: "f32[196]"; primals_31: "f32[256, 768]"; primals_32: "f32[256]"; primals_33: "f32[256]"; primals_34: "f32[256]"; primals_35: "f32[1536, 256]"; primals_36: "f32[1536]"; primals_37: "f32[768]"; primals_38: "f32[768]"; primals_39: "f32[196, 196]"; primals_40: "f32[196]"; primals_41: "f32[256, 768]"; primals_42: "f32[256]"; primals_43: "f32[256]"; primals_44: "f32[256]"; primals_45: "f32[1536, 256]"; primals_46: "f32[1536]"; primals_47: "f32[768]"; primals_48: "f32[768]"; primals_49: "f32[196, 196]"; primals_50: "f32[196]"; primals_51: "f32[256, 768]"; primals_52: "f32[256]"; primals_53: "f32[256]"; primals_54: "f32[256]"; primals_55: "f32[1536, 256]"; primals_56: "f32[1536]"; primals_57: "f32[768]"; primals_58: "f32[768]"; primals_59: "f32[196, 196]"; primals_60: "f32[196]"; primals_61: "f32[256, 768]"; primals_62: "f32[256]"; primals_63: "f32[256]"; primals_64: "f32[256]"; primals_65: "f32[1536, 256]"; primals_66: "f32[1536]"; primals_67: "f32[768]"; primals_68: "f32[768]"; primals_69: "f32[196, 196]"; primals_70: "f32[196]"; primals_71: "f32[256, 768]"; primals_72: "f32[256]"; primals_73: "f32[256]"; primals_74: "f32[256]"; primals_75: "f32[1536, 256]"; primals_76: "f32[1536]"; primals_77: "f32[768]"; primals_78: "f32[768]"; primals_79: "f32[196, 196]"; primals_80: "f32[196]"; primals_81: "f32[256, 768]"; primals_82: "f32[256]"; primals_83: "f32[256]"; primals_84: "f32[256]"; primals_85: "f32[1536, 256]"; primals_86: "f32[1536]"; primals_87: "f32[768]"; primals_88: "f32[768]"; primals_89: "f32[196, 196]"; primals_90: "f32[196]"; primals_91: "f32[256, 768]"; primals_92: "f32[256]"; primals_93: "f32[256]"; primals_94: "f32[256]"; primals_95: "f32[1536, 256]"; primals_96: "f32[1536]"; primals_97: "f32[768]"; primals_98: "f32[768]"; primals_99: "f32[196, 196]"; primals_100: "f32[196]"; primals_101: "f32[256, 768]"; primals_102: "f32[256]"; primals_103: "f32[256]"; primals_104: "f32[256]"; primals_105: "f32[1536, 256]"; primals_106: "f32[1536]"; primals_107: "f32[768]"; primals_108: "f32[768]"; primals_109: "f32[196, 196]"; primals_110: "f32[196]"; primals_111: "f32[256, 768]"; primals_112: "f32[256]"; primals_113: "f32[256]"; primals_114: "f32[256]"; primals_115: "f32[1536, 256]"; primals_116: "f32[1536]"; primals_117: "f32[768]"; primals_118: "f32[768]"; primals_119: "f32[196, 196]"; primals_120: "f32[196]"; primals_121: "f32[256, 768]"; primals_122: "f32[256]"; primals_123: "f32[256]"; primals_124: "f32[256]"; primals_125: "f32[1536, 256]"; primals_126: "f32[1536]"; primals_127: "f32[768]"; primals_128: "f32[768]"; primals_129: "f32[196, 196]"; primals_130: "f32[196]"; primals_131: "f32[256, 768]"; primals_132: "f32[256]"; primals_133: "f32[256]"; primals_134: "f32[256]"; primals_135: "f32[1536, 256]"; primals_136: "f32[1536]"; primals_137: "f32[768]"; primals_138: "f32[768]"; primals_139: "f32[196, 196]"; primals_140: "f32[196]"; primals_141: "f32[256, 768]"; primals_142: "f32[256]"; primals_143: "f32[256]"; primals_144: "f32[256]"; primals_145: "f32[1536, 256]"; primals_146: "f32[1536]"; primals_147: "f32[768]"; primals_148: "f32[768]"; primals_149: "f32[196, 196]"; primals_150: "f32[196]"; primals_151: "f32[256, 768]"; primals_152: "f32[256]"; primals_153: "f32[256]"; primals_154: "f32[256]"; primals_155: "f32[1536, 256]"; primals_156: "f32[1536]"; primals_157: "f32[768]"; primals_158: "f32[768]"; primals_159: "f32[196, 196]"; primals_160: "f32[196]"; primals_161: "f32[256, 768]"; primals_162: "f32[256]"; primals_163: "f32[256]"; primals_164: "f32[256]"; primals_165: "f32[1536, 256]"; primals_166: "f32[1536]"; primals_167: "f32[768]"; primals_168: "f32[768]"; primals_169: "f32[196, 196]"; primals_170: "f32[196]"; primals_171: "f32[256, 768]"; primals_172: "f32[256]"; primals_173: "f32[256]"; primals_174: "f32[256]"; primals_175: "f32[1536, 256]"; primals_176: "f32[1536]"; primals_177: "f32[768]"; primals_178: "f32[768]"; primals_179: "f32[196, 196]"; primals_180: "f32[196]"; primals_181: "f32[256, 768]"; primals_182: "f32[256]"; primals_183: "f32[256]"; primals_184: "f32[256]"; primals_185: "f32[1536, 256]"; primals_186: "f32[1536]"; primals_187: "f32[768]"; primals_188: "f32[768]"; primals_189: "f32[196, 196]"; primals_190: "f32[196]"; primals_191: "f32[256, 768]"; primals_192: "f32[256]"; primals_193: "f32[256]"; primals_194: "f32[256]"; primals_195: "f32[1536, 256]"; primals_196: "f32[1536]"; primals_197: "f32[768]"; primals_198: "f32[768]"; primals_199: "f32[196, 196]"; primals_200: "f32[196]"; primals_201: "f32[256, 768]"; primals_202: "f32[256]"; primals_203: "f32[256]"; primals_204: "f32[256]"; primals_205: "f32[1536, 256]"; primals_206: "f32[1536]"; primals_207: "f32[768]"; primals_208: "f32[768]"; primals_209: "f32[196, 196]"; primals_210: "f32[196]"; primals_211: "f32[256, 768]"; primals_212: "f32[256]"; primals_213: "f32[256]"; primals_214: "f32[256]"; primals_215: "f32[1536, 256]"; primals_216: "f32[1536]"; primals_217: "f32[768]"; primals_218: "f32[768]"; primals_219: "f32[196, 196]"; primals_220: "f32[196]"; primals_221: "f32[256, 768]"; primals_222: "f32[256]"; primals_223: "f32[256]"; primals_224: "f32[256]"; primals_225: "f32[1536, 256]"; primals_226: "f32[1536]"; primals_227: "f32[768]"; primals_228: "f32[768]"; primals_229: "f32[196, 196]"; primals_230: "f32[196]"; primals_231: "f32[256, 768]"; primals_232: "f32[256]"; primals_233: "f32[256]"; primals_234: "f32[256]"; primals_235: "f32[1536, 256]"; primals_236: "f32[1536]"; primals_237: "f32[768]"; primals_238: "f32[768]"; primals_239: "f32[196, 196]"; primals_240: "f32[196]"; primals_241: "f32[256, 768]"; primals_242: "f32[256]"; primals_243: "f32[256]"; primals_244: "f32[256]"; primals_245: "f32[1536, 256]"; primals_246: "f32[1536]"; primals_247: "f32[768]"; primals_248: "f32[768]"; primals_249: "f32[196, 196]"; primals_250: "f32[196]"; primals_251: "f32[256, 768]"; primals_252: "f32[256]"; primals_253: "f32[256]"; primals_254: "f32[256]"; primals_255: "f32[1536, 256]"; primals_256: "f32[1536]"; primals_257: "f32[768]"; primals_258: "f32[768]"; primals_259: "f32[196, 196]"; primals_260: "f32[196]"; primals_261: "f32[256, 768]"; primals_262: "f32[256]"; primals_263: "f32[256]"; primals_264: "f32[256]"; primals_265: "f32[1536, 256]"; primals_266: "f32[1536]"; primals_267: "f32[768]"; primals_268: "f32[768]"; primals_269: "f32[196, 196]"; primals_270: "f32[196]"; primals_271: "f32[256, 768]"; primals_272: "f32[256]"; primals_273: "f32[256]"; primals_274: "f32[256]"; primals_275: "f32[1536, 256]"; primals_276: "f32[1536]"; primals_277: "f32[768]"; primals_278: "f32[768]"; primals_279: "f32[196, 196]"; primals_280: "f32[196]"; primals_281: "f32[256, 768]"; primals_282: "f32[256]"; primals_283: "f32[256]"; primals_284: "f32[256]"; primals_285: "f32[1536, 256]"; primals_286: "f32[1536]"; primals_287: "f32[768]"; primals_288: "f32[768]"; primals_289: "f32[196, 196]"; primals_290: "f32[196]"; primals_291: "f32[256, 768]"; primals_292: "f32[256]"; primals_293: "f32[256]"; primals_294: "f32[256]"; primals_295: "f32[1536, 256]"; primals_296: "f32[1536]"; primals_297: "f32[768]"; primals_298: "f32[768]"; primals_299: "f32[196, 196]"; primals_300: "f32[196]"; primals_301: "f32[256, 768]"; primals_302: "f32[256]"; primals_303: "f32[256]"; primals_304: "f32[256]"; primals_305: "f32[1000, 256]"; primals_306: "f32[1000]"; primals_307: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
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
    sub: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = None
    mul: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul, primals_3);  mul = None
    add_1: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_1: "f32[1568, 256]" = torch.ops.aten.view.default(add_1, [1568, 256]);  add_1 = None
    permute_1: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_6, view_1, permute_1);  primals_6 = None
    view_2: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm, [8, 196, 1536]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_2: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_2, 0.5)
    mul_3: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_2, 0.7071067811865476)
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
    clone_2: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_3, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_2, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_5: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_3: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_1: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_2, getitem_5);  clone_2 = None
    mul_5: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_6: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_5, primals_7);  mul_5 = None
    add_4: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_6, primals_8);  mul_6 = primals_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_2: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_4, [0, 2, 1]);  add_4 = None
    permute_3: "f32[196, 196]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
    clone_3: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    view_3: "f32[6144, 196]" = torch.ops.aten.view.default(clone_3, [6144, 196]);  clone_3 = None
    mm: "f32[6144, 196]" = torch.ops.aten.mm.default(view_3, permute_3)
    view_4: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm, [8, 768, 196]);  mm = None
    add_5: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_4, primals_10);  view_4 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_4: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_5, [0, 2, 1]);  add_5 = None
    mul_7: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_2, permute_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_5: "f32[1568, 768]" = torch.ops.aten.view.default(mul_7, [1568, 768]);  mul_7 = None
    permute_5: "f32[768, 256]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm_1: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_12, view_5, permute_5);  primals_12 = None
    view_6: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_1, [8, 196, 256]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_4: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_6);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_6: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(permute, clone_4);  clone_4 = None
    clone_5: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_6, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_5, [2], correction = 0, keepdim = True)
    getitem_6: "f32[8, 196, 1]" = var_mean_2[0]
    getitem_7: "f32[8, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    add_7: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_2: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_2: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_5, getitem_7);  clone_5 = None
    mul_8: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_9: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_8, primals_13);  mul_8 = None
    add_8: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_9, primals_14);  mul_9 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_7: "f32[1568, 256]" = torch.ops.aten.view.default(add_8, [1568, 256]);  add_8 = None
    permute_6: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_2: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_16, view_7, permute_6);  primals_16 = None
    view_8: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_2, [8, 196, 1536]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_10: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_8, 0.5)
    mul_11: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476)
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
    clone_7: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_9, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_7, [2], correction = 0, keepdim = True)
    getitem_10: "f32[8, 196, 1]" = var_mean_3[0]
    getitem_11: "f32[8, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    add_10: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_3: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_3: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_7, getitem_11);  clone_7 = None
    mul_13: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_14: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_13, primals_17);  mul_13 = None
    add_11: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_14, primals_18);  mul_14 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_7: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_11, [0, 2, 1]);  add_11 = None
    permute_8: "f32[196, 196]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    clone_8: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_9: "f32[6144, 196]" = torch.ops.aten.view.default(clone_8, [6144, 196]);  clone_8 = None
    mm_1: "f32[6144, 196]" = torch.ops.aten.mm.default(view_9, permute_8)
    view_10: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_1, [8, 768, 196]);  mm_1 = None
    add_12: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_10, primals_20);  view_10 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_9: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_12, [0, 2, 1]);  add_12 = None
    mul_15: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_8, permute_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_11: "f32[1568, 768]" = torch.ops.aten.view.default(mul_15, [1568, 768]);  mul_15 = None
    permute_10: "f32[768, 256]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
    addmm_3: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_22, view_11, permute_10);  primals_22 = None
    view_12: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_3, [8, 196, 256]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_9: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_13: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_6, clone_9);  clone_9 = None
    clone_10: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_13, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_10, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 196, 1]" = var_mean_4[0]
    getitem_13: "f32[8, 196, 1]" = var_mean_4[1];  var_mean_4 = None
    add_14: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_4: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_4: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_10, getitem_13);  clone_10 = None
    mul_16: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_17: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_16, primals_23);  mul_16 = None
    add_15: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_17, primals_24);  mul_17 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_13: "f32[1568, 256]" = torch.ops.aten.view.default(add_15, [1568, 256]);  add_15 = None
    permute_11: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_4: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_26, view_13, permute_11);  primals_26 = None
    view_14: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_4, [8, 196, 1536]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_18: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_14, 0.5)
    mul_19: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476)
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
    clone_12: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_15, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_12, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 196, 1]" = var_mean_5[0]
    getitem_17: "f32[8, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    add_17: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_5: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_5: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_12, getitem_17);  clone_12 = None
    mul_21: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_22: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_27);  mul_21 = None
    add_18: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_22, primals_28);  mul_22 = primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_12: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_18, [0, 2, 1]);  add_18 = None
    permute_13: "f32[196, 196]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
    clone_13: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    view_15: "f32[6144, 196]" = torch.ops.aten.view.default(clone_13, [6144, 196]);  clone_13 = None
    mm_2: "f32[6144, 196]" = torch.ops.aten.mm.default(view_15, permute_13)
    view_16: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_2, [8, 768, 196]);  mm_2 = None
    add_19: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_16, primals_30);  view_16 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_14: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_19, [0, 2, 1]);  add_19 = None
    mul_23: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_14, permute_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_17: "f32[1568, 768]" = torch.ops.aten.view.default(mul_23, [1568, 768]);  mul_23 = None
    permute_15: "f32[768, 256]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    addmm_5: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_32, view_17, permute_15);  primals_32 = None
    view_18: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_5, [8, 196, 256]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_14: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_20: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_13, clone_14);  clone_14 = None
    clone_15: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_15, [2], correction = 0, keepdim = True)
    getitem_18: "f32[8, 196, 1]" = var_mean_6[0]
    getitem_19: "f32[8, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_6: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_6: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_15, getitem_19);  clone_15 = None
    mul_24: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_25: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_24, primals_33);  mul_24 = None
    add_22: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_25, primals_34);  mul_25 = primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_19: "f32[1568, 256]" = torch.ops.aten.view.default(add_22, [1568, 256]);  add_22 = None
    permute_16: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
    addmm_6: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_36, view_19, permute_16);  primals_36 = None
    view_20: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_6, [8, 196, 1536]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_26: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_20, 0.5)
    mul_27: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_20, 0.7071067811865476)
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
    clone_17: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_21, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_17, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 196, 1]" = var_mean_7[0]
    getitem_23: "f32[8, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    add_24: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_7: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_7: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_17, getitem_23);  clone_17 = None
    mul_29: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_30: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_29, primals_37);  mul_29 = None
    add_25: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_30, primals_38);  mul_30 = primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_17: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_25, [0, 2, 1]);  add_25 = None
    permute_18: "f32[196, 196]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
    clone_18: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_21: "f32[6144, 196]" = torch.ops.aten.view.default(clone_18, [6144, 196]);  clone_18 = None
    mm_3: "f32[6144, 196]" = torch.ops.aten.mm.default(view_21, permute_18)
    view_22: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_3, [8, 768, 196]);  mm_3 = None
    add_26: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_22, primals_40);  view_22 = primals_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_19: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_26, [0, 2, 1]);  add_26 = None
    mul_31: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_20, permute_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_23: "f32[1568, 768]" = torch.ops.aten.view.default(mul_31, [1568, 768]);  mul_31 = None
    permute_20: "f32[768, 256]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
    addmm_7: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_42, view_23, permute_20);  primals_42 = None
    view_24: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_7, [8, 196, 256]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_19: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_24);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_27: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_20, clone_19);  clone_19 = None
    clone_20: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_27, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_20, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 196, 1]" = var_mean_8[0]
    getitem_25: "f32[8, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_8: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_8: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_20, getitem_25);  clone_20 = None
    mul_32: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_33: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_32, primals_43);  mul_32 = None
    add_29: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_33, primals_44);  mul_33 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_25: "f32[1568, 256]" = torch.ops.aten.view.default(add_29, [1568, 256]);  add_29 = None
    permute_21: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    addmm_8: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_46, view_25, permute_21);  primals_46 = None
    view_26: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_8, [8, 196, 1536]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_34: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_26, 0.5)
    mul_35: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_26, 0.7071067811865476)
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
    clone_22: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_27, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_22, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 196, 1]" = var_mean_9[0]
    getitem_29: "f32[8, 196, 1]" = var_mean_9[1];  var_mean_9 = None
    add_31: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_9: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_9: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_22, getitem_29);  clone_22 = None
    mul_37: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_38: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_47);  mul_37 = None
    add_32: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_48);  mul_38 = primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_22: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_32, [0, 2, 1]);  add_32 = None
    permute_23: "f32[196, 196]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    clone_23: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
    view_27: "f32[6144, 196]" = torch.ops.aten.view.default(clone_23, [6144, 196]);  clone_23 = None
    mm_4: "f32[6144, 196]" = torch.ops.aten.mm.default(view_27, permute_23)
    view_28: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_4, [8, 768, 196]);  mm_4 = None
    add_33: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_28, primals_50);  view_28 = primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_24: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_33, [0, 2, 1]);  add_33 = None
    mul_39: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_26, permute_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_29: "f32[1568, 768]" = torch.ops.aten.view.default(mul_39, [1568, 768]);  mul_39 = None
    permute_25: "f32[768, 256]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    addmm_9: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_52, view_29, permute_25);  primals_52 = None
    view_30: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_9, [8, 196, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_24: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_30);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_34: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_27, clone_24);  clone_24 = None
    clone_25: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_34, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_25, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_31: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_10: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_25, getitem_31);  clone_25 = None
    mul_40: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_41: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_40, primals_53);  mul_40 = None
    add_36: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_41, primals_54);  mul_41 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_31: "f32[1568, 256]" = torch.ops.aten.view.default(add_36, [1568, 256]);  add_36 = None
    permute_26: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_10: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_56, view_31, permute_26);  primals_56 = None
    view_32: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_10, [8, 196, 1536]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_42: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_32, 0.5)
    mul_43: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_32, 0.7071067811865476)
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
    clone_27: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_33, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_27, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 196, 1]" = var_mean_11[0]
    getitem_35: "f32[8, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_11: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_11: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_27, getitem_35);  clone_27 = None
    mul_45: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_46: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_45, primals_57);  mul_45 = None
    add_39: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_46, primals_58);  mul_46 = primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_27: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_39, [0, 2, 1]);  add_39 = None
    permute_28: "f32[196, 196]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    clone_28: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    view_33: "f32[6144, 196]" = torch.ops.aten.view.default(clone_28, [6144, 196]);  clone_28 = None
    mm_5: "f32[6144, 196]" = torch.ops.aten.mm.default(view_33, permute_28)
    view_34: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_5, [8, 768, 196]);  mm_5 = None
    add_40: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_34, primals_60);  view_34 = primals_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_29: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_40, [0, 2, 1]);  add_40 = None
    mul_47: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_32, permute_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_35: "f32[1568, 768]" = torch.ops.aten.view.default(mul_47, [1568, 768]);  mul_47 = None
    permute_30: "f32[768, 256]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    addmm_11: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_62, view_35, permute_30);  primals_62 = None
    view_36: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_11, [8, 196, 256]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_29: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_41: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_34, clone_29);  clone_29 = None
    clone_30: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_30, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 196, 1]" = var_mean_12[0]
    getitem_37: "f32[8, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_12: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_12: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_30, getitem_37);  clone_30 = None
    mul_48: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_49: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_48, primals_63);  mul_48 = None
    add_43: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_49, primals_64);  mul_49 = primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_37: "f32[1568, 256]" = torch.ops.aten.view.default(add_43, [1568, 256]);  add_43 = None
    permute_31: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    addmm_12: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_66, view_37, permute_31);  primals_66 = None
    view_38: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_12, [8, 196, 1536]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_50: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_51: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
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
    clone_32: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_39, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_32, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 196, 1]" = var_mean_13[0]
    getitem_41: "f32[8, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    add_45: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_13: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_13: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_32, getitem_41);  clone_32 = None
    mul_53: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_54: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_53, primals_67);  mul_53 = None
    add_46: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_54, primals_68);  mul_54 = primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_32: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_46, [0, 2, 1]);  add_46 = None
    permute_33: "f32[196, 196]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    clone_33: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    view_39: "f32[6144, 196]" = torch.ops.aten.view.default(clone_33, [6144, 196]);  clone_33 = None
    mm_6: "f32[6144, 196]" = torch.ops.aten.mm.default(view_39, permute_33)
    view_40: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_6, [8, 768, 196]);  mm_6 = None
    add_47: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_40, primals_70);  view_40 = primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_34: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_47, [0, 2, 1]);  add_47 = None
    mul_55: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_38, permute_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_41: "f32[1568, 768]" = torch.ops.aten.view.default(mul_55, [1568, 768]);  mul_55 = None
    permute_35: "f32[768, 256]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_13: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_72, view_41, permute_35);  primals_72 = None
    view_42: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_13, [8, 196, 256]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_34: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_42);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_48: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_41, clone_34);  clone_34 = None
    clone_35: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_48, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_35, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 196, 1]" = var_mean_14[0]
    getitem_43: "f32[8, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_14: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_14: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_35, getitem_43);  clone_35 = None
    mul_56: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_57: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_56, primals_73);  mul_56 = None
    add_50: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_57, primals_74);  mul_57 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_43: "f32[1568, 256]" = torch.ops.aten.view.default(add_50, [1568, 256]);  add_50 = None
    permute_36: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    addmm_14: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_76, view_43, permute_36);  primals_76 = None
    view_44: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_14, [8, 196, 1536]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_58: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_44, 0.5)
    mul_59: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_44, 0.7071067811865476)
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
    clone_37: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_45, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_37, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 196, 1]" = var_mean_15[0]
    getitem_47: "f32[8, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    add_52: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_15: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_15: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_37, getitem_47);  clone_37 = None
    mul_61: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_62: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_61, primals_77);  mul_61 = None
    add_53: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_62, primals_78);  mul_62 = primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_37: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_53, [0, 2, 1]);  add_53 = None
    permute_38: "f32[196, 196]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    clone_38: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    view_45: "f32[6144, 196]" = torch.ops.aten.view.default(clone_38, [6144, 196]);  clone_38 = None
    mm_7: "f32[6144, 196]" = torch.ops.aten.mm.default(view_45, permute_38)
    view_46: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_7, [8, 768, 196]);  mm_7 = None
    add_54: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_46, primals_80);  view_46 = primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_39: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_54, [0, 2, 1]);  add_54 = None
    mul_63: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_44, permute_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_47: "f32[1568, 768]" = torch.ops.aten.view.default(mul_63, [1568, 768]);  mul_63 = None
    permute_40: "f32[768, 256]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_15: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_82, view_47, permute_40);  primals_82 = None
    view_48: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_15, [8, 196, 256]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_39: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_55: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_48, clone_39);  clone_39 = None
    clone_40: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_55, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_40, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 196, 1]" = var_mean_16[0]
    getitem_49: "f32[8, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    add_56: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_16: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_16: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_40, getitem_49);  clone_40 = None
    mul_64: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_65: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_64, primals_83);  mul_64 = None
    add_57: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_65, primals_84);  mul_65 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_49: "f32[1568, 256]" = torch.ops.aten.view.default(add_57, [1568, 256]);  add_57 = None
    permute_41: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_16: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_86, view_49, permute_41);  primals_86 = None
    view_50: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_16, [8, 196, 1536]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_66: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_50, 0.5)
    mul_67: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476)
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
    clone_42: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_51, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 196, 1]" = var_mean_17[0]
    getitem_53: "f32[8, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    add_59: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_17: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_17: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_42, getitem_53);  clone_42 = None
    mul_69: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_70: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_69, primals_87);  mul_69 = None
    add_60: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_70, primals_88);  mul_70 = primals_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_42: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_60, [0, 2, 1]);  add_60 = None
    permute_43: "f32[196, 196]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    clone_43: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
    view_51: "f32[6144, 196]" = torch.ops.aten.view.default(clone_43, [6144, 196]);  clone_43 = None
    mm_8: "f32[6144, 196]" = torch.ops.aten.mm.default(view_51, permute_43)
    view_52: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_8, [8, 768, 196]);  mm_8 = None
    add_61: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_52, primals_90);  view_52 = primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_44: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_61, [0, 2, 1]);  add_61 = None
    mul_71: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_50, permute_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_53: "f32[1568, 768]" = torch.ops.aten.view.default(mul_71, [1568, 768]);  mul_71 = None
    permute_45: "f32[768, 256]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_17: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_92, view_53, permute_45);  primals_92 = None
    view_54: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_17, [8, 196, 256]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_44: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_54);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_62: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_55, clone_44);  clone_44 = None
    clone_45: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_45, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 196, 1]" = var_mean_18[0]
    getitem_55: "f32[8, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    add_63: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_18: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_18: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_45, getitem_55);  clone_45 = None
    mul_72: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_73: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_72, primals_93);  mul_72 = None
    add_64: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_73, primals_94);  mul_73 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_55: "f32[1568, 256]" = torch.ops.aten.view.default(add_64, [1568, 256]);  add_64 = None
    permute_46: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_18: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_96, view_55, permute_46);  primals_96 = None
    view_56: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_18, [8, 196, 1536]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_74: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_56, 0.5)
    mul_75: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_56, 0.7071067811865476)
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
    clone_47: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_57, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_47, [2], correction = 0, keepdim = True)
    getitem_58: "f32[8, 196, 1]" = var_mean_19[0]
    getitem_59: "f32[8, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    add_66: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_19: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_19: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_47, getitem_59);  clone_47 = None
    mul_77: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_78: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_77, primals_97);  mul_77 = None
    add_67: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_78, primals_98);  mul_78 = primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_47: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_67, [0, 2, 1]);  add_67 = None
    permute_48: "f32[196, 196]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    clone_48: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    view_57: "f32[6144, 196]" = torch.ops.aten.view.default(clone_48, [6144, 196]);  clone_48 = None
    mm_9: "f32[6144, 196]" = torch.ops.aten.mm.default(view_57, permute_48)
    view_58: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_9, [8, 768, 196]);  mm_9 = None
    add_68: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_58, primals_100);  view_58 = primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_49: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_68, [0, 2, 1]);  add_68 = None
    mul_79: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_56, permute_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_59: "f32[1568, 768]" = torch.ops.aten.view.default(mul_79, [1568, 768]);  mul_79 = None
    permute_50: "f32[768, 256]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_19: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_102, view_59, permute_50);  primals_102 = None
    view_60: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_19, [8, 196, 256]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_49: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_69: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_62, clone_49);  clone_49 = None
    clone_50: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_69, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_50, [2], correction = 0, keepdim = True)
    getitem_60: "f32[8, 196, 1]" = var_mean_20[0]
    getitem_61: "f32[8, 196, 1]" = var_mean_20[1];  var_mean_20 = None
    add_70: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
    rsqrt_20: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_20: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_50, getitem_61);  clone_50 = None
    mul_80: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_81: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_80, primals_103);  mul_80 = None
    add_71: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_81, primals_104);  mul_81 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_61: "f32[1568, 256]" = torch.ops.aten.view.default(add_71, [1568, 256]);  add_71 = None
    permute_51: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_20: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_106, view_61, permute_51);  primals_106 = None
    view_62: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_20, [8, 196, 1536]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_82: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_62, 0.5)
    mul_83: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_62, 0.7071067811865476)
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
    clone_52: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_63, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_52, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 196, 1]" = var_mean_21[0]
    getitem_65: "f32[8, 196, 1]" = var_mean_21[1];  var_mean_21 = None
    add_73: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_21: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_21: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_52, getitem_65);  clone_52 = None
    mul_85: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_86: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_85, primals_107);  mul_85 = None
    add_74: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_86, primals_108);  mul_86 = primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_52: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_74, [0, 2, 1]);  add_74 = None
    permute_53: "f32[196, 196]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    clone_53: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_63: "f32[6144, 196]" = torch.ops.aten.view.default(clone_53, [6144, 196]);  clone_53 = None
    mm_10: "f32[6144, 196]" = torch.ops.aten.mm.default(view_63, permute_53)
    view_64: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_10, [8, 768, 196]);  mm_10 = None
    add_75: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_64, primals_110);  view_64 = primals_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_54: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_75, [0, 2, 1]);  add_75 = None
    mul_87: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_62, permute_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_65: "f32[1568, 768]" = torch.ops.aten.view.default(mul_87, [1568, 768]);  mul_87 = None
    permute_55: "f32[768, 256]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_21: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_112, view_65, permute_55);  primals_112 = None
    view_66: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_21, [8, 196, 256]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_54: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_66);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_76: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_69, clone_54);  clone_54 = None
    clone_55: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_55, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 196, 1]" = var_mean_22[0]
    getitem_67: "f32[8, 196, 1]" = var_mean_22[1];  var_mean_22 = None
    add_77: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_22: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_22: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_55, getitem_67);  clone_55 = None
    mul_88: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_89: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_88, primals_113);  mul_88 = None
    add_78: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_89, primals_114);  mul_89 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_67: "f32[1568, 256]" = torch.ops.aten.view.default(add_78, [1568, 256]);  add_78 = None
    permute_56: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_22: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_116, view_67, permute_56);  primals_116 = None
    view_68: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_22, [8, 196, 1536]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_90: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_68, 0.5)
    mul_91: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476)
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
    clone_57: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_69, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_57, [2], correction = 0, keepdim = True)
    getitem_70: "f32[8, 196, 1]" = var_mean_23[0]
    getitem_71: "f32[8, 196, 1]" = var_mean_23[1];  var_mean_23 = None
    add_80: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_23: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_23: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_57, getitem_71);  clone_57 = None
    mul_93: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_94: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_93, primals_117);  mul_93 = None
    add_81: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_94, primals_118);  mul_94 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_57: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_81, [0, 2, 1]);  add_81 = None
    permute_58: "f32[196, 196]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    clone_58: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    view_69: "f32[6144, 196]" = torch.ops.aten.view.default(clone_58, [6144, 196]);  clone_58 = None
    mm_11: "f32[6144, 196]" = torch.ops.aten.mm.default(view_69, permute_58)
    view_70: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_11, [8, 768, 196]);  mm_11 = None
    add_82: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_70, primals_120);  view_70 = primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_59: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_82, [0, 2, 1]);  add_82 = None
    mul_95: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_68, permute_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_71: "f32[1568, 768]" = torch.ops.aten.view.default(mul_95, [1568, 768]);  mul_95 = None
    permute_60: "f32[768, 256]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_23: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_122, view_71, permute_60);  primals_122 = None
    view_72: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_23, [8, 196, 256]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_59: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_83: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_76, clone_59);  clone_59 = None
    clone_60: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_83, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_60, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 196, 1]" = var_mean_24[0]
    getitem_73: "f32[8, 196, 1]" = var_mean_24[1];  var_mean_24 = None
    add_84: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_24: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_24: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_60, getitem_73);  clone_60 = None
    mul_96: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_97: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_96, primals_123);  mul_96 = None
    add_85: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_97, primals_124);  mul_97 = primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_73: "f32[1568, 256]" = torch.ops.aten.view.default(add_85, [1568, 256]);  add_85 = None
    permute_61: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_24: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_126, view_73, permute_61);  primals_126 = None
    view_74: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_24, [8, 196, 1536]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_98: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_74, 0.5)
    mul_99: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_74, 0.7071067811865476)
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
    clone_62: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_75, memory_format = torch.contiguous_format)
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_62, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 196, 1]" = var_mean_25[0]
    getitem_77: "f32[8, 196, 1]" = var_mean_25[1];  var_mean_25 = None
    add_87: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_25: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_25: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_62, getitem_77);  clone_62 = None
    mul_101: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    mul_102: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_101, primals_127);  mul_101 = None
    add_88: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_102, primals_128);  mul_102 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_62: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_88, [0, 2, 1]);  add_88 = None
    permute_63: "f32[196, 196]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    clone_63: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_75: "f32[6144, 196]" = torch.ops.aten.view.default(clone_63, [6144, 196]);  clone_63 = None
    mm_12: "f32[6144, 196]" = torch.ops.aten.mm.default(view_75, permute_63)
    view_76: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_12, [8, 768, 196]);  mm_12 = None
    add_89: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_76, primals_130);  view_76 = primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_64: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_89, [0, 2, 1]);  add_89 = None
    mul_103: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_74, permute_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_77: "f32[1568, 768]" = torch.ops.aten.view.default(mul_103, [1568, 768]);  mul_103 = None
    permute_65: "f32[768, 256]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_25: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_132, view_77, permute_65);  primals_132 = None
    view_78: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_25, [8, 196, 256]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_64: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_78);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_90: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_83, clone_64);  clone_64 = None
    clone_65: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_65, [2], correction = 0, keepdim = True)
    getitem_78: "f32[8, 196, 1]" = var_mean_26[0]
    getitem_79: "f32[8, 196, 1]" = var_mean_26[1];  var_mean_26 = None
    add_91: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_26: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_26: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_65, getitem_79);  clone_65 = None
    mul_104: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    mul_105: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_104, primals_133);  mul_104 = None
    add_92: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_105, primals_134);  mul_105 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_79: "f32[1568, 256]" = torch.ops.aten.view.default(add_92, [1568, 256]);  add_92 = None
    permute_66: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_26: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_136, view_79, permute_66);  primals_136 = None
    view_80: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_26, [8, 196, 1536]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_106: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_80, 0.5)
    mul_107: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476)
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
    clone_67: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_81, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_67, [2], correction = 0, keepdim = True)
    getitem_82: "f32[8, 196, 1]" = var_mean_27[0]
    getitem_83: "f32[8, 196, 1]" = var_mean_27[1];  var_mean_27 = None
    add_94: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_27: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_27: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_67, getitem_83);  clone_67 = None
    mul_109: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    mul_110: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_109, primals_137);  mul_109 = None
    add_95: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_110, primals_138);  mul_110 = primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_67: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_95, [0, 2, 1]);  add_95 = None
    permute_68: "f32[196, 196]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    clone_68: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    view_81: "f32[6144, 196]" = torch.ops.aten.view.default(clone_68, [6144, 196]);  clone_68 = None
    mm_13: "f32[6144, 196]" = torch.ops.aten.mm.default(view_81, permute_68)
    view_82: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_13, [8, 768, 196]);  mm_13 = None
    add_96: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_82, primals_140);  view_82 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_69: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_96, [0, 2, 1]);  add_96 = None
    mul_111: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_80, permute_69)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_83: "f32[1568, 768]" = torch.ops.aten.view.default(mul_111, [1568, 768]);  mul_111 = None
    permute_70: "f32[768, 256]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_27: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_142, view_83, permute_70);  primals_142 = None
    view_84: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_27, [8, 196, 256]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_69: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_84);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_97: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_90, clone_69);  clone_69 = None
    clone_70: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_97, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_70, [2], correction = 0, keepdim = True)
    getitem_84: "f32[8, 196, 1]" = var_mean_28[0]
    getitem_85: "f32[8, 196, 1]" = var_mean_28[1];  var_mean_28 = None
    add_98: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
    rsqrt_28: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_28: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_70, getitem_85);  clone_70 = None
    mul_112: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    mul_113: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_112, primals_143);  mul_112 = None
    add_99: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_113, primals_144);  mul_113 = primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_85: "f32[1568, 256]" = torch.ops.aten.view.default(add_99, [1568, 256]);  add_99 = None
    permute_71: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_28: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_146, view_85, permute_71);  primals_146 = None
    view_86: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_28, [8, 196, 1536]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_114: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_86, 0.5)
    mul_115: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_86, 0.7071067811865476)
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
    clone_72: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_87, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_72, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 196, 1]" = var_mean_29[0]
    getitem_89: "f32[8, 196, 1]" = var_mean_29[1];  var_mean_29 = None
    add_101: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_29: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_29: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_72, getitem_89);  clone_72 = None
    mul_117: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    mul_118: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_117, primals_147);  mul_117 = None
    add_102: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_118, primals_148);  mul_118 = primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_72: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_102, [0, 2, 1]);  add_102 = None
    permute_73: "f32[196, 196]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    clone_73: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    view_87: "f32[6144, 196]" = torch.ops.aten.view.default(clone_73, [6144, 196]);  clone_73 = None
    mm_14: "f32[6144, 196]" = torch.ops.aten.mm.default(view_87, permute_73)
    view_88: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_14, [8, 768, 196]);  mm_14 = None
    add_103: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_88, primals_150);  view_88 = primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_74: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_103, [0, 2, 1]);  add_103 = None
    mul_119: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_86, permute_74)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_89: "f32[1568, 768]" = torch.ops.aten.view.default(mul_119, [1568, 768]);  mul_119 = None
    permute_75: "f32[768, 256]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_29: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_152, view_89, permute_75);  primals_152 = None
    view_90: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_29, [8, 196, 256]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_74: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_90);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_104: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_97, clone_74);  clone_74 = None
    clone_75: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_104, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_75, [2], correction = 0, keepdim = True)
    getitem_90: "f32[8, 196, 1]" = var_mean_30[0]
    getitem_91: "f32[8, 196, 1]" = var_mean_30[1];  var_mean_30 = None
    add_105: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
    rsqrt_30: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_30: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_75, getitem_91);  clone_75 = None
    mul_120: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    mul_121: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_120, primals_153);  mul_120 = None
    add_106: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_121, primals_154);  mul_121 = primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_91: "f32[1568, 256]" = torch.ops.aten.view.default(add_106, [1568, 256]);  add_106 = None
    permute_76: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_30: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_156, view_91, permute_76);  primals_156 = None
    view_92: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_30, [8, 196, 1536]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_122: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_92, 0.5)
    mul_123: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_92, 0.7071067811865476)
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
    clone_77: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_93, memory_format = torch.contiguous_format)
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_77, [2], correction = 0, keepdim = True)
    getitem_94: "f32[8, 196, 1]" = var_mean_31[0]
    getitem_95: "f32[8, 196, 1]" = var_mean_31[1];  var_mean_31 = None
    add_108: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_31: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_31: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_77, getitem_95);  clone_77 = None
    mul_125: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    mul_126: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_125, primals_157);  mul_125 = None
    add_109: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_126, primals_158);  mul_126 = primals_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_77: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_109, [0, 2, 1]);  add_109 = None
    permute_78: "f32[196, 196]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    clone_78: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_93: "f32[6144, 196]" = torch.ops.aten.view.default(clone_78, [6144, 196]);  clone_78 = None
    mm_15: "f32[6144, 196]" = torch.ops.aten.mm.default(view_93, permute_78)
    view_94: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_15, [8, 768, 196]);  mm_15 = None
    add_110: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_94, primals_160);  view_94 = primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_79: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_110, [0, 2, 1]);  add_110 = None
    mul_127: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_92, permute_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_95: "f32[1568, 768]" = torch.ops.aten.view.default(mul_127, [1568, 768]);  mul_127 = None
    permute_80: "f32[768, 256]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    addmm_31: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_162, view_95, permute_80);  primals_162 = None
    view_96: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_31, [8, 196, 256]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_79: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_111: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_104, clone_79);  clone_79 = None
    clone_80: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_80, [2], correction = 0, keepdim = True)
    getitem_96: "f32[8, 196, 1]" = var_mean_32[0]
    getitem_97: "f32[8, 196, 1]" = var_mean_32[1];  var_mean_32 = None
    add_112: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
    rsqrt_32: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_32: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_80, getitem_97);  clone_80 = None
    mul_128: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    mul_129: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_128, primals_163);  mul_128 = None
    add_113: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_129, primals_164);  mul_129 = primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_97: "f32[1568, 256]" = torch.ops.aten.view.default(add_113, [1568, 256]);  add_113 = None
    permute_81: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    addmm_32: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_166, view_97, permute_81);  primals_166 = None
    view_98: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_32, [8, 196, 1536]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_130: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_131: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
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
    clone_82: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_99, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_82, [2], correction = 0, keepdim = True)
    getitem_100: "f32[8, 196, 1]" = var_mean_33[0]
    getitem_101: "f32[8, 196, 1]" = var_mean_33[1];  var_mean_33 = None
    add_115: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_33: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_33: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_82, getitem_101);  clone_82 = None
    mul_133: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    mul_134: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_133, primals_167);  mul_133 = None
    add_116: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_134, primals_168);  mul_134 = primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_82: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_116, [0, 2, 1]);  add_116 = None
    permute_83: "f32[196, 196]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    clone_83: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_99: "f32[6144, 196]" = torch.ops.aten.view.default(clone_83, [6144, 196]);  clone_83 = None
    mm_16: "f32[6144, 196]" = torch.ops.aten.mm.default(view_99, permute_83)
    view_100: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_16, [8, 768, 196]);  mm_16 = None
    add_117: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_100, primals_170);  view_100 = primals_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_84: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_117, [0, 2, 1]);  add_117 = None
    mul_135: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_98, permute_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_101: "f32[1568, 768]" = torch.ops.aten.view.default(mul_135, [1568, 768]);  mul_135 = None
    permute_85: "f32[768, 256]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm_33: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_172, view_101, permute_85);  primals_172 = None
    view_102: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_33, [8, 196, 256]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_84: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_102);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_118: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_111, clone_84);  clone_84 = None
    clone_85: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_118, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
    getitem_102: "f32[8, 196, 1]" = var_mean_34[0]
    getitem_103: "f32[8, 196, 1]" = var_mean_34[1];  var_mean_34 = None
    add_119: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-06);  getitem_102 = None
    rsqrt_34: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_34: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_85, getitem_103);  clone_85 = None
    mul_136: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    mul_137: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_136, primals_173);  mul_136 = None
    add_120: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_137, primals_174);  mul_137 = primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_103: "f32[1568, 256]" = torch.ops.aten.view.default(add_120, [1568, 256]);  add_120 = None
    permute_86: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_34: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_176, view_103, permute_86);  primals_176 = None
    view_104: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_34, [8, 196, 1536]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_138: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_104, 0.5)
    mul_139: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_104, 0.7071067811865476)
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
    clone_87: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_105, memory_format = torch.contiguous_format)
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_87, [2], correction = 0, keepdim = True)
    getitem_106: "f32[8, 196, 1]" = var_mean_35[0]
    getitem_107: "f32[8, 196, 1]" = var_mean_35[1];  var_mean_35 = None
    add_122: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_35: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_35: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_87, getitem_107);  clone_87 = None
    mul_141: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    mul_142: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_141, primals_177);  mul_141 = None
    add_123: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_142, primals_178);  mul_142 = primals_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_87: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_123, [0, 2, 1]);  add_123 = None
    permute_88: "f32[196, 196]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    clone_88: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    view_105: "f32[6144, 196]" = torch.ops.aten.view.default(clone_88, [6144, 196]);  clone_88 = None
    mm_17: "f32[6144, 196]" = torch.ops.aten.mm.default(view_105, permute_88)
    view_106: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_17, [8, 768, 196]);  mm_17 = None
    add_124: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_106, primals_180);  view_106 = primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_89: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_124, [0, 2, 1]);  add_124 = None
    mul_143: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_104, permute_89)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_107: "f32[1568, 768]" = torch.ops.aten.view.default(mul_143, [1568, 768]);  mul_143 = None
    permute_90: "f32[768, 256]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_35: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_182, view_107, permute_90);  primals_182 = None
    view_108: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_35, [8, 196, 256]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_89: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_125: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_118, clone_89);  clone_89 = None
    clone_90: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_90, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 196, 1]" = var_mean_36[0]
    getitem_109: "f32[8, 196, 1]" = var_mean_36[1];  var_mean_36 = None
    add_126: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_36: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_36: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_90, getitem_109);  clone_90 = None
    mul_144: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    mul_145: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_144, primals_183);  mul_144 = None
    add_127: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_145, primals_184);  mul_145 = primals_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_109: "f32[1568, 256]" = torch.ops.aten.view.default(add_127, [1568, 256]);  add_127 = None
    permute_91: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    addmm_36: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_186, view_109, permute_91);  primals_186 = None
    view_110: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_36, [8, 196, 1536]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_146: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_110, 0.5)
    mul_147: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_110, 0.7071067811865476)
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
    clone_92: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_111, memory_format = torch.contiguous_format)
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_92, [2], correction = 0, keepdim = True)
    getitem_112: "f32[8, 196, 1]" = var_mean_37[0]
    getitem_113: "f32[8, 196, 1]" = var_mean_37[1];  var_mean_37 = None
    add_129: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_37: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_37: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_92, getitem_113);  clone_92 = None
    mul_149: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    mul_150: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_149, primals_187);  mul_149 = None
    add_130: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_150, primals_188);  mul_150 = primals_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_92: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_130, [0, 2, 1]);  add_130 = None
    permute_93: "f32[196, 196]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    clone_93: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    view_111: "f32[6144, 196]" = torch.ops.aten.view.default(clone_93, [6144, 196]);  clone_93 = None
    mm_18: "f32[6144, 196]" = torch.ops.aten.mm.default(view_111, permute_93)
    view_112: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_18, [8, 768, 196]);  mm_18 = None
    add_131: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_112, primals_190);  view_112 = primals_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_94: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_131, [0, 2, 1]);  add_131 = None
    mul_151: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_110, permute_94)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_113: "f32[1568, 768]" = torch.ops.aten.view.default(mul_151, [1568, 768]);  mul_151 = None
    permute_95: "f32[768, 256]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
    addmm_37: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_192, view_113, permute_95);  primals_192 = None
    view_114: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_37, [8, 196, 256]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_94: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_114);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_132: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_125, clone_94);  clone_94 = None
    clone_95: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_132, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_95, [2], correction = 0, keepdim = True)
    getitem_114: "f32[8, 196, 1]" = var_mean_38[0]
    getitem_115: "f32[8, 196, 1]" = var_mean_38[1];  var_mean_38 = None
    add_133: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-06);  getitem_114 = None
    rsqrt_38: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_38: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_95, getitem_115);  clone_95 = None
    mul_152: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    mul_153: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_152, primals_193);  mul_152 = None
    add_134: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_153, primals_194);  mul_153 = primals_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_115: "f32[1568, 256]" = torch.ops.aten.view.default(add_134, [1568, 256]);  add_134 = None
    permute_96: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    addmm_38: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_196, view_115, permute_96);  primals_196 = None
    view_116: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_38, [8, 196, 1536]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_154: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_116, 0.5)
    mul_155: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476)
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
    clone_97: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_117, memory_format = torch.contiguous_format)
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_97, [2], correction = 0, keepdim = True)
    getitem_118: "f32[8, 196, 1]" = var_mean_39[0]
    getitem_119: "f32[8, 196, 1]" = var_mean_39[1];  var_mean_39 = None
    add_136: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_39: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_39: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_97, getitem_119);  clone_97 = None
    mul_157: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    mul_158: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_157, primals_197);  mul_157 = None
    add_137: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_158, primals_198);  mul_158 = primals_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_97: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_137, [0, 2, 1]);  add_137 = None
    permute_98: "f32[196, 196]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    clone_98: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    view_117: "f32[6144, 196]" = torch.ops.aten.view.default(clone_98, [6144, 196]);  clone_98 = None
    mm_19: "f32[6144, 196]" = torch.ops.aten.mm.default(view_117, permute_98)
    view_118: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_19, [8, 768, 196]);  mm_19 = None
    add_138: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_118, primals_200);  view_118 = primals_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_99: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_138, [0, 2, 1]);  add_138 = None
    mul_159: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_116, permute_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_119: "f32[1568, 768]" = torch.ops.aten.view.default(mul_159, [1568, 768]);  mul_159 = None
    permute_100: "f32[768, 256]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
    addmm_39: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_202, view_119, permute_100);  primals_202 = None
    view_120: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_39, [8, 196, 256]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_99: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_139: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_132, clone_99);  clone_99 = None
    clone_100: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_139, memory_format = torch.contiguous_format)
    var_mean_40 = torch.ops.aten.var_mean.correction(clone_100, [2], correction = 0, keepdim = True)
    getitem_120: "f32[8, 196, 1]" = var_mean_40[0]
    getitem_121: "f32[8, 196, 1]" = var_mean_40[1];  var_mean_40 = None
    add_140: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
    rsqrt_40: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_40: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_100, getitem_121);  clone_100 = None
    mul_160: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    mul_161: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_160, primals_203);  mul_160 = None
    add_141: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_161, primals_204);  mul_161 = primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_121: "f32[1568, 256]" = torch.ops.aten.view.default(add_141, [1568, 256]);  add_141 = None
    permute_101: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    addmm_40: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_206, view_121, permute_101);  primals_206 = None
    view_122: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_40, [8, 196, 1536]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_162: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_122, 0.5)
    mul_163: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_122, 0.7071067811865476)
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
    clone_102: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_123, memory_format = torch.contiguous_format)
    var_mean_41 = torch.ops.aten.var_mean.correction(clone_102, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 196, 1]" = var_mean_41[0]
    getitem_125: "f32[8, 196, 1]" = var_mean_41[1];  var_mean_41 = None
    add_143: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
    rsqrt_41: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_41: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_102, getitem_125);  clone_102 = None
    mul_165: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    mul_166: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_165, primals_207);  mul_165 = None
    add_144: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_166, primals_208);  mul_166 = primals_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_102: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_144, [0, 2, 1]);  add_144 = None
    permute_103: "f32[196, 196]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    clone_103: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    view_123: "f32[6144, 196]" = torch.ops.aten.view.default(clone_103, [6144, 196]);  clone_103 = None
    mm_20: "f32[6144, 196]" = torch.ops.aten.mm.default(view_123, permute_103)
    view_124: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_20, [8, 768, 196]);  mm_20 = None
    add_145: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_124, primals_210);  view_124 = primals_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_104: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_145, [0, 2, 1]);  add_145 = None
    mul_167: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_122, permute_104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_125: "f32[1568, 768]" = torch.ops.aten.view.default(mul_167, [1568, 768]);  mul_167 = None
    permute_105: "f32[768, 256]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
    addmm_41: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_212, view_125, permute_105);  primals_212 = None
    view_126: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_41, [8, 196, 256]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_104: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_126);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_146: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_139, clone_104);  clone_104 = None
    clone_105: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_146, memory_format = torch.contiguous_format)
    var_mean_42 = torch.ops.aten.var_mean.correction(clone_105, [2], correction = 0, keepdim = True)
    getitem_126: "f32[8, 196, 1]" = var_mean_42[0]
    getitem_127: "f32[8, 196, 1]" = var_mean_42[1];  var_mean_42 = None
    add_147: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
    rsqrt_42: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_42: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_105, getitem_127);  clone_105 = None
    mul_168: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    mul_169: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_168, primals_213);  mul_168 = None
    add_148: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_169, primals_214);  mul_169 = primals_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_127: "f32[1568, 256]" = torch.ops.aten.view.default(add_148, [1568, 256]);  add_148 = None
    permute_106: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    addmm_42: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_216, view_127, permute_106);  primals_216 = None
    view_128: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_42, [8, 196, 1536]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_170: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_128, 0.5)
    mul_171: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476)
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
    clone_107: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_129, memory_format = torch.contiguous_format)
    var_mean_43 = torch.ops.aten.var_mean.correction(clone_107, [2], correction = 0, keepdim = True)
    getitem_130: "f32[8, 196, 1]" = var_mean_43[0]
    getitem_131: "f32[8, 196, 1]" = var_mean_43[1];  var_mean_43 = None
    add_150: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
    rsqrt_43: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_43: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_107, getitem_131);  clone_107 = None
    mul_173: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    mul_174: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_173, primals_217);  mul_173 = None
    add_151: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_174, primals_218);  mul_174 = primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_107: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_151, [0, 2, 1]);  add_151 = None
    permute_108: "f32[196, 196]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
    clone_108: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    view_129: "f32[6144, 196]" = torch.ops.aten.view.default(clone_108, [6144, 196]);  clone_108 = None
    mm_21: "f32[6144, 196]" = torch.ops.aten.mm.default(view_129, permute_108)
    view_130: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_21, [8, 768, 196]);  mm_21 = None
    add_152: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_130, primals_220);  view_130 = primals_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_109: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_152, [0, 2, 1]);  add_152 = None
    mul_175: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_128, permute_109)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_131: "f32[1568, 768]" = torch.ops.aten.view.default(mul_175, [1568, 768]);  mul_175 = None
    permute_110: "f32[768, 256]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
    addmm_43: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_222, view_131, permute_110);  primals_222 = None
    view_132: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_43, [8, 196, 256]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_109: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_132);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_153: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_146, clone_109);  clone_109 = None
    clone_110: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_153, memory_format = torch.contiguous_format)
    var_mean_44 = torch.ops.aten.var_mean.correction(clone_110, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 196, 1]" = var_mean_44[0]
    getitem_133: "f32[8, 196, 1]" = var_mean_44[1];  var_mean_44 = None
    add_154: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_44: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_44: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_110, getitem_133);  clone_110 = None
    mul_176: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    mul_177: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_176, primals_223);  mul_176 = None
    add_155: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_177, primals_224);  mul_177 = primals_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_133: "f32[1568, 256]" = torch.ops.aten.view.default(add_155, [1568, 256]);  add_155 = None
    permute_111: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
    addmm_44: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_226, view_133, permute_111);  primals_226 = None
    view_134: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_44, [8, 196, 1536]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_178: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_134, 0.5)
    mul_179: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_134, 0.7071067811865476)
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
    clone_112: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_135, memory_format = torch.contiguous_format)
    var_mean_45 = torch.ops.aten.var_mean.correction(clone_112, [2], correction = 0, keepdim = True)
    getitem_136: "f32[8, 196, 1]" = var_mean_45[0]
    getitem_137: "f32[8, 196, 1]" = var_mean_45[1];  var_mean_45 = None
    add_157: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05);  getitem_136 = None
    rsqrt_45: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_45: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_112, getitem_137);  clone_112 = None
    mul_181: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    mul_182: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_181, primals_227);  mul_181 = None
    add_158: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_182, primals_228);  mul_182 = primals_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_112: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_158, [0, 2, 1]);  add_158 = None
    permute_113: "f32[196, 196]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
    clone_113: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    view_135: "f32[6144, 196]" = torch.ops.aten.view.default(clone_113, [6144, 196]);  clone_113 = None
    mm_22: "f32[6144, 196]" = torch.ops.aten.mm.default(view_135, permute_113)
    view_136: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_22, [8, 768, 196]);  mm_22 = None
    add_159: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_136, primals_230);  view_136 = primals_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_114: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_159, [0, 2, 1]);  add_159 = None
    mul_183: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_134, permute_114)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_137: "f32[1568, 768]" = torch.ops.aten.view.default(mul_183, [1568, 768]);  mul_183 = None
    permute_115: "f32[768, 256]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
    addmm_45: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_232, view_137, permute_115);  primals_232 = None
    view_138: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_45, [8, 196, 256]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_114: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_138);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_160: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_153, clone_114);  clone_114 = None
    clone_115: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_160, memory_format = torch.contiguous_format)
    var_mean_46 = torch.ops.aten.var_mean.correction(clone_115, [2], correction = 0, keepdim = True)
    getitem_138: "f32[8, 196, 1]" = var_mean_46[0]
    getitem_139: "f32[8, 196, 1]" = var_mean_46[1];  var_mean_46 = None
    add_161: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-06);  getitem_138 = None
    rsqrt_46: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_46: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_115, getitem_139);  clone_115 = None
    mul_184: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    mul_185: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_184, primals_233);  mul_184 = None
    add_162: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_185, primals_234);  mul_185 = primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_139: "f32[1568, 256]" = torch.ops.aten.view.default(add_162, [1568, 256]);  add_162 = None
    permute_116: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    addmm_46: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_236, view_139, permute_116);  primals_236 = None
    view_140: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_46, [8, 196, 1536]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_186: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_140, 0.5)
    mul_187: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_140, 0.7071067811865476)
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
    clone_117: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_141, memory_format = torch.contiguous_format)
    var_mean_47 = torch.ops.aten.var_mean.correction(clone_117, [2], correction = 0, keepdim = True)
    getitem_142: "f32[8, 196, 1]" = var_mean_47[0]
    getitem_143: "f32[8, 196, 1]" = var_mean_47[1];  var_mean_47 = None
    add_164: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
    rsqrt_47: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_47: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_117, getitem_143);  clone_117 = None
    mul_189: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    mul_190: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_189, primals_237);  mul_189 = None
    add_165: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_190, primals_238);  mul_190 = primals_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_117: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_165, [0, 2, 1]);  add_165 = None
    permute_118: "f32[196, 196]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
    clone_118: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_141: "f32[6144, 196]" = torch.ops.aten.view.default(clone_118, [6144, 196]);  clone_118 = None
    mm_23: "f32[6144, 196]" = torch.ops.aten.mm.default(view_141, permute_118)
    view_142: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_23, [8, 768, 196]);  mm_23 = None
    add_166: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_142, primals_240);  view_142 = primals_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_119: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_166, [0, 2, 1]);  add_166 = None
    mul_191: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_140, permute_119)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_143: "f32[1568, 768]" = torch.ops.aten.view.default(mul_191, [1568, 768]);  mul_191 = None
    permute_120: "f32[768, 256]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
    addmm_47: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_242, view_143, permute_120);  primals_242 = None
    view_144: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_47, [8, 196, 256]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_119: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_167: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_160, clone_119);  clone_119 = None
    clone_120: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format)
    var_mean_48 = torch.ops.aten.var_mean.correction(clone_120, [2], correction = 0, keepdim = True)
    getitem_144: "f32[8, 196, 1]" = var_mean_48[0]
    getitem_145: "f32[8, 196, 1]" = var_mean_48[1];  var_mean_48 = None
    add_168: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_48: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_48: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_120, getitem_145);  clone_120 = None
    mul_192: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    mul_193: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_192, primals_243);  mul_192 = None
    add_169: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_193, primals_244);  mul_193 = primals_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_145: "f32[1568, 256]" = torch.ops.aten.view.default(add_169, [1568, 256]);  add_169 = None
    permute_121: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
    addmm_48: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_246, view_145, permute_121);  primals_246 = None
    view_146: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_48, [8, 196, 1536]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_194: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_146, 0.5)
    mul_195: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476)
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
    clone_122: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_147, memory_format = torch.contiguous_format)
    var_mean_49 = torch.ops.aten.var_mean.correction(clone_122, [2], correction = 0, keepdim = True)
    getitem_148: "f32[8, 196, 1]" = var_mean_49[0]
    getitem_149: "f32[8, 196, 1]" = var_mean_49[1];  var_mean_49 = None
    add_171: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
    rsqrt_49: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_49: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_122, getitem_149);  clone_122 = None
    mul_197: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    mul_198: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_197, primals_247);  mul_197 = None
    add_172: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_198, primals_248);  mul_198 = primals_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_122: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_172, [0, 2, 1]);  add_172 = None
    permute_123: "f32[196, 196]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
    clone_123: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_147: "f32[6144, 196]" = torch.ops.aten.view.default(clone_123, [6144, 196]);  clone_123 = None
    mm_24: "f32[6144, 196]" = torch.ops.aten.mm.default(view_147, permute_123)
    view_148: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_24, [8, 768, 196]);  mm_24 = None
    add_173: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_148, primals_250);  view_148 = primals_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_124: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_173, [0, 2, 1]);  add_173 = None
    mul_199: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_146, permute_124)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_149: "f32[1568, 768]" = torch.ops.aten.view.default(mul_199, [1568, 768]);  mul_199 = None
    permute_125: "f32[768, 256]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
    addmm_49: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_252, view_149, permute_125);  primals_252 = None
    view_150: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_49, [8, 196, 256]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_124: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_150);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_174: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_167, clone_124);  clone_124 = None
    clone_125: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_174, memory_format = torch.contiguous_format)
    var_mean_50 = torch.ops.aten.var_mean.correction(clone_125, [2], correction = 0, keepdim = True)
    getitem_150: "f32[8, 196, 1]" = var_mean_50[0]
    getitem_151: "f32[8, 196, 1]" = var_mean_50[1];  var_mean_50 = None
    add_175: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-06);  getitem_150 = None
    rsqrt_50: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_50: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_125, getitem_151);  clone_125 = None
    mul_200: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    mul_201: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_200, primals_253);  mul_200 = None
    add_176: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_201, primals_254);  mul_201 = primals_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_151: "f32[1568, 256]" = torch.ops.aten.view.default(add_176, [1568, 256]);  add_176 = None
    permute_126: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_50: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_256, view_151, permute_126);  primals_256 = None
    view_152: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_50, [8, 196, 1536]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_202: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_152, 0.5)
    mul_203: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_152, 0.7071067811865476)
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
    clone_127: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_153, memory_format = torch.contiguous_format)
    var_mean_51 = torch.ops.aten.var_mean.correction(clone_127, [2], correction = 0, keepdim = True)
    getitem_154: "f32[8, 196, 1]" = var_mean_51[0]
    getitem_155: "f32[8, 196, 1]" = var_mean_51[1];  var_mean_51 = None
    add_178: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
    rsqrt_51: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_51: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_127, getitem_155);  clone_127 = None
    mul_205: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    mul_206: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_205, primals_257);  mul_205 = None
    add_179: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_206, primals_258);  mul_206 = primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_127: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_179, [0, 2, 1]);  add_179 = None
    permute_128: "f32[196, 196]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
    clone_128: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    view_153: "f32[6144, 196]" = torch.ops.aten.view.default(clone_128, [6144, 196]);  clone_128 = None
    mm_25: "f32[6144, 196]" = torch.ops.aten.mm.default(view_153, permute_128)
    view_154: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_25, [8, 768, 196]);  mm_25 = None
    add_180: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_154, primals_260);  view_154 = primals_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_129: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_180, [0, 2, 1]);  add_180 = None
    mul_207: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_152, permute_129)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_155: "f32[1568, 768]" = torch.ops.aten.view.default(mul_207, [1568, 768]);  mul_207 = None
    permute_130: "f32[768, 256]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
    addmm_51: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_262, view_155, permute_130);  primals_262 = None
    view_156: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_51, [8, 196, 256]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_129: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_156);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_181: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_174, clone_129);  clone_129 = None
    clone_130: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_181, memory_format = torch.contiguous_format)
    var_mean_52 = torch.ops.aten.var_mean.correction(clone_130, [2], correction = 0, keepdim = True)
    getitem_156: "f32[8, 196, 1]" = var_mean_52[0]
    getitem_157: "f32[8, 196, 1]" = var_mean_52[1];  var_mean_52 = None
    add_182: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
    rsqrt_52: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_52: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_130, getitem_157);  clone_130 = None
    mul_208: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    mul_209: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_208, primals_263);  mul_208 = None
    add_183: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_209, primals_264);  mul_209 = primals_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_157: "f32[1568, 256]" = torch.ops.aten.view.default(add_183, [1568, 256]);  add_183 = None
    permute_131: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
    addmm_52: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_266, view_157, permute_131);  primals_266 = None
    view_158: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_52, [8, 196, 1536]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_210: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
    mul_211: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476)
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
    clone_132: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_159, memory_format = torch.contiguous_format)
    var_mean_53 = torch.ops.aten.var_mean.correction(clone_132, [2], correction = 0, keepdim = True)
    getitem_160: "f32[8, 196, 1]" = var_mean_53[0]
    getitem_161: "f32[8, 196, 1]" = var_mean_53[1];  var_mean_53 = None
    add_185: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
    rsqrt_53: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_53: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_132, getitem_161);  clone_132 = None
    mul_213: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    mul_214: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_213, primals_267);  mul_213 = None
    add_186: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_214, primals_268);  mul_214 = primals_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_132: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_186, [0, 2, 1]);  add_186 = None
    permute_133: "f32[196, 196]" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
    clone_133: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    view_159: "f32[6144, 196]" = torch.ops.aten.view.default(clone_133, [6144, 196]);  clone_133 = None
    mm_26: "f32[6144, 196]" = torch.ops.aten.mm.default(view_159, permute_133)
    view_160: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_26, [8, 768, 196]);  mm_26 = None
    add_187: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_160, primals_270);  view_160 = primals_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_134: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_187, [0, 2, 1]);  add_187 = None
    mul_215: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_158, permute_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_161: "f32[1568, 768]" = torch.ops.aten.view.default(mul_215, [1568, 768]);  mul_215 = None
    permute_135: "f32[768, 256]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
    addmm_53: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_272, view_161, permute_135);  primals_272 = None
    view_162: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_53, [8, 196, 256]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_134: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_162);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_188: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_181, clone_134);  clone_134 = None
    clone_135: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_188, memory_format = torch.contiguous_format)
    var_mean_54 = torch.ops.aten.var_mean.correction(clone_135, [2], correction = 0, keepdim = True)
    getitem_162: "f32[8, 196, 1]" = var_mean_54[0]
    getitem_163: "f32[8, 196, 1]" = var_mean_54[1];  var_mean_54 = None
    add_189: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
    rsqrt_54: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_54: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_135, getitem_163);  clone_135 = None
    mul_216: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    mul_217: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_216, primals_273);  mul_216 = None
    add_190: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_217, primals_274);  mul_217 = primals_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_163: "f32[1568, 256]" = torch.ops.aten.view.default(add_190, [1568, 256]);  add_190 = None
    permute_136: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_275, [1, 0]);  primals_275 = None
    addmm_54: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_276, view_163, permute_136);  primals_276 = None
    view_164: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_54, [8, 196, 1536]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_218: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_164, 0.5)
    mul_219: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_164, 0.7071067811865476)
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
    clone_137: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_165, memory_format = torch.contiguous_format)
    var_mean_55 = torch.ops.aten.var_mean.correction(clone_137, [2], correction = 0, keepdim = True)
    getitem_166: "f32[8, 196, 1]" = var_mean_55[0]
    getitem_167: "f32[8, 196, 1]" = var_mean_55[1];  var_mean_55 = None
    add_192: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05);  getitem_166 = None
    rsqrt_55: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_55: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_137, getitem_167);  clone_137 = None
    mul_221: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    mul_222: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_221, primals_277);  mul_221 = None
    add_193: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_222, primals_278);  mul_222 = primals_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_137: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_193, [0, 2, 1]);  add_193 = None
    permute_138: "f32[196, 196]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
    clone_138: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    view_165: "f32[6144, 196]" = torch.ops.aten.view.default(clone_138, [6144, 196]);  clone_138 = None
    mm_27: "f32[6144, 196]" = torch.ops.aten.mm.default(view_165, permute_138)
    view_166: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_27, [8, 768, 196]);  mm_27 = None
    add_194: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_166, primals_280);  view_166 = primals_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_139: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_194, [0, 2, 1]);  add_194 = None
    mul_223: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_164, permute_139)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_167: "f32[1568, 768]" = torch.ops.aten.view.default(mul_223, [1568, 768]);  mul_223 = None
    permute_140: "f32[768, 256]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
    addmm_55: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_282, view_167, permute_140);  primals_282 = None
    view_168: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_55, [8, 196, 256]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_139: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_168);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_195: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_188, clone_139);  clone_139 = None
    clone_140: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_195, memory_format = torch.contiguous_format)
    var_mean_56 = torch.ops.aten.var_mean.correction(clone_140, [2], correction = 0, keepdim = True)
    getitem_168: "f32[8, 196, 1]" = var_mean_56[0]
    getitem_169: "f32[8, 196, 1]" = var_mean_56[1];  var_mean_56 = None
    add_196: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
    rsqrt_56: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    sub_56: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_140, getitem_169);  clone_140 = None
    mul_224: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    mul_225: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_224, primals_283);  mul_224 = None
    add_197: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_225, primals_284);  mul_225 = primals_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_169: "f32[1568, 256]" = torch.ops.aten.view.default(add_197, [1568, 256]);  add_197 = None
    permute_141: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
    addmm_56: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_286, view_169, permute_141);  primals_286 = None
    view_170: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_56, [8, 196, 1536]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_226: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_170, 0.5)
    mul_227: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_170, 0.7071067811865476)
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
    clone_142: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_171, memory_format = torch.contiguous_format)
    var_mean_57 = torch.ops.aten.var_mean.correction(clone_142, [2], correction = 0, keepdim = True)
    getitem_172: "f32[8, 196, 1]" = var_mean_57[0]
    getitem_173: "f32[8, 196, 1]" = var_mean_57[1];  var_mean_57 = None
    add_199: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05);  getitem_172 = None
    rsqrt_57: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_57: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_142, getitem_173);  clone_142 = None
    mul_229: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    mul_230: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_229, primals_287);  mul_229 = None
    add_200: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_230, primals_288);  mul_230 = primals_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_142: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_200, [0, 2, 1]);  add_200 = None
    permute_143: "f32[196, 196]" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
    clone_143: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    view_171: "f32[6144, 196]" = torch.ops.aten.view.default(clone_143, [6144, 196]);  clone_143 = None
    mm_28: "f32[6144, 196]" = torch.ops.aten.mm.default(view_171, permute_143)
    view_172: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_28, [8, 768, 196]);  mm_28 = None
    add_201: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_172, primals_290);  view_172 = primals_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_144: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_201, [0, 2, 1]);  add_201 = None
    mul_231: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_170, permute_144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_173: "f32[1568, 768]" = torch.ops.aten.view.default(mul_231, [1568, 768]);  mul_231 = None
    permute_145: "f32[768, 256]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
    addmm_57: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_292, view_173, permute_145);  primals_292 = None
    view_174: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_57, [8, 196, 256]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_144: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_174);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_202: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_195, clone_144);  clone_144 = None
    clone_145: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_202, memory_format = torch.contiguous_format)
    var_mean_58 = torch.ops.aten.var_mean.correction(clone_145, [2], correction = 0, keepdim = True)
    getitem_174: "f32[8, 196, 1]" = var_mean_58[0]
    getitem_175: "f32[8, 196, 1]" = var_mean_58[1];  var_mean_58 = None
    add_203: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
    rsqrt_58: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    sub_58: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_145, getitem_175);  clone_145 = None
    mul_232: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    mul_233: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_232, primals_293);  mul_232 = None
    add_204: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_233, primals_294);  mul_233 = primals_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_175: "f32[1568, 256]" = torch.ops.aten.view.default(add_204, [1568, 256]);  add_204 = None
    permute_146: "f32[256, 1536]" = torch.ops.aten.permute.default(primals_295, [1, 0]);  primals_295 = None
    addmm_58: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_296, view_175, permute_146);  primals_296 = None
    view_176: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_58, [8, 196, 1536]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_234: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_176, 0.5)
    mul_235: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476)
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
    clone_147: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_177, memory_format = torch.contiguous_format)
    var_mean_59 = torch.ops.aten.var_mean.correction(clone_147, [2], correction = 0, keepdim = True)
    getitem_178: "f32[8, 196, 1]" = var_mean_59[0]
    getitem_179: "f32[8, 196, 1]" = var_mean_59[1];  var_mean_59 = None
    add_206: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
    rsqrt_59: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
    sub_59: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_147, getitem_179);  clone_147 = None
    mul_237: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    mul_238: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_237, primals_297);  mul_237 = None
    add_207: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_238, primals_298);  mul_238 = primals_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_147: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_207, [0, 2, 1]);  add_207 = None
    permute_148: "f32[196, 196]" = torch.ops.aten.permute.default(primals_299, [1, 0]);  primals_299 = None
    clone_148: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    view_177: "f32[6144, 196]" = torch.ops.aten.view.default(clone_148, [6144, 196]);  clone_148 = None
    mm_29: "f32[6144, 196]" = torch.ops.aten.mm.default(view_177, permute_148)
    view_178: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_29, [8, 768, 196]);  mm_29 = None
    add_208: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_178, primals_300);  view_178 = primals_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_149: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_208, [0, 2, 1]);  add_208 = None
    mul_239: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_176, permute_149)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_179: "f32[1568, 768]" = torch.ops.aten.view.default(mul_239, [1568, 768]);  mul_239 = None
    permute_150: "f32[768, 256]" = torch.ops.aten.permute.default(primals_301, [1, 0]);  primals_301 = None
    addmm_59: "f32[1568, 256]" = torch.ops.aten.addmm.default(primals_302, view_179, permute_150);  primals_302 = None
    view_180: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_59, [8, 196, 256]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_149: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_209: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_202, clone_149);  clone_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    clone_150: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_209, memory_format = torch.contiguous_format)
    var_mean_60 = torch.ops.aten.var_mean.correction(clone_150, [2], correction = 0, keepdim = True)
    getitem_180: "f32[8, 196, 1]" = var_mean_60[0]
    getitem_181: "f32[8, 196, 1]" = var_mean_60[1];  var_mean_60 = None
    add_210: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-06);  getitem_180 = None
    rsqrt_60: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_60: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_150, getitem_181);  clone_150 = None
    mul_240: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    mul_241: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_240, primals_303);  mul_240 = None
    add_211: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_241, primals_304);  mul_241 = primals_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    mean: "f32[8, 256]" = torch.ops.aten.mean.dim(add_211, [1]);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:272, code: x = self.head_drop(x)
    clone_151: "f32[8, 256]" = torch.ops.aten.clone.default(mean);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    permute_151: "f32[256, 1000]" = torch.ops.aten.permute.default(primals_305, [1, 0]);  primals_305 = None
    addmm_60: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_306, clone_151, permute_151);  primals_306 = None
    permute_152: "f32[1000, 256]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    mm_30: "f32[8, 256]" = torch.ops.aten.mm.default(tangents_1, permute_152);  permute_152 = None
    permute_153: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_31: "f32[1000, 256]" = torch.ops.aten.mm.default(permute_153, clone_151);  permute_153 = clone_151 = None
    permute_154: "f32[256, 1000]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_181: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_155: "f32[1000, 256]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    unsqueeze: "f32[8, 1, 256]" = torch.ops.aten.unsqueeze.default(mm_30, 1);  mm_30 = None
    expand: "f32[8, 196, 256]" = torch.ops.aten.expand.default(unsqueeze, [8, 196, 256]);  unsqueeze = None
    div: "f32[8, 196, 256]" = torch.ops.aten.div.Scalar(expand, 196);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    clone_152: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_209, memory_format = torch.contiguous_format);  add_209 = None
    sub_61: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_152, getitem_181);  clone_152 = getitem_181 = None
    mul_242: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_60);  sub_61 = None
    mul_243: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div, primals_303);  primals_303 = None
    mul_244: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_243, 256)
    sum_2: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True)
    mul_245: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_243, mul_242);  mul_243 = None
    sum_3: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True);  mul_245 = None
    mul_246: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_242, sum_3);  sum_3 = None
    sub_62: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_244, sum_2);  mul_244 = sum_2 = None
    sub_63: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_62, mul_246);  sub_62 = mul_246 = None
    div_1: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_60, 256);  rsqrt_60 = None
    mul_247: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_1, sub_63);  div_1 = sub_63 = None
    mul_248: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div, mul_242);  mul_242 = None
    sum_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1]);  mul_248 = None
    sum_5: "f32[256]" = torch.ops.aten.sum.dim_IntList(div, [0, 1]);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_182: "f32[1568, 256]" = torch.ops.aten.view.default(mul_247, [1568, 256])
    permute_156: "f32[256, 768]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    mm_32: "f32[1568, 768]" = torch.ops.aten.mm.default(view_182, permute_156);  permute_156 = None
    permute_157: "f32[256, 1568]" = torch.ops.aten.permute.default(view_182, [1, 0])
    mm_33: "f32[256, 768]" = torch.ops.aten.mm.default(permute_157, view_179);  permute_157 = view_179 = None
    permute_158: "f32[768, 256]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_6: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True);  view_182 = None
    view_183: "f32[256]" = torch.ops.aten.view.default(sum_6, [256]);  sum_6 = None
    permute_159: "f32[256, 768]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    view_184: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_32, [8, 196, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_249: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_184, getitem_176);  getitem_176 = None
    mul_250: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_184, permute_149);  view_184 = permute_149 = None
    permute_160: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_249, [0, 2, 1]);  mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_7: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_160, [0, 1], True)
    view_185: "f32[196]" = torch.ops.aten.view.default(sum_7, [196]);  sum_7 = None
    clone_153: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    view_186: "f32[6144, 196]" = torch.ops.aten.view.default(clone_153, [6144, 196]);  clone_153 = None
    permute_161: "f32[196, 6144]" = torch.ops.aten.permute.default(view_186, [1, 0])
    mm_34: "f32[196, 196]" = torch.ops.aten.mm.default(permute_161, view_177);  permute_161 = view_177 = None
    permute_162: "f32[196, 196]" = torch.ops.aten.permute.default(mm_34, [1, 0]);  mm_34 = None
    permute_163: "f32[196, 196]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    mm_35: "f32[6144, 196]" = torch.ops.aten.mm.default(view_186, permute_163);  view_186 = permute_163 = None
    view_187: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_35, [8, 768, 196]);  mm_35 = None
    permute_164: "f32[196, 196]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    permute_165: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_154: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
    clone_155: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_177, memory_format = torch.contiguous_format);  getitem_177 = None
    sub_64: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_155, getitem_179);  clone_155 = getitem_179 = None
    mul_251: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_59);  sub_64 = None
    mul_252: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_154, primals_297);  primals_297 = None
    mul_253: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_252, 768)
    sum_8: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True)
    mul_254: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_252, mul_251);  mul_252 = None
    sum_9: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True);  mul_254 = None
    mul_255: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_251, sum_9);  sum_9 = None
    sub_65: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_253, sum_8);  mul_253 = sum_8 = None
    sub_66: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_255);  sub_65 = mul_255 = None
    div_2: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_59, 768);  rsqrt_59 = None
    mul_256: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_66);  div_2 = sub_66 = None
    mul_257: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_154, mul_251);  mul_251 = None
    sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 1]);  mul_257 = None
    sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_154, [0, 1]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_250, mul_256], 2);  mul_250 = mul_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_258: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476)
    erf_30: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_258);  mul_258 = None
    add_212: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_259: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_212, 0.5);  add_212 = None
    mul_260: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_176, view_176)
    mul_261: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_260, -0.5);  mul_260 = None
    exp: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_261);  mul_261 = None
    mul_262: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_263: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_176, mul_262);  view_176 = mul_262 = None
    add_213: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_259, mul_263);  mul_259 = mul_263 = None
    mul_264: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat, add_213);  cat = add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_188: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_264, [1568, 1536]);  mul_264 = None
    permute_166: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    mm_36: "f32[1568, 256]" = torch.ops.aten.mm.default(view_188, permute_166);  permute_166 = None
    permute_167: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_188, [1, 0])
    mm_37: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_167, view_175);  permute_167 = view_175 = None
    permute_168: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_12: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
    view_189: "f32[1536]" = torch.ops.aten.view.default(sum_12, [1536]);  sum_12 = None
    permute_169: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    view_190: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_36, [8, 196, 256]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_156: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_202, memory_format = torch.contiguous_format);  add_202 = None
    sub_67: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_156, getitem_175);  clone_156 = getitem_175 = None
    mul_265: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_58);  sub_67 = None
    mul_266: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_190, primals_293);  primals_293 = None
    mul_267: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_266, 256)
    sum_13: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True)
    mul_268: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_266, mul_265);  mul_266 = None
    sum_14: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True);  mul_268 = None
    mul_269: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_265, sum_14);  sum_14 = None
    sub_68: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_267, sum_13);  mul_267 = sum_13 = None
    sub_69: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_68, mul_269);  sub_68 = mul_269 = None
    div_3: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_58, 256);  rsqrt_58 = None
    mul_270: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_3, sub_69);  div_3 = sub_69 = None
    mul_271: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_190, mul_265);  mul_265 = None
    sum_15: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 1]);  mul_271 = None
    sum_16: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_190, [0, 1]);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_214: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_247, mul_270);  mul_247 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_191: "f32[1568, 256]" = torch.ops.aten.view.default(add_214, [1568, 256])
    permute_170: "f32[256, 768]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    mm_38: "f32[1568, 768]" = torch.ops.aten.mm.default(view_191, permute_170);  permute_170 = None
    permute_171: "f32[256, 1568]" = torch.ops.aten.permute.default(view_191, [1, 0])
    mm_39: "f32[256, 768]" = torch.ops.aten.mm.default(permute_171, view_173);  permute_171 = view_173 = None
    permute_172: "f32[768, 256]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_17: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_191, [0], True);  view_191 = None
    view_192: "f32[256]" = torch.ops.aten.view.default(sum_17, [256]);  sum_17 = None
    permute_173: "f32[256, 768]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    view_193: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_38, [8, 196, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_272: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_193, getitem_170);  getitem_170 = None
    mul_273: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_193, permute_144);  view_193 = permute_144 = None
    permute_174: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_272, [0, 2, 1]);  mul_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_18: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_174, [0, 1], True)
    view_194: "f32[196]" = torch.ops.aten.view.default(sum_18, [196]);  sum_18 = None
    clone_157: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
    view_195: "f32[6144, 196]" = torch.ops.aten.view.default(clone_157, [6144, 196]);  clone_157 = None
    permute_175: "f32[196, 6144]" = torch.ops.aten.permute.default(view_195, [1, 0])
    mm_40: "f32[196, 196]" = torch.ops.aten.mm.default(permute_175, view_171);  permute_175 = view_171 = None
    permute_176: "f32[196, 196]" = torch.ops.aten.permute.default(mm_40, [1, 0]);  mm_40 = None
    permute_177: "f32[196, 196]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    mm_41: "f32[6144, 196]" = torch.ops.aten.mm.default(view_195, permute_177);  view_195 = permute_177 = None
    view_196: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_41, [8, 768, 196]);  mm_41 = None
    permute_178: "f32[196, 196]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    permute_179: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_196, [0, 2, 1]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_158: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    clone_159: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_171, memory_format = torch.contiguous_format);  getitem_171 = None
    sub_70: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_159, getitem_173);  clone_159 = getitem_173 = None
    mul_274: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_57);  sub_70 = None
    mul_275: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_158, primals_287);  primals_287 = None
    mul_276: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_275, 768)
    sum_19: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [2], True)
    mul_277: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_275, mul_274);  mul_275 = None
    sum_20: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True);  mul_277 = None
    mul_278: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_274, sum_20);  sum_20 = None
    sub_71: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_276, sum_19);  mul_276 = sum_19 = None
    sub_72: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_278);  sub_71 = mul_278 = None
    div_4: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_57, 768);  rsqrt_57 = None
    mul_279: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_72);  div_4 = sub_72 = None
    mul_280: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_158, mul_274);  mul_274 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 1]);  mul_280 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_158, [0, 1]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_1: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_273, mul_279], 2);  mul_273 = mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_281: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_170, 0.7071067811865476)
    erf_31: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_281);  mul_281 = None
    add_215: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_282: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_215, 0.5);  add_215 = None
    mul_283: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_170, view_170)
    mul_284: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_283, -0.5);  mul_283 = None
    exp_1: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_284);  mul_284 = None
    mul_285: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_286: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_170, mul_285);  view_170 = mul_285 = None
    add_216: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_282, mul_286);  mul_282 = mul_286 = None
    mul_287: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_1, add_216);  cat_1 = add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_197: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_287, [1568, 1536]);  mul_287 = None
    permute_180: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    mm_42: "f32[1568, 256]" = torch.ops.aten.mm.default(view_197, permute_180);  permute_180 = None
    permute_181: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_197, [1, 0])
    mm_43: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_181, view_169);  permute_181 = view_169 = None
    permute_182: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_23: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_197, [0], True);  view_197 = None
    view_198: "f32[1536]" = torch.ops.aten.view.default(sum_23, [1536]);  sum_23 = None
    permute_183: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    view_199: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_42, [8, 196, 256]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_160: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_195, memory_format = torch.contiguous_format);  add_195 = None
    sub_73: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_160, getitem_169);  clone_160 = getitem_169 = None
    mul_288: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_56);  sub_73 = None
    mul_289: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_199, primals_283);  primals_283 = None
    mul_290: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_289, 256)
    sum_24: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_289, [2], True)
    mul_291: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_289, mul_288);  mul_289 = None
    sum_25: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True);  mul_291 = None
    mul_292: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_288, sum_25);  sum_25 = None
    sub_74: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_290, sum_24);  mul_290 = sum_24 = None
    sub_75: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_74, mul_292);  sub_74 = mul_292 = None
    div_5: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_56, 256);  rsqrt_56 = None
    mul_293: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_5, sub_75);  div_5 = sub_75 = None
    mul_294: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_199, mul_288);  mul_288 = None
    sum_26: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_294, [0, 1]);  mul_294 = None
    sum_27: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_199, [0, 1]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_217: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_214, mul_293);  add_214 = mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_200: "f32[1568, 256]" = torch.ops.aten.view.default(add_217, [1568, 256])
    permute_184: "f32[256, 768]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    mm_44: "f32[1568, 768]" = torch.ops.aten.mm.default(view_200, permute_184);  permute_184 = None
    permute_185: "f32[256, 1568]" = torch.ops.aten.permute.default(view_200, [1, 0])
    mm_45: "f32[256, 768]" = torch.ops.aten.mm.default(permute_185, view_167);  permute_185 = view_167 = None
    permute_186: "f32[768, 256]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_28: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_200, [0], True);  view_200 = None
    view_201: "f32[256]" = torch.ops.aten.view.default(sum_28, [256]);  sum_28 = None
    permute_187: "f32[256, 768]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    view_202: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_44, [8, 196, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_295: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_202, getitem_164);  getitem_164 = None
    mul_296: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_202, permute_139);  view_202 = permute_139 = None
    permute_188: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_295, [0, 2, 1]);  mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_29: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_188, [0, 1], True)
    view_203: "f32[196]" = torch.ops.aten.view.default(sum_29, [196]);  sum_29 = None
    clone_161: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_204: "f32[6144, 196]" = torch.ops.aten.view.default(clone_161, [6144, 196]);  clone_161 = None
    permute_189: "f32[196, 6144]" = torch.ops.aten.permute.default(view_204, [1, 0])
    mm_46: "f32[196, 196]" = torch.ops.aten.mm.default(permute_189, view_165);  permute_189 = view_165 = None
    permute_190: "f32[196, 196]" = torch.ops.aten.permute.default(mm_46, [1, 0]);  mm_46 = None
    permute_191: "f32[196, 196]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    mm_47: "f32[6144, 196]" = torch.ops.aten.mm.default(view_204, permute_191);  view_204 = permute_191 = None
    view_205: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_47, [8, 768, 196]);  mm_47 = None
    permute_192: "f32[196, 196]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    permute_193: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_162: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    clone_163: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_165, memory_format = torch.contiguous_format);  getitem_165 = None
    sub_76: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_163, getitem_167);  clone_163 = getitem_167 = None
    mul_297: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_55);  sub_76 = None
    mul_298: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_162, primals_277);  primals_277 = None
    mul_299: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_298, 768)
    sum_30: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True)
    mul_300: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_298, mul_297);  mul_298 = None
    sum_31: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True);  mul_300 = None
    mul_301: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_297, sum_31);  sum_31 = None
    sub_77: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_299, sum_30);  mul_299 = sum_30 = None
    sub_78: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_301);  sub_77 = mul_301 = None
    div_6: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_55, 768);  rsqrt_55 = None
    mul_302: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_78);  div_6 = sub_78 = None
    mul_303: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_162, mul_297);  mul_297 = None
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1]);  mul_303 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_162, [0, 1]);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_2: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_296, mul_302], 2);  mul_296 = mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_304: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_164, 0.7071067811865476)
    erf_32: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_304);  mul_304 = None
    add_218: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_305: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_218, 0.5);  add_218 = None
    mul_306: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_164, view_164)
    mul_307: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_306, -0.5);  mul_306 = None
    exp_2: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_307);  mul_307 = None
    mul_308: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_309: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_164, mul_308);  view_164 = mul_308 = None
    add_219: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_305, mul_309);  mul_305 = mul_309 = None
    mul_310: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_2, add_219);  cat_2 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_206: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_310, [1568, 1536]);  mul_310 = None
    permute_194: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    mm_48: "f32[1568, 256]" = torch.ops.aten.mm.default(view_206, permute_194);  permute_194 = None
    permute_195: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_206, [1, 0])
    mm_49: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_195, view_163);  permute_195 = view_163 = None
    permute_196: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_34: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_206, [0], True);  view_206 = None
    view_207: "f32[1536]" = torch.ops.aten.view.default(sum_34, [1536]);  sum_34 = None
    permute_197: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    view_208: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_48, [8, 196, 256]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_164: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_188, memory_format = torch.contiguous_format);  add_188 = None
    sub_79: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_164, getitem_163);  clone_164 = getitem_163 = None
    mul_311: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_54);  sub_79 = None
    mul_312: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_208, primals_273);  primals_273 = None
    mul_313: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_312, 256)
    sum_35: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True)
    mul_314: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_312, mul_311);  mul_312 = None
    sum_36: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True);  mul_314 = None
    mul_315: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_311, sum_36);  sum_36 = None
    sub_80: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_313, sum_35);  mul_313 = sum_35 = None
    sub_81: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_80, mul_315);  sub_80 = mul_315 = None
    div_7: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_54, 256);  rsqrt_54 = None
    mul_316: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_7, sub_81);  div_7 = sub_81 = None
    mul_317: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_208, mul_311);  mul_311 = None
    sum_37: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1]);  mul_317 = None
    sum_38: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_208, [0, 1]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_220: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_217, mul_316);  add_217 = mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_209: "f32[1568, 256]" = torch.ops.aten.view.default(add_220, [1568, 256])
    permute_198: "f32[256, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    mm_50: "f32[1568, 768]" = torch.ops.aten.mm.default(view_209, permute_198);  permute_198 = None
    permute_199: "f32[256, 1568]" = torch.ops.aten.permute.default(view_209, [1, 0])
    mm_51: "f32[256, 768]" = torch.ops.aten.mm.default(permute_199, view_161);  permute_199 = view_161 = None
    permute_200: "f32[768, 256]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_39: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_209, [0], True);  view_209 = None
    view_210: "f32[256]" = torch.ops.aten.view.default(sum_39, [256]);  sum_39 = None
    permute_201: "f32[256, 768]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    view_211: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_50, [8, 196, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_318: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_211, getitem_158);  getitem_158 = None
    mul_319: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_211, permute_134);  view_211 = permute_134 = None
    permute_202: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_318, [0, 2, 1]);  mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_40: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_202, [0, 1], True)
    view_212: "f32[196]" = torch.ops.aten.view.default(sum_40, [196]);  sum_40 = None
    clone_165: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
    view_213: "f32[6144, 196]" = torch.ops.aten.view.default(clone_165, [6144, 196]);  clone_165 = None
    permute_203: "f32[196, 6144]" = torch.ops.aten.permute.default(view_213, [1, 0])
    mm_52: "f32[196, 196]" = torch.ops.aten.mm.default(permute_203, view_159);  permute_203 = view_159 = None
    permute_204: "f32[196, 196]" = torch.ops.aten.permute.default(mm_52, [1, 0]);  mm_52 = None
    permute_205: "f32[196, 196]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    mm_53: "f32[6144, 196]" = torch.ops.aten.mm.default(view_213, permute_205);  view_213 = permute_205 = None
    view_214: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_53, [8, 768, 196]);  mm_53 = None
    permute_206: "f32[196, 196]" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
    permute_207: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_166: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
    clone_167: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_159, memory_format = torch.contiguous_format);  getitem_159 = None
    sub_82: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_167, getitem_161);  clone_167 = getitem_161 = None
    mul_320: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_53);  sub_82 = None
    mul_321: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_166, primals_267);  primals_267 = None
    mul_322: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_321, 768)
    sum_41: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True)
    mul_323: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_321, mul_320);  mul_321 = None
    sum_42: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True);  mul_323 = None
    mul_324: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_320, sum_42);  sum_42 = None
    sub_83: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_322, sum_41);  mul_322 = sum_41 = None
    sub_84: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_324);  sub_83 = mul_324 = None
    div_8: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_53, 768);  rsqrt_53 = None
    mul_325: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_84);  div_8 = sub_84 = None
    mul_326: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_166, mul_320);  mul_320 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 1]);  mul_326 = None
    sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_166, [0, 1]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_3: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_319, mul_325], 2);  mul_319 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_327: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476)
    erf_33: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_327);  mul_327 = None
    add_221: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_328: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_221, 0.5);  add_221 = None
    mul_329: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_158, view_158)
    mul_330: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_329, -0.5);  mul_329 = None
    exp_3: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_330);  mul_330 = None
    mul_331: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_332: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_158, mul_331);  view_158 = mul_331 = None
    add_222: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_328, mul_332);  mul_328 = mul_332 = None
    mul_333: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_3, add_222);  cat_3 = add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_215: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_333, [1568, 1536]);  mul_333 = None
    permute_208: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_54: "f32[1568, 256]" = torch.ops.aten.mm.default(view_215, permute_208);  permute_208 = None
    permute_209: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_215, [1, 0])
    mm_55: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_209, view_157);  permute_209 = view_157 = None
    permute_210: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_45: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
    view_216: "f32[1536]" = torch.ops.aten.view.default(sum_45, [1536]);  sum_45 = None
    permute_211: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_217: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_54, [8, 196, 256]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_168: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_181, memory_format = torch.contiguous_format);  add_181 = None
    sub_85: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_168, getitem_157);  clone_168 = getitem_157 = None
    mul_334: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_52);  sub_85 = None
    mul_335: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_217, primals_263);  primals_263 = None
    mul_336: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_335, 256)
    sum_46: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True)
    mul_337: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_335, mul_334);  mul_335 = None
    sum_47: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True);  mul_337 = None
    mul_338: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_334, sum_47);  sum_47 = None
    sub_86: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_336, sum_46);  mul_336 = sum_46 = None
    sub_87: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_86, mul_338);  sub_86 = mul_338 = None
    div_9: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_52, 256);  rsqrt_52 = None
    mul_339: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_9, sub_87);  div_9 = sub_87 = None
    mul_340: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_217, mul_334);  mul_334 = None
    sum_48: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 1]);  mul_340 = None
    sum_49: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 1]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_223: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_220, mul_339);  add_220 = mul_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_218: "f32[1568, 256]" = torch.ops.aten.view.default(add_223, [1568, 256])
    permute_212: "f32[256, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_56: "f32[1568, 768]" = torch.ops.aten.mm.default(view_218, permute_212);  permute_212 = None
    permute_213: "f32[256, 1568]" = torch.ops.aten.permute.default(view_218, [1, 0])
    mm_57: "f32[256, 768]" = torch.ops.aten.mm.default(permute_213, view_155);  permute_213 = view_155 = None
    permute_214: "f32[768, 256]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_50: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
    view_219: "f32[256]" = torch.ops.aten.view.default(sum_50, [256]);  sum_50 = None
    permute_215: "f32[256, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_220: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_56, [8, 196, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_341: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_220, getitem_152);  getitem_152 = None
    mul_342: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_220, permute_129);  view_220 = permute_129 = None
    permute_216: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_341, [0, 2, 1]);  mul_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_51: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_216, [0, 1], True)
    view_221: "f32[196]" = torch.ops.aten.view.default(sum_51, [196]);  sum_51 = None
    clone_169: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    view_222: "f32[6144, 196]" = torch.ops.aten.view.default(clone_169, [6144, 196]);  clone_169 = None
    permute_217: "f32[196, 6144]" = torch.ops.aten.permute.default(view_222, [1, 0])
    mm_58: "f32[196, 196]" = torch.ops.aten.mm.default(permute_217, view_153);  permute_217 = view_153 = None
    permute_218: "f32[196, 196]" = torch.ops.aten.permute.default(mm_58, [1, 0]);  mm_58 = None
    permute_219: "f32[196, 196]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    mm_59: "f32[6144, 196]" = torch.ops.aten.mm.default(view_222, permute_219);  view_222 = permute_219 = None
    view_223: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_59, [8, 768, 196]);  mm_59 = None
    permute_220: "f32[196, 196]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    permute_221: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_223, [0, 2, 1]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_170: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
    clone_171: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_153, memory_format = torch.contiguous_format);  getitem_153 = None
    sub_88: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_171, getitem_155);  clone_171 = getitem_155 = None
    mul_343: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_51);  sub_88 = None
    mul_344: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_170, primals_257);  primals_257 = None
    mul_345: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_344, 768)
    sum_52: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True)
    mul_346: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_344, mul_343);  mul_344 = None
    sum_53: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [2], True);  mul_346 = None
    mul_347: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_343, sum_53);  sum_53 = None
    sub_89: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_345, sum_52);  mul_345 = sum_52 = None
    sub_90: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_89, mul_347);  sub_89 = mul_347 = None
    div_10: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_51, 768);  rsqrt_51 = None
    mul_348: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_90);  div_10 = sub_90 = None
    mul_349: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_170, mul_343);  mul_343 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_349, [0, 1]);  mul_349 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_170, [0, 1]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_4: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_342, mul_348], 2);  mul_342 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_350: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_152, 0.7071067811865476)
    erf_34: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_350);  mul_350 = None
    add_224: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_351: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_224, 0.5);  add_224 = None
    mul_352: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_152, view_152)
    mul_353: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_352, -0.5);  mul_352 = None
    exp_4: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_353);  mul_353 = None
    mul_354: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_355: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_152, mul_354);  view_152 = mul_354 = None
    add_225: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_351, mul_355);  mul_351 = mul_355 = None
    mul_356: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_4, add_225);  cat_4 = add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_224: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_356, [1568, 1536]);  mul_356 = None
    permute_222: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    mm_60: "f32[1568, 256]" = torch.ops.aten.mm.default(view_224, permute_222);  permute_222 = None
    permute_223: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_224, [1, 0])
    mm_61: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_223, view_151);  permute_223 = view_151 = None
    permute_224: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_56: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
    view_225: "f32[1536]" = torch.ops.aten.view.default(sum_56, [1536]);  sum_56 = None
    permute_225: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    view_226: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_60, [8, 196, 256]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_172: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_174, memory_format = torch.contiguous_format);  add_174 = None
    sub_91: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_172, getitem_151);  clone_172 = getitem_151 = None
    mul_357: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_50);  sub_91 = None
    mul_358: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_226, primals_253);  primals_253 = None
    mul_359: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_358, 256)
    sum_57: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True)
    mul_360: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_358, mul_357);  mul_358 = None
    sum_58: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_360, [2], True);  mul_360 = None
    mul_361: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_357, sum_58);  sum_58 = None
    sub_92: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_359, sum_57);  mul_359 = sum_57 = None
    sub_93: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_92, mul_361);  sub_92 = mul_361 = None
    div_11: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 256);  rsqrt_50 = None
    mul_362: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_11, sub_93);  div_11 = sub_93 = None
    mul_363: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_226, mul_357);  mul_357 = None
    sum_59: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 1]);  mul_363 = None
    sum_60: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_226, [0, 1]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_226: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_223, mul_362);  add_223 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_227: "f32[1568, 256]" = torch.ops.aten.view.default(add_226, [1568, 256])
    permute_226: "f32[256, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    mm_62: "f32[1568, 768]" = torch.ops.aten.mm.default(view_227, permute_226);  permute_226 = None
    permute_227: "f32[256, 1568]" = torch.ops.aten.permute.default(view_227, [1, 0])
    mm_63: "f32[256, 768]" = torch.ops.aten.mm.default(permute_227, view_149);  permute_227 = view_149 = None
    permute_228: "f32[768, 256]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_61: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_227, [0], True);  view_227 = None
    view_228: "f32[256]" = torch.ops.aten.view.default(sum_61, [256]);  sum_61 = None
    permute_229: "f32[256, 768]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    view_229: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_62, [8, 196, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_364: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_229, getitem_146);  getitem_146 = None
    mul_365: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_229, permute_124);  view_229 = permute_124 = None
    permute_230: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_364, [0, 2, 1]);  mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_62: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_230, [0, 1], True)
    view_230: "f32[196]" = torch.ops.aten.view.default(sum_62, [196]);  sum_62 = None
    clone_173: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
    view_231: "f32[6144, 196]" = torch.ops.aten.view.default(clone_173, [6144, 196]);  clone_173 = None
    permute_231: "f32[196, 6144]" = torch.ops.aten.permute.default(view_231, [1, 0])
    mm_64: "f32[196, 196]" = torch.ops.aten.mm.default(permute_231, view_147);  permute_231 = view_147 = None
    permute_232: "f32[196, 196]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    permute_233: "f32[196, 196]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    mm_65: "f32[6144, 196]" = torch.ops.aten.mm.default(view_231, permute_233);  view_231 = permute_233 = None
    view_232: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_65, [8, 768, 196]);  mm_65 = None
    permute_234: "f32[196, 196]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    permute_235: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_174: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
    clone_175: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_147, memory_format = torch.contiguous_format);  getitem_147 = None
    sub_94: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_175, getitem_149);  clone_175 = getitem_149 = None
    mul_366: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_49);  sub_94 = None
    mul_367: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_174, primals_247);  primals_247 = None
    mul_368: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_367, 768)
    sum_63: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
    mul_369: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_367, mul_366);  mul_367 = None
    sum_64: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
    mul_370: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_366, sum_64);  sum_64 = None
    sub_95: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_368, sum_63);  mul_368 = sum_63 = None
    sub_96: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_370);  sub_95 = mul_370 = None
    div_12: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 768);  rsqrt_49 = None
    mul_371: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_96);  div_12 = sub_96 = None
    mul_372: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_174, mul_366);  mul_366 = None
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_174, [0, 1]);  clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_5: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_365, mul_371], 2);  mul_365 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_373: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476)
    erf_35: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_373);  mul_373 = None
    add_227: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_374: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_227, 0.5);  add_227 = None
    mul_375: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_146, view_146)
    mul_376: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_375, -0.5);  mul_375 = None
    exp_5: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_376);  mul_376 = None
    mul_377: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_378: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_146, mul_377);  view_146 = mul_377 = None
    add_228: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_374, mul_378);  mul_374 = mul_378 = None
    mul_379: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_5, add_228);  cat_5 = add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_233: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_379, [1568, 1536]);  mul_379 = None
    permute_236: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_66: "f32[1568, 256]" = torch.ops.aten.mm.default(view_233, permute_236);  permute_236 = None
    permute_237: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_233, [1, 0])
    mm_67: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_237, view_145);  permute_237 = view_145 = None
    permute_238: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_67: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True);  view_233 = None
    view_234: "f32[1536]" = torch.ops.aten.view.default(sum_67, [1536]);  sum_67 = None
    permute_239: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    view_235: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_66, [8, 196, 256]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_176: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format);  add_167 = None
    sub_97: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_176, getitem_145);  clone_176 = getitem_145 = None
    mul_380: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_48);  sub_97 = None
    mul_381: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_235, primals_243);  primals_243 = None
    mul_382: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_381, 256)
    sum_68: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True)
    mul_383: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_381, mul_380);  mul_381 = None
    sum_69: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
    mul_384: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_380, sum_69);  sum_69 = None
    sub_98: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_382, sum_68);  mul_382 = sum_68 = None
    sub_99: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_98, mul_384);  sub_98 = mul_384 = None
    div_13: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 256);  rsqrt_48 = None
    mul_385: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_13, sub_99);  div_13 = sub_99 = None
    mul_386: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_235, mul_380);  mul_380 = None
    sum_70: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1]);  mul_386 = None
    sum_71: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_235, [0, 1]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_229: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_226, mul_385);  add_226 = mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_236: "f32[1568, 256]" = torch.ops.aten.view.default(add_229, [1568, 256])
    permute_240: "f32[256, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    mm_68: "f32[1568, 768]" = torch.ops.aten.mm.default(view_236, permute_240);  permute_240 = None
    permute_241: "f32[256, 1568]" = torch.ops.aten.permute.default(view_236, [1, 0])
    mm_69: "f32[256, 768]" = torch.ops.aten.mm.default(permute_241, view_143);  permute_241 = view_143 = None
    permute_242: "f32[768, 256]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_72: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_236, [0], True);  view_236 = None
    view_237: "f32[256]" = torch.ops.aten.view.default(sum_72, [256]);  sum_72 = None
    permute_243: "f32[256, 768]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    view_238: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_68, [8, 196, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_387: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_238, getitem_140);  getitem_140 = None
    mul_388: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_238, permute_119);  view_238 = permute_119 = None
    permute_244: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_387, [0, 2, 1]);  mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_73: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_244, [0, 1], True)
    view_239: "f32[196]" = torch.ops.aten.view.default(sum_73, [196]);  sum_73 = None
    clone_177: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
    view_240: "f32[6144, 196]" = torch.ops.aten.view.default(clone_177, [6144, 196]);  clone_177 = None
    permute_245: "f32[196, 6144]" = torch.ops.aten.permute.default(view_240, [1, 0])
    mm_70: "f32[196, 196]" = torch.ops.aten.mm.default(permute_245, view_141);  permute_245 = view_141 = None
    permute_246: "f32[196, 196]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    permute_247: "f32[196, 196]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    mm_71: "f32[6144, 196]" = torch.ops.aten.mm.default(view_240, permute_247);  view_240 = permute_247 = None
    view_241: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_71, [8, 768, 196]);  mm_71 = None
    permute_248: "f32[196, 196]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    permute_249: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_241, [0, 2, 1]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_178: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    clone_179: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_141, memory_format = torch.contiguous_format);  getitem_141 = None
    sub_100: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_179, getitem_143);  clone_179 = getitem_143 = None
    mul_389: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_47);  sub_100 = None
    mul_390: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_178, primals_237);  primals_237 = None
    mul_391: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_390, 768)
    sum_74: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True)
    mul_392: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_390, mul_389);  mul_390 = None
    sum_75: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [2], True);  mul_392 = None
    mul_393: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_389, sum_75);  sum_75 = None
    sub_101: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_391, sum_74);  mul_391 = sum_74 = None
    sub_102: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_393);  sub_101 = mul_393 = None
    div_14: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 768);  rsqrt_47 = None
    mul_394: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_102);  div_14 = sub_102 = None
    mul_395: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_178, mul_389);  mul_389 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 1]);  mul_395 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_178, [0, 1]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_6: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_388, mul_394], 2);  mul_388 = mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_396: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_140, 0.7071067811865476)
    erf_36: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_396);  mul_396 = None
    add_230: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_397: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_230, 0.5);  add_230 = None
    mul_398: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_140, view_140)
    mul_399: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_398, -0.5);  mul_398 = None
    exp_6: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_399);  mul_399 = None
    mul_400: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_401: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_140, mul_400);  view_140 = mul_400 = None
    add_231: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_397, mul_401);  mul_397 = mul_401 = None
    mul_402: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_6, add_231);  cat_6 = add_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_242: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_402, [1568, 1536]);  mul_402 = None
    permute_250: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    mm_72: "f32[1568, 256]" = torch.ops.aten.mm.default(view_242, permute_250);  permute_250 = None
    permute_251: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_242, [1, 0])
    mm_73: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_251, view_139);  permute_251 = view_139 = None
    permute_252: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_78: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_242, [0], True);  view_242 = None
    view_243: "f32[1536]" = torch.ops.aten.view.default(sum_78, [1536]);  sum_78 = None
    permute_253: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    view_244: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_72, [8, 196, 256]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_180: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_160, memory_format = torch.contiguous_format);  add_160 = None
    sub_103: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_180, getitem_139);  clone_180 = getitem_139 = None
    mul_403: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_46);  sub_103 = None
    mul_404: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_244, primals_233);  primals_233 = None
    mul_405: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_404, 256)
    sum_79: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True)
    mul_406: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_404, mul_403);  mul_404 = None
    sum_80: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [2], True);  mul_406 = None
    mul_407: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_403, sum_80);  sum_80 = None
    sub_104: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_405, sum_79);  mul_405 = sum_79 = None
    sub_105: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_104, mul_407);  sub_104 = mul_407 = None
    div_15: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 256);  rsqrt_46 = None
    mul_408: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_15, sub_105);  div_15 = sub_105 = None
    mul_409: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_244, mul_403);  mul_403 = None
    sum_81: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 1]);  mul_409 = None
    sum_82: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_244, [0, 1]);  view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_232: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_229, mul_408);  add_229 = mul_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_245: "f32[1568, 256]" = torch.ops.aten.view.default(add_232, [1568, 256])
    permute_254: "f32[256, 768]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    mm_74: "f32[1568, 768]" = torch.ops.aten.mm.default(view_245, permute_254);  permute_254 = None
    permute_255: "f32[256, 1568]" = torch.ops.aten.permute.default(view_245, [1, 0])
    mm_75: "f32[256, 768]" = torch.ops.aten.mm.default(permute_255, view_137);  permute_255 = view_137 = None
    permute_256: "f32[768, 256]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_83: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_245, [0], True);  view_245 = None
    view_246: "f32[256]" = torch.ops.aten.view.default(sum_83, [256]);  sum_83 = None
    permute_257: "f32[256, 768]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    view_247: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_74, [8, 196, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_410: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_247, getitem_134);  getitem_134 = None
    mul_411: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_247, permute_114);  view_247 = permute_114 = None
    permute_258: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_410, [0, 2, 1]);  mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_84: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_258, [0, 1], True)
    view_248: "f32[196]" = torch.ops.aten.view.default(sum_84, [196]);  sum_84 = None
    clone_181: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
    view_249: "f32[6144, 196]" = torch.ops.aten.view.default(clone_181, [6144, 196]);  clone_181 = None
    permute_259: "f32[196, 6144]" = torch.ops.aten.permute.default(view_249, [1, 0])
    mm_76: "f32[196, 196]" = torch.ops.aten.mm.default(permute_259, view_135);  permute_259 = view_135 = None
    permute_260: "f32[196, 196]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    permute_261: "f32[196, 196]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    mm_77: "f32[6144, 196]" = torch.ops.aten.mm.default(view_249, permute_261);  view_249 = permute_261 = None
    view_250: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_77, [8, 768, 196]);  mm_77 = None
    permute_262: "f32[196, 196]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    permute_263: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_250, [0, 2, 1]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_182: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
    clone_183: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_135, memory_format = torch.contiguous_format);  getitem_135 = None
    sub_106: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_183, getitem_137);  clone_183 = getitem_137 = None
    mul_412: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_45);  sub_106 = None
    mul_413: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_182, primals_227);  primals_227 = None
    mul_414: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_413, 768)
    sum_85: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True)
    mul_415: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_413, mul_412);  mul_413 = None
    sum_86: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_415, [2], True);  mul_415 = None
    mul_416: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_412, sum_86);  sum_86 = None
    sub_107: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_414, sum_85);  mul_414 = sum_85 = None
    sub_108: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_107, mul_416);  sub_107 = mul_416 = None
    div_16: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 768);  rsqrt_45 = None
    mul_417: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_108);  div_16 = sub_108 = None
    mul_418: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_182, mul_412);  mul_412 = None
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 1]);  mul_418 = None
    sum_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_182, [0, 1]);  clone_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_7: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_411, mul_417], 2);  mul_411 = mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_419: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_134, 0.7071067811865476)
    erf_37: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_419);  mul_419 = None
    add_233: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_420: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_233, 0.5);  add_233 = None
    mul_421: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_134, view_134)
    mul_422: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_421, -0.5);  mul_421 = None
    exp_7: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_422);  mul_422 = None
    mul_423: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_424: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_134, mul_423);  view_134 = mul_423 = None
    add_234: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_420, mul_424);  mul_420 = mul_424 = None
    mul_425: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_7, add_234);  cat_7 = add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_251: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_425, [1568, 1536]);  mul_425 = None
    permute_264: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_78: "f32[1568, 256]" = torch.ops.aten.mm.default(view_251, permute_264);  permute_264 = None
    permute_265: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_251, [1, 0])
    mm_79: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_265, view_133);  permute_265 = view_133 = None
    permute_266: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_89: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_251, [0], True);  view_251 = None
    view_252: "f32[1536]" = torch.ops.aten.view.default(sum_89, [1536]);  sum_89 = None
    permute_267: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    view_253: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_78, [8, 196, 256]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_184: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_153, memory_format = torch.contiguous_format);  add_153 = None
    sub_109: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_184, getitem_133);  clone_184 = getitem_133 = None
    mul_426: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_44);  sub_109 = None
    mul_427: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_253, primals_223);  primals_223 = None
    mul_428: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_427, 256)
    sum_90: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
    mul_429: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_427, mul_426);  mul_427 = None
    sum_91: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
    mul_430: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_426, sum_91);  sum_91 = None
    sub_110: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_428, sum_90);  mul_428 = sum_90 = None
    sub_111: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_110, mul_430);  sub_110 = mul_430 = None
    div_17: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 256);  rsqrt_44 = None
    mul_431: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_17, sub_111);  div_17 = sub_111 = None
    mul_432: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_253, mul_426);  mul_426 = None
    sum_92: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
    sum_93: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_253, [0, 1]);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_235: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_232, mul_431);  add_232 = mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_254: "f32[1568, 256]" = torch.ops.aten.view.default(add_235, [1568, 256])
    permute_268: "f32[256, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_80: "f32[1568, 768]" = torch.ops.aten.mm.default(view_254, permute_268);  permute_268 = None
    permute_269: "f32[256, 1568]" = torch.ops.aten.permute.default(view_254, [1, 0])
    mm_81: "f32[256, 768]" = torch.ops.aten.mm.default(permute_269, view_131);  permute_269 = view_131 = None
    permute_270: "f32[768, 256]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_94: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
    view_255: "f32[256]" = torch.ops.aten.view.default(sum_94, [256]);  sum_94 = None
    permute_271: "f32[256, 768]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
    view_256: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_80, [8, 196, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_433: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_256, getitem_128);  getitem_128 = None
    mul_434: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_256, permute_109);  view_256 = permute_109 = None
    permute_272: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_433, [0, 2, 1]);  mul_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_95: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_272, [0, 1], True)
    view_257: "f32[196]" = torch.ops.aten.view.default(sum_95, [196]);  sum_95 = None
    clone_185: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_272, memory_format = torch.contiguous_format);  permute_272 = None
    view_258: "f32[6144, 196]" = torch.ops.aten.view.default(clone_185, [6144, 196]);  clone_185 = None
    permute_273: "f32[196, 6144]" = torch.ops.aten.permute.default(view_258, [1, 0])
    mm_82: "f32[196, 196]" = torch.ops.aten.mm.default(permute_273, view_129);  permute_273 = view_129 = None
    permute_274: "f32[196, 196]" = torch.ops.aten.permute.default(mm_82, [1, 0]);  mm_82 = None
    permute_275: "f32[196, 196]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_83: "f32[6144, 196]" = torch.ops.aten.mm.default(view_258, permute_275);  view_258 = permute_275 = None
    view_259: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_83, [8, 768, 196]);  mm_83 = None
    permute_276: "f32[196, 196]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    permute_277: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_259, [0, 2, 1]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_186: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_277, memory_format = torch.contiguous_format);  permute_277 = None
    clone_187: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_129, memory_format = torch.contiguous_format);  getitem_129 = None
    sub_112: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_187, getitem_131);  clone_187 = getitem_131 = None
    mul_435: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_43);  sub_112 = None
    mul_436: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_186, primals_217);  primals_217 = None
    mul_437: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_436, 768)
    sum_96: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_436, [2], True)
    mul_438: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_436, mul_435);  mul_436 = None
    sum_97: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_438, [2], True);  mul_438 = None
    mul_439: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_435, sum_97);  sum_97 = None
    sub_113: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_437, sum_96);  mul_437 = sum_96 = None
    sub_114: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_113, mul_439);  sub_113 = mul_439 = None
    div_18: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 768);  rsqrt_43 = None
    mul_440: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_114);  div_18 = sub_114 = None
    mul_441: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_186, mul_435);  mul_435 = None
    sum_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_441, [0, 1]);  mul_441 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_186, [0, 1]);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_8: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_434, mul_440], 2);  mul_434 = mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_442: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476)
    erf_38: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_442);  mul_442 = None
    add_236: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    mul_443: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_236, 0.5);  add_236 = None
    mul_444: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_128, view_128)
    mul_445: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_444, -0.5);  mul_444 = None
    exp_8: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_445);  mul_445 = None
    mul_446: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_447: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_128, mul_446);  view_128 = mul_446 = None
    add_237: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_443, mul_447);  mul_443 = mul_447 = None
    mul_448: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_8, add_237);  cat_8 = add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_260: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_448, [1568, 1536]);  mul_448 = None
    permute_278: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    mm_84: "f32[1568, 256]" = torch.ops.aten.mm.default(view_260, permute_278);  permute_278 = None
    permute_279: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_260, [1, 0])
    mm_85: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_279, view_127);  permute_279 = view_127 = None
    permute_280: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_100: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_260, [0], True);  view_260 = None
    view_261: "f32[1536]" = torch.ops.aten.view.default(sum_100, [1536]);  sum_100 = None
    permute_281: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_262: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_84, [8, 196, 256]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_188: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_146, memory_format = torch.contiguous_format);  add_146 = None
    sub_115: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_188, getitem_127);  clone_188 = getitem_127 = None
    mul_449: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_42);  sub_115 = None
    mul_450: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_262, primals_213);  primals_213 = None
    mul_451: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_450, 256)
    sum_101: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_450, [2], True)
    mul_452: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_450, mul_449);  mul_450 = None
    sum_102: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_452, [2], True);  mul_452 = None
    mul_453: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_449, sum_102);  sum_102 = None
    sub_116: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_451, sum_101);  mul_451 = sum_101 = None
    sub_117: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_116, mul_453);  sub_116 = mul_453 = None
    div_19: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 256);  rsqrt_42 = None
    mul_454: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_19, sub_117);  div_19 = sub_117 = None
    mul_455: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_262, mul_449);  mul_449 = None
    sum_103: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_455, [0, 1]);  mul_455 = None
    sum_104: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_262, [0, 1]);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_238: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_235, mul_454);  add_235 = mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_263: "f32[1568, 256]" = torch.ops.aten.view.default(add_238, [1568, 256])
    permute_282: "f32[256, 768]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    mm_86: "f32[1568, 768]" = torch.ops.aten.mm.default(view_263, permute_282);  permute_282 = None
    permute_283: "f32[256, 1568]" = torch.ops.aten.permute.default(view_263, [1, 0])
    mm_87: "f32[256, 768]" = torch.ops.aten.mm.default(permute_283, view_125);  permute_283 = view_125 = None
    permute_284: "f32[768, 256]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_105: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
    view_264: "f32[256]" = torch.ops.aten.view.default(sum_105, [256]);  sum_105 = None
    permute_285: "f32[256, 768]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    view_265: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_86, [8, 196, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_456: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_265, getitem_122);  getitem_122 = None
    mul_457: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_265, permute_104);  view_265 = permute_104 = None
    permute_286: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_456, [0, 2, 1]);  mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_106: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_286, [0, 1], True)
    view_266: "f32[196]" = torch.ops.aten.view.default(sum_106, [196]);  sum_106 = None
    clone_189: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    view_267: "f32[6144, 196]" = torch.ops.aten.view.default(clone_189, [6144, 196]);  clone_189 = None
    permute_287: "f32[196, 6144]" = torch.ops.aten.permute.default(view_267, [1, 0])
    mm_88: "f32[196, 196]" = torch.ops.aten.mm.default(permute_287, view_123);  permute_287 = view_123 = None
    permute_288: "f32[196, 196]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    permute_289: "f32[196, 196]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    mm_89: "f32[6144, 196]" = torch.ops.aten.mm.default(view_267, permute_289);  view_267 = permute_289 = None
    view_268: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_89, [8, 768, 196]);  mm_89 = None
    permute_290: "f32[196, 196]" = torch.ops.aten.permute.default(permute_288, [1, 0]);  permute_288 = None
    permute_291: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_268, [0, 2, 1]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_190: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_291, memory_format = torch.contiguous_format);  permute_291 = None
    clone_191: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_123, memory_format = torch.contiguous_format);  getitem_123 = None
    sub_118: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_191, getitem_125);  clone_191 = getitem_125 = None
    mul_458: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_41);  sub_118 = None
    mul_459: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_190, primals_207);  primals_207 = None
    mul_460: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_459, 768)
    sum_107: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True)
    mul_461: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_459, mul_458);  mul_459 = None
    sum_108: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True);  mul_461 = None
    mul_462: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_458, sum_108);  sum_108 = None
    sub_119: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_460, sum_107);  mul_460 = sum_107 = None
    sub_120: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_119, mul_462);  sub_119 = mul_462 = None
    div_20: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 768);  rsqrt_41 = None
    mul_463: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_120);  div_20 = sub_120 = None
    mul_464: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_190, mul_458);  mul_458 = None
    sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 1]);  mul_464 = None
    sum_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_190, [0, 1]);  clone_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_9: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_457, mul_463], 2);  mul_457 = mul_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_465: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_122, 0.7071067811865476)
    erf_39: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_465);  mul_465 = None
    add_239: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    mul_466: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_239, 0.5);  add_239 = None
    mul_467: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_122, view_122)
    mul_468: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_467, -0.5);  mul_467 = None
    exp_9: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_468);  mul_468 = None
    mul_469: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_470: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_122, mul_469);  view_122 = mul_469 = None
    add_240: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_466, mul_470);  mul_466 = mul_470 = None
    mul_471: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_9, add_240);  cat_9 = add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_269: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_471, [1568, 1536]);  mul_471 = None
    permute_292: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    mm_90: "f32[1568, 256]" = torch.ops.aten.mm.default(view_269, permute_292);  permute_292 = None
    permute_293: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_269, [1, 0])
    mm_91: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_293, view_121);  permute_293 = view_121 = None
    permute_294: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_111: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
    view_270: "f32[1536]" = torch.ops.aten.view.default(sum_111, [1536]);  sum_111 = None
    permute_295: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
    view_271: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_90, [8, 196, 256]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_192: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_139, memory_format = torch.contiguous_format);  add_139 = None
    sub_121: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_192, getitem_121);  clone_192 = getitem_121 = None
    mul_472: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_40);  sub_121 = None
    mul_473: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_271, primals_203);  primals_203 = None
    mul_474: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_473, 256)
    sum_112: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2], True)
    mul_475: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_473, mul_472);  mul_473 = None
    sum_113: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_475, [2], True);  mul_475 = None
    mul_476: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_472, sum_113);  sum_113 = None
    sub_122: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_474, sum_112);  mul_474 = sum_112 = None
    sub_123: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_122, mul_476);  sub_122 = mul_476 = None
    div_21: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 256);  rsqrt_40 = None
    mul_477: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_21, sub_123);  div_21 = sub_123 = None
    mul_478: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_271, mul_472);  mul_472 = None
    sum_114: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_478, [0, 1]);  mul_478 = None
    sum_115: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_271, [0, 1]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_241: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_238, mul_477);  add_238 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_272: "f32[1568, 256]" = torch.ops.aten.view.default(add_241, [1568, 256])
    permute_296: "f32[256, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    mm_92: "f32[1568, 768]" = torch.ops.aten.mm.default(view_272, permute_296);  permute_296 = None
    permute_297: "f32[256, 1568]" = torch.ops.aten.permute.default(view_272, [1, 0])
    mm_93: "f32[256, 768]" = torch.ops.aten.mm.default(permute_297, view_119);  permute_297 = view_119 = None
    permute_298: "f32[768, 256]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_116: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
    view_273: "f32[256]" = torch.ops.aten.view.default(sum_116, [256]);  sum_116 = None
    permute_299: "f32[256, 768]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
    view_274: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_92, [8, 196, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_479: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_274, getitem_116);  getitem_116 = None
    mul_480: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_274, permute_99);  view_274 = permute_99 = None
    permute_300: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_479, [0, 2, 1]);  mul_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_117: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_300, [0, 1], True)
    view_275: "f32[196]" = torch.ops.aten.view.default(sum_117, [196]);  sum_117 = None
    clone_193: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_300, memory_format = torch.contiguous_format);  permute_300 = None
    view_276: "f32[6144, 196]" = torch.ops.aten.view.default(clone_193, [6144, 196]);  clone_193 = None
    permute_301: "f32[196, 6144]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_94: "f32[196, 196]" = torch.ops.aten.mm.default(permute_301, view_117);  permute_301 = view_117 = None
    permute_302: "f32[196, 196]" = torch.ops.aten.permute.default(mm_94, [1, 0]);  mm_94 = None
    permute_303: "f32[196, 196]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_95: "f32[6144, 196]" = torch.ops.aten.mm.default(view_276, permute_303);  view_276 = permute_303 = None
    view_277: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_95, [8, 768, 196]);  mm_95 = None
    permute_304: "f32[196, 196]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    permute_305: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_194: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_305, memory_format = torch.contiguous_format);  permute_305 = None
    clone_195: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_117, memory_format = torch.contiguous_format);  getitem_117 = None
    sub_124: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_195, getitem_119);  clone_195 = getitem_119 = None
    mul_481: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_39);  sub_124 = None
    mul_482: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_194, primals_197);  primals_197 = None
    mul_483: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_482, 768)
    sum_118: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_482, [2], True)
    mul_484: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_482, mul_481);  mul_482 = None
    sum_119: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_484, [2], True);  mul_484 = None
    mul_485: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_481, sum_119);  sum_119 = None
    sub_125: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_483, sum_118);  mul_483 = sum_118 = None
    sub_126: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_125, mul_485);  sub_125 = mul_485 = None
    div_22: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 768);  rsqrt_39 = None
    mul_486: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_126);  div_22 = sub_126 = None
    mul_487: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_194, mul_481);  mul_481 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 1]);  mul_487 = None
    sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_194, [0, 1]);  clone_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_10: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_480, mul_486], 2);  mul_480 = mul_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_488: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476)
    erf_40: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_488);  mul_488 = None
    add_242: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
    mul_489: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_242, 0.5);  add_242 = None
    mul_490: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_116, view_116)
    mul_491: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_490, -0.5);  mul_490 = None
    exp_10: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_491);  mul_491 = None
    mul_492: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_493: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_116, mul_492);  view_116 = mul_492 = None
    add_243: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_489, mul_493);  mul_489 = mul_493 = None
    mul_494: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_10, add_243);  cat_10 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_278: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_494, [1568, 1536]);  mul_494 = None
    permute_306: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_96: "f32[1568, 256]" = torch.ops.aten.mm.default(view_278, permute_306);  permute_306 = None
    permute_307: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_278, [1, 0])
    mm_97: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_307, view_115);  permute_307 = view_115 = None
    permute_308: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_122: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
    view_279: "f32[1536]" = torch.ops.aten.view.default(sum_122, [1536]);  sum_122 = None
    permute_309: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
    view_280: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_96, [8, 196, 256]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_196: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_132, memory_format = torch.contiguous_format);  add_132 = None
    sub_127: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_196, getitem_115);  clone_196 = getitem_115 = None
    mul_495: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_38);  sub_127 = None
    mul_496: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_280, primals_193);  primals_193 = None
    mul_497: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_496, 256)
    sum_123: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_496, [2], True)
    mul_498: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_496, mul_495);  mul_496 = None
    sum_124: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_498, [2], True);  mul_498 = None
    mul_499: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_495, sum_124);  sum_124 = None
    sub_128: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_497, sum_123);  mul_497 = sum_123 = None
    sub_129: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_128, mul_499);  sub_128 = mul_499 = None
    div_23: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 256);  rsqrt_38 = None
    mul_500: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_23, sub_129);  div_23 = sub_129 = None
    mul_501: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_280, mul_495);  mul_495 = None
    sum_125: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_501, [0, 1]);  mul_501 = None
    sum_126: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_280, [0, 1]);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_244: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_241, mul_500);  add_241 = mul_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_281: "f32[1568, 256]" = torch.ops.aten.view.default(add_244, [1568, 256])
    permute_310: "f32[256, 768]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    mm_98: "f32[1568, 768]" = torch.ops.aten.mm.default(view_281, permute_310);  permute_310 = None
    permute_311: "f32[256, 1568]" = torch.ops.aten.permute.default(view_281, [1, 0])
    mm_99: "f32[256, 768]" = torch.ops.aten.mm.default(permute_311, view_113);  permute_311 = view_113 = None
    permute_312: "f32[768, 256]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_127: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_281, [0], True);  view_281 = None
    view_282: "f32[256]" = torch.ops.aten.view.default(sum_127, [256]);  sum_127 = None
    permute_313: "f32[256, 768]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_283: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_98, [8, 196, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_502: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_283, getitem_110);  getitem_110 = None
    mul_503: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_283, permute_94);  view_283 = permute_94 = None
    permute_314: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_502, [0, 2, 1]);  mul_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_128: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_314, [0, 1], True)
    view_284: "f32[196]" = torch.ops.aten.view.default(sum_128, [196]);  sum_128 = None
    clone_197: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
    view_285: "f32[6144, 196]" = torch.ops.aten.view.default(clone_197, [6144, 196]);  clone_197 = None
    permute_315: "f32[196, 6144]" = torch.ops.aten.permute.default(view_285, [1, 0])
    mm_100: "f32[196, 196]" = torch.ops.aten.mm.default(permute_315, view_111);  permute_315 = view_111 = None
    permute_316: "f32[196, 196]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    permute_317: "f32[196, 196]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    mm_101: "f32[6144, 196]" = torch.ops.aten.mm.default(view_285, permute_317);  view_285 = permute_317 = None
    view_286: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_101, [8, 768, 196]);  mm_101 = None
    permute_318: "f32[196, 196]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
    permute_319: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_198: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    clone_199: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_111, memory_format = torch.contiguous_format);  getitem_111 = None
    sub_130: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_199, getitem_113);  clone_199 = getitem_113 = None
    mul_504: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_37);  sub_130 = None
    mul_505: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_198, primals_187);  primals_187 = None
    mul_506: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_505, 768)
    sum_129: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_505, [2], True)
    mul_507: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_505, mul_504);  mul_505 = None
    sum_130: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_507, [2], True);  mul_507 = None
    mul_508: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_504, sum_130);  sum_130 = None
    sub_131: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_506, sum_129);  mul_506 = sum_129 = None
    sub_132: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_131, mul_508);  sub_131 = mul_508 = None
    div_24: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 768);  rsqrt_37 = None
    mul_509: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_132);  div_24 = sub_132 = None
    mul_510: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_198, mul_504);  mul_504 = None
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 1]);  mul_510 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_198, [0, 1]);  clone_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_11: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_503, mul_509], 2);  mul_503 = mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_511: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_110, 0.7071067811865476)
    erf_41: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_511);  mul_511 = None
    add_245: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
    mul_512: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_245, 0.5);  add_245 = None
    mul_513: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_110, view_110)
    mul_514: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_513, -0.5);  mul_513 = None
    exp_11: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_514);  mul_514 = None
    mul_515: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_516: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_110, mul_515);  view_110 = mul_515 = None
    add_246: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_512, mul_516);  mul_512 = mul_516 = None
    mul_517: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_11, add_246);  cat_11 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_287: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_517, [1568, 1536]);  mul_517 = None
    permute_320: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    mm_102: "f32[1568, 256]" = torch.ops.aten.mm.default(view_287, permute_320);  permute_320 = None
    permute_321: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_287, [1, 0])
    mm_103: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_321, view_109);  permute_321 = view_109 = None
    permute_322: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_133: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
    view_288: "f32[1536]" = torch.ops.aten.view.default(sum_133, [1536]);  sum_133 = None
    permute_323: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_322, [1, 0]);  permute_322 = None
    view_289: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_102, [8, 196, 256]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_200: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format);  add_125 = None
    sub_133: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_200, getitem_109);  clone_200 = getitem_109 = None
    mul_518: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_36);  sub_133 = None
    mul_519: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_289, primals_183);  primals_183 = None
    mul_520: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_519, 256)
    sum_134: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_519, [2], True)
    mul_521: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_519, mul_518);  mul_519 = None
    sum_135: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_521, [2], True);  mul_521 = None
    mul_522: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_518, sum_135);  sum_135 = None
    sub_134: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_520, sum_134);  mul_520 = sum_134 = None
    sub_135: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_134, mul_522);  sub_134 = mul_522 = None
    div_25: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 256);  rsqrt_36 = None
    mul_523: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_25, sub_135);  div_25 = sub_135 = None
    mul_524: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_289, mul_518);  mul_518 = None
    sum_136: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_524, [0, 1]);  mul_524 = None
    sum_137: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_289, [0, 1]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_247: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_244, mul_523);  add_244 = mul_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_290: "f32[1568, 256]" = torch.ops.aten.view.default(add_247, [1568, 256])
    permute_324: "f32[256, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    mm_104: "f32[1568, 768]" = torch.ops.aten.mm.default(view_290, permute_324);  permute_324 = None
    permute_325: "f32[256, 1568]" = torch.ops.aten.permute.default(view_290, [1, 0])
    mm_105: "f32[256, 768]" = torch.ops.aten.mm.default(permute_325, view_107);  permute_325 = view_107 = None
    permute_326: "f32[768, 256]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_138: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
    view_291: "f32[256]" = torch.ops.aten.view.default(sum_138, [256]);  sum_138 = None
    permute_327: "f32[256, 768]" = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
    view_292: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_104, [8, 196, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_525: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_292, getitem_104);  getitem_104 = None
    mul_526: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_292, permute_89);  view_292 = permute_89 = None
    permute_328: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_525, [0, 2, 1]);  mul_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_139: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_328, [0, 1], True)
    view_293: "f32[196]" = torch.ops.aten.view.default(sum_139, [196]);  sum_139 = None
    clone_201: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    view_294: "f32[6144, 196]" = torch.ops.aten.view.default(clone_201, [6144, 196]);  clone_201 = None
    permute_329: "f32[196, 6144]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_106: "f32[196, 196]" = torch.ops.aten.mm.default(permute_329, view_105);  permute_329 = view_105 = None
    permute_330: "f32[196, 196]" = torch.ops.aten.permute.default(mm_106, [1, 0]);  mm_106 = None
    permute_331: "f32[196, 196]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_107: "f32[6144, 196]" = torch.ops.aten.mm.default(view_294, permute_331);  view_294 = permute_331 = None
    view_295: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_107, [8, 768, 196]);  mm_107 = None
    permute_332: "f32[196, 196]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    permute_333: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_202: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_333, memory_format = torch.contiguous_format);  permute_333 = None
    clone_203: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_105, memory_format = torch.contiguous_format);  getitem_105 = None
    sub_136: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_203, getitem_107);  clone_203 = getitem_107 = None
    mul_527: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_136, rsqrt_35);  sub_136 = None
    mul_528: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_202, primals_177);  primals_177 = None
    mul_529: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_528, 768)
    sum_140: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_528, [2], True)
    mul_530: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_528, mul_527);  mul_528 = None
    sum_141: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_530, [2], True);  mul_530 = None
    mul_531: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_527, sum_141);  sum_141 = None
    sub_137: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_529, sum_140);  mul_529 = sum_140 = None
    sub_138: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_137, mul_531);  sub_137 = mul_531 = None
    div_26: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 768);  rsqrt_35 = None
    mul_532: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_138);  div_26 = sub_138 = None
    mul_533: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_202, mul_527);  mul_527 = None
    sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_533, [0, 1]);  mul_533 = None
    sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_202, [0, 1]);  clone_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_12: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_526, mul_532], 2);  mul_526 = mul_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_534: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_104, 0.7071067811865476)
    erf_42: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_534);  mul_534 = None
    add_248: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
    mul_535: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_248, 0.5);  add_248 = None
    mul_536: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_104, view_104)
    mul_537: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_536, -0.5);  mul_536 = None
    exp_12: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_537);  mul_537 = None
    mul_538: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_539: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_104, mul_538);  view_104 = mul_538 = None
    add_249: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_535, mul_539);  mul_535 = mul_539 = None
    mul_540: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_12, add_249);  cat_12 = add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_296: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_540, [1568, 1536]);  mul_540 = None
    permute_334: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_108: "f32[1568, 256]" = torch.ops.aten.mm.default(view_296, permute_334);  permute_334 = None
    permute_335: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_296, [1, 0])
    mm_109: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_335, view_103);  permute_335 = view_103 = None
    permute_336: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_144: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
    view_297: "f32[1536]" = torch.ops.aten.view.default(sum_144, [1536]);  sum_144 = None
    permute_337: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_336, [1, 0]);  permute_336 = None
    view_298: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_108, [8, 196, 256]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_204: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_118, memory_format = torch.contiguous_format);  add_118 = None
    sub_139: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_204, getitem_103);  clone_204 = getitem_103 = None
    mul_541: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_139, rsqrt_34);  sub_139 = None
    mul_542: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_298, primals_173);  primals_173 = None
    mul_543: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_542, 256)
    sum_145: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_542, [2], True)
    mul_544: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_542, mul_541);  mul_542 = None
    sum_146: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_544, [2], True);  mul_544 = None
    mul_545: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_541, sum_146);  sum_146 = None
    sub_140: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_543, sum_145);  mul_543 = sum_145 = None
    sub_141: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_140, mul_545);  sub_140 = mul_545 = None
    div_27: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 256);  rsqrt_34 = None
    mul_546: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_27, sub_141);  div_27 = sub_141 = None
    mul_547: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_298, mul_541);  mul_541 = None
    sum_147: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 1]);  mul_547 = None
    sum_148: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_298, [0, 1]);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_250: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_247, mul_546);  add_247 = mul_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_299: "f32[1568, 256]" = torch.ops.aten.view.default(add_250, [1568, 256])
    permute_338: "f32[256, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    mm_110: "f32[1568, 768]" = torch.ops.aten.mm.default(view_299, permute_338);  permute_338 = None
    permute_339: "f32[256, 1568]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_111: "f32[256, 768]" = torch.ops.aten.mm.default(permute_339, view_101);  permute_339 = view_101 = None
    permute_340: "f32[768, 256]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_149: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[256]" = torch.ops.aten.view.default(sum_149, [256]);  sum_149 = None
    permute_341: "f32[256, 768]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
    view_301: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_110, [8, 196, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_548: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_301, getitem_98);  getitem_98 = None
    mul_549: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_301, permute_84);  view_301 = permute_84 = None
    permute_342: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_548, [0, 2, 1]);  mul_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_150: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_342, [0, 1], True)
    view_302: "f32[196]" = torch.ops.aten.view.default(sum_150, [196]);  sum_150 = None
    clone_205: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_342, memory_format = torch.contiguous_format);  permute_342 = None
    view_303: "f32[6144, 196]" = torch.ops.aten.view.default(clone_205, [6144, 196]);  clone_205 = None
    permute_343: "f32[196, 6144]" = torch.ops.aten.permute.default(view_303, [1, 0])
    mm_112: "f32[196, 196]" = torch.ops.aten.mm.default(permute_343, view_99);  permute_343 = view_99 = None
    permute_344: "f32[196, 196]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    permute_345: "f32[196, 196]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    mm_113: "f32[6144, 196]" = torch.ops.aten.mm.default(view_303, permute_345);  view_303 = permute_345 = None
    view_304: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_113, [8, 768, 196]);  mm_113 = None
    permute_346: "f32[196, 196]" = torch.ops.aten.permute.default(permute_344, [1, 0]);  permute_344 = None
    permute_347: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_206: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
    clone_207: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_99, memory_format = torch.contiguous_format);  getitem_99 = None
    sub_142: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_207, getitem_101);  clone_207 = getitem_101 = None
    mul_550: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_142, rsqrt_33);  sub_142 = None
    mul_551: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_206, primals_167);  primals_167 = None
    mul_552: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_551, 768)
    sum_151: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_551, [2], True)
    mul_553: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_551, mul_550);  mul_551 = None
    sum_152: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_553, [2], True);  mul_553 = None
    mul_554: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_550, sum_152);  sum_152 = None
    sub_143: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_552, sum_151);  mul_552 = sum_151 = None
    sub_144: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_143, mul_554);  sub_143 = mul_554 = None
    div_28: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 768);  rsqrt_33 = None
    mul_555: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_144);  div_28 = sub_144 = None
    mul_556: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_206, mul_550);  mul_550 = None
    sum_153: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 1]);  mul_556 = None
    sum_154: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_206, [0, 1]);  clone_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_13: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_549, mul_555], 2);  mul_549 = mul_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_557: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
    erf_43: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_557);  mul_557 = None
    add_251: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
    mul_558: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_251, 0.5);  add_251 = None
    mul_559: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_98, view_98)
    mul_560: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_559, -0.5);  mul_559 = None
    exp_13: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_560);  mul_560 = None
    mul_561: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_562: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_98, mul_561);  view_98 = mul_561 = None
    add_252: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_558, mul_562);  mul_558 = mul_562 = None
    mul_563: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_13, add_252);  cat_13 = add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_305: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_563, [1568, 1536]);  mul_563 = None
    permute_348: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    mm_114: "f32[1568, 256]" = torch.ops.aten.mm.default(view_305, permute_348);  permute_348 = None
    permute_349: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_305, [1, 0])
    mm_115: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_349, view_97);  permute_349 = view_97 = None
    permute_350: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_155: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
    view_306: "f32[1536]" = torch.ops.aten.view.default(sum_155, [1536]);  sum_155 = None
    permute_351: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    view_307: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_114, [8, 196, 256]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_208: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format);  add_111 = None
    sub_145: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_208, getitem_97);  clone_208 = getitem_97 = None
    mul_564: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_32);  sub_145 = None
    mul_565: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_307, primals_163);  primals_163 = None
    mul_566: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_565, 256)
    sum_156: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_565, [2], True)
    mul_567: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_565, mul_564);  mul_565 = None
    sum_157: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_567, [2], True);  mul_567 = None
    mul_568: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_564, sum_157);  sum_157 = None
    sub_146: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_566, sum_156);  mul_566 = sum_156 = None
    sub_147: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_146, mul_568);  sub_146 = mul_568 = None
    div_29: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 256);  rsqrt_32 = None
    mul_569: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_29, sub_147);  div_29 = sub_147 = None
    mul_570: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_307, mul_564);  mul_564 = None
    sum_158: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_570, [0, 1]);  mul_570 = None
    sum_159: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_307, [0, 1]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_253: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_250, mul_569);  add_250 = mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_308: "f32[1568, 256]" = torch.ops.aten.view.default(add_253, [1568, 256])
    permute_352: "f32[256, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    mm_116: "f32[1568, 768]" = torch.ops.aten.mm.default(view_308, permute_352);  permute_352 = None
    permute_353: "f32[256, 1568]" = torch.ops.aten.permute.default(view_308, [1, 0])
    mm_117: "f32[256, 768]" = torch.ops.aten.mm.default(permute_353, view_95);  permute_353 = view_95 = None
    permute_354: "f32[768, 256]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_160: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
    view_309: "f32[256]" = torch.ops.aten.view.default(sum_160, [256]);  sum_160 = None
    permute_355: "f32[256, 768]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    view_310: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_116, [8, 196, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_571: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_310, getitem_92);  getitem_92 = None
    mul_572: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_310, permute_79);  view_310 = permute_79 = None
    permute_356: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_571, [0, 2, 1]);  mul_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_161: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_356, [0, 1], True)
    view_311: "f32[196]" = torch.ops.aten.view.default(sum_161, [196]);  sum_161 = None
    clone_209: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
    view_312: "f32[6144, 196]" = torch.ops.aten.view.default(clone_209, [6144, 196]);  clone_209 = None
    permute_357: "f32[196, 6144]" = torch.ops.aten.permute.default(view_312, [1, 0])
    mm_118: "f32[196, 196]" = torch.ops.aten.mm.default(permute_357, view_93);  permute_357 = view_93 = None
    permute_358: "f32[196, 196]" = torch.ops.aten.permute.default(mm_118, [1, 0]);  mm_118 = None
    permute_359: "f32[196, 196]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_119: "f32[6144, 196]" = torch.ops.aten.mm.default(view_312, permute_359);  view_312 = permute_359 = None
    view_313: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_119, [8, 768, 196]);  mm_119 = None
    permute_360: "f32[196, 196]" = torch.ops.aten.permute.default(permute_358, [1, 0]);  permute_358 = None
    permute_361: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_210: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_361, memory_format = torch.contiguous_format);  permute_361 = None
    clone_211: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_93, memory_format = torch.contiguous_format);  getitem_93 = None
    sub_148: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_211, getitem_95);  clone_211 = getitem_95 = None
    mul_573: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_148, rsqrt_31);  sub_148 = None
    mul_574: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_210, primals_157);  primals_157 = None
    mul_575: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_574, 768)
    sum_162: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_574, [2], True)
    mul_576: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_574, mul_573);  mul_574 = None
    sum_163: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_576, [2], True);  mul_576 = None
    mul_577: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_573, sum_163);  sum_163 = None
    sub_149: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_575, sum_162);  mul_575 = sum_162 = None
    sub_150: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_149, mul_577);  sub_149 = mul_577 = None
    div_30: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 768);  rsqrt_31 = None
    mul_578: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_150);  div_30 = sub_150 = None
    mul_579: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_210, mul_573);  mul_573 = None
    sum_164: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_579, [0, 1]);  mul_579 = None
    sum_165: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_210, [0, 1]);  clone_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_14: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_572, mul_578], 2);  mul_572 = mul_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_580: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_92, 0.7071067811865476)
    erf_44: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_580);  mul_580 = None
    add_254: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
    mul_581: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_254, 0.5);  add_254 = None
    mul_582: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_92, view_92)
    mul_583: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_582, -0.5);  mul_582 = None
    exp_14: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_583);  mul_583 = None
    mul_584: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_585: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_92, mul_584);  view_92 = mul_584 = None
    add_255: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_581, mul_585);  mul_581 = mul_585 = None
    mul_586: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_14, add_255);  cat_14 = add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_314: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_586, [1568, 1536]);  mul_586 = None
    permute_362: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_120: "f32[1568, 256]" = torch.ops.aten.mm.default(view_314, permute_362);  permute_362 = None
    permute_363: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_314, [1, 0])
    mm_121: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_363, view_91);  permute_363 = view_91 = None
    permute_364: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_166: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
    view_315: "f32[1536]" = torch.ops.aten.view.default(sum_166, [1536]);  sum_166 = None
    permute_365: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    view_316: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_120, [8, 196, 256]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_212: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_104, memory_format = torch.contiguous_format);  add_104 = None
    sub_151: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_212, getitem_91);  clone_212 = getitem_91 = None
    mul_587: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_151, rsqrt_30);  sub_151 = None
    mul_588: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_316, primals_153);  primals_153 = None
    mul_589: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_588, 256)
    sum_167: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [2], True)
    mul_590: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_588, mul_587);  mul_588 = None
    sum_168: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_590, [2], True);  mul_590 = None
    mul_591: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_587, sum_168);  sum_168 = None
    sub_152: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_589, sum_167);  mul_589 = sum_167 = None
    sub_153: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_152, mul_591);  sub_152 = mul_591 = None
    div_31: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 256);  rsqrt_30 = None
    mul_592: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_31, sub_153);  div_31 = sub_153 = None
    mul_593: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_316, mul_587);  mul_587 = None
    sum_169: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_593, [0, 1]);  mul_593 = None
    sum_170: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_316, [0, 1]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_256: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_253, mul_592);  add_253 = mul_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_317: "f32[1568, 256]" = torch.ops.aten.view.default(add_256, [1568, 256])
    permute_366: "f32[256, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_122: "f32[1568, 768]" = torch.ops.aten.mm.default(view_317, permute_366);  permute_366 = None
    permute_367: "f32[256, 1568]" = torch.ops.aten.permute.default(view_317, [1, 0])
    mm_123: "f32[256, 768]" = torch.ops.aten.mm.default(permute_367, view_89);  permute_367 = view_89 = None
    permute_368: "f32[768, 256]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_171: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
    view_318: "f32[256]" = torch.ops.aten.view.default(sum_171, [256]);  sum_171 = None
    permute_369: "f32[256, 768]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    view_319: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_122, [8, 196, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_594: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_319, getitem_86);  getitem_86 = None
    mul_595: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_319, permute_74);  view_319 = permute_74 = None
    permute_370: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_594, [0, 2, 1]);  mul_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_172: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_370, [0, 1], True)
    view_320: "f32[196]" = torch.ops.aten.view.default(sum_172, [196]);  sum_172 = None
    clone_213: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    view_321: "f32[6144, 196]" = torch.ops.aten.view.default(clone_213, [6144, 196]);  clone_213 = None
    permute_371: "f32[196, 6144]" = torch.ops.aten.permute.default(view_321, [1, 0])
    mm_124: "f32[196, 196]" = torch.ops.aten.mm.default(permute_371, view_87);  permute_371 = view_87 = None
    permute_372: "f32[196, 196]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    permute_373: "f32[196, 196]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    mm_125: "f32[6144, 196]" = torch.ops.aten.mm.default(view_321, permute_373);  view_321 = permute_373 = None
    view_322: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_125, [8, 768, 196]);  mm_125 = None
    permute_374: "f32[196, 196]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    permute_375: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_322, [0, 2, 1]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_214: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_375, memory_format = torch.contiguous_format);  permute_375 = None
    clone_215: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_87, memory_format = torch.contiguous_format);  getitem_87 = None
    sub_154: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_215, getitem_89);  clone_215 = getitem_89 = None
    mul_596: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_154, rsqrt_29);  sub_154 = None
    mul_597: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_214, primals_147);  primals_147 = None
    mul_598: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_597, 768)
    sum_173: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_597, [2], True)
    mul_599: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_597, mul_596);  mul_597 = None
    sum_174: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_599, [2], True);  mul_599 = None
    mul_600: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_596, sum_174);  sum_174 = None
    sub_155: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_598, sum_173);  mul_598 = sum_173 = None
    sub_156: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_155, mul_600);  sub_155 = mul_600 = None
    div_32: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 768);  rsqrt_29 = None
    mul_601: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_156);  div_32 = sub_156 = None
    mul_602: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_214, mul_596);  mul_596 = None
    sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_602, [0, 1]);  mul_602 = None
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_214, [0, 1]);  clone_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_15: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_595, mul_601], 2);  mul_595 = mul_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_603: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_86, 0.7071067811865476)
    erf_45: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_603);  mul_603 = None
    add_257: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
    mul_604: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_257, 0.5);  add_257 = None
    mul_605: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_86, view_86)
    mul_606: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_605, -0.5);  mul_605 = None
    exp_15: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_606);  mul_606 = None
    mul_607: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_608: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_86, mul_607);  view_86 = mul_607 = None
    add_258: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_604, mul_608);  mul_604 = mul_608 = None
    mul_609: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_15, add_258);  cat_15 = add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_323: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_609, [1568, 1536]);  mul_609 = None
    permute_376: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    mm_126: "f32[1568, 256]" = torch.ops.aten.mm.default(view_323, permute_376);  permute_376 = None
    permute_377: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_323, [1, 0])
    mm_127: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_377, view_85);  permute_377 = view_85 = None
    permute_378: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_177: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
    view_324: "f32[1536]" = torch.ops.aten.view.default(sum_177, [1536]);  sum_177 = None
    permute_379: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    view_325: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_126, [8, 196, 256]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_216: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_97, memory_format = torch.contiguous_format);  add_97 = None
    sub_157: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_216, getitem_85);  clone_216 = getitem_85 = None
    mul_610: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_157, rsqrt_28);  sub_157 = None
    mul_611: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_325, primals_143);  primals_143 = None
    mul_612: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_611, 256)
    sum_178: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_611, [2], True)
    mul_613: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_611, mul_610);  mul_611 = None
    sum_179: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_613, [2], True);  mul_613 = None
    mul_614: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_610, sum_179);  sum_179 = None
    sub_158: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_612, sum_178);  mul_612 = sum_178 = None
    sub_159: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_158, mul_614);  sub_158 = mul_614 = None
    div_33: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 256);  rsqrt_28 = None
    mul_615: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_33, sub_159);  div_33 = sub_159 = None
    mul_616: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_325, mul_610);  mul_610 = None
    sum_180: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_616, [0, 1]);  mul_616 = None
    sum_181: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_325, [0, 1]);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_259: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_256, mul_615);  add_256 = mul_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_326: "f32[1568, 256]" = torch.ops.aten.view.default(add_259, [1568, 256])
    permute_380: "f32[256, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    mm_128: "f32[1568, 768]" = torch.ops.aten.mm.default(view_326, permute_380);  permute_380 = None
    permute_381: "f32[256, 1568]" = torch.ops.aten.permute.default(view_326, [1, 0])
    mm_129: "f32[256, 768]" = torch.ops.aten.mm.default(permute_381, view_83);  permute_381 = view_83 = None
    permute_382: "f32[768, 256]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_182: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_326, [0], True);  view_326 = None
    view_327: "f32[256]" = torch.ops.aten.view.default(sum_182, [256]);  sum_182 = None
    permute_383: "f32[256, 768]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
    view_328: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_128, [8, 196, 768]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_617: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_328, getitem_80);  getitem_80 = None
    mul_618: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_328, permute_69);  view_328 = permute_69 = None
    permute_384: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_617, [0, 2, 1]);  mul_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_183: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_384, [0, 1], True)
    view_329: "f32[196]" = torch.ops.aten.view.default(sum_183, [196]);  sum_183 = None
    clone_217: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_384, memory_format = torch.contiguous_format);  permute_384 = None
    view_330: "f32[6144, 196]" = torch.ops.aten.view.default(clone_217, [6144, 196]);  clone_217 = None
    permute_385: "f32[196, 6144]" = torch.ops.aten.permute.default(view_330, [1, 0])
    mm_130: "f32[196, 196]" = torch.ops.aten.mm.default(permute_385, view_81);  permute_385 = view_81 = None
    permute_386: "f32[196, 196]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    permute_387: "f32[196, 196]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    mm_131: "f32[6144, 196]" = torch.ops.aten.mm.default(view_330, permute_387);  view_330 = permute_387 = None
    view_331: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_131, [8, 768, 196]);  mm_131 = None
    permute_388: "f32[196, 196]" = torch.ops.aten.permute.default(permute_386, [1, 0]);  permute_386 = None
    permute_389: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_331, [0, 2, 1]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_218: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_389, memory_format = torch.contiguous_format);  permute_389 = None
    clone_219: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_81, memory_format = torch.contiguous_format);  getitem_81 = None
    sub_160: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_219, getitem_83);  clone_219 = getitem_83 = None
    mul_619: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_160, rsqrt_27);  sub_160 = None
    mul_620: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_218, primals_137);  primals_137 = None
    mul_621: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_620, 768)
    sum_184: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [2], True)
    mul_622: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_620, mul_619);  mul_620 = None
    sum_185: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [2], True);  mul_622 = None
    mul_623: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_619, sum_185);  sum_185 = None
    sub_161: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_621, sum_184);  mul_621 = sum_184 = None
    sub_162: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_161, mul_623);  sub_161 = mul_623 = None
    div_34: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 768);  rsqrt_27 = None
    mul_624: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_162);  div_34 = sub_162 = None
    mul_625: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_218, mul_619);  mul_619 = None
    sum_186: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_625, [0, 1]);  mul_625 = None
    sum_187: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_218, [0, 1]);  clone_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_16: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_618, mul_624], 2);  mul_618 = mul_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_626: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476)
    erf_46: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_626);  mul_626 = None
    add_260: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
    mul_627: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_260, 0.5);  add_260 = None
    mul_628: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_80, view_80)
    mul_629: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_628, -0.5);  mul_628 = None
    exp_16: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_629);  mul_629 = None
    mul_630: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_631: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_80, mul_630);  view_80 = mul_630 = None
    add_261: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_627, mul_631);  mul_627 = mul_631 = None
    mul_632: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_16, add_261);  cat_16 = add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_332: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_632, [1568, 1536]);  mul_632 = None
    permute_390: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_132: "f32[1568, 256]" = torch.ops.aten.mm.default(view_332, permute_390);  permute_390 = None
    permute_391: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_332, [1, 0])
    mm_133: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_391, view_79);  permute_391 = view_79 = None
    permute_392: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_188: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
    view_333: "f32[1536]" = torch.ops.aten.view.default(sum_188, [1536]);  sum_188 = None
    permute_393: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
    view_334: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_132, [8, 196, 256]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_220: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format);  add_90 = None
    sub_163: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_220, getitem_79);  clone_220 = getitem_79 = None
    mul_633: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_163, rsqrt_26);  sub_163 = None
    mul_634: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_334, primals_133);  primals_133 = None
    mul_635: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_634, 256)
    sum_189: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_634, [2], True)
    mul_636: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_634, mul_633);  mul_634 = None
    sum_190: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_636, [2], True);  mul_636 = None
    mul_637: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_633, sum_190);  sum_190 = None
    sub_164: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_635, sum_189);  mul_635 = sum_189 = None
    sub_165: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_164, mul_637);  sub_164 = mul_637 = None
    div_35: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 256);  rsqrt_26 = None
    mul_638: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_35, sub_165);  div_35 = sub_165 = None
    mul_639: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_334, mul_633);  mul_633 = None
    sum_191: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_639, [0, 1]);  mul_639 = None
    sum_192: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_334, [0, 1]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_262: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_259, mul_638);  add_259 = mul_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_335: "f32[1568, 256]" = torch.ops.aten.view.default(add_262, [1568, 256])
    permute_394: "f32[256, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_134: "f32[1568, 768]" = torch.ops.aten.mm.default(view_335, permute_394);  permute_394 = None
    permute_395: "f32[256, 1568]" = torch.ops.aten.permute.default(view_335, [1, 0])
    mm_135: "f32[256, 768]" = torch.ops.aten.mm.default(permute_395, view_77);  permute_395 = view_77 = None
    permute_396: "f32[768, 256]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_193: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
    view_336: "f32[256]" = torch.ops.aten.view.default(sum_193, [256]);  sum_193 = None
    permute_397: "f32[256, 768]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_337: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_134, [8, 196, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_640: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_337, getitem_74);  getitem_74 = None
    mul_641: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_337, permute_64);  view_337 = permute_64 = None
    permute_398: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_640, [0, 2, 1]);  mul_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_194: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_398, [0, 1], True)
    view_338: "f32[196]" = torch.ops.aten.view.default(sum_194, [196]);  sum_194 = None
    clone_221: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_398, memory_format = torch.contiguous_format);  permute_398 = None
    view_339: "f32[6144, 196]" = torch.ops.aten.view.default(clone_221, [6144, 196]);  clone_221 = None
    permute_399: "f32[196, 6144]" = torch.ops.aten.permute.default(view_339, [1, 0])
    mm_136: "f32[196, 196]" = torch.ops.aten.mm.default(permute_399, view_75);  permute_399 = view_75 = None
    permute_400: "f32[196, 196]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    permute_401: "f32[196, 196]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_137: "f32[6144, 196]" = torch.ops.aten.mm.default(view_339, permute_401);  view_339 = permute_401 = None
    view_340: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_137, [8, 768, 196]);  mm_137 = None
    permute_402: "f32[196, 196]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    permute_403: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_340, [0, 2, 1]);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_222: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_403, memory_format = torch.contiguous_format);  permute_403 = None
    clone_223: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_75, memory_format = torch.contiguous_format);  getitem_75 = None
    sub_166: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_223, getitem_77);  clone_223 = getitem_77 = None
    mul_642: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_166, rsqrt_25);  sub_166 = None
    mul_643: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_222, primals_127);  primals_127 = None
    mul_644: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_643, 768)
    sum_195: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_643, [2], True)
    mul_645: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_643, mul_642);  mul_643 = None
    sum_196: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_645, [2], True);  mul_645 = None
    mul_646: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_642, sum_196);  sum_196 = None
    sub_167: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_644, sum_195);  mul_644 = sum_195 = None
    sub_168: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_167, mul_646);  sub_167 = mul_646 = None
    div_36: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
    mul_647: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_168);  div_36 = sub_168 = None
    mul_648: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_222, mul_642);  mul_642 = None
    sum_197: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_648, [0, 1]);  mul_648 = None
    sum_198: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_222, [0, 1]);  clone_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_17: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_641, mul_647], 2);  mul_641 = mul_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_649: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_74, 0.7071067811865476)
    erf_47: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_649);  mul_649 = None
    add_263: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
    mul_650: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_263, 0.5);  add_263 = None
    mul_651: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_74, view_74)
    mul_652: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_651, -0.5);  mul_651 = None
    exp_17: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_652);  mul_652 = None
    mul_653: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_654: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_74, mul_653);  view_74 = mul_653 = None
    add_264: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_650, mul_654);  mul_650 = mul_654 = None
    mul_655: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_17, add_264);  cat_17 = add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_341: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_655, [1568, 1536]);  mul_655 = None
    permute_404: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    mm_138: "f32[1568, 256]" = torch.ops.aten.mm.default(view_341, permute_404);  permute_404 = None
    permute_405: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_341, [1, 0])
    mm_139: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_405, view_73);  permute_405 = view_73 = None
    permute_406: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_199: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
    view_342: "f32[1536]" = torch.ops.aten.view.default(sum_199, [1536]);  sum_199 = None
    permute_407: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    view_343: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_138, [8, 196, 256]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_224: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_83, memory_format = torch.contiguous_format);  add_83 = None
    sub_169: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_224, getitem_73);  clone_224 = getitem_73 = None
    mul_656: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_169, rsqrt_24);  sub_169 = None
    mul_657: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_343, primals_123);  primals_123 = None
    mul_658: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_657, 256)
    sum_200: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_657, [2], True)
    mul_659: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_657, mul_656);  mul_657 = None
    sum_201: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_659, [2], True);  mul_659 = None
    mul_660: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_656, sum_201);  sum_201 = None
    sub_170: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_658, sum_200);  mul_658 = sum_200 = None
    sub_171: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_170, mul_660);  sub_170 = mul_660 = None
    div_37: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 256);  rsqrt_24 = None
    mul_661: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_37, sub_171);  div_37 = sub_171 = None
    mul_662: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_343, mul_656);  mul_656 = None
    sum_202: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 1]);  mul_662 = None
    sum_203: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_343, [0, 1]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_265: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_262, mul_661);  add_262 = mul_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_344: "f32[1568, 256]" = torch.ops.aten.view.default(add_265, [1568, 256])
    permute_408: "f32[256, 768]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    mm_140: "f32[1568, 768]" = torch.ops.aten.mm.default(view_344, permute_408);  permute_408 = None
    permute_409: "f32[256, 1568]" = torch.ops.aten.permute.default(view_344, [1, 0])
    mm_141: "f32[256, 768]" = torch.ops.aten.mm.default(permute_409, view_71);  permute_409 = view_71 = None
    permute_410: "f32[768, 256]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_204: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True);  view_344 = None
    view_345: "f32[256]" = torch.ops.aten.view.default(sum_204, [256]);  sum_204 = None
    permute_411: "f32[256, 768]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
    view_346: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_140, [8, 196, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_663: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_346, getitem_68);  getitem_68 = None
    mul_664: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_346, permute_59);  view_346 = permute_59 = None
    permute_412: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_663, [0, 2, 1]);  mul_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_205: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_412, [0, 1], True)
    view_347: "f32[196]" = torch.ops.aten.view.default(sum_205, [196]);  sum_205 = None
    clone_225: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_412, memory_format = torch.contiguous_format);  permute_412 = None
    view_348: "f32[6144, 196]" = torch.ops.aten.view.default(clone_225, [6144, 196]);  clone_225 = None
    permute_413: "f32[196, 6144]" = torch.ops.aten.permute.default(view_348, [1, 0])
    mm_142: "f32[196, 196]" = torch.ops.aten.mm.default(permute_413, view_69);  permute_413 = view_69 = None
    permute_414: "f32[196, 196]" = torch.ops.aten.permute.default(mm_142, [1, 0]);  mm_142 = None
    permute_415: "f32[196, 196]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_143: "f32[6144, 196]" = torch.ops.aten.mm.default(view_348, permute_415);  view_348 = permute_415 = None
    view_349: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_143, [8, 768, 196]);  mm_143 = None
    permute_416: "f32[196, 196]" = torch.ops.aten.permute.default(permute_414, [1, 0]);  permute_414 = None
    permute_417: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_226: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_417, memory_format = torch.contiguous_format);  permute_417 = None
    clone_227: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_69, memory_format = torch.contiguous_format);  getitem_69 = None
    sub_172: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_227, getitem_71);  clone_227 = getitem_71 = None
    mul_665: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_172, rsqrt_23);  sub_172 = None
    mul_666: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_226, primals_117);  primals_117 = None
    mul_667: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_666, 768)
    sum_206: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_666, [2], True)
    mul_668: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_666, mul_665);  mul_666 = None
    sum_207: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_668, [2], True);  mul_668 = None
    mul_669: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_665, sum_207);  sum_207 = None
    sub_173: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_667, sum_206);  mul_667 = sum_206 = None
    sub_174: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_173, mul_669);  sub_173 = mul_669 = None
    div_38: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    mul_670: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_174);  div_38 = sub_174 = None
    mul_671: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_226, mul_665);  mul_665 = None
    sum_208: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 1]);  mul_671 = None
    sum_209: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_226, [0, 1]);  clone_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_18: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_664, mul_670], 2);  mul_664 = mul_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_672: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476)
    erf_48: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_672);  mul_672 = None
    add_266: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
    mul_673: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_266, 0.5);  add_266 = None
    mul_674: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_68, view_68)
    mul_675: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_674, -0.5);  mul_674 = None
    exp_18: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_675);  mul_675 = None
    mul_676: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_677: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_68, mul_676);  view_68 = mul_676 = None
    add_267: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_673, mul_677);  mul_673 = mul_677 = None
    mul_678: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_18, add_267);  cat_18 = add_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_350: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_678, [1568, 1536]);  mul_678 = None
    permute_418: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_144: "f32[1568, 256]" = torch.ops.aten.mm.default(view_350, permute_418);  permute_418 = None
    permute_419: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_350, [1, 0])
    mm_145: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_419, view_67);  permute_419 = view_67 = None
    permute_420: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_210: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_350, [0], True);  view_350 = None
    view_351: "f32[1536]" = torch.ops.aten.view.default(sum_210, [1536]);  sum_210 = None
    permute_421: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_420, [1, 0]);  permute_420 = None
    view_352: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_144, [8, 196, 256]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_228: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format);  add_76 = None
    sub_175: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_228, getitem_67);  clone_228 = getitem_67 = None
    mul_679: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_175, rsqrt_22);  sub_175 = None
    mul_680: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_352, primals_113);  primals_113 = None
    mul_681: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_680, 256)
    sum_211: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_680, [2], True)
    mul_682: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_680, mul_679);  mul_680 = None
    sum_212: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_682, [2], True);  mul_682 = None
    mul_683: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_679, sum_212);  sum_212 = None
    sub_176: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_681, sum_211);  mul_681 = sum_211 = None
    sub_177: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_176, mul_683);  sub_176 = mul_683 = None
    div_39: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 256);  rsqrt_22 = None
    mul_684: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_39, sub_177);  div_39 = sub_177 = None
    mul_685: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_352, mul_679);  mul_679 = None
    sum_213: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_685, [0, 1]);  mul_685 = None
    sum_214: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_352, [0, 1]);  view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_268: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_265, mul_684);  add_265 = mul_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_353: "f32[1568, 256]" = torch.ops.aten.view.default(add_268, [1568, 256])
    permute_422: "f32[256, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_146: "f32[1568, 768]" = torch.ops.aten.mm.default(view_353, permute_422);  permute_422 = None
    permute_423: "f32[256, 1568]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_147: "f32[256, 768]" = torch.ops.aten.mm.default(permute_423, view_65);  permute_423 = view_65 = None
    permute_424: "f32[768, 256]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_215: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[256]" = torch.ops.aten.view.default(sum_215, [256]);  sum_215 = None
    permute_425: "f32[256, 768]" = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
    view_355: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_146, [8, 196, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_686: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_355, getitem_62);  getitem_62 = None
    mul_687: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_355, permute_54);  view_355 = permute_54 = None
    permute_426: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_686, [0, 2, 1]);  mul_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_216: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_426, [0, 1], True)
    view_356: "f32[196]" = torch.ops.aten.view.default(sum_216, [196]);  sum_216 = None
    clone_229: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    view_357: "f32[6144, 196]" = torch.ops.aten.view.default(clone_229, [6144, 196]);  clone_229 = None
    permute_427: "f32[196, 6144]" = torch.ops.aten.permute.default(view_357, [1, 0])
    mm_148: "f32[196, 196]" = torch.ops.aten.mm.default(permute_427, view_63);  permute_427 = view_63 = None
    permute_428: "f32[196, 196]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    permute_429: "f32[196, 196]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_149: "f32[6144, 196]" = torch.ops.aten.mm.default(view_357, permute_429);  view_357 = permute_429 = None
    view_358: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_149, [8, 768, 196]);  mm_149 = None
    permute_430: "f32[196, 196]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    permute_431: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_358, [0, 2, 1]);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_230: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_431, memory_format = torch.contiguous_format);  permute_431 = None
    clone_231: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_63, memory_format = torch.contiguous_format);  getitem_63 = None
    sub_178: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_231, getitem_65);  clone_231 = getitem_65 = None
    mul_688: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_178, rsqrt_21);  sub_178 = None
    mul_689: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_230, primals_107);  primals_107 = None
    mul_690: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_689, 768)
    sum_217: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_689, [2], True)
    mul_691: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_689, mul_688);  mul_689 = None
    sum_218: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_691, [2], True);  mul_691 = None
    mul_692: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_688, sum_218);  sum_218 = None
    sub_179: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_690, sum_217);  mul_690 = sum_217 = None
    sub_180: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_179, mul_692);  sub_179 = mul_692 = None
    div_40: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    mul_693: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_180);  div_40 = sub_180 = None
    mul_694: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_230, mul_688);  mul_688 = None
    sum_219: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_694, [0, 1]);  mul_694 = None
    sum_220: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_230, [0, 1]);  clone_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_19: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_687, mul_693], 2);  mul_687 = mul_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_695: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_62, 0.7071067811865476)
    erf_49: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_695);  mul_695 = None
    add_269: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
    mul_696: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_269, 0.5);  add_269 = None
    mul_697: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_62, view_62)
    mul_698: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_697, -0.5);  mul_697 = None
    exp_19: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_698);  mul_698 = None
    mul_699: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_700: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_62, mul_699);  view_62 = mul_699 = None
    add_270: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_696, mul_700);  mul_696 = mul_700 = None
    mul_701: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_19, add_270);  cat_19 = add_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_359: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_701, [1568, 1536]);  mul_701 = None
    permute_432: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    mm_150: "f32[1568, 256]" = torch.ops.aten.mm.default(view_359, permute_432);  permute_432 = None
    permute_433: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_151: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_433, view_61);  permute_433 = view_61 = None
    permute_434: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_221: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[1536]" = torch.ops.aten.view.default(sum_221, [1536]);  sum_221 = None
    permute_435: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    view_361: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_150, [8, 196, 256]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_232: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_69, memory_format = torch.contiguous_format);  add_69 = None
    sub_181: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_232, getitem_61);  clone_232 = getitem_61 = None
    mul_702: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_181, rsqrt_20);  sub_181 = None
    mul_703: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_361, primals_103);  primals_103 = None
    mul_704: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_703, 256)
    sum_222: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_703, [2], True)
    mul_705: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_703, mul_702);  mul_703 = None
    sum_223: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_705, [2], True);  mul_705 = None
    mul_706: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_702, sum_223);  sum_223 = None
    sub_182: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_704, sum_222);  mul_704 = sum_222 = None
    sub_183: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_182, mul_706);  sub_182 = mul_706 = None
    div_41: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 256);  rsqrt_20 = None
    mul_707: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_41, sub_183);  div_41 = sub_183 = None
    mul_708: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_361, mul_702);  mul_702 = None
    sum_224: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_708, [0, 1]);  mul_708 = None
    sum_225: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_361, [0, 1]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_271: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_268, mul_707);  add_268 = mul_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_362: "f32[1568, 256]" = torch.ops.aten.view.default(add_271, [1568, 256])
    permute_436: "f32[256, 768]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    mm_152: "f32[1568, 768]" = torch.ops.aten.mm.default(view_362, permute_436);  permute_436 = None
    permute_437: "f32[256, 1568]" = torch.ops.aten.permute.default(view_362, [1, 0])
    mm_153: "f32[256, 768]" = torch.ops.aten.mm.default(permute_437, view_59);  permute_437 = view_59 = None
    permute_438: "f32[768, 256]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_226: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True);  view_362 = None
    view_363: "f32[256]" = torch.ops.aten.view.default(sum_226, [256]);  sum_226 = None
    permute_439: "f32[256, 768]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
    view_364: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_152, [8, 196, 768]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_709: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_364, getitem_56);  getitem_56 = None
    mul_710: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_364, permute_49);  view_364 = permute_49 = None
    permute_440: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_709, [0, 2, 1]);  mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_227: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_440, [0, 1], True)
    view_365: "f32[196]" = torch.ops.aten.view.default(sum_227, [196]);  sum_227 = None
    clone_233: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_440, memory_format = torch.contiguous_format);  permute_440 = None
    view_366: "f32[6144, 196]" = torch.ops.aten.view.default(clone_233, [6144, 196]);  clone_233 = None
    permute_441: "f32[196, 6144]" = torch.ops.aten.permute.default(view_366, [1, 0])
    mm_154: "f32[196, 196]" = torch.ops.aten.mm.default(permute_441, view_57);  permute_441 = view_57 = None
    permute_442: "f32[196, 196]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
    permute_443: "f32[196, 196]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_155: "f32[6144, 196]" = torch.ops.aten.mm.default(view_366, permute_443);  view_366 = permute_443 = None
    view_367: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_155, [8, 768, 196]);  mm_155 = None
    permute_444: "f32[196, 196]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    permute_445: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_367, [0, 2, 1]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_234: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_445, memory_format = torch.contiguous_format);  permute_445 = None
    clone_235: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_57, memory_format = torch.contiguous_format);  getitem_57 = None
    sub_184: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_235, getitem_59);  clone_235 = getitem_59 = None
    mul_711: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_184, rsqrt_19);  sub_184 = None
    mul_712: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_234, primals_97);  primals_97 = None
    mul_713: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_712, 768)
    sum_228: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_712, [2], True)
    mul_714: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_712, mul_711);  mul_712 = None
    sum_229: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_714, [2], True);  mul_714 = None
    mul_715: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_711, sum_229);  sum_229 = None
    sub_185: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_713, sum_228);  mul_713 = sum_228 = None
    sub_186: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_185, mul_715);  sub_185 = mul_715 = None
    div_42: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    mul_716: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_186);  div_42 = sub_186 = None
    mul_717: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_234, mul_711);  mul_711 = None
    sum_230: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 1]);  mul_717 = None
    sum_231: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_234, [0, 1]);  clone_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_20: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_710, mul_716], 2);  mul_710 = mul_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_718: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_56, 0.7071067811865476)
    erf_50: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_718);  mul_718 = None
    add_272: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
    mul_719: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_272, 0.5);  add_272 = None
    mul_720: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_56, view_56)
    mul_721: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_720, -0.5);  mul_720 = None
    exp_20: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_721);  mul_721 = None
    mul_722: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_723: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_56, mul_722);  view_56 = mul_722 = None
    add_273: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_719, mul_723);  mul_719 = mul_723 = None
    mul_724: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_20, add_273);  cat_20 = add_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_368: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_724, [1568, 1536]);  mul_724 = None
    permute_446: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_156: "f32[1568, 256]" = torch.ops.aten.mm.default(view_368, permute_446);  permute_446 = None
    permute_447: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_368, [1, 0])
    mm_157: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_447, view_55);  permute_447 = view_55 = None
    permute_448: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_232: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_368, [0], True);  view_368 = None
    view_369: "f32[1536]" = torch.ops.aten.view.default(sum_232, [1536]);  sum_232 = None
    permute_449: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_448, [1, 0]);  permute_448 = None
    view_370: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_156, [8, 196, 256]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_236: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format);  add_62 = None
    sub_187: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_236, getitem_55);  clone_236 = getitem_55 = None
    mul_725: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_187, rsqrt_18);  sub_187 = None
    mul_726: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_370, primals_93);  primals_93 = None
    mul_727: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_726, 256)
    sum_233: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_726, [2], True)
    mul_728: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_726, mul_725);  mul_726 = None
    sum_234: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_728, [2], True);  mul_728 = None
    mul_729: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_725, sum_234);  sum_234 = None
    sub_188: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_727, sum_233);  mul_727 = sum_233 = None
    sub_189: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_188, mul_729);  sub_188 = mul_729 = None
    div_43: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 256);  rsqrt_18 = None
    mul_730: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_43, sub_189);  div_43 = sub_189 = None
    mul_731: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_370, mul_725);  mul_725 = None
    sum_235: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_731, [0, 1]);  mul_731 = None
    sum_236: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_370, [0, 1]);  view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_274: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_271, mul_730);  add_271 = mul_730 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_371: "f32[1568, 256]" = torch.ops.aten.view.default(add_274, [1568, 256])
    permute_450: "f32[256, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_158: "f32[1568, 768]" = torch.ops.aten.mm.default(view_371, permute_450);  permute_450 = None
    permute_451: "f32[256, 1568]" = torch.ops.aten.permute.default(view_371, [1, 0])
    mm_159: "f32[256, 768]" = torch.ops.aten.mm.default(permute_451, view_53);  permute_451 = view_53 = None
    permute_452: "f32[768, 256]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_237: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_371, [0], True);  view_371 = None
    view_372: "f32[256]" = torch.ops.aten.view.default(sum_237, [256]);  sum_237 = None
    permute_453: "f32[256, 768]" = torch.ops.aten.permute.default(permute_452, [1, 0]);  permute_452 = None
    view_373: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_158, [8, 196, 768]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_732: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_373, getitem_50);  getitem_50 = None
    mul_733: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_373, permute_44);  view_373 = permute_44 = None
    permute_454: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_732, [0, 2, 1]);  mul_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_238: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_454, [0, 1], True)
    view_374: "f32[196]" = torch.ops.aten.view.default(sum_238, [196]);  sum_238 = None
    clone_237: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_454, memory_format = torch.contiguous_format);  permute_454 = None
    view_375: "f32[6144, 196]" = torch.ops.aten.view.default(clone_237, [6144, 196]);  clone_237 = None
    permute_455: "f32[196, 6144]" = torch.ops.aten.permute.default(view_375, [1, 0])
    mm_160: "f32[196, 196]" = torch.ops.aten.mm.default(permute_455, view_51);  permute_455 = view_51 = None
    permute_456: "f32[196, 196]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    permute_457: "f32[196, 196]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_161: "f32[6144, 196]" = torch.ops.aten.mm.default(view_375, permute_457);  view_375 = permute_457 = None
    view_376: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_161, [8, 768, 196]);  mm_161 = None
    permute_458: "f32[196, 196]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
    permute_459: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_376, [0, 2, 1]);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_238: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    clone_239: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_51, memory_format = torch.contiguous_format);  getitem_51 = None
    sub_190: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_239, getitem_53);  clone_239 = getitem_53 = None
    mul_734: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_190, rsqrt_17);  sub_190 = None
    mul_735: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_238, primals_87);  primals_87 = None
    mul_736: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_735, 768)
    sum_239: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_735, [2], True)
    mul_737: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_735, mul_734);  mul_735 = None
    sum_240: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_737, [2], True);  mul_737 = None
    mul_738: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_734, sum_240);  sum_240 = None
    sub_191: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_736, sum_239);  mul_736 = sum_239 = None
    sub_192: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_191, mul_738);  sub_191 = mul_738 = None
    div_44: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    mul_739: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_44, sub_192);  div_44 = sub_192 = None
    mul_740: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_238, mul_734);  mul_734 = None
    sum_241: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_740, [0, 1]);  mul_740 = None
    sum_242: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_238, [0, 1]);  clone_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_21: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_733, mul_739], 2);  mul_733 = mul_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_741: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476)
    erf_51: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_741);  mul_741 = None
    add_275: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
    mul_742: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_275, 0.5);  add_275 = None
    mul_743: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_50, view_50)
    mul_744: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_743, -0.5);  mul_743 = None
    exp_21: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_744);  mul_744 = None
    mul_745: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_746: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_50, mul_745);  view_50 = mul_745 = None
    add_276: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_742, mul_746);  mul_742 = mul_746 = None
    mul_747: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_21, add_276);  cat_21 = add_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_377: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_747, [1568, 1536]);  mul_747 = None
    permute_460: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_162: "f32[1568, 256]" = torch.ops.aten.mm.default(view_377, permute_460);  permute_460 = None
    permute_461: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_377, [1, 0])
    mm_163: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_461, view_49);  permute_461 = view_49 = None
    permute_462: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_243: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
    view_378: "f32[1536]" = torch.ops.aten.view.default(sum_243, [1536]);  sum_243 = None
    permute_463: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    view_379: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_162, [8, 196, 256]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_240: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_55, memory_format = torch.contiguous_format);  add_55 = None
    sub_193: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_240, getitem_49);  clone_240 = getitem_49 = None
    mul_748: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_193, rsqrt_16);  sub_193 = None
    mul_749: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_379, primals_83);  primals_83 = None
    mul_750: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_749, 256)
    sum_244: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True)
    mul_751: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_749, mul_748);  mul_749 = None
    sum_245: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_751, [2], True);  mul_751 = None
    mul_752: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_748, sum_245);  sum_245 = None
    sub_194: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_750, sum_244);  mul_750 = sum_244 = None
    sub_195: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_194, mul_752);  sub_194 = mul_752 = None
    div_45: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 256);  rsqrt_16 = None
    mul_753: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_45, sub_195);  div_45 = sub_195 = None
    mul_754: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_379, mul_748);  mul_748 = None
    sum_246: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_754, [0, 1]);  mul_754 = None
    sum_247: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_379, [0, 1]);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_277: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_274, mul_753);  add_274 = mul_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_380: "f32[1568, 256]" = torch.ops.aten.view.default(add_277, [1568, 256])
    permute_464: "f32[256, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    mm_164: "f32[1568, 768]" = torch.ops.aten.mm.default(view_380, permute_464);  permute_464 = None
    permute_465: "f32[256, 1568]" = torch.ops.aten.permute.default(view_380, [1, 0])
    mm_165: "f32[256, 768]" = torch.ops.aten.mm.default(permute_465, view_47);  permute_465 = view_47 = None
    permute_466: "f32[768, 256]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_248: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
    view_381: "f32[256]" = torch.ops.aten.view.default(sum_248, [256]);  sum_248 = None
    permute_467: "f32[256, 768]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    view_382: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_164, [8, 196, 768]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_755: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_382, getitem_44);  getitem_44 = None
    mul_756: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_382, permute_39);  view_382 = permute_39 = None
    permute_468: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_755, [0, 2, 1]);  mul_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_249: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_468, [0, 1], True)
    view_383: "f32[196]" = torch.ops.aten.view.default(sum_249, [196]);  sum_249 = None
    clone_241: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
    view_384: "f32[6144, 196]" = torch.ops.aten.view.default(clone_241, [6144, 196]);  clone_241 = None
    permute_469: "f32[196, 6144]" = torch.ops.aten.permute.default(view_384, [1, 0])
    mm_166: "f32[196, 196]" = torch.ops.aten.mm.default(permute_469, view_45);  permute_469 = view_45 = None
    permute_470: "f32[196, 196]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    permute_471: "f32[196, 196]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    mm_167: "f32[6144, 196]" = torch.ops.aten.mm.default(view_384, permute_471);  view_384 = permute_471 = None
    view_385: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_167, [8, 768, 196]);  mm_167 = None
    permute_472: "f32[196, 196]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    permute_473: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_242: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_473, memory_format = torch.contiguous_format);  permute_473 = None
    clone_243: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_45, memory_format = torch.contiguous_format);  getitem_45 = None
    sub_196: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_243, getitem_47);  clone_243 = getitem_47 = None
    mul_757: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_196, rsqrt_15);  sub_196 = None
    mul_758: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_242, primals_77);  primals_77 = None
    mul_759: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_758, 768)
    sum_250: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_758, [2], True)
    mul_760: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_758, mul_757);  mul_758 = None
    sum_251: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_760, [2], True);  mul_760 = None
    mul_761: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_757, sum_251);  sum_251 = None
    sub_197: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_759, sum_250);  mul_759 = sum_250 = None
    sub_198: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_197, mul_761);  sub_197 = mul_761 = None
    div_46: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    mul_762: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_198);  div_46 = sub_198 = None
    mul_763: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_242, mul_757);  mul_757 = None
    sum_252: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_763, [0, 1]);  mul_763 = None
    sum_253: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_242, [0, 1]);  clone_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_22: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_756, mul_762], 2);  mul_756 = mul_762 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_764: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_44, 0.7071067811865476)
    erf_52: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_764);  mul_764 = None
    add_278: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
    mul_765: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_278, 0.5);  add_278 = None
    mul_766: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_44, view_44)
    mul_767: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_766, -0.5);  mul_766 = None
    exp_22: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_767);  mul_767 = None
    mul_768: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_769: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_44, mul_768);  view_44 = mul_768 = None
    add_279: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_765, mul_769);  mul_765 = mul_769 = None
    mul_770: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_22, add_279);  cat_22 = add_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_386: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_770, [1568, 1536]);  mul_770 = None
    permute_474: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_168: "f32[1568, 256]" = torch.ops.aten.mm.default(view_386, permute_474);  permute_474 = None
    permute_475: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_386, [1, 0])
    mm_169: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_475, view_43);  permute_475 = view_43 = None
    permute_476: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    sum_254: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
    view_387: "f32[1536]" = torch.ops.aten.view.default(sum_254, [1536]);  sum_254 = None
    permute_477: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_476, [1, 0]);  permute_476 = None
    view_388: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_168, [8, 196, 256]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_244: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_48, memory_format = torch.contiguous_format);  add_48 = None
    sub_199: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_244, getitem_43);  clone_244 = getitem_43 = None
    mul_771: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_199, rsqrt_14);  sub_199 = None
    mul_772: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_388, primals_73);  primals_73 = None
    mul_773: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_772, 256)
    sum_255: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_772, [2], True)
    mul_774: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_772, mul_771);  mul_772 = None
    sum_256: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_774, [2], True);  mul_774 = None
    mul_775: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_771, sum_256);  sum_256 = None
    sub_200: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_773, sum_255);  mul_773 = sum_255 = None
    sub_201: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_200, mul_775);  sub_200 = mul_775 = None
    div_47: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 256);  rsqrt_14 = None
    mul_776: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_47, sub_201);  div_47 = sub_201 = None
    mul_777: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_388, mul_771);  mul_771 = None
    sum_257: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 1]);  mul_777 = None
    sum_258: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_388, [0, 1]);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_280: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_277, mul_776);  add_277 = mul_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_389: "f32[1568, 256]" = torch.ops.aten.view.default(add_280, [1568, 256])
    permute_478: "f32[256, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_170: "f32[1568, 768]" = torch.ops.aten.mm.default(view_389, permute_478);  permute_478 = None
    permute_479: "f32[256, 1568]" = torch.ops.aten.permute.default(view_389, [1, 0])
    mm_171: "f32[256, 768]" = torch.ops.aten.mm.default(permute_479, view_41);  permute_479 = view_41 = None
    permute_480: "f32[768, 256]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_259: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
    view_390: "f32[256]" = torch.ops.aten.view.default(sum_259, [256]);  sum_259 = None
    permute_481: "f32[256, 768]" = torch.ops.aten.permute.default(permute_480, [1, 0]);  permute_480 = None
    view_391: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_170, [8, 196, 768]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_778: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_391, getitem_38);  getitem_38 = None
    mul_779: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_391, permute_34);  view_391 = permute_34 = None
    permute_482: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_778, [0, 2, 1]);  mul_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_260: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_482, [0, 1], True)
    view_392: "f32[196]" = torch.ops.aten.view.default(sum_260, [196]);  sum_260 = None
    clone_245: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_482, memory_format = torch.contiguous_format);  permute_482 = None
    view_393: "f32[6144, 196]" = torch.ops.aten.view.default(clone_245, [6144, 196]);  clone_245 = None
    permute_483: "f32[196, 6144]" = torch.ops.aten.permute.default(view_393, [1, 0])
    mm_172: "f32[196, 196]" = torch.ops.aten.mm.default(permute_483, view_39);  permute_483 = view_39 = None
    permute_484: "f32[196, 196]" = torch.ops.aten.permute.default(mm_172, [1, 0]);  mm_172 = None
    permute_485: "f32[196, 196]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_173: "f32[6144, 196]" = torch.ops.aten.mm.default(view_393, permute_485);  view_393 = permute_485 = None
    view_394: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_173, [8, 768, 196]);  mm_173 = None
    permute_486: "f32[196, 196]" = torch.ops.aten.permute.default(permute_484, [1, 0]);  permute_484 = None
    permute_487: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_394, [0, 2, 1]);  view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_246: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
    clone_247: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_39, memory_format = torch.contiguous_format);  getitem_39 = None
    sub_202: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_247, getitem_41);  clone_247 = getitem_41 = None
    mul_780: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_202, rsqrt_13);  sub_202 = None
    mul_781: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_246, primals_67);  primals_67 = None
    mul_782: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_781, 768)
    sum_261: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_781, [2], True)
    mul_783: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_781, mul_780);  mul_781 = None
    sum_262: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_783, [2], True);  mul_783 = None
    mul_784: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_780, sum_262);  sum_262 = None
    sub_203: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_782, sum_261);  mul_782 = sum_261 = None
    sub_204: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_203, mul_784);  sub_203 = mul_784 = None
    div_48: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    mul_785: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_204);  div_48 = sub_204 = None
    mul_786: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_246, mul_780);  mul_780 = None
    sum_263: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_786, [0, 1]);  mul_786 = None
    sum_264: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_246, [0, 1]);  clone_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_23: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_779, mul_785], 2);  mul_779 = mul_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_787: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_53: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_787);  mul_787 = None
    add_281: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
    mul_788: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_281, 0.5);  add_281 = None
    mul_789: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_38, view_38)
    mul_790: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_789, -0.5);  mul_789 = None
    exp_23: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_790);  mul_790 = None
    mul_791: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_792: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_38, mul_791);  view_38 = mul_791 = None
    add_282: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_788, mul_792);  mul_788 = mul_792 = None
    mul_793: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_23, add_282);  cat_23 = add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_395: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_793, [1568, 1536]);  mul_793 = None
    permute_488: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_174: "f32[1568, 256]" = torch.ops.aten.mm.default(view_395, permute_488);  permute_488 = None
    permute_489: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_395, [1, 0])
    mm_175: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_489, view_37);  permute_489 = view_37 = None
    permute_490: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_265: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
    view_396: "f32[1536]" = torch.ops.aten.view.default(sum_265, [1536]);  sum_265 = None
    permute_491: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_490, [1, 0]);  permute_490 = None
    view_397: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_174, [8, 196, 256]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_248: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format);  add_41 = None
    sub_205: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_248, getitem_37);  clone_248 = getitem_37 = None
    mul_794: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_205, rsqrt_12);  sub_205 = None
    mul_795: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_397, primals_63);  primals_63 = None
    mul_796: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_795, 256)
    sum_266: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_795, [2], True)
    mul_797: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_795, mul_794);  mul_795 = None
    sum_267: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_797, [2], True);  mul_797 = None
    mul_798: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_794, sum_267);  sum_267 = None
    sub_206: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_796, sum_266);  mul_796 = sum_266 = None
    sub_207: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_206, mul_798);  sub_206 = mul_798 = None
    div_49: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 256);  rsqrt_12 = None
    mul_799: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_49, sub_207);  div_49 = sub_207 = None
    mul_800: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_397, mul_794);  mul_794 = None
    sum_268: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_800, [0, 1]);  mul_800 = None
    sum_269: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_397, [0, 1]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_283: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_280, mul_799);  add_280 = mul_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_398: "f32[1568, 256]" = torch.ops.aten.view.default(add_283, [1568, 256])
    permute_492: "f32[256, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_176: "f32[1568, 768]" = torch.ops.aten.mm.default(view_398, permute_492);  permute_492 = None
    permute_493: "f32[256, 1568]" = torch.ops.aten.permute.default(view_398, [1, 0])
    mm_177: "f32[256, 768]" = torch.ops.aten.mm.default(permute_493, view_35);  permute_493 = view_35 = None
    permute_494: "f32[768, 256]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_270: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_398, [0], True);  view_398 = None
    view_399: "f32[256]" = torch.ops.aten.view.default(sum_270, [256]);  sum_270 = None
    permute_495: "f32[256, 768]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_400: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_176, [8, 196, 768]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_801: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_400, getitem_32);  getitem_32 = None
    mul_802: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_400, permute_29);  view_400 = permute_29 = None
    permute_496: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_801, [0, 2, 1]);  mul_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_271: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_496, [0, 1], True)
    view_401: "f32[196]" = torch.ops.aten.view.default(sum_271, [196]);  sum_271 = None
    clone_249: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_496, memory_format = torch.contiguous_format);  permute_496 = None
    view_402: "f32[6144, 196]" = torch.ops.aten.view.default(clone_249, [6144, 196]);  clone_249 = None
    permute_497: "f32[196, 6144]" = torch.ops.aten.permute.default(view_402, [1, 0])
    mm_178: "f32[196, 196]" = torch.ops.aten.mm.default(permute_497, view_33);  permute_497 = view_33 = None
    permute_498: "f32[196, 196]" = torch.ops.aten.permute.default(mm_178, [1, 0]);  mm_178 = None
    permute_499: "f32[196, 196]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    mm_179: "f32[6144, 196]" = torch.ops.aten.mm.default(view_402, permute_499);  view_402 = permute_499 = None
    view_403: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_179, [8, 768, 196]);  mm_179 = None
    permute_500: "f32[196, 196]" = torch.ops.aten.permute.default(permute_498, [1, 0]);  permute_498 = None
    permute_501: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_403, [0, 2, 1]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_250: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_501, memory_format = torch.contiguous_format);  permute_501 = None
    clone_251: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_33, memory_format = torch.contiguous_format);  getitem_33 = None
    sub_208: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_251, getitem_35);  clone_251 = getitem_35 = None
    mul_803: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_208, rsqrt_11);  sub_208 = None
    mul_804: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_250, primals_57);  primals_57 = None
    mul_805: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_804, 768)
    sum_272: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_804, [2], True)
    mul_806: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_804, mul_803);  mul_804 = None
    sum_273: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_806, [2], True);  mul_806 = None
    mul_807: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_803, sum_273);  sum_273 = None
    sub_209: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_805, sum_272);  mul_805 = sum_272 = None
    sub_210: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_209, mul_807);  sub_209 = mul_807 = None
    div_50: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_808: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_50, sub_210);  div_50 = sub_210 = None
    mul_809: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_250, mul_803);  mul_803 = None
    sum_274: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_809, [0, 1]);  mul_809 = None
    sum_275: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_250, [0, 1]);  clone_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_24: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_802, mul_808], 2);  mul_802 = mul_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_810: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_32, 0.7071067811865476)
    erf_54: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_810);  mul_810 = None
    add_284: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
    mul_811: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_284, 0.5);  add_284 = None
    mul_812: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_32, view_32)
    mul_813: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_812, -0.5);  mul_812 = None
    exp_24: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_813);  mul_813 = None
    mul_814: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_815: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_32, mul_814);  view_32 = mul_814 = None
    add_285: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_811, mul_815);  mul_811 = mul_815 = None
    mul_816: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_24, add_285);  cat_24 = add_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_404: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_816, [1568, 1536]);  mul_816 = None
    permute_502: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    mm_180: "f32[1568, 256]" = torch.ops.aten.mm.default(view_404, permute_502);  permute_502 = None
    permute_503: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_404, [1, 0])
    mm_181: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_503, view_31);  permute_503 = view_31 = None
    permute_504: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_276: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_404, [0], True);  view_404 = None
    view_405: "f32[1536]" = torch.ops.aten.view.default(sum_276, [1536]);  sum_276 = None
    permute_505: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    view_406: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_180, [8, 196, 256]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_252: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_34, memory_format = torch.contiguous_format);  add_34 = None
    sub_211: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_252, getitem_31);  clone_252 = getitem_31 = None
    mul_817: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_211, rsqrt_10);  sub_211 = None
    mul_818: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_406, primals_53);  primals_53 = None
    mul_819: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_818, 256)
    sum_277: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_818, [2], True)
    mul_820: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_818, mul_817);  mul_818 = None
    sum_278: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_820, [2], True);  mul_820 = None
    mul_821: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_817, sum_278);  sum_278 = None
    sub_212: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_819, sum_277);  mul_819 = sum_277 = None
    sub_213: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_212, mul_821);  sub_212 = mul_821 = None
    div_51: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 256);  rsqrt_10 = None
    mul_822: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_51, sub_213);  div_51 = sub_213 = None
    mul_823: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_406, mul_817);  mul_817 = None
    sum_279: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_823, [0, 1]);  mul_823 = None
    sum_280: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_406, [0, 1]);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_286: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_283, mul_822);  add_283 = mul_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_407: "f32[1568, 256]" = torch.ops.aten.view.default(add_286, [1568, 256])
    permute_506: "f32[256, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_182: "f32[1568, 768]" = torch.ops.aten.mm.default(view_407, permute_506);  permute_506 = None
    permute_507: "f32[256, 1568]" = torch.ops.aten.permute.default(view_407, [1, 0])
    mm_183: "f32[256, 768]" = torch.ops.aten.mm.default(permute_507, view_29);  permute_507 = view_29 = None
    permute_508: "f32[768, 256]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_281: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_407, [0], True);  view_407 = None
    view_408: "f32[256]" = torch.ops.aten.view.default(sum_281, [256]);  sum_281 = None
    permute_509: "f32[256, 768]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    view_409: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_182, [8, 196, 768]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_824: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_409, getitem_26);  getitem_26 = None
    mul_825: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_409, permute_24);  view_409 = permute_24 = None
    permute_510: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_824, [0, 2, 1]);  mul_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_282: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_510, [0, 1], True)
    view_410: "f32[196]" = torch.ops.aten.view.default(sum_282, [196]);  sum_282 = None
    clone_253: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_510, memory_format = torch.contiguous_format);  permute_510 = None
    view_411: "f32[6144, 196]" = torch.ops.aten.view.default(clone_253, [6144, 196]);  clone_253 = None
    permute_511: "f32[196, 6144]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_184: "f32[196, 196]" = torch.ops.aten.mm.default(permute_511, view_27);  permute_511 = view_27 = None
    permute_512: "f32[196, 196]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    permute_513: "f32[196, 196]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_185: "f32[6144, 196]" = torch.ops.aten.mm.default(view_411, permute_513);  view_411 = permute_513 = None
    view_412: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_185, [8, 768, 196]);  mm_185 = None
    permute_514: "f32[196, 196]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    permute_515: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_412, [0, 2, 1]);  view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_254: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_515, memory_format = torch.contiguous_format);  permute_515 = None
    clone_255: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_27, memory_format = torch.contiguous_format);  getitem_27 = None
    sub_214: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_255, getitem_29);  clone_255 = getitem_29 = None
    mul_826: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_214, rsqrt_9);  sub_214 = None
    mul_827: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_254, primals_47);  primals_47 = None
    mul_828: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_827, 768)
    sum_283: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_827, [2], True)
    mul_829: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_827, mul_826);  mul_827 = None
    sum_284: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_829, [2], True);  mul_829 = None
    mul_830: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_826, sum_284);  sum_284 = None
    sub_215: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_828, sum_283);  mul_828 = sum_283 = None
    sub_216: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_215, mul_830);  sub_215 = mul_830 = None
    div_52: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_831: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_216);  div_52 = sub_216 = None
    mul_832: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_254, mul_826);  mul_826 = None
    sum_285: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_832, [0, 1]);  mul_832 = None
    sum_286: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_254, [0, 1]);  clone_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_25: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_825, mul_831], 2);  mul_825 = mul_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_833: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_26, 0.7071067811865476)
    erf_55: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_833);  mul_833 = None
    add_287: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
    mul_834: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_287, 0.5);  add_287 = None
    mul_835: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_26, view_26)
    mul_836: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_835, -0.5);  mul_835 = None
    exp_25: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_836);  mul_836 = None
    mul_837: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_838: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_26, mul_837);  view_26 = mul_837 = None
    add_288: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_834, mul_838);  mul_834 = mul_838 = None
    mul_839: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_25, add_288);  cat_25 = add_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_413: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_839, [1568, 1536]);  mul_839 = None
    permute_516: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_186: "f32[1568, 256]" = torch.ops.aten.mm.default(view_413, permute_516);  permute_516 = None
    permute_517: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_413, [1, 0])
    mm_187: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_517, view_25);  permute_517 = view_25 = None
    permute_518: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_287: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_413, [0], True);  view_413 = None
    view_414: "f32[1536]" = torch.ops.aten.view.default(sum_287, [1536]);  sum_287 = None
    permute_519: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_518, [1, 0]);  permute_518 = None
    view_415: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_186, [8, 196, 256]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_256: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_27, memory_format = torch.contiguous_format);  add_27 = None
    sub_217: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_256, getitem_25);  clone_256 = getitem_25 = None
    mul_840: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_217, rsqrt_8);  sub_217 = None
    mul_841: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_415, primals_43);  primals_43 = None
    mul_842: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_841, 256)
    sum_288: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_841, [2], True)
    mul_843: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_841, mul_840);  mul_841 = None
    sum_289: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_843, [2], True);  mul_843 = None
    mul_844: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_840, sum_289);  sum_289 = None
    sub_218: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_842, sum_288);  mul_842 = sum_288 = None
    sub_219: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_218, mul_844);  sub_218 = mul_844 = None
    div_53: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None
    mul_845: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_53, sub_219);  div_53 = sub_219 = None
    mul_846: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_415, mul_840);  mul_840 = None
    sum_290: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_846, [0, 1]);  mul_846 = None
    sum_291: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_415, [0, 1]);  view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_289: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_286, mul_845);  add_286 = mul_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_416: "f32[1568, 256]" = torch.ops.aten.view.default(add_289, [1568, 256])
    permute_520: "f32[256, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_188: "f32[1568, 768]" = torch.ops.aten.mm.default(view_416, permute_520);  permute_520 = None
    permute_521: "f32[256, 1568]" = torch.ops.aten.permute.default(view_416, [1, 0])
    mm_189: "f32[256, 768]" = torch.ops.aten.mm.default(permute_521, view_23);  permute_521 = view_23 = None
    permute_522: "f32[768, 256]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_292: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
    view_417: "f32[256]" = torch.ops.aten.view.default(sum_292, [256]);  sum_292 = None
    permute_523: "f32[256, 768]" = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
    view_418: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_188, [8, 196, 768]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_847: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_418, getitem_20);  getitem_20 = None
    mul_848: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_418, permute_19);  view_418 = permute_19 = None
    permute_524: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_847, [0, 2, 1]);  mul_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_293: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_524, [0, 1], True)
    view_419: "f32[196]" = torch.ops.aten.view.default(sum_293, [196]);  sum_293 = None
    clone_257: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_420: "f32[6144, 196]" = torch.ops.aten.view.default(clone_257, [6144, 196]);  clone_257 = None
    permute_525: "f32[196, 6144]" = torch.ops.aten.permute.default(view_420, [1, 0])
    mm_190: "f32[196, 196]" = torch.ops.aten.mm.default(permute_525, view_21);  permute_525 = view_21 = None
    permute_526: "f32[196, 196]" = torch.ops.aten.permute.default(mm_190, [1, 0]);  mm_190 = None
    permute_527: "f32[196, 196]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_191: "f32[6144, 196]" = torch.ops.aten.mm.default(view_420, permute_527);  view_420 = permute_527 = None
    view_421: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_191, [8, 768, 196]);  mm_191 = None
    permute_528: "f32[196, 196]" = torch.ops.aten.permute.default(permute_526, [1, 0]);  permute_526 = None
    permute_529: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_421, [0, 2, 1]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_258: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_529, memory_format = torch.contiguous_format);  permute_529 = None
    clone_259: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_21, memory_format = torch.contiguous_format);  getitem_21 = None
    sub_220: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_259, getitem_23);  clone_259 = getitem_23 = None
    mul_849: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_220, rsqrt_7);  sub_220 = None
    mul_850: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_258, primals_37);  primals_37 = None
    mul_851: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_850, 768)
    sum_294: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_850, [2], True)
    mul_852: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_850, mul_849);  mul_850 = None
    sum_295: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_852, [2], True);  mul_852 = None
    mul_853: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_849, sum_295);  sum_295 = None
    sub_221: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_851, sum_294);  mul_851 = sum_294 = None
    sub_222: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_221, mul_853);  sub_221 = mul_853 = None
    div_54: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_854: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_222);  div_54 = sub_222 = None
    mul_855: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_258, mul_849);  mul_849 = None
    sum_296: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_855, [0, 1]);  mul_855 = None
    sum_297: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_258, [0, 1]);  clone_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_26: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_848, mul_854], 2);  mul_848 = mul_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_856: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_20, 0.7071067811865476)
    erf_56: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_856);  mul_856 = None
    add_290: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
    mul_857: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_290, 0.5);  add_290 = None
    mul_858: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_20, view_20)
    mul_859: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_858, -0.5);  mul_858 = None
    exp_26: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_859);  mul_859 = None
    mul_860: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_861: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_20, mul_860);  view_20 = mul_860 = None
    add_291: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_857, mul_861);  mul_857 = mul_861 = None
    mul_862: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_26, add_291);  cat_26 = add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_422: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_862, [1568, 1536]);  mul_862 = None
    permute_530: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    mm_192: "f32[1568, 256]" = torch.ops.aten.mm.default(view_422, permute_530);  permute_530 = None
    permute_531: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_422, [1, 0])
    mm_193: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_531, view_19);  permute_531 = view_19 = None
    permute_532: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_298: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
    view_423: "f32[1536]" = torch.ops.aten.view.default(sum_298, [1536]);  sum_298 = None
    permute_533: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    view_424: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_192, [8, 196, 256]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_260: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format);  add_20 = None
    sub_223: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_260, getitem_19);  clone_260 = getitem_19 = None
    mul_863: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_223, rsqrt_6);  sub_223 = None
    mul_864: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_424, primals_33);  primals_33 = None
    mul_865: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_864, 256)
    sum_299: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_864, [2], True)
    mul_866: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_864, mul_863);  mul_864 = None
    sum_300: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_866, [2], True);  mul_866 = None
    mul_867: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_863, sum_300);  sum_300 = None
    sub_224: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_865, sum_299);  mul_865 = sum_299 = None
    sub_225: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_224, mul_867);  sub_224 = mul_867 = None
    div_55: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
    mul_868: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_55, sub_225);  div_55 = sub_225 = None
    mul_869: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_424, mul_863);  mul_863 = None
    sum_301: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_869, [0, 1]);  mul_869 = None
    sum_302: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_424, [0, 1]);  view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_292: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_289, mul_868);  add_289 = mul_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_425: "f32[1568, 256]" = torch.ops.aten.view.default(add_292, [1568, 256])
    permute_534: "f32[256, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    mm_194: "f32[1568, 768]" = torch.ops.aten.mm.default(view_425, permute_534);  permute_534 = None
    permute_535: "f32[256, 1568]" = torch.ops.aten.permute.default(view_425, [1, 0])
    mm_195: "f32[256, 768]" = torch.ops.aten.mm.default(permute_535, view_17);  permute_535 = view_17 = None
    permute_536: "f32[768, 256]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_303: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
    view_426: "f32[256]" = torch.ops.aten.view.default(sum_303, [256]);  sum_303 = None
    permute_537: "f32[256, 768]" = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
    view_427: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_194, [8, 196, 768]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_870: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_427, getitem_14);  getitem_14 = None
    mul_871: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_427, permute_14);  view_427 = permute_14 = None
    permute_538: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_870, [0, 2, 1]);  mul_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_304: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_538, [0, 1], True)
    view_428: "f32[196]" = torch.ops.aten.view.default(sum_304, [196]);  sum_304 = None
    clone_261: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_538, memory_format = torch.contiguous_format);  permute_538 = None
    view_429: "f32[6144, 196]" = torch.ops.aten.view.default(clone_261, [6144, 196]);  clone_261 = None
    permute_539: "f32[196, 6144]" = torch.ops.aten.permute.default(view_429, [1, 0])
    mm_196: "f32[196, 196]" = torch.ops.aten.mm.default(permute_539, view_15);  permute_539 = view_15 = None
    permute_540: "f32[196, 196]" = torch.ops.aten.permute.default(mm_196, [1, 0]);  mm_196 = None
    permute_541: "f32[196, 196]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_197: "f32[6144, 196]" = torch.ops.aten.mm.default(view_429, permute_541);  view_429 = permute_541 = None
    view_430: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_197, [8, 768, 196]);  mm_197 = None
    permute_542: "f32[196, 196]" = torch.ops.aten.permute.default(permute_540, [1, 0]);  permute_540 = None
    permute_543: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_430, [0, 2, 1]);  view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_262: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_543, memory_format = torch.contiguous_format);  permute_543 = None
    clone_263: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_15, memory_format = torch.contiguous_format);  getitem_15 = None
    sub_226: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_263, getitem_17);  clone_263 = getitem_17 = None
    mul_872: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_226, rsqrt_5);  sub_226 = None
    mul_873: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_262, primals_27);  primals_27 = None
    mul_874: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_873, 768)
    sum_305: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_873, [2], True)
    mul_875: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_873, mul_872);  mul_873 = None
    sum_306: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_875, [2], True);  mul_875 = None
    mul_876: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_872, sum_306);  sum_306 = None
    sub_227: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_874, sum_305);  mul_874 = sum_305 = None
    sub_228: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_227, mul_876);  sub_227 = mul_876 = None
    div_56: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_877: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_56, sub_228);  div_56 = sub_228 = None
    mul_878: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_262, mul_872);  mul_872 = None
    sum_307: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 1]);  mul_878 = None
    sum_308: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_262, [0, 1]);  clone_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_27: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_871, mul_877], 2);  mul_871 = mul_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_879: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476)
    erf_57: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_879);  mul_879 = None
    add_293: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
    mul_880: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_293, 0.5);  add_293 = None
    mul_881: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_14, view_14)
    mul_882: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_881, -0.5);  mul_881 = None
    exp_27: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_882);  mul_882 = None
    mul_883: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_884: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_14, mul_883);  view_14 = mul_883 = None
    add_294: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_880, mul_884);  mul_880 = mul_884 = None
    mul_885: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_27, add_294);  cat_27 = add_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_431: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_885, [1568, 1536]);  mul_885 = None
    permute_544: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_198: "f32[1568, 256]" = torch.ops.aten.mm.default(view_431, permute_544);  permute_544 = None
    permute_545: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_431, [1, 0])
    mm_199: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_545, view_13);  permute_545 = view_13 = None
    permute_546: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_309: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_431, [0], True);  view_431 = None
    view_432: "f32[1536]" = torch.ops.aten.view.default(sum_309, [1536]);  sum_309 = None
    permute_547: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_546, [1, 0]);  permute_546 = None
    view_433: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_198, [8, 196, 256]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_264: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_13, memory_format = torch.contiguous_format);  add_13 = None
    sub_229: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_264, getitem_13);  clone_264 = getitem_13 = None
    mul_886: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_229, rsqrt_4);  sub_229 = None
    mul_887: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_433, primals_23);  primals_23 = None
    mul_888: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_887, 256)
    sum_310: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_887, [2], True)
    mul_889: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_887, mul_886);  mul_887 = None
    sum_311: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_889, [2], True);  mul_889 = None
    mul_890: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_886, sum_311);  sum_311 = None
    sub_230: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_888, sum_310);  mul_888 = sum_310 = None
    sub_231: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_230, mul_890);  sub_230 = mul_890 = None
    div_57: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 256);  rsqrt_4 = None
    mul_891: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_57, sub_231);  div_57 = sub_231 = None
    mul_892: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_433, mul_886);  mul_886 = None
    sum_312: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 1]);  mul_892 = None
    sum_313: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_433, [0, 1]);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_295: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_292, mul_891);  add_292 = mul_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_434: "f32[1568, 256]" = torch.ops.aten.view.default(add_295, [1568, 256])
    permute_548: "f32[256, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_200: "f32[1568, 768]" = torch.ops.aten.mm.default(view_434, permute_548);  permute_548 = None
    permute_549: "f32[256, 1568]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_201: "f32[256, 768]" = torch.ops.aten.mm.default(permute_549, view_11);  permute_549 = view_11 = None
    permute_550: "f32[768, 256]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    sum_314: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[256]" = torch.ops.aten.view.default(sum_314, [256]);  sum_314 = None
    permute_551: "f32[256, 768]" = torch.ops.aten.permute.default(permute_550, [1, 0]);  permute_550 = None
    view_436: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_200, [8, 196, 768]);  mm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_893: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_436, getitem_8);  getitem_8 = None
    mul_894: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_436, permute_9);  view_436 = permute_9 = None
    permute_552: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_893, [0, 2, 1]);  mul_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_315: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_552, [0, 1], True)
    view_437: "f32[196]" = torch.ops.aten.view.default(sum_315, [196]);  sum_315 = None
    clone_265: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_552, memory_format = torch.contiguous_format);  permute_552 = None
    view_438: "f32[6144, 196]" = torch.ops.aten.view.default(clone_265, [6144, 196]);  clone_265 = None
    permute_553: "f32[196, 6144]" = torch.ops.aten.permute.default(view_438, [1, 0])
    mm_202: "f32[196, 196]" = torch.ops.aten.mm.default(permute_553, view_9);  permute_553 = view_9 = None
    permute_554: "f32[196, 196]" = torch.ops.aten.permute.default(mm_202, [1, 0]);  mm_202 = None
    permute_555: "f32[196, 196]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_203: "f32[6144, 196]" = torch.ops.aten.mm.default(view_438, permute_555);  view_438 = permute_555 = None
    view_439: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_203, [8, 768, 196]);  mm_203 = None
    permute_556: "f32[196, 196]" = torch.ops.aten.permute.default(permute_554, [1, 0]);  permute_554 = None
    permute_557: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_439, [0, 2, 1]);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_266: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_557, memory_format = torch.contiguous_format);  permute_557 = None
    clone_267: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_9, memory_format = torch.contiguous_format);  getitem_9 = None
    sub_232: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_267, getitem_11);  clone_267 = getitem_11 = None
    mul_895: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_232, rsqrt_3);  sub_232 = None
    mul_896: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_266, primals_17);  primals_17 = None
    mul_897: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_896, 768)
    sum_316: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_896, [2], True)
    mul_898: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_896, mul_895);  mul_896 = None
    sum_317: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_898, [2], True);  mul_898 = None
    mul_899: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_895, sum_317);  sum_317 = None
    sub_233: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_897, sum_316);  mul_897 = sum_316 = None
    sub_234: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_233, mul_899);  sub_233 = mul_899 = None
    div_58: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_900: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_234);  div_58 = sub_234 = None
    mul_901: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_266, mul_895);  mul_895 = None
    sum_318: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_901, [0, 1]);  mul_901 = None
    sum_319: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_266, [0, 1]);  clone_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_28: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_894, mul_900], 2);  mul_894 = mul_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_902: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476)
    erf_58: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_902);  mul_902 = None
    add_296: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
    mul_903: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_296, 0.5);  add_296 = None
    mul_904: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_8, view_8)
    mul_905: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_904, -0.5);  mul_904 = None
    exp_28: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_905);  mul_905 = None
    mul_906: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_907: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_8, mul_906);  view_8 = mul_906 = None
    add_297: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_903, mul_907);  mul_903 = mul_907 = None
    mul_908: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_28, add_297);  cat_28 = add_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_440: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_908, [1568, 1536]);  mul_908 = None
    permute_558: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    mm_204: "f32[1568, 256]" = torch.ops.aten.mm.default(view_440, permute_558);  permute_558 = None
    permute_559: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_440, [1, 0])
    mm_205: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_559, view_7);  permute_559 = view_7 = None
    permute_560: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_320: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
    view_441: "f32[1536]" = torch.ops.aten.view.default(sum_320, [1536]);  sum_320 = None
    permute_561: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_560, [1, 0]);  permute_560 = None
    view_442: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_204, [8, 196, 256]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_268: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_6, memory_format = torch.contiguous_format);  add_6 = None
    sub_235: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_268, getitem_7);  clone_268 = getitem_7 = None
    mul_909: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_235, rsqrt_2);  sub_235 = None
    mul_910: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_442, primals_13);  primals_13 = None
    mul_911: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_910, 256)
    sum_321: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_910, [2], True)
    mul_912: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_910, mul_909);  mul_910 = None
    sum_322: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_912, [2], True);  mul_912 = None
    mul_913: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_909, sum_322);  sum_322 = None
    sub_236: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_911, sum_321);  mul_911 = sum_321 = None
    sub_237: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_236, mul_913);  sub_236 = mul_913 = None
    div_59: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 256);  rsqrt_2 = None
    mul_914: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_59, sub_237);  div_59 = sub_237 = None
    mul_915: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_442, mul_909);  mul_909 = None
    sum_323: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_915, [0, 1]);  mul_915 = None
    sum_324: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_442, [0, 1]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_298: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_295, mul_914);  add_295 = mul_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_443: "f32[1568, 256]" = torch.ops.aten.view.default(add_298, [1568, 256])
    permute_562: "f32[256, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    mm_206: "f32[1568, 768]" = torch.ops.aten.mm.default(view_443, permute_562);  permute_562 = None
    permute_563: "f32[256, 1568]" = torch.ops.aten.permute.default(view_443, [1, 0])
    mm_207: "f32[256, 768]" = torch.ops.aten.mm.default(permute_563, view_5);  permute_563 = view_5 = None
    permute_564: "f32[768, 256]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    sum_325: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_443, [0], True);  view_443 = None
    view_444: "f32[256]" = torch.ops.aten.view.default(sum_325, [256]);  sum_325 = None
    permute_565: "f32[256, 768]" = torch.ops.aten.permute.default(permute_564, [1, 0]);  permute_564 = None
    view_445: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_206, [8, 196, 768]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_916: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_445, getitem_2);  getitem_2 = None
    mul_917: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_445, permute_4);  view_445 = permute_4 = None
    permute_566: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_916, [0, 2, 1]);  mul_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_326: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_566, [0, 1], True)
    view_446: "f32[196]" = torch.ops.aten.view.default(sum_326, [196]);  sum_326 = None
    clone_269: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_566, memory_format = torch.contiguous_format);  permute_566 = None
    view_447: "f32[6144, 196]" = torch.ops.aten.view.default(clone_269, [6144, 196]);  clone_269 = None
    permute_567: "f32[196, 6144]" = torch.ops.aten.permute.default(view_447, [1, 0])
    mm_208: "f32[196, 196]" = torch.ops.aten.mm.default(permute_567, view_3);  permute_567 = view_3 = None
    permute_568: "f32[196, 196]" = torch.ops.aten.permute.default(mm_208, [1, 0]);  mm_208 = None
    permute_569: "f32[196, 196]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_209: "f32[6144, 196]" = torch.ops.aten.mm.default(view_447, permute_569);  view_447 = permute_569 = None
    view_448: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_209, [8, 768, 196]);  mm_209 = None
    permute_570: "f32[196, 196]" = torch.ops.aten.permute.default(permute_568, [1, 0]);  permute_568 = None
    permute_571: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_448, [0, 2, 1]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_270: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_571, memory_format = torch.contiguous_format);  permute_571 = None
    clone_271: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_3, memory_format = torch.contiguous_format);  getitem_3 = None
    sub_238: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_271, getitem_5);  clone_271 = getitem_5 = None
    mul_918: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_238, rsqrt_1);  sub_238 = None
    mul_919: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_270, primals_7);  primals_7 = None
    mul_920: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_919, 768)
    sum_327: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_919, [2], True)
    mul_921: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_919, mul_918);  mul_919 = None
    sum_328: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_921, [2], True);  mul_921 = None
    mul_922: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_918, sum_328);  sum_328 = None
    sub_239: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_920, sum_327);  mul_920 = sum_327 = None
    sub_240: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_239, mul_922);  sub_239 = mul_922 = None
    div_60: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_923: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_240);  div_60 = sub_240 = None
    mul_924: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_270, mul_918);  mul_918 = None
    sum_329: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_924, [0, 1]);  mul_924 = None
    sum_330: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_270, [0, 1]);  clone_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_29: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_917, mul_923], 2);  mul_917 = mul_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_925: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_2, 0.7071067811865476)
    erf_59: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_925);  mul_925 = None
    add_299: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
    mul_926: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_299, 0.5);  add_299 = None
    mul_927: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_2, view_2)
    mul_928: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_927, -0.5);  mul_927 = None
    exp_29: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_928);  mul_928 = None
    mul_929: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_930: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_2, mul_929);  view_2 = mul_929 = None
    add_300: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_926, mul_930);  mul_926 = mul_930 = None
    mul_931: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_29, add_300);  cat_29 = add_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_449: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_931, [1568, 1536]);  mul_931 = None
    permute_572: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_210: "f32[1568, 256]" = torch.ops.aten.mm.default(view_449, permute_572);  permute_572 = None
    permute_573: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_449, [1, 0])
    mm_211: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_573, view_1);  permute_573 = view_1 = None
    permute_574: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    sum_331: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_449, [0], True);  view_449 = None
    view_450: "f32[1536]" = torch.ops.aten.view.default(sum_331, [1536]);  sum_331 = None
    permute_575: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_574, [1, 0]);  permute_574 = None
    view_451: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_210, [8, 196, 256]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone_272: "f32[8, 196, 256]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    sub_241: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_272, getitem_1);  clone_272 = getitem_1 = None
    mul_932: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_241, rsqrt);  sub_241 = None
    mul_933: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_451, primals_3);  primals_3 = None
    mul_934: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_933, 256)
    sum_332: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_933, [2], True)
    mul_935: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_933, mul_932);  mul_933 = None
    sum_333: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_935, [2], True);  mul_935 = None
    mul_936: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_932, sum_333);  sum_333 = None
    sub_242: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_934, sum_332);  mul_934 = sum_332 = None
    sub_243: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_242, mul_936);  sub_242 = mul_936 = None
    div_61: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt, 256);  rsqrt = None
    mul_937: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_61, sub_243);  div_61 = sub_243 = None
    mul_938: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_451, mul_932);  mul_932 = None
    sum_334: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_938, [0, 1]);  mul_938 = None
    sum_335: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_451, [0, 1]);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_301: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_298, mul_937);  add_298 = mul_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_576: "f32[8, 256, 196]" = torch.ops.aten.permute.default(add_301, [0, 2, 1]);  add_301 = None
    view_452: "f32[8, 256, 14, 14]" = torch.ops.aten.view.default(permute_576, [8, 256, 14, 14]);  permute_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(view_452, primals_307, primals_1, [256], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_452 = primals_307 = primals_1 = None
    getitem_183: "f32[256, 3, 16, 16]" = convolution_backward[1]
    getitem_184: "f32[256]" = convolution_backward[2];  convolution_backward = None
    return pytree.tree_unflatten([addmm_60, getitem_183, getitem_184, sum_334, sum_335, permute_575, view_450, sum_329, sum_330, permute_570, view_446, permute_565, view_444, sum_323, sum_324, permute_561, view_441, sum_318, sum_319, permute_556, view_437, permute_551, view_435, sum_312, sum_313, permute_547, view_432, sum_307, sum_308, permute_542, view_428, permute_537, view_426, sum_301, sum_302, permute_533, view_423, sum_296, sum_297, permute_528, view_419, permute_523, view_417, sum_290, sum_291, permute_519, view_414, sum_285, sum_286, permute_514, view_410, permute_509, view_408, sum_279, sum_280, permute_505, view_405, sum_274, sum_275, permute_500, view_401, permute_495, view_399, sum_268, sum_269, permute_491, view_396, sum_263, sum_264, permute_486, view_392, permute_481, view_390, sum_257, sum_258, permute_477, view_387, sum_252, sum_253, permute_472, view_383, permute_467, view_381, sum_246, sum_247, permute_463, view_378, sum_241, sum_242, permute_458, view_374, permute_453, view_372, sum_235, sum_236, permute_449, view_369, sum_230, sum_231, permute_444, view_365, permute_439, view_363, sum_224, sum_225, permute_435, view_360, sum_219, sum_220, permute_430, view_356, permute_425, view_354, sum_213, sum_214, permute_421, view_351, sum_208, sum_209, permute_416, view_347, permute_411, view_345, sum_202, sum_203, permute_407, view_342, sum_197, sum_198, permute_402, view_338, permute_397, view_336, sum_191, sum_192, permute_393, view_333, sum_186, sum_187, permute_388, view_329, permute_383, view_327, sum_180, sum_181, permute_379, view_324, sum_175, sum_176, permute_374, view_320, permute_369, view_318, sum_169, sum_170, permute_365, view_315, sum_164, sum_165, permute_360, view_311, permute_355, view_309, sum_158, sum_159, permute_351, view_306, sum_153, sum_154, permute_346, view_302, permute_341, view_300, sum_147, sum_148, permute_337, view_297, sum_142, sum_143, permute_332, view_293, permute_327, view_291, sum_136, sum_137, permute_323, view_288, sum_131, sum_132, permute_318, view_284, permute_313, view_282, sum_125, sum_126, permute_309, view_279, sum_120, sum_121, permute_304, view_275, permute_299, view_273, sum_114, sum_115, permute_295, view_270, sum_109, sum_110, permute_290, view_266, permute_285, view_264, sum_103, sum_104, permute_281, view_261, sum_98, sum_99, permute_276, view_257, permute_271, view_255, sum_92, sum_93, permute_267, view_252, sum_87, sum_88, permute_262, view_248, permute_257, view_246, sum_81, sum_82, permute_253, view_243, sum_76, sum_77, permute_248, view_239, permute_243, view_237, sum_70, sum_71, permute_239, view_234, sum_65, sum_66, permute_234, view_230, permute_229, view_228, sum_59, sum_60, permute_225, view_225, sum_54, sum_55, permute_220, view_221, permute_215, view_219, sum_48, sum_49, permute_211, view_216, sum_43, sum_44, permute_206, view_212, permute_201, view_210, sum_37, sum_38, permute_197, view_207, sum_32, sum_33, permute_192, view_203, permute_187, view_201, sum_26, sum_27, permute_183, view_198, sum_21, sum_22, permute_178, view_194, permute_173, view_192, sum_15, sum_16, permute_169, view_189, sum_10, sum_11, permute_164, view_185, permute_159, view_183, sum_4, sum_5, permute_155, view_181, None], self._out_spec)
    