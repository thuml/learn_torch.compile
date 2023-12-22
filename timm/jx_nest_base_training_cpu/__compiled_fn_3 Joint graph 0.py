from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[1, 16, 196, 128]"; primals_2: "f32[128]"; primals_3: "f32[128]"; primals_4: "f32[128]"; primals_5: "f32[128]"; primals_6: "f32[128]"; primals_7: "f32[128]"; primals_8: "f32[128]"; primals_9: "f32[128]"; primals_10: "f32[256]"; primals_11: "f32[256]"; primals_12: "f32[1, 4, 196, 256]"; primals_13: "f32[256]"; primals_14: "f32[256]"; primals_15: "f32[256]"; primals_16: "f32[256]"; primals_17: "f32[256]"; primals_18: "f32[256]"; primals_19: "f32[256]"; primals_20: "f32[256]"; primals_21: "f32[512]"; primals_22: "f32[512]"; primals_23: "f32[1, 1, 196, 512]"; primals_24: "f32[512]"; primals_25: "f32[512]"; primals_26: "f32[512]"; primals_27: "f32[512]"; primals_28: "f32[512]"; primals_29: "f32[512]"; primals_30: "f32[512]"; primals_31: "f32[512]"; primals_32: "f32[512]"; primals_33: "f32[512]"; primals_34: "f32[512]"; primals_35: "f32[512]"; primals_36: "f32[512]"; primals_37: "f32[512]"; primals_38: "f32[512]"; primals_39: "f32[512]"; primals_40: "f32[512]"; primals_41: "f32[512]"; primals_42: "f32[512]"; primals_43: "f32[512]"; primals_44: "f32[512]"; primals_45: "f32[512]"; primals_46: "f32[512]"; primals_47: "f32[512]"; primals_48: "f32[512]"; primals_49: "f32[512]"; primals_50: "f32[512]"; primals_51: "f32[512]"; primals_52: "f32[512]"; primals_53: "f32[512]"; primals_54: "f32[512]"; primals_55: "f32[512]"; primals_56: "f32[512]"; primals_57: "f32[512]"; primals_58: "f32[512]"; primals_59: "f32[512]"; primals_60: "f32[512]"; primals_61: "f32[512]"; primals_62: "f32[512]"; primals_63: "f32[512]"; primals_64: "f32[512]"; primals_65: "f32[512]"; primals_66: "f32[512]"; primals_67: "f32[512]"; primals_68: "f32[512]"; primals_69: "f32[512]"; primals_70: "f32[512]"; primals_71: "f32[512]"; primals_72: "f32[512]"; primals_73: "f32[512]"; primals_74: "f32[512]"; primals_75: "f32[512]"; primals_76: "f32[512]"; primals_77: "f32[512]"; primals_78: "f32[512]"; primals_79: "f32[512]"; primals_80: "f32[512]"; primals_81: "f32[512]"; primals_82: "f32[512]"; primals_83: "f32[512]"; primals_84: "f32[512]"; primals_85: "f32[512]"; primals_86: "f32[512]"; primals_87: "f32[512]"; primals_88: "f32[512]"; primals_89: "f32[512]"; primals_90: "f32[512]"; primals_91: "f32[512]"; primals_92: "f32[512]"; primals_93: "f32[512]"; primals_94: "f32[512]"; primals_95: "f32[512]"; primals_96: "f32[512]"; primals_97: "f32[512]"; primals_98: "f32[512]"; primals_99: "f32[512]"; primals_100: "f32[512]"; primals_101: "f32[512]"; primals_102: "f32[512]"; primals_103: "f32[512]"; primals_104: "f32[512]"; primals_105: "f32[512]"; primals_106: "f32[128, 3, 4, 4]"; primals_107: "f32[128]"; primals_108: "f32[384, 128]"; primals_109: "f32[384]"; primals_110: "f32[128, 128]"; primals_111: "f32[128]"; primals_112: "f32[512, 128]"; primals_113: "f32[512]"; primals_114: "f32[128, 512]"; primals_115: "f32[128]"; primals_116: "f32[384, 128]"; primals_117: "f32[384]"; primals_118: "f32[128, 128]"; primals_119: "f32[128]"; primals_120: "f32[512, 128]"; primals_121: "f32[512]"; primals_122: "f32[128, 512]"; primals_123: "f32[128]"; primals_124: "f32[256, 128, 3, 3]"; primals_125: "f32[256]"; primals_126: "f32[768, 256]"; primals_127: "f32[768]"; primals_128: "f32[256, 256]"; primals_129: "f32[256]"; primals_130: "f32[1024, 256]"; primals_131: "f32[1024]"; primals_132: "f32[256, 1024]"; primals_133: "f32[256]"; primals_134: "f32[768, 256]"; primals_135: "f32[768]"; primals_136: "f32[256, 256]"; primals_137: "f32[256]"; primals_138: "f32[1024, 256]"; primals_139: "f32[1024]"; primals_140: "f32[256, 1024]"; primals_141: "f32[256]"; primals_142: "f32[512, 256, 3, 3]"; primals_143: "f32[512]"; primals_144: "f32[1536, 512]"; primals_145: "f32[1536]"; primals_146: "f32[512, 512]"; primals_147: "f32[512]"; primals_148: "f32[2048, 512]"; primals_149: "f32[2048]"; primals_150: "f32[512, 2048]"; primals_151: "f32[512]"; primals_152: "f32[1536, 512]"; primals_153: "f32[1536]"; primals_154: "f32[512, 512]"; primals_155: "f32[512]"; primals_156: "f32[2048, 512]"; primals_157: "f32[2048]"; primals_158: "f32[512, 2048]"; primals_159: "f32[512]"; primals_160: "f32[1536, 512]"; primals_161: "f32[1536]"; primals_162: "f32[512, 512]"; primals_163: "f32[512]"; primals_164: "f32[2048, 512]"; primals_165: "f32[2048]"; primals_166: "f32[512, 2048]"; primals_167: "f32[512]"; primals_168: "f32[1536, 512]"; primals_169: "f32[1536]"; primals_170: "f32[512, 512]"; primals_171: "f32[512]"; primals_172: "f32[2048, 512]"; primals_173: "f32[2048]"; primals_174: "f32[512, 2048]"; primals_175: "f32[512]"; primals_176: "f32[1536, 512]"; primals_177: "f32[1536]"; primals_178: "f32[512, 512]"; primals_179: "f32[512]"; primals_180: "f32[2048, 512]"; primals_181: "f32[2048]"; primals_182: "f32[512, 2048]"; primals_183: "f32[512]"; primals_184: "f32[1536, 512]"; primals_185: "f32[1536]"; primals_186: "f32[512, 512]"; primals_187: "f32[512]"; primals_188: "f32[2048, 512]"; primals_189: "f32[2048]"; primals_190: "f32[512, 2048]"; primals_191: "f32[512]"; primals_192: "f32[1536, 512]"; primals_193: "f32[1536]"; primals_194: "f32[512, 512]"; primals_195: "f32[512]"; primals_196: "f32[2048, 512]"; primals_197: "f32[2048]"; primals_198: "f32[512, 2048]"; primals_199: "f32[512]"; primals_200: "f32[1536, 512]"; primals_201: "f32[1536]"; primals_202: "f32[512, 512]"; primals_203: "f32[512]"; primals_204: "f32[2048, 512]"; primals_205: "f32[2048]"; primals_206: "f32[512, 2048]"; primals_207: "f32[512]"; primals_208: "f32[1536, 512]"; primals_209: "f32[1536]"; primals_210: "f32[512, 512]"; primals_211: "f32[512]"; primals_212: "f32[2048, 512]"; primals_213: "f32[2048]"; primals_214: "f32[512, 2048]"; primals_215: "f32[512]"; primals_216: "f32[1536, 512]"; primals_217: "f32[1536]"; primals_218: "f32[512, 512]"; primals_219: "f32[512]"; primals_220: "f32[2048, 512]"; primals_221: "f32[2048]"; primals_222: "f32[512, 2048]"; primals_223: "f32[512]"; primals_224: "f32[1536, 512]"; primals_225: "f32[1536]"; primals_226: "f32[512, 512]"; primals_227: "f32[512]"; primals_228: "f32[2048, 512]"; primals_229: "f32[2048]"; primals_230: "f32[512, 2048]"; primals_231: "f32[512]"; primals_232: "f32[1536, 512]"; primals_233: "f32[1536]"; primals_234: "f32[512, 512]"; primals_235: "f32[512]"; primals_236: "f32[2048, 512]"; primals_237: "f32[2048]"; primals_238: "f32[512, 2048]"; primals_239: "f32[512]"; primals_240: "f32[1536, 512]"; primals_241: "f32[1536]"; primals_242: "f32[512, 512]"; primals_243: "f32[512]"; primals_244: "f32[2048, 512]"; primals_245: "f32[2048]"; primals_246: "f32[512, 2048]"; primals_247: "f32[512]"; primals_248: "f32[1536, 512]"; primals_249: "f32[1536]"; primals_250: "f32[512, 512]"; primals_251: "f32[512]"; primals_252: "f32[2048, 512]"; primals_253: "f32[2048]"; primals_254: "f32[512, 2048]"; primals_255: "f32[512]"; primals_256: "f32[1536, 512]"; primals_257: "f32[1536]"; primals_258: "f32[512, 512]"; primals_259: "f32[512]"; primals_260: "f32[2048, 512]"; primals_261: "f32[2048]"; primals_262: "f32[512, 2048]"; primals_263: "f32[512]"; primals_264: "f32[1536, 512]"; primals_265: "f32[1536]"; primals_266: "f32[512, 512]"; primals_267: "f32[512]"; primals_268: "f32[2048, 512]"; primals_269: "f32[2048]"; primals_270: "f32[512, 2048]"; primals_271: "f32[512]"; primals_272: "f32[1536, 512]"; primals_273: "f32[1536]"; primals_274: "f32[512, 512]"; primals_275: "f32[512]"; primals_276: "f32[2048, 512]"; primals_277: "f32[2048]"; primals_278: "f32[512, 2048]"; primals_279: "f32[512]"; primals_280: "f32[1536, 512]"; primals_281: "f32[1536]"; primals_282: "f32[512, 512]"; primals_283: "f32[512]"; primals_284: "f32[2048, 512]"; primals_285: "f32[2048]"; primals_286: "f32[512, 2048]"; primals_287: "f32[512]"; primals_288: "f32[1536, 512]"; primals_289: "f32[1536]"; primals_290: "f32[512, 512]"; primals_291: "f32[512]"; primals_292: "f32[2048, 512]"; primals_293: "f32[2048]"; primals_294: "f32[512, 2048]"; primals_295: "f32[512]"; primals_296: "f32[1536, 512]"; primals_297: "f32[1536]"; primals_298: "f32[512, 512]"; primals_299: "f32[512]"; primals_300: "f32[2048, 512]"; primals_301: "f32[2048]"; primals_302: "f32[512, 2048]"; primals_303: "f32[512]"; primals_304: "f32[1000, 512]"; primals_305: "f32[1000]"; primals_306: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(primals_306, primals_106, primals_107, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    view: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.view.default(permute, [8, 4, 14, 4, 14, 128]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    permute_1: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2, 4, 5]);  view = None
    clone: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_1: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(clone, [8, 16, 196, 128]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    add: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(view_1, primals_1);  view_1 = primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean = torch.ops.aten.var_mean.correction(add, [3], correction = 0, keepdim = True)
    getitem: "f32[8, 16, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 16, 196, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add, getitem_1)
    mul: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul, primals_2);  mul = None
    add_2: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_3);  mul_1 = primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_2: "f32[25088, 128]" = torch.ops.aten.view.default(add_2, [25088, 128]);  add_2 = None
    permute_2: "f32[128, 384]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm: "f32[25088, 384]" = torch.ops.aten.addmm.default(primals_109, view_2, permute_2);  primals_109 = None
    view_3: "f32[8, 16, 196, 384]" = torch.ops.aten.view.default(addmm, [8, 16, 196, 384]);  addmm = None
    view_4: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.view.default(view_3, [8, 16, 196, 3, 4, 32]);  view_3 = None
    permute_3: "f32[3, 8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(view_4, [3, 0, 4, 1, 2, 5]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
    getitem_2: "f32[8, 4, 16, 196, 32]" = unbind[0]
    getitem_3: "f32[8, 4, 16, 196, 32]" = unbind[1]
    getitem_4: "f32[8, 4, 16, 196, 32]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_2: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_2, 0.42044820762685725);  getitem_2 = None
    permute_4: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.permute.default(getitem_3, [0, 1, 2, 4, 3]);  getitem_3 = None
    mul_3: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.mul.Scalar(permute_4, 0.42044820762685725);  permute_4 = None
    expand: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(mul_2, [8, 4, 16, 196, 32]);  mul_2 = None
    clone_1: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_5: "f32[512, 196, 32]" = torch.ops.aten.view.default(clone_1, [512, 196, 32]);  clone_1 = None
    expand_1: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.expand.default(mul_3, [8, 4, 16, 32, 196]);  mul_3 = None
    clone_2: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_6: "f32[512, 32, 196]" = torch.ops.aten.view.default(clone_2, [512, 32, 196]);  clone_2 = None
    bmm: "f32[512, 196, 196]" = torch.ops.aten.bmm.default(view_5, view_6)
    view_7: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.view.default(bmm, [8, 4, 16, 196, 196]);  bmm = None
    amax: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.amax.default(view_7, [-1], True)
    sub_1: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.sub.Tensor(view_7, amax);  view_7 = amax = None
    exp: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.alias.default(div)
    expand_2: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.expand.default(div, [8, 4, 16, 196, 196]);  div = None
    view_8: "f32[512, 196, 196]" = torch.ops.aten.view.default(expand_2, [512, 196, 196]);  expand_2 = None
    expand_3: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(getitem_4, [8, 4, 16, 196, 32]);  getitem_4 = None
    clone_3: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_9: "f32[512, 196, 32]" = torch.ops.aten.view.default(clone_3, [512, 196, 32]);  clone_3 = None
    bmm_1: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(view_8, view_9)
    view_10: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.view.default(bmm_1, [8, 4, 16, 196, 32]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_5: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.permute.default(view_10, [0, 2, 3, 4, 1]);  view_10 = None
    clone_4: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_11: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(clone_4, [8, 16, 196, 128]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_12: "f32[25088, 128]" = torch.ops.aten.view.default(view_11, [25088, 128]);  view_11 = None
    permute_6: "f32[128, 128]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    addmm_1: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_111, view_12, permute_6);  primals_111 = None
    view_13: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(addmm_1, [8, 16, 196, 128]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_5: "f32[8, 16, 196, 128]" = torch.ops.aten.clone.default(view_13);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_3: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add, clone_5);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [3], correction = 0, keepdim = True)
    getitem_5: "f32[8, 16, 196, 1]" = var_mean_1[0]
    getitem_6: "f32[8, 16, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem_5, 1e-06);  getitem_5 = None
    rsqrt_1: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_3, getitem_6)
    mul_4: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_5: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_4, primals_4);  mul_4 = None
    add_5: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_5, primals_5);  mul_5 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_14: "f32[25088, 128]" = torch.ops.aten.view.default(add_5, [25088, 128]);  add_5 = None
    permute_7: "f32[128, 512]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    addmm_2: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_113, view_14, permute_7);  primals_113 = None
    view_15: "f32[8, 16, 196, 512]" = torch.ops.aten.view.default(addmm_2, [8, 16, 196, 512]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_6: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
    mul_7: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476)
    erf: "f32[8, 16, 196, 512]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_6: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(mul_6, add_6);  mul_6 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_6: "f32[8, 16, 196, 512]" = torch.ops.aten.clone.default(mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_16: "f32[25088, 512]" = torch.ops.aten.view.default(clone_6, [25088, 512]);  clone_6 = None
    permute_8: "f32[512, 128]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    addmm_3: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_115, view_16, permute_8);  primals_115 = None
    view_17: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(addmm_3, [8, 16, 196, 128]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_7: "f32[8, 16, 196, 128]" = torch.ops.aten.clone.default(view_17);  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_7: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_3, clone_7);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [3], correction = 0, keepdim = True)
    getitem_7: "f32[8, 16, 196, 1]" = var_mean_2[0]
    getitem_8: "f32[8, 16, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem_7, 1e-06);  getitem_7 = None
    rsqrt_2: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_3: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_7, getitem_8)
    mul_9: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_10: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_9, primals_6);  mul_9 = None
    add_9: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_10, primals_7);  mul_10 = primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_18: "f32[25088, 128]" = torch.ops.aten.view.default(add_9, [25088, 128]);  add_9 = None
    permute_9: "f32[128, 384]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    addmm_4: "f32[25088, 384]" = torch.ops.aten.addmm.default(primals_117, view_18, permute_9);  primals_117 = None
    view_19: "f32[8, 16, 196, 384]" = torch.ops.aten.view.default(addmm_4, [8, 16, 196, 384]);  addmm_4 = None
    view_20: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.view.default(view_19, [8, 16, 196, 3, 4, 32]);  view_19 = None
    permute_10: "f32[3, 8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(view_20, [3, 0, 4, 1, 2, 5]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_1 = torch.ops.aten.unbind.int(permute_10);  permute_10 = None
    getitem_9: "f32[8, 4, 16, 196, 32]" = unbind_1[0]
    getitem_10: "f32[8, 4, 16, 196, 32]" = unbind_1[1]
    getitem_11: "f32[8, 4, 16, 196, 32]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_11: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_9, 0.42044820762685725);  getitem_9 = None
    permute_11: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.permute.default(getitem_10, [0, 1, 2, 4, 3]);  getitem_10 = None
    mul_12: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.mul.Scalar(permute_11, 0.42044820762685725);  permute_11 = None
    expand_4: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(mul_11, [8, 4, 16, 196, 32]);  mul_11 = None
    clone_8: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_21: "f32[512, 196, 32]" = torch.ops.aten.view.default(clone_8, [512, 196, 32]);  clone_8 = None
    expand_5: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.expand.default(mul_12, [8, 4, 16, 32, 196]);  mul_12 = None
    clone_9: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_22: "f32[512, 32, 196]" = torch.ops.aten.view.default(clone_9, [512, 32, 196]);  clone_9 = None
    bmm_2: "f32[512, 196, 196]" = torch.ops.aten.bmm.default(view_21, view_22)
    view_23: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.view.default(bmm_2, [8, 4, 16, 196, 196]);  bmm_2 = None
    amax_1: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.amax.default(view_23, [-1], True)
    sub_4: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.sub.Tensor(view_23, amax_1);  view_23 = amax_1 = None
    exp_1: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.alias.default(div_1)
    expand_6: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.expand.default(div_1, [8, 4, 16, 196, 196]);  div_1 = None
    view_24: "f32[512, 196, 196]" = torch.ops.aten.view.default(expand_6, [512, 196, 196]);  expand_6 = None
    expand_7: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(getitem_11, [8, 4, 16, 196, 32]);  getitem_11 = None
    clone_10: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_25: "f32[512, 196, 32]" = torch.ops.aten.view.default(clone_10, [512, 196, 32]);  clone_10 = None
    bmm_3: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(view_24, view_25)
    view_26: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.view.default(bmm_3, [8, 4, 16, 196, 32]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_12: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.permute.default(view_26, [0, 2, 3, 4, 1]);  view_26 = None
    clone_11: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    view_27: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(clone_11, [8, 16, 196, 128]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_28: "f32[25088, 128]" = torch.ops.aten.view.default(view_27, [25088, 128]);  view_27 = None
    permute_13: "f32[128, 128]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_5: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_119, view_28, permute_13);  primals_119 = None
    view_29: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(addmm_5, [8, 16, 196, 128]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_12: "f32[8, 16, 196, 128]" = torch.ops.aten.clone.default(view_29);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty, 0.9782608691602945);  empty = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_2: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli, 0.9782608691602945);  bernoulli = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_13: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(clone_12, div_2);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_10: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_7, mul_13);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [3], correction = 0, keepdim = True)
    getitem_12: "f32[8, 16, 196, 1]" = var_mean_3[0]
    getitem_13: "f32[8, 16, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_3: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_5: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_10, getitem_13)
    mul_14: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_15: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_14, primals_8);  mul_14 = None
    add_12: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_15, primals_9);  mul_15 = primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_30: "f32[25088, 128]" = torch.ops.aten.view.default(add_12, [25088, 128]);  add_12 = None
    permute_14: "f32[128, 512]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_6: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_121, view_30, permute_14);  primals_121 = None
    view_31: "f32[8, 16, 196, 512]" = torch.ops.aten.view.default(addmm_6, [8, 16, 196, 512]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_16: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
    mul_17: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476)
    erf_1: "f32[8, 16, 196, 512]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
    add_13: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_18: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(mul_16, add_13);  mul_16 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_13: "f32[8, 16, 196, 512]" = torch.ops.aten.clone.default(mul_18);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_32: "f32[25088, 512]" = torch.ops.aten.view.default(clone_13, [25088, 512]);  clone_13 = None
    permute_15: "f32[512, 128]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_7: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_123, view_32, permute_15);  primals_123 = None
    view_33: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(addmm_7, [8, 16, 196, 128]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_14: "f32[8, 16, 196, 128]" = torch.ops.aten.clone.default(view_33);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_1: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_1: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_1, 0.9782608691602945);  empty_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_3: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_1, 0.9782608691602945);  bernoulli_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_19: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(clone_14, div_3);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_14: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_10, mul_19);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    view_34: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.view.default(add_14, [8, 4, 4, 14, 14, 128]);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    permute_16: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.permute.default(view_34, [0, 1, 3, 2, 4, 5]);  view_34 = None
    clone_15: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    view_35: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(clone_15, [8, 56, 56, 128]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_17: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(view_35, [0, 3, 1, 2]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:141, code: x = self.conv(x)
    convolution_1: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(permute_17, primals_124, primals_125, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_18: "f32[8, 56, 56, 256]" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_4 = torch.ops.aten.var_mean.correction(permute_18, [3], correction = 0, keepdim = True)
    getitem_14: "f32[8, 56, 56, 1]" = var_mean_4[0]
    getitem_15: "f32[8, 56, 56, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_4: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_6: "f32[8, 56, 56, 256]" = torch.ops.aten.sub.Tensor(permute_18, getitem_15)
    mul_20: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_21: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(mul_20, primals_10);  mul_20 = None
    add_16: "f32[8, 56, 56, 256]" = torch.ops.aten.add.Tensor(mul_21, primals_11);  mul_21 = primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_19: "f32[8, 256, 56, 56]" = torch.ops.aten.permute.default(add_16, [0, 3, 1, 2]);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd: "f32[8, 256, 57, 57]" = torch.ops.aten.constant_pad_nd.default(permute_19, [0, 1, 0, 1], -inf);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd, [3, 3], [2, 2])
    getitem_16: "f32[8, 256, 28, 28]" = max_pool2d_with_indices[0]
    getitem_17: "i64[8, 256, 28, 28]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute_20: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 3, 1]);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    view_36: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.view.default(permute_20, [8, 2, 14, 2, 14, 256]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    permute_21: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.permute.default(view_36, [0, 1, 3, 2, 4, 5]);  view_36 = None
    clone_16: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
    view_37: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(clone_16, [8, 4, 196, 256]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    add_17: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(view_37, primals_12);  view_37 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [3], correction = 0, keepdim = True)
    getitem_18: "f32[8, 4, 196, 1]" = var_mean_5[0]
    getitem_19: "f32[8, 4, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_5: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_7: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_17, getitem_19)
    mul_22: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
    mul_23: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_22, primals_13);  mul_22 = None
    add_19: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_23, primals_14);  mul_23 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_38: "f32[6272, 256]" = torch.ops.aten.view.default(add_19, [6272, 256]);  add_19 = None
    permute_22: "f32[256, 768]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    addmm_8: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_127, view_38, permute_22);  primals_127 = None
    view_39: "f32[8, 4, 196, 768]" = torch.ops.aten.view.default(addmm_8, [8, 4, 196, 768]);  addmm_8 = None
    view_40: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.view.default(view_39, [8, 4, 196, 3, 8, 32]);  view_39 = None
    permute_23: "f32[3, 8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(view_40, [3, 0, 4, 1, 2, 5]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_2 = torch.ops.aten.unbind.int(permute_23);  permute_23 = None
    getitem_20: "f32[8, 8, 4, 196, 32]" = unbind_2[0]
    getitem_21: "f32[8, 8, 4, 196, 32]" = unbind_2[1]
    getitem_22: "f32[8, 8, 4, 196, 32]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_24: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_20, 0.42044820762685725);  getitem_20 = None
    permute_24: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.permute.default(getitem_21, [0, 1, 2, 4, 3]);  getitem_21 = None
    mul_25: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.mul.Scalar(permute_24, 0.42044820762685725);  permute_24 = None
    expand_8: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(mul_24, [8, 8, 4, 196, 32]);  mul_24 = None
    clone_17: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_41: "f32[256, 196, 32]" = torch.ops.aten.view.default(clone_17, [256, 196, 32]);  clone_17 = None
    expand_9: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.expand.default(mul_25, [8, 8, 4, 32, 196]);  mul_25 = None
    clone_18: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_42: "f32[256, 32, 196]" = torch.ops.aten.view.default(clone_18, [256, 32, 196]);  clone_18 = None
    bmm_4: "f32[256, 196, 196]" = torch.ops.aten.bmm.default(view_41, view_42)
    view_43: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_4, [8, 8, 4, 196, 196]);  bmm_4 = None
    amax_2: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.amax.default(view_43, [-1], True)
    sub_8: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(view_43, amax_2);  view_43 = amax_2 = None
    exp_2: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_3: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_4: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.alias.default(div_4)
    expand_10: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.expand.default(div_4, [8, 8, 4, 196, 196]);  div_4 = None
    view_44: "f32[256, 196, 196]" = torch.ops.aten.view.default(expand_10, [256, 196, 196]);  expand_10 = None
    expand_11: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(getitem_22, [8, 8, 4, 196, 32]);  getitem_22 = None
    clone_19: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_45: "f32[256, 196, 32]" = torch.ops.aten.view.default(clone_19, [256, 196, 32]);  clone_19 = None
    bmm_5: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(view_44, view_45)
    view_46: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_5, [8, 8, 4, 196, 32]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_25: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.permute.default(view_46, [0, 2, 3, 4, 1]);  view_46 = None
    clone_20: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    view_47: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(clone_20, [8, 4, 196, 256]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_48: "f32[6272, 256]" = torch.ops.aten.view.default(view_47, [6272, 256]);  view_47 = None
    permute_26: "f32[256, 256]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_9: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_129, view_48, permute_26);  primals_129 = None
    view_49: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(addmm_9, [8, 4, 196, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_21: "f32[8, 4, 196, 256]" = torch.ops.aten.clone.default(view_49);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_2: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_2: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_2, 0.9565217383205891);  empty_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_5: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_2, 0.9565217383205891);  bernoulli_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_26: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(clone_21, div_5);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_20: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_17, mul_26);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_20, [3], correction = 0, keepdim = True)
    getitem_23: "f32[8, 4, 196, 1]" = var_mean_6[0]
    getitem_24: "f32[8, 4, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_23, 1e-06);  getitem_23 = None
    rsqrt_6: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_9: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_20, getitem_24)
    mul_27: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_28: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_27, primals_15);  mul_27 = None
    add_22: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_28, primals_16);  mul_28 = primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_50: "f32[6272, 256]" = torch.ops.aten.view.default(add_22, [6272, 256]);  add_22 = None
    permute_27: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_10: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_131, view_50, permute_27);  primals_131 = None
    view_51: "f32[8, 4, 196, 1024]" = torch.ops.aten.view.default(addmm_10, [8, 4, 196, 1024]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_29: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    mul_30: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476)
    erf_2: "f32[8, 4, 196, 1024]" = torch.ops.aten.erf.default(mul_30);  mul_30 = None
    add_23: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_31: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(mul_29, add_23);  mul_29 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_22: "f32[8, 4, 196, 1024]" = torch.ops.aten.clone.default(mul_31);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_52: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_22, [6272, 1024]);  clone_22 = None
    permute_28: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    addmm_11: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_133, view_52, permute_28);  primals_133 = None
    view_53: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(addmm_11, [8, 4, 196, 256]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_23: "f32[8, 4, 196, 256]" = torch.ops.aten.clone.default(view_53);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_3: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_3: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_3, 0.9565217383205891);  empty_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_6: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_3, 0.9565217383205891);  bernoulli_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_32: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(clone_23, div_6);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_24: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_20, mul_32);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [3], correction = 0, keepdim = True)
    getitem_25: "f32[8, 4, 196, 1]" = var_mean_7[0]
    getitem_26: "f32[8, 4, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_25, 1e-06);  getitem_25 = None
    rsqrt_7: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_10: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_24, getitem_26)
    mul_33: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = None
    mul_34: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_33, primals_17);  mul_33 = None
    add_26: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_34, primals_18);  mul_34 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_54: "f32[6272, 256]" = torch.ops.aten.view.default(add_26, [6272, 256]);  add_26 = None
    permute_29: "f32[256, 768]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_12: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_135, view_54, permute_29);  primals_135 = None
    view_55: "f32[8, 4, 196, 768]" = torch.ops.aten.view.default(addmm_12, [8, 4, 196, 768]);  addmm_12 = None
    view_56: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.view.default(view_55, [8, 4, 196, 3, 8, 32]);  view_55 = None
    permute_30: "f32[3, 8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(view_56, [3, 0, 4, 1, 2, 5]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_3 = torch.ops.aten.unbind.int(permute_30);  permute_30 = None
    getitem_27: "f32[8, 8, 4, 196, 32]" = unbind_3[0]
    getitem_28: "f32[8, 8, 4, 196, 32]" = unbind_3[1]
    getitem_29: "f32[8, 8, 4, 196, 32]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_35: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_27, 0.42044820762685725);  getitem_27 = None
    permute_31: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.permute.default(getitem_28, [0, 1, 2, 4, 3]);  getitem_28 = None
    mul_36: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.mul.Scalar(permute_31, 0.42044820762685725);  permute_31 = None
    expand_12: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(mul_35, [8, 8, 4, 196, 32]);  mul_35 = None
    clone_24: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_57: "f32[256, 196, 32]" = torch.ops.aten.view.default(clone_24, [256, 196, 32]);  clone_24 = None
    expand_13: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.expand.default(mul_36, [8, 8, 4, 32, 196]);  mul_36 = None
    clone_25: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_58: "f32[256, 32, 196]" = torch.ops.aten.view.default(clone_25, [256, 32, 196]);  clone_25 = None
    bmm_6: "f32[256, 196, 196]" = torch.ops.aten.bmm.default(view_57, view_58)
    view_59: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_6, [8, 8, 4, 196, 196]);  bmm_6 = None
    amax_3: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.amax.default(view_59, [-1], True)
    sub_11: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(view_59, amax_3);  view_59 = amax_3 = None
    exp_3: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_4: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.alias.default(div_7)
    expand_14: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.expand.default(div_7, [8, 8, 4, 196, 196]);  div_7 = None
    view_60: "f32[256, 196, 196]" = torch.ops.aten.view.default(expand_14, [256, 196, 196]);  expand_14 = None
    expand_15: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(getitem_29, [8, 8, 4, 196, 32]);  getitem_29 = None
    clone_26: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_61: "f32[256, 196, 32]" = torch.ops.aten.view.default(clone_26, [256, 196, 32]);  clone_26 = None
    bmm_7: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(view_60, view_61)
    view_62: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_7, [8, 8, 4, 196, 32]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_32: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.permute.default(view_62, [0, 2, 3, 4, 1]);  view_62 = None
    clone_27: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    view_63: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(clone_27, [8, 4, 196, 256]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_64: "f32[6272, 256]" = torch.ops.aten.view.default(view_63, [6272, 256]);  view_63 = None
    permute_33: "f32[256, 256]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_13: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_137, view_64, permute_33);  primals_137 = None
    view_65: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(addmm_13, [8, 4, 196, 256]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_28: "f32[8, 4, 196, 256]" = torch.ops.aten.clone.default(view_65);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_4: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_4: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_4, 0.9347826093435287);  empty_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_8: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_4, 0.9347826093435287);  bernoulli_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_37: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(clone_28, div_8);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_27: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_24, mul_37);  mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_27, [3], correction = 0, keepdim = True)
    getitem_30: "f32[8, 4, 196, 1]" = var_mean_8[0]
    getitem_31: "f32[8, 4, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_8: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_12: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_27, getitem_31)
    mul_38: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_39: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_38, primals_19);  mul_38 = None
    add_29: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_39, primals_20);  mul_39 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_66: "f32[6272, 256]" = torch.ops.aten.view.default(add_29, [6272, 256]);  add_29 = None
    permute_34: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_14: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_139, view_66, permute_34);  primals_139 = None
    view_67: "f32[8, 4, 196, 1024]" = torch.ops.aten.view.default(addmm_14, [8, 4, 196, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_40: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_67, 0.5)
    mul_41: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_67, 0.7071067811865476)
    erf_3: "f32[8, 4, 196, 1024]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_30: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_42: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(mul_40, add_30);  mul_40 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_29: "f32[8, 4, 196, 1024]" = torch.ops.aten.clone.default(mul_42);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_68: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_29, [6272, 1024]);  clone_29 = None
    permute_35: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_15: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_141, view_68, permute_35);  primals_141 = None
    view_69: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(addmm_15, [8, 4, 196, 256]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_30: "f32[8, 4, 196, 256]" = torch.ops.aten.clone.default(view_69);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_5: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_5: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_5, 0.9347826093435287);  empty_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_9: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_5, 0.9347826093435287);  bernoulli_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_43: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(clone_30, div_9);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_31: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_27, mul_43);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    view_70: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.view.default(add_31, [8, 2, 2, 14, 14, 256]);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    permute_36: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.permute.default(view_70, [0, 1, 3, 2, 4, 5]);  view_70 = None
    clone_31: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
    view_71: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(clone_31, [8, 28, 28, 256]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_37: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(view_71, [0, 3, 1, 2]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:141, code: x = self.conv(x)
    convolution_2: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(permute_37, primals_142, primals_143, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_38: "f32[8, 28, 28, 512]" = torch.ops.aten.permute.default(convolution_2, [0, 2, 3, 1]);  convolution_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_9 = torch.ops.aten.var_mean.correction(permute_38, [3], correction = 0, keepdim = True)
    getitem_32: "f32[8, 28, 28, 1]" = var_mean_9[0]
    getitem_33: "f32[8, 28, 28, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_9: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_13: "f32[8, 28, 28, 512]" = torch.ops.aten.sub.Tensor(permute_38, getitem_33)
    mul_44: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_9);  sub_13 = None
    mul_45: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_44, primals_21);  mul_44 = None
    add_33: "f32[8, 28, 28, 512]" = torch.ops.aten.add.Tensor(mul_45, primals_22);  mul_45 = primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_39: "f32[8, 512, 28, 28]" = torch.ops.aten.permute.default(add_33, [0, 3, 1, 2]);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_1: "f32[8, 512, 29, 29]" = torch.ops.aten.constant_pad_nd.default(permute_39, [0, 1, 0, 1], -inf);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(constant_pad_nd_1, [3, 3], [2, 2])
    getitem_34: "f32[8, 512, 14, 14]" = max_pool2d_with_indices_1[0]
    getitem_35: "i64[8, 512, 14, 14]" = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute_40: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 3, 1]);  getitem_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    view_72: "f32[8, 1, 14, 1, 14, 512]" = torch.ops.aten.view.default(permute_40, [8, 1, 14, 1, 14, 512]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    permute_41: "f32[8, 1, 1, 14, 14, 512]" = torch.ops.aten.permute.default(view_72, [0, 1, 3, 2, 4, 5]);  view_72 = None
    view_73: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(permute_41, [8, 1, -1, 512]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    add_34: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(view_73, primals_23);  view_73 = primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_34, [3], correction = 0, keepdim = True)
    getitem_36: "f32[8, 1, 196, 1]" = var_mean_10[0]
    getitem_37: "f32[8, 1, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_10: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_14: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_34, getitem_37)
    mul_46: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = None
    mul_47: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_46, primals_24);  mul_46 = None
    add_36: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_47, primals_25);  mul_47 = primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_74: "f32[1568, 512]" = torch.ops.aten.view.default(add_36, [1568, 512]);  add_36 = None
    permute_42: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_16: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_145, view_74, permute_42);  primals_145 = None
    view_75: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_16, [8, 1, 196, 1536]);  addmm_16 = None
    view_76: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_75, [8, 1, 196, 3, 16, 32]);  view_75 = None
    permute_43: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_76, [3, 0, 4, 1, 2, 5]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_4 = torch.ops.aten.unbind.int(permute_43);  permute_43 = None
    getitem_38: "f32[8, 16, 1, 196, 32]" = unbind_4[0]
    getitem_39: "f32[8, 16, 1, 196, 32]" = unbind_4[1]
    getitem_40: "f32[8, 16, 1, 196, 32]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_48: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_38, 0.42044820762685725);  getitem_38 = None
    permute_44: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_39, [0, 1, 2, 4, 3]);  getitem_39 = None
    mul_49: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_44, 0.42044820762685725);  permute_44 = None
    expand_16: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_48, [8, 16, 1, 196, 32]);  mul_48 = None
    clone_32: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_77: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_32, [128, 196, 32]);  clone_32 = None
    expand_17: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_49, [8, 16, 1, 32, 196]);  mul_49 = None
    clone_33: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_78: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_33, [128, 32, 196]);  clone_33 = None
    bmm_8: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_77, view_78)
    view_79: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_8, [8, 16, 1, 196, 196]);  bmm_8 = None
    amax_4: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_79, [-1], True)
    sub_15: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_79, amax_4);  view_79 = amax_4 = None
    exp_4: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_5: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_10: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_10)
    expand_18: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_10, [8, 16, 1, 196, 196]);  div_10 = None
    view_80: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_18, [128, 196, 196]);  expand_18 = None
    expand_19: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_40, [8, 16, 1, 196, 32]);  getitem_40 = None
    clone_34: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_81: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_34, [128, 196, 32]);  clone_34 = None
    bmm_9: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_80, view_81)
    view_82: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_9, [8, 16, 1, 196, 32]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_45: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_82, [0, 2, 3, 4, 1]);  view_82 = None
    clone_35: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_83: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_35, [8, 1, 196, 512]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_84: "f32[1568, 512]" = torch.ops.aten.view.default(view_83, [1568, 512]);  view_83 = None
    permute_46: "f32[512, 512]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_17: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_147, view_84, permute_46);  primals_147 = None
    view_85: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_17, [8, 1, 196, 512]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_36: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_85);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_6: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_6: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_6, 0.9130434766411781);  empty_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_11: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_6, 0.9130434766411781);  bernoulli_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_50: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_36, div_11);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_37: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_34, mul_50);  mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_37, [3], correction = 0, keepdim = True)
    getitem_41: "f32[8, 1, 196, 1]" = var_mean_11[0]
    getitem_42: "f32[8, 1, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-06);  getitem_41 = None
    rsqrt_11: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_16: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_37, getitem_42)
    mul_51: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_11);  sub_16 = None
    mul_52: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_51, primals_26);  mul_51 = None
    add_39: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_52, primals_27);  mul_52 = primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_86: "f32[1568, 512]" = torch.ops.aten.view.default(add_39, [1568, 512]);  add_39 = None
    permute_47: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_18: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_149, view_86, permute_47);  primals_149 = None
    view_87: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_18, [8, 1, 196, 2048]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_53: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    mul_54: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476)
    erf_4: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_40: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_55: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_53, add_40);  mul_53 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_37: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_55);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_88: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_37, [1568, 2048]);  clone_37 = None
    permute_48: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_19: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_151, view_88, permute_48);  primals_151 = None
    view_89: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_19, [8, 1, 196, 512]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_38: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_89);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_7: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_7: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_7, 0.9130434766411781);  empty_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_12: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_7, 0.9130434766411781);  bernoulli_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_56: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_38, div_12);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_41: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_37, mul_56);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_41, [3], correction = 0, keepdim = True)
    getitem_43: "f32[8, 1, 196, 1]" = var_mean_12[0]
    getitem_44: "f32[8, 1, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_43, 1e-06);  getitem_43 = None
    rsqrt_12: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_17: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_41, getitem_44)
    mul_57: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = None
    mul_58: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_57, primals_28);  mul_57 = None
    add_43: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_58, primals_29);  mul_58 = primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_90: "f32[1568, 512]" = torch.ops.aten.view.default(add_43, [1568, 512]);  add_43 = None
    permute_49: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_20: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_153, view_90, permute_49);  primals_153 = None
    view_91: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_20, [8, 1, 196, 1536]);  addmm_20 = None
    view_92: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_91, [8, 1, 196, 3, 16, 32]);  view_91 = None
    permute_50: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_92, [3, 0, 4, 1, 2, 5]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_5 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
    getitem_45: "f32[8, 16, 1, 196, 32]" = unbind_5[0]
    getitem_46: "f32[8, 16, 1, 196, 32]" = unbind_5[1]
    getitem_47: "f32[8, 16, 1, 196, 32]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_59: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_45, 0.42044820762685725);  getitem_45 = None
    permute_51: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_46, [0, 1, 2, 4, 3]);  getitem_46 = None
    mul_60: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_51, 0.42044820762685725);  permute_51 = None
    expand_20: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_59, [8, 16, 1, 196, 32]);  mul_59 = None
    clone_39: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_93: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_39, [128, 196, 32]);  clone_39 = None
    expand_21: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_60, [8, 16, 1, 32, 196]);  mul_60 = None
    clone_40: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_94: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_40, [128, 32, 196]);  clone_40 = None
    bmm_10: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_93, view_94)
    view_95: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_10, [8, 16, 1, 196, 196]);  bmm_10 = None
    amax_5: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_95, [-1], True)
    sub_18: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_95, amax_5);  view_95 = amax_5 = None
    exp_5: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_6: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_13: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_13)
    expand_22: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_13, [8, 16, 1, 196, 196]);  div_13 = None
    view_96: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_22, [128, 196, 196]);  expand_22 = None
    expand_23: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_47, [8, 16, 1, 196, 32]);  getitem_47 = None
    clone_41: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_97: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_41, [128, 196, 32]);  clone_41 = None
    bmm_11: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_96, view_97)
    view_98: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_11, [8, 16, 1, 196, 32]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_52: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_98, [0, 2, 3, 4, 1]);  view_98 = None
    clone_42: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_99: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_42, [8, 1, 196, 512]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_100: "f32[1568, 512]" = torch.ops.aten.view.default(view_99, [1568, 512]);  view_99 = None
    permute_53: "f32[512, 512]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_21: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_155, view_100, permute_53);  primals_155 = None
    view_101: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_21, [8, 1, 196, 512]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_43: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_101);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_8: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_8: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_8, 0.8913043439388275);  empty_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_14: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_8, 0.8913043439388275);  bernoulli_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_61: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_43, div_14);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_44: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_41, mul_61);  mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_44, [3], correction = 0, keepdim = True)
    getitem_48: "f32[8, 1, 196, 1]" = var_mean_13[0]
    getitem_49: "f32[8, 1, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    add_45: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_13: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_19: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_44, getitem_49)
    mul_62: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = None
    mul_63: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_62, primals_30);  mul_62 = None
    add_46: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_63, primals_31);  mul_63 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_102: "f32[1568, 512]" = torch.ops.aten.view.default(add_46, [1568, 512]);  add_46 = None
    permute_54: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_22: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_157, view_102, permute_54);  primals_157 = None
    view_103: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_22, [8, 1, 196, 2048]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_64: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_103, 0.5)
    mul_65: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_103, 0.7071067811865476)
    erf_5: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_65);  mul_65 = None
    add_47: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_66: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_64, add_47);  mul_64 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_44: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_66);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_104: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_44, [1568, 2048]);  clone_44 = None
    permute_55: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    addmm_23: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_159, view_104, permute_55);  primals_159 = None
    view_105: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_23, [8, 1, 196, 512]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_45: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_105);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_9: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_9: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_9, 0.8913043439388275);  empty_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_15: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_9, 0.8913043439388275);  bernoulli_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_67: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_45, div_15);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_48: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_44, mul_67);  mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_48, [3], correction = 0, keepdim = True)
    getitem_50: "f32[8, 1, 196, 1]" = var_mean_14[0]
    getitem_51: "f32[8, 1, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
    rsqrt_14: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_20: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_48, getitem_51)
    mul_68: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = None
    mul_69: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_68, primals_32);  mul_68 = None
    add_50: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_69, primals_33);  mul_69 = primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_106: "f32[1568, 512]" = torch.ops.aten.view.default(add_50, [1568, 512]);  add_50 = None
    permute_56: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_24: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_161, view_106, permute_56);  primals_161 = None
    view_107: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_24, [8, 1, 196, 1536]);  addmm_24 = None
    view_108: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_107, [8, 1, 196, 3, 16, 32]);  view_107 = None
    permute_57: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_108, [3, 0, 4, 1, 2, 5]);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_6 = torch.ops.aten.unbind.int(permute_57);  permute_57 = None
    getitem_52: "f32[8, 16, 1, 196, 32]" = unbind_6[0]
    getitem_53: "f32[8, 16, 1, 196, 32]" = unbind_6[1]
    getitem_54: "f32[8, 16, 1, 196, 32]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_70: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_52, 0.42044820762685725);  getitem_52 = None
    permute_58: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_53, [0, 1, 2, 4, 3]);  getitem_53 = None
    mul_71: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_58, 0.42044820762685725);  permute_58 = None
    expand_24: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_70, [8, 16, 1, 196, 32]);  mul_70 = None
    clone_46: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_109: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_46, [128, 196, 32]);  clone_46 = None
    expand_25: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_71, [8, 16, 1, 32, 196]);  mul_71 = None
    clone_47: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_110: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_47, [128, 32, 196]);  clone_47 = None
    bmm_12: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_109, view_110)
    view_111: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_12, [8, 16, 1, 196, 196]);  bmm_12 = None
    amax_6: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_111, [-1], True)
    sub_21: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_111, amax_6);  view_111 = amax_6 = None
    exp_6: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_7: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_16: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_16)
    expand_26: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_16, [8, 16, 1, 196, 196]);  div_16 = None
    view_112: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_26, [128, 196, 196]);  expand_26 = None
    expand_27: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_54, [8, 16, 1, 196, 32]);  getitem_54 = None
    clone_48: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_113: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_48, [128, 196, 32]);  clone_48 = None
    bmm_13: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_112, view_113)
    view_114: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_13, [8, 16, 1, 196, 32]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_59: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_114, [0, 2, 3, 4, 1]);  view_114 = None
    clone_49: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    view_115: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_49, [8, 1, 196, 512]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_116: "f32[1568, 512]" = torch.ops.aten.view.default(view_115, [1568, 512]);  view_115 = None
    permute_60: "f32[512, 512]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_25: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_163, view_116, permute_60);  primals_163 = None
    view_117: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_25, [8, 1, 196, 512]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_50: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_117);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_10: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_10: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_10, 0.8695652186870575);  empty_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_17: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_10, 0.8695652186870575);  bernoulli_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_72: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_50, div_17);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_51: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_48, mul_72);  mul_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_51, [3], correction = 0, keepdim = True)
    getitem_55: "f32[8, 1, 196, 1]" = var_mean_15[0]
    getitem_56: "f32[8, 1, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    add_52: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-06);  getitem_55 = None
    rsqrt_15: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_22: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_51, getitem_56)
    mul_73: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_15);  sub_22 = None
    mul_74: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_73, primals_34);  mul_73 = None
    add_53: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_74, primals_35);  mul_74 = primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_118: "f32[1568, 512]" = torch.ops.aten.view.default(add_53, [1568, 512]);  add_53 = None
    permute_61: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
    addmm_26: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_165, view_118, permute_61);  primals_165 = None
    view_119: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_26, [8, 1, 196, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_75: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_119, 0.5)
    mul_76: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_119, 0.7071067811865476)
    erf_6: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_54: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_77: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_75, add_54);  mul_75 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_51: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_77);  mul_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_120: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_51, [1568, 2048]);  clone_51 = None
    permute_62: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_27: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_167, view_120, permute_62);  primals_167 = None
    view_121: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_27, [8, 1, 196, 512]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_52: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_121);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_11: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_11: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_11, 0.8695652186870575);  empty_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_18: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_11, 0.8695652186870575);  bernoulli_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_78: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_52, div_18);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_55: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_51, mul_78);  mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_55, [3], correction = 0, keepdim = True)
    getitem_57: "f32[8, 1, 196, 1]" = var_mean_16[0]
    getitem_58: "f32[8, 1, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    add_56: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_57, 1e-06);  getitem_57 = None
    rsqrt_16: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_23: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_55, getitem_58)
    mul_79: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_16);  sub_23 = None
    mul_80: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_79, primals_36);  mul_79 = None
    add_57: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_80, primals_37);  mul_80 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_122: "f32[1568, 512]" = torch.ops.aten.view.default(add_57, [1568, 512]);  add_57 = None
    permute_63: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_28: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_169, view_122, permute_63);  primals_169 = None
    view_123: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_28, [8, 1, 196, 1536]);  addmm_28 = None
    view_124: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_123, [8, 1, 196, 3, 16, 32]);  view_123 = None
    permute_64: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_124, [3, 0, 4, 1, 2, 5]);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_7 = torch.ops.aten.unbind.int(permute_64);  permute_64 = None
    getitem_59: "f32[8, 16, 1, 196, 32]" = unbind_7[0]
    getitem_60: "f32[8, 16, 1, 196, 32]" = unbind_7[1]
    getitem_61: "f32[8, 16, 1, 196, 32]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_81: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_59, 0.42044820762685725);  getitem_59 = None
    permute_65: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_60, [0, 1, 2, 4, 3]);  getitem_60 = None
    mul_82: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_65, 0.42044820762685725);  permute_65 = None
    expand_28: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_81, [8, 16, 1, 196, 32]);  mul_81 = None
    clone_53: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_125: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_53, [128, 196, 32]);  clone_53 = None
    expand_29: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_82, [8, 16, 1, 32, 196]);  mul_82 = None
    clone_54: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_126: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_54, [128, 32, 196]);  clone_54 = None
    bmm_14: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_125, view_126)
    view_127: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_14, [8, 16, 1, 196, 196]);  bmm_14 = None
    amax_7: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_127, [-1], True)
    sub_24: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_127, amax_7);  view_127 = amax_7 = None
    exp_7: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_8: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_19: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_19)
    expand_30: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_19, [8, 16, 1, 196, 196]);  div_19 = None
    view_128: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_30, [128, 196, 196]);  expand_30 = None
    expand_31: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_61, [8, 16, 1, 196, 32]);  getitem_61 = None
    clone_55: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_129: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_55, [128, 196, 32]);  clone_55 = None
    bmm_15: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_128, view_129)
    view_130: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_15, [8, 16, 1, 196, 32]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_66: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_130, [0, 2, 3, 4, 1]);  view_130 = None
    clone_56: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    view_131: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_56, [8, 1, 196, 512]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_132: "f32[1568, 512]" = torch.ops.aten.view.default(view_131, [1568, 512]);  view_131 = None
    permute_67: "f32[512, 512]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_29: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_171, view_132, permute_67);  primals_171 = None
    view_133: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_29, [8, 1, 196, 512]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_57: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_133);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_12: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_12: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_12, 0.8478260785341263);  empty_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_20: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_12, 0.8478260785341263);  bernoulli_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_83: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_57, div_20);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_58: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_55, mul_83);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_58, [3], correction = 0, keepdim = True)
    getitem_62: "f32[8, 1, 196, 1]" = var_mean_17[0]
    getitem_63: "f32[8, 1, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    add_59: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
    rsqrt_17: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_25: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_58, getitem_63)
    mul_84: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_17);  sub_25 = None
    mul_85: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_84, primals_38);  mul_84 = None
    add_60: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_85, primals_39);  mul_85 = primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_134: "f32[1568, 512]" = torch.ops.aten.view.default(add_60, [1568, 512]);  add_60 = None
    permute_68: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_30: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_173, view_134, permute_68);  primals_173 = None
    view_135: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_30, [8, 1, 196, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_135, 0.5)
    mul_87: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_135, 0.7071067811865476)
    erf_7: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_61: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_88: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_86, add_61);  mul_86 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_58: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_136: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_58, [1568, 2048]);  clone_58 = None
    permute_69: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    addmm_31: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_175, view_136, permute_69);  primals_175 = None
    view_137: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_31, [8, 1, 196, 512]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_59: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_137);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_13: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_13: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_13, 0.8478260785341263);  empty_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_21: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_13, 0.8478260785341263);  bernoulli_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_89: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_59, div_21);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_62: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_58, mul_89);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_62, [3], correction = 0, keepdim = True)
    getitem_64: "f32[8, 1, 196, 1]" = var_mean_18[0]
    getitem_65: "f32[8, 1, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    add_63: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_18: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_26: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_62, getitem_65)
    mul_90: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_18);  sub_26 = None
    mul_91: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_90, primals_40);  mul_90 = None
    add_64: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_91, primals_41);  mul_91 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_138: "f32[1568, 512]" = torch.ops.aten.view.default(add_64, [1568, 512]);  add_64 = None
    permute_70: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_32: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_177, view_138, permute_70);  primals_177 = None
    view_139: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_32, [8, 1, 196, 1536]);  addmm_32 = None
    view_140: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_139, [8, 1, 196, 3, 16, 32]);  view_139 = None
    permute_71: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_140, [3, 0, 4, 1, 2, 5]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_8 = torch.ops.aten.unbind.int(permute_71);  permute_71 = None
    getitem_66: "f32[8, 16, 1, 196, 32]" = unbind_8[0]
    getitem_67: "f32[8, 16, 1, 196, 32]" = unbind_8[1]
    getitem_68: "f32[8, 16, 1, 196, 32]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_92: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_66, 0.42044820762685725);  getitem_66 = None
    permute_72: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_67, [0, 1, 2, 4, 3]);  getitem_67 = None
    mul_93: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_72, 0.42044820762685725);  permute_72 = None
    expand_32: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_92, [8, 16, 1, 196, 32]);  mul_92 = None
    clone_60: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_141: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_60, [128, 196, 32]);  clone_60 = None
    expand_33: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_93, [8, 16, 1, 32, 196]);  mul_93 = None
    clone_61: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_142: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_61, [128, 32, 196]);  clone_61 = None
    bmm_16: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_141, view_142)
    view_143: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_16, [8, 16, 1, 196, 196]);  bmm_16 = None
    amax_8: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_143, [-1], True)
    sub_27: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_143, amax_8);  view_143 = amax_8 = None
    exp_8: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_9: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_22: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_22)
    expand_34: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_22, [8, 16, 1, 196, 196]);  div_22 = None
    view_144: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_34, [128, 196, 196]);  expand_34 = None
    expand_35: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_68, [8, 16, 1, 196, 32]);  getitem_68 = None
    clone_62: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_145: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_62, [128, 196, 32]);  clone_62 = None
    bmm_17: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_144, view_145)
    view_146: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_17, [8, 16, 1, 196, 32]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_73: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_146, [0, 2, 3, 4, 1]);  view_146 = None
    clone_63: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_147: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_63, [8, 1, 196, 512]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_148: "f32[1568, 512]" = torch.ops.aten.view.default(view_147, [1568, 512]);  view_147 = None
    permute_74: "f32[512, 512]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    addmm_33: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_179, view_148, permute_74);  primals_179 = None
    view_149: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_33, [8, 1, 196, 512]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_64: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_149);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_14: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_14: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_14, 0.8260869532823563);  empty_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_23: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_14, 0.8260869532823563);  bernoulli_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_94: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_64, div_23);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_65: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_62, mul_94);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_65, [3], correction = 0, keepdim = True)
    getitem_69: "f32[8, 1, 196, 1]" = var_mean_19[0]
    getitem_70: "f32[8, 1, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    add_66: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_69, 1e-06);  getitem_69 = None
    rsqrt_19: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_28: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_65, getitem_70)
    mul_95: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_19);  sub_28 = None
    mul_96: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_95, primals_42);  mul_95 = None
    add_67: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_96, primals_43);  mul_96 = primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_150: "f32[1568, 512]" = torch.ops.aten.view.default(add_67, [1568, 512]);  add_67 = None
    permute_75: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    addmm_34: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_181, view_150, permute_75);  primals_181 = None
    view_151: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_34, [8, 1, 196, 2048]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_97: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_98: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_8: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
    add_68: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_99: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_97, add_68);  mul_97 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_65: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_99);  mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_152: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_65, [1568, 2048]);  clone_65 = None
    permute_76: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    addmm_35: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_183, view_152, permute_76);  primals_183 = None
    view_153: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_35, [8, 1, 196, 512]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_66: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_153);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_15: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_15: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_15, 0.8260869532823563);  empty_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_24: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_15, 0.8260869532823563);  bernoulli_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_100: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_66, div_24);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_69: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_65, mul_100);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_69, [3], correction = 0, keepdim = True)
    getitem_71: "f32[8, 1, 196, 1]" = var_mean_20[0]
    getitem_72: "f32[8, 1, 196, 1]" = var_mean_20[1];  var_mean_20 = None
    add_70: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_71, 1e-06);  getitem_71 = None
    rsqrt_20: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_29: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_69, getitem_72)
    mul_101: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_20);  sub_29 = None
    mul_102: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_101, primals_44);  mul_101 = None
    add_71: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_102, primals_45);  mul_102 = primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_154: "f32[1568, 512]" = torch.ops.aten.view.default(add_71, [1568, 512]);  add_71 = None
    permute_77: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_36: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_185, view_154, permute_77);  primals_185 = None
    view_155: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_36, [8, 1, 196, 1536]);  addmm_36 = None
    view_156: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_155, [8, 1, 196, 3, 16, 32]);  view_155 = None
    permute_78: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_156, [3, 0, 4, 1, 2, 5]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_9 = torch.ops.aten.unbind.int(permute_78);  permute_78 = None
    getitem_73: "f32[8, 16, 1, 196, 32]" = unbind_9[0]
    getitem_74: "f32[8, 16, 1, 196, 32]" = unbind_9[1]
    getitem_75: "f32[8, 16, 1, 196, 32]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_103: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_73, 0.42044820762685725);  getitem_73 = None
    permute_79: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_74, [0, 1, 2, 4, 3]);  getitem_74 = None
    mul_104: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_79, 0.42044820762685725);  permute_79 = None
    expand_36: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_103, [8, 16, 1, 196, 32]);  mul_103 = None
    clone_67: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_157: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_67, [128, 196, 32]);  clone_67 = None
    expand_37: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_104, [8, 16, 1, 32, 196]);  mul_104 = None
    clone_68: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_158: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_68, [128, 32, 196]);  clone_68 = None
    bmm_18: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_157, view_158)
    view_159: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_18, [8, 16, 1, 196, 196]);  bmm_18 = None
    amax_9: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_159, [-1], True)
    sub_30: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_159, amax_9);  view_159 = amax_9 = None
    exp_9: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_10: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_25: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_25)
    expand_38: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_25, [8, 16, 1, 196, 196]);  div_25 = None
    view_160: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_38, [128, 196, 196]);  expand_38 = None
    expand_39: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_75, [8, 16, 1, 196, 32]);  getitem_75 = None
    clone_69: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_161: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_69, [128, 196, 32]);  clone_69 = None
    bmm_19: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_160, view_161)
    view_162: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_19, [8, 16, 1, 196, 32]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_80: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_162, [0, 2, 3, 4, 1]);  view_162 = None
    clone_70: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    view_163: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_70, [8, 1, 196, 512]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_164: "f32[1568, 512]" = torch.ops.aten.view.default(view_163, [1568, 512]);  view_163 = None
    permute_81: "f32[512, 512]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    addmm_37: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_187, view_164, permute_81);  primals_187 = None
    view_165: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_37, [8, 1, 196, 512]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_71: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_165);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_16: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_16: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_16, 0.8043478280305862);  empty_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_26: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_16, 0.8043478280305862);  bernoulli_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_105: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_71, div_26);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_72: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_69, mul_105);  mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_72, [3], correction = 0, keepdim = True)
    getitem_76: "f32[8, 1, 196, 1]" = var_mean_21[0]
    getitem_77: "f32[8, 1, 196, 1]" = var_mean_21[1];  var_mean_21 = None
    add_73: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_21: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_31: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_72, getitem_77)
    mul_106: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_21);  sub_31 = None
    mul_107: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_106, primals_46);  mul_106 = None
    add_74: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_107, primals_47);  mul_107 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_166: "f32[1568, 512]" = torch.ops.aten.view.default(add_74, [1568, 512]);  add_74 = None
    permute_82: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_38: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_189, view_166, permute_82);  primals_189 = None
    view_167: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_38, [8, 1, 196, 2048]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_108: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_167, 0.5)
    mul_109: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_167, 0.7071067811865476)
    erf_9: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_109);  mul_109 = None
    add_75: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_110: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_108, add_75);  mul_108 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_72: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_110);  mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_168: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_72, [1568, 2048]);  clone_72 = None
    permute_83: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    addmm_39: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_191, view_168, permute_83);  primals_191 = None
    view_169: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_39, [8, 1, 196, 512]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_73: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_169);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_17: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_17: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_17, 0.8043478280305862);  empty_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_27: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_17, 0.8043478280305862);  bernoulli_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_111: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_73, div_27);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_76: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_72, mul_111);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_76, [3], correction = 0, keepdim = True)
    getitem_78: "f32[8, 1, 196, 1]" = var_mean_22[0]
    getitem_79: "f32[8, 1, 196, 1]" = var_mean_22[1];  var_mean_22 = None
    add_77: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_22: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_32: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_76, getitem_79)
    mul_112: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_22);  sub_32 = None
    mul_113: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_112, primals_48);  mul_112 = None
    add_78: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_113, primals_49);  mul_113 = primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_170: "f32[1568, 512]" = torch.ops.aten.view.default(add_78, [1568, 512]);  add_78 = None
    permute_84: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    addmm_40: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_193, view_170, permute_84);  primals_193 = None
    view_171: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_40, [8, 1, 196, 1536]);  addmm_40 = None
    view_172: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_171, [8, 1, 196, 3, 16, 32]);  view_171 = None
    permute_85: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_172, [3, 0, 4, 1, 2, 5]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_10 = torch.ops.aten.unbind.int(permute_85);  permute_85 = None
    getitem_80: "f32[8, 16, 1, 196, 32]" = unbind_10[0]
    getitem_81: "f32[8, 16, 1, 196, 32]" = unbind_10[1]
    getitem_82: "f32[8, 16, 1, 196, 32]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_114: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_80, 0.42044820762685725);  getitem_80 = None
    permute_86: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_81, [0, 1, 2, 4, 3]);  getitem_81 = None
    mul_115: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_86, 0.42044820762685725);  permute_86 = None
    expand_40: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_114, [8, 16, 1, 196, 32]);  mul_114 = None
    clone_74: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_173: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_74, [128, 196, 32]);  clone_74 = None
    expand_41: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_115, [8, 16, 1, 32, 196]);  mul_115 = None
    clone_75: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_174: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_75, [128, 32, 196]);  clone_75 = None
    bmm_20: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_173, view_174)
    view_175: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_20, [8, 16, 1, 196, 196]);  bmm_20 = None
    amax_10: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_175, [-1], True)
    sub_33: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_175, amax_10);  view_175 = amax_10 = None
    exp_10: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_11: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_28: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_28)
    expand_42: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_28, [8, 16, 1, 196, 196]);  div_28 = None
    view_176: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_42, [128, 196, 196]);  expand_42 = None
    expand_43: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_82, [8, 16, 1, 196, 32]);  getitem_82 = None
    clone_76: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_177: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_76, [128, 196, 32]);  clone_76 = None
    bmm_21: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_176, view_177)
    view_178: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_21, [8, 16, 1, 196, 32]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_87: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_178, [0, 2, 3, 4, 1]);  view_178 = None
    clone_77: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    view_179: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_77, [8, 1, 196, 512]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_180: "f32[1568, 512]" = torch.ops.aten.view.default(view_179, [1568, 512]);  view_179 = None
    permute_88: "f32[512, 512]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_41: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_195, view_180, permute_88);  primals_195 = None
    view_181: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_41, [8, 1, 196, 512]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_78: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_181);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_18: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_18: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_18, 0.782608687877655);  empty_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_29: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_18, 0.782608687877655);  bernoulli_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_116: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_78, div_29);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_79: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_76, mul_116);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_79, [3], correction = 0, keepdim = True)
    getitem_83: "f32[8, 1, 196, 1]" = var_mean_23[0]
    getitem_84: "f32[8, 1, 196, 1]" = var_mean_23[1];  var_mean_23 = None
    add_80: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_83, 1e-06);  getitem_83 = None
    rsqrt_23: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_34: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_79, getitem_84)
    mul_117: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_23);  sub_34 = None
    mul_118: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_117, primals_50);  mul_117 = None
    add_81: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_118, primals_51);  mul_118 = primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_182: "f32[1568, 512]" = torch.ops.aten.view.default(add_81, [1568, 512]);  add_81 = None
    permute_89: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm_42: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_197, view_182, permute_89);  primals_197 = None
    view_183: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_42, [8, 1, 196, 2048]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_119: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_183, 0.5)
    mul_120: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_183, 0.7071067811865476)
    erf_10: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_120);  mul_120 = None
    add_82: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_121: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_119, add_82);  mul_119 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_79: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_121);  mul_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_184: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_79, [1568, 2048]);  clone_79 = None
    permute_90: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_43: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_199, view_184, permute_90);  primals_199 = None
    view_185: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_43, [8, 1, 196, 512]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_80: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_185);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_19: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_19: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_19, 0.782608687877655);  empty_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_30: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_19, 0.782608687877655);  bernoulli_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_122: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_80, div_30);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_83: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_79, mul_122);  mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_83, [3], correction = 0, keepdim = True)
    getitem_85: "f32[8, 1, 196, 1]" = var_mean_24[0]
    getitem_86: "f32[8, 1, 196, 1]" = var_mean_24[1];  var_mean_24 = None
    add_84: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_85, 1e-06);  getitem_85 = None
    rsqrt_24: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_35: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_83, getitem_86)
    mul_123: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_24);  sub_35 = None
    mul_124: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_123, primals_52);  mul_123 = None
    add_85: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_124, primals_53);  mul_124 = primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_186: "f32[1568, 512]" = torch.ops.aten.view.default(add_85, [1568, 512]);  add_85 = None
    permute_91: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
    addmm_44: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_201, view_186, permute_91);  primals_201 = None
    view_187: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_44, [8, 1, 196, 1536]);  addmm_44 = None
    view_188: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_187, [8, 1, 196, 3, 16, 32]);  view_187 = None
    permute_92: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_188, [3, 0, 4, 1, 2, 5]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_11 = torch.ops.aten.unbind.int(permute_92);  permute_92 = None
    getitem_87: "f32[8, 16, 1, 196, 32]" = unbind_11[0]
    getitem_88: "f32[8, 16, 1, 196, 32]" = unbind_11[1]
    getitem_89: "f32[8, 16, 1, 196, 32]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_125: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_87, 0.42044820762685725);  getitem_87 = None
    permute_93: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_88, [0, 1, 2, 4, 3]);  getitem_88 = None
    mul_126: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_93, 0.42044820762685725);  permute_93 = None
    expand_44: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_125, [8, 16, 1, 196, 32]);  mul_125 = None
    clone_81: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_189: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_81, [128, 196, 32]);  clone_81 = None
    expand_45: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_126, [8, 16, 1, 32, 196]);  mul_126 = None
    clone_82: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_190: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_82, [128, 32, 196]);  clone_82 = None
    bmm_22: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_189, view_190)
    view_191: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_22, [8, 16, 1, 196, 196]);  bmm_22 = None
    amax_11: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_191, [-1], True)
    sub_36: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_191, amax_11);  view_191 = amax_11 = None
    exp_11: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
    sum_12: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_31: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_31)
    expand_46: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_31, [8, 16, 1, 196, 196]);  div_31 = None
    view_192: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_46, [128, 196, 196]);  expand_46 = None
    expand_47: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_89, [8, 16, 1, 196, 32]);  getitem_89 = None
    clone_83: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_193: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_83, [128, 196, 32]);  clone_83 = None
    bmm_23: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_192, view_193)
    view_194: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_23, [8, 16, 1, 196, 32]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_94: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_194, [0, 2, 3, 4, 1]);  view_194 = None
    clone_84: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
    view_195: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_84, [8, 1, 196, 512]);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_196: "f32[1568, 512]" = torch.ops.aten.view.default(view_195, [1568, 512]);  view_195 = None
    permute_95: "f32[512, 512]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    addmm_45: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_203, view_196, permute_95);  primals_203 = None
    view_197: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_45, [8, 1, 196, 512]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_85: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_197);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_20: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_20: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_20, 0.760869562625885);  empty_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_32: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_20, 0.760869562625885);  bernoulli_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_127: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_85, div_32);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_86: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_83, mul_127);  mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_86, [3], correction = 0, keepdim = True)
    getitem_90: "f32[8, 1, 196, 1]" = var_mean_25[0]
    getitem_91: "f32[8, 1, 196, 1]" = var_mean_25[1];  var_mean_25 = None
    add_87: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
    rsqrt_25: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_37: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_86, getitem_91)
    mul_128: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
    mul_129: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_128, primals_54);  mul_128 = None
    add_88: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_129, primals_55);  mul_129 = primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_198: "f32[1568, 512]" = torch.ops.aten.view.default(add_88, [1568, 512]);  add_88 = None
    permute_96: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
    addmm_46: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_205, view_198, permute_96);  primals_205 = None
    view_199: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_46, [8, 1, 196, 2048]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_130: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_199, 0.5)
    mul_131: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_199, 0.7071067811865476)
    erf_11: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
    add_89: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_132: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_130, add_89);  mul_130 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_86: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_132);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_200: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_86, [1568, 2048]);  clone_86 = None
    permute_97: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_206, [1, 0]);  primals_206 = None
    addmm_47: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_207, view_200, permute_97);  primals_207 = None
    view_201: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_47, [8, 1, 196, 512]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_87: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_201);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_21: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_21: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_21, 0.760869562625885);  empty_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_33: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_21, 0.760869562625885);  bernoulli_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_133: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_87, div_33);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_90: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_86, mul_133);  mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_90, [3], correction = 0, keepdim = True)
    getitem_92: "f32[8, 1, 196, 1]" = var_mean_26[0]
    getitem_93: "f32[8, 1, 196, 1]" = var_mean_26[1];  var_mean_26 = None
    add_91: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
    rsqrt_26: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_38: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_90, getitem_93)
    mul_134: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_26);  sub_38 = None
    mul_135: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_134, primals_56);  mul_134 = None
    add_92: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_135, primals_57);  mul_135 = primals_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_202: "f32[1568, 512]" = torch.ops.aten.view.default(add_92, [1568, 512]);  add_92 = None
    permute_98: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_208, [1, 0]);  primals_208 = None
    addmm_48: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_209, view_202, permute_98);  primals_209 = None
    view_203: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_48, [8, 1, 196, 1536]);  addmm_48 = None
    view_204: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_203, [8, 1, 196, 3, 16, 32]);  view_203 = None
    permute_99: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_204, [3, 0, 4, 1, 2, 5]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_12 = torch.ops.aten.unbind.int(permute_99);  permute_99 = None
    getitem_94: "f32[8, 16, 1, 196, 32]" = unbind_12[0]
    getitem_95: "f32[8, 16, 1, 196, 32]" = unbind_12[1]
    getitem_96: "f32[8, 16, 1, 196, 32]" = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_136: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_94, 0.42044820762685725);  getitem_94 = None
    permute_100: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_95, [0, 1, 2, 4, 3]);  getitem_95 = None
    mul_137: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_100, 0.42044820762685725);  permute_100 = None
    expand_48: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_136, [8, 16, 1, 196, 32]);  mul_136 = None
    clone_88: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_205: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_88, [128, 196, 32]);  clone_88 = None
    expand_49: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_137, [8, 16, 1, 32, 196]);  mul_137 = None
    clone_89: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_206: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_89, [128, 32, 196]);  clone_89 = None
    bmm_24: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_205, view_206)
    view_207: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_24, [8, 16, 1, 196, 196]);  bmm_24 = None
    amax_12: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_207, [-1], True)
    sub_39: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_207, amax_12);  view_207 = amax_12 = None
    exp_12: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_13: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_34: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_12: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_34)
    expand_50: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_34, [8, 16, 1, 196, 196]);  div_34 = None
    view_208: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_50, [128, 196, 196]);  expand_50 = None
    expand_51: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_96, [8, 16, 1, 196, 32]);  getitem_96 = None
    clone_90: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_209: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_90, [128, 196, 32]);  clone_90 = None
    bmm_25: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_208, view_209)
    view_210: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_25, [8, 16, 1, 196, 32]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_101: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_210, [0, 2, 3, 4, 1]);  view_210 = None
    clone_91: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    view_211: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_91, [8, 1, 196, 512]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_212: "f32[1568, 512]" = torch.ops.aten.view.default(view_211, [1568, 512]);  view_211 = None
    permute_102: "f32[512, 512]" = torch.ops.aten.permute.default(primals_210, [1, 0]);  primals_210 = None
    addmm_49: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_211, view_212, permute_102);  primals_211 = None
    view_213: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_49, [8, 1, 196, 512]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_92: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_213);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_22: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_22: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_22, 0.739130437374115);  empty_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_35: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_22, 0.739130437374115);  bernoulli_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_138: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_92, div_35);  clone_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_93: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_90, mul_138);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_93, [3], correction = 0, keepdim = True)
    getitem_97: "f32[8, 1, 196, 1]" = var_mean_27[0]
    getitem_98: "f32[8, 1, 196, 1]" = var_mean_27[1];  var_mean_27 = None
    add_94: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-06);  getitem_97 = None
    rsqrt_27: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_40: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_93, getitem_98)
    mul_139: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_27);  sub_40 = None
    mul_140: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_139, primals_58);  mul_139 = None
    add_95: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_140, primals_59);  mul_140 = primals_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_214: "f32[1568, 512]" = torch.ops.aten.view.default(add_95, [1568, 512]);  add_95 = None
    permute_103: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_212, [1, 0]);  primals_212 = None
    addmm_50: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_213, view_214, permute_103);  primals_213 = None
    view_215: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_50, [8, 1, 196, 2048]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_141: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_215, 0.5)
    mul_142: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_215, 0.7071067811865476)
    erf_12: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_142);  mul_142 = None
    add_96: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_143: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_141, add_96);  mul_141 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_93: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_143);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_216: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_93, [1568, 2048]);  clone_93 = None
    permute_104: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
    addmm_51: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_215, view_216, permute_104);  primals_215 = None
    view_217: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_51, [8, 1, 196, 512]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_94: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_217);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_23: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_23: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_23, 0.739130437374115);  empty_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_36: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_23, 0.739130437374115);  bernoulli_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_144: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_94, div_36);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_97: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_93, mul_144);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_97, [3], correction = 0, keepdim = True)
    getitem_99: "f32[8, 1, 196, 1]" = var_mean_28[0]
    getitem_100: "f32[8, 1, 196, 1]" = var_mean_28[1];  var_mean_28 = None
    add_98: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-06);  getitem_99 = None
    rsqrt_28: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_41: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_97, getitem_100)
    mul_145: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_28);  sub_41 = None
    mul_146: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_145, primals_60);  mul_145 = None
    add_99: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_146, primals_61);  mul_146 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_218: "f32[1568, 512]" = torch.ops.aten.view.default(add_99, [1568, 512]);  add_99 = None
    permute_105: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
    addmm_52: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_217, view_218, permute_105);  primals_217 = None
    view_219: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_52, [8, 1, 196, 1536]);  addmm_52 = None
    view_220: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_219, [8, 1, 196, 3, 16, 32]);  view_219 = None
    permute_106: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_220, [3, 0, 4, 1, 2, 5]);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_13 = torch.ops.aten.unbind.int(permute_106);  permute_106 = None
    getitem_101: "f32[8, 16, 1, 196, 32]" = unbind_13[0]
    getitem_102: "f32[8, 16, 1, 196, 32]" = unbind_13[1]
    getitem_103: "f32[8, 16, 1, 196, 32]" = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_147: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_101, 0.42044820762685725);  getitem_101 = None
    permute_107: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_102, [0, 1, 2, 4, 3]);  getitem_102 = None
    mul_148: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_107, 0.42044820762685725);  permute_107 = None
    expand_52: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_147, [8, 16, 1, 196, 32]);  mul_147 = None
    clone_95: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_221: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_95, [128, 196, 32]);  clone_95 = None
    expand_53: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_148, [8, 16, 1, 32, 196]);  mul_148 = None
    clone_96: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_222: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_96, [128, 32, 196]);  clone_96 = None
    bmm_26: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_221, view_222)
    view_223: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_26, [8, 16, 1, 196, 196]);  bmm_26 = None
    amax_13: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_223, [-1], True)
    sub_42: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_223, amax_13);  view_223 = amax_13 = None
    exp_13: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    sum_14: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_37: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_13: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_37)
    expand_54: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_37, [8, 16, 1, 196, 196]);  div_37 = None
    view_224: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_54, [128, 196, 196]);  expand_54 = None
    expand_55: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_103, [8, 16, 1, 196, 32]);  getitem_103 = None
    clone_97: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    view_225: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_97, [128, 196, 32]);  clone_97 = None
    bmm_27: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_224, view_225)
    view_226: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_27, [8, 16, 1, 196, 32]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_108: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_226, [0, 2, 3, 4, 1]);  view_226 = None
    clone_98: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
    view_227: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_98, [8, 1, 196, 512]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_228: "f32[1568, 512]" = torch.ops.aten.view.default(view_227, [1568, 512]);  view_227 = None
    permute_109: "f32[512, 512]" = torch.ops.aten.permute.default(primals_218, [1, 0]);  primals_218 = None
    addmm_53: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_219, view_228, permute_109);  primals_219 = None
    view_229: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_53, [8, 1, 196, 512]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_99: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_229);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_24: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_24: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_24, 0.717391312122345);  empty_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_38: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_24, 0.717391312122345);  bernoulli_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_149: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_99, div_38);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_100: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_97, mul_149);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_100, [3], correction = 0, keepdim = True)
    getitem_104: "f32[8, 1, 196, 1]" = var_mean_29[0]
    getitem_105: "f32[8, 1, 196, 1]" = var_mean_29[1];  var_mean_29 = None
    add_101: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
    rsqrt_29: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_43: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_100, getitem_105)
    mul_150: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_29);  sub_43 = None
    mul_151: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_150, primals_62);  mul_150 = None
    add_102: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_151, primals_63);  mul_151 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_230: "f32[1568, 512]" = torch.ops.aten.view.default(add_102, [1568, 512]);  add_102 = None
    permute_110: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
    addmm_54: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_221, view_230, permute_110);  primals_221 = None
    view_231: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_54, [8, 1, 196, 2048]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_152: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_231, 0.5)
    mul_153: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_231, 0.7071067811865476)
    erf_13: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
    add_103: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_154: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_152, add_103);  mul_152 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_100: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_154);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_232: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_100, [1568, 2048]);  clone_100 = None
    permute_111: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_222, [1, 0]);  primals_222 = None
    addmm_55: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_223, view_232, permute_111);  primals_223 = None
    view_233: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_55, [8, 1, 196, 512]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_101: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_233);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_25: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_25: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_25, 0.717391312122345);  empty_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_39: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_25, 0.717391312122345);  bernoulli_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_155: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_101, div_39);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_104: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_100, mul_155);  mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_104, [3], correction = 0, keepdim = True)
    getitem_106: "f32[8, 1, 196, 1]" = var_mean_30[0]
    getitem_107: "f32[8, 1, 196, 1]" = var_mean_30[1];  var_mean_30 = None
    add_105: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
    rsqrt_30: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_44: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_104, getitem_107)
    mul_156: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_30);  sub_44 = None
    mul_157: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_156, primals_64);  mul_156 = None
    add_106: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_157, primals_65);  mul_157 = primals_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_234: "f32[1568, 512]" = torch.ops.aten.view.default(add_106, [1568, 512]);  add_106 = None
    permute_112: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_224, [1, 0]);  primals_224 = None
    addmm_56: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_225, view_234, permute_112);  primals_225 = None
    view_235: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_56, [8, 1, 196, 1536]);  addmm_56 = None
    view_236: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_235, [8, 1, 196, 3, 16, 32]);  view_235 = None
    permute_113: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_236, [3, 0, 4, 1, 2, 5]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_14 = torch.ops.aten.unbind.int(permute_113);  permute_113 = None
    getitem_108: "f32[8, 16, 1, 196, 32]" = unbind_14[0]
    getitem_109: "f32[8, 16, 1, 196, 32]" = unbind_14[1]
    getitem_110: "f32[8, 16, 1, 196, 32]" = unbind_14[2];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_158: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_108, 0.42044820762685725);  getitem_108 = None
    permute_114: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_109, [0, 1, 2, 4, 3]);  getitem_109 = None
    mul_159: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_114, 0.42044820762685725);  permute_114 = None
    expand_56: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_158, [8, 16, 1, 196, 32]);  mul_158 = None
    clone_102: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_237: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_102, [128, 196, 32]);  clone_102 = None
    expand_57: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_159, [8, 16, 1, 32, 196]);  mul_159 = None
    clone_103: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_238: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_103, [128, 32, 196]);  clone_103 = None
    bmm_28: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_237, view_238)
    view_239: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_28, [8, 16, 1, 196, 196]);  bmm_28 = None
    amax_14: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_239, [-1], True)
    sub_45: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_239, amax_14);  view_239 = amax_14 = None
    exp_14: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_15: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_40: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_14: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_40)
    expand_58: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_40, [8, 16, 1, 196, 196]);  div_40 = None
    view_240: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_58, [128, 196, 196]);  expand_58 = None
    expand_59: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_110, [8, 16, 1, 196, 32]);  getitem_110 = None
    clone_104: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_241: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_104, [128, 196, 32]);  clone_104 = None
    bmm_29: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_240, view_241)
    view_242: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_29, [8, 16, 1, 196, 32]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_115: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_242, [0, 2, 3, 4, 1]);  view_242 = None
    clone_105: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    view_243: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_105, [8, 1, 196, 512]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_244: "f32[1568, 512]" = torch.ops.aten.view.default(view_243, [1568, 512]);  view_243 = None
    permute_116: "f32[512, 512]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
    addmm_57: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_227, view_244, permute_116);  primals_227 = None
    view_245: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_57, [8, 1, 196, 512]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_106: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_245);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_26: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_26: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_26, 0.695652186870575);  empty_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_41: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_26, 0.695652186870575);  bernoulli_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_160: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_106, div_41);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_107: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_104, mul_160);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_107, [3], correction = 0, keepdim = True)
    getitem_111: "f32[8, 1, 196, 1]" = var_mean_31[0]
    getitem_112: "f32[8, 1, 196, 1]" = var_mean_31[1];  var_mean_31 = None
    add_108: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_111, 1e-06);  getitem_111 = None
    rsqrt_31: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_46: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_107, getitem_112)
    mul_161: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_31);  sub_46 = None
    mul_162: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_161, primals_66);  mul_161 = None
    add_109: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_162, primals_67);  mul_162 = primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_246: "f32[1568, 512]" = torch.ops.aten.view.default(add_109, [1568, 512]);  add_109 = None
    permute_117: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_228, [1, 0]);  primals_228 = None
    addmm_58: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_229, view_246, permute_117);  primals_229 = None
    view_247: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_58, [8, 1, 196, 2048]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_163: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_247, 0.5)
    mul_164: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_247, 0.7071067811865476)
    erf_14: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
    add_110: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_165: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_163, add_110);  mul_163 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_107: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_165);  mul_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_248: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_107, [1568, 2048]);  clone_107 = None
    permute_118: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_230, [1, 0]);  primals_230 = None
    addmm_59: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_231, view_248, permute_118);  primals_231 = None
    view_249: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_59, [8, 1, 196, 512]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_108: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_249);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_27: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_27: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_27, 0.695652186870575);  empty_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_42: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_27, 0.695652186870575);  bernoulli_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_166: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_108, div_42);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_111: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_107, mul_166);  mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_111, [3], correction = 0, keepdim = True)
    getitem_113: "f32[8, 1, 196, 1]" = var_mean_32[0]
    getitem_114: "f32[8, 1, 196, 1]" = var_mean_32[1];  var_mean_32 = None
    add_112: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_113, 1e-06);  getitem_113 = None
    rsqrt_32: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_47: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_111, getitem_114)
    mul_167: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_32);  sub_47 = None
    mul_168: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_167, primals_68);  mul_167 = None
    add_113: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_168, primals_69);  mul_168 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_250: "f32[1568, 512]" = torch.ops.aten.view.default(add_113, [1568, 512]);  add_113 = None
    permute_119: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
    addmm_60: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_233, view_250, permute_119);  primals_233 = None
    view_251: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_60, [8, 1, 196, 1536]);  addmm_60 = None
    view_252: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_251, [8, 1, 196, 3, 16, 32]);  view_251 = None
    permute_120: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_252, [3, 0, 4, 1, 2, 5]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_15 = torch.ops.aten.unbind.int(permute_120);  permute_120 = None
    getitem_115: "f32[8, 16, 1, 196, 32]" = unbind_15[0]
    getitem_116: "f32[8, 16, 1, 196, 32]" = unbind_15[1]
    getitem_117: "f32[8, 16, 1, 196, 32]" = unbind_15[2];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_169: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_115, 0.42044820762685725);  getitem_115 = None
    permute_121: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_116, [0, 1, 2, 4, 3]);  getitem_116 = None
    mul_170: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_121, 0.42044820762685725);  permute_121 = None
    expand_60: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_169, [8, 16, 1, 196, 32]);  mul_169 = None
    clone_109: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_253: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_109, [128, 196, 32]);  clone_109 = None
    expand_61: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_170, [8, 16, 1, 32, 196]);  mul_170 = None
    clone_110: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_254: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_110, [128, 32, 196]);  clone_110 = None
    bmm_30: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_253, view_254)
    view_255: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_30, [8, 16, 1, 196, 196]);  bmm_30 = None
    amax_15: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_255, [-1], True)
    sub_48: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_255, amax_15);  view_255 = amax_15 = None
    exp_15: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_48);  sub_48 = None
    sum_16: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_43: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_15: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_43)
    expand_62: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_43, [8, 16, 1, 196, 196]);  div_43 = None
    view_256: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_62, [128, 196, 196]);  expand_62 = None
    expand_63: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_117, [8, 16, 1, 196, 32]);  getitem_117 = None
    clone_111: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_257: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_111, [128, 196, 32]);  clone_111 = None
    bmm_31: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_256, view_257)
    view_258: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_31, [8, 16, 1, 196, 32]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_122: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_258, [0, 2, 3, 4, 1]);  view_258 = None
    clone_112: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_259: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_112, [8, 1, 196, 512]);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_260: "f32[1568, 512]" = torch.ops.aten.view.default(view_259, [1568, 512]);  view_259 = None
    permute_123: "f32[512, 512]" = torch.ops.aten.permute.default(primals_234, [1, 0]);  primals_234 = None
    addmm_61: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_235, view_260, permute_123);  primals_235 = None
    view_261: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_61, [8, 1, 196, 512]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_113: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_261);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_28: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_28: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_28, 0.6739130616188049);  empty_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_44: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_28, 0.6739130616188049);  bernoulli_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_171: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_113, div_44);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_114: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_111, mul_171);  mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_114, [3], correction = 0, keepdim = True)
    getitem_118: "f32[8, 1, 196, 1]" = var_mean_33[0]
    getitem_119: "f32[8, 1, 196, 1]" = var_mean_33[1];  var_mean_33 = None
    add_115: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-06);  getitem_118 = None
    rsqrt_33: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_49: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_114, getitem_119)
    mul_172: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_33);  sub_49 = None
    mul_173: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_172, primals_70);  mul_172 = None
    add_116: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_173, primals_71);  mul_173 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_262: "f32[1568, 512]" = torch.ops.aten.view.default(add_116, [1568, 512]);  add_116 = None
    permute_124: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_236, [1, 0]);  primals_236 = None
    addmm_62: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_237, view_262, permute_124);  primals_237 = None
    view_263: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_62, [8, 1, 196, 2048]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_174: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    mul_175: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476)
    erf_15: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_175);  mul_175 = None
    add_117: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_176: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_174, add_117);  mul_174 = add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_114: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_176);  mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_264: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_114, [1568, 2048]);  clone_114 = None
    permute_125: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_238, [1, 0]);  primals_238 = None
    addmm_63: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_239, view_264, permute_125);  primals_239 = None
    view_265: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_63, [8, 1, 196, 512]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_115: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_265);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_29: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_29: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_29, 0.6739130616188049);  empty_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_45: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_29, 0.6739130616188049);  bernoulli_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_177: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_115, div_45);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_118: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_114, mul_177);  mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_118, [3], correction = 0, keepdim = True)
    getitem_120: "f32[8, 1, 196, 1]" = var_mean_34[0]
    getitem_121: "f32[8, 1, 196, 1]" = var_mean_34[1];  var_mean_34 = None
    add_119: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
    rsqrt_34: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_50: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_118, getitem_121)
    mul_178: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_34);  sub_50 = None
    mul_179: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_178, primals_72);  mul_178 = None
    add_120: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_179, primals_73);  mul_179 = primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_266: "f32[1568, 512]" = torch.ops.aten.view.default(add_120, [1568, 512]);  add_120 = None
    permute_126: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
    addmm_64: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_241, view_266, permute_126);  primals_241 = None
    view_267: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_64, [8, 1, 196, 1536]);  addmm_64 = None
    view_268: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_267, [8, 1, 196, 3, 16, 32]);  view_267 = None
    permute_127: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_268, [3, 0, 4, 1, 2, 5]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_16 = torch.ops.aten.unbind.int(permute_127);  permute_127 = None
    getitem_122: "f32[8, 16, 1, 196, 32]" = unbind_16[0]
    getitem_123: "f32[8, 16, 1, 196, 32]" = unbind_16[1]
    getitem_124: "f32[8, 16, 1, 196, 32]" = unbind_16[2];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_180: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_122, 0.42044820762685725);  getitem_122 = None
    permute_128: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_123, [0, 1, 2, 4, 3]);  getitem_123 = None
    mul_181: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_128, 0.42044820762685725);  permute_128 = None
    expand_64: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_180, [8, 16, 1, 196, 32]);  mul_180 = None
    clone_116: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_269: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_116, [128, 196, 32]);  clone_116 = None
    expand_65: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_181, [8, 16, 1, 32, 196]);  mul_181 = None
    clone_117: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_270: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_117, [128, 32, 196]);  clone_117 = None
    bmm_32: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_269, view_270)
    view_271: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_32, [8, 16, 1, 196, 196]);  bmm_32 = None
    amax_16: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_271, [-1], True)
    sub_51: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_271, amax_16);  view_271 = amax_16 = None
    exp_16: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_17: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_46: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_16: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_46)
    expand_66: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_46, [8, 16, 1, 196, 196]);  div_46 = None
    view_272: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_66, [128, 196, 196]);  expand_66 = None
    expand_67: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_124, [8, 16, 1, 196, 32]);  getitem_124 = None
    clone_118: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
    view_273: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_118, [128, 196, 32]);  clone_118 = None
    bmm_33: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_272, view_273)
    view_274: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_33, [8, 16, 1, 196, 32]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_129: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_274, [0, 2, 3, 4, 1]);  view_274 = None
    clone_119: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    view_275: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_119, [8, 1, 196, 512]);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_276: "f32[1568, 512]" = torch.ops.aten.view.default(view_275, [1568, 512]);  view_275 = None
    permute_130: "f32[512, 512]" = torch.ops.aten.permute.default(primals_242, [1, 0]);  primals_242 = None
    addmm_65: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_243, view_276, permute_130);  primals_243 = None
    view_277: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_65, [8, 1, 196, 512]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_120: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_277);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_30: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_30: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_30, 0.6521739065647125);  empty_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_47: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_30, 0.6521739065647125);  bernoulli_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_182: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_120, div_47);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_121: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_118, mul_182);  mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_121, [3], correction = 0, keepdim = True)
    getitem_125: "f32[8, 1, 196, 1]" = var_mean_35[0]
    getitem_126: "f32[8, 1, 196, 1]" = var_mean_35[1];  var_mean_35 = None
    add_122: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_125, 1e-06);  getitem_125 = None
    rsqrt_35: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_52: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_121, getitem_126)
    mul_183: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_35);  sub_52 = None
    mul_184: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_183, primals_74);  mul_183 = None
    add_123: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_184, primals_75);  mul_184 = primals_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_278: "f32[1568, 512]" = torch.ops.aten.view.default(add_123, [1568, 512]);  add_123 = None
    permute_131: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_244, [1, 0]);  primals_244 = None
    addmm_66: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_245, view_278, permute_131);  primals_245 = None
    view_279: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_66, [8, 1, 196, 2048]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_185: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_279, 0.5)
    mul_186: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_279, 0.7071067811865476)
    erf_16: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_186);  mul_186 = None
    add_124: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_187: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_185, add_124);  mul_185 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_121: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_187);  mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_280: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_121, [1568, 2048]);  clone_121 = None
    permute_132: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_246, [1, 0]);  primals_246 = None
    addmm_67: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_247, view_280, permute_132);  primals_247 = None
    view_281: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_67, [8, 1, 196, 512]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_122: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_281);  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_31: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_31: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_31, 0.6521739065647125);  empty_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_48: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_31, 0.6521739065647125);  bernoulli_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_188: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_122, div_48);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_125: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_121, mul_188);  mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_125, [3], correction = 0, keepdim = True)
    getitem_127: "f32[8, 1, 196, 1]" = var_mean_36[0]
    getitem_128: "f32[8, 1, 196, 1]" = var_mean_36[1];  var_mean_36 = None
    add_126: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_127, 1e-06);  getitem_127 = None
    rsqrt_36: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_53: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_125, getitem_128)
    mul_189: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_36);  sub_53 = None
    mul_190: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_189, primals_76);  mul_189 = None
    add_127: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_190, primals_77);  mul_190 = primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_282: "f32[1568, 512]" = torch.ops.aten.view.default(add_127, [1568, 512]);  add_127 = None
    permute_133: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
    addmm_68: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_249, view_282, permute_133);  primals_249 = None
    view_283: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_68, [8, 1, 196, 1536]);  addmm_68 = None
    view_284: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_283, [8, 1, 196, 3, 16, 32]);  view_283 = None
    permute_134: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_284, [3, 0, 4, 1, 2, 5]);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_17 = torch.ops.aten.unbind.int(permute_134);  permute_134 = None
    getitem_129: "f32[8, 16, 1, 196, 32]" = unbind_17[0]
    getitem_130: "f32[8, 16, 1, 196, 32]" = unbind_17[1]
    getitem_131: "f32[8, 16, 1, 196, 32]" = unbind_17[2];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_191: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_129, 0.42044820762685725);  getitem_129 = None
    permute_135: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_130, [0, 1, 2, 4, 3]);  getitem_130 = None
    mul_192: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_135, 0.42044820762685725);  permute_135 = None
    expand_68: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_191, [8, 16, 1, 196, 32]);  mul_191 = None
    clone_123: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_285: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_123, [128, 196, 32]);  clone_123 = None
    expand_69: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_192, [8, 16, 1, 32, 196]);  mul_192 = None
    clone_124: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    view_286: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_124, [128, 32, 196]);  clone_124 = None
    bmm_34: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_285, view_286)
    view_287: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_34, [8, 16, 1, 196, 196]);  bmm_34 = None
    amax_17: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_287, [-1], True)
    sub_54: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_287, amax_17);  view_287 = amax_17 = None
    exp_17: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_54);  sub_54 = None
    sum_18: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_49: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_17: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_49)
    expand_70: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_49, [8, 16, 1, 196, 196]);  div_49 = None
    view_288: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_70, [128, 196, 196]);  expand_70 = None
    expand_71: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_131, [8, 16, 1, 196, 32]);  getitem_131 = None
    clone_125: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    view_289: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_125, [128, 196, 32]);  clone_125 = None
    bmm_35: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_288, view_289)
    view_290: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_35, [8, 16, 1, 196, 32]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_136: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_290, [0, 2, 3, 4, 1]);  view_290 = None
    clone_126: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    view_291: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_126, [8, 1, 196, 512]);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_292: "f32[1568, 512]" = torch.ops.aten.view.default(view_291, [1568, 512]);  view_291 = None
    permute_137: "f32[512, 512]" = torch.ops.aten.permute.default(primals_250, [1, 0]);  primals_250 = None
    addmm_69: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_251, view_292, permute_137);  primals_251 = None
    view_293: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_69, [8, 1, 196, 512]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_127: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_293);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_32: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_32: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_32, 0.6304347813129425);  empty_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_50: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_32, 0.6304347813129425);  bernoulli_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_193: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_127, div_50);  clone_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_128: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_125, mul_193);  mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_128, [3], correction = 0, keepdim = True)
    getitem_132: "f32[8, 1, 196, 1]" = var_mean_37[0]
    getitem_133: "f32[8, 1, 196, 1]" = var_mean_37[1];  var_mean_37 = None
    add_129: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_37: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_55: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_128, getitem_133)
    mul_194: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_37);  sub_55 = None
    mul_195: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_194, primals_78);  mul_194 = None
    add_130: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_195, primals_79);  mul_195 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_294: "f32[1568, 512]" = torch.ops.aten.view.default(add_130, [1568, 512]);  add_130 = None
    permute_138: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_252, [1, 0]);  primals_252 = None
    addmm_70: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_253, view_294, permute_138);  primals_253 = None
    view_295: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_70, [8, 1, 196, 2048]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_196: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_295, 0.5)
    mul_197: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_295, 0.7071067811865476)
    erf_17: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_197);  mul_197 = None
    add_131: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_198: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_196, add_131);  mul_196 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_128: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_198);  mul_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_296: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_128, [1568, 2048]);  clone_128 = None
    permute_139: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_254, [1, 0]);  primals_254 = None
    addmm_71: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_255, view_296, permute_139);  primals_255 = None
    view_297: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_71, [8, 1, 196, 512]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_129: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_297);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_33: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_33: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_33, 0.6304347813129425);  empty_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_51: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_33, 0.6304347813129425);  bernoulli_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_199: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_129, div_51);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_132: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_128, mul_199);  mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_132, [3], correction = 0, keepdim = True)
    getitem_134: "f32[8, 1, 196, 1]" = var_mean_38[0]
    getitem_135: "f32[8, 1, 196, 1]" = var_mean_38[1];  var_mean_38 = None
    add_133: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-06);  getitem_134 = None
    rsqrt_38: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_56: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_132, getitem_135)
    mul_200: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_38);  sub_56 = None
    mul_201: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_200, primals_80);  mul_200 = None
    add_134: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_201, primals_81);  mul_201 = primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_298: "f32[1568, 512]" = torch.ops.aten.view.default(add_134, [1568, 512]);  add_134 = None
    permute_140: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_256, [1, 0]);  primals_256 = None
    addmm_72: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_257, view_298, permute_140);  primals_257 = None
    view_299: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_72, [8, 1, 196, 1536]);  addmm_72 = None
    view_300: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_299, [8, 1, 196, 3, 16, 32]);  view_299 = None
    permute_141: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_300, [3, 0, 4, 1, 2, 5]);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_18 = torch.ops.aten.unbind.int(permute_141);  permute_141 = None
    getitem_136: "f32[8, 16, 1, 196, 32]" = unbind_18[0]
    getitem_137: "f32[8, 16, 1, 196, 32]" = unbind_18[1]
    getitem_138: "f32[8, 16, 1, 196, 32]" = unbind_18[2];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_202: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_136, 0.42044820762685725);  getitem_136 = None
    permute_142: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_137, [0, 1, 2, 4, 3]);  getitem_137 = None
    mul_203: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_142, 0.42044820762685725);  permute_142 = None
    expand_72: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_202, [8, 16, 1, 196, 32]);  mul_202 = None
    clone_130: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_301: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_130, [128, 196, 32]);  clone_130 = None
    expand_73: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_203, [8, 16, 1, 32, 196]);  mul_203 = None
    clone_131: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
    view_302: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_131, [128, 32, 196]);  clone_131 = None
    bmm_36: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_301, view_302)
    view_303: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_36, [8, 16, 1, 196, 196]);  bmm_36 = None
    amax_18: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_303, [-1], True)
    sub_57: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_303, amax_18);  view_303 = amax_18 = None
    exp_18: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_19: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_52: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_18: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_52)
    expand_74: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_52, [8, 16, 1, 196, 196]);  div_52 = None
    view_304: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_74, [128, 196, 196]);  expand_74 = None
    expand_75: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_138, [8, 16, 1, 196, 32]);  getitem_138 = None
    clone_132: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    view_305: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_132, [128, 196, 32]);  clone_132 = None
    bmm_37: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_304, view_305)
    view_306: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_37, [8, 16, 1, 196, 32]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_143: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_306, [0, 2, 3, 4, 1]);  view_306 = None
    clone_133: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
    view_307: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_133, [8, 1, 196, 512]);  clone_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_308: "f32[1568, 512]" = torch.ops.aten.view.default(view_307, [1568, 512]);  view_307 = None
    permute_144: "f32[512, 512]" = torch.ops.aten.permute.default(primals_258, [1, 0]);  primals_258 = None
    addmm_73: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_259, view_308, permute_144);  primals_259 = None
    view_309: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_73, [8, 1, 196, 512]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_134: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_309);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_34: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_34: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_34, 0.6086956560611725);  empty_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_53: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_34, 0.6086956560611725);  bernoulli_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_204: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_134, div_53);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_135: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_132, mul_204);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_135, [3], correction = 0, keepdim = True)
    getitem_139: "f32[8, 1, 196, 1]" = var_mean_39[0]
    getitem_140: "f32[8, 1, 196, 1]" = var_mean_39[1];  var_mean_39 = None
    add_136: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_139, 1e-06);  getitem_139 = None
    rsqrt_39: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_58: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_135, getitem_140)
    mul_205: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_39);  sub_58 = None
    mul_206: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_205, primals_82);  mul_205 = None
    add_137: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_206, primals_83);  mul_206 = primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_310: "f32[1568, 512]" = torch.ops.aten.view.default(add_137, [1568, 512]);  add_137 = None
    permute_145: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_260, [1, 0]);  primals_260 = None
    addmm_74: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_261, view_310, permute_145);  primals_261 = None
    view_311: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_74, [8, 1, 196, 2048]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_207: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_311, 0.5)
    mul_208: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_311, 0.7071067811865476)
    erf_18: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_208);  mul_208 = None
    add_138: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_209: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_207, add_138);  mul_207 = add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_135: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_209);  mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_312: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_135, [1568, 2048]);  clone_135 = None
    permute_146: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_262, [1, 0]);  primals_262 = None
    addmm_75: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_263, view_312, permute_146);  primals_263 = None
    view_313: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_75, [8, 1, 196, 512]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_136: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_313);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_35: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_35: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_35, 0.6086956560611725);  empty_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_54: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_35, 0.6086956560611725);  bernoulli_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_210: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_136, div_54);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_139: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_135, mul_210);  mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_139, [3], correction = 0, keepdim = True)
    getitem_141: "f32[8, 1, 196, 1]" = var_mean_40[0]
    getitem_142: "f32[8, 1, 196, 1]" = var_mean_40[1];  var_mean_40 = None
    add_140: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_141, 1e-06);  getitem_141 = None
    rsqrt_40: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_59: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_139, getitem_142)
    mul_211: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_40);  sub_59 = None
    mul_212: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_211, primals_84);  mul_211 = None
    add_141: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_212, primals_85);  mul_212 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_314: "f32[1568, 512]" = torch.ops.aten.view.default(add_141, [1568, 512]);  add_141 = None
    permute_147: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_264, [1, 0]);  primals_264 = None
    addmm_76: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_265, view_314, permute_147);  primals_265 = None
    view_315: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_76, [8, 1, 196, 1536]);  addmm_76 = None
    view_316: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_315, [8, 1, 196, 3, 16, 32]);  view_315 = None
    permute_148: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_316, [3, 0, 4, 1, 2, 5]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_19 = torch.ops.aten.unbind.int(permute_148);  permute_148 = None
    getitem_143: "f32[8, 16, 1, 196, 32]" = unbind_19[0]
    getitem_144: "f32[8, 16, 1, 196, 32]" = unbind_19[1]
    getitem_145: "f32[8, 16, 1, 196, 32]" = unbind_19[2];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_213: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_143, 0.42044820762685725);  getitem_143 = None
    permute_149: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_144, [0, 1, 2, 4, 3]);  getitem_144 = None
    mul_214: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_149, 0.42044820762685725);  permute_149 = None
    expand_76: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_213, [8, 16, 1, 196, 32]);  mul_213 = None
    clone_137: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
    view_317: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_137, [128, 196, 32]);  clone_137 = None
    expand_77: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_214, [8, 16, 1, 32, 196]);  mul_214 = None
    clone_138: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
    view_318: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_138, [128, 32, 196]);  clone_138 = None
    bmm_38: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_317, view_318)
    view_319: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_38, [8, 16, 1, 196, 196]);  bmm_38 = None
    amax_19: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_319, [-1], True)
    sub_60: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_319, amax_19);  view_319 = amax_19 = None
    exp_19: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
    sum_20: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_55: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    alias_19: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_55)
    expand_78: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_55, [8, 16, 1, 196, 196]);  div_55 = None
    view_320: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_78, [128, 196, 196]);  expand_78 = None
    expand_79: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_145, [8, 16, 1, 196, 32]);  getitem_145 = None
    clone_139: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
    view_321: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_139, [128, 196, 32]);  clone_139 = None
    bmm_39: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_320, view_321)
    view_322: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_39, [8, 16, 1, 196, 32]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_150: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_322, [0, 2, 3, 4, 1]);  view_322 = None
    clone_140: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    view_323: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_140, [8, 1, 196, 512]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_324: "f32[1568, 512]" = torch.ops.aten.view.default(view_323, [1568, 512]);  view_323 = None
    permute_151: "f32[512, 512]" = torch.ops.aten.permute.default(primals_266, [1, 0]);  primals_266 = None
    addmm_77: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_267, view_324, permute_151);  primals_267 = None
    view_325: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_77, [8, 1, 196, 512]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_141: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_325);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_36: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_36: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_36, 0.5869565308094025);  empty_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_56: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_36, 0.5869565308094025);  bernoulli_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_215: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_141, div_56);  clone_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_142: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_139, mul_215);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_142, [3], correction = 0, keepdim = True)
    getitem_146: "f32[8, 1, 196, 1]" = var_mean_41[0]
    getitem_147: "f32[8, 1, 196, 1]" = var_mean_41[1];  var_mean_41 = None
    add_143: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-06);  getitem_146 = None
    rsqrt_41: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_61: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_142, getitem_147)
    mul_216: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_41);  sub_61 = None
    mul_217: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_216, primals_86);  mul_216 = None
    add_144: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_217, primals_87);  mul_217 = primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_326: "f32[1568, 512]" = torch.ops.aten.view.default(add_144, [1568, 512]);  add_144 = None
    permute_152: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_268, [1, 0]);  primals_268 = None
    addmm_78: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_269, view_326, permute_152);  primals_269 = None
    view_327: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_78, [8, 1, 196, 2048]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_218: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
    mul_219: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476)
    erf_19: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_219);  mul_219 = None
    add_145: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_220: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_218, add_145);  mul_218 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_142: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_220);  mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_328: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_142, [1568, 2048]);  clone_142 = None
    permute_153: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_270, [1, 0]);  primals_270 = None
    addmm_79: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_271, view_328, permute_153);  primals_271 = None
    view_329: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_79, [8, 1, 196, 512]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_143: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_329);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_37: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_37: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_37, 0.5869565308094025);  empty_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_57: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_37, 0.5869565308094025);  bernoulli_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_221: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_143, div_57);  clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_146: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_142, mul_221);  mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_146, [3], correction = 0, keepdim = True)
    getitem_148: "f32[8, 1, 196, 1]" = var_mean_42[0]
    getitem_149: "f32[8, 1, 196, 1]" = var_mean_42[1];  var_mean_42 = None
    add_147: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-06);  getitem_148 = None
    rsqrt_42: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_62: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_146, getitem_149)
    mul_222: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_42);  sub_62 = None
    mul_223: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_222, primals_88);  mul_222 = None
    add_148: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_223, primals_89);  mul_223 = primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_330: "f32[1568, 512]" = torch.ops.aten.view.default(add_148, [1568, 512]);  add_148 = None
    permute_154: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_272, [1, 0]);  primals_272 = None
    addmm_80: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_273, view_330, permute_154);  primals_273 = None
    view_331: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_80, [8, 1, 196, 1536]);  addmm_80 = None
    view_332: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_331, [8, 1, 196, 3, 16, 32]);  view_331 = None
    permute_155: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_332, [3, 0, 4, 1, 2, 5]);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_20 = torch.ops.aten.unbind.int(permute_155);  permute_155 = None
    getitem_150: "f32[8, 16, 1, 196, 32]" = unbind_20[0]
    getitem_151: "f32[8, 16, 1, 196, 32]" = unbind_20[1]
    getitem_152: "f32[8, 16, 1, 196, 32]" = unbind_20[2];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_224: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_150, 0.42044820762685725);  getitem_150 = None
    permute_156: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_151, [0, 1, 2, 4, 3]);  getitem_151 = None
    mul_225: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_156, 0.42044820762685725);  permute_156 = None
    expand_80: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_224, [8, 16, 1, 196, 32]);  mul_224 = None
    clone_144: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    view_333: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_144, [128, 196, 32]);  clone_144 = None
    expand_81: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_225, [8, 16, 1, 32, 196]);  mul_225 = None
    clone_145: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    view_334: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_145, [128, 32, 196]);  clone_145 = None
    bmm_40: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_333, view_334)
    view_335: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_40, [8, 16, 1, 196, 196]);  bmm_40 = None
    amax_20: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_335, [-1], True)
    sub_63: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_335, amax_20);  view_335 = amax_20 = None
    exp_20: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
    sum_21: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_58: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    alias_20: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_58)
    expand_82: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_58, [8, 16, 1, 196, 196]);  div_58 = None
    view_336: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_82, [128, 196, 196]);  expand_82 = None
    expand_83: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_152, [8, 16, 1, 196, 32]);  getitem_152 = None
    clone_146: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
    view_337: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_146, [128, 196, 32]);  clone_146 = None
    bmm_41: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_336, view_337)
    view_338: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_41, [8, 16, 1, 196, 32]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_157: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_338, [0, 2, 3, 4, 1]);  view_338 = None
    clone_147: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    view_339: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_147, [8, 1, 196, 512]);  clone_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_340: "f32[1568, 512]" = torch.ops.aten.view.default(view_339, [1568, 512]);  view_339 = None
    permute_158: "f32[512, 512]" = torch.ops.aten.permute.default(primals_274, [1, 0]);  primals_274 = None
    addmm_81: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_275, view_340, permute_158);  primals_275 = None
    view_341: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_81, [8, 1, 196, 512]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_148: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_341);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_38: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_38: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_38, 0.5652174055576324);  empty_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_59: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_38, 0.5652174055576324);  bernoulli_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_226: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_148, div_59);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_149: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_146, mul_226);  mul_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_149, [3], correction = 0, keepdim = True)
    getitem_153: "f32[8, 1, 196, 1]" = var_mean_43[0]
    getitem_154: "f32[8, 1, 196, 1]" = var_mean_43[1];  var_mean_43 = None
    add_150: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_153, 1e-06);  getitem_153 = None
    rsqrt_43: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_64: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_149, getitem_154)
    mul_227: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_43);  sub_64 = None
    mul_228: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_227, primals_90);  mul_227 = None
    add_151: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_228, primals_91);  mul_228 = primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_342: "f32[1568, 512]" = torch.ops.aten.view.default(add_151, [1568, 512]);  add_151 = None
    permute_159: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_276, [1, 0]);  primals_276 = None
    addmm_82: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_277, view_342, permute_159);  primals_277 = None
    view_343: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_82, [8, 1, 196, 2048]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_229: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_343, 0.5)
    mul_230: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_343, 0.7071067811865476)
    erf_20: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_230);  mul_230 = None
    add_152: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_231: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_229, add_152);  mul_229 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_149: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_231);  mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_344: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_149, [1568, 2048]);  clone_149 = None
    permute_160: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_278, [1, 0]);  primals_278 = None
    addmm_83: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_279, view_344, permute_160);  primals_279 = None
    view_345: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_83, [8, 1, 196, 512]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_150: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_345);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_39: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_39: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_39, 0.5652174055576324);  empty_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_60: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_39, 0.5652174055576324);  bernoulli_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_232: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_150, div_60);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_153: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_149, mul_232);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_153, [3], correction = 0, keepdim = True)
    getitem_155: "f32[8, 1, 196, 1]" = var_mean_44[0]
    getitem_156: "f32[8, 1, 196, 1]" = var_mean_44[1];  var_mean_44 = None
    add_154: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_155, 1e-06);  getitem_155 = None
    rsqrt_44: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_65: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_153, getitem_156)
    mul_233: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_44);  sub_65 = None
    mul_234: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_233, primals_92);  mul_233 = None
    add_155: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_234, primals_93);  mul_234 = primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_346: "f32[1568, 512]" = torch.ops.aten.view.default(add_155, [1568, 512]);  add_155 = None
    permute_161: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_280, [1, 0]);  primals_280 = None
    addmm_84: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_281, view_346, permute_161);  primals_281 = None
    view_347: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_84, [8, 1, 196, 1536]);  addmm_84 = None
    view_348: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_347, [8, 1, 196, 3, 16, 32]);  view_347 = None
    permute_162: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_348, [3, 0, 4, 1, 2, 5]);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_21 = torch.ops.aten.unbind.int(permute_162);  permute_162 = None
    getitem_157: "f32[8, 16, 1, 196, 32]" = unbind_21[0]
    getitem_158: "f32[8, 16, 1, 196, 32]" = unbind_21[1]
    getitem_159: "f32[8, 16, 1, 196, 32]" = unbind_21[2];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_235: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_157, 0.42044820762685725);  getitem_157 = None
    permute_163: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_158, [0, 1, 2, 4, 3]);  getitem_158 = None
    mul_236: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_163, 0.42044820762685725);  permute_163 = None
    expand_84: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_235, [8, 16, 1, 196, 32]);  mul_235 = None
    clone_151: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
    view_349: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_151, [128, 196, 32]);  clone_151 = None
    expand_85: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_236, [8, 16, 1, 32, 196]);  mul_236 = None
    clone_152: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
    view_350: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_152, [128, 32, 196]);  clone_152 = None
    bmm_42: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_349, view_350)
    view_351: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_42, [8, 16, 1, 196, 196]);  bmm_42 = None
    amax_21: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_351, [-1], True)
    sub_66: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_351, amax_21);  view_351 = amax_21 = None
    exp_21: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_66);  sub_66 = None
    sum_22: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_61: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    alias_21: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_61)
    expand_86: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_61, [8, 16, 1, 196, 196]);  div_61 = None
    view_352: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_86, [128, 196, 196]);  expand_86 = None
    expand_87: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_159, [8, 16, 1, 196, 32]);  getitem_159 = None
    clone_153: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
    view_353: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_153, [128, 196, 32]);  clone_153 = None
    bmm_43: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_352, view_353)
    view_354: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_43, [8, 16, 1, 196, 32]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_164: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_354, [0, 2, 3, 4, 1]);  view_354 = None
    clone_154: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    view_355: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_154, [8, 1, 196, 512]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_356: "f32[1568, 512]" = torch.ops.aten.view.default(view_355, [1568, 512]);  view_355 = None
    permute_165: "f32[512, 512]" = torch.ops.aten.permute.default(primals_282, [1, 0]);  primals_282 = None
    addmm_85: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_283, view_356, permute_165);  primals_283 = None
    view_357: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_85, [8, 1, 196, 512]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_155: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_357);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_40: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_40: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_40, 0.54347825050354);  empty_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_62: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_40, 0.54347825050354);  bernoulli_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_237: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_155, div_62);  clone_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_156: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_153, mul_237);  mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_156, [3], correction = 0, keepdim = True)
    getitem_160: "f32[8, 1, 196, 1]" = var_mean_45[0]
    getitem_161: "f32[8, 1, 196, 1]" = var_mean_45[1];  var_mean_45 = None
    add_157: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
    rsqrt_45: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_67: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_156, getitem_161)
    mul_238: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_45);  sub_67 = None
    mul_239: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_238, primals_94);  mul_238 = None
    add_158: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_239, primals_95);  mul_239 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_358: "f32[1568, 512]" = torch.ops.aten.view.default(add_158, [1568, 512]);  add_158 = None
    permute_166: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_284, [1, 0]);  primals_284 = None
    addmm_86: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_285, view_358, permute_166);  primals_285 = None
    view_359: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_86, [8, 1, 196, 2048]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_240: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_359, 0.5)
    mul_241: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_359, 0.7071067811865476)
    erf_21: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_241);  mul_241 = None
    add_159: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_242: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_240, add_159);  mul_240 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_156: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_242);  mul_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_360: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_156, [1568, 2048]);  clone_156 = None
    permute_167: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_286, [1, 0]);  primals_286 = None
    addmm_87: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_287, view_360, permute_167);  primals_287 = None
    view_361: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_87, [8, 1, 196, 512]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_157: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_361);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_41: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_41: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_41, 0.54347825050354);  empty_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_63: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_41, 0.54347825050354);  bernoulli_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_243: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_157, div_63);  clone_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_160: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_156, mul_243);  mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_160, [3], correction = 0, keepdim = True)
    getitem_162: "f32[8, 1, 196, 1]" = var_mean_46[0]
    getitem_163: "f32[8, 1, 196, 1]" = var_mean_46[1];  var_mean_46 = None
    add_161: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
    rsqrt_46: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_68: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_160, getitem_163)
    mul_244: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_46);  sub_68 = None
    mul_245: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_244, primals_96);  mul_244 = None
    add_162: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_245, primals_97);  mul_245 = primals_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_362: "f32[1568, 512]" = torch.ops.aten.view.default(add_162, [1568, 512]);  add_162 = None
    permute_168: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_288, [1, 0]);  primals_288 = None
    addmm_88: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_289, view_362, permute_168);  primals_289 = None
    view_363: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_88, [8, 1, 196, 1536]);  addmm_88 = None
    view_364: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_363, [8, 1, 196, 3, 16, 32]);  view_363 = None
    permute_169: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_364, [3, 0, 4, 1, 2, 5]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_22 = torch.ops.aten.unbind.int(permute_169);  permute_169 = None
    getitem_164: "f32[8, 16, 1, 196, 32]" = unbind_22[0]
    getitem_165: "f32[8, 16, 1, 196, 32]" = unbind_22[1]
    getitem_166: "f32[8, 16, 1, 196, 32]" = unbind_22[2];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_246: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_164, 0.42044820762685725);  getitem_164 = None
    permute_170: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_165, [0, 1, 2, 4, 3]);  getitem_165 = None
    mul_247: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_170, 0.42044820762685725);  permute_170 = None
    expand_88: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_246, [8, 16, 1, 196, 32]);  mul_246 = None
    clone_158: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
    view_365: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_158, [128, 196, 32]);  clone_158 = None
    expand_89: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_247, [8, 16, 1, 32, 196]);  mul_247 = None
    clone_159: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    view_366: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_159, [128, 32, 196]);  clone_159 = None
    bmm_44: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_365, view_366)
    view_367: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_44, [8, 16, 1, 196, 196]);  bmm_44 = None
    amax_22: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_367, [-1], True)
    sub_69: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_367, amax_22);  view_367 = amax_22 = None
    exp_22: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_69);  sub_69 = None
    sum_23: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_64: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    alias_22: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_64)
    expand_90: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_64, [8, 16, 1, 196, 196]);  div_64 = None
    view_368: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_90, [128, 196, 196]);  expand_90 = None
    expand_91: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_166, [8, 16, 1, 196, 32]);  getitem_166 = None
    clone_160: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
    view_369: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_160, [128, 196, 32]);  clone_160 = None
    bmm_45: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_368, view_369)
    view_370: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_45, [8, 16, 1, 196, 32]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_171: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_370, [0, 2, 3, 4, 1]);  view_370 = None
    clone_161: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    view_371: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_161, [8, 1, 196, 512]);  clone_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_372: "f32[1568, 512]" = torch.ops.aten.view.default(view_371, [1568, 512]);  view_371 = None
    permute_172: "f32[512, 512]" = torch.ops.aten.permute.default(primals_290, [1, 0]);  primals_290 = None
    addmm_89: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_291, view_372, permute_172);  primals_291 = None
    view_373: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_89, [8, 1, 196, 512]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_162: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_373);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_42: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_42: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_42, 0.52173912525177);  empty_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_65: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_42, 0.52173912525177);  bernoulli_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_248: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_162, div_65);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_163: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_160, mul_248);  mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_163, [3], correction = 0, keepdim = True)
    getitem_167: "f32[8, 1, 196, 1]" = var_mean_47[0]
    getitem_168: "f32[8, 1, 196, 1]" = var_mean_47[1];  var_mean_47 = None
    add_164: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-06);  getitem_167 = None
    rsqrt_47: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_70: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_163, getitem_168)
    mul_249: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_47);  sub_70 = None
    mul_250: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_249, primals_98);  mul_249 = None
    add_165: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_250, primals_99);  mul_250 = primals_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_374: "f32[1568, 512]" = torch.ops.aten.view.default(add_165, [1568, 512]);  add_165 = None
    permute_173: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_292, [1, 0]);  primals_292 = None
    addmm_90: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_293, view_374, permute_173);  primals_293 = None
    view_375: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_90, [8, 1, 196, 2048]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_251: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_375, 0.5)
    mul_252: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_375, 0.7071067811865476)
    erf_22: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_252);  mul_252 = None
    add_166: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_253: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_251, add_166);  mul_251 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_163: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_253);  mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_376: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_163, [1568, 2048]);  clone_163 = None
    permute_174: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_294, [1, 0]);  primals_294 = None
    addmm_91: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_295, view_376, permute_174);  primals_295 = None
    view_377: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_91, [8, 1, 196, 512]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_164: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_377);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_43: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_43: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_43, 0.52173912525177);  empty_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_66: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_43, 0.52173912525177);  bernoulli_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_254: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_164, div_66);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_167: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_163, mul_254);  mul_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_167, [3], correction = 0, keepdim = True)
    getitem_169: "f32[8, 1, 196, 1]" = var_mean_48[0]
    getitem_170: "f32[8, 1, 196, 1]" = var_mean_48[1];  var_mean_48 = None
    add_168: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_169, 1e-06);  getitem_169 = None
    rsqrt_48: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_71: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_167, getitem_170)
    mul_255: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_48);  sub_71 = None
    mul_256: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_255, primals_100);  mul_255 = None
    add_169: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_256, primals_101);  mul_256 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    view_378: "f32[1568, 512]" = torch.ops.aten.view.default(add_169, [1568, 512]);  add_169 = None
    permute_175: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_296, [1, 0]);  primals_296 = None
    addmm_92: "f32[1568, 1536]" = torch.ops.aten.addmm.default(primals_297, view_378, permute_175);  primals_297 = None
    view_379: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(addmm_92, [8, 1, 196, 1536]);  addmm_92 = None
    view_380: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.view.default(view_379, [8, 1, 196, 3, 16, 32]);  view_379 = None
    permute_176: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_380, [3, 0, 4, 1, 2, 5]);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_23 = torch.ops.aten.unbind.int(permute_176);  permute_176 = None
    getitem_171: "f32[8, 16, 1, 196, 32]" = unbind_23[0]
    getitem_172: "f32[8, 16, 1, 196, 32]" = unbind_23[1]
    getitem_173: "f32[8, 16, 1, 196, 32]" = unbind_23[2];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    mul_257: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_171, 0.42044820762685725);  getitem_171 = None
    permute_177: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_172, [0, 1, 2, 4, 3]);  getitem_172 = None
    mul_258: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_177, 0.42044820762685725);  permute_177 = None
    expand_92: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_257, [8, 16, 1, 196, 32]);  mul_257 = None
    clone_165: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
    view_381: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_165, [128, 196, 32]);  clone_165 = None
    expand_93: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_258, [8, 16, 1, 32, 196]);  mul_258 = None
    clone_166: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
    view_382: "f32[128, 32, 196]" = torch.ops.aten.view.default(clone_166, [128, 32, 196]);  clone_166 = None
    bmm_46: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_381, view_382)
    view_383: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_46, [8, 16, 1, 196, 196]);  bmm_46 = None
    amax_23: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_383, [-1], True)
    sub_72: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_383, amax_23);  view_383 = amax_23 = None
    exp_23: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_72);  sub_72 = None
    sum_24: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_67: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    alias_23: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(div_67)
    expand_94: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(div_67, [8, 16, 1, 196, 196]);  div_67 = None
    view_384: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_94, [128, 196, 196]);  expand_94 = None
    expand_95: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_173, [8, 16, 1, 196, 32]);  getitem_173 = None
    clone_167: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
    view_385: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_167, [128, 196, 32]);  clone_167 = None
    bmm_47: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_384, view_385)
    view_386: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_47, [8, 16, 1, 196, 32]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_178: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_386, [0, 2, 3, 4, 1]);  view_386 = None
    clone_168: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
    view_387: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(clone_168, [8, 1, 196, 512]);  clone_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_388: "f32[1568, 512]" = torch.ops.aten.view.default(view_387, [1568, 512]);  view_387 = None
    permute_179: "f32[512, 512]" = torch.ops.aten.permute.default(primals_298, [1, 0]);  primals_298 = None
    addmm_93: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_299, view_388, permute_179);  primals_299 = None
    view_389: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_93, [8, 1, 196, 512]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    clone_169: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_389);  view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_44: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_44: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_44, 0.5);  empty_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_68: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_44, 0.5);  bernoulli_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_259: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_169, div_68);  clone_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    add_170: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_167, mul_259);  mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_49 = torch.ops.aten.var_mean.correction(add_170, [3], correction = 0, keepdim = True)
    getitem_174: "f32[8, 1, 196, 1]" = var_mean_49[0]
    getitem_175: "f32[8, 1, 196, 1]" = var_mean_49[1];  var_mean_49 = None
    add_171: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
    rsqrt_49: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_73: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_170, getitem_175)
    mul_260: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_49);  sub_73 = None
    mul_261: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_260, primals_102);  mul_260 = None
    add_172: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_261, primals_103);  mul_261 = primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_390: "f32[1568, 512]" = torch.ops.aten.view.default(add_172, [1568, 512]);  add_172 = None
    permute_180: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_300, [1, 0]);  primals_300 = None
    addmm_94: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_301, view_390, permute_180);  primals_301 = None
    view_391: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_94, [8, 1, 196, 2048]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_262: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_391, 0.5)
    mul_263: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_391, 0.7071067811865476)
    erf_23: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_263);  mul_263 = None
    add_173: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_264: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_262, add_173);  mul_262 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_170: "f32[8, 1, 196, 2048]" = torch.ops.aten.clone.default(mul_264);  mul_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_392: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_170, [1568, 2048]);  clone_170 = None
    permute_181: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_302, [1, 0]);  primals_302 = None
    addmm_95: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_303, view_392, permute_181);  primals_303 = None
    view_393: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(addmm_95, [8, 1, 196, 512]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_171: "f32[8, 1, 196, 512]" = torch.ops.aten.clone.default(view_393);  view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:151, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    empty_45: "f32[8, 1, 1, 1]" = torch.ops.aten.empty.memory_format([8, 1, 1, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    bernoulli_45: "f32[8, 1, 1, 1]" = torch.ops.aten.bernoulli.p(empty_45, 0.5);  empty_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_69: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_45, 0.5);  bernoulli_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_265: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(clone_171, div_69);  clone_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_174: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_170, mul_265);  mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    view_394: "f32[8, 1, 1, 14, 14, 512]" = torch.ops.aten.view.default(add_174, [8, 1, 1, 14, 14, 512]);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    permute_182: "f32[8, 1, 14, 1, 14, 512]" = torch.ops.aten.permute.default(view_394, [0, 1, 3, 2, 4, 5]);  view_394 = None
    view_395: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(permute_182, [8, 14, 14, 512]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_183: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_395, [0, 3, 1, 2]);  view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:427, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_184: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(permute_183, [0, 2, 3, 1]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_50 = torch.ops.aten.var_mean.correction(permute_184, [3], correction = 0, keepdim = True)
    getitem_176: "f32[8, 14, 14, 1]" = var_mean_50[0]
    getitem_177: "f32[8, 14, 14, 1]" = var_mean_50[1];  var_mean_50 = None
    add_175: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
    rsqrt_50: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_74: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_184, getitem_177)
    mul_266: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_50);  sub_74 = None
    mul_267: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_266, primals_104);  mul_266 = None
    add_176: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_267, primals_105);  mul_267 = primals_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:427, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_185: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(add_176, [0, 3, 1, 2]);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(permute_185, [-1, -2], True);  permute_185 = None
    as_strided: "f32[8, 512, 1, 1]" = torch.ops.aten.as_strided.default(mean, [8, 512, 1, 1], [512, 1, 512, 512]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_396: "f32[8, 512]" = torch.ops.aten.view.default(as_strided, [8, 512]);  as_strided = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:432, code: x = self.head_drop(x)
    clone_172: "f32[8, 512]" = torch.ops.aten.clone.default(view_396);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:433, code: return x if pre_logits else self.head(x)
    permute_186: "f32[512, 1000]" = torch.ops.aten.permute.default(primals_304, [1, 0]);  primals_304 = None
    addmm_96: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_305, clone_172, permute_186);  primals_305 = None
    permute_187: "f32[1000, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    mm: "f32[8, 512]" = torch.ops.aten.mm.default(tangents_1, permute_187);  permute_187 = None
    permute_188: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 512]" = torch.ops.aten.mm.default(permute_188, clone_172);  permute_188 = clone_172 = None
    permute_189: "f32[512, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_25: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_397: "f32[1000]" = torch.ops.aten.view.default(sum_25, [1000]);  sum_25 = None
    permute_190: "f32[1000, 512]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_398: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(mm, [8, 512, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    squeeze: "f32[8, 512, 1]" = torch.ops.aten.squeeze.dim(view_398, 3);  view_398 = None
    squeeze_1: "f32[8, 512]" = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
    full: "f32[4096]" = torch.ops.aten.full.default([4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_1: "f32[8, 512]" = torch.ops.aten.as_strided.default(full, [8, 512], [512, 1], 0)
    copy: "f32[8, 512]" = torch.ops.aten.copy.default(as_strided_1, squeeze_1);  as_strided_1 = squeeze_1 = None
    as_strided_scatter: "f32[4096]" = torch.ops.aten.as_strided_scatter.default(full, copy, [8, 512], [512, 1], 0);  full = copy = None
    as_strided_4: "f32[8, 512, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, [8, 512, 1, 1], [512, 1, 1, 1], 0);  as_strided_scatter = None
    expand_97: "f32[8, 512, 14, 14]" = torch.ops.aten.expand.default(as_strided_4, [8, 512, 14, 14]);  as_strided_4 = None
    div_70: "f32[8, 512, 14, 14]" = torch.ops.aten.div.Scalar(expand_97, 196);  expand_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:427, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_191: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(div_70, [0, 2, 3, 1]);  div_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_173: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
    sub_75: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_184, getitem_177);  permute_184 = getitem_177 = None
    mul_268: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_50);  sub_75 = None
    mul_269: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(clone_173, primals_104);  primals_104 = None
    mul_270: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_269, 512)
    sum_26: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [3], True)
    mul_271: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_269, mul_268);  mul_269 = None
    sum_27: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [3], True);  mul_271 = None
    mul_272: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_268, sum_27);  sum_27 = None
    sub_76: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_270, sum_26);  mul_270 = sum_26 = None
    sub_77: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_76, mul_272);  sub_76 = mul_272 = None
    div_71: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 512);  rsqrt_50 = None
    mul_273: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_71, sub_77);  div_71 = sub_77 = None
    mul_274: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(clone_173, mul_268);  mul_268 = None
    sum_28: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1, 2]);  mul_274 = None
    sum_29: "f32[512]" = torch.ops.aten.sum.dim_IntList(clone_173, [0, 1, 2]);  clone_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:427, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_192: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_273, [0, 3, 1, 2]);  mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_193: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(permute_192, [0, 2, 3, 1]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    view_399: "f32[8, 1, 14, 1, 14, 512]" = torch.ops.aten.view.default(permute_193, [8, 1, 14, 1, 14, 512]);  permute_193 = None
    permute_194: "f32[8, 1, 1, 14, 14, 512]" = torch.ops.aten.permute.default(view_399, [0, 1, 3, 2, 4, 5]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    view_400: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(permute_194, [8, 1, 196, 512]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_275: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_400, div_69);  div_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_401: "f32[1568, 512]" = torch.ops.aten.view.default(mul_275, [1568, 512]);  mul_275 = None
    permute_195: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    mm_2: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_401, permute_195);  permute_195 = None
    permute_196: "f32[512, 1568]" = torch.ops.aten.permute.default(view_401, [1, 0])
    mm_3: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_196, view_392);  permute_196 = view_392 = None
    permute_197: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_30: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_401, [0], True);  view_401 = None
    view_402: "f32[512]" = torch.ops.aten.view.default(sum_30, [512]);  sum_30 = None
    permute_198: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_403: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_2, [8, 1, 196, 2048]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_276: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_391, 0.7071067811865476)
    erf_24: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_276);  mul_276 = None
    add_177: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_277: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_177, 0.5);  add_177 = None
    mul_278: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_391, view_391)
    mul_279: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_278, -0.5);  mul_278 = None
    exp_24: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_279);  mul_279 = None
    mul_280: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_281: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_391, mul_280);  view_391 = mul_280 = None
    add_178: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_277, mul_281);  mul_277 = mul_281 = None
    mul_282: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_403, add_178);  view_403 = add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_404: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_282, [1568, 2048]);  mul_282 = None
    permute_199: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    mm_4: "f32[1568, 512]" = torch.ops.aten.mm.default(view_404, permute_199);  permute_199 = None
    permute_200: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_404, [1, 0])
    mm_5: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_200, view_390);  permute_200 = view_390 = None
    permute_201: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_31: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_404, [0], True);  view_404 = None
    view_405: "f32[2048]" = torch.ops.aten.view.default(sum_31, [2048]);  sum_31 = None
    permute_202: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_406: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_4, [8, 1, 196, 512]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_78: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_170, getitem_175);  add_170 = getitem_175 = None
    mul_283: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_49);  sub_78 = None
    mul_284: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_406, primals_102);  primals_102 = None
    mul_285: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_284, 512)
    sum_32: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [3], True)
    mul_286: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_284, mul_283);  mul_284 = None
    sum_33: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [3], True);  mul_286 = None
    mul_287: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_283, sum_33);  sum_33 = None
    sub_79: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_285, sum_32);  mul_285 = sum_32 = None
    sub_80: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_79, mul_287);  sub_79 = mul_287 = None
    div_72: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 512);  rsqrt_49 = None
    mul_288: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_72, sub_80);  div_72 = sub_80 = None
    mul_289: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_406, mul_283);  mul_283 = None
    sum_34: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 1, 2]);  mul_289 = None
    sum_35: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_406, [0, 1, 2]);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_179: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(view_400, mul_288);  view_400 = mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_290: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_179, div_68);  div_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_407: "f32[1568, 512]" = torch.ops.aten.view.default(mul_290, [1568, 512]);  mul_290 = None
    permute_203: "f32[512, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
    mm_6: "f32[1568, 512]" = torch.ops.aten.mm.default(view_407, permute_203);  permute_203 = None
    permute_204: "f32[512, 1568]" = torch.ops.aten.permute.default(view_407, [1, 0])
    mm_7: "f32[512, 512]" = torch.ops.aten.mm.default(permute_204, view_388);  permute_204 = view_388 = None
    permute_205: "f32[512, 512]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_36: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_407, [0], True);  view_407 = None
    view_408: "f32[512]" = torch.ops.aten.view.default(sum_36, [512]);  sum_36 = None
    permute_206: "f32[512, 512]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
    view_409: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_6, [8, 1, 196, 512]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_410: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_409, [8, 1, 196, 32, 16]);  view_409 = None
    permute_207: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_410, [0, 4, 1, 2, 3]);  view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_174: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
    view_411: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_174, [128, 196, 32]);  clone_174 = None
    permute_208: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_384, [0, 2, 1]);  view_384 = None
    bmm_48: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_208, view_411);  permute_208 = None
    permute_209: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    bmm_49: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_411, permute_209);  view_411 = permute_209 = None
    view_412: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_48, [8, 16, 1, 196, 32]);  bmm_48 = None
    view_413: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_49, [8, 16, 1, 196, 196]);  bmm_49 = None
    alias_24: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_291: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_413, alias_24);  view_413 = None
    sum_37: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [-1], True)
    mul_292: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_24, sum_37);  alias_24 = sum_37 = None
    sub_81: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    view_414: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_81, [128, 196, 196]);  sub_81 = None
    permute_210: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
    bmm_50: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_210, view_414);  permute_210 = None
    permute_211: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_382, [0, 2, 1]);  view_382 = None
    bmm_51: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_414, permute_211);  view_414 = permute_211 = None
    view_415: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_50, [8, 16, 1, 32, 196]);  bmm_50 = None
    view_416: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_51, [8, 16, 1, 196, 32]);  bmm_51 = None
    mul_293: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_415, 0.42044820762685725);  view_415 = None
    permute_212: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_293, [0, 1, 2, 4, 3]);  mul_293 = None
    mul_294: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_416, 0.42044820762685725);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_294, permute_212, view_412]);  mul_294 = permute_212 = view_412 = None
    view_417: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat, [3, 8, 16, 1, 196, 32]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_213: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_417, [1, 3, 4, 0, 2, 5]);  view_417 = None
    clone_175: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
    view_418: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_175, [8, 1, 196, 1536]);  clone_175 = None
    view_419: "f32[1568, 1536]" = torch.ops.aten.view.default(view_418, [1568, 1536]);  view_418 = None
    permute_214: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    mm_8: "f32[1568, 512]" = torch.ops.aten.mm.default(view_419, permute_214);  permute_214 = None
    permute_215: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_419, [1, 0])
    mm_9: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_215, view_378);  permute_215 = view_378 = None
    permute_216: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_38: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
    view_420: "f32[1536]" = torch.ops.aten.view.default(sum_38, [1536]);  sum_38 = None
    permute_217: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
    view_421: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_8, [8, 1, 196, 512]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_82: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_167, getitem_170);  add_167 = getitem_170 = None
    mul_295: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_48);  sub_82 = None
    mul_296: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_421, primals_100);  primals_100 = None
    mul_297: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_296, 512)
    sum_39: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [3], True)
    mul_298: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_296, mul_295);  mul_296 = None
    sum_40: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [3], True);  mul_298 = None
    mul_299: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_295, sum_40);  sum_40 = None
    sub_83: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_297, sum_39);  mul_297 = sum_39 = None
    sub_84: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_83, mul_299);  sub_83 = mul_299 = None
    div_73: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 512);  rsqrt_48 = None
    mul_300: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_73, sub_84);  div_73 = sub_84 = None
    mul_301: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_421, mul_295);  mul_295 = None
    sum_41: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1, 2]);  mul_301 = None
    sum_42: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_421, [0, 1, 2]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_180: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_179, mul_300);  add_179 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_302: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_180, div_66);  div_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_422: "f32[1568, 512]" = torch.ops.aten.view.default(mul_302, [1568, 512]);  mul_302 = None
    permute_218: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    mm_10: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_422, permute_218);  permute_218 = None
    permute_219: "f32[512, 1568]" = torch.ops.aten.permute.default(view_422, [1, 0])
    mm_11: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_219, view_376);  permute_219 = view_376 = None
    permute_220: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_43: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
    view_423: "f32[512]" = torch.ops.aten.view.default(sum_43, [512]);  sum_43 = None
    permute_221: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    view_424: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_10, [8, 1, 196, 2048]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_303: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_375, 0.7071067811865476)
    erf_25: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_303);  mul_303 = None
    add_181: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_304: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_181, 0.5);  add_181 = None
    mul_305: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_375, view_375)
    mul_306: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_305, -0.5);  mul_305 = None
    exp_25: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_306);  mul_306 = None
    mul_307: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_308: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_375, mul_307);  view_375 = mul_307 = None
    add_182: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_304, mul_308);  mul_304 = mul_308 = None
    mul_309: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_424, add_182);  view_424 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_425: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_309, [1568, 2048]);  mul_309 = None
    permute_222: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    mm_12: "f32[1568, 512]" = torch.ops.aten.mm.default(view_425, permute_222);  permute_222 = None
    permute_223: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_425, [1, 0])
    mm_13: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_223, view_374);  permute_223 = view_374 = None
    permute_224: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_44: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
    view_426: "f32[2048]" = torch.ops.aten.view.default(sum_44, [2048]);  sum_44 = None
    permute_225: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    view_427: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_12, [8, 1, 196, 512]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_85: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_163, getitem_168);  add_163 = getitem_168 = None
    mul_310: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_47);  sub_85 = None
    mul_311: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_427, primals_98);  primals_98 = None
    mul_312: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_311, 512)
    sum_45: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [3], True)
    mul_313: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_311, mul_310);  mul_311 = None
    sum_46: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [3], True);  mul_313 = None
    mul_314: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_310, sum_46);  sum_46 = None
    sub_86: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_312, sum_45);  mul_312 = sum_45 = None
    sub_87: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_86, mul_314);  sub_86 = mul_314 = None
    div_74: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 512);  rsqrt_47 = None
    mul_315: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_74, sub_87);  div_74 = sub_87 = None
    mul_316: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_427, mul_310);  mul_310 = None
    sum_47: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1, 2]);  mul_316 = None
    sum_48: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_427, [0, 1, 2]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_183: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_180, mul_315);  add_180 = mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_317: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_183, div_65);  div_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_428: "f32[1568, 512]" = torch.ops.aten.view.default(mul_317, [1568, 512]);  mul_317 = None
    permute_226: "f32[512, 512]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    mm_14: "f32[1568, 512]" = torch.ops.aten.mm.default(view_428, permute_226);  permute_226 = None
    permute_227: "f32[512, 1568]" = torch.ops.aten.permute.default(view_428, [1, 0])
    mm_15: "f32[512, 512]" = torch.ops.aten.mm.default(permute_227, view_372);  permute_227 = view_372 = None
    permute_228: "f32[512, 512]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_49: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_428, [0], True);  view_428 = None
    view_429: "f32[512]" = torch.ops.aten.view.default(sum_49, [512]);  sum_49 = None
    permute_229: "f32[512, 512]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    view_430: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_14, [8, 1, 196, 512]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_431: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_430, [8, 1, 196, 32, 16]);  view_430 = None
    permute_230: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_431, [0, 4, 1, 2, 3]);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_176: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
    view_432: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_176, [128, 196, 32]);  clone_176 = None
    permute_231: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_368, [0, 2, 1]);  view_368 = None
    bmm_52: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_231, view_432);  permute_231 = None
    permute_232: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_369, [0, 2, 1]);  view_369 = None
    bmm_53: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_432, permute_232);  view_432 = permute_232 = None
    view_433: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_52, [8, 16, 1, 196, 32]);  bmm_52 = None
    view_434: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_53, [8, 16, 1, 196, 196]);  bmm_53 = None
    alias_25: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    mul_318: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_434, alias_25);  view_434 = None
    sum_50: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [-1], True)
    mul_319: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_25, sum_50);  alias_25 = sum_50 = None
    sub_88: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
    view_435: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_88, [128, 196, 196]);  sub_88 = None
    permute_233: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None
    bmm_54: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_233, view_435);  permute_233 = None
    permute_234: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_366, [0, 2, 1]);  view_366 = None
    bmm_55: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_435, permute_234);  view_435 = permute_234 = None
    view_436: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_54, [8, 16, 1, 32, 196]);  bmm_54 = None
    view_437: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_55, [8, 16, 1, 196, 32]);  bmm_55 = None
    mul_320: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_436, 0.42044820762685725);  view_436 = None
    permute_235: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_320, [0, 1, 2, 4, 3]);  mul_320 = None
    mul_321: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_437, 0.42044820762685725);  view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_1: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_321, permute_235, view_433]);  mul_321 = permute_235 = view_433 = None
    view_438: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_1, [3, 8, 16, 1, 196, 32]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_236: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_438, [1, 3, 4, 0, 2, 5]);  view_438 = None
    clone_177: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    view_439: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_177, [8, 1, 196, 1536]);  clone_177 = None
    view_440: "f32[1568, 1536]" = torch.ops.aten.view.default(view_439, [1568, 1536]);  view_439 = None
    permute_237: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    mm_16: "f32[1568, 512]" = torch.ops.aten.mm.default(view_440, permute_237);  permute_237 = None
    permute_238: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_440, [1, 0])
    mm_17: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_238, view_362);  permute_238 = view_362 = None
    permute_239: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_51: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
    view_441: "f32[1536]" = torch.ops.aten.view.default(sum_51, [1536]);  sum_51 = None
    permute_240: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    view_442: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_16, [8, 1, 196, 512]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_89: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_160, getitem_163);  add_160 = getitem_163 = None
    mul_322: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_46);  sub_89 = None
    mul_323: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_442, primals_96);  primals_96 = None
    mul_324: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_323, 512)
    sum_52: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [3], True)
    mul_325: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_323, mul_322);  mul_323 = None
    sum_53: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [3], True);  mul_325 = None
    mul_326: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_322, sum_53);  sum_53 = None
    sub_90: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_324, sum_52);  mul_324 = sum_52 = None
    sub_91: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_90, mul_326);  sub_90 = mul_326 = None
    div_75: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 512);  rsqrt_46 = None
    mul_327: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_75, sub_91);  div_75 = sub_91 = None
    mul_328: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_442, mul_322);  mul_322 = None
    sum_54: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1, 2]);  mul_328 = None
    sum_55: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_442, [0, 1, 2]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_184: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_183, mul_327);  add_183 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_329: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_184, div_63);  div_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_443: "f32[1568, 512]" = torch.ops.aten.view.default(mul_329, [1568, 512]);  mul_329 = None
    permute_241: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    mm_18: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_443, permute_241);  permute_241 = None
    permute_242: "f32[512, 1568]" = torch.ops.aten.permute.default(view_443, [1, 0])
    mm_19: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_242, view_360);  permute_242 = view_360 = None
    permute_243: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_56: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_443, [0], True);  view_443 = None
    view_444: "f32[512]" = torch.ops.aten.view.default(sum_56, [512]);  sum_56 = None
    permute_244: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_445: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_18, [8, 1, 196, 2048]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_330: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_359, 0.7071067811865476)
    erf_26: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_330);  mul_330 = None
    add_185: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_331: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_185, 0.5);  add_185 = None
    mul_332: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_359, view_359)
    mul_333: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_332, -0.5);  mul_332 = None
    exp_26: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_333);  mul_333 = None
    mul_334: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_335: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_359, mul_334);  view_359 = mul_334 = None
    add_186: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_331, mul_335);  mul_331 = mul_335 = None
    mul_336: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_445, add_186);  view_445 = add_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_446: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_336, [1568, 2048]);  mul_336 = None
    permute_245: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    mm_20: "f32[1568, 512]" = torch.ops.aten.mm.default(view_446, permute_245);  permute_245 = None
    permute_246: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_446, [1, 0])
    mm_21: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_246, view_358);  permute_246 = view_358 = None
    permute_247: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_57: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_446, [0], True);  view_446 = None
    view_447: "f32[2048]" = torch.ops.aten.view.default(sum_57, [2048]);  sum_57 = None
    permute_248: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_448: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_20, [8, 1, 196, 512]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_92: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_156, getitem_161);  add_156 = getitem_161 = None
    mul_337: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_45);  sub_92 = None
    mul_338: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_448, primals_94);  primals_94 = None
    mul_339: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_338, 512)
    sum_58: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [3], True)
    mul_340: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_338, mul_337);  mul_338 = None
    sum_59: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [3], True);  mul_340 = None
    mul_341: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_337, sum_59);  sum_59 = None
    sub_93: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_339, sum_58);  mul_339 = sum_58 = None
    sub_94: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_93, mul_341);  sub_93 = mul_341 = None
    div_76: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 512);  rsqrt_45 = None
    mul_342: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_76, sub_94);  div_76 = sub_94 = None
    mul_343: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_448, mul_337);  mul_337 = None
    sum_60: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1, 2]);  mul_343 = None
    sum_61: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_448, [0, 1, 2]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_187: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_184, mul_342);  add_184 = mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_344: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_187, div_62);  div_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_449: "f32[1568, 512]" = torch.ops.aten.view.default(mul_344, [1568, 512]);  mul_344 = None
    permute_249: "f32[512, 512]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    mm_22: "f32[1568, 512]" = torch.ops.aten.mm.default(view_449, permute_249);  permute_249 = None
    permute_250: "f32[512, 1568]" = torch.ops.aten.permute.default(view_449, [1, 0])
    mm_23: "f32[512, 512]" = torch.ops.aten.mm.default(permute_250, view_356);  permute_250 = view_356 = None
    permute_251: "f32[512, 512]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_62: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_449, [0], True);  view_449 = None
    view_450: "f32[512]" = torch.ops.aten.view.default(sum_62, [512]);  sum_62 = None
    permute_252: "f32[512, 512]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    view_451: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_22, [8, 1, 196, 512]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_452: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_451, [8, 1, 196, 32, 16]);  view_451 = None
    permute_253: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_452, [0, 4, 1, 2, 3]);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_178: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
    view_453: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_178, [128, 196, 32]);  clone_178 = None
    permute_254: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None
    bmm_56: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_254, view_453);  permute_254 = None
    permute_255: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_353, [0, 2, 1]);  view_353 = None
    bmm_57: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_453, permute_255);  view_453 = permute_255 = None
    view_454: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_56, [8, 16, 1, 196, 32]);  bmm_56 = None
    view_455: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_57, [8, 16, 1, 196, 196]);  bmm_57 = None
    alias_26: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_345: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_455, alias_26);  view_455 = None
    sum_63: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_345, [-1], True)
    mul_346: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_26, sum_63);  alias_26 = sum_63 = None
    sub_95: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_345, mul_346);  mul_345 = mul_346 = None
    view_456: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_95, [128, 196, 196]);  sub_95 = None
    permute_256: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
    bmm_58: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_256, view_456);  permute_256 = None
    permute_257: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_350, [0, 2, 1]);  view_350 = None
    bmm_59: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_456, permute_257);  view_456 = permute_257 = None
    view_457: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_58, [8, 16, 1, 32, 196]);  bmm_58 = None
    view_458: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_59, [8, 16, 1, 196, 32]);  bmm_59 = None
    mul_347: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_457, 0.42044820762685725);  view_457 = None
    permute_258: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_347, [0, 1, 2, 4, 3]);  mul_347 = None
    mul_348: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_458, 0.42044820762685725);  view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_2: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_348, permute_258, view_454]);  mul_348 = permute_258 = view_454 = None
    view_459: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_2, [3, 8, 16, 1, 196, 32]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_259: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_459, [1, 3, 4, 0, 2, 5]);  view_459 = None
    clone_179: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
    view_460: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_179, [8, 1, 196, 1536]);  clone_179 = None
    view_461: "f32[1568, 1536]" = torch.ops.aten.view.default(view_460, [1568, 1536]);  view_460 = None
    permute_260: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    mm_24: "f32[1568, 512]" = torch.ops.aten.mm.default(view_461, permute_260);  permute_260 = None
    permute_261: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_461, [1, 0])
    mm_25: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_261, view_346);  permute_261 = view_346 = None
    permute_262: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_64: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_461, [0], True);  view_461 = None
    view_462: "f32[1536]" = torch.ops.aten.view.default(sum_64, [1536]);  sum_64 = None
    permute_263: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    view_463: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_24, [8, 1, 196, 512]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_96: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_153, getitem_156);  add_153 = getitem_156 = None
    mul_349: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_44);  sub_96 = None
    mul_350: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_463, primals_92);  primals_92 = None
    mul_351: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_350, 512)
    sum_65: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [3], True)
    mul_352: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_350, mul_349);  mul_350 = None
    sum_66: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [3], True);  mul_352 = None
    mul_353: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_349, sum_66);  sum_66 = None
    sub_97: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_351, sum_65);  mul_351 = sum_65 = None
    sub_98: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_97, mul_353);  sub_97 = mul_353 = None
    div_77: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 512);  rsqrt_44 = None
    mul_354: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_77, sub_98);  div_77 = sub_98 = None
    mul_355: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_463, mul_349);  mul_349 = None
    sum_67: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_355, [0, 1, 2]);  mul_355 = None
    sum_68: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_463, [0, 1, 2]);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_188: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_187, mul_354);  add_187 = mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_356: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_188, div_60);  div_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_464: "f32[1568, 512]" = torch.ops.aten.view.default(mul_356, [1568, 512]);  mul_356 = None
    permute_264: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
    mm_26: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_464, permute_264);  permute_264 = None
    permute_265: "f32[512, 1568]" = torch.ops.aten.permute.default(view_464, [1, 0])
    mm_27: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_265, view_344);  permute_265 = view_344 = None
    permute_266: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_69: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
    view_465: "f32[512]" = torch.ops.aten.view.default(sum_69, [512]);  sum_69 = None
    permute_267: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    view_466: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_26, [8, 1, 196, 2048]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_357: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_343, 0.7071067811865476)
    erf_27: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_357);  mul_357 = None
    add_189: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_358: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_189, 0.5);  add_189 = None
    mul_359: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_343, view_343)
    mul_360: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_359, -0.5);  mul_359 = None
    exp_27: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_360);  mul_360 = None
    mul_361: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_362: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_343, mul_361);  view_343 = mul_361 = None
    add_190: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_358, mul_362);  mul_358 = mul_362 = None
    mul_363: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_466, add_190);  view_466 = add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_467: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_363, [1568, 2048]);  mul_363 = None
    permute_268: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    mm_28: "f32[1568, 512]" = torch.ops.aten.mm.default(view_467, permute_268);  permute_268 = None
    permute_269: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_467, [1, 0])
    mm_29: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_269, view_342);  permute_269 = view_342 = None
    permute_270: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_70: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
    view_468: "f32[2048]" = torch.ops.aten.view.default(sum_70, [2048]);  sum_70 = None
    permute_271: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
    view_469: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_28, [8, 1, 196, 512]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_99: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_149, getitem_154);  add_149 = getitem_154 = None
    mul_364: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_43);  sub_99 = None
    mul_365: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_469, primals_90);  primals_90 = None
    mul_366: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_365, 512)
    sum_71: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [3], True)
    mul_367: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_365, mul_364);  mul_365 = None
    sum_72: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [3], True);  mul_367 = None
    mul_368: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_364, sum_72);  sum_72 = None
    sub_100: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_366, sum_71);  mul_366 = sum_71 = None
    sub_101: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_100, mul_368);  sub_100 = mul_368 = None
    div_78: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 512);  rsqrt_43 = None
    mul_369: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_78, sub_101);  div_78 = sub_101 = None
    mul_370: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_469, mul_364);  mul_364 = None
    sum_73: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_370, [0, 1, 2]);  mul_370 = None
    sum_74: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_469, [0, 1, 2]);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_191: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_188, mul_369);  add_188 = mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_371: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_191, div_59);  div_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_470: "f32[1568, 512]" = torch.ops.aten.view.default(mul_371, [1568, 512]);  mul_371 = None
    permute_272: "f32[512, 512]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    mm_30: "f32[1568, 512]" = torch.ops.aten.mm.default(view_470, permute_272);  permute_272 = None
    permute_273: "f32[512, 1568]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_31: "f32[512, 512]" = torch.ops.aten.mm.default(permute_273, view_340);  permute_273 = view_340 = None
    permute_274: "f32[512, 512]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_75: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
    view_471: "f32[512]" = torch.ops.aten.view.default(sum_75, [512]);  sum_75 = None
    permute_275: "f32[512, 512]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    view_472: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_30, [8, 1, 196, 512]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_473: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_472, [8, 1, 196, 32, 16]);  view_472 = None
    permute_276: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_473, [0, 4, 1, 2, 3]);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_180: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
    view_474: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_180, [128, 196, 32]);  clone_180 = None
    permute_277: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_336, [0, 2, 1]);  view_336 = None
    bmm_60: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_277, view_474);  permute_277 = None
    permute_278: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
    bmm_61: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_474, permute_278);  view_474 = permute_278 = None
    view_475: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_60, [8, 16, 1, 196, 32]);  bmm_60 = None
    view_476: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_61, [8, 16, 1, 196, 196]);  bmm_61 = None
    alias_27: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    mul_372: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_476, alias_27);  view_476 = None
    sum_76: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [-1], True)
    mul_373: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_27, sum_76);  alias_27 = sum_76 = None
    sub_102: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    view_477: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_102, [128, 196, 196]);  sub_102 = None
    permute_279: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
    bmm_62: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_279, view_477);  permute_279 = None
    permute_280: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_334, [0, 2, 1]);  view_334 = None
    bmm_63: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_477, permute_280);  view_477 = permute_280 = None
    view_478: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_62, [8, 16, 1, 32, 196]);  bmm_62 = None
    view_479: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_63, [8, 16, 1, 196, 32]);  bmm_63 = None
    mul_374: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_478, 0.42044820762685725);  view_478 = None
    permute_281: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_374, [0, 1, 2, 4, 3]);  mul_374 = None
    mul_375: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_479, 0.42044820762685725);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_3: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_375, permute_281, view_475]);  mul_375 = permute_281 = view_475 = None
    view_480: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_3, [3, 8, 16, 1, 196, 32]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_282: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_480, [1, 3, 4, 0, 2, 5]);  view_480 = None
    clone_181: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format);  permute_282 = None
    view_481: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_181, [8, 1, 196, 1536]);  clone_181 = None
    view_482: "f32[1568, 1536]" = torch.ops.aten.view.default(view_481, [1568, 1536]);  view_481 = None
    permute_283: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    mm_32: "f32[1568, 512]" = torch.ops.aten.mm.default(view_482, permute_283);  permute_283 = None
    permute_284: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_482, [1, 0])
    mm_33: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_284, view_330);  permute_284 = view_330 = None
    permute_285: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_77: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_482, [0], True);  view_482 = None
    view_483: "f32[1536]" = torch.ops.aten.view.default(sum_77, [1536]);  sum_77 = None
    permute_286: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    view_484: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_32, [8, 1, 196, 512]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_103: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_146, getitem_149);  add_146 = getitem_149 = None
    mul_376: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_42);  sub_103 = None
    mul_377: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_484, primals_88);  primals_88 = None
    mul_378: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_377, 512)
    sum_78: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [3], True)
    mul_379: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_377, mul_376);  mul_377 = None
    sum_79: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [3], True);  mul_379 = None
    mul_380: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_376, sum_79);  sum_79 = None
    sub_104: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_378, sum_78);  mul_378 = sum_78 = None
    sub_105: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_104, mul_380);  sub_104 = mul_380 = None
    div_79: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 512);  rsqrt_42 = None
    mul_381: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_79, sub_105);  div_79 = sub_105 = None
    mul_382: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_484, mul_376);  mul_376 = None
    sum_80: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 1, 2]);  mul_382 = None
    sum_81: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_484, [0, 1, 2]);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_192: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_191, mul_381);  add_191 = mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_383: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_192, div_57);  div_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_485: "f32[1568, 512]" = torch.ops.aten.view.default(mul_383, [1568, 512]);  mul_383 = None
    permute_287: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    mm_34: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_485, permute_287);  permute_287 = None
    permute_288: "f32[512, 1568]" = torch.ops.aten.permute.default(view_485, [1, 0])
    mm_35: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_288, view_328);  permute_288 = view_328 = None
    permute_289: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_82: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_485, [0], True);  view_485 = None
    view_486: "f32[512]" = torch.ops.aten.view.default(sum_82, [512]);  sum_82 = None
    permute_290: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
    view_487: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_34, [8, 1, 196, 2048]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_384: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476)
    erf_28: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_384);  mul_384 = None
    add_193: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_385: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_193, 0.5);  add_193 = None
    mul_386: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_327, view_327)
    mul_387: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_386, -0.5);  mul_386 = None
    exp_28: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_387);  mul_387 = None
    mul_388: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_389: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_327, mul_388);  view_327 = mul_388 = None
    add_194: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_385, mul_389);  mul_385 = mul_389 = None
    mul_390: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_487, add_194);  view_487 = add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_488: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_390, [1568, 2048]);  mul_390 = None
    permute_291: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    mm_36: "f32[1568, 512]" = torch.ops.aten.mm.default(view_488, permute_291);  permute_291 = None
    permute_292: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_488, [1, 0])
    mm_37: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_292, view_326);  permute_292 = view_326 = None
    permute_293: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_83: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_488, [0], True);  view_488 = None
    view_489: "f32[2048]" = torch.ops.aten.view.default(sum_83, [2048]);  sum_83 = None
    permute_294: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
    view_490: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_36, [8, 1, 196, 512]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_106: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_142, getitem_147);  add_142 = getitem_147 = None
    mul_391: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_41);  sub_106 = None
    mul_392: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_490, primals_86);  primals_86 = None
    mul_393: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_392, 512)
    sum_84: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [3], True)
    mul_394: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_392, mul_391);  mul_392 = None
    sum_85: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [3], True);  mul_394 = None
    mul_395: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_391, sum_85);  sum_85 = None
    sub_107: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_393, sum_84);  mul_393 = sum_84 = None
    sub_108: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_107, mul_395);  sub_107 = mul_395 = None
    div_80: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 512);  rsqrt_41 = None
    mul_396: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_80, sub_108);  div_80 = sub_108 = None
    mul_397: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_490, mul_391);  mul_391 = None
    sum_86: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 1, 2]);  mul_397 = None
    sum_87: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_490, [0, 1, 2]);  view_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_195: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_192, mul_396);  add_192 = mul_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_398: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_195, div_56);  div_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_491: "f32[1568, 512]" = torch.ops.aten.view.default(mul_398, [1568, 512]);  mul_398 = None
    permute_295: "f32[512, 512]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    mm_38: "f32[1568, 512]" = torch.ops.aten.mm.default(view_491, permute_295);  permute_295 = None
    permute_296: "f32[512, 1568]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_39: "f32[512, 512]" = torch.ops.aten.mm.default(permute_296, view_324);  permute_296 = view_324 = None
    permute_297: "f32[512, 512]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_88: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_491, [0], True);  view_491 = None
    view_492: "f32[512]" = torch.ops.aten.view.default(sum_88, [512]);  sum_88 = None
    permute_298: "f32[512, 512]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    view_493: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_38, [8, 1, 196, 512]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_494: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_493, [8, 1, 196, 32, 16]);  view_493 = None
    permute_299: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_494, [0, 4, 1, 2, 3]);  view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_182: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
    view_495: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_182, [128, 196, 32]);  clone_182 = None
    permute_300: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
    bmm_64: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_300, view_495);  permute_300 = None
    permute_301: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None
    bmm_65: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_495, permute_301);  view_495 = permute_301 = None
    view_496: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_64, [8, 16, 1, 196, 32]);  bmm_64 = None
    view_497: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_65, [8, 16, 1, 196, 196]);  bmm_65 = None
    alias_28: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    mul_399: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_497, alias_28);  view_497 = None
    sum_89: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [-1], True)
    mul_400: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_28, sum_89);  alias_28 = sum_89 = None
    sub_109: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_399, mul_400);  mul_399 = mul_400 = None
    view_498: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_109, [128, 196, 196]);  sub_109 = None
    permute_302: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
    bmm_66: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_302, view_498);  permute_302 = None
    permute_303: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None
    bmm_67: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_498, permute_303);  view_498 = permute_303 = None
    view_499: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_66, [8, 16, 1, 32, 196]);  bmm_66 = None
    view_500: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_67, [8, 16, 1, 196, 32]);  bmm_67 = None
    mul_401: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_499, 0.42044820762685725);  view_499 = None
    permute_304: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_401, [0, 1, 2, 4, 3]);  mul_401 = None
    mul_402: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_500, 0.42044820762685725);  view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_4: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_402, permute_304, view_496]);  mul_402 = permute_304 = view_496 = None
    view_501: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_4, [3, 8, 16, 1, 196, 32]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_305: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_501, [1, 3, 4, 0, 2, 5]);  view_501 = None
    clone_183: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_305, memory_format = torch.contiguous_format);  permute_305 = None
    view_502: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_183, [8, 1, 196, 1536]);  clone_183 = None
    view_503: "f32[1568, 1536]" = torch.ops.aten.view.default(view_502, [1568, 1536]);  view_502 = None
    permute_306: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    mm_40: "f32[1568, 512]" = torch.ops.aten.mm.default(view_503, permute_306);  permute_306 = None
    permute_307: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_503, [1, 0])
    mm_41: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_307, view_314);  permute_307 = view_314 = None
    permute_308: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_90: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
    view_504: "f32[1536]" = torch.ops.aten.view.default(sum_90, [1536]);  sum_90 = None
    permute_309: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
    view_505: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_40, [8, 1, 196, 512]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_110: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_139, getitem_142);  add_139 = getitem_142 = None
    mul_403: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_40);  sub_110 = None
    mul_404: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_505, primals_84);  primals_84 = None
    mul_405: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_404, 512)
    sum_91: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [3], True)
    mul_406: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_404, mul_403);  mul_404 = None
    sum_92: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [3], True);  mul_406 = None
    mul_407: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_403, sum_92);  sum_92 = None
    sub_111: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_405, sum_91);  mul_405 = sum_91 = None
    sub_112: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_111, mul_407);  sub_111 = mul_407 = None
    div_81: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 512);  rsqrt_40 = None
    mul_408: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_81, sub_112);  div_81 = sub_112 = None
    mul_409: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_505, mul_403);  mul_403 = None
    sum_93: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 1, 2]);  mul_409 = None
    sum_94: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_505, [0, 1, 2]);  view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_196: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_195, mul_408);  add_195 = mul_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_410: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_196, div_54);  div_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_506: "f32[1568, 512]" = torch.ops.aten.view.default(mul_410, [1568, 512]);  mul_410 = None
    permute_310: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    mm_42: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_506, permute_310);  permute_310 = None
    permute_311: "f32[512, 1568]" = torch.ops.aten.permute.default(view_506, [1, 0])
    mm_43: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_311, view_312);  permute_311 = view_312 = None
    permute_312: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_95: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
    view_507: "f32[512]" = torch.ops.aten.view.default(sum_95, [512]);  sum_95 = None
    permute_313: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_508: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_42, [8, 1, 196, 2048]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_411: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_311, 0.7071067811865476)
    erf_29: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_411);  mul_411 = None
    add_197: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_412: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_197, 0.5);  add_197 = None
    mul_413: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_311, view_311)
    mul_414: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_413, -0.5);  mul_413 = None
    exp_29: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_414);  mul_414 = None
    mul_415: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_416: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_311, mul_415);  view_311 = mul_415 = None
    add_198: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_412, mul_416);  mul_412 = mul_416 = None
    mul_417: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_508, add_198);  view_508 = add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_509: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_417, [1568, 2048]);  mul_417 = None
    permute_314: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    mm_44: "f32[1568, 512]" = torch.ops.aten.mm.default(view_509, permute_314);  permute_314 = None
    permute_315: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_509, [1, 0])
    mm_45: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_315, view_310);  permute_315 = view_310 = None
    permute_316: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_96: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
    view_510: "f32[2048]" = torch.ops.aten.view.default(sum_96, [2048]);  sum_96 = None
    permute_317: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
    view_511: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_44, [8, 1, 196, 512]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_113: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_135, getitem_140);  add_135 = getitem_140 = None
    mul_418: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_39);  sub_113 = None
    mul_419: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_511, primals_82);  primals_82 = None
    mul_420: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_419, 512)
    sum_97: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [3], True)
    mul_421: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_419, mul_418);  mul_419 = None
    sum_98: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [3], True);  mul_421 = None
    mul_422: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_418, sum_98);  sum_98 = None
    sub_114: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_420, sum_97);  mul_420 = sum_97 = None
    sub_115: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_114, mul_422);  sub_114 = mul_422 = None
    div_82: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 512);  rsqrt_39 = None
    mul_423: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_82, sub_115);  div_82 = sub_115 = None
    mul_424: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_511, mul_418);  mul_418 = None
    sum_99: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_424, [0, 1, 2]);  mul_424 = None
    sum_100: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_511, [0, 1, 2]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_199: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_196, mul_423);  add_196 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_425: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_199, div_53);  div_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_512: "f32[1568, 512]" = torch.ops.aten.view.default(mul_425, [1568, 512]);  mul_425 = None
    permute_318: "f32[512, 512]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    mm_46: "f32[1568, 512]" = torch.ops.aten.mm.default(view_512, permute_318);  permute_318 = None
    permute_319: "f32[512, 1568]" = torch.ops.aten.permute.default(view_512, [1, 0])
    mm_47: "f32[512, 512]" = torch.ops.aten.mm.default(permute_319, view_308);  permute_319 = view_308 = None
    permute_320: "f32[512, 512]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_101: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_512, [0], True);  view_512 = None
    view_513: "f32[512]" = torch.ops.aten.view.default(sum_101, [512]);  sum_101 = None
    permute_321: "f32[512, 512]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
    view_514: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_46, [8, 1, 196, 512]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_515: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_514, [8, 1, 196, 32, 16]);  view_514 = None
    permute_322: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_515, [0, 4, 1, 2, 3]);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_184: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
    view_516: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_184, [128, 196, 32]);  clone_184 = None
    permute_323: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
    bmm_68: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_323, view_516);  permute_323 = None
    permute_324: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_305, [0, 2, 1]);  view_305 = None
    bmm_69: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_516, permute_324);  view_516 = permute_324 = None
    view_517: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_68, [8, 16, 1, 196, 32]);  bmm_68 = None
    view_518: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_69, [8, 16, 1, 196, 196]);  bmm_69 = None
    alias_29: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    mul_426: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_518, alias_29);  view_518 = None
    sum_102: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [-1], True)
    mul_427: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_29, sum_102);  alias_29 = sum_102 = None
    sub_116: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_426, mul_427);  mul_426 = mul_427 = None
    view_519: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_116, [128, 196, 196]);  sub_116 = None
    permute_325: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_301, [0, 2, 1]);  view_301 = None
    bmm_70: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_325, view_519);  permute_325 = None
    permute_326: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_302, [0, 2, 1]);  view_302 = None
    bmm_71: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_519, permute_326);  view_519 = permute_326 = None
    view_520: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_70, [8, 16, 1, 32, 196]);  bmm_70 = None
    view_521: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_71, [8, 16, 1, 196, 32]);  bmm_71 = None
    mul_428: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_520, 0.42044820762685725);  view_520 = None
    permute_327: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_428, [0, 1, 2, 4, 3]);  mul_428 = None
    mul_429: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_521, 0.42044820762685725);  view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_5: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_429, permute_327, view_517]);  mul_429 = permute_327 = view_517 = None
    view_522: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_5, [3, 8, 16, 1, 196, 32]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_328: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_522, [1, 3, 4, 0, 2, 5]);  view_522 = None
    clone_185: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    view_523: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_185, [8, 1, 196, 1536]);  clone_185 = None
    view_524: "f32[1568, 1536]" = torch.ops.aten.view.default(view_523, [1568, 1536]);  view_523 = None
    permute_329: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    mm_48: "f32[1568, 512]" = torch.ops.aten.mm.default(view_524, permute_329);  permute_329 = None
    permute_330: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_524, [1, 0])
    mm_49: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_330, view_298);  permute_330 = view_298 = None
    permute_331: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_103: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_524, [0], True);  view_524 = None
    view_525: "f32[1536]" = torch.ops.aten.view.default(sum_103, [1536]);  sum_103 = None
    permute_332: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
    view_526: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_48, [8, 1, 196, 512]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_117: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_132, getitem_135);  add_132 = getitem_135 = None
    mul_430: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_38);  sub_117 = None
    mul_431: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_526, primals_80);  primals_80 = None
    mul_432: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_431, 512)
    sum_104: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [3], True)
    mul_433: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_431, mul_430);  mul_431 = None
    sum_105: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_433, [3], True);  mul_433 = None
    mul_434: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_430, sum_105);  sum_105 = None
    sub_118: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_432, sum_104);  mul_432 = sum_104 = None
    sub_119: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_118, mul_434);  sub_118 = mul_434 = None
    div_83: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 512);  rsqrt_38 = None
    mul_435: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_83, sub_119);  div_83 = sub_119 = None
    mul_436: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_526, mul_430);  mul_430 = None
    sum_106: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 1, 2]);  mul_436 = None
    sum_107: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_526, [0, 1, 2]);  view_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_200: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_199, mul_435);  add_199 = mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_437: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_200, div_51);  div_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_527: "f32[1568, 512]" = torch.ops.aten.view.default(mul_437, [1568, 512]);  mul_437 = None
    permute_333: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    mm_50: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_527, permute_333);  permute_333 = None
    permute_334: "f32[512, 1568]" = torch.ops.aten.permute.default(view_527, [1, 0])
    mm_51: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_334, view_296);  permute_334 = view_296 = None
    permute_335: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_108: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_527, [0], True);  view_527 = None
    view_528: "f32[512]" = torch.ops.aten.view.default(sum_108, [512]);  sum_108 = None
    permute_336: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    view_529: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_50, [8, 1, 196, 2048]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_438: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_295, 0.7071067811865476)
    erf_30: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_438);  mul_438 = None
    add_201: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_439: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_201, 0.5);  add_201 = None
    mul_440: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_295, view_295)
    mul_441: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_440, -0.5);  mul_440 = None
    exp_30: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_441);  mul_441 = None
    mul_442: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_443: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_295, mul_442);  view_295 = mul_442 = None
    add_202: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_439, mul_443);  mul_439 = mul_443 = None
    mul_444: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_529, add_202);  view_529 = add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_530: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_444, [1568, 2048]);  mul_444 = None
    permute_337: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    mm_52: "f32[1568, 512]" = torch.ops.aten.mm.default(view_530, permute_337);  permute_337 = None
    permute_338: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_530, [1, 0])
    mm_53: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_338, view_294);  permute_338 = view_294 = None
    permute_339: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_109: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_530, [0], True);  view_530 = None
    view_531: "f32[2048]" = torch.ops.aten.view.default(sum_109, [2048]);  sum_109 = None
    permute_340: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    view_532: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_52, [8, 1, 196, 512]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_120: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_128, getitem_133);  add_128 = getitem_133 = None
    mul_445: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_37);  sub_120 = None
    mul_446: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_532, primals_78);  primals_78 = None
    mul_447: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_446, 512)
    sum_110: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [3], True)
    mul_448: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_446, mul_445);  mul_446 = None
    sum_111: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_448, [3], True);  mul_448 = None
    mul_449: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_445, sum_111);  sum_111 = None
    sub_121: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_447, sum_110);  mul_447 = sum_110 = None
    sub_122: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_121, mul_449);  sub_121 = mul_449 = None
    div_84: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 512);  rsqrt_37 = None
    mul_450: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_84, sub_122);  div_84 = sub_122 = None
    mul_451: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_532, mul_445);  mul_445 = None
    sum_112: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 1, 2]);  mul_451 = None
    sum_113: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_532, [0, 1, 2]);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_203: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_200, mul_450);  add_200 = mul_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_452: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_203, div_50);  div_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_533: "f32[1568, 512]" = torch.ops.aten.view.default(mul_452, [1568, 512]);  mul_452 = None
    permute_341: "f32[512, 512]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    mm_54: "f32[1568, 512]" = torch.ops.aten.mm.default(view_533, permute_341);  permute_341 = None
    permute_342: "f32[512, 1568]" = torch.ops.aten.permute.default(view_533, [1, 0])
    mm_55: "f32[512, 512]" = torch.ops.aten.mm.default(permute_342, view_292);  permute_342 = view_292 = None
    permute_343: "f32[512, 512]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_114: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_533, [0], True);  view_533 = None
    view_534: "f32[512]" = torch.ops.aten.view.default(sum_114, [512]);  sum_114 = None
    permute_344: "f32[512, 512]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    view_535: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_54, [8, 1, 196, 512]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_536: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_535, [8, 1, 196, 32, 16]);  view_535 = None
    permute_345: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_536, [0, 4, 1, 2, 3]);  view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_186: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_345, memory_format = torch.contiguous_format);  permute_345 = None
    view_537: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_186, [128, 196, 32]);  clone_186 = None
    permute_346: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
    bmm_72: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_346, view_537);  permute_346 = None
    permute_347: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_289, [0, 2, 1]);  view_289 = None
    bmm_73: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_537, permute_347);  view_537 = permute_347 = None
    view_538: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_72, [8, 16, 1, 196, 32]);  bmm_72 = None
    view_539: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_73, [8, 16, 1, 196, 196]);  bmm_73 = None
    alias_30: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_453: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_539, alias_30);  view_539 = None
    sum_115: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [-1], True)
    mul_454: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_30, sum_115);  alias_30 = sum_115 = None
    sub_123: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
    view_540: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_123, [128, 196, 196]);  sub_123 = None
    permute_348: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
    bmm_74: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_348, view_540);  permute_348 = None
    permute_349: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
    bmm_75: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_540, permute_349);  view_540 = permute_349 = None
    view_541: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_74, [8, 16, 1, 32, 196]);  bmm_74 = None
    view_542: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_75, [8, 16, 1, 196, 32]);  bmm_75 = None
    mul_455: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_541, 0.42044820762685725);  view_541 = None
    permute_350: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_455, [0, 1, 2, 4, 3]);  mul_455 = None
    mul_456: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_542, 0.42044820762685725);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_6: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_456, permute_350, view_538]);  mul_456 = permute_350 = view_538 = None
    view_543: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_6, [3, 8, 16, 1, 196, 32]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_351: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_543, [1, 3, 4, 0, 2, 5]);  view_543 = None
    clone_187: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_351, memory_format = torch.contiguous_format);  permute_351 = None
    view_544: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_187, [8, 1, 196, 1536]);  clone_187 = None
    view_545: "f32[1568, 1536]" = torch.ops.aten.view.default(view_544, [1568, 1536]);  view_544 = None
    permute_352: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    mm_56: "f32[1568, 512]" = torch.ops.aten.mm.default(view_545, permute_352);  permute_352 = None
    permute_353: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_545, [1, 0])
    mm_57: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_353, view_282);  permute_353 = view_282 = None
    permute_354: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_116: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_545, [0], True);  view_545 = None
    view_546: "f32[1536]" = torch.ops.aten.view.default(sum_116, [1536]);  sum_116 = None
    permute_355: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    view_547: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_56, [8, 1, 196, 512]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_124: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_125, getitem_128);  add_125 = getitem_128 = None
    mul_457: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_36);  sub_124 = None
    mul_458: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_547, primals_76);  primals_76 = None
    mul_459: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_458, 512)
    sum_117: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_458, [3], True)
    mul_460: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_458, mul_457);  mul_458 = None
    sum_118: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [3], True);  mul_460 = None
    mul_461: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_457, sum_118);  sum_118 = None
    sub_125: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_459, sum_117);  mul_459 = sum_117 = None
    sub_126: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_125, mul_461);  sub_125 = mul_461 = None
    div_85: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 512);  rsqrt_36 = None
    mul_462: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_85, sub_126);  div_85 = sub_126 = None
    mul_463: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_547, mul_457);  mul_457 = None
    sum_119: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 1, 2]);  mul_463 = None
    sum_120: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_547, [0, 1, 2]);  view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_204: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_203, mul_462);  add_203 = mul_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_464: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_204, div_48);  div_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_548: "f32[1568, 512]" = torch.ops.aten.view.default(mul_464, [1568, 512]);  mul_464 = None
    permute_356: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_58: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_548, permute_356);  permute_356 = None
    permute_357: "f32[512, 1568]" = torch.ops.aten.permute.default(view_548, [1, 0])
    mm_59: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_357, view_280);  permute_357 = view_280 = None
    permute_358: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_121: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
    view_549: "f32[512]" = torch.ops.aten.view.default(sum_121, [512]);  sum_121 = None
    permute_359: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_358, [1, 0]);  permute_358 = None
    view_550: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_58, [8, 1, 196, 2048]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_465: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_279, 0.7071067811865476)
    erf_31: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_465);  mul_465 = None
    add_205: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_466: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_205, 0.5);  add_205 = None
    mul_467: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_279, view_279)
    mul_468: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_467, -0.5);  mul_467 = None
    exp_31: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_468);  mul_468 = None
    mul_469: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_470: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_279, mul_469);  view_279 = mul_469 = None
    add_206: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_466, mul_470);  mul_466 = mul_470 = None
    mul_471: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_550, add_206);  view_550 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_551: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_471, [1568, 2048]);  mul_471 = None
    permute_360: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_60: "f32[1568, 512]" = torch.ops.aten.mm.default(view_551, permute_360);  permute_360 = None
    permute_361: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_551, [1, 0])
    mm_61: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_361, view_278);  permute_361 = view_278 = None
    permute_362: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_122: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
    view_552: "f32[2048]" = torch.ops.aten.view.default(sum_122, [2048]);  sum_122 = None
    permute_363: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_553: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_60, [8, 1, 196, 512]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_127: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_121, getitem_126);  add_121 = getitem_126 = None
    mul_472: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_35);  sub_127 = None
    mul_473: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_553, primals_74);  primals_74 = None
    mul_474: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_473, 512)
    sum_123: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [3], True)
    mul_475: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_473, mul_472);  mul_473 = None
    sum_124: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_475, [3], True);  mul_475 = None
    mul_476: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_472, sum_124);  sum_124 = None
    sub_128: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_474, sum_123);  mul_474 = sum_123 = None
    sub_129: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_128, mul_476);  sub_128 = mul_476 = None
    div_86: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 512);  rsqrt_35 = None
    mul_477: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_86, sub_129);  div_86 = sub_129 = None
    mul_478: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_553, mul_472);  mul_472 = None
    sum_125: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_478, [0, 1, 2]);  mul_478 = None
    sum_126: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_553, [0, 1, 2]);  view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_207: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_204, mul_477);  add_204 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_479: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_207, div_47);  div_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_554: "f32[1568, 512]" = torch.ops.aten.view.default(mul_479, [1568, 512]);  mul_479 = None
    permute_364: "f32[512, 512]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_62: "f32[1568, 512]" = torch.ops.aten.mm.default(view_554, permute_364);  permute_364 = None
    permute_365: "f32[512, 1568]" = torch.ops.aten.permute.default(view_554, [1, 0])
    mm_63: "f32[512, 512]" = torch.ops.aten.mm.default(permute_365, view_276);  permute_365 = view_276 = None
    permute_366: "f32[512, 512]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_127: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
    view_555: "f32[512]" = torch.ops.aten.view.default(sum_127, [512]);  sum_127 = None
    permute_367: "f32[512, 512]" = torch.ops.aten.permute.default(permute_366, [1, 0]);  permute_366 = None
    view_556: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_62, [8, 1, 196, 512]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_557: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_556, [8, 1, 196, 32, 16]);  view_556 = None
    permute_368: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_557, [0, 4, 1, 2, 3]);  view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_188: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
    view_558: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_188, [128, 196, 32]);  clone_188 = None
    permute_369: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_272, [0, 2, 1]);  view_272 = None
    bmm_76: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_369, view_558);  permute_369 = None
    permute_370: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
    bmm_77: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_558, permute_370);  view_558 = permute_370 = None
    view_559: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_76, [8, 16, 1, 196, 32]);  bmm_76 = None
    view_560: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_77, [8, 16, 1, 196, 196]);  bmm_77 = None
    alias_31: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    mul_480: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_560, alias_31);  view_560 = None
    sum_128: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_480, [-1], True)
    mul_481: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_31, sum_128);  alias_31 = sum_128 = None
    sub_130: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
    view_561: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_130, [128, 196, 196]);  sub_130 = None
    permute_371: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
    bmm_78: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_371, view_561);  permute_371 = None
    permute_372: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_270, [0, 2, 1]);  view_270 = None
    bmm_79: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_561, permute_372);  view_561 = permute_372 = None
    view_562: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_78, [8, 16, 1, 32, 196]);  bmm_78 = None
    view_563: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_79, [8, 16, 1, 196, 32]);  bmm_79 = None
    mul_482: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_562, 0.42044820762685725);  view_562 = None
    permute_373: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_482, [0, 1, 2, 4, 3]);  mul_482 = None
    mul_483: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_563, 0.42044820762685725);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_7: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_483, permute_373, view_559]);  mul_483 = permute_373 = view_559 = None
    view_564: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_7, [3, 8, 16, 1, 196, 32]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_374: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_564, [1, 3, 4, 0, 2, 5]);  view_564 = None
    clone_189: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
    view_565: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_189, [8, 1, 196, 1536]);  clone_189 = None
    view_566: "f32[1568, 1536]" = torch.ops.aten.view.default(view_565, [1568, 1536]);  view_565 = None
    permute_375: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    mm_64: "f32[1568, 512]" = torch.ops.aten.mm.default(view_566, permute_375);  permute_375 = None
    permute_376: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_566, [1, 0])
    mm_65: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_376, view_266);  permute_376 = view_266 = None
    permute_377: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_129: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_566, [0], True);  view_566 = None
    view_567: "f32[1536]" = torch.ops.aten.view.default(sum_129, [1536]);  sum_129 = None
    permute_378: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    view_568: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_64, [8, 1, 196, 512]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_131: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_118, getitem_121);  add_118 = getitem_121 = None
    mul_484: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_34);  sub_131 = None
    mul_485: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_568, primals_72);  primals_72 = None
    mul_486: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_485, 512)
    sum_130: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [3], True)
    mul_487: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_485, mul_484);  mul_485 = None
    sum_131: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_487, [3], True);  mul_487 = None
    mul_488: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_484, sum_131);  sum_131 = None
    sub_132: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_486, sum_130);  mul_486 = sum_130 = None
    sub_133: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_132, mul_488);  sub_132 = mul_488 = None
    div_87: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 512);  rsqrt_34 = None
    mul_489: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_87, sub_133);  div_87 = sub_133 = None
    mul_490: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_568, mul_484);  mul_484 = None
    sum_132: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 1, 2]);  mul_490 = None
    sum_133: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_568, [0, 1, 2]);  view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_208: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_207, mul_489);  add_207 = mul_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_491: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_208, div_45);  div_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_569: "f32[1568, 512]" = torch.ops.aten.view.default(mul_491, [1568, 512]);  mul_491 = None
    permute_379: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    mm_66: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_569, permute_379);  permute_379 = None
    permute_380: "f32[512, 1568]" = torch.ops.aten.permute.default(view_569, [1, 0])
    mm_67: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_380, view_264);  permute_380 = view_264 = None
    permute_381: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_134: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_569, [0], True);  view_569 = None
    view_570: "f32[512]" = torch.ops.aten.view.default(sum_134, [512]);  sum_134 = None
    permute_382: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_381, [1, 0]);  permute_381 = None
    view_571: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_66, [8, 1, 196, 2048]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_492: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476)
    erf_32: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_492);  mul_492 = None
    add_209: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_493: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_209, 0.5);  add_209 = None
    mul_494: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_263, view_263)
    mul_495: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_494, -0.5);  mul_494 = None
    exp_32: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_495);  mul_495 = None
    mul_496: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_497: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_263, mul_496);  view_263 = mul_496 = None
    add_210: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_493, mul_497);  mul_493 = mul_497 = None
    mul_498: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_571, add_210);  view_571 = add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_572: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_498, [1568, 2048]);  mul_498 = None
    permute_383: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    mm_68: "f32[1568, 512]" = torch.ops.aten.mm.default(view_572, permute_383);  permute_383 = None
    permute_384: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_572, [1, 0])
    mm_69: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_384, view_262);  permute_384 = view_262 = None
    permute_385: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_135: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_572, [0], True);  view_572 = None
    view_573: "f32[2048]" = torch.ops.aten.view.default(sum_135, [2048]);  sum_135 = None
    permute_386: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
    view_574: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_68, [8, 1, 196, 512]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_134: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_114, getitem_119);  add_114 = getitem_119 = None
    mul_499: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_33);  sub_134 = None
    mul_500: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_574, primals_70);  primals_70 = None
    mul_501: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_500, 512)
    sum_136: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [3], True)
    mul_502: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_500, mul_499);  mul_500 = None
    sum_137: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_502, [3], True);  mul_502 = None
    mul_503: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_499, sum_137);  sum_137 = None
    sub_135: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_501, sum_136);  mul_501 = sum_136 = None
    sub_136: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_135, mul_503);  sub_135 = mul_503 = None
    div_88: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 512);  rsqrt_33 = None
    mul_504: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_88, sub_136);  div_88 = sub_136 = None
    mul_505: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_574, mul_499);  mul_499 = None
    sum_138: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_505, [0, 1, 2]);  mul_505 = None
    sum_139: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_574, [0, 1, 2]);  view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_211: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_208, mul_504);  add_208 = mul_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_506: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_211, div_44);  div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_575: "f32[1568, 512]" = torch.ops.aten.view.default(mul_506, [1568, 512]);  mul_506 = None
    permute_387: "f32[512, 512]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    mm_70: "f32[1568, 512]" = torch.ops.aten.mm.default(view_575, permute_387);  permute_387 = None
    permute_388: "f32[512, 1568]" = torch.ops.aten.permute.default(view_575, [1, 0])
    mm_71: "f32[512, 512]" = torch.ops.aten.mm.default(permute_388, view_260);  permute_388 = view_260 = None
    permute_389: "f32[512, 512]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_140: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
    view_576: "f32[512]" = torch.ops.aten.view.default(sum_140, [512]);  sum_140 = None
    permute_390: "f32[512, 512]" = torch.ops.aten.permute.default(permute_389, [1, 0]);  permute_389 = None
    view_577: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_70, [8, 1, 196, 512]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_578: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_577, [8, 1, 196, 32, 16]);  view_577 = None
    permute_391: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_578, [0, 4, 1, 2, 3]);  view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_190: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
    view_579: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_190, [128, 196, 32]);  clone_190 = None
    permute_392: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
    bmm_80: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_392, view_579);  permute_392 = None
    permute_393: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
    bmm_81: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_579, permute_393);  view_579 = permute_393 = None
    view_580: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_80, [8, 16, 1, 196, 32]);  bmm_80 = None
    view_581: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_81, [8, 16, 1, 196, 196]);  bmm_81 = None
    alias_32: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_507: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_581, alias_32);  view_581 = None
    sum_141: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_507, [-1], True)
    mul_508: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_32, sum_141);  alias_32 = sum_141 = None
    sub_137: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_507, mul_508);  mul_507 = mul_508 = None
    view_582: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_137, [128, 196, 196]);  sub_137 = None
    permute_394: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
    bmm_82: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_394, view_582);  permute_394 = None
    permute_395: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    bmm_83: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_582, permute_395);  view_582 = permute_395 = None
    view_583: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_82, [8, 16, 1, 32, 196]);  bmm_82 = None
    view_584: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_83, [8, 16, 1, 196, 32]);  bmm_83 = None
    mul_509: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_583, 0.42044820762685725);  view_583 = None
    permute_396: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_509, [0, 1, 2, 4, 3]);  mul_509 = None
    mul_510: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_584, 0.42044820762685725);  view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_8: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_510, permute_396, view_580]);  mul_510 = permute_396 = view_580 = None
    view_585: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_8, [3, 8, 16, 1, 196, 32]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_397: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_585, [1, 3, 4, 0, 2, 5]);  view_585 = None
    clone_191: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_397, memory_format = torch.contiguous_format);  permute_397 = None
    view_586: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_191, [8, 1, 196, 1536]);  clone_191 = None
    view_587: "f32[1568, 1536]" = torch.ops.aten.view.default(view_586, [1568, 1536]);  view_586 = None
    permute_398: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_72: "f32[1568, 512]" = torch.ops.aten.mm.default(view_587, permute_398);  permute_398 = None
    permute_399: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_73: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_399, view_250);  permute_399 = view_250 = None
    permute_400: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_142: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
    view_588: "f32[1536]" = torch.ops.aten.view.default(sum_142, [1536]);  sum_142 = None
    permute_401: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_589: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_72, [8, 1, 196, 512]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_138: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_111, getitem_114);  add_111 = getitem_114 = None
    mul_511: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_138, rsqrt_32);  sub_138 = None
    mul_512: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_589, primals_68);  primals_68 = None
    mul_513: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_512, 512)
    sum_143: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [3], True)
    mul_514: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_512, mul_511);  mul_512 = None
    sum_144: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_514, [3], True);  mul_514 = None
    mul_515: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_511, sum_144);  sum_144 = None
    sub_139: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_513, sum_143);  mul_513 = sum_143 = None
    sub_140: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_139, mul_515);  sub_139 = mul_515 = None
    div_89: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 512);  rsqrt_32 = None
    mul_516: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_89, sub_140);  div_89 = sub_140 = None
    mul_517: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_589, mul_511);  mul_511 = None
    sum_145: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 1, 2]);  mul_517 = None
    sum_146: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_589, [0, 1, 2]);  view_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_212: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_211, mul_516);  add_211 = mul_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_518: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_212, div_42);  div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_590: "f32[1568, 512]" = torch.ops.aten.view.default(mul_518, [1568, 512]);  mul_518 = None
    permute_402: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    mm_74: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_590, permute_402);  permute_402 = None
    permute_403: "f32[512, 1568]" = torch.ops.aten.permute.default(view_590, [1, 0])
    mm_75: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_403, view_248);  permute_403 = view_248 = None
    permute_404: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_147: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
    view_591: "f32[512]" = torch.ops.aten.view.default(sum_147, [512]);  sum_147 = None
    permute_405: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    view_592: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_74, [8, 1, 196, 2048]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_519: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_247, 0.7071067811865476)
    erf_33: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_519);  mul_519 = None
    add_213: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_520: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_213, 0.5);  add_213 = None
    mul_521: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_247, view_247)
    mul_522: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_521, -0.5);  mul_521 = None
    exp_33: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_522);  mul_522 = None
    mul_523: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_524: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_247, mul_523);  view_247 = mul_523 = None
    add_214: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_520, mul_524);  mul_520 = mul_524 = None
    mul_525: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_592, add_214);  view_592 = add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_593: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_525, [1568, 2048]);  mul_525 = None
    permute_406: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    mm_76: "f32[1568, 512]" = torch.ops.aten.mm.default(view_593, permute_406);  permute_406 = None
    permute_407: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_593, [1, 0])
    mm_77: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_407, view_246);  permute_407 = view_246 = None
    permute_408: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_148: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_593, [0], True);  view_593 = None
    view_594: "f32[2048]" = torch.ops.aten.view.default(sum_148, [2048]);  sum_148 = None
    permute_409: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    view_595: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_76, [8, 1, 196, 512]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_141: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_107, getitem_112);  add_107 = getitem_112 = None
    mul_526: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_141, rsqrt_31);  sub_141 = None
    mul_527: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_595, primals_66);  primals_66 = None
    mul_528: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_527, 512)
    sum_149: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [3], True)
    mul_529: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_527, mul_526);  mul_527 = None
    sum_150: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_529, [3], True);  mul_529 = None
    mul_530: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_526, sum_150);  sum_150 = None
    sub_142: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_528, sum_149);  mul_528 = sum_149 = None
    sub_143: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_142, mul_530);  sub_142 = mul_530 = None
    div_90: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 512);  rsqrt_31 = None
    mul_531: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_90, sub_143);  div_90 = sub_143 = None
    mul_532: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_595, mul_526);  mul_526 = None
    sum_151: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_532, [0, 1, 2]);  mul_532 = None
    sum_152: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_595, [0, 1, 2]);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_215: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_212, mul_531);  add_212 = mul_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_533: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_215, div_41);  div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_596: "f32[1568, 512]" = torch.ops.aten.view.default(mul_533, [1568, 512]);  mul_533 = None
    permute_410: "f32[512, 512]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    mm_78: "f32[1568, 512]" = torch.ops.aten.mm.default(view_596, permute_410);  permute_410 = None
    permute_411: "f32[512, 1568]" = torch.ops.aten.permute.default(view_596, [1, 0])
    mm_79: "f32[512, 512]" = torch.ops.aten.mm.default(permute_411, view_244);  permute_411 = view_244 = None
    permute_412: "f32[512, 512]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_153: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_596, [0], True);  view_596 = None
    view_597: "f32[512]" = torch.ops.aten.view.default(sum_153, [512]);  sum_153 = None
    permute_413: "f32[512, 512]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_598: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_78, [8, 1, 196, 512]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_599: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_598, [8, 1, 196, 32, 16]);  view_598 = None
    permute_414: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_599, [0, 4, 1, 2, 3]);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_192: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_414, memory_format = torch.contiguous_format);  permute_414 = None
    view_600: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_192, [128, 196, 32]);  clone_192 = None
    permute_415: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_240, [0, 2, 1]);  view_240 = None
    bmm_84: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_415, view_600);  permute_415 = None
    permute_416: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_241, [0, 2, 1]);  view_241 = None
    bmm_85: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_600, permute_416);  view_600 = permute_416 = None
    view_601: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_84, [8, 16, 1, 196, 32]);  bmm_84 = None
    view_602: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_85, [8, 16, 1, 196, 196]);  bmm_85 = None
    alias_33: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    mul_534: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_602, alias_33);  view_602 = None
    sum_154: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_534, [-1], True)
    mul_535: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_33, sum_154);  alias_33 = sum_154 = None
    sub_144: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_534, mul_535);  mul_534 = mul_535 = None
    view_603: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_144, [128, 196, 196]);  sub_144 = None
    permute_417: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
    bmm_86: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_417, view_603);  permute_417 = None
    permute_418: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_238, [0, 2, 1]);  view_238 = None
    bmm_87: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_603, permute_418);  view_603 = permute_418 = None
    view_604: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_86, [8, 16, 1, 32, 196]);  bmm_86 = None
    view_605: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_87, [8, 16, 1, 196, 32]);  bmm_87 = None
    mul_536: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_604, 0.42044820762685725);  view_604 = None
    permute_419: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_536, [0, 1, 2, 4, 3]);  mul_536 = None
    mul_537: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_605, 0.42044820762685725);  view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_9: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_537, permute_419, view_601]);  mul_537 = permute_419 = view_601 = None
    view_606: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_9, [3, 8, 16, 1, 196, 32]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_420: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_606, [1, 3, 4, 0, 2, 5]);  view_606 = None
    clone_193: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_607: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_193, [8, 1, 196, 1536]);  clone_193 = None
    view_608: "f32[1568, 1536]" = torch.ops.aten.view.default(view_607, [1568, 1536]);  view_607 = None
    permute_421: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_80: "f32[1568, 512]" = torch.ops.aten.mm.default(view_608, permute_421);  permute_421 = None
    permute_422: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_608, [1, 0])
    mm_81: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_422, view_234);  permute_422 = view_234 = None
    permute_423: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_155: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_608, [0], True);  view_608 = None
    view_609: "f32[1536]" = torch.ops.aten.view.default(sum_155, [1536]);  sum_155 = None
    permute_424: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    view_610: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_80, [8, 1, 196, 512]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_145: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_104, getitem_107);  add_104 = getitem_107 = None
    mul_538: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_30);  sub_145 = None
    mul_539: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_610, primals_64);  primals_64 = None
    mul_540: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_539, 512)
    sum_156: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_539, [3], True)
    mul_541: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_539, mul_538);  mul_539 = None
    sum_157: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_541, [3], True);  mul_541 = None
    mul_542: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_538, sum_157);  sum_157 = None
    sub_146: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_540, sum_156);  mul_540 = sum_156 = None
    sub_147: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_146, mul_542);  sub_146 = mul_542 = None
    div_91: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 512);  rsqrt_30 = None
    mul_543: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_91, sub_147);  div_91 = sub_147 = None
    mul_544: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_610, mul_538);  mul_538 = None
    sum_158: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 1, 2]);  mul_544 = None
    sum_159: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_610, [0, 1, 2]);  view_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_216: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_215, mul_543);  add_215 = mul_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_545: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_216, div_39);  div_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_611: "f32[1568, 512]" = torch.ops.aten.view.default(mul_545, [1568, 512]);  mul_545 = None
    permute_425: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_82: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_611, permute_425);  permute_425 = None
    permute_426: "f32[512, 1568]" = torch.ops.aten.permute.default(view_611, [1, 0])
    mm_83: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_426, view_232);  permute_426 = view_232 = None
    permute_427: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_160: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_611, [0], True);  view_611 = None
    view_612: "f32[512]" = torch.ops.aten.view.default(sum_160, [512]);  sum_160 = None
    permute_428: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    view_613: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_82, [8, 1, 196, 2048]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_546: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_231, 0.7071067811865476)
    erf_34: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_546);  mul_546 = None
    add_217: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_547: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_217, 0.5);  add_217 = None
    mul_548: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_231, view_231)
    mul_549: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_548, -0.5);  mul_548 = None
    exp_34: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_549);  mul_549 = None
    mul_550: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_551: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_231, mul_550);  view_231 = mul_550 = None
    add_218: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_547, mul_551);  mul_547 = mul_551 = None
    mul_552: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_613, add_218);  view_613 = add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_614: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_552, [1568, 2048]);  mul_552 = None
    permute_429: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_84: "f32[1568, 512]" = torch.ops.aten.mm.default(view_614, permute_429);  permute_429 = None
    permute_430: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_614, [1, 0])
    mm_85: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_430, view_230);  permute_430 = view_230 = None
    permute_431: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_161: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_614, [0], True);  view_614 = None
    view_615: "f32[2048]" = torch.ops.aten.view.default(sum_161, [2048]);  sum_161 = None
    permute_432: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_431, [1, 0]);  permute_431 = None
    view_616: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_84, [8, 1, 196, 512]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_148: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_100, getitem_105);  add_100 = getitem_105 = None
    mul_553: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_148, rsqrt_29);  sub_148 = None
    mul_554: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_616, primals_62);  primals_62 = None
    mul_555: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_554, 512)
    sum_162: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_554, [3], True)
    mul_556: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_554, mul_553);  mul_554 = None
    sum_163: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_556, [3], True);  mul_556 = None
    mul_557: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_553, sum_163);  sum_163 = None
    sub_149: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_555, sum_162);  mul_555 = sum_162 = None
    sub_150: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_149, mul_557);  sub_149 = mul_557 = None
    div_92: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 512);  rsqrt_29 = None
    mul_558: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_92, sub_150);  div_92 = sub_150 = None
    mul_559: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_616, mul_553);  mul_553 = None
    sum_164: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 1, 2]);  mul_559 = None
    sum_165: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_616, [0, 1, 2]);  view_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_219: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_216, mul_558);  add_216 = mul_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_560: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_219, div_38);  div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_617: "f32[1568, 512]" = torch.ops.aten.view.default(mul_560, [1568, 512]);  mul_560 = None
    permute_433: "f32[512, 512]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_86: "f32[1568, 512]" = torch.ops.aten.mm.default(view_617, permute_433);  permute_433 = None
    permute_434: "f32[512, 1568]" = torch.ops.aten.permute.default(view_617, [1, 0])
    mm_87: "f32[512, 512]" = torch.ops.aten.mm.default(permute_434, view_228);  permute_434 = view_228 = None
    permute_435: "f32[512, 512]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_166: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_617, [0], True);  view_617 = None
    view_618: "f32[512]" = torch.ops.aten.view.default(sum_166, [512]);  sum_166 = None
    permute_436: "f32[512, 512]" = torch.ops.aten.permute.default(permute_435, [1, 0]);  permute_435 = None
    view_619: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_86, [8, 1, 196, 512]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_620: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_619, [8, 1, 196, 32, 16]);  view_619 = None
    permute_437: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_620, [0, 4, 1, 2, 3]);  view_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_194: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_437, memory_format = torch.contiguous_format);  permute_437 = None
    view_621: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_194, [128, 196, 32]);  clone_194 = None
    permute_438: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_224, [0, 2, 1]);  view_224 = None
    bmm_88: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_438, view_621);  permute_438 = None
    permute_439: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_225, [0, 2, 1]);  view_225 = None
    bmm_89: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_621, permute_439);  view_621 = permute_439 = None
    view_622: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_88, [8, 16, 1, 196, 32]);  bmm_88 = None
    view_623: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_89, [8, 16, 1, 196, 196]);  bmm_89 = None
    alias_34: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_561: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_623, alias_34);  view_623 = None
    sum_167: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_561, [-1], True)
    mul_562: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_34, sum_167);  alias_34 = sum_167 = None
    sub_151: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    view_624: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_151, [128, 196, 196]);  sub_151 = None
    permute_440: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_221, [0, 2, 1]);  view_221 = None
    bmm_90: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_440, view_624);  permute_440 = None
    permute_441: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_222, [0, 2, 1]);  view_222 = None
    bmm_91: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_624, permute_441);  view_624 = permute_441 = None
    view_625: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_90, [8, 16, 1, 32, 196]);  bmm_90 = None
    view_626: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_91, [8, 16, 1, 196, 32]);  bmm_91 = None
    mul_563: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_625, 0.42044820762685725);  view_625 = None
    permute_442: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_563, [0, 1, 2, 4, 3]);  mul_563 = None
    mul_564: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_626, 0.42044820762685725);  view_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_10: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_564, permute_442, view_622]);  mul_564 = permute_442 = view_622 = None
    view_627: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_10, [3, 8, 16, 1, 196, 32]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_443: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_627, [1, 3, 4, 0, 2, 5]);  view_627 = None
    clone_195: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_443, memory_format = torch.contiguous_format);  permute_443 = None
    view_628: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_195, [8, 1, 196, 1536]);  clone_195 = None
    view_629: "f32[1568, 1536]" = torch.ops.aten.view.default(view_628, [1568, 1536]);  view_628 = None
    permute_444: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    mm_88: "f32[1568, 512]" = torch.ops.aten.mm.default(view_629, permute_444);  permute_444 = None
    permute_445: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_629, [1, 0])
    mm_89: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_445, view_218);  permute_445 = view_218 = None
    permute_446: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_168: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_629, [0], True);  view_629 = None
    view_630: "f32[1536]" = torch.ops.aten.view.default(sum_168, [1536]);  sum_168 = None
    permute_447: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    view_631: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_88, [8, 1, 196, 512]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_152: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_97, getitem_100);  add_97 = getitem_100 = None
    mul_565: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_152, rsqrt_28);  sub_152 = None
    mul_566: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_631, primals_60);  primals_60 = None
    mul_567: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_566, 512)
    sum_169: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_566, [3], True)
    mul_568: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_566, mul_565);  mul_566 = None
    sum_170: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_568, [3], True);  mul_568 = None
    mul_569: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_565, sum_170);  sum_170 = None
    sub_153: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_567, sum_169);  mul_567 = sum_169 = None
    sub_154: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_153, mul_569);  sub_153 = mul_569 = None
    div_93: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 512);  rsqrt_28 = None
    mul_570: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_93, sub_154);  div_93 = sub_154 = None
    mul_571: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_631, mul_565);  mul_565 = None
    sum_171: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 1, 2]);  mul_571 = None
    sum_172: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_631, [0, 1, 2]);  view_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_220: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_219, mul_570);  add_219 = mul_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_572: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_220, div_36);  div_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_632: "f32[1568, 512]" = torch.ops.aten.view.default(mul_572, [1568, 512]);  mul_572 = None
    permute_448: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    mm_90: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_632, permute_448);  permute_448 = None
    permute_449: "f32[512, 1568]" = torch.ops.aten.permute.default(view_632, [1, 0])
    mm_91: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_449, view_216);  permute_449 = view_216 = None
    permute_450: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_173: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_632, [0], True);  view_632 = None
    view_633: "f32[512]" = torch.ops.aten.view.default(sum_173, [512]);  sum_173 = None
    permute_451: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    view_634: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_90, [8, 1, 196, 2048]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_573: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_215, 0.7071067811865476)
    erf_35: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_573);  mul_573 = None
    add_221: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_574: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_221, 0.5);  add_221 = None
    mul_575: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_215, view_215)
    mul_576: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_575, -0.5);  mul_575 = None
    exp_35: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_576);  mul_576 = None
    mul_577: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_578: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_215, mul_577);  view_215 = mul_577 = None
    add_222: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_574, mul_578);  mul_574 = mul_578 = None
    mul_579: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_634, add_222);  view_634 = add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_635: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_579, [1568, 2048]);  mul_579 = None
    permute_452: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    mm_92: "f32[1568, 512]" = torch.ops.aten.mm.default(view_635, permute_452);  permute_452 = None
    permute_453: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_635, [1, 0])
    mm_93: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_453, view_214);  permute_453 = view_214 = None
    permute_454: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_174: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_635, [0], True);  view_635 = None
    view_636: "f32[2048]" = torch.ops.aten.view.default(sum_174, [2048]);  sum_174 = None
    permute_455: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
    view_637: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_92, [8, 1, 196, 512]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_155: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_93, getitem_98);  add_93 = getitem_98 = None
    mul_580: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_155, rsqrt_27);  sub_155 = None
    mul_581: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_637, primals_58);  primals_58 = None
    mul_582: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_581, 512)
    sum_175: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [3], True)
    mul_583: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_581, mul_580);  mul_581 = None
    sum_176: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_583, [3], True);  mul_583 = None
    mul_584: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_580, sum_176);  sum_176 = None
    sub_156: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_582, sum_175);  mul_582 = sum_175 = None
    sub_157: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_156, mul_584);  sub_156 = mul_584 = None
    div_94: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 512);  rsqrt_27 = None
    mul_585: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_94, sub_157);  div_94 = sub_157 = None
    mul_586: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_637, mul_580);  mul_580 = None
    sum_177: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_586, [0, 1, 2]);  mul_586 = None
    sum_178: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_637, [0, 1, 2]);  view_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_223: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_220, mul_585);  add_220 = mul_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_587: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_223, div_35);  div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_638: "f32[1568, 512]" = torch.ops.aten.view.default(mul_587, [1568, 512]);  mul_587 = None
    permute_456: "f32[512, 512]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    mm_94: "f32[1568, 512]" = torch.ops.aten.mm.default(view_638, permute_456);  permute_456 = None
    permute_457: "f32[512, 1568]" = torch.ops.aten.permute.default(view_638, [1, 0])
    mm_95: "f32[512, 512]" = torch.ops.aten.mm.default(permute_457, view_212);  permute_457 = view_212 = None
    permute_458: "f32[512, 512]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_179: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_638, [0], True);  view_638 = None
    view_639: "f32[512]" = torch.ops.aten.view.default(sum_179, [512]);  sum_179 = None
    permute_459: "f32[512, 512]" = torch.ops.aten.permute.default(permute_458, [1, 0]);  permute_458 = None
    view_640: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_94, [8, 1, 196, 512]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_641: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_640, [8, 1, 196, 32, 16]);  view_640 = None
    permute_460: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_641, [0, 4, 1, 2, 3]);  view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_196: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
    view_642: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_196, [128, 196, 32]);  clone_196 = None
    permute_461: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    bmm_92: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_461, view_642);  permute_461 = None
    permute_462: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    bmm_93: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_642, permute_462);  view_642 = permute_462 = None
    view_643: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_92, [8, 16, 1, 196, 32]);  bmm_92 = None
    view_644: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_93, [8, 16, 1, 196, 196]);  bmm_93 = None
    alias_35: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_588: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_644, alias_35);  view_644 = None
    sum_180: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [-1], True)
    mul_589: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_35, sum_180);  alias_35 = sum_180 = None
    sub_158: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_588, mul_589);  mul_588 = mul_589 = None
    view_645: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_158, [128, 196, 196]);  sub_158 = None
    permute_463: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    bmm_94: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_463, view_645);  permute_463 = None
    permute_464: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    bmm_95: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_645, permute_464);  view_645 = permute_464 = None
    view_646: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_94, [8, 16, 1, 32, 196]);  bmm_94 = None
    view_647: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_95, [8, 16, 1, 196, 32]);  bmm_95 = None
    mul_590: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_646, 0.42044820762685725);  view_646 = None
    permute_465: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_590, [0, 1, 2, 4, 3]);  mul_590 = None
    mul_591: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_647, 0.42044820762685725);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_11: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_591, permute_465, view_643]);  mul_591 = permute_465 = view_643 = None
    view_648: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_11, [3, 8, 16, 1, 196, 32]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_466: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_648, [1, 3, 4, 0, 2, 5]);  view_648 = None
    clone_197: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_466, memory_format = torch.contiguous_format);  permute_466 = None
    view_649: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_197, [8, 1, 196, 1536]);  clone_197 = None
    view_650: "f32[1568, 1536]" = torch.ops.aten.view.default(view_649, [1568, 1536]);  view_649 = None
    permute_467: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_96: "f32[1568, 512]" = torch.ops.aten.mm.default(view_650, permute_467);  permute_467 = None
    permute_468: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_650, [1, 0])
    mm_97: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_468, view_202);  permute_468 = view_202 = None
    permute_469: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_181: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
    view_651: "f32[1536]" = torch.ops.aten.view.default(sum_181, [1536]);  sum_181 = None
    permute_470: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
    view_652: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_96, [8, 1, 196, 512]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_159: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_90, getitem_93);  add_90 = getitem_93 = None
    mul_592: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_159, rsqrt_26);  sub_159 = None
    mul_593: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_652, primals_56);  primals_56 = None
    mul_594: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_593, 512)
    sum_182: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_593, [3], True)
    mul_595: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_593, mul_592);  mul_593 = None
    sum_183: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_595, [3], True);  mul_595 = None
    mul_596: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_592, sum_183);  sum_183 = None
    sub_160: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_594, sum_182);  mul_594 = sum_182 = None
    sub_161: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_160, mul_596);  sub_160 = mul_596 = None
    div_95: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 512);  rsqrt_26 = None
    mul_597: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_95, sub_161);  div_95 = sub_161 = None
    mul_598: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_652, mul_592);  mul_592 = None
    sum_184: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 1, 2]);  mul_598 = None
    sum_185: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_652, [0, 1, 2]);  view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_224: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_223, mul_597);  add_223 = mul_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_599: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_224, div_33);  div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_653: "f32[1568, 512]" = torch.ops.aten.view.default(mul_599, [1568, 512]);  mul_599 = None
    permute_471: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_98: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_653, permute_471);  permute_471 = None
    permute_472: "f32[512, 1568]" = torch.ops.aten.permute.default(view_653, [1, 0])
    mm_99: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_472, view_200);  permute_472 = view_200 = None
    permute_473: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_186: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_653, [0], True);  view_653 = None
    view_654: "f32[512]" = torch.ops.aten.view.default(sum_186, [512]);  sum_186 = None
    permute_474: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_655: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_98, [8, 1, 196, 2048]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_600: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_199, 0.7071067811865476)
    erf_36: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_600);  mul_600 = None
    add_225: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_601: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_225, 0.5);  add_225 = None
    mul_602: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_199, view_199)
    mul_603: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_602, -0.5);  mul_602 = None
    exp_36: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_603);  mul_603 = None
    mul_604: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_605: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_199, mul_604);  view_199 = mul_604 = None
    add_226: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_601, mul_605);  mul_601 = mul_605 = None
    mul_606: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_655, add_226);  view_655 = add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_656: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_606, [1568, 2048]);  mul_606 = None
    permute_475: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_100: "f32[1568, 512]" = torch.ops.aten.mm.default(view_656, permute_475);  permute_475 = None
    permute_476: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_656, [1, 0])
    mm_101: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_476, view_198);  permute_476 = view_198 = None
    permute_477: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_187: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_656, [0], True);  view_656 = None
    view_657: "f32[2048]" = torch.ops.aten.view.default(sum_187, [2048]);  sum_187 = None
    permute_478: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_658: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_100, [8, 1, 196, 512]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_162: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_86, getitem_91);  add_86 = getitem_91 = None
    mul_607: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_162, rsqrt_25);  sub_162 = None
    mul_608: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_658, primals_54);  primals_54 = None
    mul_609: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_608, 512)
    sum_188: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [3], True)
    mul_610: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_608, mul_607);  mul_608 = None
    sum_189: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_610, [3], True);  mul_610 = None
    mul_611: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_607, sum_189);  sum_189 = None
    sub_163: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_609, sum_188);  mul_609 = sum_188 = None
    sub_164: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_163, mul_611);  sub_163 = mul_611 = None
    div_96: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 512);  rsqrt_25 = None
    mul_612: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_96, sub_164);  div_96 = sub_164 = None
    mul_613: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_658, mul_607);  mul_607 = None
    sum_190: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 1, 2]);  mul_613 = None
    sum_191: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_658, [0, 1, 2]);  view_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_227: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_224, mul_612);  add_224 = mul_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_614: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_227, div_32);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_659: "f32[1568, 512]" = torch.ops.aten.view.default(mul_614, [1568, 512]);  mul_614 = None
    permute_479: "f32[512, 512]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    mm_102: "f32[1568, 512]" = torch.ops.aten.mm.default(view_659, permute_479);  permute_479 = None
    permute_480: "f32[512, 1568]" = torch.ops.aten.permute.default(view_659, [1, 0])
    mm_103: "f32[512, 512]" = torch.ops.aten.mm.default(permute_480, view_196);  permute_480 = view_196 = None
    permute_481: "f32[512, 512]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_192: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_659, [0], True);  view_659 = None
    view_660: "f32[512]" = torch.ops.aten.view.default(sum_192, [512]);  sum_192 = None
    permute_482: "f32[512, 512]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
    view_661: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_102, [8, 1, 196, 512]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_662: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_661, [8, 1, 196, 32, 16]);  view_661 = None
    permute_483: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_662, [0, 4, 1, 2, 3]);  view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_198: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_483, memory_format = torch.contiguous_format);  permute_483 = None
    view_663: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_198, [128, 196, 32]);  clone_198 = None
    permute_484: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_192, [0, 2, 1]);  view_192 = None
    bmm_96: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_484, view_663);  permute_484 = None
    permute_485: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
    bmm_97: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_663, permute_485);  view_663 = permute_485 = None
    view_664: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_96, [8, 16, 1, 196, 32]);  bmm_96 = None
    view_665: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_97, [8, 16, 1, 196, 196]);  bmm_97 = None
    alias_36: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_615: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_665, alias_36);  view_665 = None
    sum_193: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_615, [-1], True)
    mul_616: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_36, sum_193);  alias_36 = sum_193 = None
    sub_165: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_615, mul_616);  mul_615 = mul_616 = None
    view_666: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_165, [128, 196, 196]);  sub_165 = None
    permute_486: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    bmm_98: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_486, view_666);  permute_486 = None
    permute_487: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    bmm_99: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_666, permute_487);  view_666 = permute_487 = None
    view_667: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_98, [8, 16, 1, 32, 196]);  bmm_98 = None
    view_668: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_99, [8, 16, 1, 196, 32]);  bmm_99 = None
    mul_617: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_667, 0.42044820762685725);  view_667 = None
    permute_488: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_617, [0, 1, 2, 4, 3]);  mul_617 = None
    mul_618: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_668, 0.42044820762685725);  view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_12: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_618, permute_488, view_664]);  mul_618 = permute_488 = view_664 = None
    view_669: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_12, [3, 8, 16, 1, 196, 32]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_489: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_669, [1, 3, 4, 0, 2, 5]);  view_669 = None
    clone_199: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
    view_670: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_199, [8, 1, 196, 1536]);  clone_199 = None
    view_671: "f32[1568, 1536]" = torch.ops.aten.view.default(view_670, [1568, 1536]);  view_670 = None
    permute_490: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    mm_104: "f32[1568, 512]" = torch.ops.aten.mm.default(view_671, permute_490);  permute_490 = None
    permute_491: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_671, [1, 0])
    mm_105: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_491, view_186);  permute_491 = view_186 = None
    permute_492: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_194: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_671, [0], True);  view_671 = None
    view_672: "f32[1536]" = torch.ops.aten.view.default(sum_194, [1536]);  sum_194 = None
    permute_493: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_492, [1, 0]);  permute_492 = None
    view_673: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_104, [8, 1, 196, 512]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_166: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_83, getitem_86);  add_83 = getitem_86 = None
    mul_619: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_166, rsqrt_24);  sub_166 = None
    mul_620: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_673, primals_52);  primals_52 = None
    mul_621: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_620, 512)
    sum_195: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [3], True)
    mul_622: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_620, mul_619);  mul_620 = None
    sum_196: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [3], True);  mul_622 = None
    mul_623: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_619, sum_196);  sum_196 = None
    sub_167: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_621, sum_195);  mul_621 = sum_195 = None
    sub_168: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_167, mul_623);  sub_167 = mul_623 = None
    div_97: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 512);  rsqrt_24 = None
    mul_624: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_97, sub_168);  div_97 = sub_168 = None
    mul_625: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_673, mul_619);  mul_619 = None
    sum_197: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_625, [0, 1, 2]);  mul_625 = None
    sum_198: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_673, [0, 1, 2]);  view_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_228: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_227, mul_624);  add_227 = mul_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_626: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_228, div_30);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_674: "f32[1568, 512]" = torch.ops.aten.view.default(mul_626, [1568, 512]);  mul_626 = None
    permute_494: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    mm_106: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_674, permute_494);  permute_494 = None
    permute_495: "f32[512, 1568]" = torch.ops.aten.permute.default(view_674, [1, 0])
    mm_107: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_495, view_184);  permute_495 = view_184 = None
    permute_496: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_199: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_674, [0], True);  view_674 = None
    view_675: "f32[512]" = torch.ops.aten.view.default(sum_199, [512]);  sum_199 = None
    permute_497: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_496, [1, 0]);  permute_496 = None
    view_676: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_106, [8, 1, 196, 2048]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_627: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_183, 0.7071067811865476)
    erf_37: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_627);  mul_627 = None
    add_229: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_628: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_229, 0.5);  add_229 = None
    mul_629: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_183, view_183)
    mul_630: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_629, -0.5);  mul_629 = None
    exp_37: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_630);  mul_630 = None
    mul_631: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_632: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_183, mul_631);  view_183 = mul_631 = None
    add_230: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_628, mul_632);  mul_628 = mul_632 = None
    mul_633: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_676, add_230);  view_676 = add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_677: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_633, [1568, 2048]);  mul_633 = None
    permute_498: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_108: "f32[1568, 512]" = torch.ops.aten.mm.default(view_677, permute_498);  permute_498 = None
    permute_499: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_677, [1, 0])
    mm_109: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_499, view_182);  permute_499 = view_182 = None
    permute_500: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_200: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_677, [0], True);  view_677 = None
    view_678: "f32[2048]" = torch.ops.aten.view.default(sum_200, [2048]);  sum_200 = None
    permute_501: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    view_679: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_108, [8, 1, 196, 512]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_169: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_79, getitem_84);  add_79 = getitem_84 = None
    mul_634: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_169, rsqrt_23);  sub_169 = None
    mul_635: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_679, primals_50);  primals_50 = None
    mul_636: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_635, 512)
    sum_201: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_635, [3], True)
    mul_637: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_635, mul_634);  mul_635 = None
    sum_202: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_637, [3], True);  mul_637 = None
    mul_638: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_634, sum_202);  sum_202 = None
    sub_170: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_636, sum_201);  mul_636 = sum_201 = None
    sub_171: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_170, mul_638);  sub_170 = mul_638 = None
    div_98: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 512);  rsqrt_23 = None
    mul_639: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_98, sub_171);  div_98 = sub_171 = None
    mul_640: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_679, mul_634);  mul_634 = None
    sum_203: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 1, 2]);  mul_640 = None
    sum_204: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_679, [0, 1, 2]);  view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_231: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_228, mul_639);  add_228 = mul_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_641: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_231, div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_680: "f32[1568, 512]" = torch.ops.aten.view.default(mul_641, [1568, 512]);  mul_641 = None
    permute_502: "f32[512, 512]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_110: "f32[1568, 512]" = torch.ops.aten.mm.default(view_680, permute_502);  permute_502 = None
    permute_503: "f32[512, 1568]" = torch.ops.aten.permute.default(view_680, [1, 0])
    mm_111: "f32[512, 512]" = torch.ops.aten.mm.default(permute_503, view_180);  permute_503 = view_180 = None
    permute_504: "f32[512, 512]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_205: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_680, [0], True);  view_680 = None
    view_681: "f32[512]" = torch.ops.aten.view.default(sum_205, [512]);  sum_205 = None
    permute_505: "f32[512, 512]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    view_682: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_110, [8, 1, 196, 512]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_683: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_682, [8, 1, 196, 32, 16]);  view_682 = None
    permute_506: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_683, [0, 4, 1, 2, 3]);  view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_200: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_506, memory_format = torch.contiguous_format);  permute_506 = None
    view_684: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_200, [128, 196, 32]);  clone_200 = None
    permute_507: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
    bmm_100: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_507, view_684);  permute_507 = None
    permute_508: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_177, [0, 2, 1]);  view_177 = None
    bmm_101: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_684, permute_508);  view_684 = permute_508 = None
    view_685: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_100, [8, 16, 1, 196, 32]);  bmm_100 = None
    view_686: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_101, [8, 16, 1, 196, 196]);  bmm_101 = None
    alias_37: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_642: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_686, alias_37);  view_686 = None
    sum_206: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [-1], True)
    mul_643: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_37, sum_206);  alias_37 = sum_206 = None
    sub_172: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_642, mul_643);  mul_642 = mul_643 = None
    view_687: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_172, [128, 196, 196]);  sub_172 = None
    permute_509: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
    bmm_102: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_509, view_687);  permute_509 = None
    permute_510: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_174, [0, 2, 1]);  view_174 = None
    bmm_103: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_687, permute_510);  view_687 = permute_510 = None
    view_688: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_102, [8, 16, 1, 32, 196]);  bmm_102 = None
    view_689: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_103, [8, 16, 1, 196, 32]);  bmm_103 = None
    mul_644: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_688, 0.42044820762685725);  view_688 = None
    permute_511: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_644, [0, 1, 2, 4, 3]);  mul_644 = None
    mul_645: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_689, 0.42044820762685725);  view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_13: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_645, permute_511, view_685]);  mul_645 = permute_511 = view_685 = None
    view_690: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_13, [3, 8, 16, 1, 196, 32]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_512: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_690, [1, 3, 4, 0, 2, 5]);  view_690 = None
    clone_201: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_512, memory_format = torch.contiguous_format);  permute_512 = None
    view_691: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_201, [8, 1, 196, 1536]);  clone_201 = None
    view_692: "f32[1568, 1536]" = torch.ops.aten.view.default(view_691, [1568, 1536]);  view_691 = None
    permute_513: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    mm_112: "f32[1568, 512]" = torch.ops.aten.mm.default(view_692, permute_513);  permute_513 = None
    permute_514: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_692, [1, 0])
    mm_113: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_514, view_170);  permute_514 = view_170 = None
    permute_515: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_207: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_692, [0], True);  view_692 = None
    view_693: "f32[1536]" = torch.ops.aten.view.default(sum_207, [1536]);  sum_207 = None
    permute_516: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
    view_694: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_112, [8, 1, 196, 512]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_173: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_76, getitem_79);  add_76 = getitem_79 = None
    mul_646: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_173, rsqrt_22);  sub_173 = None
    mul_647: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_694, primals_48);  primals_48 = None
    mul_648: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_647, 512)
    sum_208: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_647, [3], True)
    mul_649: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_647, mul_646);  mul_647 = None
    sum_209: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_649, [3], True);  mul_649 = None
    mul_650: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_646, sum_209);  sum_209 = None
    sub_174: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_648, sum_208);  mul_648 = sum_208 = None
    sub_175: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_174, mul_650);  sub_174 = mul_650 = None
    div_99: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 512);  rsqrt_22 = None
    mul_651: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_99, sub_175);  div_99 = sub_175 = None
    mul_652: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_694, mul_646);  mul_646 = None
    sum_210: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_652, [0, 1, 2]);  mul_652 = None
    sum_211: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_694, [0, 1, 2]);  view_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_232: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_231, mul_651);  add_231 = mul_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_653: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_232, div_27);  div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_695: "f32[1568, 512]" = torch.ops.aten.view.default(mul_653, [1568, 512]);  mul_653 = None
    permute_517: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    mm_114: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_695, permute_517);  permute_517 = None
    permute_518: "f32[512, 1568]" = torch.ops.aten.permute.default(view_695, [1, 0])
    mm_115: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_518, view_168);  permute_518 = view_168 = None
    permute_519: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_212: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_695, [0], True);  view_695 = None
    view_696: "f32[512]" = torch.ops.aten.view.default(sum_212, [512]);  sum_212 = None
    permute_520: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_519, [1, 0]);  permute_519 = None
    view_697: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_114, [8, 1, 196, 2048]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_654: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_167, 0.7071067811865476)
    erf_38: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_654);  mul_654 = None
    add_233: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    mul_655: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_233, 0.5);  add_233 = None
    mul_656: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_167, view_167)
    mul_657: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_656, -0.5);  mul_656 = None
    exp_38: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_657);  mul_657 = None
    mul_658: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_659: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_167, mul_658);  view_167 = mul_658 = None
    add_234: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_655, mul_659);  mul_655 = mul_659 = None
    mul_660: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_697, add_234);  view_697 = add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_698: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_660, [1568, 2048]);  mul_660 = None
    permute_521: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    mm_116: "f32[1568, 512]" = torch.ops.aten.mm.default(view_698, permute_521);  permute_521 = None
    permute_522: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_698, [1, 0])
    mm_117: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_522, view_166);  permute_522 = view_166 = None
    permute_523: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_213: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_698, [0], True);  view_698 = None
    view_699: "f32[2048]" = torch.ops.aten.view.default(sum_213, [2048]);  sum_213 = None
    permute_524: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_523, [1, 0]);  permute_523 = None
    view_700: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_116, [8, 1, 196, 512]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_176: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_72, getitem_77);  add_72 = getitem_77 = None
    mul_661: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_176, rsqrt_21);  sub_176 = None
    mul_662: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_700, primals_46);  primals_46 = None
    mul_663: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_662, 512)
    sum_214: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_662, [3], True)
    mul_664: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_662, mul_661);  mul_662 = None
    sum_215: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_664, [3], True);  mul_664 = None
    mul_665: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_661, sum_215);  sum_215 = None
    sub_177: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_663, sum_214);  mul_663 = sum_214 = None
    sub_178: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_177, mul_665);  sub_177 = mul_665 = None
    div_100: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 512);  rsqrt_21 = None
    mul_666: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_100, sub_178);  div_100 = sub_178 = None
    mul_667: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_700, mul_661);  mul_661 = None
    sum_216: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 1, 2]);  mul_667 = None
    sum_217: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_700, [0, 1, 2]);  view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_235: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_232, mul_666);  add_232 = mul_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_668: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_235, div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_701: "f32[1568, 512]" = torch.ops.aten.view.default(mul_668, [1568, 512]);  mul_668 = None
    permute_525: "f32[512, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    mm_118: "f32[1568, 512]" = torch.ops.aten.mm.default(view_701, permute_525);  permute_525 = None
    permute_526: "f32[512, 1568]" = torch.ops.aten.permute.default(view_701, [1, 0])
    mm_119: "f32[512, 512]" = torch.ops.aten.mm.default(permute_526, view_164);  permute_526 = view_164 = None
    permute_527: "f32[512, 512]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_218: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_701, [0], True);  view_701 = None
    view_702: "f32[512]" = torch.ops.aten.view.default(sum_218, [512]);  sum_218 = None
    permute_528: "f32[512, 512]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_703: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_118, [8, 1, 196, 512]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_704: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_703, [8, 1, 196, 32, 16]);  view_703 = None
    permute_529: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_704, [0, 4, 1, 2, 3]);  view_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_202: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_529, memory_format = torch.contiguous_format);  permute_529 = None
    view_705: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_202, [128, 196, 32]);  clone_202 = None
    permute_530: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_160, [0, 2, 1]);  view_160 = None
    bmm_104: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_530, view_705);  permute_530 = None
    permute_531: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    bmm_105: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_705, permute_531);  view_705 = permute_531 = None
    view_706: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_104, [8, 16, 1, 196, 32]);  bmm_104 = None
    view_707: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_105, [8, 16, 1, 196, 196]);  bmm_105 = None
    alias_38: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_669: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_707, alias_38);  view_707 = None
    sum_219: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_669, [-1], True)
    mul_670: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_38, sum_219);  alias_38 = sum_219 = None
    sub_179: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    view_708: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_179, [128, 196, 196]);  sub_179 = None
    permute_532: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
    bmm_106: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_532, view_708);  permute_532 = None
    permute_533: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
    bmm_107: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_708, permute_533);  view_708 = permute_533 = None
    view_709: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_106, [8, 16, 1, 32, 196]);  bmm_106 = None
    view_710: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_107, [8, 16, 1, 196, 32]);  bmm_107 = None
    mul_671: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_709, 0.42044820762685725);  view_709 = None
    permute_534: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_671, [0, 1, 2, 4, 3]);  mul_671 = None
    mul_672: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_710, 0.42044820762685725);  view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_14: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_672, permute_534, view_706]);  mul_672 = permute_534 = view_706 = None
    view_711: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_14, [3, 8, 16, 1, 196, 32]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_535: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_711, [1, 3, 4, 0, 2, 5]);  view_711 = None
    clone_203: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_535, memory_format = torch.contiguous_format);  permute_535 = None
    view_712: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_203, [8, 1, 196, 1536]);  clone_203 = None
    view_713: "f32[1568, 1536]" = torch.ops.aten.view.default(view_712, [1568, 1536]);  view_712 = None
    permute_536: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_120: "f32[1568, 512]" = torch.ops.aten.mm.default(view_713, permute_536);  permute_536 = None
    permute_537: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_713, [1, 0])
    mm_121: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_537, view_154);  permute_537 = view_154 = None
    permute_538: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_220: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_713, [0], True);  view_713 = None
    view_714: "f32[1536]" = torch.ops.aten.view.default(sum_220, [1536]);  sum_220 = None
    permute_539: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_538, [1, 0]);  permute_538 = None
    view_715: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_120, [8, 1, 196, 512]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_180: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_69, getitem_72);  add_69 = getitem_72 = None
    mul_673: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_180, rsqrt_20);  sub_180 = None
    mul_674: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_715, primals_44);  primals_44 = None
    mul_675: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_674, 512)
    sum_221: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_674, [3], True)
    mul_676: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_674, mul_673);  mul_674 = None
    sum_222: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_676, [3], True);  mul_676 = None
    mul_677: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_673, sum_222);  sum_222 = None
    sub_181: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_675, sum_221);  mul_675 = sum_221 = None
    sub_182: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_181, mul_677);  sub_181 = mul_677 = None
    div_101: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 512);  rsqrt_20 = None
    mul_678: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_101, sub_182);  div_101 = sub_182 = None
    mul_679: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_715, mul_673);  mul_673 = None
    sum_223: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 1, 2]);  mul_679 = None
    sum_224: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_715, [0, 1, 2]);  view_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_236: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_235, mul_678);  add_235 = mul_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_680: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_236, div_24);  div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_716: "f32[1568, 512]" = torch.ops.aten.view.default(mul_680, [1568, 512]);  mul_680 = None
    permute_540: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_122: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_716, permute_540);  permute_540 = None
    permute_541: "f32[512, 1568]" = torch.ops.aten.permute.default(view_716, [1, 0])
    mm_123: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_541, view_152);  permute_541 = view_152 = None
    permute_542: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_225: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_716, [0], True);  view_716 = None
    view_717: "f32[512]" = torch.ops.aten.view.default(sum_225, [512]);  sum_225 = None
    permute_543: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_542, [1, 0]);  permute_542 = None
    view_718: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_122, [8, 1, 196, 2048]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_681: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_39: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_681);  mul_681 = None
    add_237: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    mul_682: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_237, 0.5);  add_237 = None
    mul_683: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_684: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_683, -0.5);  mul_683 = None
    exp_39: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_684);  mul_684 = None
    mul_685: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_686: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_151, mul_685);  view_151 = mul_685 = None
    add_238: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_682, mul_686);  mul_682 = mul_686 = None
    mul_687: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_718, add_238);  view_718 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_719: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_687, [1568, 2048]);  mul_687 = None
    permute_544: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_124: "f32[1568, 512]" = torch.ops.aten.mm.default(view_719, permute_544);  permute_544 = None
    permute_545: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_719, [1, 0])
    mm_125: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_545, view_150);  permute_545 = view_150 = None
    permute_546: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_226: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_719, [0], True);  view_719 = None
    view_720: "f32[2048]" = torch.ops.aten.view.default(sum_226, [2048]);  sum_226 = None
    permute_547: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_546, [1, 0]);  permute_546 = None
    view_721: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_124, [8, 1, 196, 512]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_183: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_65, getitem_70);  add_65 = getitem_70 = None
    mul_688: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_183, rsqrt_19);  sub_183 = None
    mul_689: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_721, primals_42);  primals_42 = None
    mul_690: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_689, 512)
    sum_227: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_689, [3], True)
    mul_691: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_689, mul_688);  mul_689 = None
    sum_228: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_691, [3], True);  mul_691 = None
    mul_692: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_688, sum_228);  sum_228 = None
    sub_184: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_690, sum_227);  mul_690 = sum_227 = None
    sub_185: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_184, mul_692);  sub_184 = mul_692 = None
    div_102: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 512);  rsqrt_19 = None
    mul_693: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_102, sub_185);  div_102 = sub_185 = None
    mul_694: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_721, mul_688);  mul_688 = None
    sum_229: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_694, [0, 1, 2]);  mul_694 = None
    sum_230: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_721, [0, 1, 2]);  view_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_239: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_236, mul_693);  add_236 = mul_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_695: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_239, div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_722: "f32[1568, 512]" = torch.ops.aten.view.default(mul_695, [1568, 512]);  mul_695 = None
    permute_548: "f32[512, 512]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    mm_126: "f32[1568, 512]" = torch.ops.aten.mm.default(view_722, permute_548);  permute_548 = None
    permute_549: "f32[512, 1568]" = torch.ops.aten.permute.default(view_722, [1, 0])
    mm_127: "f32[512, 512]" = torch.ops.aten.mm.default(permute_549, view_148);  permute_549 = view_148 = None
    permute_550: "f32[512, 512]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_231: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_722, [0], True);  view_722 = None
    view_723: "f32[512]" = torch.ops.aten.view.default(sum_231, [512]);  sum_231 = None
    permute_551: "f32[512, 512]" = torch.ops.aten.permute.default(permute_550, [1, 0]);  permute_550 = None
    view_724: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_126, [8, 1, 196, 512]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_725: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_724, [8, 1, 196, 32, 16]);  view_724 = None
    permute_552: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_725, [0, 4, 1, 2, 3]);  view_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_204: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_552, memory_format = torch.contiguous_format);  permute_552 = None
    view_726: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_204, [128, 196, 32]);  clone_204 = None
    permute_553: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    bmm_108: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_553, view_726);  permute_553 = None
    permute_554: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
    bmm_109: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_726, permute_554);  view_726 = permute_554 = None
    view_727: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_108, [8, 16, 1, 196, 32]);  bmm_108 = None
    view_728: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_109, [8, 16, 1, 196, 196]);  bmm_109 = None
    alias_39: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_696: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_728, alias_39);  view_728 = None
    sum_232: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_696, [-1], True)
    mul_697: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_39, sum_232);  alias_39 = sum_232 = None
    sub_186: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_696, mul_697);  mul_696 = mul_697 = None
    view_729: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_186, [128, 196, 196]);  sub_186 = None
    permute_555: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
    bmm_110: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_555, view_729);  permute_555 = None
    permute_556: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None
    bmm_111: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_729, permute_556);  view_729 = permute_556 = None
    view_730: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_110, [8, 16, 1, 32, 196]);  bmm_110 = None
    view_731: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_111, [8, 16, 1, 196, 32]);  bmm_111 = None
    mul_698: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_730, 0.42044820762685725);  view_730 = None
    permute_557: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_698, [0, 1, 2, 4, 3]);  mul_698 = None
    mul_699: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_731, 0.42044820762685725);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_15: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_699, permute_557, view_727]);  mul_699 = permute_557 = view_727 = None
    view_732: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_15, [3, 8, 16, 1, 196, 32]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_558: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_732, [1, 3, 4, 0, 2, 5]);  view_732 = None
    clone_205: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_558, memory_format = torch.contiguous_format);  permute_558 = None
    view_733: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_205, [8, 1, 196, 1536]);  clone_205 = None
    view_734: "f32[1568, 1536]" = torch.ops.aten.view.default(view_733, [1568, 1536]);  view_733 = None
    permute_559: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    mm_128: "f32[1568, 512]" = torch.ops.aten.mm.default(view_734, permute_559);  permute_559 = None
    permute_560: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_734, [1, 0])
    mm_129: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_560, view_138);  permute_560 = view_138 = None
    permute_561: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_233: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
    view_735: "f32[1536]" = torch.ops.aten.view.default(sum_233, [1536]);  sum_233 = None
    permute_562: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_561, [1, 0]);  permute_561 = None
    view_736: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_128, [8, 1, 196, 512]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_187: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_62, getitem_65);  add_62 = getitem_65 = None
    mul_700: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_187, rsqrt_18);  sub_187 = None
    mul_701: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_736, primals_40);  primals_40 = None
    mul_702: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_701, 512)
    sum_234: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_701, [3], True)
    mul_703: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_701, mul_700);  mul_701 = None
    sum_235: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_703, [3], True);  mul_703 = None
    mul_704: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_700, sum_235);  sum_235 = None
    sub_188: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_702, sum_234);  mul_702 = sum_234 = None
    sub_189: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_188, mul_704);  sub_188 = mul_704 = None
    div_103: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 512);  rsqrt_18 = None
    mul_705: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_103, sub_189);  div_103 = sub_189 = None
    mul_706: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_736, mul_700);  mul_700 = None
    sum_236: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 1, 2]);  mul_706 = None
    sum_237: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_736, [0, 1, 2]);  view_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_240: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_239, mul_705);  add_239 = mul_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_707: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_240, div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_737: "f32[1568, 512]" = torch.ops.aten.view.default(mul_707, [1568, 512]);  mul_707 = None
    permute_563: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    mm_130: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_737, permute_563);  permute_563 = None
    permute_564: "f32[512, 1568]" = torch.ops.aten.permute.default(view_737, [1, 0])
    mm_131: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_564, view_136);  permute_564 = view_136 = None
    permute_565: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_238: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
    view_738: "f32[512]" = torch.ops.aten.view.default(sum_238, [512]);  sum_238 = None
    permute_566: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_565, [1, 0]);  permute_565 = None
    view_739: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_130, [8, 1, 196, 2048]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_708: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_135, 0.7071067811865476)
    erf_40: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_708);  mul_708 = None
    add_241: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
    mul_709: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_241, 0.5);  add_241 = None
    mul_710: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_135, view_135)
    mul_711: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_710, -0.5);  mul_710 = None
    exp_40: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_711);  mul_711 = None
    mul_712: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
    mul_713: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_135, mul_712);  view_135 = mul_712 = None
    add_242: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_709, mul_713);  mul_709 = mul_713 = None
    mul_714: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_739, add_242);  view_739 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_740: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_714, [1568, 2048]);  mul_714 = None
    permute_567: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    mm_132: "f32[1568, 512]" = torch.ops.aten.mm.default(view_740, permute_567);  permute_567 = None
    permute_568: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_740, [1, 0])
    mm_133: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_568, view_134);  permute_568 = view_134 = None
    permute_569: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_239: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
    view_741: "f32[2048]" = torch.ops.aten.view.default(sum_239, [2048]);  sum_239 = None
    permute_570: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_569, [1, 0]);  permute_569 = None
    view_742: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_132, [8, 1, 196, 512]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_190: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_58, getitem_63);  add_58 = getitem_63 = None
    mul_715: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_190, rsqrt_17);  sub_190 = None
    mul_716: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_742, primals_38);  primals_38 = None
    mul_717: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_716, 512)
    sum_240: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_716, [3], True)
    mul_718: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_716, mul_715);  mul_716 = None
    sum_241: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_718, [3], True);  mul_718 = None
    mul_719: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_715, sum_241);  sum_241 = None
    sub_191: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_717, sum_240);  mul_717 = sum_240 = None
    sub_192: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_191, mul_719);  sub_191 = mul_719 = None
    div_104: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 512);  rsqrt_17 = None
    mul_720: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_104, sub_192);  div_104 = sub_192 = None
    mul_721: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_742, mul_715);  mul_715 = None
    sum_242: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 1, 2]);  mul_721 = None
    sum_243: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_742, [0, 1, 2]);  view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_243: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_240, mul_720);  add_240 = mul_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_722: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_243, div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_743: "f32[1568, 512]" = torch.ops.aten.view.default(mul_722, [1568, 512]);  mul_722 = None
    permute_571: "f32[512, 512]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_134: "f32[1568, 512]" = torch.ops.aten.mm.default(view_743, permute_571);  permute_571 = None
    permute_572: "f32[512, 1568]" = torch.ops.aten.permute.default(view_743, [1, 0])
    mm_135: "f32[512, 512]" = torch.ops.aten.mm.default(permute_572, view_132);  permute_572 = view_132 = None
    permute_573: "f32[512, 512]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_244: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_743, [0], True);  view_743 = None
    view_744: "f32[512]" = torch.ops.aten.view.default(sum_244, [512]);  sum_244 = None
    permute_574: "f32[512, 512]" = torch.ops.aten.permute.default(permute_573, [1, 0]);  permute_573 = None
    view_745: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_134, [8, 1, 196, 512]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_746: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_745, [8, 1, 196, 32, 16]);  view_745 = None
    permute_575: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_746, [0, 4, 1, 2, 3]);  view_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_206: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_575, memory_format = torch.contiguous_format);  permute_575 = None
    view_747: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_206, [128, 196, 32]);  clone_206 = None
    permute_576: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
    bmm_112: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_576, view_747);  permute_576 = None
    permute_577: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None
    bmm_113: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_747, permute_577);  view_747 = permute_577 = None
    view_748: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_112, [8, 16, 1, 196, 32]);  bmm_112 = None
    view_749: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_113, [8, 16, 1, 196, 196]);  bmm_113 = None
    alias_40: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_723: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_749, alias_40);  view_749 = None
    sum_245: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_723, [-1], True)
    mul_724: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_40, sum_245);  alias_40 = sum_245 = None
    sub_193: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_723, mul_724);  mul_723 = mul_724 = None
    view_750: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_193, [128, 196, 196]);  sub_193 = None
    permute_578: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    bmm_114: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_578, view_750);  permute_578 = None
    permute_579: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_126, [0, 2, 1]);  view_126 = None
    bmm_115: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_750, permute_579);  view_750 = permute_579 = None
    view_751: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_114, [8, 16, 1, 32, 196]);  bmm_114 = None
    view_752: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_115, [8, 16, 1, 196, 32]);  bmm_115 = None
    mul_725: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_751, 0.42044820762685725);  view_751 = None
    permute_580: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_725, [0, 1, 2, 4, 3]);  mul_725 = None
    mul_726: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_752, 0.42044820762685725);  view_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_16: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_726, permute_580, view_748]);  mul_726 = permute_580 = view_748 = None
    view_753: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_16, [3, 8, 16, 1, 196, 32]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_581: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_753, [1, 3, 4, 0, 2, 5]);  view_753 = None
    clone_207: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_581, memory_format = torch.contiguous_format);  permute_581 = None
    view_754: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_207, [8, 1, 196, 1536]);  clone_207 = None
    view_755: "f32[1568, 1536]" = torch.ops.aten.view.default(view_754, [1568, 1536]);  view_754 = None
    permute_582: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_136: "f32[1568, 512]" = torch.ops.aten.mm.default(view_755, permute_582);  permute_582 = None
    permute_583: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_755, [1, 0])
    mm_137: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_583, view_122);  permute_583 = view_122 = None
    permute_584: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_246: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_755, [0], True);  view_755 = None
    view_756: "f32[1536]" = torch.ops.aten.view.default(sum_246, [1536]);  sum_246 = None
    permute_585: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_584, [1, 0]);  permute_584 = None
    view_757: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_136, [8, 1, 196, 512]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_194: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_55, getitem_58);  add_55 = getitem_58 = None
    mul_727: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_194, rsqrt_16);  sub_194 = None
    mul_728: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_757, primals_36);  primals_36 = None
    mul_729: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_728, 512)
    sum_247: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_728, [3], True)
    mul_730: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_728, mul_727);  mul_728 = None
    sum_248: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_730, [3], True);  mul_730 = None
    mul_731: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_727, sum_248);  sum_248 = None
    sub_195: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_729, sum_247);  mul_729 = sum_247 = None
    sub_196: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_195, mul_731);  sub_195 = mul_731 = None
    div_105: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 512);  rsqrt_16 = None
    mul_732: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_105, sub_196);  div_105 = sub_196 = None
    mul_733: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_757, mul_727);  mul_727 = None
    sum_249: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 1, 2]);  mul_733 = None
    sum_250: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_757, [0, 1, 2]);  view_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_244: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_243, mul_732);  add_243 = mul_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_734: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_244, div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_758: "f32[1568, 512]" = torch.ops.aten.view.default(mul_734, [1568, 512]);  mul_734 = None
    permute_586: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    mm_138: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_758, permute_586);  permute_586 = None
    permute_587: "f32[512, 1568]" = torch.ops.aten.permute.default(view_758, [1, 0])
    mm_139: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_587, view_120);  permute_587 = view_120 = None
    permute_588: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_251: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_758, [0], True);  view_758 = None
    view_759: "f32[512]" = torch.ops.aten.view.default(sum_251, [512]);  sum_251 = None
    permute_589: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_588, [1, 0]);  permute_588 = None
    view_760: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_138, [8, 1, 196, 2048]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_735: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_119, 0.7071067811865476)
    erf_41: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_735);  mul_735 = None
    add_245: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
    mul_736: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_245, 0.5);  add_245 = None
    mul_737: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_119, view_119)
    mul_738: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_737, -0.5);  mul_737 = None
    exp_41: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_738);  mul_738 = None
    mul_739: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
    mul_740: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_119, mul_739);  view_119 = mul_739 = None
    add_246: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_736, mul_740);  mul_736 = mul_740 = None
    mul_741: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_760, add_246);  view_760 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_761: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_741, [1568, 2048]);  mul_741 = None
    permute_590: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    mm_140: "f32[1568, 512]" = torch.ops.aten.mm.default(view_761, permute_590);  permute_590 = None
    permute_591: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_761, [1, 0])
    mm_141: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_591, view_118);  permute_591 = view_118 = None
    permute_592: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_252: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_761, [0], True);  view_761 = None
    view_762: "f32[2048]" = torch.ops.aten.view.default(sum_252, [2048]);  sum_252 = None
    permute_593: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_592, [1, 0]);  permute_592 = None
    view_763: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_140, [8, 1, 196, 512]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_197: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_51, getitem_56);  add_51 = getitem_56 = None
    mul_742: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_197, rsqrt_15);  sub_197 = None
    mul_743: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_763, primals_34);  primals_34 = None
    mul_744: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_743, 512)
    sum_253: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_743, [3], True)
    mul_745: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_743, mul_742);  mul_743 = None
    sum_254: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_745, [3], True);  mul_745 = None
    mul_746: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_742, sum_254);  sum_254 = None
    sub_198: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_744, sum_253);  mul_744 = sum_253 = None
    sub_199: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_198, mul_746);  sub_198 = mul_746 = None
    div_106: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 512);  rsqrt_15 = None
    mul_747: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_106, sub_199);  div_106 = sub_199 = None
    mul_748: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_763, mul_742);  mul_742 = None
    sum_255: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_748, [0, 1, 2]);  mul_748 = None
    sum_256: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_763, [0, 1, 2]);  view_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_247: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_244, mul_747);  add_244 = mul_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_749: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_247, div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_764: "f32[1568, 512]" = torch.ops.aten.view.default(mul_749, [1568, 512]);  mul_749 = None
    permute_594: "f32[512, 512]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    mm_142: "f32[1568, 512]" = torch.ops.aten.mm.default(view_764, permute_594);  permute_594 = None
    permute_595: "f32[512, 1568]" = torch.ops.aten.permute.default(view_764, [1, 0])
    mm_143: "f32[512, 512]" = torch.ops.aten.mm.default(permute_595, view_116);  permute_595 = view_116 = None
    permute_596: "f32[512, 512]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_257: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_764, [0], True);  view_764 = None
    view_765: "f32[512]" = torch.ops.aten.view.default(sum_257, [512]);  sum_257 = None
    permute_597: "f32[512, 512]" = torch.ops.aten.permute.default(permute_596, [1, 0]);  permute_596 = None
    view_766: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_142, [8, 1, 196, 512]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_767: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_766, [8, 1, 196, 32, 16]);  view_766 = None
    permute_598: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_767, [0, 4, 1, 2, 3]);  view_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_208: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_598, memory_format = torch.contiguous_format);  permute_598 = None
    view_768: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_208, [128, 196, 32]);  clone_208 = None
    permute_599: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
    bmm_116: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_599, view_768);  permute_599 = None
    permute_600: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
    bmm_117: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_768, permute_600);  view_768 = permute_600 = None
    view_769: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_116, [8, 16, 1, 196, 32]);  bmm_116 = None
    view_770: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_117, [8, 16, 1, 196, 196]);  bmm_117 = None
    alias_41: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_750: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_770, alias_41);  view_770 = None
    sum_258: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_750, [-1], True)
    mul_751: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_41, sum_258);  alias_41 = sum_258 = None
    sub_200: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_750, mul_751);  mul_750 = mul_751 = None
    view_771: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_200, [128, 196, 196]);  sub_200 = None
    permute_601: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_109, [0, 2, 1]);  view_109 = None
    bmm_118: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_601, view_771);  permute_601 = None
    permute_602: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
    bmm_119: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_771, permute_602);  view_771 = permute_602 = None
    view_772: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_118, [8, 16, 1, 32, 196]);  bmm_118 = None
    view_773: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_119, [8, 16, 1, 196, 32]);  bmm_119 = None
    mul_752: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_772, 0.42044820762685725);  view_772 = None
    permute_603: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_752, [0, 1, 2, 4, 3]);  mul_752 = None
    mul_753: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_773, 0.42044820762685725);  view_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_17: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_753, permute_603, view_769]);  mul_753 = permute_603 = view_769 = None
    view_774: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_17, [3, 8, 16, 1, 196, 32]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_604: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_774, [1, 3, 4, 0, 2, 5]);  view_774 = None
    clone_209: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_604, memory_format = torch.contiguous_format);  permute_604 = None
    view_775: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_209, [8, 1, 196, 1536]);  clone_209 = None
    view_776: "f32[1568, 1536]" = torch.ops.aten.view.default(view_775, [1568, 1536]);  view_775 = None
    permute_605: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_144: "f32[1568, 512]" = torch.ops.aten.mm.default(view_776, permute_605);  permute_605 = None
    permute_606: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_776, [1, 0])
    mm_145: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_606, view_106);  permute_606 = view_106 = None
    permute_607: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_259: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_776, [0], True);  view_776 = None
    view_777: "f32[1536]" = torch.ops.aten.view.default(sum_259, [1536]);  sum_259 = None
    permute_608: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_607, [1, 0]);  permute_607 = None
    view_778: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_144, [8, 1, 196, 512]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_201: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_48, getitem_51);  add_48 = getitem_51 = None
    mul_754: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_201, rsqrt_14);  sub_201 = None
    mul_755: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_778, primals_32);  primals_32 = None
    mul_756: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_755, 512)
    sum_260: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_755, [3], True)
    mul_757: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_755, mul_754);  mul_755 = None
    sum_261: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_757, [3], True);  mul_757 = None
    mul_758: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_754, sum_261);  sum_261 = None
    sub_202: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_756, sum_260);  mul_756 = sum_260 = None
    sub_203: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_202, mul_758);  sub_202 = mul_758 = None
    div_107: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 512);  rsqrt_14 = None
    mul_759: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_107, sub_203);  div_107 = sub_203 = None
    mul_760: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_778, mul_754);  mul_754 = None
    sum_262: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_760, [0, 1, 2]);  mul_760 = None
    sum_263: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_778, [0, 1, 2]);  view_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_248: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_247, mul_759);  add_247 = mul_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_761: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_248, div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_779: "f32[1568, 512]" = torch.ops.aten.view.default(mul_761, [1568, 512]);  mul_761 = None
    permute_609: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_146: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_779, permute_609);  permute_609 = None
    permute_610: "f32[512, 1568]" = torch.ops.aten.permute.default(view_779, [1, 0])
    mm_147: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_610, view_104);  permute_610 = view_104 = None
    permute_611: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_264: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_779, [0], True);  view_779 = None
    view_780: "f32[512]" = torch.ops.aten.view.default(sum_264, [512]);  sum_264 = None
    permute_612: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_611, [1, 0]);  permute_611 = None
    view_781: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_146, [8, 1, 196, 2048]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_762: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_103, 0.7071067811865476)
    erf_42: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_762);  mul_762 = None
    add_249: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
    mul_763: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_249, 0.5);  add_249 = None
    mul_764: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_103, view_103)
    mul_765: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_764, -0.5);  mul_764 = None
    exp_42: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_765);  mul_765 = None
    mul_766: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
    mul_767: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_103, mul_766);  view_103 = mul_766 = None
    add_250: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_763, mul_767);  mul_763 = mul_767 = None
    mul_768: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_781, add_250);  view_781 = add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_782: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_768, [1568, 2048]);  mul_768 = None
    permute_613: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_148: "f32[1568, 512]" = torch.ops.aten.mm.default(view_782, permute_613);  permute_613 = None
    permute_614: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_782, [1, 0])
    mm_149: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_614, view_102);  permute_614 = view_102 = None
    permute_615: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_265: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_782, [0], True);  view_782 = None
    view_783: "f32[2048]" = torch.ops.aten.view.default(sum_265, [2048]);  sum_265 = None
    permute_616: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_615, [1, 0]);  permute_615 = None
    view_784: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_148, [8, 1, 196, 512]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_204: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_44, getitem_49);  add_44 = getitem_49 = None
    mul_769: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_204, rsqrt_13);  sub_204 = None
    mul_770: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_784, primals_30);  primals_30 = None
    mul_771: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_770, 512)
    sum_266: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_770, [3], True)
    mul_772: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_770, mul_769);  mul_770 = None
    sum_267: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_772, [3], True);  mul_772 = None
    mul_773: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_769, sum_267);  sum_267 = None
    sub_205: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_771, sum_266);  mul_771 = sum_266 = None
    sub_206: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_205, mul_773);  sub_205 = mul_773 = None
    div_108: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 512);  rsqrt_13 = None
    mul_774: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_108, sub_206);  div_108 = sub_206 = None
    mul_775: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_784, mul_769);  mul_769 = None
    sum_268: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_775, [0, 1, 2]);  mul_775 = None
    sum_269: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_784, [0, 1, 2]);  view_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_251: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_248, mul_774);  add_248 = mul_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_776: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_251, div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_785: "f32[1568, 512]" = torch.ops.aten.view.default(mul_776, [1568, 512]);  mul_776 = None
    permute_617: "f32[512, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_150: "f32[1568, 512]" = torch.ops.aten.mm.default(view_785, permute_617);  permute_617 = None
    permute_618: "f32[512, 1568]" = torch.ops.aten.permute.default(view_785, [1, 0])
    mm_151: "f32[512, 512]" = torch.ops.aten.mm.default(permute_618, view_100);  permute_618 = view_100 = None
    permute_619: "f32[512, 512]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_270: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_785, [0], True);  view_785 = None
    view_786: "f32[512]" = torch.ops.aten.view.default(sum_270, [512]);  sum_270 = None
    permute_620: "f32[512, 512]" = torch.ops.aten.permute.default(permute_619, [1, 0]);  permute_619 = None
    view_787: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_150, [8, 1, 196, 512]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_788: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_787, [8, 1, 196, 32, 16]);  view_787 = None
    permute_621: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_788, [0, 4, 1, 2, 3]);  view_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_210: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_621, memory_format = torch.contiguous_format);  permute_621 = None
    view_789: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_210, [128, 196, 32]);  clone_210 = None
    permute_622: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None
    bmm_120: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_622, view_789);  permute_622 = None
    permute_623: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    bmm_121: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_789, permute_623);  view_789 = permute_623 = None
    view_790: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_120, [8, 16, 1, 196, 32]);  bmm_120 = None
    view_791: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_121, [8, 16, 1, 196, 196]);  bmm_121 = None
    alias_42: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_777: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_791, alias_42);  view_791 = None
    sum_271: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_777, [-1], True)
    mul_778: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_42, sum_271);  alias_42 = sum_271 = None
    sub_207: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_777, mul_778);  mul_777 = mul_778 = None
    view_792: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_207, [128, 196, 196]);  sub_207 = None
    permute_624: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None
    bmm_122: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_624, view_792);  permute_624 = None
    permute_625: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
    bmm_123: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_792, permute_625);  view_792 = permute_625 = None
    view_793: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_122, [8, 16, 1, 32, 196]);  bmm_122 = None
    view_794: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_123, [8, 16, 1, 196, 32]);  bmm_123 = None
    mul_779: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_793, 0.42044820762685725);  view_793 = None
    permute_626: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_779, [0, 1, 2, 4, 3]);  mul_779 = None
    mul_780: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_794, 0.42044820762685725);  view_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_18: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_780, permute_626, view_790]);  mul_780 = permute_626 = view_790 = None
    view_795: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_18, [3, 8, 16, 1, 196, 32]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_627: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_795, [1, 3, 4, 0, 2, 5]);  view_795 = None
    clone_211: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_627, memory_format = torch.contiguous_format);  permute_627 = None
    view_796: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_211, [8, 1, 196, 1536]);  clone_211 = None
    view_797: "f32[1568, 1536]" = torch.ops.aten.view.default(view_796, [1568, 1536]);  view_796 = None
    permute_628: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    mm_152: "f32[1568, 512]" = torch.ops.aten.mm.default(view_797, permute_628);  permute_628 = None
    permute_629: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_797, [1, 0])
    mm_153: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_629, view_90);  permute_629 = view_90 = None
    permute_630: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_272: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_797, [0], True);  view_797 = None
    view_798: "f32[1536]" = torch.ops.aten.view.default(sum_272, [1536]);  sum_272 = None
    permute_631: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_630, [1, 0]);  permute_630 = None
    view_799: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_152, [8, 1, 196, 512]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_208: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_41, getitem_44);  add_41 = getitem_44 = None
    mul_781: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_208, rsqrt_12);  sub_208 = None
    mul_782: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_799, primals_28);  primals_28 = None
    mul_783: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_782, 512)
    sum_273: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_782, [3], True)
    mul_784: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_782, mul_781);  mul_782 = None
    sum_274: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_784, [3], True);  mul_784 = None
    mul_785: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_781, sum_274);  sum_274 = None
    sub_209: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_783, sum_273);  mul_783 = sum_273 = None
    sub_210: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_209, mul_785);  sub_209 = mul_785 = None
    div_109: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 512);  rsqrt_12 = None
    mul_786: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_109, sub_210);  div_109 = sub_210 = None
    mul_787: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_799, mul_781);  mul_781 = None
    sum_275: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 1, 2]);  mul_787 = None
    sum_276: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_799, [0, 1, 2]);  view_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_252: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_251, mul_786);  add_251 = mul_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_788: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_252, div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_800: "f32[1568, 512]" = torch.ops.aten.view.default(mul_788, [1568, 512]);  mul_788 = None
    permute_632: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_154: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_800, permute_632);  permute_632 = None
    permute_633: "f32[512, 1568]" = torch.ops.aten.permute.default(view_800, [1, 0])
    mm_155: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_633, view_88);  permute_633 = view_88 = None
    permute_634: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_277: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_800, [0], True);  view_800 = None
    view_801: "f32[512]" = torch.ops.aten.view.default(sum_277, [512]);  sum_277 = None
    permute_635: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_634, [1, 0]);  permute_634 = None
    view_802: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_154, [8, 1, 196, 2048]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_789: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476)
    erf_43: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_789);  mul_789 = None
    add_253: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
    mul_790: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_253, 0.5);  add_253 = None
    mul_791: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_87, view_87)
    mul_792: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_791, -0.5);  mul_791 = None
    exp_43: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_792);  mul_792 = None
    mul_793: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
    mul_794: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_87, mul_793);  view_87 = mul_793 = None
    add_254: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_790, mul_794);  mul_790 = mul_794 = None
    mul_795: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_802, add_254);  view_802 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_803: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_795, [1568, 2048]);  mul_795 = None
    permute_636: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_156: "f32[1568, 512]" = torch.ops.aten.mm.default(view_803, permute_636);  permute_636 = None
    permute_637: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_803, [1, 0])
    mm_157: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_637, view_86);  permute_637 = view_86 = None
    permute_638: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_278: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_803, [0], True);  view_803 = None
    view_804: "f32[2048]" = torch.ops.aten.view.default(sum_278, [2048]);  sum_278 = None
    permute_639: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_638, [1, 0]);  permute_638 = None
    view_805: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_156, [8, 1, 196, 512]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_211: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_37, getitem_42);  add_37 = getitem_42 = None
    mul_796: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_211, rsqrt_11);  sub_211 = None
    mul_797: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_805, primals_26);  primals_26 = None
    mul_798: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_797, 512)
    sum_279: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_797, [3], True)
    mul_799: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_797, mul_796);  mul_797 = None
    sum_280: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_799, [3], True);  mul_799 = None
    mul_800: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_796, sum_280);  sum_280 = None
    sub_212: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_798, sum_279);  mul_798 = sum_279 = None
    sub_213: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_212, mul_800);  sub_212 = mul_800 = None
    div_110: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 512);  rsqrt_11 = None
    mul_801: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_110, sub_213);  div_110 = sub_213 = None
    mul_802: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_805, mul_796);  mul_796 = None
    sum_281: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_802, [0, 1, 2]);  mul_802 = None
    sum_282: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_805, [0, 1, 2]);  view_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_255: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_252, mul_801);  add_252 = mul_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_803: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_255, div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_806: "f32[1568, 512]" = torch.ops.aten.view.default(mul_803, [1568, 512]);  mul_803 = None
    permute_640: "f32[512, 512]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_158: "f32[1568, 512]" = torch.ops.aten.mm.default(view_806, permute_640);  permute_640 = None
    permute_641: "f32[512, 1568]" = torch.ops.aten.permute.default(view_806, [1, 0])
    mm_159: "f32[512, 512]" = torch.ops.aten.mm.default(permute_641, view_84);  permute_641 = view_84 = None
    permute_642: "f32[512, 512]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_283: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_806, [0], True);  view_806 = None
    view_807: "f32[512]" = torch.ops.aten.view.default(sum_283, [512]);  sum_283 = None
    permute_643: "f32[512, 512]" = torch.ops.aten.permute.default(permute_642, [1, 0]);  permute_642 = None
    view_808: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_158, [8, 1, 196, 512]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_809: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_808, [8, 1, 196, 32, 16]);  view_808 = None
    permute_644: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_809, [0, 4, 1, 2, 3]);  view_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_212: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_644, memory_format = torch.contiguous_format);  permute_644 = None
    view_810: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_212, [128, 196, 32]);  clone_212 = None
    permute_645: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    bmm_124: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_645, view_810);  permute_645 = None
    permute_646: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    bmm_125: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_810, permute_646);  view_810 = permute_646 = None
    view_811: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_124, [8, 16, 1, 196, 32]);  bmm_124 = None
    view_812: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_125, [8, 16, 1, 196, 196]);  bmm_125 = None
    alias_43: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_804: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_812, alias_43);  view_812 = None
    sum_284: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_804, [-1], True)
    mul_805: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_43, sum_284);  alias_43 = sum_284 = None
    sub_214: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_804, mul_805);  mul_804 = mul_805 = None
    view_813: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_214, [128, 196, 196]);  sub_214 = None
    permute_647: "f32[128, 32, 196]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    bmm_126: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_647, view_813);  permute_647 = None
    permute_648: "f32[128, 196, 32]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_127: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_813, permute_648);  view_813 = permute_648 = None
    view_814: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_126, [8, 16, 1, 32, 196]);  bmm_126 = None
    view_815: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_127, [8, 16, 1, 196, 32]);  bmm_127 = None
    mul_806: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_814, 0.42044820762685725);  view_814 = None
    permute_649: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_806, [0, 1, 2, 4, 3]);  mul_806 = None
    mul_807: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_815, 0.42044820762685725);  view_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_19: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_807, permute_649, view_811]);  mul_807 = permute_649 = view_811 = None
    view_816: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_19, [3, 8, 16, 1, 196, 32]);  cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_650: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_816, [1, 3, 4, 0, 2, 5]);  view_816 = None
    clone_213: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
    view_817: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_213, [8, 1, 196, 1536]);  clone_213 = None
    view_818: "f32[1568, 1536]" = torch.ops.aten.view.default(view_817, [1568, 1536]);  view_817 = None
    permute_651: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_160: "f32[1568, 512]" = torch.ops.aten.mm.default(view_818, permute_651);  permute_651 = None
    permute_652: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_818, [1, 0])
    mm_161: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_652, view_74);  permute_652 = view_74 = None
    permute_653: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    sum_285: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
    view_819: "f32[1536]" = torch.ops.aten.view.default(sum_285, [1536]);  sum_285 = None
    permute_654: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_653, [1, 0]);  permute_653 = None
    view_820: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_160, [8, 1, 196, 512]);  mm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_215: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_34, getitem_37);  add_34 = getitem_37 = None
    mul_808: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_215, rsqrt_10);  sub_215 = None
    mul_809: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_820, primals_24);  primals_24 = None
    mul_810: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_809, 512)
    sum_286: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_809, [3], True)
    mul_811: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_809, mul_808);  mul_809 = None
    sum_287: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_811, [3], True);  mul_811 = None
    mul_812: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_808, sum_287);  sum_287 = None
    sub_216: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_810, sum_286);  mul_810 = sum_286 = None
    sub_217: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_216, mul_812);  sub_216 = mul_812 = None
    div_111: "f32[8, 1, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 512);  rsqrt_10 = None
    mul_813: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_111, sub_217);  div_111 = sub_217 = None
    mul_814: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_820, mul_808);  mul_808 = None
    sum_288: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 1, 2]);  mul_814 = None
    sum_289: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_820, [0, 1, 2]);  view_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_256: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_255, mul_813);  add_255 = mul_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    sum_290: "f32[1, 1, 196, 512]" = torch.ops.aten.sum.dim_IntList(add_256, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    view_821: "f32[8, 1, 1, 14, 14, 512]" = torch.ops.aten.view.default(add_256, [8, 1, 1, 14, 14, 512]);  add_256 = None
    permute_655: "f32[8, 1, 14, 1, 14, 512]" = torch.ops.aten.permute.default(view_821, [0, 1, 3, 2, 4, 5]);  view_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    view_822: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(permute_655, [8, 14, 14, 512]);  permute_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute_656: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_822, [0, 3, 1, 2]);  view_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward: "f32[8, 512, 29, 29]" = torch.ops.aten.max_pool2d_with_indices_backward.default(permute_656, constant_pad_nd_1, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_35);  permute_656 = constant_pad_nd_1 = getitem_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_2: "f32[8, 512, 28, 28]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward, [0, -1, 0, -1]);  max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_657: "f32[8, 28, 28, 512]" = torch.ops.aten.permute.default(constant_pad_nd_2, [0, 2, 3, 1]);  constant_pad_nd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_218: "f32[8, 28, 28, 512]" = torch.ops.aten.sub.Tensor(permute_38, getitem_33);  permute_38 = getitem_33 = None
    mul_815: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(sub_218, rsqrt_9);  sub_218 = None
    mul_816: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(permute_657, primals_21);  primals_21 = None
    mul_817: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_816, 512)
    sum_291: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_816, [3], True)
    mul_818: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_816, mul_815);  mul_816 = None
    sum_292: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_818, [3], True);  mul_818 = None
    mul_819: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_815, sum_292);  sum_292 = None
    sub_219: "f32[8, 28, 28, 512]" = torch.ops.aten.sub.Tensor(mul_817, sum_291);  mul_817 = sum_291 = None
    sub_220: "f32[8, 28, 28, 512]" = torch.ops.aten.sub.Tensor(sub_219, mul_819);  sub_219 = mul_819 = None
    div_112: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 512);  rsqrt_9 = None
    mul_820: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(div_112, sub_220);  div_112 = sub_220 = None
    mul_821: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(permute_657, mul_815);  mul_815 = None
    sum_293: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_821, [0, 1, 2]);  mul_821 = None
    sum_294: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_657, [0, 1, 2]);  permute_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_658: "f32[8, 512, 28, 28]" = torch.ops.aten.permute.default(mul_820, [0, 3, 1, 2]);  mul_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:141, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(permute_658, permute_37, primals_142, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  permute_658 = permute_37 = primals_142 = None
    getitem_178: "f32[8, 256, 28, 28]" = convolution_backward[0]
    getitem_179: "f32[512, 256, 3, 3]" = convolution_backward[1]
    getitem_180: "f32[512]" = convolution_backward[2];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_659: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(getitem_178, [0, 2, 3, 1]);  getitem_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    view_823: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.view.default(permute_659, [8, 2, 14, 2, 14, 256]);  permute_659 = None
    permute_660: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.permute.default(view_823, [0, 1, 3, 2, 4, 5]);  view_823 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    clone_214: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.clone.default(permute_660, memory_format = torch.contiguous_format);  permute_660 = None
    view_824: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(clone_214, [8, 4, 196, 256]);  clone_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_822: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_824, div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_825: "f32[6272, 256]" = torch.ops.aten.view.default(mul_822, [6272, 256]);  mul_822 = None
    permute_661: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_162: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_825, permute_661);  permute_661 = None
    permute_662: "f32[256, 6272]" = torch.ops.aten.permute.default(view_825, [1, 0])
    mm_163: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_662, view_68);  permute_662 = view_68 = None
    permute_663: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_295: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_825, [0], True);  view_825 = None
    view_826: "f32[256]" = torch.ops.aten.view.default(sum_295, [256]);  sum_295 = None
    permute_664: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_663, [1, 0]);  permute_663 = None
    view_827: "f32[8, 4, 196, 1024]" = torch.ops.aten.view.default(mm_162, [8, 4, 196, 1024]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_823: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_67, 0.7071067811865476)
    erf_44: "f32[8, 4, 196, 1024]" = torch.ops.aten.erf.default(mul_823);  mul_823 = None
    add_257: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
    mul_824: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(add_257, 0.5);  add_257 = None
    mul_825: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_67, view_67)
    mul_826: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(mul_825, -0.5);  mul_825 = None
    exp_44: "f32[8, 4, 196, 1024]" = torch.ops.aten.exp.default(mul_826);  mul_826 = None
    mul_827: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
    mul_828: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_67, mul_827);  view_67 = mul_827 = None
    add_258: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(mul_824, mul_828);  mul_824 = mul_828 = None
    mul_829: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_827, add_258);  view_827 = add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_828: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_829, [6272, 1024]);  mul_829 = None
    permute_665: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_164: "f32[6272, 256]" = torch.ops.aten.mm.default(view_828, permute_665);  permute_665 = None
    permute_666: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_828, [1, 0])
    mm_165: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_666, view_66);  permute_666 = view_66 = None
    permute_667: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_296: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_828, [0], True);  view_828 = None
    view_829: "f32[1024]" = torch.ops.aten.view.default(sum_296, [1024]);  sum_296 = None
    permute_668: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_667, [1, 0]);  permute_667 = None
    view_830: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(mm_164, [8, 4, 196, 256]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_221: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_27, getitem_31);  add_27 = getitem_31 = None
    mul_830: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_221, rsqrt_8);  sub_221 = None
    mul_831: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_830, primals_19);  primals_19 = None
    mul_832: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_831, 256)
    sum_297: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_831, [3], True)
    mul_833: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_831, mul_830);  mul_831 = None
    sum_298: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_833, [3], True);  mul_833 = None
    mul_834: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_830, sum_298);  sum_298 = None
    sub_222: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(mul_832, sum_297);  mul_832 = sum_297 = None
    sub_223: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(sub_222, mul_834);  sub_222 = mul_834 = None
    div_113: "f32[8, 4, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None
    mul_835: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(div_113, sub_223);  div_113 = sub_223 = None
    mul_836: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_830, mul_830);  mul_830 = None
    sum_299: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 1, 2]);  mul_836 = None
    sum_300: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_830, [0, 1, 2]);  view_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_259: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(view_824, mul_835);  view_824 = mul_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_837: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(add_259, div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_831: "f32[6272, 256]" = torch.ops.aten.view.default(mul_837, [6272, 256]);  mul_837 = None
    permute_669: "f32[256, 256]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_166: "f32[6272, 256]" = torch.ops.aten.mm.default(view_831, permute_669);  permute_669 = None
    permute_670: "f32[256, 6272]" = torch.ops.aten.permute.default(view_831, [1, 0])
    mm_167: "f32[256, 256]" = torch.ops.aten.mm.default(permute_670, view_64);  permute_670 = view_64 = None
    permute_671: "f32[256, 256]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    sum_301: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_831, [0], True);  view_831 = None
    view_832: "f32[256]" = torch.ops.aten.view.default(sum_301, [256]);  sum_301 = None
    permute_672: "f32[256, 256]" = torch.ops.aten.permute.default(permute_671, [1, 0]);  permute_671 = None
    view_833: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(mm_166, [8, 4, 196, 256]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_834: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.view.default(view_833, [8, 4, 196, 32, 8]);  view_833 = None
    permute_673: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(view_834, [0, 4, 1, 2, 3]);  view_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_215: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(permute_673, memory_format = torch.contiguous_format);  permute_673 = None
    view_835: "f32[256, 196, 32]" = torch.ops.aten.view.default(clone_215, [256, 196, 32]);  clone_215 = None
    permute_674: "f32[256, 196, 196]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    bmm_128: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(permute_674, view_835);  permute_674 = None
    permute_675: "f32[256, 32, 196]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    bmm_129: "f32[256, 196, 196]" = torch.ops.aten.bmm.default(view_835, permute_675);  view_835 = permute_675 = None
    view_836: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_128, [8, 8, 4, 196, 32]);  bmm_128 = None
    view_837: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_129, [8, 8, 4, 196, 196]);  bmm_129 = None
    alias_44: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_838: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_837, alias_44);  view_837 = None
    sum_302: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_838, [-1], True)
    mul_839: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(alias_44, sum_302);  alias_44 = sum_302 = None
    sub_224: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(mul_838, mul_839);  mul_838 = mul_839 = None
    view_838: "f32[256, 196, 196]" = torch.ops.aten.view.default(sub_224, [256, 196, 196]);  sub_224 = None
    permute_676: "f32[256, 32, 196]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
    bmm_130: "f32[256, 32, 196]" = torch.ops.aten.bmm.default(permute_676, view_838);  permute_676 = None
    permute_677: "f32[256, 196, 32]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    bmm_131: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(view_838, permute_677);  view_838 = permute_677 = None
    view_839: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.view.default(bmm_130, [8, 8, 4, 32, 196]);  bmm_130 = None
    view_840: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_131, [8, 8, 4, 196, 32]);  bmm_131 = None
    mul_840: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.mul.Scalar(view_839, 0.42044820762685725);  view_839 = None
    permute_678: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(mul_840, [0, 1, 2, 4, 3]);  mul_840 = None
    mul_841: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.mul.Scalar(view_840, 0.42044820762685725);  view_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_20: "f32[24, 8, 4, 196, 32]" = torch.ops.aten.cat.default([mul_841, permute_678, view_836]);  mul_841 = permute_678 = view_836 = None
    view_841: "f32[3, 8, 8, 4, 196, 32]" = torch.ops.aten.view.default(cat_20, [3, 8, 8, 4, 196, 32]);  cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_679: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.permute.default(view_841, [1, 3, 4, 0, 2, 5]);  view_841 = None
    clone_216: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.clone.default(permute_679, memory_format = torch.contiguous_format);  permute_679 = None
    view_842: "f32[8, 4, 196, 768]" = torch.ops.aten.view.default(clone_216, [8, 4, 196, 768]);  clone_216 = None
    view_843: "f32[6272, 768]" = torch.ops.aten.view.default(view_842, [6272, 768]);  view_842 = None
    permute_680: "f32[768, 256]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    mm_168: "f32[6272, 256]" = torch.ops.aten.mm.default(view_843, permute_680);  permute_680 = None
    permute_681: "f32[768, 6272]" = torch.ops.aten.permute.default(view_843, [1, 0])
    mm_169: "f32[768, 256]" = torch.ops.aten.mm.default(permute_681, view_54);  permute_681 = view_54 = None
    permute_682: "f32[256, 768]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    sum_303: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_843, [0], True);  view_843 = None
    view_844: "f32[768]" = torch.ops.aten.view.default(sum_303, [768]);  sum_303 = None
    permute_683: "f32[768, 256]" = torch.ops.aten.permute.default(permute_682, [1, 0]);  permute_682 = None
    view_845: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(mm_168, [8, 4, 196, 256]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_225: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_24, getitem_26);  add_24 = getitem_26 = None
    mul_842: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_225, rsqrt_7);  sub_225 = None
    mul_843: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_845, primals_17);  primals_17 = None
    mul_844: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_843, 256)
    sum_304: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_843, [3], True)
    mul_845: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_843, mul_842);  mul_843 = None
    sum_305: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_845, [3], True);  mul_845 = None
    mul_846: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_842, sum_305);  sum_305 = None
    sub_226: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(mul_844, sum_304);  mul_844 = sum_304 = None
    sub_227: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(sub_226, mul_846);  sub_226 = mul_846 = None
    div_114: "f32[8, 4, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None
    mul_847: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(div_114, sub_227);  div_114 = sub_227 = None
    mul_848: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_845, mul_842);  mul_842 = None
    sum_306: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_848, [0, 1, 2]);  mul_848 = None
    sum_307: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_845, [0, 1, 2]);  view_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_260: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_259, mul_847);  add_259 = mul_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_849: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(add_260, div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_846: "f32[6272, 256]" = torch.ops.aten.view.default(mul_849, [6272, 256]);  mul_849 = None
    permute_684: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    mm_170: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_846, permute_684);  permute_684 = None
    permute_685: "f32[256, 6272]" = torch.ops.aten.permute.default(view_846, [1, 0])
    mm_171: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_685, view_52);  permute_685 = view_52 = None
    permute_686: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_308: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_846, [0], True);  view_846 = None
    view_847: "f32[256]" = torch.ops.aten.view.default(sum_308, [256]);  sum_308 = None
    permute_687: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_686, [1, 0]);  permute_686 = None
    view_848: "f32[8, 4, 196, 1024]" = torch.ops.aten.view.default(mm_170, [8, 4, 196, 1024]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_850: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476)
    erf_45: "f32[8, 4, 196, 1024]" = torch.ops.aten.erf.default(mul_850);  mul_850 = None
    add_261: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
    mul_851: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(add_261, 0.5);  add_261 = None
    mul_852: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_51, view_51)
    mul_853: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(mul_852, -0.5);  mul_852 = None
    exp_45: "f32[8, 4, 196, 1024]" = torch.ops.aten.exp.default(mul_853);  mul_853 = None
    mul_854: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
    mul_855: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_51, mul_854);  view_51 = mul_854 = None
    add_262: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(mul_851, mul_855);  mul_851 = mul_855 = None
    mul_856: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_848, add_262);  view_848 = add_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_849: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_856, [6272, 1024]);  mul_856 = None
    permute_688: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    mm_172: "f32[6272, 256]" = torch.ops.aten.mm.default(view_849, permute_688);  permute_688 = None
    permute_689: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_849, [1, 0])
    mm_173: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_689, view_50);  permute_689 = view_50 = None
    permute_690: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_309: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
    view_850: "f32[1024]" = torch.ops.aten.view.default(sum_309, [1024]);  sum_309 = None
    permute_691: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_690, [1, 0]);  permute_690 = None
    view_851: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(mm_172, [8, 4, 196, 256]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_228: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_20, getitem_24);  add_20 = getitem_24 = None
    mul_857: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_228, rsqrt_6);  sub_228 = None
    mul_858: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_851, primals_15);  primals_15 = None
    mul_859: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_858, 256)
    sum_310: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_858, [3], True)
    mul_860: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_858, mul_857);  mul_858 = None
    sum_311: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_860, [3], True);  mul_860 = None
    mul_861: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_857, sum_311);  sum_311 = None
    sub_229: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(mul_859, sum_310);  mul_859 = sum_310 = None
    sub_230: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(sub_229, mul_861);  sub_229 = mul_861 = None
    div_115: "f32[8, 4, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
    mul_862: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(div_115, sub_230);  div_115 = sub_230 = None
    mul_863: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_851, mul_857);  mul_857 = None
    sum_312: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_863, [0, 1, 2]);  mul_863 = None
    sum_313: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_851, [0, 1, 2]);  view_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_263: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_260, mul_862);  add_260 = mul_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_864: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(add_263, div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_852: "f32[6272, 256]" = torch.ops.aten.view.default(mul_864, [6272, 256]);  mul_864 = None
    permute_692: "f32[256, 256]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    mm_174: "f32[6272, 256]" = torch.ops.aten.mm.default(view_852, permute_692);  permute_692 = None
    permute_693: "f32[256, 6272]" = torch.ops.aten.permute.default(view_852, [1, 0])
    mm_175: "f32[256, 256]" = torch.ops.aten.mm.default(permute_693, view_48);  permute_693 = view_48 = None
    permute_694: "f32[256, 256]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_314: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_852, [0], True);  view_852 = None
    view_853: "f32[256]" = torch.ops.aten.view.default(sum_314, [256]);  sum_314 = None
    permute_695: "f32[256, 256]" = torch.ops.aten.permute.default(permute_694, [1, 0]);  permute_694 = None
    view_854: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(mm_174, [8, 4, 196, 256]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_855: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.view.default(view_854, [8, 4, 196, 32, 8]);  view_854 = None
    permute_696: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(view_855, [0, 4, 1, 2, 3]);  view_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_217: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(permute_696, memory_format = torch.contiguous_format);  permute_696 = None
    view_856: "f32[256, 196, 32]" = torch.ops.aten.view.default(clone_217, [256, 196, 32]);  clone_217 = None
    permute_697: "f32[256, 196, 196]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    bmm_132: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(permute_697, view_856);  permute_697 = None
    permute_698: "f32[256, 32, 196]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None
    bmm_133: "f32[256, 196, 196]" = torch.ops.aten.bmm.default(view_856, permute_698);  view_856 = permute_698 = None
    view_857: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_132, [8, 8, 4, 196, 32]);  bmm_132 = None
    view_858: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_133, [8, 8, 4, 196, 196]);  bmm_133 = None
    alias_45: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_865: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_858, alias_45);  view_858 = None
    sum_315: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_865, [-1], True)
    mul_866: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(alias_45, sum_315);  alias_45 = sum_315 = None
    sub_231: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(mul_865, mul_866);  mul_865 = mul_866 = None
    view_859: "f32[256, 196, 196]" = torch.ops.aten.view.default(sub_231, [256, 196, 196]);  sub_231 = None
    permute_699: "f32[256, 32, 196]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
    bmm_134: "f32[256, 32, 196]" = torch.ops.aten.bmm.default(permute_699, view_859);  permute_699 = None
    permute_700: "f32[256, 196, 32]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None
    bmm_135: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(view_859, permute_700);  view_859 = permute_700 = None
    view_860: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.view.default(bmm_134, [8, 8, 4, 32, 196]);  bmm_134 = None
    view_861: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_135, [8, 8, 4, 196, 32]);  bmm_135 = None
    mul_867: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.mul.Scalar(view_860, 0.42044820762685725);  view_860 = None
    permute_701: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(mul_867, [0, 1, 2, 4, 3]);  mul_867 = None
    mul_868: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.mul.Scalar(view_861, 0.42044820762685725);  view_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_21: "f32[24, 8, 4, 196, 32]" = torch.ops.aten.cat.default([mul_868, permute_701, view_857]);  mul_868 = permute_701 = view_857 = None
    view_862: "f32[3, 8, 8, 4, 196, 32]" = torch.ops.aten.view.default(cat_21, [3, 8, 8, 4, 196, 32]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_702: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.permute.default(view_862, [1, 3, 4, 0, 2, 5]);  view_862 = None
    clone_218: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.clone.default(permute_702, memory_format = torch.contiguous_format);  permute_702 = None
    view_863: "f32[8, 4, 196, 768]" = torch.ops.aten.view.default(clone_218, [8, 4, 196, 768]);  clone_218 = None
    view_864: "f32[6272, 768]" = torch.ops.aten.view.default(view_863, [6272, 768]);  view_863 = None
    permute_703: "f32[768, 256]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_176: "f32[6272, 256]" = torch.ops.aten.mm.default(view_864, permute_703);  permute_703 = None
    permute_704: "f32[768, 6272]" = torch.ops.aten.permute.default(view_864, [1, 0])
    mm_177: "f32[768, 256]" = torch.ops.aten.mm.default(permute_704, view_38);  permute_704 = view_38 = None
    permute_705: "f32[256, 768]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_316: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_864, [0], True);  view_864 = None
    view_865: "f32[768]" = torch.ops.aten.view.default(sum_316, [768]);  sum_316 = None
    permute_706: "f32[768, 256]" = torch.ops.aten.permute.default(permute_705, [1, 0]);  permute_705 = None
    view_866: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(mm_176, [8, 4, 196, 256]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_232: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_17, getitem_19);  add_17 = getitem_19 = None
    mul_869: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_232, rsqrt_5);  sub_232 = None
    mul_870: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_866, primals_13);  primals_13 = None
    mul_871: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_870, 256)
    sum_317: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_870, [3], True)
    mul_872: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_870, mul_869);  mul_870 = None
    sum_318: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_872, [3], True);  mul_872 = None
    mul_873: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_869, sum_318);  sum_318 = None
    sub_233: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(mul_871, sum_317);  mul_871 = sum_317 = None
    sub_234: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(sub_233, mul_873);  sub_233 = mul_873 = None
    div_116: "f32[8, 4, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 256);  rsqrt_5 = None
    mul_874: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(div_116, sub_234);  div_116 = sub_234 = None
    mul_875: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_866, mul_869);  mul_869 = None
    sum_319: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_875, [0, 1, 2]);  mul_875 = None
    sum_320: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_866, [0, 1, 2]);  view_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_264: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_263, mul_874);  add_263 = mul_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    sum_321: "f32[1, 4, 196, 256]" = torch.ops.aten.sum.dim_IntList(add_264, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    view_867: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.view.default(add_264, [8, 2, 2, 14, 14, 256]);  add_264 = None
    permute_707: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.permute.default(view_867, [0, 1, 3, 2, 4, 5]);  view_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    clone_219: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.clone.default(permute_707, memory_format = torch.contiguous_format);  permute_707 = None
    view_868: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(clone_219, [8, 28, 28, 256]);  clone_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute_708: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(view_868, [0, 3, 1, 2]);  view_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_1: "f32[8, 256, 57, 57]" = torch.ops.aten.max_pool2d_with_indices_backward.default(permute_708, constant_pad_nd, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_17);  permute_708 = constant_pad_nd = getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_3: "f32[8, 256, 56, 56]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_1, [0, -1, 0, -1]);  max_pool2d_with_indices_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_709: "f32[8, 56, 56, 256]" = torch.ops.aten.permute.default(constant_pad_nd_3, [0, 2, 3, 1]);  constant_pad_nd_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_235: "f32[8, 56, 56, 256]" = torch.ops.aten.sub.Tensor(permute_18, getitem_15);  permute_18 = getitem_15 = None
    mul_876: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(sub_235, rsqrt_4);  sub_235 = None
    mul_877: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(permute_709, primals_10);  primals_10 = None
    mul_878: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(mul_877, 256)
    sum_322: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_877, [3], True)
    mul_879: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(mul_877, mul_876);  mul_877 = None
    sum_323: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_879, [3], True);  mul_879 = None
    mul_880: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(mul_876, sum_323);  sum_323 = None
    sub_236: "f32[8, 56, 56, 256]" = torch.ops.aten.sub.Tensor(mul_878, sum_322);  mul_878 = sum_322 = None
    sub_237: "f32[8, 56, 56, 256]" = torch.ops.aten.sub.Tensor(sub_236, mul_880);  sub_236 = mul_880 = None
    div_117: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 256);  rsqrt_4 = None
    mul_881: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(div_117, sub_237);  div_117 = sub_237 = None
    mul_882: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(permute_709, mul_876);  mul_876 = None
    sum_324: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_882, [0, 1, 2]);  mul_882 = None
    sum_325: "f32[256]" = torch.ops.aten.sum.dim_IntList(permute_709, [0, 1, 2]);  permute_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_710: "f32[8, 256, 56, 56]" = torch.ops.aten.permute.default(mul_881, [0, 3, 1, 2]);  mul_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:141, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(permute_710, permute_17, primals_124, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  permute_710 = permute_17 = primals_124 = None
    getitem_181: "f32[8, 128, 56, 56]" = convolution_backward_1[0]
    getitem_182: "f32[256, 128, 3, 3]" = convolution_backward_1[1]
    getitem_183: "f32[256]" = convolution_backward_1[2];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_711: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 3, 1]);  getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    view_869: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.view.default(permute_711, [8, 4, 14, 4, 14, 128]);  permute_711 = None
    permute_712: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.permute.default(view_869, [0, 1, 3, 2, 4, 5]);  view_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    clone_220: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.clone.default(permute_712, memory_format = torch.contiguous_format);  permute_712 = None
    view_870: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(clone_220, [8, 16, 196, 128]);  clone_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_883: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_870, div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_871: "f32[25088, 128]" = torch.ops.aten.view.default(mul_883, [25088, 128]);  mul_883 = None
    permute_713: "f32[128, 512]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    mm_178: "f32[25088, 512]" = torch.ops.aten.mm.default(view_871, permute_713);  permute_713 = None
    permute_714: "f32[128, 25088]" = torch.ops.aten.permute.default(view_871, [1, 0])
    mm_179: "f32[128, 512]" = torch.ops.aten.mm.default(permute_714, view_32);  permute_714 = view_32 = None
    permute_715: "f32[512, 128]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    sum_326: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_871, [0], True);  view_871 = None
    view_872: "f32[128]" = torch.ops.aten.view.default(sum_326, [128]);  sum_326 = None
    permute_716: "f32[128, 512]" = torch.ops.aten.permute.default(permute_715, [1, 0]);  permute_715 = None
    view_873: "f32[8, 16, 196, 512]" = torch.ops.aten.view.default(mm_178, [8, 16, 196, 512]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_884: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476)
    erf_46: "f32[8, 16, 196, 512]" = torch.ops.aten.erf.default(mul_884);  mul_884 = None
    add_265: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
    mul_885: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(add_265, 0.5);  add_265 = None
    mul_886: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_31, view_31)
    mul_887: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(mul_886, -0.5);  mul_886 = None
    exp_46: "f32[8, 16, 196, 512]" = torch.ops.aten.exp.default(mul_887);  mul_887 = None
    mul_888: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
    mul_889: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_31, mul_888);  view_31 = mul_888 = None
    add_266: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(mul_885, mul_889);  mul_885 = mul_889 = None
    mul_890: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_873, add_266);  view_873 = add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_874: "f32[25088, 512]" = torch.ops.aten.view.default(mul_890, [25088, 512]);  mul_890 = None
    permute_717: "f32[512, 128]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_180: "f32[25088, 128]" = torch.ops.aten.mm.default(view_874, permute_717);  permute_717 = None
    permute_718: "f32[512, 25088]" = torch.ops.aten.permute.default(view_874, [1, 0])
    mm_181: "f32[512, 128]" = torch.ops.aten.mm.default(permute_718, view_30);  permute_718 = view_30 = None
    permute_719: "f32[128, 512]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_327: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_874, [0], True);  view_874 = None
    view_875: "f32[512]" = torch.ops.aten.view.default(sum_327, [512]);  sum_327 = None
    permute_720: "f32[512, 128]" = torch.ops.aten.permute.default(permute_719, [1, 0]);  permute_719 = None
    view_876: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(mm_180, [8, 16, 196, 128]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_238: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_10, getitem_13);  add_10 = getitem_13 = None
    mul_891: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_238, rsqrt_3);  sub_238 = None
    mul_892: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_876, primals_8);  primals_8 = None
    mul_893: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_892, 128)
    sum_328: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_892, [3], True)
    mul_894: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_892, mul_891);  mul_892 = None
    sum_329: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_894, [3], True);  mul_894 = None
    mul_895: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_891, sum_329);  sum_329 = None
    sub_239: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(mul_893, sum_328);  mul_893 = sum_328 = None
    sub_240: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(sub_239, mul_895);  sub_239 = mul_895 = None
    div_118: "f32[8, 16, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 128);  rsqrt_3 = None
    mul_896: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(div_118, sub_240);  div_118 = sub_240 = None
    mul_897: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_876, mul_891);  mul_891 = None
    sum_330: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 1, 2]);  mul_897 = None
    sum_331: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_876, [0, 1, 2]);  view_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_267: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(view_870, mul_896);  view_870 = mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_898: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(add_267, div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_877: "f32[25088, 128]" = torch.ops.aten.view.default(mul_898, [25088, 128]);  mul_898 = None
    permute_721: "f32[128, 128]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_182: "f32[25088, 128]" = torch.ops.aten.mm.default(view_877, permute_721);  permute_721 = None
    permute_722: "f32[128, 25088]" = torch.ops.aten.permute.default(view_877, [1, 0])
    mm_183: "f32[128, 128]" = torch.ops.aten.mm.default(permute_722, view_28);  permute_722 = view_28 = None
    permute_723: "f32[128, 128]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_332: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
    view_878: "f32[128]" = torch.ops.aten.view.default(sum_332, [128]);  sum_332 = None
    permute_724: "f32[128, 128]" = torch.ops.aten.permute.default(permute_723, [1, 0]);  permute_723 = None
    view_879: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(mm_182, [8, 16, 196, 128]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_880: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.view.default(view_879, [8, 16, 196, 32, 4]);  view_879 = None
    permute_725: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(view_880, [0, 4, 1, 2, 3]);  view_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_221: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(permute_725, memory_format = torch.contiguous_format);  permute_725 = None
    view_881: "f32[512, 196, 32]" = torch.ops.aten.view.default(clone_221, [512, 196, 32]);  clone_221 = None
    permute_726: "f32[512, 196, 196]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
    bmm_136: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(permute_726, view_881);  permute_726 = None
    permute_727: "f32[512, 32, 196]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    bmm_137: "f32[512, 196, 196]" = torch.ops.aten.bmm.default(view_881, permute_727);  view_881 = permute_727 = None
    view_882: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.view.default(bmm_136, [8, 4, 16, 196, 32]);  bmm_136 = None
    view_883: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.view.default(bmm_137, [8, 4, 16, 196, 196]);  bmm_137 = None
    alias_46: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_899: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_883, alias_46);  view_883 = None
    sum_333: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_899, [-1], True)
    mul_900: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_46, sum_333);  alias_46 = sum_333 = None
    sub_241: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_899, mul_900);  mul_899 = mul_900 = None
    view_884: "f32[512, 196, 196]" = torch.ops.aten.view.default(sub_241, [512, 196, 196]);  sub_241 = None
    permute_728: "f32[512, 32, 196]" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
    bmm_138: "f32[512, 32, 196]" = torch.ops.aten.bmm.default(permute_728, view_884);  permute_728 = None
    permute_729: "f32[512, 196, 32]" = torch.ops.aten.permute.default(view_22, [0, 2, 1]);  view_22 = None
    bmm_139: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(view_884, permute_729);  view_884 = permute_729 = None
    view_885: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.view.default(bmm_138, [8, 4, 16, 32, 196]);  bmm_138 = None
    view_886: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.view.default(bmm_139, [8, 4, 16, 196, 32]);  bmm_139 = None
    mul_901: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.mul.Scalar(view_885, 0.42044820762685725);  view_885 = None
    permute_730: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(mul_901, [0, 1, 2, 4, 3]);  mul_901 = None
    mul_902: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.mul.Scalar(view_886, 0.42044820762685725);  view_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_22: "f32[24, 4, 16, 196, 32]" = torch.ops.aten.cat.default([mul_902, permute_730, view_882]);  mul_902 = permute_730 = view_882 = None
    view_887: "f32[3, 8, 4, 16, 196, 32]" = torch.ops.aten.view.default(cat_22, [3, 8, 4, 16, 196, 32]);  cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_731: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.permute.default(view_887, [1, 3, 4, 0, 2, 5]);  view_887 = None
    clone_222: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.clone.default(permute_731, memory_format = torch.contiguous_format);  permute_731 = None
    view_888: "f32[8, 16, 196, 384]" = torch.ops.aten.view.default(clone_222, [8, 16, 196, 384]);  clone_222 = None
    view_889: "f32[25088, 384]" = torch.ops.aten.view.default(view_888, [25088, 384]);  view_888 = None
    permute_732: "f32[384, 128]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_184: "f32[25088, 128]" = torch.ops.aten.mm.default(view_889, permute_732);  permute_732 = None
    permute_733: "f32[384, 25088]" = torch.ops.aten.permute.default(view_889, [1, 0])
    mm_185: "f32[384, 128]" = torch.ops.aten.mm.default(permute_733, view_18);  permute_733 = view_18 = None
    permute_734: "f32[128, 384]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    sum_334: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_889, [0], True);  view_889 = None
    view_890: "f32[384]" = torch.ops.aten.view.default(sum_334, [384]);  sum_334 = None
    permute_735: "f32[384, 128]" = torch.ops.aten.permute.default(permute_734, [1, 0]);  permute_734 = None
    view_891: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(mm_184, [8, 16, 196, 128]);  mm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_242: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_7, getitem_8);  add_7 = getitem_8 = None
    mul_903: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_242, rsqrt_2);  sub_242 = None
    mul_904: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_891, primals_6);  primals_6 = None
    mul_905: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_904, 128)
    sum_335: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_904, [3], True)
    mul_906: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_904, mul_903);  mul_904 = None
    sum_336: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_906, [3], True);  mul_906 = None
    mul_907: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_903, sum_336);  sum_336 = None
    sub_243: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(mul_905, sum_335);  mul_905 = sum_335 = None
    sub_244: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(sub_243, mul_907);  sub_243 = mul_907 = None
    div_119: "f32[8, 16, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
    mul_908: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(div_119, sub_244);  div_119 = sub_244 = None
    mul_909: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_891, mul_903);  mul_903 = None
    sum_337: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_909, [0, 1, 2]);  mul_909 = None
    sum_338: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_891, [0, 1, 2]);  view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_268: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_267, mul_908);  add_267 = mul_908 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_892: "f32[25088, 128]" = torch.ops.aten.view.default(add_268, [25088, 128])
    permute_736: "f32[128, 512]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_186: "f32[25088, 512]" = torch.ops.aten.mm.default(view_892, permute_736);  permute_736 = None
    permute_737: "f32[128, 25088]" = torch.ops.aten.permute.default(view_892, [1, 0])
    mm_187: "f32[128, 512]" = torch.ops.aten.mm.default(permute_737, view_16);  permute_737 = view_16 = None
    permute_738: "f32[512, 128]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_339: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_892, [0], True);  view_892 = None
    view_893: "f32[128]" = torch.ops.aten.view.default(sum_339, [128]);  sum_339 = None
    permute_739: "f32[128, 512]" = torch.ops.aten.permute.default(permute_738, [1, 0]);  permute_738 = None
    view_894: "f32[8, 16, 196, 512]" = torch.ops.aten.view.default(mm_186, [8, 16, 196, 512]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_910: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476)
    erf_47: "f32[8, 16, 196, 512]" = torch.ops.aten.erf.default(mul_910);  mul_910 = None
    add_269: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
    mul_911: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(add_269, 0.5);  add_269 = None
    mul_912: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_15, view_15)
    mul_913: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(mul_912, -0.5);  mul_912 = None
    exp_47: "f32[8, 16, 196, 512]" = torch.ops.aten.exp.default(mul_913);  mul_913 = None
    mul_914: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
    mul_915: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_15, mul_914);  view_15 = mul_914 = None
    add_270: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(mul_911, mul_915);  mul_911 = mul_915 = None
    mul_916: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_894, add_270);  view_894 = add_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_895: "f32[25088, 512]" = torch.ops.aten.view.default(mul_916, [25088, 512]);  mul_916 = None
    permute_740: "f32[512, 128]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    mm_188: "f32[25088, 128]" = torch.ops.aten.mm.default(view_895, permute_740);  permute_740 = None
    permute_741: "f32[512, 25088]" = torch.ops.aten.permute.default(view_895, [1, 0])
    mm_189: "f32[512, 128]" = torch.ops.aten.mm.default(permute_741, view_14);  permute_741 = view_14 = None
    permute_742: "f32[128, 512]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_340: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_895, [0], True);  view_895 = None
    view_896: "f32[512]" = torch.ops.aten.view.default(sum_340, [512]);  sum_340 = None
    permute_743: "f32[512, 128]" = torch.ops.aten.permute.default(permute_742, [1, 0]);  permute_742 = None
    view_897: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(mm_188, [8, 16, 196, 128]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_245: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_3, getitem_6);  add_3 = getitem_6 = None
    mul_917: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_245, rsqrt_1);  sub_245 = None
    mul_918: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_897, primals_4);  primals_4 = None
    mul_919: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_918, 128)
    sum_341: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_918, [3], True)
    mul_920: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_918, mul_917);  mul_918 = None
    sum_342: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_920, [3], True);  mul_920 = None
    mul_921: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_917, sum_342);  sum_342 = None
    sub_246: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(mul_919, sum_341);  mul_919 = sum_341 = None
    sub_247: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(sub_246, mul_921);  sub_246 = mul_921 = None
    div_120: "f32[8, 16, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
    mul_922: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(div_120, sub_247);  div_120 = sub_247 = None
    mul_923: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_897, mul_917);  mul_917 = None
    sum_343: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_923, [0, 1, 2]);  mul_923 = None
    sum_344: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_897, [0, 1, 2]);  view_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_271: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_268, mul_922);  add_268 = mul_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_898: "f32[25088, 128]" = torch.ops.aten.view.default(add_271, [25088, 128])
    permute_744: "f32[128, 128]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    mm_190: "f32[25088, 128]" = torch.ops.aten.mm.default(view_898, permute_744);  permute_744 = None
    permute_745: "f32[128, 25088]" = torch.ops.aten.permute.default(view_898, [1, 0])
    mm_191: "f32[128, 128]" = torch.ops.aten.mm.default(permute_745, view_12);  permute_745 = view_12 = None
    permute_746: "f32[128, 128]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    sum_345: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_898, [0], True);  view_898 = None
    view_899: "f32[128]" = torch.ops.aten.view.default(sum_345, [128]);  sum_345 = None
    permute_747: "f32[128, 128]" = torch.ops.aten.permute.default(permute_746, [1, 0]);  permute_746 = None
    view_900: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(mm_190, [8, 16, 196, 128]);  mm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_901: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.view.default(view_900, [8, 16, 196, 32, 4]);  view_900 = None
    permute_748: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(view_901, [0, 4, 1, 2, 3]);  view_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_223: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(permute_748, memory_format = torch.contiguous_format);  permute_748 = None
    view_902: "f32[512, 196, 32]" = torch.ops.aten.view.default(clone_223, [512, 196, 32]);  clone_223 = None
    permute_749: "f32[512, 196, 196]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    bmm_140: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(permute_749, view_902);  permute_749 = None
    permute_750: "f32[512, 32, 196]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_141: "f32[512, 196, 196]" = torch.ops.aten.bmm.default(view_902, permute_750);  view_902 = permute_750 = None
    view_903: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.view.default(bmm_140, [8, 4, 16, 196, 32]);  bmm_140 = None
    view_904: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.view.default(bmm_141, [8, 4, 16, 196, 196]);  bmm_141 = None
    alias_47: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_924: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_904, alias_47);  view_904 = None
    sum_346: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_924, [-1], True)
    mul_925: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_47, sum_346);  alias_47 = sum_346 = None
    sub_248: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_924, mul_925);  mul_924 = mul_925 = None
    view_905: "f32[512, 196, 196]" = torch.ops.aten.view.default(sub_248, [512, 196, 196]);  sub_248 = None
    permute_751: "f32[512, 32, 196]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    bmm_142: "f32[512, 32, 196]" = torch.ops.aten.bmm.default(permute_751, view_905);  permute_751 = None
    permute_752: "f32[512, 196, 32]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
    bmm_143: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(view_905, permute_752);  view_905 = permute_752 = None
    view_906: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.view.default(bmm_142, [8, 4, 16, 32, 196]);  bmm_142 = None
    view_907: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.view.default(bmm_143, [8, 4, 16, 196, 32]);  bmm_143 = None
    mul_926: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.mul.Scalar(view_906, 0.42044820762685725);  view_906 = None
    permute_753: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(mul_926, [0, 1, 2, 4, 3]);  mul_926 = None
    mul_927: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.mul.Scalar(view_907, 0.42044820762685725);  view_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_23: "f32[24, 4, 16, 196, 32]" = torch.ops.aten.cat.default([mul_927, permute_753, view_903]);  mul_927 = permute_753 = view_903 = None
    view_908: "f32[3, 8, 4, 16, 196, 32]" = torch.ops.aten.view.default(cat_23, [3, 8, 4, 16, 196, 32]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_754: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.permute.default(view_908, [1, 3, 4, 0, 2, 5]);  view_908 = None
    clone_224: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
    view_909: "f32[8, 16, 196, 384]" = torch.ops.aten.view.default(clone_224, [8, 16, 196, 384]);  clone_224 = None
    view_910: "f32[25088, 384]" = torch.ops.aten.view.default(view_909, [25088, 384]);  view_909 = None
    permute_755: "f32[384, 128]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_192: "f32[25088, 128]" = torch.ops.aten.mm.default(view_910, permute_755);  permute_755 = None
    permute_756: "f32[384, 25088]" = torch.ops.aten.permute.default(view_910, [1, 0])
    mm_193: "f32[384, 128]" = torch.ops.aten.mm.default(permute_756, view_2);  permute_756 = view_2 = None
    permute_757: "f32[128, 384]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_347: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_910, [0], True);  view_910 = None
    view_911: "f32[384]" = torch.ops.aten.view.default(sum_347, [384]);  sum_347 = None
    permute_758: "f32[384, 128]" = torch.ops.aten.permute.default(permute_757, [1, 0]);  permute_757 = None
    view_912: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(mm_192, [8, 16, 196, 128]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_249: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
    mul_928: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_249, rsqrt);  sub_249 = None
    mul_929: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_912, primals_2);  primals_2 = None
    mul_930: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_929, 128)
    sum_348: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_929, [3], True)
    mul_931: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_929, mul_928);  mul_929 = None
    sum_349: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_931, [3], True);  mul_931 = None
    mul_932: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_928, sum_349);  sum_349 = None
    sub_250: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(mul_930, sum_348);  mul_930 = sum_348 = None
    sub_251: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(sub_250, mul_932);  sub_250 = mul_932 = None
    div_121: "f32[8, 16, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
    mul_933: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(div_121, sub_251);  div_121 = sub_251 = None
    mul_934: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_912, mul_928);  mul_928 = None
    sum_350: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_934, [0, 1, 2]);  mul_934 = None
    sum_351: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_912, [0, 1, 2]);  view_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_272: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_271, mul_933);  add_271 = mul_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    sum_352: "f32[1, 16, 196, 128]" = torch.ops.aten.sum.dim_IntList(add_272, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    view_913: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.view.default(add_272, [8, 4, 4, 14, 14, 128]);  add_272 = None
    permute_759: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.permute.default(view_913, [0, 1, 3, 2, 4, 5]);  view_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    clone_225: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.clone.default(permute_759, memory_format = torch.contiguous_format);  permute_759 = None
    view_914: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(clone_225, [8, 56, 56, 128]);  clone_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute_760: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(view_914, [0, 3, 1, 2]);  view_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(permute_760, primals_306, primals_106, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  permute_760 = primals_306 = primals_106 = None
    getitem_185: "f32[128, 3, 4, 4]" = convolution_backward_2[1]
    getitem_186: "f32[128]" = convolution_backward_2[2];  convolution_backward_2 = None
    return pytree.tree_unflatten([addmm_96, sum_352, sum_350, sum_351, sum_343, sum_344, sum_337, sum_338, sum_330, sum_331, sum_324, sum_325, sum_321, sum_319, sum_320, sum_312, sum_313, sum_306, sum_307, sum_299, sum_300, sum_293, sum_294, sum_290, sum_288, sum_289, sum_281, sum_282, sum_275, sum_276, sum_268, sum_269, sum_262, sum_263, sum_255, sum_256, sum_249, sum_250, sum_242, sum_243, sum_236, sum_237, sum_229, sum_230, sum_223, sum_224, sum_216, sum_217, sum_210, sum_211, sum_203, sum_204, sum_197, sum_198, sum_190, sum_191, sum_184, sum_185, sum_177, sum_178, sum_171, sum_172, sum_164, sum_165, sum_158, sum_159, sum_151, sum_152, sum_145, sum_146, sum_138, sum_139, sum_132, sum_133, sum_125, sum_126, sum_119, sum_120, sum_112, sum_113, sum_106, sum_107, sum_99, sum_100, sum_93, sum_94, sum_86, sum_87, sum_80, sum_81, sum_73, sum_74, sum_67, sum_68, sum_60, sum_61, sum_54, sum_55, sum_47, sum_48, sum_41, sum_42, sum_34, sum_35, sum_28, sum_29, getitem_185, getitem_186, permute_758, view_911, permute_747, view_899, permute_743, view_896, permute_739, view_893, permute_735, view_890, permute_724, view_878, permute_720, view_875, permute_716, view_872, getitem_182, getitem_183, permute_706, view_865, permute_695, view_853, permute_691, view_850, permute_687, view_847, permute_683, view_844, permute_672, view_832, permute_668, view_829, permute_664, view_826, getitem_179, getitem_180, permute_654, view_819, permute_643, view_807, permute_639, view_804, permute_635, view_801, permute_631, view_798, permute_620, view_786, permute_616, view_783, permute_612, view_780, permute_608, view_777, permute_597, view_765, permute_593, view_762, permute_589, view_759, permute_585, view_756, permute_574, view_744, permute_570, view_741, permute_566, view_738, permute_562, view_735, permute_551, view_723, permute_547, view_720, permute_543, view_717, permute_539, view_714, permute_528, view_702, permute_524, view_699, permute_520, view_696, permute_516, view_693, permute_505, view_681, permute_501, view_678, permute_497, view_675, permute_493, view_672, permute_482, view_660, permute_478, view_657, permute_474, view_654, permute_470, view_651, permute_459, view_639, permute_455, view_636, permute_451, view_633, permute_447, view_630, permute_436, view_618, permute_432, view_615, permute_428, view_612, permute_424, view_609, permute_413, view_597, permute_409, view_594, permute_405, view_591, permute_401, view_588, permute_390, view_576, permute_386, view_573, permute_382, view_570, permute_378, view_567, permute_367, view_555, permute_363, view_552, permute_359, view_549, permute_355, view_546, permute_344, view_534, permute_340, view_531, permute_336, view_528, permute_332, view_525, permute_321, view_513, permute_317, view_510, permute_313, view_507, permute_309, view_504, permute_298, view_492, permute_294, view_489, permute_290, view_486, permute_286, view_483, permute_275, view_471, permute_271, view_468, permute_267, view_465, permute_263, view_462, permute_252, view_450, permute_248, view_447, permute_244, view_444, permute_240, view_441, permute_229, view_429, permute_225, view_426, permute_221, view_423, permute_217, view_420, permute_206, view_408, permute_202, view_405, permute_198, view_402, permute_190, view_397, None], self._out_spec)
    