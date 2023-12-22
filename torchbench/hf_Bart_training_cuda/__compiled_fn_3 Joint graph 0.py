from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[1026, 768]"; primals_2: "f32[1026, 768]"; primals_3: "f32[50265, 768]"; primals_4: "f32[768]"; primals_5: "f32[768]"; primals_6: "f32[768, 768]"; primals_7: "f32[768]"; primals_8: "f32[768, 768]"; primals_9: "f32[768]"; primals_10: "f32[768, 768]"; primals_11: "f32[768]"; primals_12: "f32[768, 768]"; primals_13: "f32[768]"; primals_14: "f32[768]"; primals_15: "f32[768]"; primals_16: "f32[3072, 768]"; primals_17: "f32[3072]"; primals_18: "f32[768, 3072]"; primals_19: "f32[768]"; primals_20: "f32[768]"; primals_21: "f32[768]"; primals_22: "f32[768, 768]"; primals_23: "f32[768]"; primals_24: "f32[768, 768]"; primals_25: "f32[768]"; primals_26: "f32[768, 768]"; primals_27: "f32[768]"; primals_28: "f32[768, 768]"; primals_29: "f32[768]"; primals_30: "f32[768]"; primals_31: "f32[768]"; primals_32: "f32[3072, 768]"; primals_33: "f32[3072]"; primals_34: "f32[768, 3072]"; primals_35: "f32[768]"; primals_36: "f32[768]"; primals_37: "f32[768]"; primals_38: "f32[768, 768]"; primals_39: "f32[768]"; primals_40: "f32[768, 768]"; primals_41: "f32[768]"; primals_42: "f32[768, 768]"; primals_43: "f32[768]"; primals_44: "f32[768, 768]"; primals_45: "f32[768]"; primals_46: "f32[768]"; primals_47: "f32[768]"; primals_48: "f32[3072, 768]"; primals_49: "f32[3072]"; primals_50: "f32[768, 3072]"; primals_51: "f32[768]"; primals_52: "f32[768]"; primals_53: "f32[768]"; primals_54: "f32[768, 768]"; primals_55: "f32[768]"; primals_56: "f32[768, 768]"; primals_57: "f32[768]"; primals_58: "f32[768, 768]"; primals_59: "f32[768]"; primals_60: "f32[768, 768]"; primals_61: "f32[768]"; primals_62: "f32[768]"; primals_63: "f32[768]"; primals_64: "f32[3072, 768]"; primals_65: "f32[3072]"; primals_66: "f32[768, 3072]"; primals_67: "f32[768]"; primals_68: "f32[768]"; primals_69: "f32[768]"; primals_70: "f32[768, 768]"; primals_71: "f32[768]"; primals_72: "f32[768, 768]"; primals_73: "f32[768]"; primals_74: "f32[768, 768]"; primals_75: "f32[768]"; primals_76: "f32[768, 768]"; primals_77: "f32[768]"; primals_78: "f32[768]"; primals_79: "f32[768]"; primals_80: "f32[3072, 768]"; primals_81: "f32[3072]"; primals_82: "f32[768, 3072]"; primals_83: "f32[768]"; primals_84: "f32[768]"; primals_85: "f32[768]"; primals_86: "f32[768, 768]"; primals_87: "f32[768]"; primals_88: "f32[768, 768]"; primals_89: "f32[768]"; primals_90: "f32[768, 768]"; primals_91: "f32[768]"; primals_92: "f32[768, 768]"; primals_93: "f32[768]"; primals_94: "f32[768]"; primals_95: "f32[768]"; primals_96: "f32[3072, 768]"; primals_97: "f32[3072]"; primals_98: "f32[768, 3072]"; primals_99: "f32[768]"; primals_100: "f32[768]"; primals_101: "f32[768]"; primals_102: "f32[50265, 768]"; primals_103: "f32[768]"; primals_104: "f32[768]"; primals_105: "f32[768, 768]"; primals_106: "f32[768]"; primals_107: "f32[768, 768]"; primals_108: "f32[768]"; primals_109: "f32[768, 768]"; primals_110: "f32[768]"; primals_111: "f32[768, 768]"; primals_112: "f32[768]"; primals_113: "f32[768]"; primals_114: "f32[768]"; primals_115: "f32[768, 768]"; primals_116: "f32[768]"; primals_117: "f32[768, 768]"; primals_118: "f32[768]"; primals_119: "f32[768, 768]"; primals_120: "f32[768]"; primals_121: "f32[768, 768]"; primals_122: "f32[768]"; primals_123: "f32[768]"; primals_124: "f32[768]"; primals_125: "f32[3072, 768]"; primals_126: "f32[3072]"; primals_127: "f32[768, 3072]"; primals_128: "f32[768]"; primals_129: "f32[768]"; primals_130: "f32[768]"; primals_131: "f32[768, 768]"; primals_132: "f32[768]"; primals_133: "f32[768, 768]"; primals_134: "f32[768]"; primals_135: "f32[768, 768]"; primals_136: "f32[768]"; primals_137: "f32[768, 768]"; primals_138: "f32[768]"; primals_139: "f32[768]"; primals_140: "f32[768]"; primals_141: "f32[768, 768]"; primals_142: "f32[768]"; primals_143: "f32[768, 768]"; primals_144: "f32[768]"; primals_145: "f32[768, 768]"; primals_146: "f32[768]"; primals_147: "f32[768, 768]"; primals_148: "f32[768]"; primals_149: "f32[768]"; primals_150: "f32[768]"; primals_151: "f32[3072, 768]"; primals_152: "f32[3072]"; primals_153: "f32[768, 3072]"; primals_154: "f32[768]"; primals_155: "f32[768]"; primals_156: "f32[768]"; primals_157: "f32[768, 768]"; primals_158: "f32[768]"; primals_159: "f32[768, 768]"; primals_160: "f32[768]"; primals_161: "f32[768, 768]"; primals_162: "f32[768]"; primals_163: "f32[768, 768]"; primals_164: "f32[768]"; primals_165: "f32[768]"; primals_166: "f32[768]"; primals_167: "f32[768, 768]"; primals_168: "f32[768]"; primals_169: "f32[768, 768]"; primals_170: "f32[768]"; primals_171: "f32[768, 768]"; primals_172: "f32[768]"; primals_173: "f32[768, 768]"; primals_174: "f32[768]"; primals_175: "f32[768]"; primals_176: "f32[768]"; primals_177: "f32[3072, 768]"; primals_178: "f32[3072]"; primals_179: "f32[768, 3072]"; primals_180: "f32[768]"; primals_181: "f32[768]"; primals_182: "f32[768]"; primals_183: "f32[768, 768]"; primals_184: "f32[768]"; primals_185: "f32[768, 768]"; primals_186: "f32[768]"; primals_187: "f32[768, 768]"; primals_188: "f32[768]"; primals_189: "f32[768, 768]"; primals_190: "f32[768]"; primals_191: "f32[768]"; primals_192: "f32[768]"; primals_193: "f32[768, 768]"; primals_194: "f32[768]"; primals_195: "f32[768, 768]"; primals_196: "f32[768]"; primals_197: "f32[768, 768]"; primals_198: "f32[768]"; primals_199: "f32[768, 768]"; primals_200: "f32[768]"; primals_201: "f32[768]"; primals_202: "f32[768]"; primals_203: "f32[3072, 768]"; primals_204: "f32[3072]"; primals_205: "f32[768, 3072]"; primals_206: "f32[768]"; primals_207: "f32[768]"; primals_208: "f32[768]"; primals_209: "f32[768, 768]"; primals_210: "f32[768]"; primals_211: "f32[768, 768]"; primals_212: "f32[768]"; primals_213: "f32[768, 768]"; primals_214: "f32[768]"; primals_215: "f32[768, 768]"; primals_216: "f32[768]"; primals_217: "f32[768]"; primals_218: "f32[768]"; primals_219: "f32[768, 768]"; primals_220: "f32[768]"; primals_221: "f32[768, 768]"; primals_222: "f32[768]"; primals_223: "f32[768, 768]"; primals_224: "f32[768]"; primals_225: "f32[768, 768]"; primals_226: "f32[768]"; primals_227: "f32[768]"; primals_228: "f32[768]"; primals_229: "f32[3072, 768]"; primals_230: "f32[3072]"; primals_231: "f32[768, 3072]"; primals_232: "f32[768]"; primals_233: "f32[768]"; primals_234: "f32[768]"; primals_235: "f32[768, 768]"; primals_236: "f32[768]"; primals_237: "f32[768, 768]"; primals_238: "f32[768]"; primals_239: "f32[768, 768]"; primals_240: "f32[768]"; primals_241: "f32[768, 768]"; primals_242: "f32[768]"; primals_243: "f32[768]"; primals_244: "f32[768]"; primals_245: "f32[768, 768]"; primals_246: "f32[768]"; primals_247: "f32[768, 768]"; primals_248: "f32[768]"; primals_249: "f32[768, 768]"; primals_250: "f32[768]"; primals_251: "f32[768, 768]"; primals_252: "f32[768]"; primals_253: "f32[768]"; primals_254: "f32[768]"; primals_255: "f32[3072, 768]"; primals_256: "f32[3072]"; primals_257: "f32[768, 3072]"; primals_258: "f32[768]"; primals_259: "f32[768]"; primals_260: "f32[768]"; primals_261: "f32[50265, 768]"; primals_262: "f32[1, 50265]"; primals_263: "i64[4, 512]"; primals_264: "i64[4, 512]"; tangents_1: "f32[4, 512, 50265]"; tangents_2: "f32[4, 12, 512, 64]"; tangents_3: "f32[4, 12, 512, 64]"; tangents_4: "f32[4, 12, 512, 64]"; tangents_5: "f32[4, 12, 512, 64]"; tangents_6: "f32[4, 12, 512, 64]"; tangents_7: "f32[4, 12, 512, 64]"; tangents_8: "f32[4, 12, 512, 64]"; tangents_9: "f32[4, 12, 512, 64]"; tangents_10: "f32[4, 12, 512, 64]"; tangents_11: "f32[4, 12, 512, 64]"; tangents_12: "f32[4, 12, 512, 64]"; tangents_13: "f32[4, 12, 512, 64]"; tangents_14: "f32[4, 12, 512, 64]"; tangents_15: "f32[4, 12, 512, 64]"; tangents_16: "f32[4, 12, 512, 64]"; tangents_17: "f32[4, 12, 512, 64]"; tangents_18: "f32[4, 12, 512, 64]"; tangents_19: "f32[4, 12, 512, 64]"; tangents_20: "f32[4, 12, 512, 64]"; tangents_21: "f32[4, 12, 512, 64]"; tangents_22: "f32[4, 12, 512, 64]"; tangents_23: "f32[4, 12, 512, 64]"; tangents_24: "f32[4, 12, 512, 64]"; tangents_25: "f32[4, 12, 512, 64]"; tangents_26: "f32[4, 512, 768]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:811, code: input_ids = input_ids.view(-1, input_ids.shape[-1])
    view: "i64[4, 512]" = torch.ops.aten.view.default(primals_263, [-1, 512]);  primals_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:818, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(primals_3, view, 1);  primals_3 = None
    mul: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:135, code: positions = torch.arange(
    iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:137, code: ).expand(bsz, -1)
    expand: "i64[4, 512]" = torch.ops.aten.expand.default(iota, [4, -1]);  iota = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:139, code: return super().forward(positions + self.offset)
    add: "i64[4, 512]" = torch.ops.aten.add.Tensor(expand, 2);  expand = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_1: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(primals_1, add);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:823, code: hidden_states = inputs_embeds + embed_pos
    add_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:824, code: hidden_states = self.layernorm_embedding(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[4, 512, 1]" = var_mean[0]
    getitem_1: "f32[4, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1)
    mul_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_2: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, primals_4);  mul_1 = None
    add_3: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:825, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone: "f32[4, 512, 768]" = torch.ops.aten.clone.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_1: "f32[2048, 768]" = torch.ops.aten.view.default(clone, [2048, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    addmm: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_7, view_1, permute);  primals_7 = None
    view_2: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm, [4, 512, 768]);  addmm = None
    mul_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_2, 0.125);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_3: "f32[2048, 768]" = torch.ops.aten.view.default(clone, [2048, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm_1: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_3, permute_1);  primals_9 = None
    view_4: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_1, [4, 512, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_5: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_4, [4, -1, 12, 64]);  view_4 = None
    permute_2: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    clone_1: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_6: "f32[2048, 768]" = torch.ops.aten.view.default(clone, [2048, 768])
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_2: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_6, permute_3);  primals_11 = None
    view_7: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_2, [4, 512, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_7, [4, -1, 12, 64]);  view_7 = None
    permute_4: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_3, [4, 512, 12, 64]);  mul_3 = None
    permute_5: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    clone_3: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_10: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_3, [48, -1, 64]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_11: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_1, [48, -1, 64]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_12: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_2, [48, -1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_10, permute_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm, [-1], True)
    sub_1: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm, amax);  bmm = amax = None
    exp: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_4: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_4, view_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_13: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_1, [4, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_5: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_14: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_5, [4, 512, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_15: "f32[2048, 768]" = torch.ops.aten.view.default(view_14, [2048, 768]);  view_14 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_3: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_15, permute_8);  primals_13 = None
    view_16: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_3, [4, 512, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:338, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_6: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_16);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_4: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(clone, clone_6);  clone = clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem_2: "f32[4, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[4, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_5: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_2: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_3)
    mul_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_5: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, primals_14);  mul_4 = None
    add_6: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, primals_15);  mul_5 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_17: "f32[2048, 768]" = torch.ops.aten.view.default(add_6, [2048, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    addmm_4: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_17, view_17, permute_9);  primals_17 = None
    view_18: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_4, [4, 512, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_7: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_7: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_7);  mul_6 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:344, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_7: "f32[4, 512, 3072]" = torch.ops.aten.clone.default(mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_19: "f32[2048, 3072]" = torch.ops.aten.view.default(clone_7, [2048, 3072]);  clone_7 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm_5: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_19, view_19, permute_10);  primals_19 = None
    view_20: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_5, [4, 512, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:346, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_8: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_8: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_6, clone_8);  add_6 = clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_4: "f32[4, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[4, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_3: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_5)
    mul_9: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_10: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_20);  mul_9 = None
    add_10: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_21);  mul_10 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_21: "f32[2048, 768]" = torch.ops.aten.view.default(add_10, [2048, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    addmm_6: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_23, view_21, permute_11);  primals_23 = None
    view_22: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_6, [4, 512, 768]);  addmm_6 = None
    mul_11: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_22, 0.125);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_23: "f32[2048, 768]" = torch.ops.aten.view.default(add_10, [2048, 768])
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    addmm_7: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_25, view_23, permute_12);  primals_25 = None
    view_24: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_7, [4, 512, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_24, [4, -1, 12, 64]);  view_24 = None
    permute_13: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    clone_9: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_26: "f32[2048, 768]" = torch.ops.aten.view.default(add_10, [2048, 768])
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    addmm_8: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_27, view_26, permute_14);  primals_27 = None
    view_27: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_8, [4, 512, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_28: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_27, [4, -1, 12, 64]);  view_27 = None
    permute_15: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_10: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_29: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_11, [4, 512, 12, 64]);  mul_11 = None
    permute_16: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_11: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_30: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_11, [48, -1, 64]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_31: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_9, [48, -1, 64]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_32: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_10, [48, -1, 64]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_17: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
    bmm_2: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_30, permute_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_2, [-1], True)
    sub_4: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_2, amax_1);  bmm_2 = amax_1 = None
    exp_1: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_12: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_3: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_12, view_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_33: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_3, [4, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_13: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_34: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_13, [4, 512, 768]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_35: "f32[2048, 768]" = torch.ops.aten.view.default(view_34, [2048, 768]);  view_34 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_9: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_29, view_35, permute_19);  primals_29 = None
    view_36: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_9, [4, 512, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:338, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_14: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_11: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_10, clone_14);  add_10 = clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_6: "f32[4, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[4, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_7)
    mul_12: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_13: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, primals_30);  mul_12 = None
    add_13: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_13, primals_31);  mul_13 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_37: "f32[2048, 768]" = torch.ops.aten.view.default(add_13, [2048, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    addmm_10: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_33, view_37, permute_20);  primals_33 = None
    view_38: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_10, [4, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_15: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_1: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_14: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_14, add_14);  mul_14 = add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:344, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_15: "f32[4, 512, 3072]" = torch.ops.aten.clone.default(mul_16);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_39: "f32[2048, 3072]" = torch.ops.aten.view.default(clone_15, [2048, 3072]);  clone_15 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_11: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_35, view_39, permute_21);  primals_35 = None
    view_40: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_11, [4, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:346, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_16: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_15: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_13, clone_16);  add_13 = clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_8: "f32[4, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[4, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_16: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_6: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_9)
    mul_17: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_18: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, primals_36);  mul_17 = None
    add_17: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, primals_37);  mul_18 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_41: "f32[2048, 768]" = torch.ops.aten.view.default(add_17, [2048, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_12: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_39, view_41, permute_22);  primals_39 = None
    view_42: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_12, [4, 512, 768]);  addmm_12 = None
    mul_19: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_42, 0.125);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_43: "f32[2048, 768]" = torch.ops.aten.view.default(add_17, [2048, 768])
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_13: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_41, view_43, permute_23);  primals_41 = None
    view_44: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_13, [4, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_45: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_44, [4, -1, 12, 64]);  view_44 = None
    permute_24: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
    clone_17: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_46: "f32[2048, 768]" = torch.ops.aten.view.default(add_17, [2048, 768])
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    addmm_14: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_43, view_46, permute_25);  primals_43 = None
    view_47: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_14, [4, 512, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_48: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_47, [4, -1, 12, 64]);  view_47 = None
    permute_26: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    clone_18: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_49: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_19, [4, 512, 12, 64]);  mul_19 = None
    permute_27: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    clone_19: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_50: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_19, [48, -1, 64]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_51: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_17, [48, -1, 64]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_52: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_18, [48, -1, 64]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_28: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
    bmm_4: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_50, permute_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_4, [-1], True)
    sub_7: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_4, amax_2);  bmm_4 = amax_2 = None
    exp_2: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_20: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_5: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_20, view_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_53: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_5, [4, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_21: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_54: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_21, [4, 512, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_55: "f32[2048, 768]" = torch.ops.aten.view.default(view_54, [2048, 768]);  view_54 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    addmm_15: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_45, view_55, permute_30);  primals_45 = None
    view_56: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_15, [4, 512, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:338, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_22: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_56);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_18: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_17, clone_22);  add_17 = clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
    getitem_10: "f32[4, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[4, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_19: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_8: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_18, getitem_11)
    mul_20: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_21: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, primals_46);  mul_20 = None
    add_20: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_21, primals_47);  mul_21 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_57: "f32[2048, 768]" = torch.ops.aten.view.default(add_20, [2048, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    addmm_16: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_49, view_57, permute_31);  primals_49 = None
    view_58: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_16, [4, 512, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_23: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
    erf_2: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_21: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_24: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_22, add_21);  mul_22 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:344, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_23: "f32[4, 512, 3072]" = torch.ops.aten.clone.default(mul_24);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_59: "f32[2048, 3072]" = torch.ops.aten.view.default(clone_23, [2048, 3072]);  clone_23 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_17: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_51, view_59, permute_32);  primals_51 = None
    view_60: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_17, [4, 512, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:346, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_24: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_22: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_20, clone_24);  add_20 = clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
    getitem_12: "f32[4, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[4, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_23: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_9: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_13)
    mul_25: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_26: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_52);  mul_25 = None
    add_24: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_53);  mul_26 = primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_61: "f32[2048, 768]" = torch.ops.aten.view.default(add_24, [2048, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    addmm_18: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_55, view_61, permute_33);  primals_55 = None
    view_62: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_18, [4, 512, 768]);  addmm_18 = None
    mul_27: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_62, 0.125);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_63: "f32[2048, 768]" = torch.ops.aten.view.default(add_24, [2048, 768])
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_19: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_57, view_63, permute_34);  primals_57 = None
    view_64: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_19, [4, 512, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_65: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_64, [4, -1, 12, 64]);  view_64 = None
    permute_35: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
    clone_25: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_66: "f32[2048, 768]" = torch.ops.aten.view.default(add_24, [2048, 768])
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_20: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_59, view_66, permute_36);  primals_59 = None
    view_67: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_20, [4, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_68: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_67, [4, -1, 12, 64]);  view_67 = None
    permute_37: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    clone_26: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_69: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_27, [4, 512, 12, 64]);  mul_27 = None
    permute_38: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
    clone_27: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_70: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_27, [48, -1, 64]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_71: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_25, [48, -1, 64]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_72: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_26, [48, -1, 64]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_39: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
    bmm_6: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_70, permute_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_6, [-1], True)
    sub_10: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_6, amax_3);  bmm_6 = amax_3 = None
    exp_3: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_28: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_7: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_28, view_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_73: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_7, [4, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_29: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_74: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_29, [4, 512, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_75: "f32[2048, 768]" = torch.ops.aten.view.default(view_74, [2048, 768]);  view_74 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_21: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_61, view_75, permute_41);  primals_61 = None
    view_76: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_21, [4, 512, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:338, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_30: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_76);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_25: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_24, clone_30);  add_24 = clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_14: "f32[4, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[4, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_26: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_11: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_15)
    mul_28: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_29: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_62);  mul_28 = None
    add_27: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_29, primals_63);  mul_29 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_77: "f32[2048, 768]" = torch.ops.aten.view.default(add_27, [2048, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_22: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_65, view_77, permute_42);  primals_65 = None
    view_78: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_22, [4, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_30: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_31: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476)
    erf_3: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_28: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_32: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_30, add_28);  mul_30 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:344, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_31: "f32[4, 512, 3072]" = torch.ops.aten.clone.default(mul_32);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_79: "f32[2048, 3072]" = torch.ops.aten.view.default(clone_31, [2048, 3072]);  clone_31 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_23: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_67, view_79, permute_43);  primals_67 = None
    view_80: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_23, [4, 512, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:346, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_32: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_29: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_27, clone_32);  add_27 = clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_16: "f32[4, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[4, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_30: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_12: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_29, getitem_17)
    mul_33: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_34: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, primals_68);  mul_33 = None
    add_31: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_69);  mul_34 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_81: "f32[2048, 768]" = torch.ops.aten.view.default(add_31, [2048, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    addmm_24: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_71, view_81, permute_44);  primals_71 = None
    view_82: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_24, [4, 512, 768]);  addmm_24 = None
    mul_35: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_82, 0.125);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_83: "f32[2048, 768]" = torch.ops.aten.view.default(add_31, [2048, 768])
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    addmm_25: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_73, view_83, permute_45);  primals_73 = None
    view_84: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_25, [4, 512, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_85: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_84, [4, -1, 12, 64]);  view_84 = None
    permute_46: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
    clone_33: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_86: "f32[2048, 768]" = torch.ops.aten.view.default(add_31, [2048, 768])
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_26: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_75, view_86, permute_47);  primals_75 = None
    view_87: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_26, [4, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_88: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_87, [4, -1, 12, 64]);  view_87 = None
    permute_48: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
    clone_34: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_89: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_35, [4, 512, 12, 64]);  mul_35 = None
    permute_49: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
    clone_35: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_90: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_35, [48, -1, 64]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_91: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_33, [48, -1, 64]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_92: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_34, [48, -1, 64]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_50: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
    bmm_8: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_90, permute_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_8, [-1], True)
    sub_13: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_8, amax_4);  bmm_8 = amax_4 = None
    exp_4: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_36: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_9: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_36, view_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_93: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_9, [4, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_37: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_94: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_37, [4, 512, 768]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_95: "f32[2048, 768]" = torch.ops.aten.view.default(view_94, [2048, 768]);  view_94 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    addmm_27: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_77, view_95, permute_52);  primals_77 = None
    view_96: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_27, [4, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:338, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_38: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_32: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_31, clone_38);  add_31 = clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_18: "f32[4, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[4, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_33: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_14: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_19)
    mul_36: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_37: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, primals_78);  mul_36 = None
    add_34: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, primals_79);  mul_37 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_97: "f32[2048, 768]" = torch.ops.aten.view.default(add_34, [2048, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_28: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_81, view_97, permute_53);  primals_81 = None
    view_98: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_28, [4, 512, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_38: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_39: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
    erf_4: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_35: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_40: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_38, add_35);  mul_38 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:344, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_39: "f32[4, 512, 3072]" = torch.ops.aten.clone.default(mul_40);  mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_99: "f32[2048, 3072]" = torch.ops.aten.view.default(clone_39, [2048, 3072]);  clone_39 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_29: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_83, view_99, permute_54);  primals_83 = None
    view_100: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_29, [4, 512, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:346, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_40: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_36: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_34, clone_40);  add_34 = clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
    getitem_20: "f32[4, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[4, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_37: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_15: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_21)
    mul_41: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_42: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, primals_84);  mul_41 = None
    add_38: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, primals_85);  mul_42 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_101: "f32[2048, 768]" = torch.ops.aten.view.default(add_38, [2048, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_30: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_87, view_101, permute_55);  primals_87 = None
    view_102: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_30, [4, 512, 768]);  addmm_30 = None
    mul_43: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_102, 0.125);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_103: "f32[2048, 768]" = torch.ops.aten.view.default(add_38, [2048, 768])
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_31: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_89, view_103, permute_56);  primals_89 = None
    view_104: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_31, [4, 512, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_105: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_104, [4, -1, 12, 64]);  view_104 = None
    permute_57: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
    clone_41: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_106: "f32[2048, 768]" = torch.ops.aten.view.default(add_38, [2048, 768])
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_32: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_91, view_106, permute_58);  primals_91 = None
    view_107: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_32, [4, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_108: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_107, [4, -1, 12, 64]);  view_107 = None
    permute_59: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
    clone_42: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_109: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_43, [4, 512, 12, 64]);  mul_43 = None
    permute_60: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
    clone_43: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_110: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_43, [48, -1, 64]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_111: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_41, [48, -1, 64]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_112: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_42, [48, -1, 64]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_61: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
    bmm_10: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_110, permute_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_10, [-1], True)
    sub_16: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_10, amax_5);  bmm_10 = amax_5 = None
    exp_5: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_44: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_11: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_44, view_112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_113: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_11, [4, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_45: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_114: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_45, [4, 512, 768]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_115: "f32[2048, 768]" = torch.ops.aten.view.default(view_114, [2048, 768]);  view_114 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_33: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_93, view_115, permute_63);  primals_93 = None
    view_116: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_33, [4, 512, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:338, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_46: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_116);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_39: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_38, clone_46);  add_38 = clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_22: "f32[4, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[4, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_40: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_17: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_23)
    mul_44: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_45: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_94);  mul_44 = None
    add_41: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_95);  mul_45 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_117: "f32[2048, 768]" = torch.ops.aten.view.default(add_41, [2048, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    addmm_34: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_97, view_117, permute_64);  primals_97 = None
    view_118: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_34, [4, 512, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_46: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_47: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
    erf_5: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_42: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_48: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_42);  mul_46 = add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:344, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_47: "f32[4, 512, 3072]" = torch.ops.aten.clone.default(mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_119: "f32[2048, 3072]" = torch.ops.aten.view.default(clone_47, [2048, 3072]);  clone_47 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    addmm_35: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_99, view_119, permute_65);  primals_99 = None
    view_120: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_35, [4, 512, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:346, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_48: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_43: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_41, clone_48);  add_41 = clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_24: "f32[4, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[4, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_44: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_18: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_25)
    mul_49: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_50: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_100);  mul_49 = None
    add_45: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_101);  mul_50 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1059, code: inputs_embeds = self.embed_tokens(input) * self.embed_scale
    embedding_2: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(primals_102, primals_264, 1);  primals_102 = None
    mul_51: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(embedding_2, 1.0);  embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:96, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:97, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:98, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_46: "i64[512]" = torch.ops.aten.add.Tensor(iota_1, 1)
    view_122: "i64[512, 1]" = torch.ops.aten.view.default(add_46, [512, 1]);  add_46 = None
    lt: "b8[512, 512]" = torch.ops.aten.lt.Tensor(iota_1, view_122);  iota_1 = view_122 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[512, 512]" = torch.ops.aten.where.self(lt, scalar_tensor, full);  lt = scalar_tensor = full = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:135, code: positions = torch.arange(
    iota_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:137, code: ).expand(bsz, -1)
    expand_2: "i64[4, 512]" = torch.ops.aten.expand.default(iota_2, [4, -1]);  iota_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:139, code: return super().forward(positions + self.offset)
    add_47: "i64[4, 512]" = torch.ops.aten.add.Tensor(expand_2, 2);  expand_2 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_3: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(primals_2, add_47);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1074, code: hidden_states = inputs_embeds + positions
    add_48: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, embedding_3);  mul_51 = embedding_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1075, code: hidden_states = self.layernorm_embedding(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_26: "f32[4, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[4, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_49: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_19: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_27)
    mul_52: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = None
    mul_53: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, primals_103);  mul_52 = None
    add_50: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, primals_104);  mul_53 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1077, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_49: "f32[4, 512, 768]" = torch.ops.aten.clone.default(add_50);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_123: "f32[2048, 768]" = torch.ops.aten.view.default(clone_49, [2048, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_36: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_106, view_123, permute_66);  primals_106 = None
    view_124: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_36, [4, 512, 768]);  addmm_36 = None
    mul_54: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_124, 0.125);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_125: "f32[2048, 768]" = torch.ops.aten.view.default(clone_49, [2048, 768])
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_37: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_108, view_125, permute_67);  primals_108 = None
    view_126: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_37, [4, 512, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_127: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_126, [4, -1, 12, 64]);  view_126 = None
    permute_68: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
    clone_50: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_128: "f32[2048, 768]" = torch.ops.aten.view.default(clone_49, [2048, 768])
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    addmm_38: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_110, view_128, permute_69);  primals_110 = None
    view_129: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_38, [4, 512, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_130: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_129, [4, -1, 12, 64]);  view_129 = None
    permute_70: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    clone_51: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_131: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_54, [4, 512, 12, 64]);  mul_54 = None
    permute_71: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
    clone_52: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_132: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_52, [48, -1, 64]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_133: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_50, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_134: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_51, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_72: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    bmm_12: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_132, permute_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_135: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_12, [4, 12, 512, 512]);  bmm_12 = None
    unsqueeze_2: "f32[1, 512, 512]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_3: "f32[1, 1, 512, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
    slice_3: "f32[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(unsqueeze_3, 2, 0, 9223372036854775807);  unsqueeze_3 = None
    slice_4: "f32[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 9223372036854775807);  slice_3 = None
    expand_3: "f32[4, 1, 512, 512]" = torch.ops.aten.expand.default(slice_4, [4, 1, 512, 512]);  slice_4 = None
    add_51: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_135, expand_3);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_136: "f32[48, 512, 512]" = torch.ops.aten.view.default(add_51, [48, 512, 512]);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[48, 512, 1]" = torch.ops.aten.amax.default(view_136, [-1], True)
    sub_20: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(view_136, amax_6);  view_136 = amax_6 = None
    exp_6: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_7: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_53: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_13: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_53, view_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_137: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_13, [4, 12, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_54: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_138: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_54, [4, 512, 768]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_139: "f32[2048, 768]" = torch.ops.aten.view.default(view_138, [2048, 768]);  view_138 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_39: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_112, view_139, permute_74);  primals_112 = None
    view_140: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_39, [4, 512, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_55: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_52: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(clone_49, clone_55);  clone_49 = clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_28: "f32[4, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[4, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_53: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_21: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_29)
    mul_55: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
    mul_56: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_55, primals_113);  mul_55 = None
    add_54: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_56, primals_114);  mul_56 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_141: "f32[2048, 768]" = torch.ops.aten.view.default(add_54, [2048, 768])
    permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_40: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_116, view_141, permute_75);  primals_116 = None
    view_142: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_40, [4, 512, 768]);  addmm_40 = None
    mul_57: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_142, 0.125);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_143: "f32[2048, 768]" = torch.ops.aten.view.default(add_45, [2048, 768])
    permute_76: "f32[768, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    addmm_41: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_118, view_143, permute_76);  primals_118 = None
    view_144: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_41, [4, 512, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_145: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_144, [4, -1, 12, 64]);  view_144 = None
    permute_77: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
    clone_56: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_146: "f32[2048, 768]" = torch.ops.aten.view.default(add_45, [2048, 768])
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_42: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_120, view_146, permute_78);  primals_120 = None
    view_147: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_42, [4, 512, 768]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_148: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_147, [4, -1, 12, 64]);  view_147 = None
    permute_79: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_57: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_149: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_57, [4, 512, 12, 64]);  mul_57 = None
    permute_80: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    clone_58: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_150: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_58, [48, -1, 64]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_151: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_56, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_152: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_57, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_81: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
    bmm_14: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_150, permute_81)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_14, [-1], True)
    sub_22: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_14, amax_7);  bmm_14 = amax_7 = None
    exp_7: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_59: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_15: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_59, view_152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_153: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_15, [4, 12, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_82: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_60: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_154: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_60, [4, 512, 768]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_155: "f32[2048, 768]" = torch.ops.aten.view.default(view_154, [2048, 768]);  view_154 = None
    permute_83: "f32[768, 768]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_43: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_122, view_155, permute_83);  primals_122 = None
    view_156: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_43, [4, 512, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_61: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_156);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_55: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_54, clone_61);  add_54 = clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_30: "f32[4, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[4, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_56: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_23: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_31)
    mul_58: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
    mul_59: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_123);  mul_58 = None
    add_57: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_124);  mul_59 = primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_157: "f32[2048, 768]" = torch.ops.aten.view.default(add_57, [2048, 768])
    permute_84: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_44: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_126, view_157, permute_84);  primals_126 = None
    view_158: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_44, [4, 512, 3072]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_60: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
    mul_61: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476)
    erf_6: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_58: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_62: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_58);  mul_60 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_62: "f32[4, 512, 3072]" = torch.ops.aten.clone.default(mul_62);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_159: "f32[2048, 3072]" = torch.ops.aten.view.default(clone_62, [2048, 3072]);  clone_62 = None
    permute_85: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    addmm_45: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_128, view_159, permute_85);  primals_128 = None
    view_160: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_45, [4, 512, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_63: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_160);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_59: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_57, clone_63);  add_57 = clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_32: "f32[4, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[4, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_60: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_24: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_33)
    mul_63: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
    mul_64: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_63, primals_129);  mul_63 = None
    add_61: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_64, primals_130);  mul_64 = primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_161: "f32[2048, 768]" = torch.ops.aten.view.default(add_61, [2048, 768])
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_46: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_132, view_161, permute_86);  primals_132 = None
    view_162: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_46, [4, 512, 768]);  addmm_46 = None
    mul_65: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_162, 0.125);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_163: "f32[2048, 768]" = torch.ops.aten.view.default(add_61, [2048, 768])
    permute_87: "f32[768, 768]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    addmm_47: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_134, view_163, permute_87);  primals_134 = None
    view_164: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_47, [4, 512, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_165: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_164, [4, -1, 12, 64]);  view_164 = None
    permute_88: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
    clone_64: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_166: "f32[2048, 768]" = torch.ops.aten.view.default(add_61, [2048, 768])
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_48: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_136, view_166, permute_89);  primals_136 = None
    view_167: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_48, [4, 512, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_168: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_167, [4, -1, 12, 64]);  view_167 = None
    permute_90: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    clone_65: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_169: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_65, [4, 512, 12, 64]);  mul_65 = None
    permute_91: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
    clone_66: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_170: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_66, [48, -1, 64]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_171: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_64, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_172: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_65, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_92: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    bmm_16: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_170, permute_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_173: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_16, [4, 12, 512, 512]);  bmm_16 = None
    add_62: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_173, expand_3);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_174: "f32[48, 512, 512]" = torch.ops.aten.view.default(add_62, [48, 512, 512]);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[48, 512, 1]" = torch.ops.aten.amax.default(view_174, [-1], True)
    sub_25: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(view_174, amax_8);  view_174 = amax_8 = None
    exp_8: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_67: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_17: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_67, view_172)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_175: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_17, [4, 12, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_93: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_68: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    view_176: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_68, [4, 512, 768]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_177: "f32[2048, 768]" = torch.ops.aten.view.default(view_176, [2048, 768]);  view_176 = None
    permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_49: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_138, view_177, permute_94);  primals_138 = None
    view_178: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_49, [4, 512, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_69: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_178);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_63: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_61, clone_69);  add_61 = clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_34: "f32[4, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[4, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_64: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_26: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_35)
    mul_66: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
    mul_67: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_139);  mul_66 = None
    add_65: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_140);  mul_67 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_179: "f32[2048, 768]" = torch.ops.aten.view.default(add_65, [2048, 768])
    permute_95: "f32[768, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_50: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_142, view_179, permute_95);  primals_142 = None
    view_180: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_50, [4, 512, 768]);  addmm_50 = None
    mul_68: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_180, 0.125);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_181: "f32[2048, 768]" = torch.ops.aten.view.default(add_45, [2048, 768])
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_51: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_144, view_181, permute_96);  primals_144 = None
    view_182: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_51, [4, 512, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_183: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_182, [4, -1, 12, 64]);  view_182 = None
    permute_97: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    clone_70: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_184: "f32[2048, 768]" = torch.ops.aten.view.default(add_45, [2048, 768])
    permute_98: "f32[768, 768]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_52: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_146, view_184, permute_98);  primals_146 = None
    view_185: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_52, [4, 512, 768]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_186: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_185, [4, -1, 12, 64]);  view_185 = None
    permute_99: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    clone_71: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_187: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_68, [4, 512, 12, 64]);  mul_68 = None
    permute_100: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
    clone_72: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_188: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_72, [48, -1, 64]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_189: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_70, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_190: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_71, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_101: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    bmm_18: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_188, permute_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_18, [-1], True)
    sub_27: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_18, amax_9);  bmm_18 = amax_9 = None
    exp_9: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_10: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_73: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_19: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_73, view_190)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_191: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_19, [4, 12, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_102: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_74: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    view_192: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_74, [4, 512, 768]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_193: "f32[2048, 768]" = torch.ops.aten.view.default(view_192, [2048, 768]);  view_192 = None
    permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm_53: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_148, view_193, permute_103);  primals_148 = None
    view_194: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_53, [4, 512, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_75: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_194);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_66: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_65, clone_75);  add_65 = clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_36: "f32[4, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[4, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_67: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_28: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_66, getitem_37)
    mul_69: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
    mul_70: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_69, primals_149);  mul_69 = None
    add_68: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_70, primals_150);  mul_70 = primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_195: "f32[2048, 768]" = torch.ops.aten.view.default(add_68, [2048, 768])
    permute_104: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_54: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_152, view_195, permute_104);  primals_152 = None
    view_196: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_54, [4, 512, 3072]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_71: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.5)
    mul_72: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.7071067811865476)
    erf_7: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_69: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_73: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_71, add_69);  mul_71 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_76: "f32[4, 512, 3072]" = torch.ops.aten.clone.default(mul_73);  mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_197: "f32[2048, 3072]" = torch.ops.aten.view.default(clone_76, [2048, 3072]);  clone_76 = None
    permute_105: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    addmm_55: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_154, view_197, permute_105);  primals_154 = None
    view_198: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_55, [4, 512, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_77: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_198);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_70: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_68, clone_77);  add_68 = clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_38: "f32[4, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[4, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_71: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_29: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_39)
    mul_74: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
    mul_75: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, primals_155);  mul_74 = None
    add_72: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_156);  mul_75 = primals_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_199: "f32[2048, 768]" = torch.ops.aten.view.default(add_72, [2048, 768])
    permute_106: "f32[768, 768]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm_56: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_158, view_199, permute_106);  primals_158 = None
    view_200: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_56, [4, 512, 768]);  addmm_56 = None
    mul_76: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_200, 0.125);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_201: "f32[2048, 768]" = torch.ops.aten.view.default(add_72, [2048, 768])
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    addmm_57: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_160, view_201, permute_107);  primals_160 = None
    view_202: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_57, [4, 512, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_203: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_202, [4, -1, 12, 64]);  view_202 = None
    permute_108: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    clone_78: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_204: "f32[2048, 768]" = torch.ops.aten.view.default(add_72, [2048, 768])
    permute_109: "f32[768, 768]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    addmm_58: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_162, view_204, permute_109);  primals_162 = None
    view_205: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_58, [4, 512, 768]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_206: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_205, [4, -1, 12, 64]);  view_205 = None
    permute_110: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    clone_79: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_207: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_76, [4, 512, 12, 64]);  mul_76 = None
    permute_111: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    clone_80: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_208: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_80, [48, -1, 64]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_209: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_78, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_210: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_79, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_112: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    bmm_20: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_208, permute_112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_211: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_20, [4, 12, 512, 512]);  bmm_20 = None
    add_73: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_211, expand_3);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_212: "f32[48, 512, 512]" = torch.ops.aten.view.default(add_73, [48, 512, 512]);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[48, 512, 1]" = torch.ops.aten.amax.default(view_212, [-1], True)
    sub_30: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(view_212, amax_10);  view_212 = amax_10 = None
    exp_10: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_11: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_81: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_21: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_81, view_210)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_213: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_21, [4, 12, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_113: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_82: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    view_214: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_82, [4, 512, 768]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_215: "f32[2048, 768]" = torch.ops.aten.view.default(view_214, [2048, 768]);  view_214 = None
    permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    addmm_59: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_164, view_215, permute_114);  primals_164 = None
    view_216: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_59, [4, 512, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_83: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_216);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_74: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_72, clone_83);  add_72 = clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
    getitem_40: "f32[4, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[4, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_75: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    sub_31: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_74, getitem_41)
    mul_77: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
    mul_78: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_77, primals_165);  mul_77 = None
    add_76: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_78, primals_166);  mul_78 = primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_217: "f32[2048, 768]" = torch.ops.aten.view.default(add_76, [2048, 768])
    permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
    addmm_60: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_168, view_217, permute_115);  primals_168 = None
    view_218: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_60, [4, 512, 768]);  addmm_60 = None
    mul_79: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_218, 0.125);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_219: "f32[2048, 768]" = torch.ops.aten.view.default(add_45, [2048, 768])
    permute_116: "f32[768, 768]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    addmm_61: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_170, view_219, permute_116);  primals_170 = None
    view_220: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_61, [4, 512, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_221: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_220, [4, -1, 12, 64]);  view_220 = None
    permute_117: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None
    clone_84: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_222: "f32[2048, 768]" = torch.ops.aten.view.default(add_45, [2048, 768])
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm_62: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_172, view_222, permute_118);  primals_172 = None
    view_223: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_62, [4, 512, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_224: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_223, [4, -1, 12, 64]);  view_223 = None
    permute_119: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    clone_85: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_225: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_79, [4, 512, 12, 64]);  mul_79 = None
    permute_120: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
    clone_86: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_226: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_86, [48, -1, 64]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_227: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_84, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_228: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_85, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_121: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_227, [0, 2, 1]);  view_227 = None
    bmm_22: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_226, permute_121)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_22, [-1], True)
    sub_32: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_22, amax_11);  bmm_22 = amax_11 = None
    exp_11: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_12: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_87: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_23: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_87, view_228)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_229: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, [4, 12, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_122: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_88: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_230: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_88, [4, 512, 768]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_231: "f32[2048, 768]" = torch.ops.aten.view.default(view_230, [2048, 768]);  view_230 = None
    permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
    addmm_63: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_174, view_231, permute_123);  primals_174 = None
    view_232: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_63, [4, 512, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_89: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_232);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_77: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_76, clone_89);  add_76 = clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_42: "f32[4, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[4, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_78: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_33: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_43)
    mul_80: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
    mul_81: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_175);  mul_80 = None
    add_79: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_176);  mul_81 = primals_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_233: "f32[2048, 768]" = torch.ops.aten.view.default(add_79, [2048, 768])
    permute_124: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    addmm_64: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_178, view_233, permute_124);  primals_178 = None
    view_234: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_64, [4, 512, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_234, 0.5)
    mul_83: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_234, 0.7071067811865476)
    erf_8: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_80: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_84: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_82, add_80);  mul_82 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_90: "f32[4, 512, 3072]" = torch.ops.aten.clone.default(mul_84);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_235: "f32[2048, 3072]" = torch.ops.aten.view.default(clone_90, [2048, 3072]);  clone_90 = None
    permute_125: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    addmm_65: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_180, view_235, permute_125);  primals_180 = None
    view_236: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_65, [4, 512, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_91: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_236);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_81: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_79, clone_91);  add_79 = clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_44: "f32[4, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[4, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_82: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_34: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_45)
    mul_85: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
    mul_86: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, primals_181);  mul_85 = None
    add_83: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, primals_182);  mul_86 = primals_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_237: "f32[2048, 768]" = torch.ops.aten.view.default(add_83, [2048, 768])
    permute_126: "f32[768, 768]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    addmm_66: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_184, view_237, permute_126);  primals_184 = None
    view_238: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_66, [4, 512, 768]);  addmm_66 = None
    mul_87: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_238, 0.125);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_239: "f32[2048, 768]" = torch.ops.aten.view.default(add_83, [2048, 768])
    permute_127: "f32[768, 768]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    addmm_67: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_186, view_239, permute_127);  primals_186 = None
    view_240: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_67, [4, 512, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_241: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_240, [4, -1, 12, 64]);  view_240 = None
    permute_128: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
    clone_92: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_242: "f32[2048, 768]" = torch.ops.aten.view.default(add_83, [2048, 768])
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    addmm_68: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_188, view_242, permute_129);  primals_188 = None
    view_243: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_68, [4, 512, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_244: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_243, [4, -1, 12, 64]);  view_243 = None
    permute_130: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
    clone_93: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_245: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_87, [4, 512, 12, 64]);  mul_87 = None
    permute_131: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
    clone_94: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_131, memory_format = torch.contiguous_format);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_246: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_94, [48, -1, 64]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_247: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_92, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_248: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_93, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_132: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_247, [0, 2, 1]);  view_247 = None
    bmm_24: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_246, permute_132)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_249: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_24, [4, 12, 512, 512]);  bmm_24 = None
    add_84: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_249, expand_3);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_250: "f32[48, 512, 512]" = torch.ops.aten.view.default(add_84, [48, 512, 512]);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[48, 512, 1]" = torch.ops.aten.amax.default(view_250, [-1], True)
    sub_35: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(view_250, amax_12);  view_250 = amax_12 = None
    exp_12: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_13: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_12: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_95: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_25: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_95, view_248)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_251: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_25, [4, 12, 512, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_133: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_96: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    view_252: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_96, [4, 512, 768]);  clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_253: "f32[2048, 768]" = torch.ops.aten.view.default(view_252, [2048, 768]);  view_252 = None
    permute_134: "f32[768, 768]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    addmm_69: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_190, view_253, permute_134);  primals_190 = None
    view_254: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_69, [4, 512, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_97: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_254);  view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_85: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_83, clone_97);  add_83 = clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_46: "f32[4, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[4, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_86: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_36: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_47)
    mul_88: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
    mul_89: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_88, primals_191);  mul_88 = None
    add_87: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_89, primals_192);  mul_89 = primals_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_255: "f32[2048, 768]" = torch.ops.aten.view.default(add_87, [2048, 768])
    permute_135: "f32[768, 768]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
    addmm_70: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_194, view_255, permute_135);  primals_194 = None
    view_256: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_70, [4, 512, 768]);  addmm_70 = None
    mul_90: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_256, 0.125);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_257: "f32[2048, 768]" = torch.ops.aten.view.default(add_45, [2048, 768])
    permute_136: "f32[768, 768]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    addmm_71: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_196, view_257, permute_136);  primals_196 = None
    view_258: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_71, [4, 512, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_259: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_258, [4, -1, 12, 64]);  view_258 = None
    permute_137: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    clone_98: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_260: "f32[2048, 768]" = torch.ops.aten.view.default(add_45, [2048, 768])
    permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
    addmm_72: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_198, view_260, permute_138);  primals_198 = None
    view_261: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_72, [4, 512, 768]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_262: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_261, [4, -1, 12, 64]);  view_261 = None
    permute_139: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
    clone_99: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_263: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_90, [4, 512, 12, 64]);  mul_90 = None
    permute_140: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
    clone_100: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_264: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_100, [48, -1, 64]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_265: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_98, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_266: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_99, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_141: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_265, [0, 2, 1]);  view_265 = None
    bmm_26: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_264, permute_141)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_13: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_26, [-1], True)
    sub_37: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_26, amax_13);  bmm_26 = amax_13 = None
    exp_13: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_14: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_13: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_101: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_27: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_101, view_266)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_267: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_27, [4, 12, 512, 64]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_142: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_102: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    view_268: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_102, [4, 512, 768]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_269: "f32[2048, 768]" = torch.ops.aten.view.default(view_268, [2048, 768]);  view_268 = None
    permute_143: "f32[768, 768]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    addmm_73: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_200, view_269, permute_143);  primals_200 = None
    view_270: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_73, [4, 512, 768]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_103: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_270);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_88: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_87, clone_103);  add_87 = clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_48: "f32[4, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[4, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_89: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_38: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_49)
    mul_91: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_24);  sub_38 = None
    mul_92: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_91, primals_201);  mul_91 = None
    add_90: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_92, primals_202);  mul_92 = primals_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_271: "f32[2048, 768]" = torch.ops.aten.view.default(add_90, [2048, 768])
    permute_144: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
    addmm_74: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_204, view_271, permute_144);  primals_204 = None
    view_272: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_74, [4, 512, 3072]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_93: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_272, 0.5)
    mul_94: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_272, 0.7071067811865476)
    erf_9: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
    add_91: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_95: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_93, add_91);  mul_93 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_104: "f32[4, 512, 3072]" = torch.ops.aten.clone.default(mul_95);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_273: "f32[2048, 3072]" = torch.ops.aten.view.default(clone_104, [2048, 3072]);  clone_104 = None
    permute_145: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    addmm_75: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_206, view_273, permute_145);  primals_206 = None
    view_274: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_75, [4, 512, 768]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_105: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_274);  view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_92: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_90, clone_105);  add_90 = clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_50: "f32[4, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[4, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_93: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_39: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_51)
    mul_96: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_25);  sub_39 = None
    mul_97: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, primals_207);  mul_96 = None
    add_94: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_97, primals_208);  mul_97 = primals_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_275: "f32[2048, 768]" = torch.ops.aten.view.default(add_94, [2048, 768])
    permute_146: "f32[768, 768]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    addmm_76: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_210, view_275, permute_146);  primals_210 = None
    view_276: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_76, [4, 512, 768]);  addmm_76 = None
    mul_98: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, 0.125);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_277: "f32[2048, 768]" = torch.ops.aten.view.default(add_94, [2048, 768])
    permute_147: "f32[768, 768]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
    addmm_77: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_212, view_277, permute_147);  primals_212 = None
    view_278: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_77, [4, 512, 768]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_279: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_278, [4, -1, 12, 64]);  view_278 = None
    permute_148: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
    clone_106: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_280: "f32[2048, 768]" = torch.ops.aten.view.default(add_94, [2048, 768])
    permute_149: "f32[768, 768]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
    addmm_78: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_214, view_280, permute_149);  primals_214 = None
    view_281: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_78, [4, 512, 768]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_282: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_281, [4, -1, 12, 64]);  view_281 = None
    permute_150: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
    clone_107: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_283: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_98, [4, 512, 12, 64]);  mul_98 = None
    permute_151: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
    clone_108: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_151, memory_format = torch.contiguous_format);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_284: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_108, [48, -1, 64]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_285: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_106, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_286: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_107, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_152: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
    bmm_28: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_284, permute_152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_287: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_28, [4, 12, 512, 512]);  bmm_28 = None
    add_95: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_287, expand_3);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_288: "f32[48, 512, 512]" = torch.ops.aten.view.default(add_95, [48, 512, 512]);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[48, 512, 1]" = torch.ops.aten.amax.default(view_288, [-1], True)
    sub_40: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(view_288, amax_14);  view_288 = amax_14 = None
    exp_14: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_15: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_14: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_109: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_29: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_109, view_286)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_289: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_29, [4, 12, 512, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_153: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_110: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
    view_290: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_110, [4, 512, 768]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_291: "f32[2048, 768]" = torch.ops.aten.view.default(view_290, [2048, 768]);  view_290 = None
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    addmm_79: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_216, view_291, permute_154);  primals_216 = None
    view_292: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_79, [4, 512, 768]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_111: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_292);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_96: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_94, clone_111);  add_94 = clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_52: "f32[4, 512, 1]" = var_mean_26[0]
    getitem_53: "f32[4, 512, 1]" = var_mean_26[1];  var_mean_26 = None
    add_97: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_41: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_53)
    mul_99: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_26);  sub_41 = None
    mul_100: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_99, primals_217);  mul_99 = None
    add_98: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_100, primals_218);  mul_100 = primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_293: "f32[2048, 768]" = torch.ops.aten.view.default(add_98, [2048, 768])
    permute_155: "f32[768, 768]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
    addmm_80: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_220, view_293, permute_155);  primals_220 = None
    view_294: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_80, [4, 512, 768]);  addmm_80 = None
    mul_101: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_294, 0.125);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_295: "f32[2048, 768]" = torch.ops.aten.view.default(add_45, [2048, 768])
    permute_156: "f32[768, 768]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
    addmm_81: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_222, view_295, permute_156);  primals_222 = None
    view_296: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_81, [4, 512, 768]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_297: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_296, [4, -1, 12, 64]);  view_296 = None
    permute_157: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None
    clone_112: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_298: "f32[2048, 768]" = torch.ops.aten.view.default(add_45, [2048, 768])
    permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
    addmm_82: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_224, view_298, permute_158);  primals_224 = None
    view_299: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_82, [4, 512, 768]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_300: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_299, [4, -1, 12, 64]);  view_299 = None
    permute_159: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
    clone_113: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_301: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_101, [4, 512, 12, 64]);  mul_101 = None
    permute_160: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None
    clone_114: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_302: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_114, [48, -1, 64]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_303: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_112, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_304: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_113, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_161: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
    bmm_30: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_302, permute_161)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_15: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_30, [-1], True)
    sub_42: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_30, amax_15);  bmm_30 = amax_15 = None
    exp_15: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    sum_16: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_15: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_115: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_31: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_115, view_304)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_305: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_31, [4, 12, 512, 64]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_162: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_116: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    view_306: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_116, [4, 512, 768]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_307: "f32[2048, 768]" = torch.ops.aten.view.default(view_306, [2048, 768]);  view_306 = None
    permute_163: "f32[768, 768]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
    addmm_83: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_226, view_307, permute_163);  primals_226 = None
    view_308: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_83, [4, 512, 768]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_117: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_308);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_99: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_98, clone_117);  add_98 = clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
    getitem_54: "f32[4, 512, 1]" = var_mean_27[0]
    getitem_55: "f32[4, 512, 1]" = var_mean_27[1];  var_mean_27 = None
    add_100: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_43: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_55)
    mul_102: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_27);  sub_43 = None
    mul_103: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, primals_227);  mul_102 = None
    add_101: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_103, primals_228);  mul_103 = primals_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_309: "f32[2048, 768]" = torch.ops.aten.view.default(add_101, [2048, 768])
    permute_164: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
    addmm_84: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_230, view_309, permute_164);  primals_230 = None
    view_310: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_84, [4, 512, 3072]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_104: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_310, 0.5)
    mul_105: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_310, 0.7071067811865476)
    erf_10: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_102: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_106: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_104, add_102);  mul_104 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_118: "f32[4, 512, 3072]" = torch.ops.aten.clone.default(mul_106);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_311: "f32[2048, 3072]" = torch.ops.aten.view.default(clone_118, [2048, 3072]);  clone_118 = None
    permute_165: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
    addmm_85: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_232, view_311, permute_165);  primals_232 = None
    view_312: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_85, [4, 512, 768]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_119: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_312);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_103: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_101, clone_119);  add_101 = clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_56: "f32[4, 512, 1]" = var_mean_28[0]
    getitem_57: "f32[4, 512, 1]" = var_mean_28[1];  var_mean_28 = None
    add_104: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_44: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_57)
    mul_107: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_28);  sub_44 = None
    mul_108: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, primals_233);  mul_107 = None
    add_105: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_108, primals_234);  mul_108 = primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_313: "f32[2048, 768]" = torch.ops.aten.view.default(add_105, [2048, 768])
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    addmm_86: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_236, view_313, permute_166);  primals_236 = None
    view_314: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_86, [4, 512, 768]);  addmm_86 = None
    mul_109: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_314, 0.125);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_315: "f32[2048, 768]" = torch.ops.aten.view.default(add_105, [2048, 768])
    permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
    addmm_87: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_238, view_315, permute_167);  primals_238 = None
    view_316: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_87, [4, 512, 768]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_317: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_316, [4, -1, 12, 64]);  view_316 = None
    permute_168: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
    clone_120: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_318: "f32[2048, 768]" = torch.ops.aten.view.default(add_105, [2048, 768])
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
    addmm_88: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_240, view_318, permute_169);  primals_240 = None
    view_319: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_88, [4, 512, 768]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_320: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_319, [4, -1, 12, 64]);  view_319 = None
    permute_170: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
    clone_121: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_321: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_109, [4, 512, 12, 64]);  mul_109 = None
    permute_171: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
    clone_122: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_322: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_122, [48, -1, 64]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_323: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_120, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_324: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_121, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_172: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
    bmm_32: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_322, permute_172)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_325: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_32, [4, 12, 512, 512]);  bmm_32 = None
    add_106: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_325, expand_3);  view_325 = expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_326: "f32[48, 512, 512]" = torch.ops.aten.view.default(add_106, [48, 512, 512]);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[48, 512, 1]" = torch.ops.aten.amax.default(view_326, [-1], True)
    sub_45: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(view_326, amax_16);  view_326 = amax_16 = None
    exp_16: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_17: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_16: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_123: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_16);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_33: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_123, view_324)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_327: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_33, [4, 12, 512, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_173: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_124: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
    view_328: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_124, [4, 512, 768]);  clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_329: "f32[2048, 768]" = torch.ops.aten.view.default(view_328, [2048, 768]);  view_328 = None
    permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
    addmm_89: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_242, view_329, permute_174);  primals_242 = None
    view_330: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_89, [4, 512, 768]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_125: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_330);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_107: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_105, clone_125);  add_105 = clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_58: "f32[4, 512, 1]" = var_mean_29[0]
    getitem_59: "f32[4, 512, 1]" = var_mean_29[1];  var_mean_29 = None
    add_108: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_46: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_107, getitem_59)
    mul_110: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_29);  sub_46 = None
    mul_111: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_110, primals_243);  mul_110 = None
    add_109: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_111, primals_244);  mul_111 = primals_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_331: "f32[2048, 768]" = torch.ops.aten.view.default(add_109, [2048, 768])
    permute_175: "f32[768, 768]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
    addmm_90: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_246, view_331, permute_175);  primals_246 = None
    view_332: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_90, [4, 512, 768]);  addmm_90 = None
    mul_112: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_332, 0.125);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_333: "f32[2048, 768]" = torch.ops.aten.view.default(add_45, [2048, 768])
    permute_176: "f32[768, 768]" = torch.ops.aten.permute.default(primals_247, [1, 0]);  primals_247 = None
    addmm_91: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_248, view_333, permute_176);  primals_248 = None
    view_334: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_91, [4, 512, 768]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_335: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_334, [4, -1, 12, 64]);  view_334 = None
    permute_177: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None
    clone_126: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_336: "f32[2048, 768]" = torch.ops.aten.view.default(add_45, [2048, 768])
    permute_178: "f32[768, 768]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
    addmm_92: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_250, view_336, permute_178);  primals_250 = None
    view_337: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_92, [4, 512, 768]);  addmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_338: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_337, [4, -1, 12, 64]);  view_337 = None
    permute_179: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    clone_127: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_339: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_112, [4, 512, 12, 64]);  mul_112 = None
    permute_180: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
    clone_128: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_340: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_128, [48, -1, 64]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_341: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_126, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_342: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_127, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_181: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None
    bmm_34: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_340, permute_181)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_17: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_34, [-1], True)
    sub_47: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_34, amax_17);  bmm_34 = amax_17 = None
    exp_17: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
    sum_18: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_17: "f32[48, 512, 512]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_129: "f32[48, 512, 512]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_35: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(clone_129, view_342)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_343: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_35, [4, 12, 512, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_182: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_130: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
    view_344: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_130, [4, 512, 768]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_345: "f32[2048, 768]" = torch.ops.aten.view.default(view_344, [2048, 768]);  view_344 = None
    permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
    addmm_93: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_252, view_345, permute_183);  primals_252 = None
    view_346: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_93, [4, 512, 768]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:454, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_131: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_346);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_110: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_109, clone_131);  add_109 = clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
    getitem_60: "f32[4, 512, 1]" = var_mean_30[0]
    getitem_61: "f32[4, 512, 1]" = var_mean_30[1];  var_mean_30 = None
    add_111: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_48: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_110, getitem_61)
    mul_113: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_30);  sub_48 = None
    mul_114: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_113, primals_253);  mul_113 = None
    add_112: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_114, primals_254);  mul_114 = primals_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_347: "f32[2048, 768]" = torch.ops.aten.view.default(add_112, [2048, 768])
    permute_184: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_94: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_256, view_347, permute_184);  primals_256 = None
    view_348: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_94, [4, 512, 3072]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_115: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_348, 0.5)
    mul_116: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_348, 0.7071067811865476)
    erf_11: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
    add_113: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_117: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_115, add_113);  mul_115 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_132: "f32[4, 512, 3072]" = torch.ops.aten.clone.default(mul_117);  mul_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_349: "f32[2048, 3072]" = torch.ops.aten.view.default(clone_132, [2048, 3072]);  clone_132 = None
    permute_185: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_257, [1, 0]);  primals_257 = None
    addmm_95: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_258, view_349, permute_185);  primals_258 = None
    view_350: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_95, [4, 512, 768]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_133: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_350);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_114: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_112, clone_133);  add_112 = clone_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_62: "f32[4, 512, 1]" = var_mean_31[0]
    getitem_63: "f32[4, 512, 1]" = var_mean_31[1];  var_mean_31 = None
    add_115: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_49: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_114, getitem_63)
    mul_118: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_31);  sub_49 = None
    mul_119: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_118, primals_259);  mul_118 = None
    add_116: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_119, primals_260);  mul_119 = primals_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1406, code: lm_logits = self.lm_head(outputs[0])
    permute_186: "f32[768, 50265]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
    view_351: "f32[2048, 768]" = torch.ops.aten.view.default(add_116, [2048, 768]);  add_116 = None
    mm: "f32[2048, 50265]" = torch.ops.aten.mm.default(view_351, permute_186)
    view_352: "f32[4, 512, 50265]" = torch.ops.aten.view.default(mm, [4, 512, 50265]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1407, code: lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
    add_117: "f32[4, 512, 50265]" = torch.ops.aten.add.Tensor(view_352, primals_262);  view_352 = primals_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1406, code: lm_logits = self.lm_head(outputs[0])
    view_353: "f32[2048, 50265]" = torch.ops.aten.view.default(tangents_1, [2048, 50265]);  tangents_1 = None
    permute_187: "f32[50265, 2048]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_1: "f32[50265, 768]" = torch.ops.aten.mm.default(permute_187, view_351);  permute_187 = view_351 = None
    permute_188: "f32[768, 50265]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    permute_189: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    mm_2: "f32[2048, 768]" = torch.ops.aten.mm.default(view_353, permute_189);  view_353 = permute_189 = None
    view_354: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_2, [4, 512, 768]);  mm_2 = None
    permute_190: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_50: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_114, getitem_63);  add_114 = getitem_63 = None
    mul_120: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_31);  sub_50 = None
    mul_121: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_354, primals_259);  primals_259 = None
    mul_122: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, 768)
    sum_19: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True)
    mul_123: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, mul_120);  mul_121 = None
    sum_20: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True);  mul_123 = None
    mul_124: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_120, sum_20);  sum_20 = None
    sub_51: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_122, sum_19);  mul_122 = sum_19 = None
    sub_52: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_51, mul_124);  sub_51 = mul_124 = None
    div_18: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 768);  rsqrt_31 = None
    mul_125: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_52);  div_18 = sub_52 = None
    mul_126: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_354, mul_120);  mul_120 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_126, [0, 1]);  mul_126 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_354, [0, 1]);  view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_355: "f32[2048, 768]" = torch.ops.aten.view.default(mul_125, [2048, 768])
    permute_191: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    mm_3: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_355, permute_191);  permute_191 = None
    permute_192: "f32[768, 2048]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_4: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_192, view_349);  permute_192 = view_349 = None
    permute_193: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_4, [1, 0]);  mm_4 = None
    sum_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
    view_356: "f32[768]" = torch.ops.aten.view.default(sum_23, [768]);  sum_23 = None
    permute_194: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    view_357: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_3, [4, 512, 3072]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_127: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_348, 0.7071067811865476)
    erf_12: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_127);  mul_127 = None
    add_118: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_128: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_118, 0.5);  add_118 = None
    mul_129: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_348, view_348)
    mul_130: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_129, -0.5);  mul_129 = None
    exp_18: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_130);  mul_130 = None
    mul_131: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_132: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_348, mul_131);  view_348 = mul_131 = None
    add_119: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_128, mul_132);  mul_128 = mul_132 = None
    mul_133: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, add_119);  view_357 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_358: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_133, [2048, 3072]);  mul_133 = None
    permute_195: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    mm_5: "f32[2048, 768]" = torch.ops.aten.mm.default(view_358, permute_195);  permute_195 = None
    permute_196: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_358, [1, 0])
    mm_6: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_196, view_347);  permute_196 = view_347 = None
    permute_197: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_6, [1, 0]);  mm_6 = None
    sum_24: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
    view_359: "f32[3072]" = torch.ops.aten.view.default(sum_24, [3072]);  sum_24 = None
    permute_198: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_360: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_5, [4, 512, 768]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_120: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_125, view_360);  mul_125 = view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    sub_53: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_110, getitem_61);  add_110 = getitem_61 = None
    mul_134: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_30);  sub_53 = None
    mul_135: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, primals_253);  primals_253 = None
    mul_136: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_135, 768)
    sum_25: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [2], True)
    mul_137: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_135, mul_134);  mul_135 = None
    sum_26: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [2], True);  mul_137 = None
    mul_138: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_134, sum_26);  sum_26 = None
    sub_54: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_136, sum_25);  mul_136 = sum_25 = None
    sub_55: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_54, mul_138);  sub_54 = mul_138 = None
    div_19: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 768);  rsqrt_30 = None
    mul_139: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_55);  div_19 = sub_55 = None
    mul_140: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, mul_134);  mul_134 = None
    sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_140, [0, 1]);  mul_140 = None
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_361: "f32[2048, 768]" = torch.ops.aten.view.default(mul_139, [2048, 768])
    permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
    mm_7: "f32[2048, 768]" = torch.ops.aten.mm.default(view_361, permute_199);  permute_199 = None
    permute_200: "f32[768, 2048]" = torch.ops.aten.permute.default(view_361, [1, 0])
    mm_8: "f32[768, 768]" = torch.ops.aten.mm.default(permute_200, view_345);  permute_200 = view_345 = None
    permute_201: "f32[768, 768]" = torch.ops.aten.permute.default(mm_8, [1, 0]);  mm_8 = None
    sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
    view_362: "f32[768]" = torch.ops.aten.view.default(sum_29, [768]);  sum_29 = None
    permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_363: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_7, [4, 512, 768]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_364: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_363, [4, 512, 12, 64]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_203: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_134: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    view_365: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_134, [48, 512, 64]);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_204: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_129, [0, 2, 1]);  clone_129 = None
    bmm_36: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_204, view_365);  permute_204 = None
    permute_205: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_342, [0, 2, 1]);  view_342 = None
    bmm_37: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_365, permute_205);  view_365 = permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_18: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_141: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_37, alias_18);  bmm_37 = None
    sum_30: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_141, [-1], True)
    mul_142: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_18, sum_30);  alias_18 = sum_30 = None
    sub_56: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_206: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_340, [0, 2, 1]);  view_340 = None
    bmm_38: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_206, sub_56);  permute_206 = None
    permute_207: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_181, [0, 2, 1]);  permute_181 = None
    bmm_39: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_56, permute_207);  sub_56 = permute_207 = None
    permute_208: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_38, [0, 2, 1]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_366: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_36, [4, 12, 512, 64]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_121: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_25, view_366);  tangents_25 = view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_367: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_208, [4, 12, 512, 64]);  permute_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_122: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_24, view_367);  tangents_24 = view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_368: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_39, [4, 12, 512, 64]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_209: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    clone_135: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
    view_369: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_135, [4, 512, 768]);  clone_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_210: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_121, [0, 2, 1, 3]);  add_121 = None
    clone_136: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
    view_370: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_136, [4, 512, 768]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_371: "f32[2048, 768]" = torch.ops.aten.view.default(view_370, [2048, 768]);  view_370 = None
    permute_211: "f32[768, 768]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    mm_9: "f32[2048, 768]" = torch.ops.aten.mm.default(view_371, permute_211);  permute_211 = None
    permute_212: "f32[768, 2048]" = torch.ops.aten.permute.default(view_371, [1, 0])
    mm_10: "f32[768, 768]" = torch.ops.aten.mm.default(permute_212, view_336);  permute_212 = view_336 = None
    permute_213: "f32[768, 768]" = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
    sum_31: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_371, [0], True);  view_371 = None
    view_372: "f32[768]" = torch.ops.aten.view.default(sum_31, [768]);  sum_31 = None
    permute_214: "f32[768, 768]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    view_373: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_9, [4, 512, 768]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_123: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(tangents_26, view_373);  tangents_26 = view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_215: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_122, [0, 2, 1, 3]);  add_122 = None
    clone_137: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
    view_374: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_137, [4, 512, 768]);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_375: "f32[2048, 768]" = torch.ops.aten.view.default(view_374, [2048, 768]);  view_374 = None
    permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    mm_11: "f32[2048, 768]" = torch.ops.aten.mm.default(view_375, permute_216);  permute_216 = None
    permute_217: "f32[768, 2048]" = torch.ops.aten.permute.default(view_375, [1, 0])
    mm_12: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_333);  permute_217 = view_333 = None
    permute_218: "f32[768, 768]" = torch.ops.aten.permute.default(mm_12, [1, 0]);  mm_12 = None
    sum_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_375, [0], True);  view_375 = None
    view_376: "f32[768]" = torch.ops.aten.view.default(sum_32, [768]);  sum_32 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    view_377: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_11, [4, 512, 768]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_124: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_123, view_377);  add_123 = view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_143: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_369, 0.125);  view_369 = None
    view_378: "f32[2048, 768]" = torch.ops.aten.view.default(mul_143, [2048, 768]);  mul_143 = None
    permute_220: "f32[768, 768]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    mm_13: "f32[2048, 768]" = torch.ops.aten.mm.default(view_378, permute_220);  permute_220 = None
    permute_221: "f32[768, 2048]" = torch.ops.aten.permute.default(view_378, [1, 0])
    mm_14: "f32[768, 768]" = torch.ops.aten.mm.default(permute_221, view_331);  permute_221 = view_331 = None
    permute_222: "f32[768, 768]" = torch.ops.aten.permute.default(mm_14, [1, 0]);  mm_14 = None
    sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True);  view_378 = None
    view_379: "f32[768]" = torch.ops.aten.view.default(sum_33, [768]);  sum_33 = None
    permute_223: "f32[768, 768]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    view_380: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_13, [4, 512, 768]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_125: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_139, view_380);  mul_139 = view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_57: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_107, getitem_59);  add_107 = getitem_59 = None
    mul_144: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_29);  sub_57 = None
    mul_145: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_243);  primals_243 = None
    mul_146: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_145, 768)
    sum_34: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True)
    mul_147: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_145, mul_144);  mul_145 = None
    sum_35: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True);  mul_147 = None
    mul_148: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_144, sum_35);  sum_35 = None
    sub_58: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_146, sum_34);  mul_146 = sum_34 = None
    sub_59: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_58, mul_148);  sub_58 = mul_148 = None
    div_20: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 768);  rsqrt_29 = None
    mul_149: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_59);  div_20 = sub_59 = None
    mul_150: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, mul_144);  mul_144 = None
    sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_150, [0, 1]);  mul_150 = None
    sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_381: "f32[2048, 768]" = torch.ops.aten.view.default(mul_149, [2048, 768])
    permute_224: "f32[768, 768]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    mm_15: "f32[2048, 768]" = torch.ops.aten.mm.default(view_381, permute_224);  permute_224 = None
    permute_225: "f32[768, 2048]" = torch.ops.aten.permute.default(view_381, [1, 0])
    mm_16: "f32[768, 768]" = torch.ops.aten.mm.default(permute_225, view_329);  permute_225 = view_329 = None
    permute_226: "f32[768, 768]" = torch.ops.aten.permute.default(mm_16, [1, 0]);  mm_16 = None
    sum_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_381, [0], True);  view_381 = None
    view_382: "f32[768]" = torch.ops.aten.view.default(sum_38, [768]);  sum_38 = None
    permute_227: "f32[768, 768]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    view_383: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_15, [4, 512, 768]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_384: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_383, [4, 512, 12, 64]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_228: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_138: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_385: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_138, [48, 512, 64]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_229: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_123, [0, 2, 1]);  clone_123 = None
    bmm_40: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_229, view_385);  permute_229 = None
    permute_230: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None
    bmm_41: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_385, permute_230);  view_385 = permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_19: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    mul_151: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_41, alias_19);  bmm_41 = None
    sum_39: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [-1], True)
    mul_152: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_19, sum_39);  alias_19 = sum_39 = None
    sub_60: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_386: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(sub_60, [4, 12, 512, 512]);  sub_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_387: "f32[48, 512, 512]" = torch.ops.aten.view.default(view_386, [48, 512, 512]);  view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_231: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_322, [0, 2, 1]);  view_322 = None
    bmm_42: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_231, view_387);  permute_231 = None
    permute_232: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_172, [0, 2, 1]);  permute_172 = None
    bmm_43: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_387, permute_232);  view_387 = permute_232 = None
    permute_233: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_42, [0, 2, 1]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_388: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_40, [4, 12, 512, 64]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_126: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_23, view_388);  tangents_23 = view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_389: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_233, [4, 12, 512, 64]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_127: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_22, view_389);  tangents_22 = view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_390: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_43, [4, 12, 512, 64]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_234: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
    clone_139: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
    view_391: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_139, [4, 512, 768]);  clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_235: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_126, [0, 2, 1, 3]);  add_126 = None
    clone_140: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
    view_392: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_140, [4, 512, 768]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_393: "f32[2048, 768]" = torch.ops.aten.view.default(view_392, [2048, 768]);  view_392 = None
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    mm_17: "f32[2048, 768]" = torch.ops.aten.mm.default(view_393, permute_236);  permute_236 = None
    permute_237: "f32[768, 2048]" = torch.ops.aten.permute.default(view_393, [1, 0])
    mm_18: "f32[768, 768]" = torch.ops.aten.mm.default(permute_237, view_318);  permute_237 = view_318 = None
    permute_238: "f32[768, 768]" = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
    sum_40: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_393, [0], True);  view_393 = None
    view_394: "f32[768]" = torch.ops.aten.view.default(sum_40, [768]);  sum_40 = None
    permute_239: "f32[768, 768]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    view_395: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_17, [4, 512, 768]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_128: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_149, view_395);  mul_149 = view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_240: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_127, [0, 2, 1, 3]);  add_127 = None
    clone_141: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
    view_396: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_141, [4, 512, 768]);  clone_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_397: "f32[2048, 768]" = torch.ops.aten.view.default(view_396, [2048, 768]);  view_396 = None
    permute_241: "f32[768, 768]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    mm_19: "f32[2048, 768]" = torch.ops.aten.mm.default(view_397, permute_241);  permute_241 = None
    permute_242: "f32[768, 2048]" = torch.ops.aten.permute.default(view_397, [1, 0])
    mm_20: "f32[768, 768]" = torch.ops.aten.mm.default(permute_242, view_315);  permute_242 = view_315 = None
    permute_243: "f32[768, 768]" = torch.ops.aten.permute.default(mm_20, [1, 0]);  mm_20 = None
    sum_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
    view_398: "f32[768]" = torch.ops.aten.view.default(sum_41, [768]);  sum_41 = None
    permute_244: "f32[768, 768]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_399: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_19, [4, 512, 768]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_129: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_128, view_399);  add_128 = view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_153: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_391, 0.125);  view_391 = None
    view_400: "f32[2048, 768]" = torch.ops.aten.view.default(mul_153, [2048, 768]);  mul_153 = None
    permute_245: "f32[768, 768]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    mm_21: "f32[2048, 768]" = torch.ops.aten.mm.default(view_400, permute_245);  permute_245 = None
    permute_246: "f32[768, 2048]" = torch.ops.aten.permute.default(view_400, [1, 0])
    mm_22: "f32[768, 768]" = torch.ops.aten.mm.default(permute_246, view_313);  permute_246 = view_313 = None
    permute_247: "f32[768, 768]" = torch.ops.aten.permute.default(mm_22, [1, 0]);  mm_22 = None
    sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_400, [0], True);  view_400 = None
    view_401: "f32[768]" = torch.ops.aten.view.default(sum_42, [768]);  sum_42 = None
    permute_248: "f32[768, 768]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_402: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_21, [4, 512, 768]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_130: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_129, view_402);  add_129 = view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_61: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_57);  add_103 = getitem_57 = None
    mul_154: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_28);  sub_61 = None
    mul_155: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_130, primals_233);  primals_233 = None
    mul_156: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_155, 768)
    sum_43: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [2], True)
    mul_157: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_155, mul_154);  mul_155 = None
    sum_44: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [2], True);  mul_157 = None
    mul_158: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_154, sum_44);  sum_44 = None
    sub_62: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_156, sum_43);  mul_156 = sum_43 = None
    sub_63: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_158);  sub_62 = mul_158 = None
    div_21: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 768);  rsqrt_28 = None
    mul_159: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_63);  div_21 = sub_63 = None
    mul_160: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_130, mul_154);  mul_154 = None
    sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_160, [0, 1]);  mul_160 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_130, [0, 1]);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_403: "f32[2048, 768]" = torch.ops.aten.view.default(mul_159, [2048, 768])
    permute_249: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    mm_23: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_403, permute_249);  permute_249 = None
    permute_250: "f32[768, 2048]" = torch.ops.aten.permute.default(view_403, [1, 0])
    mm_24: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_250, view_311);  permute_250 = view_311 = None
    permute_251: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_24, [1, 0]);  mm_24 = None
    sum_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_403, [0], True);  view_403 = None
    view_404: "f32[768]" = torch.ops.aten.view.default(sum_47, [768]);  sum_47 = None
    permute_252: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    view_405: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_23, [4, 512, 3072]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_161: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_310, 0.7071067811865476)
    erf_13: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_161);  mul_161 = None
    add_131: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_162: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_131, 0.5);  add_131 = None
    mul_163: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_310, view_310)
    mul_164: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_163, -0.5);  mul_163 = None
    exp_19: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_164);  mul_164 = None
    mul_165: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_166: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_310, mul_165);  view_310 = mul_165 = None
    add_132: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_162, mul_166);  mul_162 = mul_166 = None
    mul_167: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_405, add_132);  view_405 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_406: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_167, [2048, 3072]);  mul_167 = None
    permute_253: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    mm_25: "f32[2048, 768]" = torch.ops.aten.mm.default(view_406, permute_253);  permute_253 = None
    permute_254: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_406, [1, 0])
    mm_26: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_254, view_309);  permute_254 = view_309 = None
    permute_255: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_26, [1, 0]);  mm_26 = None
    sum_48: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
    view_407: "f32[3072]" = torch.ops.aten.view.default(sum_48, [3072]);  sum_48 = None
    permute_256: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
    view_408: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_25, [4, 512, 768]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_133: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_159, view_408);  mul_159 = view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    sub_64: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_55);  add_99 = getitem_55 = None
    mul_168: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_27);  sub_64 = None
    mul_169: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, primals_227);  primals_227 = None
    mul_170: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_169, 768)
    sum_49: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True)
    mul_171: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_169, mul_168);  mul_169 = None
    sum_50: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True);  mul_171 = None
    mul_172: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_168, sum_50);  sum_50 = None
    sub_65: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_170, sum_49);  mul_170 = sum_49 = None
    sub_66: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_172);  sub_65 = mul_172 = None
    div_22: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 768);  rsqrt_27 = None
    mul_173: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_66);  div_22 = sub_66 = None
    mul_174: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, mul_168);  mul_168 = None
    sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_174, [0, 1]);  mul_174 = None
    sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 1]);  add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_409: "f32[2048, 768]" = torch.ops.aten.view.default(mul_173, [2048, 768])
    permute_257: "f32[768, 768]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    mm_27: "f32[2048, 768]" = torch.ops.aten.mm.default(view_409, permute_257);  permute_257 = None
    permute_258: "f32[768, 2048]" = torch.ops.aten.permute.default(view_409, [1, 0])
    mm_28: "f32[768, 768]" = torch.ops.aten.mm.default(permute_258, view_307);  permute_258 = view_307 = None
    permute_259: "f32[768, 768]" = torch.ops.aten.permute.default(mm_28, [1, 0]);  mm_28 = None
    sum_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_409, [0], True);  view_409 = None
    view_410: "f32[768]" = torch.ops.aten.view.default(sum_53, [768]);  sum_53 = None
    permute_260: "f32[768, 768]" = torch.ops.aten.permute.default(permute_259, [1, 0]);  permute_259 = None
    view_411: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_27, [4, 512, 768]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_412: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_411, [4, 512, 12, 64]);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_261: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_142: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
    view_413: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_142, [48, 512, 64]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_262: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_115, [0, 2, 1]);  clone_115 = None
    bmm_44: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_262, view_413);  permute_262 = None
    permute_263: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
    bmm_45: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_413, permute_263);  view_413 = permute_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_20: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_175: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_45, alias_20);  bmm_45 = None
    sum_54: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_175, [-1], True)
    mul_176: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_20, sum_54);  alias_20 = sum_54 = None
    sub_67: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_264: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_302, [0, 2, 1]);  view_302 = None
    bmm_46: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_264, sub_67);  permute_264 = None
    permute_265: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_161, [0, 2, 1]);  permute_161 = None
    bmm_47: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_67, permute_265);  sub_67 = permute_265 = None
    permute_266: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_46, [0, 2, 1]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_414: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_44, [4, 12, 512, 64]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_134: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_21, view_414);  tangents_21 = view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_415: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_266, [4, 12, 512, 64]);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_135: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_20, view_415);  tangents_20 = view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_416: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_47, [4, 12, 512, 64]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_267: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
    clone_143: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
    view_417: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_143, [4, 512, 768]);  clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_268: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_134, [0, 2, 1, 3]);  add_134 = None
    clone_144: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
    view_418: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_144, [4, 512, 768]);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_419: "f32[2048, 768]" = torch.ops.aten.view.default(view_418, [2048, 768]);  view_418 = None
    permute_269: "f32[768, 768]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    mm_29: "f32[2048, 768]" = torch.ops.aten.mm.default(view_419, permute_269);  permute_269 = None
    permute_270: "f32[768, 2048]" = torch.ops.aten.permute.default(view_419, [1, 0])
    mm_30: "f32[768, 768]" = torch.ops.aten.mm.default(permute_270, view_298);  permute_270 = view_298 = None
    permute_271: "f32[768, 768]" = torch.ops.aten.permute.default(mm_30, [1, 0]);  mm_30 = None
    sum_55: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
    view_420: "f32[768]" = torch.ops.aten.view.default(sum_55, [768]);  sum_55 = None
    permute_272: "f32[768, 768]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    view_421: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_29, [4, 512, 768]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_136: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_124, view_421);  add_124 = view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_273: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_135, [0, 2, 1, 3]);  add_135 = None
    clone_145: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_273, memory_format = torch.contiguous_format);  permute_273 = None
    view_422: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_145, [4, 512, 768]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_423: "f32[2048, 768]" = torch.ops.aten.view.default(view_422, [2048, 768]);  view_422 = None
    permute_274: "f32[768, 768]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    mm_31: "f32[2048, 768]" = torch.ops.aten.mm.default(view_423, permute_274);  permute_274 = None
    permute_275: "f32[768, 2048]" = torch.ops.aten.permute.default(view_423, [1, 0])
    mm_32: "f32[768, 768]" = torch.ops.aten.mm.default(permute_275, view_295);  permute_275 = view_295 = None
    permute_276: "f32[768, 768]" = torch.ops.aten.permute.default(mm_32, [1, 0]);  mm_32 = None
    sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_423, [0], True);  view_423 = None
    view_424: "f32[768]" = torch.ops.aten.view.default(sum_56, [768]);  sum_56 = None
    permute_277: "f32[768, 768]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_425: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_31, [4, 512, 768]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_137: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_136, view_425);  add_136 = view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_177: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_417, 0.125);  view_417 = None
    view_426: "f32[2048, 768]" = torch.ops.aten.view.default(mul_177, [2048, 768]);  mul_177 = None
    permute_278: "f32[768, 768]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    mm_33: "f32[2048, 768]" = torch.ops.aten.mm.default(view_426, permute_278);  permute_278 = None
    permute_279: "f32[768, 2048]" = torch.ops.aten.permute.default(view_426, [1, 0])
    mm_34: "f32[768, 768]" = torch.ops.aten.mm.default(permute_279, view_293);  permute_279 = view_293 = None
    permute_280: "f32[768, 768]" = torch.ops.aten.permute.default(mm_34, [1, 0]);  mm_34 = None
    sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_426, [0], True);  view_426 = None
    view_427: "f32[768]" = torch.ops.aten.view.default(sum_57, [768]);  sum_57 = None
    permute_281: "f32[768, 768]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_428: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_33, [4, 512, 768]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_138: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_173, view_428);  mul_173 = view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_68: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_53);  add_96 = getitem_53 = None
    mul_178: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_26);  sub_68 = None
    mul_179: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, primals_217);  primals_217 = None
    mul_180: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_179, 768)
    sum_58: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
    mul_181: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_179, mul_178);  mul_179 = None
    sum_59: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
    mul_182: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_178, sum_59);  sum_59 = None
    sub_69: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_180, sum_58);  mul_180 = sum_58 = None
    sub_70: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_182);  sub_69 = mul_182 = None
    div_23: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 768);  rsqrt_26 = None
    mul_183: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_70);  div_23 = sub_70 = None
    mul_184: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, mul_178);  mul_178 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
    sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_429: "f32[2048, 768]" = torch.ops.aten.view.default(mul_183, [2048, 768])
    permute_282: "f32[768, 768]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    mm_35: "f32[2048, 768]" = torch.ops.aten.mm.default(view_429, permute_282);  permute_282 = None
    permute_283: "f32[768, 2048]" = torch.ops.aten.permute.default(view_429, [1, 0])
    mm_36: "f32[768, 768]" = torch.ops.aten.mm.default(permute_283, view_291);  permute_283 = view_291 = None
    permute_284: "f32[768, 768]" = torch.ops.aten.permute.default(mm_36, [1, 0]);  mm_36 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_429, [0], True);  view_429 = None
    view_430: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    view_431: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_35, [4, 512, 768]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_432: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_431, [4, 512, 12, 64]);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_286: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_146: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    view_433: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_146, [48, 512, 64]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_287: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_109, [0, 2, 1]);  clone_109 = None
    bmm_48: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_287, view_433);  permute_287 = None
    permute_288: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
    bmm_49: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_433, permute_288);  view_433 = permute_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_21: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    mul_185: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_49, alias_21);  bmm_49 = None
    sum_63: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [-1], True)
    mul_186: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_21, sum_63);  alias_21 = sum_63 = None
    sub_71: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_434: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(sub_71, [4, 12, 512, 512]);  sub_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_435: "f32[48, 512, 512]" = torch.ops.aten.view.default(view_434, [48, 512, 512]);  view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_289: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
    bmm_50: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_289, view_435);  permute_289 = None
    permute_290: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_152, [0, 2, 1]);  permute_152 = None
    bmm_51: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_435, permute_290);  view_435 = permute_290 = None
    permute_291: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_50, [0, 2, 1]);  bmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_436: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_48, [4, 12, 512, 64]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_139: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_19, view_436);  tangents_19 = view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_437: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_291, [4, 12, 512, 64]);  permute_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_140: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_18, view_437);  tangents_18 = view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_438: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_51, [4, 12, 512, 64]);  bmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_292: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_438, [0, 2, 1, 3]);  view_438 = None
    clone_147: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    view_439: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_147, [4, 512, 768]);  clone_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_293: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_139, [0, 2, 1, 3]);  add_139 = None
    clone_148: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_440: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_148, [4, 512, 768]);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_441: "f32[2048, 768]" = torch.ops.aten.view.default(view_440, [2048, 768]);  view_440 = None
    permute_294: "f32[768, 768]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    mm_37: "f32[2048, 768]" = torch.ops.aten.mm.default(view_441, permute_294);  permute_294 = None
    permute_295: "f32[768, 2048]" = torch.ops.aten.permute.default(view_441, [1, 0])
    mm_38: "f32[768, 768]" = torch.ops.aten.mm.default(permute_295, view_280);  permute_295 = view_280 = None
    permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(mm_38, [1, 0]);  mm_38 = None
    sum_64: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_441, [0], True);  view_441 = None
    view_442: "f32[768]" = torch.ops.aten.view.default(sum_64, [768]);  sum_64 = None
    permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_443: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_37, [4, 512, 768]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_141: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_183, view_443);  mul_183 = view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_298: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_140, [0, 2, 1, 3]);  add_140 = None
    clone_149: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_298, memory_format = torch.contiguous_format);  permute_298 = None
    view_444: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_149, [4, 512, 768]);  clone_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_445: "f32[2048, 768]" = torch.ops.aten.view.default(view_444, [2048, 768]);  view_444 = None
    permute_299: "f32[768, 768]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    mm_39: "f32[2048, 768]" = torch.ops.aten.mm.default(view_445, permute_299);  permute_299 = None
    permute_300: "f32[768, 2048]" = torch.ops.aten.permute.default(view_445, [1, 0])
    mm_40: "f32[768, 768]" = torch.ops.aten.mm.default(permute_300, view_277);  permute_300 = view_277 = None
    permute_301: "f32[768, 768]" = torch.ops.aten.permute.default(mm_40, [1, 0]);  mm_40 = None
    sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
    view_446: "f32[768]" = torch.ops.aten.view.default(sum_65, [768]);  sum_65 = None
    permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    view_447: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_39, [4, 512, 768]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_142: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_141, view_447);  add_141 = view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_187: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_439, 0.125);  view_439 = None
    view_448: "f32[2048, 768]" = torch.ops.aten.view.default(mul_187, [2048, 768]);  mul_187 = None
    permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    mm_41: "f32[2048, 768]" = torch.ops.aten.mm.default(view_448, permute_303);  permute_303 = None
    permute_304: "f32[768, 2048]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_42: "f32[768, 768]" = torch.ops.aten.mm.default(permute_304, view_275);  permute_304 = view_275 = None
    permute_305: "f32[768, 768]" = torch.ops.aten.permute.default(mm_42, [1, 0]);  mm_42 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
    view_449: "f32[768]" = torch.ops.aten.view.default(sum_66, [768]);  sum_66 = None
    permute_306: "f32[768, 768]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    view_450: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_41, [4, 512, 768]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_143: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_142, view_450);  add_142 = view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_72: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_51);  add_92 = getitem_51 = None
    mul_188: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_25);  sub_72 = None
    mul_189: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, primals_207);  primals_207 = None
    mul_190: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_189, 768)
    sum_67: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [2], True)
    mul_191: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_189, mul_188);  mul_189 = None
    sum_68: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_191, [2], True);  mul_191 = None
    mul_192: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_188, sum_68);  sum_68 = None
    sub_73: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_190, sum_67);  mul_190 = sum_67 = None
    sub_74: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_73, mul_192);  sub_73 = mul_192 = None
    div_24: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
    mul_193: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_74);  div_24 = sub_74 = None
    mul_194: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, mul_188);  mul_188 = None
    sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_194, [0, 1]);  mul_194 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_451: "f32[2048, 768]" = torch.ops.aten.view.default(mul_193, [2048, 768])
    permute_307: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    mm_43: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_451, permute_307);  permute_307 = None
    permute_308: "f32[768, 2048]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_44: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_308, view_273);  permute_308 = view_273 = None
    permute_309: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_44, [1, 0]);  mm_44 = None
    sum_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
    view_452: "f32[768]" = torch.ops.aten.view.default(sum_71, [768]);  sum_71 = None
    permute_310: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    view_453: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_43, [4, 512, 3072]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_195: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_272, 0.7071067811865476)
    erf_14: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_144: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_196: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_144, 0.5);  add_144 = None
    mul_197: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_272, view_272)
    mul_198: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_197, -0.5);  mul_197 = None
    exp_20: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_198);  mul_198 = None
    mul_199: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_200: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_272, mul_199);  view_272 = mul_199 = None
    add_145: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_196, mul_200);  mul_196 = mul_200 = None
    mul_201: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_453, add_145);  view_453 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_454: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_201, [2048, 3072]);  mul_201 = None
    permute_311: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    mm_45: "f32[2048, 768]" = torch.ops.aten.mm.default(view_454, permute_311);  permute_311 = None
    permute_312: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_454, [1, 0])
    mm_46: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_312, view_271);  permute_312 = view_271 = None
    permute_313: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_46, [1, 0]);  mm_46 = None
    sum_72: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_454, [0], True);  view_454 = None
    view_455: "f32[3072]" = torch.ops.aten.view.default(sum_72, [3072]);  sum_72 = None
    permute_314: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    view_456: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_45, [4, 512, 768]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_146: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_193, view_456);  mul_193 = view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    sub_75: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_49);  add_88 = getitem_49 = None
    mul_202: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_24);  sub_75 = None
    mul_203: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, primals_201);  primals_201 = None
    mul_204: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_203, 768)
    sum_73: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True)
    mul_205: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_203, mul_202);  mul_203 = None
    sum_74: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_205, [2], True);  mul_205 = None
    mul_206: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_202, sum_74);  sum_74 = None
    sub_76: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_204, sum_73);  mul_204 = sum_73 = None
    sub_77: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_76, mul_206);  sub_76 = mul_206 = None
    div_25: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    mul_207: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_77);  div_25 = sub_77 = None
    mul_208: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_202);  mul_202 = None
    sum_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_208, [0, 1]);  mul_208 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_457: "f32[2048, 768]" = torch.ops.aten.view.default(mul_207, [2048, 768])
    permute_315: "f32[768, 768]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    mm_47: "f32[2048, 768]" = torch.ops.aten.mm.default(view_457, permute_315);  permute_315 = None
    permute_316: "f32[768, 2048]" = torch.ops.aten.permute.default(view_457, [1, 0])
    mm_48: "f32[768, 768]" = torch.ops.aten.mm.default(permute_316, view_269);  permute_316 = view_269 = None
    permute_317: "f32[768, 768]" = torch.ops.aten.permute.default(mm_48, [1, 0]);  mm_48 = None
    sum_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_457, [0], True);  view_457 = None
    view_458: "f32[768]" = torch.ops.aten.view.default(sum_77, [768]);  sum_77 = None
    permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    view_459: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_47, [4, 512, 768]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_460: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_459, [4, 512, 12, 64]);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_319: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_150: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_461: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_150, [48, 512, 64]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_320: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_101, [0, 2, 1]);  clone_101 = None
    bmm_52: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_320, view_461);  permute_320 = None
    permute_321: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
    bmm_53: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_461, permute_321);  view_461 = permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_22: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_209: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_53, alias_22);  bmm_53 = None
    sum_78: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [-1], True)
    mul_210: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_22, sum_78);  alias_22 = sum_78 = None
    sub_78: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_209, mul_210);  mul_209 = mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_322: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
    bmm_54: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_322, sub_78);  permute_322 = None
    permute_323: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_141, [0, 2, 1]);  permute_141 = None
    bmm_55: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_78, permute_323);  sub_78 = permute_323 = None
    permute_324: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_54, [0, 2, 1]);  bmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_462: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_52, [4, 12, 512, 64]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_147: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_17, view_462);  tangents_17 = view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_463: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_324, [4, 12, 512, 64]);  permute_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_148: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_16, view_463);  tangents_16 = view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_464: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_55, [4, 12, 512, 64]);  bmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_325: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None
    clone_151: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    view_465: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_151, [4, 512, 768]);  clone_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_326: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_147, [0, 2, 1, 3]);  add_147 = None
    clone_152: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_466: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_152, [4, 512, 768]);  clone_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_467: "f32[2048, 768]" = torch.ops.aten.view.default(view_466, [2048, 768]);  view_466 = None
    permute_327: "f32[768, 768]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    mm_49: "f32[2048, 768]" = torch.ops.aten.mm.default(view_467, permute_327);  permute_327 = None
    permute_328: "f32[768, 2048]" = torch.ops.aten.permute.default(view_467, [1, 0])
    mm_50: "f32[768, 768]" = torch.ops.aten.mm.default(permute_328, view_260);  permute_328 = view_260 = None
    permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(mm_50, [1, 0]);  mm_50 = None
    sum_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
    view_468: "f32[768]" = torch.ops.aten.view.default(sum_79, [768]);  sum_79 = None
    permute_330: "f32[768, 768]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    view_469: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_49, [4, 512, 768]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_149: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_137, view_469);  add_137 = view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_331: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_148, [0, 2, 1, 3]);  add_148 = None
    clone_153: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_331, memory_format = torch.contiguous_format);  permute_331 = None
    view_470: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_153, [4, 512, 768]);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_471: "f32[2048, 768]" = torch.ops.aten.view.default(view_470, [2048, 768]);  view_470 = None
    permute_332: "f32[768, 768]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    mm_51: "f32[2048, 768]" = torch.ops.aten.mm.default(view_471, permute_332);  permute_332 = None
    permute_333: "f32[768, 2048]" = torch.ops.aten.permute.default(view_471, [1, 0])
    mm_52: "f32[768, 768]" = torch.ops.aten.mm.default(permute_333, view_257);  permute_333 = view_257 = None
    permute_334: "f32[768, 768]" = torch.ops.aten.permute.default(mm_52, [1, 0]);  mm_52 = None
    sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_471, [0], True);  view_471 = None
    view_472: "f32[768]" = torch.ops.aten.view.default(sum_80, [768]);  sum_80 = None
    permute_335: "f32[768, 768]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_473: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_51, [4, 512, 768]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_150: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_149, view_473);  add_149 = view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_211: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_465, 0.125);  view_465 = None
    view_474: "f32[2048, 768]" = torch.ops.aten.view.default(mul_211, [2048, 768]);  mul_211 = None
    permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    mm_53: "f32[2048, 768]" = torch.ops.aten.mm.default(view_474, permute_336);  permute_336 = None
    permute_337: "f32[768, 2048]" = torch.ops.aten.permute.default(view_474, [1, 0])
    mm_54: "f32[768, 768]" = torch.ops.aten.mm.default(permute_337, view_255);  permute_337 = view_255 = None
    permute_338: "f32[768, 768]" = torch.ops.aten.permute.default(mm_54, [1, 0]);  mm_54 = None
    sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_474, [0], True);  view_474 = None
    view_475: "f32[768]" = torch.ops.aten.view.default(sum_81, [768]);  sum_81 = None
    permute_339: "f32[768, 768]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_476: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_53, [4, 512, 768]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_151: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_207, view_476);  mul_207 = view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_79: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_47);  add_85 = getitem_47 = None
    mul_212: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_23);  sub_79 = None
    mul_213: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_151, primals_191);  primals_191 = None
    mul_214: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
    sum_82: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_212);  mul_213 = None
    sum_83: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_212, sum_83);  sum_83 = None
    sub_80: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_82);  mul_214 = sum_82 = None
    sub_81: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_216);  sub_80 = mul_216 = None
    div_26: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    mul_217: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_81);  div_26 = sub_81 = None
    mul_218: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_151, mul_212);  mul_212 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_151, [0, 1]);  add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_477: "f32[2048, 768]" = torch.ops.aten.view.default(mul_217, [2048, 768])
    permute_340: "f32[768, 768]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    mm_55: "f32[2048, 768]" = torch.ops.aten.mm.default(view_477, permute_340);  permute_340 = None
    permute_341: "f32[768, 2048]" = torch.ops.aten.permute.default(view_477, [1, 0])
    mm_56: "f32[768, 768]" = torch.ops.aten.mm.default(permute_341, view_253);  permute_341 = view_253 = None
    permute_342: "f32[768, 768]" = torch.ops.aten.permute.default(mm_56, [1, 0]);  mm_56 = None
    sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_477, [0], True);  view_477 = None
    view_478: "f32[768]" = torch.ops.aten.view.default(sum_86, [768]);  sum_86 = None
    permute_343: "f32[768, 768]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_479: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_55, [4, 512, 768]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_480: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_479, [4, 512, 12, 64]);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_344: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_480, [0, 2, 1, 3]);  view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_154: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_344, memory_format = torch.contiguous_format);  permute_344 = None
    view_481: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_154, [48, 512, 64]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_345: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_95, [0, 2, 1]);  clone_95 = None
    bmm_56: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_345, view_481);  permute_345 = None
    permute_346: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
    bmm_57: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_481, permute_346);  view_481 = permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_23: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_219: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_57, alias_23);  bmm_57 = None
    sum_87: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [-1], True)
    mul_220: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_23, sum_87);  alias_23 = sum_87 = None
    sub_82: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_482: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(sub_82, [4, 12, 512, 512]);  sub_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_483: "f32[48, 512, 512]" = torch.ops.aten.view.default(view_482, [48, 512, 512]);  view_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_347: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_246, [0, 2, 1]);  view_246 = None
    bmm_58: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_347, view_483);  permute_347 = None
    permute_348: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_132, [0, 2, 1]);  permute_132 = None
    bmm_59: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_483, permute_348);  view_483 = permute_348 = None
    permute_349: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_58, [0, 2, 1]);  bmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_484: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_56, [4, 12, 512, 64]);  bmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_152: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_15, view_484);  tangents_15 = view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_485: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_349, [4, 12, 512, 64]);  permute_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_153: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_14, view_485);  tangents_14 = view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_486: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_59, [4, 12, 512, 64]);  bmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_350: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
    clone_155: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_350, memory_format = torch.contiguous_format);  permute_350 = None
    view_487: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_155, [4, 512, 768]);  clone_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_351: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_152, [0, 2, 1, 3]);  add_152 = None
    clone_156: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_351, memory_format = torch.contiguous_format);  permute_351 = None
    view_488: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_156, [4, 512, 768]);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_489: "f32[2048, 768]" = torch.ops.aten.view.default(view_488, [2048, 768]);  view_488 = None
    permute_352: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    mm_57: "f32[2048, 768]" = torch.ops.aten.mm.default(view_489, permute_352);  permute_352 = None
    permute_353: "f32[768, 2048]" = torch.ops.aten.permute.default(view_489, [1, 0])
    mm_58: "f32[768, 768]" = torch.ops.aten.mm.default(permute_353, view_242);  permute_353 = view_242 = None
    permute_354: "f32[768, 768]" = torch.ops.aten.permute.default(mm_58, [1, 0]);  mm_58 = None
    sum_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_489, [0], True);  view_489 = None
    view_490: "f32[768]" = torch.ops.aten.view.default(sum_88, [768]);  sum_88 = None
    permute_355: "f32[768, 768]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    view_491: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_57, [4, 512, 768]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_154: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_217, view_491);  mul_217 = view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_356: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_153, [0, 2, 1, 3]);  add_153 = None
    clone_157: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
    view_492: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_157, [4, 512, 768]);  clone_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_493: "f32[2048, 768]" = torch.ops.aten.view.default(view_492, [2048, 768]);  view_492 = None
    permute_357: "f32[768, 768]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    mm_59: "f32[2048, 768]" = torch.ops.aten.mm.default(view_493, permute_357);  permute_357 = None
    permute_358: "f32[768, 2048]" = torch.ops.aten.permute.default(view_493, [1, 0])
    mm_60: "f32[768, 768]" = torch.ops.aten.mm.default(permute_358, view_239);  permute_358 = view_239 = None
    permute_359: "f32[768, 768]" = torch.ops.aten.permute.default(mm_60, [1, 0]);  mm_60 = None
    sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
    view_494: "f32[768]" = torch.ops.aten.view.default(sum_89, [768]);  sum_89 = None
    permute_360: "f32[768, 768]" = torch.ops.aten.permute.default(permute_359, [1, 0]);  permute_359 = None
    view_495: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_59, [4, 512, 768]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_155: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_154, view_495);  add_154 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_221: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_487, 0.125);  view_487 = None
    view_496: "f32[2048, 768]" = torch.ops.aten.view.default(mul_221, [2048, 768]);  mul_221 = None
    permute_361: "f32[768, 768]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    mm_61: "f32[2048, 768]" = torch.ops.aten.mm.default(view_496, permute_361);  permute_361 = None
    permute_362: "f32[768, 2048]" = torch.ops.aten.permute.default(view_496, [1, 0])
    mm_62: "f32[768, 768]" = torch.ops.aten.mm.default(permute_362, view_237);  permute_362 = view_237 = None
    permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(mm_62, [1, 0]);  mm_62 = None
    sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
    view_497: "f32[768]" = torch.ops.aten.view.default(sum_90, [768]);  sum_90 = None
    permute_364: "f32[768, 768]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
    view_498: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_61, [4, 512, 768]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_156: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_155, view_498);  add_155 = view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_83: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_45);  add_81 = getitem_45 = None
    mul_222: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_22);  sub_83 = None
    mul_223: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, primals_181);  primals_181 = None
    mul_224: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_223, 768)
    sum_91: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True)
    mul_225: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_223, mul_222);  mul_223 = None
    sum_92: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True);  mul_225 = None
    mul_226: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_222, sum_92);  sum_92 = None
    sub_84: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_224, sum_91);  mul_224 = sum_91 = None
    sub_85: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_84, mul_226);  sub_84 = mul_226 = None
    div_27: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    mul_227: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_85);  div_27 = sub_85 = None
    mul_228: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, mul_222);  mul_222 = None
    sum_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_228, [0, 1]);  mul_228 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_499: "f32[2048, 768]" = torch.ops.aten.view.default(mul_227, [2048, 768])
    permute_365: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    mm_63: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_499, permute_365);  permute_365 = None
    permute_366: "f32[768, 2048]" = torch.ops.aten.permute.default(view_499, [1, 0])
    mm_64: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_366, view_235);  permute_366 = view_235 = None
    permute_367: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_499, [0], True);  view_499 = None
    view_500: "f32[768]" = torch.ops.aten.view.default(sum_95, [768]);  sum_95 = None
    permute_368: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    view_501: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_63, [4, 512, 3072]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_229: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_234, 0.7071067811865476)
    erf_15: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_229);  mul_229 = None
    add_157: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_230: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_157, 0.5);  add_157 = None
    mul_231: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_234, view_234)
    mul_232: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_231, -0.5);  mul_231 = None
    exp_21: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_232);  mul_232 = None
    mul_233: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_234: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_234, mul_233);  view_234 = mul_233 = None
    add_158: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_230, mul_234);  mul_230 = mul_234 = None
    mul_235: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_501, add_158);  view_501 = add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_502: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_235, [2048, 3072]);  mul_235 = None
    permute_369: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    mm_65: "f32[2048, 768]" = torch.ops.aten.mm.default(view_502, permute_369);  permute_369 = None
    permute_370: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_502, [1, 0])
    mm_66: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_370, view_233);  permute_370 = view_233 = None
    permute_371: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_66, [1, 0]);  mm_66 = None
    sum_96: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_502, [0], True);  view_502 = None
    view_503: "f32[3072]" = torch.ops.aten.view.default(sum_96, [3072]);  sum_96 = None
    permute_372: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    view_504: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_65, [4, 512, 768]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_159: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_227, view_504);  mul_227 = view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    sub_86: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_43);  add_77 = getitem_43 = None
    mul_236: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_21);  sub_86 = None
    mul_237: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_159, primals_175);  primals_175 = None
    mul_238: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_237, 768)
    sum_97: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [2], True)
    mul_239: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_237, mul_236);  mul_237 = None
    sum_98: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True);  mul_239 = None
    mul_240: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_236, sum_98);  sum_98 = None
    sub_87: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_238, sum_97);  mul_238 = sum_97 = None
    sub_88: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_240);  sub_87 = mul_240 = None
    div_28: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    mul_241: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_88);  div_28 = sub_88 = None
    mul_242: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_159, mul_236);  mul_236 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1]);  mul_242 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_159, [0, 1]);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_505: "f32[2048, 768]" = torch.ops.aten.view.default(mul_241, [2048, 768])
    permute_373: "f32[768, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    mm_67: "f32[2048, 768]" = torch.ops.aten.mm.default(view_505, permute_373);  permute_373 = None
    permute_374: "f32[768, 2048]" = torch.ops.aten.permute.default(view_505, [1, 0])
    mm_68: "f32[768, 768]" = torch.ops.aten.mm.default(permute_374, view_231);  permute_374 = view_231 = None
    permute_375: "f32[768, 768]" = torch.ops.aten.permute.default(mm_68, [1, 0]);  mm_68 = None
    sum_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_505, [0], True);  view_505 = None
    view_506: "f32[768]" = torch.ops.aten.view.default(sum_101, [768]);  sum_101 = None
    permute_376: "f32[768, 768]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    view_507: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_67, [4, 512, 768]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_508: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_507, [4, 512, 12, 64]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_377: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_158: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_377, memory_format = torch.contiguous_format);  permute_377 = None
    view_509: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_158, [48, 512, 64]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_378: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_87, [0, 2, 1]);  clone_87 = None
    bmm_60: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_378, view_509);  permute_378 = None
    permute_379: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
    bmm_61: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_509, permute_379);  view_509 = permute_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_24: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_243: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_61, alias_24);  bmm_61 = None
    sum_102: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [-1], True)
    mul_244: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_24, sum_102);  alias_24 = sum_102 = None
    sub_89: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_380: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_226, [0, 2, 1]);  view_226 = None
    bmm_62: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_380, sub_89);  permute_380 = None
    permute_381: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_121, [0, 2, 1]);  permute_121 = None
    bmm_63: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_89, permute_381);  sub_89 = permute_381 = None
    permute_382: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_62, [0, 2, 1]);  bmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_510: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_60, [4, 12, 512, 64]);  bmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_160: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_13, view_510);  tangents_13 = view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_511: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_382, [4, 12, 512, 64]);  permute_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_161: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_12, view_511);  tangents_12 = view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_512: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_63, [4, 12, 512, 64]);  bmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_383: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
    clone_159: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_383, memory_format = torch.contiguous_format);  permute_383 = None
    view_513: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_159, [4, 512, 768]);  clone_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_384: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_160, [0, 2, 1, 3]);  add_160 = None
    clone_160: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_384, memory_format = torch.contiguous_format);  permute_384 = None
    view_514: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_160, [4, 512, 768]);  clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_515: "f32[2048, 768]" = torch.ops.aten.view.default(view_514, [2048, 768]);  view_514 = None
    permute_385: "f32[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    mm_69: "f32[2048, 768]" = torch.ops.aten.mm.default(view_515, permute_385);  permute_385 = None
    permute_386: "f32[768, 2048]" = torch.ops.aten.permute.default(view_515, [1, 0])
    mm_70: "f32[768, 768]" = torch.ops.aten.mm.default(permute_386, view_222);  permute_386 = view_222 = None
    permute_387: "f32[768, 768]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    sum_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_515, [0], True);  view_515 = None
    view_516: "f32[768]" = torch.ops.aten.view.default(sum_103, [768]);  sum_103 = None
    permute_388: "f32[768, 768]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
    view_517: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_69, [4, 512, 768]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_162: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_150, view_517);  add_150 = view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_389: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_161, [0, 2, 1, 3]);  add_161 = None
    clone_161: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_389, memory_format = torch.contiguous_format);  permute_389 = None
    view_518: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_161, [4, 512, 768]);  clone_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_519: "f32[2048, 768]" = torch.ops.aten.view.default(view_518, [2048, 768]);  view_518 = None
    permute_390: "f32[768, 768]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    mm_71: "f32[2048, 768]" = torch.ops.aten.mm.default(view_519, permute_390);  permute_390 = None
    permute_391: "f32[768, 2048]" = torch.ops.aten.permute.default(view_519, [1, 0])
    mm_72: "f32[768, 768]" = torch.ops.aten.mm.default(permute_391, view_219);  permute_391 = view_219 = None
    permute_392: "f32[768, 768]" = torch.ops.aten.permute.default(mm_72, [1, 0]);  mm_72 = None
    sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_519, [0], True);  view_519 = None
    view_520: "f32[768]" = torch.ops.aten.view.default(sum_104, [768]);  sum_104 = None
    permute_393: "f32[768, 768]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
    view_521: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_71, [4, 512, 768]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_163: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_162, view_521);  add_162 = view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_245: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_513, 0.125);  view_513 = None
    view_522: "f32[2048, 768]" = torch.ops.aten.view.default(mul_245, [2048, 768]);  mul_245 = None
    permute_394: "f32[768, 768]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    mm_73: "f32[2048, 768]" = torch.ops.aten.mm.default(view_522, permute_394);  permute_394 = None
    permute_395: "f32[768, 2048]" = torch.ops.aten.permute.default(view_522, [1, 0])
    mm_74: "f32[768, 768]" = torch.ops.aten.mm.default(permute_395, view_217);  permute_395 = view_217 = None
    permute_396: "f32[768, 768]" = torch.ops.aten.permute.default(mm_74, [1, 0]);  mm_74 = None
    sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_522, [0], True);  view_522 = None
    view_523: "f32[768]" = torch.ops.aten.view.default(sum_105, [768]);  sum_105 = None
    permute_397: "f32[768, 768]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_524: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_73, [4, 512, 768]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_164: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_241, view_524);  mul_241 = view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_90: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_74, getitem_41);  add_74 = getitem_41 = None
    mul_246: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_20);  sub_90 = None
    mul_247: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, primals_165);  primals_165 = None
    mul_248: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_247, 768)
    sum_106: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True)
    mul_249: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_247, mul_246);  mul_247 = None
    sum_107: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [2], True);  mul_249 = None
    mul_250: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_246, sum_107);  sum_107 = None
    sub_91: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_248, sum_106);  mul_248 = sum_106 = None
    sub_92: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_91, mul_250);  sub_91 = mul_250 = None
    div_29: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    mul_251: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_29, sub_92);  div_29 = sub_92 = None
    mul_252: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, mul_246);  mul_246 = None
    sum_108: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 1]);  mul_252 = None
    sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_525: "f32[2048, 768]" = torch.ops.aten.view.default(mul_251, [2048, 768])
    permute_398: "f32[768, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    mm_75: "f32[2048, 768]" = torch.ops.aten.mm.default(view_525, permute_398);  permute_398 = None
    permute_399: "f32[768, 2048]" = torch.ops.aten.permute.default(view_525, [1, 0])
    mm_76: "f32[768, 768]" = torch.ops.aten.mm.default(permute_399, view_215);  permute_399 = view_215 = None
    permute_400: "f32[768, 768]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_525, [0], True);  view_525 = None
    view_526: "f32[768]" = torch.ops.aten.view.default(sum_110, [768]);  sum_110 = None
    permute_401: "f32[768, 768]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_527: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_75, [4, 512, 768]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_528: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_527, [4, 512, 12, 64]);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_402: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_162: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_402, memory_format = torch.contiguous_format);  permute_402 = None
    view_529: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_162, [48, 512, 64]);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_403: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_81, [0, 2, 1]);  clone_81 = None
    bmm_64: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_403, view_529);  permute_403 = None
    permute_404: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    bmm_65: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_529, permute_404);  view_529 = permute_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_25: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_253: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_65, alias_25);  bmm_65 = None
    sum_111: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [-1], True)
    mul_254: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_25, sum_111);  alias_25 = sum_111 = None
    sub_93: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_530: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(sub_93, [4, 12, 512, 512]);  sub_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_531: "f32[48, 512, 512]" = torch.ops.aten.view.default(view_530, [48, 512, 512]);  view_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_405: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    bmm_66: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_405, view_531);  permute_405 = None
    permute_406: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_112, [0, 2, 1]);  permute_112 = None
    bmm_67: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_531, permute_406);  view_531 = permute_406 = None
    permute_407: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_66, [0, 2, 1]);  bmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_532: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_64, [4, 12, 512, 64]);  bmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_165: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_11, view_532);  tangents_11 = view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_533: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_407, [4, 12, 512, 64]);  permute_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_166: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_10, view_533);  tangents_10 = view_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_534: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_67, [4, 12, 512, 64]);  bmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_408: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
    clone_163: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_408, memory_format = torch.contiguous_format);  permute_408 = None
    view_535: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_163, [4, 512, 768]);  clone_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_409: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_165, [0, 2, 1, 3]);  add_165 = None
    clone_164: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_409, memory_format = torch.contiguous_format);  permute_409 = None
    view_536: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_164, [4, 512, 768]);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_537: "f32[2048, 768]" = torch.ops.aten.view.default(view_536, [2048, 768]);  view_536 = None
    permute_410: "f32[768, 768]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_77: "f32[2048, 768]" = torch.ops.aten.mm.default(view_537, permute_410);  permute_410 = None
    permute_411: "f32[768, 2048]" = torch.ops.aten.permute.default(view_537, [1, 0])
    mm_78: "f32[768, 768]" = torch.ops.aten.mm.default(permute_411, view_204);  permute_411 = view_204 = None
    permute_412: "f32[768, 768]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    sum_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
    view_538: "f32[768]" = torch.ops.aten.view.default(sum_112, [768]);  sum_112 = None
    permute_413: "f32[768, 768]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_539: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_77, [4, 512, 768]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_167: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_251, view_539);  mul_251 = view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_414: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_166, [0, 2, 1, 3]);  add_166 = None
    clone_165: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_414, memory_format = torch.contiguous_format);  permute_414 = None
    view_540: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_165, [4, 512, 768]);  clone_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_541: "f32[2048, 768]" = torch.ops.aten.view.default(view_540, [2048, 768]);  view_540 = None
    permute_415: "f32[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_79: "f32[2048, 768]" = torch.ops.aten.mm.default(view_541, permute_415);  permute_415 = None
    permute_416: "f32[768, 2048]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_80: "f32[768, 768]" = torch.ops.aten.mm.default(permute_416, view_201);  permute_416 = view_201 = None
    permute_417: "f32[768, 768]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    sum_113: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
    view_542: "f32[768]" = torch.ops.aten.view.default(sum_113, [768]);  sum_113 = None
    permute_418: "f32[768, 768]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    view_543: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_79, [4, 512, 768]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_168: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_167, view_543);  add_167 = view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_255: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_535, 0.125);  view_535 = None
    view_544: "f32[2048, 768]" = torch.ops.aten.view.default(mul_255, [2048, 768]);  mul_255 = None
    permute_419: "f32[768, 768]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    mm_81: "f32[2048, 768]" = torch.ops.aten.mm.default(view_544, permute_419);  permute_419 = None
    permute_420: "f32[768, 2048]" = torch.ops.aten.permute.default(view_544, [1, 0])
    mm_82: "f32[768, 768]" = torch.ops.aten.mm.default(permute_420, view_199);  permute_420 = view_199 = None
    permute_421: "f32[768, 768]" = torch.ops.aten.permute.default(mm_82, [1, 0]);  mm_82 = None
    sum_114: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_544, [0], True);  view_544 = None
    view_545: "f32[768]" = torch.ops.aten.view.default(sum_114, [768]);  sum_114 = None
    permute_422: "f32[768, 768]" = torch.ops.aten.permute.default(permute_421, [1, 0]);  permute_421 = None
    view_546: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_81, [4, 512, 768]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_169: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_168, view_546);  add_168 = view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_94: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_39);  add_70 = getitem_39 = None
    mul_256: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_19);  sub_94 = None
    mul_257: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_169, primals_155);  primals_155 = None
    mul_258: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_257, 768)
    sum_115: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True)
    mul_259: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_257, mul_256);  mul_257 = None
    sum_116: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [2], True);  mul_259 = None
    mul_260: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_256, sum_116);  sum_116 = None
    sub_95: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_258, sum_115);  mul_258 = sum_115 = None
    sub_96: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_260);  sub_95 = mul_260 = None
    div_30: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    mul_261: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_96);  div_30 = sub_96 = None
    mul_262: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_169, mul_256);  mul_256 = None
    sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_262, [0, 1]);  mul_262 = None
    sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_169, [0, 1]);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_547: "f32[2048, 768]" = torch.ops.aten.view.default(mul_261, [2048, 768])
    permute_423: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    mm_83: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_547, permute_423);  permute_423 = None
    permute_424: "f32[768, 2048]" = torch.ops.aten.permute.default(view_547, [1, 0])
    mm_84: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_424, view_197);  permute_424 = view_197 = None
    permute_425: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    sum_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
    view_548: "f32[768]" = torch.ops.aten.view.default(sum_119, [768]);  sum_119 = None
    permute_426: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_425, [1, 0]);  permute_425 = None
    view_549: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_83, [4, 512, 3072]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_263: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.7071067811865476)
    erf_16: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_263);  mul_263 = None
    add_170: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_264: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_170, 0.5);  add_170 = None
    mul_265: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_196, view_196)
    mul_266: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_265, -0.5);  mul_265 = None
    exp_22: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_266);  mul_266 = None
    mul_267: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_268: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_196, mul_267);  view_196 = mul_267 = None
    add_171: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_264, mul_268);  mul_264 = mul_268 = None
    mul_269: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_549, add_171);  view_549 = add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_550: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_269, [2048, 3072]);  mul_269 = None
    permute_427: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    mm_85: "f32[2048, 768]" = torch.ops.aten.mm.default(view_550, permute_427);  permute_427 = None
    permute_428: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_550, [1, 0])
    mm_86: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_428, view_195);  permute_428 = view_195 = None
    permute_429: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
    sum_120: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_550, [0], True);  view_550 = None
    view_551: "f32[3072]" = torch.ops.aten.view.default(sum_120, [3072]);  sum_120 = None
    permute_430: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    view_552: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_85, [4, 512, 768]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_172: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_261, view_552);  mul_261 = view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    sub_97: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_66, getitem_37);  add_66 = getitem_37 = None
    mul_270: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_18);  sub_97 = None
    mul_271: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_172, primals_149);  primals_149 = None
    mul_272: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_271, 768)
    sum_121: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True)
    mul_273: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_271, mul_270);  mul_271 = None
    sum_122: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2], True);  mul_273 = None
    mul_274: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_270, sum_122);  sum_122 = None
    sub_98: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_272, sum_121);  mul_272 = sum_121 = None
    sub_99: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_274);  sub_98 = mul_274 = None
    div_31: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    mul_275: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_99);  div_31 = sub_99 = None
    mul_276: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_172, mul_270);  mul_270 = None
    sum_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1]);  mul_276 = None
    sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_172, [0, 1]);  add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_553: "f32[2048, 768]" = torch.ops.aten.view.default(mul_275, [2048, 768])
    permute_431: "f32[768, 768]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    mm_87: "f32[2048, 768]" = torch.ops.aten.mm.default(view_553, permute_431);  permute_431 = None
    permute_432: "f32[768, 2048]" = torch.ops.aten.permute.default(view_553, [1, 0])
    mm_88: "f32[768, 768]" = torch.ops.aten.mm.default(permute_432, view_193);  permute_432 = view_193 = None
    permute_433: "f32[768, 768]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_553, [0], True);  view_553 = None
    view_554: "f32[768]" = torch.ops.aten.view.default(sum_125, [768]);  sum_125 = None
    permute_434: "f32[768, 768]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    view_555: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_87, [4, 512, 768]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_556: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_555, [4, 512, 12, 64]);  view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_435: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_166: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_435, memory_format = torch.contiguous_format);  permute_435 = None
    view_557: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_166, [48, 512, 64]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_436: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_73, [0, 2, 1]);  clone_73 = None
    bmm_68: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_436, view_557);  permute_436 = None
    permute_437: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    bmm_69: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_557, permute_437);  view_557 = permute_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_26: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_277: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_69, alias_26);  bmm_69 = None
    sum_126: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [-1], True)
    mul_278: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_26, sum_126);  alias_26 = sum_126 = None
    sub_100: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_438: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_70: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_438, sub_100);  permute_438 = None
    permute_439: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_101, [0, 2, 1]);  permute_101 = None
    bmm_71: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_100, permute_439);  sub_100 = permute_439 = None
    permute_440: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_70, [0, 2, 1]);  bmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_558: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_68, [4, 12, 512, 64]);  bmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_173: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_9, view_558);  tangents_9 = view_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_559: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_440, [4, 12, 512, 64]);  permute_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_174: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_8, view_559);  tangents_8 = view_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_560: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_71, [4, 12, 512, 64]);  bmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_441: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
    clone_167: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_441, memory_format = torch.contiguous_format);  permute_441 = None
    view_561: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_167, [4, 512, 768]);  clone_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_442: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_173, [0, 2, 1, 3]);  add_173 = None
    clone_168: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_442, memory_format = torch.contiguous_format);  permute_442 = None
    view_562: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_168, [4, 512, 768]);  clone_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_563: "f32[2048, 768]" = torch.ops.aten.view.default(view_562, [2048, 768]);  view_562 = None
    permute_443: "f32[768, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_89: "f32[2048, 768]" = torch.ops.aten.mm.default(view_563, permute_443);  permute_443 = None
    permute_444: "f32[768, 2048]" = torch.ops.aten.permute.default(view_563, [1, 0])
    mm_90: "f32[768, 768]" = torch.ops.aten.mm.default(permute_444, view_184);  permute_444 = view_184 = None
    permute_445: "f32[768, 768]" = torch.ops.aten.permute.default(mm_90, [1, 0]);  mm_90 = None
    sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
    view_564: "f32[768]" = torch.ops.aten.view.default(sum_127, [768]);  sum_127 = None
    permute_446: "f32[768, 768]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    view_565: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_89, [4, 512, 768]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_175: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_163, view_565);  add_163 = view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_447: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_174, [0, 2, 1, 3]);  add_174 = None
    clone_169: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
    view_566: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_169, [4, 512, 768]);  clone_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_567: "f32[2048, 768]" = torch.ops.aten.view.default(view_566, [2048, 768]);  view_566 = None
    permute_448: "f32[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_91: "f32[2048, 768]" = torch.ops.aten.mm.default(view_567, permute_448);  permute_448 = None
    permute_449: "f32[768, 2048]" = torch.ops.aten.permute.default(view_567, [1, 0])
    mm_92: "f32[768, 768]" = torch.ops.aten.mm.default(permute_449, view_181);  permute_449 = view_181 = None
    permute_450: "f32[768, 768]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    sum_128: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
    view_568: "f32[768]" = torch.ops.aten.view.default(sum_128, [768]);  sum_128 = None
    permute_451: "f32[768, 768]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    view_569: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_91, [4, 512, 768]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_176: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_175, view_569);  add_175 = view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_279: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_561, 0.125);  view_561 = None
    view_570: "f32[2048, 768]" = torch.ops.aten.view.default(mul_279, [2048, 768]);  mul_279 = None
    permute_452: "f32[768, 768]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    mm_93: "f32[2048, 768]" = torch.ops.aten.mm.default(view_570, permute_452);  permute_452 = None
    permute_453: "f32[768, 2048]" = torch.ops.aten.permute.default(view_570, [1, 0])
    mm_94: "f32[768, 768]" = torch.ops.aten.mm.default(permute_453, view_179);  permute_453 = view_179 = None
    permute_454: "f32[768, 768]" = torch.ops.aten.permute.default(mm_94, [1, 0]);  mm_94 = None
    sum_129: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
    view_571: "f32[768]" = torch.ops.aten.view.default(sum_129, [768]);  sum_129 = None
    permute_455: "f32[768, 768]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
    view_572: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_93, [4, 512, 768]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_177: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_275, view_572);  mul_275 = view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_101: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_35);  add_63 = getitem_35 = None
    mul_280: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_17);  sub_101 = None
    mul_281: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_177, primals_139);  primals_139 = None
    mul_282: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_281, 768)
    sum_130: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_281, [2], True)
    mul_283: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_281, mul_280);  mul_281 = None
    sum_131: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True);  mul_283 = None
    mul_284: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_280, sum_131);  sum_131 = None
    sub_102: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_282, sum_130);  mul_282 = sum_130 = None
    sub_103: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_102, mul_284);  sub_102 = mul_284 = None
    div_32: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    mul_285: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_103);  div_32 = sub_103 = None
    mul_286: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_177, mul_280);  mul_280 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_286, [0, 1]);  mul_286 = None
    sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_177, [0, 1]);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_573: "f32[2048, 768]" = torch.ops.aten.view.default(mul_285, [2048, 768])
    permute_456: "f32[768, 768]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    mm_95: "f32[2048, 768]" = torch.ops.aten.mm.default(view_573, permute_456);  permute_456 = None
    permute_457: "f32[768, 2048]" = torch.ops.aten.permute.default(view_573, [1, 0])
    mm_96: "f32[768, 768]" = torch.ops.aten.mm.default(permute_457, view_177);  permute_457 = view_177 = None
    permute_458: "f32[768, 768]" = torch.ops.aten.permute.default(mm_96, [1, 0]);  mm_96 = None
    sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_573, [0], True);  view_573 = None
    view_574: "f32[768]" = torch.ops.aten.view.default(sum_134, [768]);  sum_134 = None
    permute_459: "f32[768, 768]" = torch.ops.aten.permute.default(permute_458, [1, 0]);  permute_458 = None
    view_575: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_95, [4, 512, 768]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_576: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_575, [4, 512, 12, 64]);  view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_460: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_576, [0, 2, 1, 3]);  view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_170: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
    view_577: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_170, [48, 512, 64]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_461: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_67, [0, 2, 1]);  clone_67 = None
    bmm_72: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_461, view_577);  permute_461 = None
    permute_462: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    bmm_73: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_577, permute_462);  view_577 = permute_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_27: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_287: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_73, alias_27);  bmm_73 = None
    sum_135: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [-1], True)
    mul_288: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_27, sum_135);  alias_27 = sum_135 = None
    sub_104: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_287, mul_288);  mul_287 = mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_578: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(sub_104, [4, 12, 512, 512]);  sub_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_579: "f32[48, 512, 512]" = torch.ops.aten.view.default(view_578, [48, 512, 512]);  view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_463: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
    bmm_74: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_463, view_579);  permute_463 = None
    permute_464: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_92, [0, 2, 1]);  permute_92 = None
    bmm_75: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_579, permute_464);  view_579 = permute_464 = None
    permute_465: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_74, [0, 2, 1]);  bmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_580: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_72, [4, 12, 512, 64]);  bmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_178: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_7, view_580);  tangents_7 = view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_581: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_465, [4, 12, 512, 64]);  permute_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_179: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_6, view_581);  tangents_6 = view_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_582: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_75, [4, 12, 512, 64]);  bmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_466: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None
    clone_171: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_466, memory_format = torch.contiguous_format);  permute_466 = None
    view_583: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_171, [4, 512, 768]);  clone_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_467: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_178, [0, 2, 1, 3]);  add_178 = None
    clone_172: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
    view_584: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_172, [4, 512, 768]);  clone_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_585: "f32[2048, 768]" = torch.ops.aten.view.default(view_584, [2048, 768]);  view_584 = None
    permute_468: "f32[768, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_97: "f32[2048, 768]" = torch.ops.aten.mm.default(view_585, permute_468);  permute_468 = None
    permute_469: "f32[768, 2048]" = torch.ops.aten.permute.default(view_585, [1, 0])
    mm_98: "f32[768, 768]" = torch.ops.aten.mm.default(permute_469, view_166);  permute_469 = view_166 = None
    permute_470: "f32[768, 768]" = torch.ops.aten.permute.default(mm_98, [1, 0]);  mm_98 = None
    sum_136: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
    view_586: "f32[768]" = torch.ops.aten.view.default(sum_136, [768]);  sum_136 = None
    permute_471: "f32[768, 768]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    view_587: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_97, [4, 512, 768]);  mm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_180: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_285, view_587);  mul_285 = view_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_472: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_179, [0, 2, 1, 3]);  add_179 = None
    clone_173: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_472, memory_format = torch.contiguous_format);  permute_472 = None
    view_588: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_173, [4, 512, 768]);  clone_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_589: "f32[2048, 768]" = torch.ops.aten.view.default(view_588, [2048, 768]);  view_588 = None
    permute_473: "f32[768, 768]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_99: "f32[2048, 768]" = torch.ops.aten.mm.default(view_589, permute_473);  permute_473 = None
    permute_474: "f32[768, 2048]" = torch.ops.aten.permute.default(view_589, [1, 0])
    mm_100: "f32[768, 768]" = torch.ops.aten.mm.default(permute_474, view_163);  permute_474 = view_163 = None
    permute_475: "f32[768, 768]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    sum_137: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_589, [0], True);  view_589 = None
    view_590: "f32[768]" = torch.ops.aten.view.default(sum_137, [768]);  sum_137 = None
    permute_476: "f32[768, 768]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
    view_591: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_99, [4, 512, 768]);  mm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_181: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_180, view_591);  add_180 = view_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_289: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_583, 0.125);  view_583 = None
    view_592: "f32[2048, 768]" = torch.ops.aten.view.default(mul_289, [2048, 768]);  mul_289 = None
    permute_477: "f32[768, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_101: "f32[2048, 768]" = torch.ops.aten.mm.default(view_592, permute_477);  permute_477 = None
    permute_478: "f32[768, 2048]" = torch.ops.aten.permute.default(view_592, [1, 0])
    mm_102: "f32[768, 768]" = torch.ops.aten.mm.default(permute_478, view_161);  permute_478 = view_161 = None
    permute_479: "f32[768, 768]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_592, [0], True);  view_592 = None
    view_593: "f32[768]" = torch.ops.aten.view.default(sum_138, [768]);  sum_138 = None
    permute_480: "f32[768, 768]" = torch.ops.aten.permute.default(permute_479, [1, 0]);  permute_479 = None
    view_594: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_101, [4, 512, 768]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_182: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_181, view_594);  add_181 = view_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_105: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_33);  add_59 = getitem_33 = None
    mul_290: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_16);  sub_105 = None
    mul_291: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_182, primals_129);  primals_129 = None
    mul_292: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_291, 768)
    sum_139: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True)
    mul_293: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_291, mul_290);  mul_291 = None
    sum_140: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True);  mul_293 = None
    mul_294: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_290, sum_140);  sum_140 = None
    sub_106: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_292, sum_139);  mul_292 = sum_139 = None
    sub_107: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_106, mul_294);  sub_106 = mul_294 = None
    div_33: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    mul_295: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_107);  div_33 = sub_107 = None
    mul_296: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_182, mul_290);  mul_290 = None
    sum_141: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 1]);  mul_296 = None
    sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_182, [0, 1]);  add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_595: "f32[2048, 768]" = torch.ops.aten.view.default(mul_295, [2048, 768])
    permute_481: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    mm_103: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_595, permute_481);  permute_481 = None
    permute_482: "f32[768, 2048]" = torch.ops.aten.permute.default(view_595, [1, 0])
    mm_104: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_482, view_159);  permute_482 = view_159 = None
    permute_483: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    sum_143: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_595, [0], True);  view_595 = None
    view_596: "f32[768]" = torch.ops.aten.view.default(sum_143, [768]);  sum_143 = None
    permute_484: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_483, [1, 0]);  permute_483 = None
    view_597: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_103, [4, 512, 3072]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_297: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476)
    erf_17: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_297);  mul_297 = None
    add_183: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_298: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_183, 0.5);  add_183 = None
    mul_299: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_158, view_158)
    mul_300: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_299, -0.5);  mul_299 = None
    exp_23: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_300);  mul_300 = None
    mul_301: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_302: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_158, mul_301);  view_158 = mul_301 = None
    add_184: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_298, mul_302);  mul_298 = mul_302 = None
    mul_303: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_597, add_184);  view_597 = add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_598: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_303, [2048, 3072]);  mul_303 = None
    permute_485: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    mm_105: "f32[2048, 768]" = torch.ops.aten.mm.default(view_598, permute_485);  permute_485 = None
    permute_486: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_598, [1, 0])
    mm_106: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_486, view_157);  permute_486 = view_157 = None
    permute_487: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_106, [1, 0]);  mm_106 = None
    sum_144: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_598, [0], True);  view_598 = None
    view_599: "f32[3072]" = torch.ops.aten.view.default(sum_144, [3072]);  sum_144 = None
    permute_488: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_487, [1, 0]);  permute_487 = None
    view_600: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_105, [4, 512, 768]);  mm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_185: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_295, view_600);  mul_295 = view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    sub_108: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_31);  add_55 = getitem_31 = None
    mul_304: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_15);  sub_108 = None
    mul_305: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_185, primals_123);  primals_123 = None
    mul_306: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_305, 768)
    sum_145: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True)
    mul_307: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_305, mul_304);  mul_305 = None
    sum_146: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [2], True);  mul_307 = None
    mul_308: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_304, sum_146);  sum_146 = None
    sub_109: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_306, sum_145);  mul_306 = sum_145 = None
    sub_110: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_109, mul_308);  sub_109 = mul_308 = None
    div_34: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    mul_309: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_110);  div_34 = sub_110 = None
    mul_310: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_185, mul_304);  mul_304 = None
    sum_147: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_310, [0, 1]);  mul_310 = None
    sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_185, [0, 1]);  add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_601: "f32[2048, 768]" = torch.ops.aten.view.default(mul_309, [2048, 768])
    permute_489: "f32[768, 768]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    mm_107: "f32[2048, 768]" = torch.ops.aten.mm.default(view_601, permute_489);  permute_489 = None
    permute_490: "f32[768, 2048]" = torch.ops.aten.permute.default(view_601, [1, 0])
    mm_108: "f32[768, 768]" = torch.ops.aten.mm.default(permute_490, view_155);  permute_490 = view_155 = None
    permute_491: "f32[768, 768]" = torch.ops.aten.permute.default(mm_108, [1, 0]);  mm_108 = None
    sum_149: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_601, [0], True);  view_601 = None
    view_602: "f32[768]" = torch.ops.aten.view.default(sum_149, [768]);  sum_149 = None
    permute_492: "f32[768, 768]" = torch.ops.aten.permute.default(permute_491, [1, 0]);  permute_491 = None
    view_603: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_107, [4, 512, 768]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_604: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_603, [4, 512, 12, 64]);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_493: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_604, [0, 2, 1, 3]);  view_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_174: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_493, memory_format = torch.contiguous_format);  permute_493 = None
    view_605: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_174, [48, 512, 64]);  clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_494: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_59, [0, 2, 1]);  clone_59 = None
    bmm_76: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_494, view_605);  permute_494 = None
    permute_495: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    bmm_77: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_605, permute_495);  view_605 = permute_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_28: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_311: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_77, alias_28);  bmm_77 = None
    sum_150: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [-1], True)
    mul_312: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_28, sum_150);  alias_28 = sum_150 = None
    sub_111: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_311, mul_312);  mul_311 = mul_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_496: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_150, [0, 2, 1]);  view_150 = None
    bmm_78: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_496, sub_111);  permute_496 = None
    permute_497: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_81, [0, 2, 1]);  permute_81 = None
    bmm_79: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_111, permute_497);  sub_111 = permute_497 = None
    permute_498: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_78, [0, 2, 1]);  bmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_606: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_76, [4, 12, 512, 64]);  bmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_186: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_5, view_606);  tangents_5 = view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_607: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_498, [4, 12, 512, 64]);  permute_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_187: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_4, view_607);  tangents_4 = view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_608: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_79, [4, 12, 512, 64]);  bmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_499: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None
    clone_175: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_499, memory_format = torch.contiguous_format);  permute_499 = None
    view_609: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_175, [4, 512, 768]);  clone_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_500: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_186, [0, 2, 1, 3]);  add_186 = None
    clone_176: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
    view_610: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_176, [4, 512, 768]);  clone_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_611: "f32[2048, 768]" = torch.ops.aten.view.default(view_610, [2048, 768]);  view_610 = None
    permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_109: "f32[2048, 768]" = torch.ops.aten.mm.default(view_611, permute_501);  permute_501 = None
    permute_502: "f32[768, 2048]" = torch.ops.aten.permute.default(view_611, [1, 0])
    mm_110: "f32[768, 768]" = torch.ops.aten.mm.default(permute_502, view_146);  permute_502 = view_146 = None
    permute_503: "f32[768, 768]" = torch.ops.aten.permute.default(mm_110, [1, 0]);  mm_110 = None
    sum_151: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_611, [0], True);  view_611 = None
    view_612: "f32[768]" = torch.ops.aten.view.default(sum_151, [768]);  sum_151 = None
    permute_504: "f32[768, 768]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    view_613: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_109, [4, 512, 768]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_188: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_176, view_613);  add_176 = view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_505: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_187, [0, 2, 1, 3]);  add_187 = None
    clone_177: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_505, memory_format = torch.contiguous_format);  permute_505 = None
    view_614: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_177, [4, 512, 768]);  clone_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_615: "f32[2048, 768]" = torch.ops.aten.view.default(view_614, [2048, 768]);  view_614 = None
    permute_506: "f32[768, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_111: "f32[2048, 768]" = torch.ops.aten.mm.default(view_615, permute_506);  permute_506 = None
    permute_507: "f32[768, 2048]" = torch.ops.aten.permute.default(view_615, [1, 0])
    mm_112: "f32[768, 768]" = torch.ops.aten.mm.default(permute_507, view_143);  permute_507 = view_143 = None
    permute_508: "f32[768, 768]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    sum_152: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
    view_616: "f32[768]" = torch.ops.aten.view.default(sum_152, [768]);  sum_152 = None
    permute_509: "f32[768, 768]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    view_617: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_111, [4, 512, 768]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_189: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_188, view_617);  add_188 = view_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_313: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_609, 0.125);  view_609 = None
    view_618: "f32[2048, 768]" = torch.ops.aten.view.default(mul_313, [2048, 768]);  mul_313 = None
    permute_510: "f32[768, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_113: "f32[2048, 768]" = torch.ops.aten.mm.default(view_618, permute_510);  permute_510 = None
    permute_511: "f32[768, 2048]" = torch.ops.aten.permute.default(view_618, [1, 0])
    mm_114: "f32[768, 768]" = torch.ops.aten.mm.default(permute_511, view_141);  permute_511 = view_141 = None
    permute_512: "f32[768, 768]" = torch.ops.aten.permute.default(mm_114, [1, 0]);  mm_114 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_618, [0], True);  view_618 = None
    view_619: "f32[768]" = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
    permute_513: "f32[768, 768]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    view_620: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_113, [4, 512, 768]);  mm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_190: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_309, view_620);  mul_309 = view_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_112: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_29);  add_52 = getitem_29 = None
    mul_314: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_14);  sub_112 = None
    mul_315: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_190, primals_113);  primals_113 = None
    mul_316: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_315, 768)
    sum_154: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True)
    mul_317: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_315, mul_314);  mul_315 = None
    sum_155: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True);  mul_317 = None
    mul_318: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_314, sum_155);  sum_155 = None
    sub_113: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_316, sum_154);  mul_316 = sum_154 = None
    sub_114: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_113, mul_318);  sub_113 = mul_318 = None
    div_35: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    mul_319: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_114);  div_35 = sub_114 = None
    mul_320: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_190, mul_314);  mul_314 = None
    sum_156: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_320, [0, 1]);  mul_320 = None
    sum_157: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_190, [0, 1]);  add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_621: "f32[2048, 768]" = torch.ops.aten.view.default(mul_319, [2048, 768])
    permute_514: "f32[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    mm_115: "f32[2048, 768]" = torch.ops.aten.mm.default(view_621, permute_514);  permute_514 = None
    permute_515: "f32[768, 2048]" = torch.ops.aten.permute.default(view_621, [1, 0])
    mm_116: "f32[768, 768]" = torch.ops.aten.mm.default(permute_515, view_139);  permute_515 = view_139 = None
    permute_516: "f32[768, 768]" = torch.ops.aten.permute.default(mm_116, [1, 0]);  mm_116 = None
    sum_158: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_621, [0], True);  view_621 = None
    view_622: "f32[768]" = torch.ops.aten.view.default(sum_158, [768]);  sum_158 = None
    permute_517: "f32[768, 768]" = torch.ops.aten.permute.default(permute_516, [1, 0]);  permute_516 = None
    view_623: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_115, [4, 512, 768]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_624: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_623, [4, 512, 12, 64]);  view_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_518: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_624, [0, 2, 1, 3]);  view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_178: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
    view_625: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_178, [48, 512, 64]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_519: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_53, [0, 2, 1]);  clone_53 = None
    bmm_80: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_519, view_625);  permute_519 = None
    permute_520: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    bmm_81: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_625, permute_520);  view_625 = permute_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_29: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_321: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_81, alias_29);  bmm_81 = None
    sum_159: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [-1], True)
    mul_322: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_29, sum_159);  alias_29 = sum_159 = None
    sub_115: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_321, mul_322);  mul_321 = mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_626: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(sub_115, [4, 12, 512, 512]);  sub_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_627: "f32[48, 512, 512]" = torch.ops.aten.view.default(view_626, [48, 512, 512]);  view_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_521: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
    bmm_82: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_521, view_627);  permute_521 = None
    permute_522: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_72, [0, 2, 1]);  permute_72 = None
    bmm_83: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_627, permute_522);  view_627 = permute_522 = None
    permute_523: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_82, [0, 2, 1]);  bmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_628: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_80, [4, 12, 512, 64]);  bmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    add_191: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_3, view_628);  tangents_3 = view_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_629: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_523, [4, 12, 512, 64]);  permute_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    add_192: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_2, view_629);  tangents_2 = view_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_630: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_83, [4, 12, 512, 64]);  bmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_524: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
    clone_179: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_631: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_179, [4, 512, 768]);  clone_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_525: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_191, [0, 2, 1, 3]);  add_191 = None
    clone_180: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
    view_632: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_180, [4, 512, 768]);  clone_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_633: "f32[2048, 768]" = torch.ops.aten.view.default(view_632, [2048, 768]);  view_632 = None
    permute_526: "f32[768, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    mm_117: "f32[2048, 768]" = torch.ops.aten.mm.default(view_633, permute_526);  permute_526 = None
    permute_527: "f32[768, 2048]" = torch.ops.aten.permute.default(view_633, [1, 0])
    mm_118: "f32[768, 768]" = torch.ops.aten.mm.default(permute_527, view_128);  permute_527 = view_128 = None
    permute_528: "f32[768, 768]" = torch.ops.aten.permute.default(mm_118, [1, 0]);  mm_118 = None
    sum_160: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_633, [0], True);  view_633 = None
    view_634: "f32[768]" = torch.ops.aten.view.default(sum_160, [768]);  sum_160 = None
    permute_529: "f32[768, 768]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
    view_635: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_117, [4, 512, 768]);  mm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_193: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_319, view_635);  mul_319 = view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_530: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_192, [0, 2, 1, 3]);  add_192 = None
    clone_181: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_530, memory_format = torch.contiguous_format);  permute_530 = None
    view_636: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_181, [4, 512, 768]);  clone_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_637: "f32[2048, 768]" = torch.ops.aten.view.default(view_636, [2048, 768]);  view_636 = None
    permute_531: "f32[768, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_119: "f32[2048, 768]" = torch.ops.aten.mm.default(view_637, permute_531);  permute_531 = None
    permute_532: "f32[768, 2048]" = torch.ops.aten.permute.default(view_637, [1, 0])
    mm_120: "f32[768, 768]" = torch.ops.aten.mm.default(permute_532, view_125);  permute_532 = view_125 = None
    permute_533: "f32[768, 768]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    sum_161: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_637, [0], True);  view_637 = None
    view_638: "f32[768]" = torch.ops.aten.view.default(sum_161, [768]);  sum_161 = None
    permute_534: "f32[768, 768]" = torch.ops.aten.permute.default(permute_533, [1, 0]);  permute_533 = None
    view_639: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_119, [4, 512, 768]);  mm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_194: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_193, view_639);  add_193 = view_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_323: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_631, 0.125);  view_631 = None
    view_640: "f32[2048, 768]" = torch.ops.aten.view.default(mul_323, [2048, 768]);  mul_323 = None
    permute_535: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_121: "f32[2048, 768]" = torch.ops.aten.mm.default(view_640, permute_535);  permute_535 = None
    permute_536: "f32[768, 2048]" = torch.ops.aten.permute.default(view_640, [1, 0])
    mm_122: "f32[768, 768]" = torch.ops.aten.mm.default(permute_536, view_123);  permute_536 = view_123 = None
    permute_537: "f32[768, 768]" = torch.ops.aten.permute.default(mm_122, [1, 0]);  mm_122 = None
    sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_640, [0], True);  view_640 = None
    view_641: "f32[768]" = torch.ops.aten.view.default(sum_162, [768]);  sum_162 = None
    permute_538: "f32[768, 768]" = torch.ops.aten.permute.default(permute_537, [1, 0]);  permute_537 = None
    view_642: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_121, [4, 512, 768]);  mm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_195: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_194, view_642);  add_194 = view_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1075, code: hidden_states = self.layernorm_embedding(hidden_states)
    sub_116: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_27);  add_48 = getitem_27 = None
    mul_324: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_13);  sub_116 = None
    mul_325: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_195, primals_103);  primals_103 = None
    mul_326: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_325, 768)
    sum_163: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)
    mul_327: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_325, mul_324);  mul_325 = None
    sum_164: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
    mul_328: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_324, sum_164);  sum_164 = None
    sub_117: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_326, sum_163);  mul_326 = sum_163 = None
    sub_118: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_117, mul_328);  sub_117 = mul_328 = None
    div_36: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    mul_329: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_118);  div_36 = sub_118 = None
    mul_330: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_195, mul_324);  mul_324 = None
    sum_165: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
    sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_195, [0, 1]);  add_195 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    eq: "b8[4, 512]" = torch.ops.aten.eq.Scalar(add_47, -1)
    unsqueeze_4: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[4, 512, 768]" = torch.ops.aten.where.self(unsqueeze_4, scalar_tensor_1, mul_329);  unsqueeze_4 = scalar_tensor_1 = None
    full_1: "f32[1026, 768]" = torch.ops.aten.full.default([1026, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[1026, 768]" = torch.ops.aten._unsafe_index_put.default(full_1, [add_47], where_1, True);  full_1 = add_47 = where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1059, code: inputs_embeds = self.embed_tokens(input) * self.embed_scale
    mul_331: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_329, 1.0);  mul_329 = None
    eq_1: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_264, 1)
    unsqueeze_5: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[4, 512, 768]" = torch.ops.aten.where.self(unsqueeze_5, scalar_tensor_2, mul_331);  unsqueeze_5 = scalar_tensor_2 = mul_331 = None
    full_2: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[50265, 768]" = torch.ops.aten._unsafe_index_put.default(full_2, [primals_264], where_2, True);  full_2 = primals_264 = where_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_119: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_25);  add_43 = getitem_25 = None
    mul_332: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_12);  sub_119 = None
    mul_333: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, primals_100);  primals_100 = None
    mul_334: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_333, 768)
    sum_167: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [2], True)
    mul_335: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_333, mul_332);  mul_333 = None
    sum_168: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True);  mul_335 = None
    mul_336: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_332, sum_168);  sum_168 = None
    sub_120: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_334, sum_167);  mul_334 = sum_167 = None
    sub_121: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_120, mul_336);  sub_120 = mul_336 = None
    div_37: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    mul_337: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_121);  div_37 = sub_121 = None
    mul_338: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, mul_332);  mul_332 = None
    sum_169: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_338, [0, 1]);  mul_338 = None
    sum_170: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_189, [0, 1]);  add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_643: "f32[2048, 768]" = torch.ops.aten.view.default(mul_337, [2048, 768])
    permute_539: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_123: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_643, permute_539);  permute_539 = None
    permute_540: "f32[768, 2048]" = torch.ops.aten.permute.default(view_643, [1, 0])
    mm_124: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_540, view_119);  permute_540 = view_119 = None
    permute_541: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    sum_171: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_643, [0], True);  view_643 = None
    view_644: "f32[768]" = torch.ops.aten.view.default(sum_171, [768]);  sum_171 = None
    permute_542: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_541, [1, 0]);  permute_541 = None
    view_645: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_123, [4, 512, 3072]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_339: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
    erf_18: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_339);  mul_339 = None
    add_196: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_340: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_196, 0.5);  add_196 = None
    mul_341: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_118, view_118)
    mul_342: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_341, -0.5);  mul_341 = None
    exp_24: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_342);  mul_342 = None
    mul_343: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_344: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_118, mul_343);  view_118 = mul_343 = None
    add_197: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_340, mul_344);  mul_340 = mul_344 = None
    mul_345: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_645, add_197);  view_645 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_646: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_345, [2048, 3072]);  mul_345 = None
    permute_543: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_125: "f32[2048, 768]" = torch.ops.aten.mm.default(view_646, permute_543);  permute_543 = None
    permute_544: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_646, [1, 0])
    mm_126: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_544, view_117);  permute_544 = view_117 = None
    permute_545: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_126, [1, 0]);  mm_126 = None
    sum_172: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_646, [0], True);  view_646 = None
    view_647: "f32[3072]" = torch.ops.aten.view.default(sum_172, [3072]);  sum_172 = None
    permute_546: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_545, [1, 0]);  permute_545 = None
    view_648: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_125, [4, 512, 768]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_198: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_337, view_648);  mul_337 = view_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_122: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_23);  add_39 = getitem_23 = None
    mul_346: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_11);  sub_122 = None
    mul_347: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_198, primals_94);  primals_94 = None
    mul_348: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_347, 768)
    sum_173: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
    mul_349: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_347, mul_346);  mul_347 = None
    sum_174: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
    mul_350: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_346, sum_174);  sum_174 = None
    sub_123: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_348, sum_173);  mul_348 = sum_173 = None
    sub_124: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_123, mul_350);  sub_123 = mul_350 = None
    div_38: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_351: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_124);  div_38 = sub_124 = None
    mul_352: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_198, mul_346);  mul_346 = None
    sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_198, [0, 1]);  add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_649: "f32[2048, 768]" = torch.ops.aten.view.default(mul_351, [2048, 768])
    permute_547: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_127: "f32[2048, 768]" = torch.ops.aten.mm.default(view_649, permute_547);  permute_547 = None
    permute_548: "f32[768, 2048]" = torch.ops.aten.permute.default(view_649, [1, 0])
    mm_128: "f32[768, 768]" = torch.ops.aten.mm.default(permute_548, view_115);  permute_548 = view_115 = None
    permute_549: "f32[768, 768]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    sum_177: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_649, [0], True);  view_649 = None
    view_650: "f32[768]" = torch.ops.aten.view.default(sum_177, [768]);  sum_177 = None
    permute_550: "f32[768, 768]" = torch.ops.aten.permute.default(permute_549, [1, 0]);  permute_549 = None
    view_651: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_127, [4, 512, 768]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_652: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_651, [4, 512, 12, 64]);  view_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_551: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_652, [0, 2, 1, 3]);  view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_182: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
    view_653: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_182, [48, 512, 64]);  clone_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_552: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_44, [0, 2, 1]);  clone_44 = None
    bmm_84: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_552, view_653);  permute_552 = None
    permute_553: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
    bmm_85: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_653, permute_553);  view_653 = permute_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_30: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_353: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_85, alias_30);  bmm_85 = None
    sum_178: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [-1], True)
    mul_354: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_30, sum_178);  alias_30 = sum_178 = None
    sub_125: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_353, mul_354);  mul_353 = mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_554: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
    bmm_86: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_554, sub_125);  permute_554 = None
    permute_555: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_61, [0, 2, 1]);  permute_61 = None
    bmm_87: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_125, permute_555);  sub_125 = permute_555 = None
    permute_556: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_86, [0, 2, 1]);  bmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_654: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_84, [4, 12, 512, 64]);  bmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_655: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_556, [4, 12, 512, 64]);  permute_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_656: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_87, [4, 12, 512, 64]);  bmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_557: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_656, [0, 2, 1, 3]);  view_656 = None
    clone_183: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_557, memory_format = torch.contiguous_format);  permute_557 = None
    view_657: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_183, [4, 512, 768]);  clone_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_558: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    clone_184: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_558, memory_format = torch.contiguous_format);  permute_558 = None
    view_658: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_184, [4, 512, 768]);  clone_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_659: "f32[2048, 768]" = torch.ops.aten.view.default(view_658, [2048, 768]);  view_658 = None
    permute_559: "f32[768, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_129: "f32[2048, 768]" = torch.ops.aten.mm.default(view_659, permute_559);  permute_559 = None
    permute_560: "f32[768, 2048]" = torch.ops.aten.permute.default(view_659, [1, 0])
    mm_130: "f32[768, 768]" = torch.ops.aten.mm.default(permute_560, view_106);  permute_560 = view_106 = None
    permute_561: "f32[768, 768]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    sum_179: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_659, [0], True);  view_659 = None
    view_660: "f32[768]" = torch.ops.aten.view.default(sum_179, [768]);  sum_179 = None
    permute_562: "f32[768, 768]" = torch.ops.aten.permute.default(permute_561, [1, 0]);  permute_561 = None
    view_661: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_129, [4, 512, 768]);  mm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_199: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_351, view_661);  mul_351 = view_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_563: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_655, [0, 2, 1, 3]);  view_655 = None
    view_662: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_563, [4, 512, 768]);  permute_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    clone_185: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_662, memory_format = torch.contiguous_format);  view_662 = None
    view_663: "f32[2048, 768]" = torch.ops.aten.view.default(clone_185, [2048, 768]);  clone_185 = None
    permute_564: "f32[768, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_131: "f32[2048, 768]" = torch.ops.aten.mm.default(view_663, permute_564);  permute_564 = None
    permute_565: "f32[768, 2048]" = torch.ops.aten.permute.default(view_663, [1, 0])
    mm_132: "f32[768, 768]" = torch.ops.aten.mm.default(permute_565, view_103);  permute_565 = view_103 = None
    permute_566: "f32[768, 768]" = torch.ops.aten.permute.default(mm_132, [1, 0]);  mm_132 = None
    sum_180: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_663, [0], True);  view_663 = None
    view_664: "f32[768]" = torch.ops.aten.view.default(sum_180, [768]);  sum_180 = None
    permute_567: "f32[768, 768]" = torch.ops.aten.permute.default(permute_566, [1, 0]);  permute_566 = None
    view_665: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_131, [4, 512, 768]);  mm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_200: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_199, view_665);  add_199 = view_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_355: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_657, 0.125);  view_657 = None
    view_666: "f32[2048, 768]" = torch.ops.aten.view.default(mul_355, [2048, 768]);  mul_355 = None
    permute_568: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_133: "f32[2048, 768]" = torch.ops.aten.mm.default(view_666, permute_568);  permute_568 = None
    permute_569: "f32[768, 2048]" = torch.ops.aten.permute.default(view_666, [1, 0])
    mm_134: "f32[768, 768]" = torch.ops.aten.mm.default(permute_569, view_101);  permute_569 = view_101 = None
    permute_570: "f32[768, 768]" = torch.ops.aten.permute.default(mm_134, [1, 0]);  mm_134 = None
    sum_181: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_666, [0], True);  view_666 = None
    view_667: "f32[768]" = torch.ops.aten.view.default(sum_181, [768]);  sum_181 = None
    permute_571: "f32[768, 768]" = torch.ops.aten.permute.default(permute_570, [1, 0]);  permute_570 = None
    view_668: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_133, [4, 512, 768]);  mm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_201: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_200, view_668);  add_200 = view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_126: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_21);  add_36 = getitem_21 = None
    mul_356: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_10);  sub_126 = None
    mul_357: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_201, primals_84);  primals_84 = None
    mul_358: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_357, 768)
    sum_182: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True)
    mul_359: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_357, mul_356);  mul_357 = None
    sum_183: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_359, [2], True);  mul_359 = None
    mul_360: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_356, sum_183);  sum_183 = None
    sub_127: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_358, sum_182);  mul_358 = sum_182 = None
    sub_128: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_127, mul_360);  sub_127 = mul_360 = None
    div_39: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    mul_361: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_128);  div_39 = sub_128 = None
    mul_362: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_201, mul_356);  mul_356 = None
    sum_184: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 1]);  mul_362 = None
    sum_185: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_201, [0, 1]);  add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_669: "f32[2048, 768]" = torch.ops.aten.view.default(mul_361, [2048, 768])
    permute_572: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_135: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_669, permute_572);  permute_572 = None
    permute_573: "f32[768, 2048]" = torch.ops.aten.permute.default(view_669, [1, 0])
    mm_136: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_573, view_99);  permute_573 = view_99 = None
    permute_574: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_669, [0], True);  view_669 = None
    view_670: "f32[768]" = torch.ops.aten.view.default(sum_186, [768]);  sum_186 = None
    permute_575: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_574, [1, 0]);  permute_574 = None
    view_671: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_135, [4, 512, 3072]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_363: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
    erf_19: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_363);  mul_363 = None
    add_202: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_364: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_202, 0.5);  add_202 = None
    mul_365: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_98, view_98)
    mul_366: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_365, -0.5);  mul_365 = None
    exp_25: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_366);  mul_366 = None
    mul_367: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_368: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_98, mul_367);  view_98 = mul_367 = None
    add_203: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_364, mul_368);  mul_364 = mul_368 = None
    mul_369: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_671, add_203);  view_671 = add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_672: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_369, [2048, 3072]);  mul_369 = None
    permute_576: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_137: "f32[2048, 768]" = torch.ops.aten.mm.default(view_672, permute_576);  permute_576 = None
    permute_577: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_672, [1, 0])
    mm_138: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_577, view_97);  permute_577 = view_97 = None
    permute_578: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_138, [1, 0]);  mm_138 = None
    sum_187: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_672, [0], True);  view_672 = None
    view_673: "f32[3072]" = torch.ops.aten.view.default(sum_187, [3072]);  sum_187 = None
    permute_579: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_578, [1, 0]);  permute_578 = None
    view_674: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_137, [4, 512, 768]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_204: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_361, view_674);  mul_361 = view_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_129: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_19);  add_32 = getitem_19 = None
    mul_370: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_129, rsqrt_9);  sub_129 = None
    mul_371: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_204, primals_78);  primals_78 = None
    mul_372: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_371, 768)
    sum_188: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True)
    mul_373: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_371, mul_370);  mul_371 = None
    sum_189: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True);  mul_373 = None
    mul_374: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_370, sum_189);  sum_189 = None
    sub_130: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_372, sum_188);  mul_372 = sum_188 = None
    sub_131: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_130, mul_374);  sub_130 = mul_374 = None
    div_40: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_375: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_131);  div_40 = sub_131 = None
    mul_376: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_204, mul_370);  mul_370 = None
    sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1]);  mul_376 = None
    sum_191: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_204, [0, 1]);  add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_675: "f32[2048, 768]" = torch.ops.aten.view.default(mul_375, [2048, 768])
    permute_580: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    mm_139: "f32[2048, 768]" = torch.ops.aten.mm.default(view_675, permute_580);  permute_580 = None
    permute_581: "f32[768, 2048]" = torch.ops.aten.permute.default(view_675, [1, 0])
    mm_140: "f32[768, 768]" = torch.ops.aten.mm.default(permute_581, view_95);  permute_581 = view_95 = None
    permute_582: "f32[768, 768]" = torch.ops.aten.permute.default(mm_140, [1, 0]);  mm_140 = None
    sum_192: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_675, [0], True);  view_675 = None
    view_676: "f32[768]" = torch.ops.aten.view.default(sum_192, [768]);  sum_192 = None
    permute_583: "f32[768, 768]" = torch.ops.aten.permute.default(permute_582, [1, 0]);  permute_582 = None
    view_677: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_139, [4, 512, 768]);  mm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_678: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_677, [4, 512, 12, 64]);  view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_584: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_678, [0, 2, 1, 3]);  view_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_186: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_584, memory_format = torch.contiguous_format);  permute_584 = None
    view_679: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_186, [48, 512, 64]);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_585: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_36, [0, 2, 1]);  clone_36 = None
    bmm_88: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_585, view_679);  permute_585 = None
    permute_586: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
    bmm_89: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_679, permute_586);  view_679 = permute_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_31: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_377: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_89, alias_31);  bmm_89 = None
    sum_193: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [-1], True)
    mul_378: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_31, sum_193);  alias_31 = sum_193 = None
    sub_132: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_587: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
    bmm_90: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_587, sub_132);  permute_587 = None
    permute_588: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_50, [0, 2, 1]);  permute_50 = None
    bmm_91: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_132, permute_588);  sub_132 = permute_588 = None
    permute_589: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_90, [0, 2, 1]);  bmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_680: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_88, [4, 12, 512, 64]);  bmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_681: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_589, [4, 12, 512, 64]);  permute_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_682: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_91, [4, 12, 512, 64]);  bmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_590: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_682, [0, 2, 1, 3]);  view_682 = None
    clone_187: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_590, memory_format = torch.contiguous_format);  permute_590 = None
    view_683: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_187, [4, 512, 768]);  clone_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_591: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_680, [0, 2, 1, 3]);  view_680 = None
    clone_188: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_591, memory_format = torch.contiguous_format);  permute_591 = None
    view_684: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_188, [4, 512, 768]);  clone_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_685: "f32[2048, 768]" = torch.ops.aten.view.default(view_684, [2048, 768]);  view_684 = None
    permute_592: "f32[768, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_141: "f32[2048, 768]" = torch.ops.aten.mm.default(view_685, permute_592);  permute_592 = None
    permute_593: "f32[768, 2048]" = torch.ops.aten.permute.default(view_685, [1, 0])
    mm_142: "f32[768, 768]" = torch.ops.aten.mm.default(permute_593, view_86);  permute_593 = view_86 = None
    permute_594: "f32[768, 768]" = torch.ops.aten.permute.default(mm_142, [1, 0]);  mm_142 = None
    sum_194: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_685, [0], True);  view_685 = None
    view_686: "f32[768]" = torch.ops.aten.view.default(sum_194, [768]);  sum_194 = None
    permute_595: "f32[768, 768]" = torch.ops.aten.permute.default(permute_594, [1, 0]);  permute_594 = None
    view_687: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_141, [4, 512, 768]);  mm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_205: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_375, view_687);  mul_375 = view_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_596: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_681, [0, 2, 1, 3]);  view_681 = None
    view_688: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_596, [4, 512, 768]);  permute_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    clone_189: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_688, memory_format = torch.contiguous_format);  view_688 = None
    view_689: "f32[2048, 768]" = torch.ops.aten.view.default(clone_189, [2048, 768]);  clone_189 = None
    permute_597: "f32[768, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_143: "f32[2048, 768]" = torch.ops.aten.mm.default(view_689, permute_597);  permute_597 = None
    permute_598: "f32[768, 2048]" = torch.ops.aten.permute.default(view_689, [1, 0])
    mm_144: "f32[768, 768]" = torch.ops.aten.mm.default(permute_598, view_83);  permute_598 = view_83 = None
    permute_599: "f32[768, 768]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    sum_195: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_689, [0], True);  view_689 = None
    view_690: "f32[768]" = torch.ops.aten.view.default(sum_195, [768]);  sum_195 = None
    permute_600: "f32[768, 768]" = torch.ops.aten.permute.default(permute_599, [1, 0]);  permute_599 = None
    view_691: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_143, [4, 512, 768]);  mm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_206: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_205, view_691);  add_205 = view_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_379: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_683, 0.125);  view_683 = None
    view_692: "f32[2048, 768]" = torch.ops.aten.view.default(mul_379, [2048, 768]);  mul_379 = None
    permute_601: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_145: "f32[2048, 768]" = torch.ops.aten.mm.default(view_692, permute_601);  permute_601 = None
    permute_602: "f32[768, 2048]" = torch.ops.aten.permute.default(view_692, [1, 0])
    mm_146: "f32[768, 768]" = torch.ops.aten.mm.default(permute_602, view_81);  permute_602 = view_81 = None
    permute_603: "f32[768, 768]" = torch.ops.aten.permute.default(mm_146, [1, 0]);  mm_146 = None
    sum_196: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_692, [0], True);  view_692 = None
    view_693: "f32[768]" = torch.ops.aten.view.default(sum_196, [768]);  sum_196 = None
    permute_604: "f32[768, 768]" = torch.ops.aten.permute.default(permute_603, [1, 0]);  permute_603 = None
    view_694: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_145, [4, 512, 768]);  mm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_207: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_206, view_694);  add_206 = view_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_133: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_29, getitem_17);  add_29 = getitem_17 = None
    mul_380: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_8);  sub_133 = None
    mul_381: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_207, primals_68);  primals_68 = None
    mul_382: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_381, 768)
    sum_197: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True)
    mul_383: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_381, mul_380);  mul_381 = None
    sum_198: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
    mul_384: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_380, sum_198);  sum_198 = None
    sub_134: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_382, sum_197);  mul_382 = sum_197 = None
    sub_135: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_134, mul_384);  sub_134 = mul_384 = None
    div_41: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    mul_385: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_41, sub_135);  div_41 = sub_135 = None
    mul_386: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_207, mul_380);  mul_380 = None
    sum_199: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1]);  mul_386 = None
    sum_200: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_207, [0, 1]);  add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_695: "f32[2048, 768]" = torch.ops.aten.view.default(mul_385, [2048, 768])
    permute_605: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_147: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_695, permute_605);  permute_605 = None
    permute_606: "f32[768, 2048]" = torch.ops.aten.permute.default(view_695, [1, 0])
    mm_148: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_606, view_79);  permute_606 = view_79 = None
    permute_607: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    sum_201: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_695, [0], True);  view_695 = None
    view_696: "f32[768]" = torch.ops.aten.view.default(sum_201, [768]);  sum_201 = None
    permute_608: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_607, [1, 0]);  permute_607 = None
    view_697: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_147, [4, 512, 3072]);  mm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_387: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476)
    erf_20: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_387);  mul_387 = None
    add_208: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_388: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_208, 0.5);  add_208 = None
    mul_389: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_78, view_78)
    mul_390: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_389, -0.5);  mul_389 = None
    exp_26: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_390);  mul_390 = None
    mul_391: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_392: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_78, mul_391);  view_78 = mul_391 = None
    add_209: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_388, mul_392);  mul_388 = mul_392 = None
    mul_393: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_697, add_209);  view_697 = add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_698: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_393, [2048, 3072]);  mul_393 = None
    permute_609: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_149: "f32[2048, 768]" = torch.ops.aten.mm.default(view_698, permute_609);  permute_609 = None
    permute_610: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_698, [1, 0])
    mm_150: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_610, view_77);  permute_610 = view_77 = None
    permute_611: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_150, [1, 0]);  mm_150 = None
    sum_202: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_698, [0], True);  view_698 = None
    view_699: "f32[3072]" = torch.ops.aten.view.default(sum_202, [3072]);  sum_202 = None
    permute_612: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_611, [1, 0]);  permute_611 = None
    view_700: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_149, [4, 512, 768]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_210: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_385, view_700);  mul_385 = view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_136: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_15);  add_25 = getitem_15 = None
    mul_394: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_136, rsqrt_7);  sub_136 = None
    mul_395: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_210, primals_62);  primals_62 = None
    mul_396: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_395, 768)
    sum_203: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
    mul_397: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_395, mul_394);  mul_395 = None
    sum_204: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
    mul_398: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_394, sum_204);  sum_204 = None
    sub_137: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_396, sum_203);  mul_396 = sum_203 = None
    sub_138: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_137, mul_398);  sub_137 = mul_398 = None
    div_42: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_399: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_138);  div_42 = sub_138 = None
    mul_400: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_210, mul_394);  mul_394 = None
    sum_205: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
    sum_206: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_210, [0, 1]);  add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_701: "f32[2048, 768]" = torch.ops.aten.view.default(mul_399, [2048, 768])
    permute_613: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_151: "f32[2048, 768]" = torch.ops.aten.mm.default(view_701, permute_613);  permute_613 = None
    permute_614: "f32[768, 2048]" = torch.ops.aten.permute.default(view_701, [1, 0])
    mm_152: "f32[768, 768]" = torch.ops.aten.mm.default(permute_614, view_75);  permute_614 = view_75 = None
    permute_615: "f32[768, 768]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    sum_207: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_701, [0], True);  view_701 = None
    view_702: "f32[768]" = torch.ops.aten.view.default(sum_207, [768]);  sum_207 = None
    permute_616: "f32[768, 768]" = torch.ops.aten.permute.default(permute_615, [1, 0]);  permute_615 = None
    view_703: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_151, [4, 512, 768]);  mm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_704: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_703, [4, 512, 12, 64]);  view_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_617: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_704, [0, 2, 1, 3]);  view_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_190: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_617, memory_format = torch.contiguous_format);  permute_617 = None
    view_705: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_190, [48, 512, 64]);  clone_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_618: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_28, [0, 2, 1]);  clone_28 = None
    bmm_92: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_618, view_705);  permute_618 = None
    permute_619: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None
    bmm_93: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_705, permute_619);  view_705 = permute_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_32: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_401: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_93, alias_32);  bmm_93 = None
    sum_208: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [-1], True)
    mul_402: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_32, sum_208);  alias_32 = sum_208 = None
    sub_139: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_620: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
    bmm_94: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_620, sub_139);  permute_620 = None
    permute_621: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_39, [0, 2, 1]);  permute_39 = None
    bmm_95: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_139, permute_621);  sub_139 = permute_621 = None
    permute_622: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_94, [0, 2, 1]);  bmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_706: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_92, [4, 12, 512, 64]);  bmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_707: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_622, [4, 12, 512, 64]);  permute_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_708: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_95, [4, 12, 512, 64]);  bmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_623: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_708, [0, 2, 1, 3]);  view_708 = None
    clone_191: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_623, memory_format = torch.contiguous_format);  permute_623 = None
    view_709: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_191, [4, 512, 768]);  clone_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_624: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_706, [0, 2, 1, 3]);  view_706 = None
    clone_192: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_624, memory_format = torch.contiguous_format);  permute_624 = None
    view_710: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_192, [4, 512, 768]);  clone_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_711: "f32[2048, 768]" = torch.ops.aten.view.default(view_710, [2048, 768]);  view_710 = None
    permute_625: "f32[768, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_153: "f32[2048, 768]" = torch.ops.aten.mm.default(view_711, permute_625);  permute_625 = None
    permute_626: "f32[768, 2048]" = torch.ops.aten.permute.default(view_711, [1, 0])
    mm_154: "f32[768, 768]" = torch.ops.aten.mm.default(permute_626, view_66);  permute_626 = view_66 = None
    permute_627: "f32[768, 768]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
    sum_209: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_711, [0], True);  view_711 = None
    view_712: "f32[768]" = torch.ops.aten.view.default(sum_209, [768]);  sum_209 = None
    permute_628: "f32[768, 768]" = torch.ops.aten.permute.default(permute_627, [1, 0]);  permute_627 = None
    view_713: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_153, [4, 512, 768]);  mm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_211: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_399, view_713);  mul_399 = view_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_629: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_707, [0, 2, 1, 3]);  view_707 = None
    view_714: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_629, [4, 512, 768]);  permute_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    clone_193: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_714, memory_format = torch.contiguous_format);  view_714 = None
    view_715: "f32[2048, 768]" = torch.ops.aten.view.default(clone_193, [2048, 768]);  clone_193 = None
    permute_630: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_155: "f32[2048, 768]" = torch.ops.aten.mm.default(view_715, permute_630);  permute_630 = None
    permute_631: "f32[768, 2048]" = torch.ops.aten.permute.default(view_715, [1, 0])
    mm_156: "f32[768, 768]" = torch.ops.aten.mm.default(permute_631, view_63);  permute_631 = view_63 = None
    permute_632: "f32[768, 768]" = torch.ops.aten.permute.default(mm_156, [1, 0]);  mm_156 = None
    sum_210: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
    view_716: "f32[768]" = torch.ops.aten.view.default(sum_210, [768]);  sum_210 = None
    permute_633: "f32[768, 768]" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None
    view_717: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_155, [4, 512, 768]);  mm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_212: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_211, view_717);  add_211 = view_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_403: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_709, 0.125);  view_709 = None
    view_718: "f32[2048, 768]" = torch.ops.aten.view.default(mul_403, [2048, 768]);  mul_403 = None
    permute_634: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_157: "f32[2048, 768]" = torch.ops.aten.mm.default(view_718, permute_634);  permute_634 = None
    permute_635: "f32[768, 2048]" = torch.ops.aten.permute.default(view_718, [1, 0])
    mm_158: "f32[768, 768]" = torch.ops.aten.mm.default(permute_635, view_61);  permute_635 = view_61 = None
    permute_636: "f32[768, 768]" = torch.ops.aten.permute.default(mm_158, [1, 0]);  mm_158 = None
    sum_211: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_718, [0], True);  view_718 = None
    view_719: "f32[768]" = torch.ops.aten.view.default(sum_211, [768]);  sum_211 = None
    permute_637: "f32[768, 768]" = torch.ops.aten.permute.default(permute_636, [1, 0]);  permute_636 = None
    view_720: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_157, [4, 512, 768]);  mm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_213: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_212, view_720);  add_212 = view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_140: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_13);  add_22 = getitem_13 = None
    mul_404: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_6);  sub_140 = None
    mul_405: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_213, primals_52);  primals_52 = None
    mul_406: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_405, 768)
    sum_212: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_405, [2], True)
    mul_407: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_405, mul_404);  mul_405 = None
    sum_213: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_407, [2], True);  mul_407 = None
    mul_408: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_404, sum_213);  sum_213 = None
    sub_141: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_406, sum_212);  mul_406 = sum_212 = None
    sub_142: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_141, mul_408);  sub_141 = mul_408 = None
    div_43: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    mul_409: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_142);  div_43 = sub_142 = None
    mul_410: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_213, mul_404);  mul_404 = None
    sum_214: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_410, [0, 1]);  mul_410 = None
    sum_215: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_213, [0, 1]);  add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_721: "f32[2048, 768]" = torch.ops.aten.view.default(mul_409, [2048, 768])
    permute_638: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_159: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_721, permute_638);  permute_638 = None
    permute_639: "f32[768, 2048]" = torch.ops.aten.permute.default(view_721, [1, 0])
    mm_160: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_639, view_59);  permute_639 = view_59 = None
    permute_640: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    sum_216: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_721, [0], True);  view_721 = None
    view_722: "f32[768]" = torch.ops.aten.view.default(sum_216, [768]);  sum_216 = None
    permute_641: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_640, [1, 0]);  permute_640 = None
    view_723: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_159, [4, 512, 3072]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_411: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
    erf_21: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_411);  mul_411 = None
    add_214: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_412: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_214, 0.5);  add_214 = None
    mul_413: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, view_58)
    mul_414: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_413, -0.5);  mul_413 = None
    exp_27: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_414);  mul_414 = None
    mul_415: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_416: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, mul_415);  view_58 = mul_415 = None
    add_215: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_412, mul_416);  mul_412 = mul_416 = None
    mul_417: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_723, add_215);  view_723 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_724: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_417, [2048, 3072]);  mul_417 = None
    permute_642: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_161: "f32[2048, 768]" = torch.ops.aten.mm.default(view_724, permute_642);  permute_642 = None
    permute_643: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_724, [1, 0])
    mm_162: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_643, view_57);  permute_643 = view_57 = None
    permute_644: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_162, [1, 0]);  mm_162 = None
    sum_217: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_724, [0], True);  view_724 = None
    view_725: "f32[3072]" = torch.ops.aten.view.default(sum_217, [3072]);  sum_217 = None
    permute_645: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_644, [1, 0]);  permute_644 = None
    view_726: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_161, [4, 512, 768]);  mm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_216: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_409, view_726);  mul_409 = view_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_143: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_18, getitem_11);  add_18 = getitem_11 = None
    mul_418: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_143, rsqrt_5);  sub_143 = None
    mul_419: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_216, primals_46);  primals_46 = None
    mul_420: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_419, 768)
    sum_218: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [2], True)
    mul_421: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_419, mul_418);  mul_419 = None
    sum_219: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True);  mul_421 = None
    mul_422: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_418, sum_219);  sum_219 = None
    sub_144: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_420, sum_218);  mul_420 = sum_218 = None
    sub_145: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_144, mul_422);  sub_144 = mul_422 = None
    div_44: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_423: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_44, sub_145);  div_44 = sub_145 = None
    mul_424: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_216, mul_418);  mul_418 = None
    sum_220: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_424, [0, 1]);  mul_424 = None
    sum_221: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_216, [0, 1]);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_727: "f32[2048, 768]" = torch.ops.aten.view.default(mul_423, [2048, 768])
    permute_646: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_163: "f32[2048, 768]" = torch.ops.aten.mm.default(view_727, permute_646);  permute_646 = None
    permute_647: "f32[768, 2048]" = torch.ops.aten.permute.default(view_727, [1, 0])
    mm_164: "f32[768, 768]" = torch.ops.aten.mm.default(permute_647, view_55);  permute_647 = view_55 = None
    permute_648: "f32[768, 768]" = torch.ops.aten.permute.default(mm_164, [1, 0]);  mm_164 = None
    sum_222: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_727, [0], True);  view_727 = None
    view_728: "f32[768]" = torch.ops.aten.view.default(sum_222, [768]);  sum_222 = None
    permute_649: "f32[768, 768]" = torch.ops.aten.permute.default(permute_648, [1, 0]);  permute_648 = None
    view_729: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_163, [4, 512, 768]);  mm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_730: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_729, [4, 512, 12, 64]);  view_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_650: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_730, [0, 2, 1, 3]);  view_730 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_194: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
    view_731: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_194, [48, 512, 64]);  clone_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_651: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_20, [0, 2, 1]);  clone_20 = None
    bmm_96: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_651, view_731);  permute_651 = None
    permute_652: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
    bmm_97: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_731, permute_652);  view_731 = permute_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_33: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_425: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_97, alias_33);  bmm_97 = None
    sum_223: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [-1], True)
    mul_426: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_33, sum_223);  alias_33 = sum_223 = None
    sub_146: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_425, mul_426);  mul_425 = mul_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_653: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    bmm_98: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_653, sub_146);  permute_653 = None
    permute_654: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_28, [0, 2, 1]);  permute_28 = None
    bmm_99: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_146, permute_654);  sub_146 = permute_654 = None
    permute_655: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_98, [0, 2, 1]);  bmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_732: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_96, [4, 12, 512, 64]);  bmm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_733: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_655, [4, 12, 512, 64]);  permute_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_734: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_99, [4, 12, 512, 64]);  bmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_656: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None
    clone_195: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_656, memory_format = torch.contiguous_format);  permute_656 = None
    view_735: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_195, [4, 512, 768]);  clone_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_657: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None
    clone_196: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_657, memory_format = torch.contiguous_format);  permute_657 = None
    view_736: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_196, [4, 512, 768]);  clone_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_737: "f32[2048, 768]" = torch.ops.aten.view.default(view_736, [2048, 768]);  view_736 = None
    permute_658: "f32[768, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_165: "f32[2048, 768]" = torch.ops.aten.mm.default(view_737, permute_658);  permute_658 = None
    permute_659: "f32[768, 2048]" = torch.ops.aten.permute.default(view_737, [1, 0])
    mm_166: "f32[768, 768]" = torch.ops.aten.mm.default(permute_659, view_46);  permute_659 = view_46 = None
    permute_660: "f32[768, 768]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    sum_224: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
    view_738: "f32[768]" = torch.ops.aten.view.default(sum_224, [768]);  sum_224 = None
    permute_661: "f32[768, 768]" = torch.ops.aten.permute.default(permute_660, [1, 0]);  permute_660 = None
    view_739: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_165, [4, 512, 768]);  mm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_217: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_423, view_739);  mul_423 = view_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_662: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_733, [0, 2, 1, 3]);  view_733 = None
    view_740: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_662, [4, 512, 768]);  permute_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    clone_197: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_740, memory_format = torch.contiguous_format);  view_740 = None
    view_741: "f32[2048, 768]" = torch.ops.aten.view.default(clone_197, [2048, 768]);  clone_197 = None
    permute_663: "f32[768, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_167: "f32[2048, 768]" = torch.ops.aten.mm.default(view_741, permute_663);  permute_663 = None
    permute_664: "f32[768, 2048]" = torch.ops.aten.permute.default(view_741, [1, 0])
    mm_168: "f32[768, 768]" = torch.ops.aten.mm.default(permute_664, view_43);  permute_664 = view_43 = None
    permute_665: "f32[768, 768]" = torch.ops.aten.permute.default(mm_168, [1, 0]);  mm_168 = None
    sum_225: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_741, [0], True);  view_741 = None
    view_742: "f32[768]" = torch.ops.aten.view.default(sum_225, [768]);  sum_225 = None
    permute_666: "f32[768, 768]" = torch.ops.aten.permute.default(permute_665, [1, 0]);  permute_665 = None
    view_743: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_167, [4, 512, 768]);  mm_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_218: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_217, view_743);  add_217 = view_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_427: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_735, 0.125);  view_735 = None
    view_744: "f32[2048, 768]" = torch.ops.aten.view.default(mul_427, [2048, 768]);  mul_427 = None
    permute_667: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_169: "f32[2048, 768]" = torch.ops.aten.mm.default(view_744, permute_667);  permute_667 = None
    permute_668: "f32[768, 2048]" = torch.ops.aten.permute.default(view_744, [1, 0])
    mm_170: "f32[768, 768]" = torch.ops.aten.mm.default(permute_668, view_41);  permute_668 = view_41 = None
    permute_669: "f32[768, 768]" = torch.ops.aten.permute.default(mm_170, [1, 0]);  mm_170 = None
    sum_226: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_744, [0], True);  view_744 = None
    view_745: "f32[768]" = torch.ops.aten.view.default(sum_226, [768]);  sum_226 = None
    permute_670: "f32[768, 768]" = torch.ops.aten.permute.default(permute_669, [1, 0]);  permute_669 = None
    view_746: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_169, [4, 512, 768]);  mm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_219: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_218, view_746);  add_218 = view_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_147: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_9);  add_15 = getitem_9 = None
    mul_428: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_147, rsqrt_4);  sub_147 = None
    mul_429: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_219, primals_36);  primals_36 = None
    mul_430: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_429, 768)
    sum_227: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True)
    mul_431: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_429, mul_428);  mul_429 = None
    sum_228: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [2], True);  mul_431 = None
    mul_432: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_428, sum_228);  sum_228 = None
    sub_148: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_430, sum_227);  mul_430 = sum_227 = None
    sub_149: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_148, mul_432);  sub_148 = mul_432 = None
    div_45: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    mul_433: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_149);  div_45 = sub_149 = None
    mul_434: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_219, mul_428);  mul_428 = None
    sum_229: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_434, [0, 1]);  mul_434 = None
    sum_230: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_219, [0, 1]);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_747: "f32[2048, 768]" = torch.ops.aten.view.default(mul_433, [2048, 768])
    permute_671: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_171: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_747, permute_671);  permute_671 = None
    permute_672: "f32[768, 2048]" = torch.ops.aten.permute.default(view_747, [1, 0])
    mm_172: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_672, view_39);  permute_672 = view_39 = None
    permute_673: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_172, [1, 0]);  mm_172 = None
    sum_231: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_747, [0], True);  view_747 = None
    view_748: "f32[768]" = torch.ops.aten.view.default(sum_231, [768]);  sum_231 = None
    permute_674: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_673, [1, 0]);  permute_673 = None
    view_749: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_171, [4, 512, 3072]);  mm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_435: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_22: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_435);  mul_435 = None
    add_220: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_436: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_220, 0.5);  add_220 = None
    mul_437: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_38, view_38)
    mul_438: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_437, -0.5);  mul_437 = None
    exp_28: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_438);  mul_438 = None
    mul_439: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_440: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_38, mul_439);  view_38 = mul_439 = None
    add_221: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_436, mul_440);  mul_436 = mul_440 = None
    mul_441: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_749, add_221);  view_749 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_750: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_441, [2048, 3072]);  mul_441 = None
    permute_675: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_173: "f32[2048, 768]" = torch.ops.aten.mm.default(view_750, permute_675);  permute_675 = None
    permute_676: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_750, [1, 0])
    mm_174: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_676, view_37);  permute_676 = view_37 = None
    permute_677: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_174, [1, 0]);  mm_174 = None
    sum_232: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_750, [0], True);  view_750 = None
    view_751: "f32[3072]" = torch.ops.aten.view.default(sum_232, [3072]);  sum_232 = None
    permute_678: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_677, [1, 0]);  permute_677 = None
    view_752: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_173, [4, 512, 768]);  mm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_222: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_433, view_752);  mul_433 = view_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_150: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_7);  add_11 = getitem_7 = None
    mul_442: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_150, rsqrt_3);  sub_150 = None
    mul_443: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_222, primals_30);  primals_30 = None
    mul_444: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_443, 768)
    sum_233: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True)
    mul_445: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_443, mul_442);  mul_443 = None
    sum_234: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True);  mul_445 = None
    mul_446: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_442, sum_234);  sum_234 = None
    sub_151: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_444, sum_233);  mul_444 = sum_233 = None
    sub_152: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_151, mul_446);  sub_151 = mul_446 = None
    div_46: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_447: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_152);  div_46 = sub_152 = None
    mul_448: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_222, mul_442);  mul_442 = None
    sum_235: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 1]);  mul_448 = None
    sum_236: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_222, [0, 1]);  add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_753: "f32[2048, 768]" = torch.ops.aten.view.default(mul_447, [2048, 768])
    permute_679: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm_175: "f32[2048, 768]" = torch.ops.aten.mm.default(view_753, permute_679);  permute_679 = None
    permute_680: "f32[768, 2048]" = torch.ops.aten.permute.default(view_753, [1, 0])
    mm_176: "f32[768, 768]" = torch.ops.aten.mm.default(permute_680, view_35);  permute_680 = view_35 = None
    permute_681: "f32[768, 768]" = torch.ops.aten.permute.default(mm_176, [1, 0]);  mm_176 = None
    sum_237: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_753, [0], True);  view_753 = None
    view_754: "f32[768]" = torch.ops.aten.view.default(sum_237, [768]);  sum_237 = None
    permute_682: "f32[768, 768]" = torch.ops.aten.permute.default(permute_681, [1, 0]);  permute_681 = None
    view_755: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_175, [4, 512, 768]);  mm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_756: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_755, [4, 512, 12, 64]);  view_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_683: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_756, [0, 2, 1, 3]);  view_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_198: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_683, memory_format = torch.contiguous_format);  permute_683 = None
    view_757: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_198, [48, 512, 64]);  clone_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_684: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_12, [0, 2, 1]);  clone_12 = None
    bmm_100: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_684, view_757);  permute_684 = None
    permute_685: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    bmm_101: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_757, permute_685);  view_757 = permute_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_34: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_449: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_101, alias_34);  bmm_101 = None
    sum_238: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_449, [-1], True)
    mul_450: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_34, sum_238);  alias_34 = sum_238 = None
    sub_153: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_686: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
    bmm_102: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_686, sub_153);  permute_686 = None
    permute_687: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_17, [0, 2, 1]);  permute_17 = None
    bmm_103: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_153, permute_687);  sub_153 = permute_687 = None
    permute_688: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_102, [0, 2, 1]);  bmm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_758: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_100, [4, 12, 512, 64]);  bmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_759: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_688, [4, 12, 512, 64]);  permute_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_760: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_103, [4, 12, 512, 64]);  bmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_689: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_760, [0, 2, 1, 3]);  view_760 = None
    clone_199: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_689, memory_format = torch.contiguous_format);  permute_689 = None
    view_761: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_199, [4, 512, 768]);  clone_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_690: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_758, [0, 2, 1, 3]);  view_758 = None
    clone_200: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_690, memory_format = torch.contiguous_format);  permute_690 = None
    view_762: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_200, [4, 512, 768]);  clone_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_763: "f32[2048, 768]" = torch.ops.aten.view.default(view_762, [2048, 768]);  view_762 = None
    permute_691: "f32[768, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_177: "f32[2048, 768]" = torch.ops.aten.mm.default(view_763, permute_691);  permute_691 = None
    permute_692: "f32[768, 2048]" = torch.ops.aten.permute.default(view_763, [1, 0])
    mm_178: "f32[768, 768]" = torch.ops.aten.mm.default(permute_692, view_26);  permute_692 = view_26 = None
    permute_693: "f32[768, 768]" = torch.ops.aten.permute.default(mm_178, [1, 0]);  mm_178 = None
    sum_239: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_763, [0], True);  view_763 = None
    view_764: "f32[768]" = torch.ops.aten.view.default(sum_239, [768]);  sum_239 = None
    permute_694: "f32[768, 768]" = torch.ops.aten.permute.default(permute_693, [1, 0]);  permute_693 = None
    view_765: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_177, [4, 512, 768]);  mm_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_223: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_447, view_765);  mul_447 = view_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_695: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_759, [0, 2, 1, 3]);  view_759 = None
    view_766: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_695, [4, 512, 768]);  permute_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    clone_201: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_766, memory_format = torch.contiguous_format);  view_766 = None
    view_767: "f32[2048, 768]" = torch.ops.aten.view.default(clone_201, [2048, 768]);  clone_201 = None
    permute_696: "f32[768, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_179: "f32[2048, 768]" = torch.ops.aten.mm.default(view_767, permute_696);  permute_696 = None
    permute_697: "f32[768, 2048]" = torch.ops.aten.permute.default(view_767, [1, 0])
    mm_180: "f32[768, 768]" = torch.ops.aten.mm.default(permute_697, view_23);  permute_697 = view_23 = None
    permute_698: "f32[768, 768]" = torch.ops.aten.permute.default(mm_180, [1, 0]);  mm_180 = None
    sum_240: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_767, [0], True);  view_767 = None
    view_768: "f32[768]" = torch.ops.aten.view.default(sum_240, [768]);  sum_240 = None
    permute_699: "f32[768, 768]" = torch.ops.aten.permute.default(permute_698, [1, 0]);  permute_698 = None
    view_769: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_179, [4, 512, 768]);  mm_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_224: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_223, view_769);  add_223 = view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_451: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_761, 0.125);  view_761 = None
    view_770: "f32[2048, 768]" = torch.ops.aten.view.default(mul_451, [2048, 768]);  mul_451 = None
    permute_700: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_181: "f32[2048, 768]" = torch.ops.aten.mm.default(view_770, permute_700);  permute_700 = None
    permute_701: "f32[768, 2048]" = torch.ops.aten.permute.default(view_770, [1, 0])
    mm_182: "f32[768, 768]" = torch.ops.aten.mm.default(permute_701, view_21);  permute_701 = view_21 = None
    permute_702: "f32[768, 768]" = torch.ops.aten.permute.default(mm_182, [1, 0]);  mm_182 = None
    sum_241: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_770, [0], True);  view_770 = None
    view_771: "f32[768]" = torch.ops.aten.view.default(sum_241, [768]);  sum_241 = None
    permute_703: "f32[768, 768]" = torch.ops.aten.permute.default(permute_702, [1, 0]);  permute_702 = None
    view_772: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_181, [4, 512, 768]);  mm_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_225: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_224, view_772);  add_224 = view_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_154: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_5);  add_8 = getitem_5 = None
    mul_452: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_154, rsqrt_2);  sub_154 = None
    mul_453: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_225, primals_20);  primals_20 = None
    mul_454: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_453, 768)
    sum_242: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True)
    mul_455: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_453, mul_452);  mul_453 = None
    sum_243: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True);  mul_455 = None
    mul_456: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_452, sum_243);  sum_243 = None
    sub_155: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_454, sum_242);  mul_454 = sum_242 = None
    sub_156: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_155, mul_456);  sub_155 = mul_456 = None
    div_47: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    mul_457: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_47, sub_156);  div_47 = sub_156 = None
    mul_458: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_225, mul_452);  mul_452 = None
    sum_244: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 1]);  mul_458 = None
    sum_245: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_225, [0, 1]);  add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_773: "f32[2048, 768]" = torch.ops.aten.view.default(mul_457, [2048, 768])
    permute_704: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_183: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_773, permute_704);  permute_704 = None
    permute_705: "f32[768, 2048]" = torch.ops.aten.permute.default(view_773, [1, 0])
    mm_184: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_705, view_19);  permute_705 = view_19 = None
    permute_706: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    sum_246: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_773, [0], True);  view_773 = None
    view_774: "f32[768]" = torch.ops.aten.view.default(sum_246, [768]);  sum_246 = None
    permute_707: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_706, [1, 0]);  permute_706 = None
    view_775: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_183, [4, 512, 3072]);  mm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_459: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf_23: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_459);  mul_459 = None
    add_226: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_460: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_226, 0.5);  add_226 = None
    mul_461: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_18, view_18)
    mul_462: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_461, -0.5);  mul_461 = None
    exp_29: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_462);  mul_462 = None
    mul_463: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_464: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_18, mul_463);  view_18 = mul_463 = None
    add_227: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_460, mul_464);  mul_460 = mul_464 = None
    mul_465: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_775, add_227);  view_775 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_776: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_465, [2048, 3072]);  mul_465 = None
    permute_708: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_185: "f32[2048, 768]" = torch.ops.aten.mm.default(view_776, permute_708);  permute_708 = None
    permute_709: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_776, [1, 0])
    mm_186: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_709, view_17);  permute_709 = view_17 = None
    permute_710: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_186, [1, 0]);  mm_186 = None
    sum_247: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_776, [0], True);  view_776 = None
    view_777: "f32[3072]" = torch.ops.aten.view.default(sum_247, [3072]);  sum_247 = None
    permute_711: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_710, [1, 0]);  permute_710 = None
    view_778: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_185, [4, 512, 768]);  mm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_228: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_457, view_778);  mul_457 = view_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_157: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_3);  add_4 = getitem_3 = None
    mul_466: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_157, rsqrt_1);  sub_157 = None
    mul_467: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_228, primals_14);  primals_14 = None
    mul_468: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_467, 768)
    sum_248: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True)
    mul_469: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_467, mul_466);  mul_467 = None
    sum_249: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [2], True);  mul_469 = None
    mul_470: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_466, sum_249);  sum_249 = None
    sub_158: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_468, sum_248);  mul_468 = sum_248 = None
    sub_159: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_158, mul_470);  sub_158 = mul_470 = None
    div_48: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_471: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_159);  div_48 = sub_159 = None
    mul_472: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_228, mul_466);  mul_466 = None
    sum_250: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1]);  mul_472 = None
    sum_251: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_228, [0, 1]);  add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_779: "f32[2048, 768]" = torch.ops.aten.view.default(mul_471, [2048, 768])
    permute_712: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_187: "f32[2048, 768]" = torch.ops.aten.mm.default(view_779, permute_712);  permute_712 = None
    permute_713: "f32[768, 2048]" = torch.ops.aten.permute.default(view_779, [1, 0])
    mm_188: "f32[768, 768]" = torch.ops.aten.mm.default(permute_713, view_15);  permute_713 = view_15 = None
    permute_714: "f32[768, 768]" = torch.ops.aten.permute.default(mm_188, [1, 0]);  mm_188 = None
    sum_252: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_779, [0], True);  view_779 = None
    view_780: "f32[768]" = torch.ops.aten.view.default(sum_252, [768]);  sum_252 = None
    permute_715: "f32[768, 768]" = torch.ops.aten.permute.default(permute_714, [1, 0]);  permute_714 = None
    view_781: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_187, [4, 512, 768]);  mm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_782: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_781, [4, 512, 12, 64]);  view_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_716: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_782, [0, 2, 1, 3]);  view_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    clone_202: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_716, memory_format = torch.contiguous_format);  permute_716 = None
    view_783: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_202, [48, 512, 64]);  clone_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_717: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_4, [0, 2, 1]);  clone_4 = None
    bmm_104: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_717, view_783);  permute_717 = None
    permute_718: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm_105: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_783, permute_718);  view_783 = permute_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_35: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_473: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(bmm_105, alias_35);  bmm_105 = None
    sum_253: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [-1], True)
    mul_474: "f32[48, 512, 512]" = torch.ops.aten.mul.Tensor(alias_35, sum_253);  alias_35 = sum_253 = None
    sub_160: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_719: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_106: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_719, sub_160);  permute_719 = None
    permute_720: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
    bmm_107: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(sub_160, permute_720);  sub_160 = permute_720 = None
    permute_721: "f32[48, 512, 64]" = torch.ops.aten.permute.default(bmm_106, [0, 2, 1]);  bmm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_784: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_104, [4, 12, 512, 64]);  bmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_785: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(permute_721, [4, 12, 512, 64]);  permute_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_786: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_107, [4, 12, 512, 64]);  bmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_722: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_786, [0, 2, 1, 3]);  view_786 = None
    clone_203: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_722, memory_format = torch.contiguous_format);  permute_722 = None
    view_787: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_203, [4, 512, 768]);  clone_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_723: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_784, [0, 2, 1, 3]);  view_784 = None
    clone_204: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_723, memory_format = torch.contiguous_format);  permute_723 = None
    view_788: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_204, [4, 512, 768]);  clone_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_789: "f32[2048, 768]" = torch.ops.aten.view.default(view_788, [2048, 768]);  view_788 = None
    permute_724: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_189: "f32[2048, 768]" = torch.ops.aten.mm.default(view_789, permute_724);  permute_724 = None
    permute_725: "f32[768, 2048]" = torch.ops.aten.permute.default(view_789, [1, 0])
    mm_190: "f32[768, 768]" = torch.ops.aten.mm.default(permute_725, view_6);  permute_725 = view_6 = None
    permute_726: "f32[768, 768]" = torch.ops.aten.permute.default(mm_190, [1, 0]);  mm_190 = None
    sum_254: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_789, [0], True);  view_789 = None
    view_790: "f32[768]" = torch.ops.aten.view.default(sum_254, [768]);  sum_254 = None
    permute_727: "f32[768, 768]" = torch.ops.aten.permute.default(permute_726, [1, 0]);  permute_726 = None
    view_791: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_189, [4, 512, 768]);  mm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_229: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_471, view_791);  mul_471 = view_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_728: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_785, [0, 2, 1, 3]);  view_785 = None
    view_792: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_728, [4, 512, 768]);  permute_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    clone_205: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_792, memory_format = torch.contiguous_format);  view_792 = None
    view_793: "f32[2048, 768]" = torch.ops.aten.view.default(clone_205, [2048, 768]);  clone_205 = None
    permute_729: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_191: "f32[2048, 768]" = torch.ops.aten.mm.default(view_793, permute_729);  permute_729 = None
    permute_730: "f32[768, 2048]" = torch.ops.aten.permute.default(view_793, [1, 0])
    mm_192: "f32[768, 768]" = torch.ops.aten.mm.default(permute_730, view_3);  permute_730 = view_3 = None
    permute_731: "f32[768, 768]" = torch.ops.aten.permute.default(mm_192, [1, 0]);  mm_192 = None
    sum_255: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_793, [0], True);  view_793 = None
    view_794: "f32[768]" = torch.ops.aten.view.default(sum_255, [768]);  sum_255 = None
    permute_732: "f32[768, 768]" = torch.ops.aten.permute.default(permute_731, [1, 0]);  permute_731 = None
    view_795: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_191, [4, 512, 768]);  mm_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_230: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_229, view_795);  add_229 = view_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_475: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_787, 0.125);  view_787 = None
    view_796: "f32[2048, 768]" = torch.ops.aten.view.default(mul_475, [2048, 768]);  mul_475 = None
    permute_733: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_193: "f32[2048, 768]" = torch.ops.aten.mm.default(view_796, permute_733);  permute_733 = None
    permute_734: "f32[768, 2048]" = torch.ops.aten.permute.default(view_796, [1, 0])
    mm_194: "f32[768, 768]" = torch.ops.aten.mm.default(permute_734, view_1);  permute_734 = view_1 = None
    permute_735: "f32[768, 768]" = torch.ops.aten.permute.default(mm_194, [1, 0]);  mm_194 = None
    sum_256: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
    view_797: "f32[768]" = torch.ops.aten.view.default(sum_256, [768]);  sum_256 = None
    permute_736: "f32[768, 768]" = torch.ops.aten.permute.default(permute_735, [1, 0]);  permute_735 = None
    view_798: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_193, [4, 512, 768]);  mm_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_231: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_230, view_798);  add_230 = view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:824, code: hidden_states = self.layernorm_embedding(hidden_states)
    sub_161: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_476: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_161, rsqrt);  sub_161 = None
    mul_477: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_231, primals_4);  primals_4 = None
    mul_478: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_477, 768)
    sum_257: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_477, [2], True)
    mul_479: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_477, mul_476);  mul_477 = None
    sum_258: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_479, [2], True);  mul_479 = None
    mul_480: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_476, sum_258);  sum_258 = None
    sub_162: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_478, sum_257);  mul_478 = sum_257 = None
    sub_163: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_162, mul_480);  sub_162 = mul_480 = None
    div_49: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_481: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_163);  div_49 = sub_163 = None
    mul_482: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_231, mul_476);  mul_476 = None
    sum_259: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 1]);  mul_482 = None
    sum_260: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_231, [0, 1]);  add_231 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    eq_2: "b8[4, 512]" = torch.ops.aten.eq.Scalar(add, -1)
    unsqueeze_6: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[4, 512, 768]" = torch.ops.aten.where.self(unsqueeze_6, scalar_tensor_3, mul_481);  unsqueeze_6 = scalar_tensor_3 = None
    full_3: "f32[1026, 768]" = torch.ops.aten.full.default([1026, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_2: "f32[1026, 768]" = torch.ops.aten._unsafe_index_put.default(full_3, [add], where_3, True);  full_3 = add = where_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:818, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    mul_483: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_481, 1.0);  mul_481 = None
    eq_3: "b8[4, 512]" = torch.ops.aten.eq.Scalar(view, 1)
    unsqueeze_7: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_3, -1);  eq_3 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[4, 512, 768]" = torch.ops.aten.where.self(unsqueeze_7, scalar_tensor_4, mul_483);  unsqueeze_7 = scalar_tensor_4 = mul_483 = None
    full_4: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_3: "f32[50265, 768]" = torch.ops.aten._unsafe_index_put.default(full_4, [view], where_4, True);  full_4 = view = where_4 = None
    return pytree.tree_unflatten([add_117, clone_50, clone_51, clone_56, clone_57, clone_64, clone_65, clone_70, clone_71, clone_78, clone_79, clone_84, clone_85, clone_92, clone_93, clone_98, clone_99, clone_106, clone_107, clone_112, clone_113, clone_120, clone_121, clone_126, clone_127, add_45, _unsafe_index_put_2, _unsafe_index_put, _unsafe_index_put_3, sum_259, sum_260, permute_736, view_797, permute_732, view_794, permute_727, view_790, permute_715, view_780, sum_250, sum_251, permute_711, view_777, permute_707, view_774, sum_244, sum_245, permute_703, view_771, permute_699, view_768, permute_694, view_764, permute_682, view_754, sum_235, sum_236, permute_678, view_751, permute_674, view_748, sum_229, sum_230, permute_670, view_745, permute_666, view_742, permute_661, view_738, permute_649, view_728, sum_220, sum_221, permute_645, view_725, permute_641, view_722, sum_214, sum_215, permute_637, view_719, permute_633, view_716, permute_628, view_712, permute_616, view_702, sum_205, sum_206, permute_612, view_699, permute_608, view_696, sum_199, sum_200, permute_604, view_693, permute_600, view_690, permute_595, view_686, permute_583, view_676, sum_190, sum_191, permute_579, view_673, permute_575, view_670, sum_184, sum_185, permute_571, view_667, permute_567, view_664, permute_562, view_660, permute_550, view_650, sum_175, sum_176, permute_546, view_647, permute_542, view_644, sum_169, sum_170, _unsafe_index_put_1, sum_165, sum_166, permute_538, view_641, permute_534, view_638, permute_529, view_634, permute_517, view_622, sum_156, sum_157, permute_513, view_619, permute_509, view_616, permute_504, view_612, permute_492, view_602, sum_147, sum_148, permute_488, view_599, permute_484, view_596, sum_141, sum_142, permute_480, view_593, permute_476, view_590, permute_471, view_586, permute_459, view_574, sum_132, sum_133, permute_455, view_571, permute_451, view_568, permute_446, view_564, permute_434, view_554, sum_123, sum_124, permute_430, view_551, permute_426, view_548, sum_117, sum_118, permute_422, view_545, permute_418, view_542, permute_413, view_538, permute_401, view_526, sum_108, sum_109, permute_397, view_523, permute_393, view_520, permute_388, view_516, permute_376, view_506, sum_99, sum_100, permute_372, view_503, permute_368, view_500, sum_93, sum_94, permute_364, view_497, permute_360, view_494, permute_355, view_490, permute_343, view_478, sum_84, sum_85, permute_339, view_475, permute_335, view_472, permute_330, view_468, permute_318, view_458, sum_75, sum_76, permute_314, view_455, permute_310, view_452, sum_69, sum_70, permute_306, view_449, permute_302, view_446, permute_297, view_442, permute_285, view_430, sum_60, sum_61, permute_281, view_427, permute_277, view_424, permute_272, view_420, permute_260, view_410, sum_51, sum_52, permute_256, view_407, permute_252, view_404, sum_45, sum_46, permute_248, view_401, permute_244, view_398, permute_239, view_394, permute_227, view_382, sum_36, sum_37, permute_223, view_379, permute_219, view_376, permute_214, view_372, permute_202, view_362, sum_27, sum_28, permute_198, view_359, permute_194, view_356, sum_21, sum_22, permute_190, None, None, None], self._out_spec)
    